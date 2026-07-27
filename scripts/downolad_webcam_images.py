#!/usr/bin/env python3
"""
Proste, sekwencyjne pobieranie zdjec z webcam MAGIC (magic.iac.es).
Parametry przez YAML (--config) albo CLI. CLI nadpisuje YAML, YAML nadpisuje domyslne.

Wszystkie pliki laduja plasko w jednym katalogu output_dir (bez podkatalogow
rok/miesiac/dzien), z nazwa zawierajaca pelna date, np. 20240301_0900_la.jpg.

Przyklady:
    python download_magic_webcam.py --start-date 2024-03-25 --output-dir ./out
    python download_magic_webcam.py --config config.yaml
    python download_magic_webcam.py --config config.yaml --insecure

Przykladowy config.yaml:
    start_date: "2024-03-01"
    end_date: "2024-03-05"
    start_time: "08:00"
    end_time: "18:00"
    interval_min: 10
    camera: webcam2
    output_dir: /home/pgliwny/Praca/.../data/webcam
    retries: 2
    overwrite: false
    verify_ssl: true
    ca_bundle: null

Wymaga: pip install requests pyyaml
"""

import argparse
import csv
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

try:
    import yaml
except ImportError:
    yaml = None

BASE_URL = "https://www.magic.iac.es/webcams/{camera}/{year:04d}/{month:02d}/{day:02d}/{hhmm}_la.jpg"

DEFAULTS = {
    "start_date": None,
    "end_date": None,
    "start_time": "00:00",
    "end_time": "23:50",
    "interval_min": 10,
    "camera": "webcam2",
    "output_dir": None,
    "timeout": 15.0,
    "retries": 2,
    "overwrite": False,
    "verify_ssl": True,
    "ca_bundle": None,
}


def parse_date(s) -> date:
    if isinstance(s, date):
        return s
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Nieprawidlowa data '{s}', oczekiwano YYYY-MM-DD")


def parse_time_hhmm(s):
    raw = str(s).replace(":", "")
    if len(raw) != 4 or not raw.isdigit():
        raise ValueError(f"Nieprawidlowy czas '{s}', oczekiwano HHMM lub HH:MM")
    h, m = int(raw[:2]), int(raw[2:])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Nieprawidlowy czas '{s}'")
    return h, m


def daterange(d1: date, d2: date):
    cur = d1
    while cur <= d2:
        yield cur
        cur += timedelta(days=1)


def build_timestamps(day: date, start_hm, end_hm, interval_min: int):
    start_dt = datetime.combine(day, datetime.min.time()).replace(hour=start_hm[0], minute=start_hm[1])
    end_dt = datetime.combine(day, datetime.min.time()).replace(hour=end_hm[0], minute=end_hm[1])
    if end_dt < start_dt:
        return
    cur = start_dt
    while cur <= end_dt:
        yield cur
        cur += timedelta(minutes=interval_min)


def download_one(session, camera, dt, out_dir, timeout, retries, overwrite, verify):
    hhmm = f"{dt.hour:02d}{dt.minute:02d}"
    url = BASE_URL.format(camera=camera, year=dt.year, month=dt.month, day=dt.day, hhmm=hhmm)
    dest = out_dir / f"{dt.strftime('%Y%m%d')}_{hhmm}_la.jpg"

    if dest.exists() and not overwrite:
        return {"timestamp": dt.isoformat(), "url": url, "status": "skipped_exists", "path": str(dest)}

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout, verify=verify)
            if resp.status_code == 200 and resp.content:
                dest.write_bytes(resp.content)
                return {"timestamp": dt.isoformat(), "url": url, "status": "ok", "path": str(dest)}
            elif resp.status_code == 404:
                return {"timestamp": dt.isoformat(), "url": url, "status": "not_found", "path": ""}
            else:
                last_err = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        if attempt < retries:
            time.sleep(1)

    return {"timestamp": dt.isoformat(), "url": url, "status": f"failed: {last_err}", "path": ""}


def build_arg_parser():
    p = argparse.ArgumentParser(description="Pobiera zdjecia z webcam MAGIC (magic.iac.es).")
    p.add_argument("--config", type=Path, default=None, help="Sciezka do pliku YAML z parametrami")
    p.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD (domyslnie = start-date)")
    p.add_argument("--start-time", default=None, help="HH:MM lub HHMM, domyslnie 00:00")
    p.add_argument("--end-time", default=None, help="HH:MM lub HHMM, domyslnie 23:50")
    p.add_argument("--interval-min", type=int, default=None, help="Odstep w minutach (domyslnie 10)")
    p.add_argument("--camera", default=None, help="Nazwa kamery, np. webcam2 (domyslnie) lub webcam1")
    p.add_argument("--output-dir", default=None, help="Katalog docelowy (plaski, bez podkatalogow)")
    p.add_argument("--timeout", type=float, default=None, help="Timeout pojedynczego zadania (s)")
    p.add_argument("--retries", type=int, default=None, help="Liczba prob przy bledzie (domyslnie 2)")
    p.add_argument("--overwrite", action="store_true", default=None, help="Pobierz ponownie istniejace pliki")
    p.add_argument("--insecure", dest="verify_ssl", action="store_false", default=None,
                    help="Wylacz weryfikacje certyfikatu SSL (self-signed cert serwera)")
    p.add_argument("--ca-bundle", dest="ca_bundle", default=None,
                    help="Sciezka do pliku CA bundle (PEM) do weryfikacji SSL")
    return p


def load_yaml_config(path: Path) -> dict:
    if yaml is None:
        sys.exit("Do uzycia --config potrzebny jest pakiet pyyaml: pip install pyyaml")
    if not path.exists():
        sys.exit(f"Nie znaleziono pliku config: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        sys.exit(f"Plik {path} musi zawierac mapowanie klucz: wartosc (YAML dict)")
    return {str(k).replace("-", "_"): v for k, v in data.items()}


def resolve_config(cli_args: argparse.Namespace) -> dict:
    merged = dict(DEFAULTS)

    if cli_args.config:
        yaml_cfg = load_yaml_config(cli_args.config)
        unknown = set(yaml_cfg) - set(DEFAULTS)
        if unknown:
            print(f"Uwaga: nieznane klucze w YAML zostana zignorowane: {sorted(unknown)}", file=sys.stderr)
        for k, v in yaml_cfg.items():
            if k in DEFAULTS:
                merged[k] = v

    cli_dict = vars(cli_args)
    for k in DEFAULTS:
        if k in cli_dict and cli_dict[k] is not None:
            merged[k] = cli_dict[k]

    if merged["start_date"] is None:
        sys.exit("Brak wymaganego parametru: start_date (podaj w --config lub jako --start-date)")
    if merged["output_dir"] is None:
        sys.exit("Brak wymaganego parametru: output_dir (podaj w --config lub jako --output-dir)")

    merged["start_date"] = parse_date(merged["start_date"])
    merged["end_date"] = parse_date(merged["end_date"]) if merged["end_date"] else merged["start_date"]
    merged["start_time"] = parse_time_hhmm(merged["start_time"])
    merged["end_time"] = parse_time_hhmm(merged["end_time"])
    merged["output_dir"] = Path(merged["output_dir"])
    merged["interval_min"] = int(merged["interval_min"])
    merged["timeout"] = float(merged["timeout"])
    merged["retries"] = int(merged["retries"])
    merged["overwrite"] = bool(merged["overwrite"])
    merged["verify_ssl"] = bool(merged["verify_ssl"])
    merged["ca_bundle"] = str(merged["ca_bundle"]) if merged["ca_bundle"] else None
    merged["verify"] = merged["ca_bundle"] if merged["ca_bundle"] else merged["verify_ssl"]

    if merged["end_date"] < merged["start_date"]:
        sys.exit("end_date musi byc >= start_date")

    return merged


def main():
    cli_args = build_arg_parser().parse_args()
    cfg = resolve_config(cli_args)

    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = cfg["output_dir"].resolve()
    print(f"Pliki beda zapisywane do: {cfg['output_dir']}")

    tasks = []
    for day in daterange(cfg["start_date"], cfg["end_date"]):
        tasks.extend(build_timestamps(day, cfg["start_time"], cfg["end_time"], cfg["interval_min"]))

    print(f"Zaplanowano {len(tasks)} zdjec: {cfg['start_date']} - {cfg['end_date']}, "
          f"{cfg['start_time'][0]:02d}:{cfg['start_time'][1]:02d}-{cfg['end_time'][0]:02d}:{cfg['end_time'][1]:02d}, "
          f"co {cfg['interval_min']} min, kamera={cfg['camera']}")

    if cfg["verify"] is False:
        print("UWAGA: weryfikacja certyfikatu SSL jest wylaczona (--insecure).", file=sys.stderr)
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    session = requests.Session()
    session.headers.update({"User-Agent": "magic-webcam-downloader/1.0"})

    results = []
    for i, dt in enumerate(tasks, start=1):
        res = download_one(session, cfg["camera"], dt, cfg["output_dir"],
                            cfg["timeout"], cfg["retries"], cfg["overwrite"], cfg["verify"])
        results.append(res)
        if i % 25 == 0 or i == len(tasks):
            print(f"  {i}/{len(tasks)} przetworzono")

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped_exists")
    not_found = sum(1 for r in results if r["status"] == "not_found")
    failed = sum(1 for r in results if r["status"].startswith("failed"))

    print(f"\nGotowe. ok={ok} pominieto_istniejace={skipped} brak_404={not_found} bledy={failed}")

    log_path = cfg["output_dir"] / "download_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "url", "status", "path"])
        writer.writeheader()
        for r in sorted(results, key=lambda r: r["timestamp"]):
            writer.writerow(r)
    print(f"Log zapisany do {log_path}")


if __name__ == "__main__":
    main()