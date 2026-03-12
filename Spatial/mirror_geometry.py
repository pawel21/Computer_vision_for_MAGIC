import numpy as np
from collections import defaultdict


def build_hex_mirror_graph(row_sizes):
    """
    Buduje graf sąsiedztwa dla luster w układzie heksagonalnym.
    row_sizes: lista z liczbą luster w kolejnych rzędach,
               np. [14, 16, 18, 20, 22, 24, 26, 28, 26, 24, 22, 20, 18, 16, 14]
    """
    mirrors = {}  # (row, col) -> mirror_id
    positions = {}  # mirror_id -> (x, y) fizyczna pozycja
    mirror_id = 0

    max_width = max(row_sizes)

    for row_idx, n_mirrors in enumerate(row_sizes):
        # Offset - centrowanie każdego rzędu
        offset = (max_width - n_mirrors) / 2.0
        for col_idx in range(n_mirrors):
            mirrors[(row_idx, col_idx)] = mirror_id

            # Pozycja fizyczna (uwzględniając hex staggering)
            x = offset + col_idx
            y = row_idx * np.sqrt(3) / 2  # odstęp hex między rzędami
            positions[mirror_id] = (x, y)
            mirror_id += 1

    total_mirrors = mirror_id
    adjacency = defaultdict(list)

    for row_idx, n_mirrors in enumerate(row_sizes):
        for col_idx in range(n_mirrors):
            mid = mirrors[(row_idx, col_idx)]

            # Sąsiedzi w tym samym rzędzie
            if col_idx > 0:
                adjacency[mid].append(mirrors[(row_idx, col_idx - 1)])
            if col_idx < n_mirrors - 1:
                adjacency[mid].append(mirrors[(row_idx, col_idx + 1)])

            # Sąsiedzi w rzędzie powyżej i poniżej
            for d_row, neighbor_row_idx in [(-1, row_idx - 1), (1, row_idx + 1)]:
                if 0 <= neighbor_row_idx < len(row_sizes):
                    n_neighbor = row_sizes[neighbor_row_idx]
                    delta = (n_neighbor - n_mirrors)

                    # W siatce hex, sąsiedztwo zależy od tego,
                    # czy sąsiedni rząd jest szerszy czy węższy
                    if delta > 0:  # sąsiedni rząd szerszy
                        neighbor_cols = [col_idx, col_idx + 1]
                    elif delta < 0:  # sąsiedni rząd węższy
                        neighbor_cols = [col_idx - 1, col_idx]
                    else:  # ten sam rozmiar
                        neighbor_cols = [col_idx - 1, col_idx, col_idx + 1]

                    for nc in neighbor_cols:
                        if 0 <= nc < n_neighbor:
                            nid = mirrors[(neighbor_row_idx, nc)]
                            if nid not in adjacency[mid]:
                                adjacency[mid].append(nid)

    return adjacency, positions, total_mirrors

row_sizes = [14, 16, 18, 20, 22, 24, 26, 28, 26, 24, 22, 20, 18, 16, 14]
adj, pos, n_total = build_hex_mirror_graph(row_sizes)

print(f"Łącznie luster: {n_total}")
print(f"Lustro 50 -> sąsiedzi: {adj[0]}")