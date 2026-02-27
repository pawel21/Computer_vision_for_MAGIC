"""
Mirror Ranking Tracker — spatial ranking + temporal degradation detection.

For each image:
  - Extract texture features per mirror
  - Rank mirrors within the frame (spatial comparison)

Over time:
  - Track each mirror's rank history
  - Detect mirrors that are systematically dropping in ranking
  - Flag anomalies using both z-score and trend analysis

Usage:
    tracker = MirrorRankingTracker(mirror_points_json="mirror_points.json")

    # Process a batch of images
    tracker.process_images(image_paths)

    # Get degradation report
    report = tracker.detect_degradation()

    # Visualize
    tracker.plot_ranking_history(mirror_ids=[47, 128, 200])
    tracker.plot_degradation_heatmap()
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import warnings


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class FrameResult:
    """Results from processing a single image."""
    timestamp: datetime
    image_path: str
    feature_matrix: np.ndarray  # (249, n_features)
    ranks: np.ndarray  # (249,) — rank per mirror (1=best)
    composite_score: np.ndarray  # (249,) — single quality score per mirror
    z_scores: np.ndarray  # (249,) — how far from population median


@dataclass
class MirrorHistory:
    """Temporal history for a single mirror."""
    mirror_id: int
    timestamps: list = field(default_factory=list)
    ranks: list = field(default_factory=list)
    composite_scores: list = field(default_factory=list)
    z_scores: list = field(default_factory=list)


# ─────────────────────────────────────────────
# Core tracker
# ─────────────────────────────────────────────

class MirrorRankingTracker:
    """
    Tracks mirror quality rankings over time and detects degradation.

    The key insight: we don't compare absolute feature values between days
    (because lighting changes everything). Instead we compare the RANKING
    of mirrors within each frame — if a mirror drops from rank 30 to rank 150
    over a week, something changed on its surface, regardless of weather.
    """

    N_MIRRORS = 249

    # Features where HIGHER = better mirror quality
    HIGHER_IS_BETTER = ['glcm_homogeneity', 'glcm_energy', 'glcm_correlation']
    # Features where LOWER = better mirror quality
    LOWER_IS_BETTER = ['glcm_contrast', 'glcm_dissimilarity', 'lbp_entropy']
    # Informational features (used in composite but direction depends on context)
    INFORMATIONAL = ['lbp_mean', 'lbp_std']

    FEATURE_NAMES = [
        'lbp_mean', 'lbp_std', 'lbp_entropy',
        'glcm_contrast', 'glcm_dissimilarity',
        'glcm_homogeneity', 'glcm_energy', 'glcm_correlation'
    ]

    def __init__(self, mirror_points_json: str,
                 mirror_extractor=None,
                 feature_extractor=None):
        """
        Args:
            mirror_points_json: Path to JSON with mirror point coordinates
            mirror_extractor: SimpleMirrorExtractor instance (or None to create)
            feature_extractor: MirrorFeatureExtractor instance (or None to create)
        """
        self.mirror_points_json = mirror_points_json
        self.mirror_extractor = mirror_extractor
        self.feature_extractor = feature_extractor

        # Results storage
        self.frame_results: list[FrameResult] = []
        self.mirror_histories: dict[int, MirrorHistory] = {
            i: MirrorHistory(mirror_id=i) for i in range(self.N_MIRRORS)
        }

        # Baseline (built from first N frames)
        self.baseline_ranks: Optional[np.ndarray] = None  # (249,) median ranks
        self.baseline_rank_std: Optional[np.ndarray] = None  # (249,) rank std
        self.baseline_n_frames: int = 0

    # ─────────────────────────────────────────
    # Feature extraction for one frame
    # ─────────────────────────────────────────

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract texture features for all 249 mirrors from one image.
        Returns: feature_matrix (249, 8)
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        feature_matrix = np.zeros((self.N_MIRRORS, len(self.FEATURE_NAMES)))

        for mirror_id in range(self.N_MIRRORS):
            try:
                mirror_crop = self.mirror_extractor.extract_mirror_gray(
                    img_gray, mirror_id=mirror_id
                )
                feat = self.feature_extractor.extract_texture_features(mirror_crop)
                feature_matrix[mirror_id] = list(feat.values())
            except Exception as e:
                warnings.warn(f"Mirror {mirror_id}: extraction failed ({e})")
                feature_matrix[mirror_id] = np.nan

        return feature_matrix

    # ─────────────────────────────────────────
    # Composite quality score
    # ─────────────────────────────────────────

    def compute_composite_score(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Combine multiple features into a single quality score per mirror.

        Strategy:
        - z-score normalize each feature within the frame
        - flip sign for features where lower = better
        - average → single composite where HIGHER = better quality

        This is brightness-invariant because we normalize within each frame.
        """
        n_mirrors, n_features = feature_matrix.shape
        z_features = np.zeros_like(feature_matrix)

        for j in range(n_features):
            col = feature_matrix[:, j]
            valid = ~np.isnan(col)
            if valid.sum() < 2:
                continue

            med = np.median(col[valid])
            mad = np.median(np.abs(col[valid] - med))
            # Robust z-score using MAD (more resistant to outliers than std)
            if mad > 1e-10:
                z_features[:, j] = (col - med) / (mad * 1.4826)
            else:
                z_features[:, j] = 0.0

        # Flip sign so that HIGHER always = better quality
        for j, name in enumerate(self.FEATURE_NAMES):
            if name in self.LOWER_IS_BETTER:
                z_features[:, j] *= -1

        # Composite = mean of z-scored features
        # (excluding lbp_mean and lbp_std which are ambiguous)
        quality_indices = [
            i for i, name in enumerate(self.FEATURE_NAMES)
            if name not in self.INFORMATIONAL
        ]
        composite = np.nanmean(z_features[:, quality_indices], axis=1)

        return composite

    # ─────────────────────────────────────────
    # Ranking
    # ─────────────────────────────────────────

    def rank_mirrors(self, composite_score: np.ndarray) -> np.ndarray:
        """
        Rank mirrors by composite quality score.
        Rank 1 = best quality mirror, Rank 249 = worst.
        Ties get average rank.
        """
        from scipy.stats import rankdata
        # Higher composite = better → rank descending
        ranks = rankdata(-composite_score, method='average')
        return ranks.astype(int)

    def compute_z_scores(self, composite_score: np.ndarray) -> np.ndarray:
        """
        Z-score of each mirror's composite score vs population.
        Negative z-score = worse than average.
        """
        valid = ~np.isnan(composite_score)
        med = np.median(composite_score[valid])
        mad = np.median(np.abs(composite_score[valid] - med))
        if mad > 1e-10:
            return (composite_score - med) / (mad * 1.4826)
        return np.zeros_like(composite_score)

    # ─────────────────────────────────────────
    # Process images
    # ─────────────────────────────────────────

    def parse_timestamp(self, image_path: str) -> datetime:
        """
        Extract timestamp from MAGIC camera filename.
        Format: IRCamM1T20251005_075000M.jpg → 2025-10-05 07:50:00
        """
        fname = Path(image_path).stem
        try:
            # Find the date-time portion: 8 digits _ 6 digits
            # Pattern: ...YYYYMMDD_HHMMSS...
            import re
            match = re.search(r'(\d{8})_(\d{6})', fname)
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                return datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        except Exception:
            pass
        # Fallback: use file modification time
        return datetime.fromtimestamp(os.path.getmtime(image_path))

    def process_single_image(self, image_path: str) -> FrameResult:
        """Process one image: extract features, rank, store."""
        timestamp = self.parse_timestamp(image_path)
        feature_matrix = self.extract_features(image_path)
        composite = self.compute_composite_score(feature_matrix)
        ranks = self.rank_mirrors(composite)
        z_scores = self.compute_z_scores(composite)

        result = FrameResult(
            timestamp=timestamp,
            image_path=image_path,
            feature_matrix=feature_matrix,
            ranks=ranks,
            composite_score=composite,
            z_scores=z_scores,
        )

        # Store
        self.frame_results.append(result)
        for mid in range(self.N_MIRRORS):
            h = self.mirror_histories[mid]
            h.timestamps.append(timestamp)
            h.ranks.append(ranks[mid])
            h.composite_scores.append(composite[mid])
            h.z_scores.append(z_scores[mid])

        return result

    def process_images(self, image_paths: list[str],
                       verbose: bool = True) -> list[FrameResult]:
        """
        Process a batch of images chronologically.

        Args:
            image_paths: List of image file paths
            verbose: Print progress
        """
        # Sort by timestamp
        paths_with_ts = [(p, self.parse_timestamp(p)) for p in image_paths]
        paths_with_ts.sort(key=lambda x: x[1])

        results = []
        for i, (path, ts) in enumerate(paths_with_ts):
            if verbose:
                print(f"[{i + 1}/{len(paths_with_ts)}] {Path(path).name} ({ts})")
            try:
                result = self.process_single_image(path)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Failed to process {path}: {e}")

        if verbose:
            print(f"\nProcessed {len(results)} images successfully.")

        return results

    # ─────────────────────────────────────────
    # Baseline building
    # ─────────────────────────────────────────

    def build_baseline(self, n_frames: Optional[int] = None):
        """
        Build baseline statistics from the first N frames.

        The baseline captures each mirror's "normal" rank range.
        Default: use first 30% of frames or at least 10.
        """
        if not self.frame_results:
            raise ValueError("No frames processed yet. Call process_images first.")

        if n_frames is None:
            n_frames = max(10, int(len(self.frame_results) * 0.3))
        n_frames = min(n_frames, len(self.frame_results))

        # Collect ranks from baseline frames
        rank_matrix = np.array([
            self.frame_results[i].ranks for i in range(n_frames)
        ])  # (n_frames, 249)

        self.baseline_ranks = np.median(rank_matrix, axis=0)
        self.baseline_rank_std = np.std(rank_matrix, axis=0)
        # Minimum std to avoid division by zero for very stable mirrors
        self.baseline_rank_std = np.maximum(self.baseline_rank_std, 5.0)
        self.baseline_n_frames = n_frames

        print(f"Baseline built from {n_frames} frames.")
        print(f"Median rank range: {self.baseline_ranks.min():.0f} - "
              f"{self.baseline_ranks.max():.0f}")

    # ─────────────────────────────────────────
    # Degradation detection
    # ─────────────────────────────────────────

    def detect_degradation(self,
                           rank_drop_threshold: float = 3.0,
                           trend_window: int = 5,
                           min_consecutive_days: int = 3
                           ) -> dict:
        """
        Detect mirrors showing signs of degradation.

        Three complementary signals:
        1. RANK DROP: Mirror's recent rank is significantly worse than baseline
        2. TREND: Mirror shows consistent downward trend in ranking
        3. SPATIAL OUTLIER: Mirror is consistently in bottom percentile

        Args:
            rank_drop_threshold: How many baseline-stds of rank drop to flag
            trend_window: Number of recent frames to compute trend
            min_consecutive_days: Minimum consecutive bad frames to flag

        Returns:
            Dict with flagged mirrors and their diagnostics
        """
        if self.baseline_ranks is None:
            self.build_baseline()

        n_frames = len(self.frame_results)
        if n_frames < self.baseline_n_frames + trend_window:
            warnings.warn("Not enough frames after baseline for trend analysis.")

        results = {
            'rank_drop': [],  # mirrors with significant rank drop
            'trending_down': [],  # mirrors with consistent negative trend
            'persistent_outlier': [],  # mirrors consistently in bottom 10%
            'all_flagged': set(),
        }

        for mid in range(self.N_MIRRORS):
            h = self.mirror_histories[mid]
            if len(h.ranks) < self.baseline_n_frames + 2:
                continue

            recent_ranks = np.array(h.ranks[self.baseline_n_frames:])
            baseline_rank = self.baseline_ranks[mid]
            baseline_std = self.baseline_rank_std[mid]

            # ── Signal 1: Rank drop ──
            # Is the recent median rank significantly worse (higher number)?
            recent_median = np.median(recent_ranks[-trend_window:])
            rank_drop = (recent_median - baseline_rank) / baseline_std

            if rank_drop > rank_drop_threshold:
                results['rank_drop'].append({
                    'mirror_id': mid,
                    'baseline_rank': baseline_rank,
                    'recent_rank': recent_median,
                    'drop_sigma': rank_drop,
                })
                results['all_flagged'].add(mid)

            # ── Signal 2: Consistent downward trend ──
            # Linear regression on recent ranks (positive slope = getting worse)
            if len(recent_ranks) >= trend_window:
                window = recent_ranks[-trend_window:]
                x = np.arange(len(window))
                slope = np.polyfit(x, window, 1)[0]

                # Slope > 2 ranks per frame = concerning trend
                if slope > 2.0:
                    results['trending_down'].append({
                        'mirror_id': mid,
                        'slope': slope,
                        'ranks_per_frame': slope,
                        'recent_ranks': window.tolist(),
                    })
                    results['all_flagged'].add(mid)

            # ── Signal 3: Persistent outlier ──
            # Mirror consistently in bottom 10% (rank > 225)
            bottom_threshold = self.N_MIRRORS * 0.9
            recent_bottom = recent_ranks > bottom_threshold

            # Check for consecutive bad frames
            max_consecutive = self._max_consecutive_true(recent_bottom)
            if max_consecutive >= min_consecutive_days:
                results['persistent_outlier'].append({
                    'mirror_id': mid,
                    'consecutive_bottom_frames': max_consecutive,
                    'pct_in_bottom_10': recent_bottom.mean() * 100,
                })
                results['all_flagged'].add(mid)

        # Sort by severity
        results['rank_drop'].sort(key=lambda x: x['drop_sigma'], reverse=True)
        results['trending_down'].sort(key=lambda x: x['slope'], reverse=True)
        results['persistent_outlier'].sort(
            key=lambda x: x['consecutive_bottom_frames'], reverse=True
        )

        return results

    @staticmethod
    def _max_consecutive_true(arr: np.ndarray) -> int:
        """Find longest run of True values in boolean array."""
        max_run = 0
        current_run = 0
        for val in arr:
            if val:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    # ─────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────

    def print_report(self, report: Optional[dict] = None):
        """Print a human-readable degradation report."""
        if report is None:
            report = self.detect_degradation()

        print("=" * 70)
        print("MIRROR DEGRADATION REPORT")
        print(f"Based on {len(self.frame_results)} frames, "
              f"baseline from first {self.baseline_n_frames}")
        print("=" * 70)

        n_flagged = len(report['all_flagged'])
        print(f"\nTotal flagged mirrors: {n_flagged} / {self.N_MIRRORS}")

        if report['rank_drop']:
            print(f"\n── SIGNIFICANT RANK DROPS ({len(report['rank_drop'])}) ──")
            for item in report['rank_drop'][:10]:
                print(f"  Mirror {item['mirror_id']:3d}: "
                      f"baseline rank {item['baseline_rank']:5.0f} → "
                      f"recent {item['recent_rank']:5.0f} "
                      f"({item['drop_sigma']:+.1f}σ)")

        if report['trending_down']:
            print(f"\n── TRENDING DOWNWARD ({len(report['trending_down'])}) ──")
            for item in report['trending_down'][:10]:
                print(f"  Mirror {item['mirror_id']:3d}: "
                      f"slope = {item['slope']:+.1f} ranks/frame "
                      f"(recent: {item['recent_ranks']})")

        if report['persistent_outlier']:
            print(f"\n── PERSISTENT BOTTOM 10% ({len(report['persistent_outlier'])}) ──")
            for item in report['persistent_outlier'][:10]:
                print(f"  Mirror {item['mirror_id']:3d}: "
                      f"{item['consecutive_bottom_frames']} consecutive frames, "
                      f"{item['pct_in_bottom_10']:.0f}% of time in bottom 10%")

        if not n_flagged:
            print("\n  ✓ No mirrors flagged for degradation.")

        print("=" * 70)

    def get_frame_summary(self, frame_idx: int = -1) -> dict:
        """
        Get a summary for a single frame — useful for quick inspection.
        Returns top 5 best and worst mirrors.
        """
        result = self.frame_results[frame_idx]
        order = np.argsort(result.ranks)

        return {
            'timestamp': result.timestamp,
            'image': result.image_path,
            'best_5': [
                {'mirror_id': int(order[i]),
                 'rank': int(result.ranks[order[i]]),
                 'z_score': float(result.z_scores[order[i]])}
                for i in range(5)
            ],
            'worst_5': [
                {'mirror_id': int(order[-(i + 1)]),
                 'rank': int(result.ranks[order[-(i + 1)]]),
                 'z_score': float(result.z_scores[order[-(i + 1)]])}
                for i in range(5)
            ],
        }

    # ─────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────

    def plot_ranking_history(self, mirror_ids: list[int],
                             save_path: Optional[str] = None):
        """
        Plot rank history over time for selected mirrors.
        Useful for inspecting flagged mirrors.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))

        for mid in mirror_ids:
            h = self.mirror_histories[mid]
            ax.plot(h.timestamps, h.ranks,
                    marker='.', markersize=4, label=f'Mirror {mid}')

        # Baseline period shading
        if self.baseline_n_frames > 0:
            baseline_end = self.frame_results[self.baseline_n_frames - 1].timestamp
            ax.axvspan(h.timestamps[0], baseline_end,
                       alpha=0.1, color='green', label='Baseline period')

        ax.set_xlabel('Time')
        ax.set_ylabel('Rank (1 = best, 249 = worst)')
        ax.set_title('Mirror Ranking Over Time')
        ax.invert_yaxis()  # Rank 1 at top
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()

    def plot_rank_heatmap(self, save_path: Optional[str] = None):
        """
        Heatmap: mirrors × time, colored by rank.
        Quickly shows which mirrors degrade over time.
        """
        import matplotlib.pyplot as plt

        n_frames = len(self.frame_results)
        rank_matrix = np.zeros((self.N_MIRRORS, n_frames))

        for i, result in enumerate(self.frame_results):
            rank_matrix[:, i] = result.ranks

        # Sort mirrors by their median rank for cleaner visualization
        median_ranks = np.median(rank_matrix, axis=1)
        sort_order = np.argsort(median_ranks)

        fig, ax = plt.subplots(figsize=(16, 10))
        im = ax.imshow(rank_matrix[sort_order],
                       aspect='auto', cmap='RdYlGn_r',
                       interpolation='nearest')

        ax.set_xlabel('Frame index')
        ax.set_ylabel('Mirror (sorted by median rank)')
        ax.set_title('Mirror Rank Heatmap Over Time')
        plt.colorbar(im, ax=ax, label='Rank (1=best, 249=worst)')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()

    def plot_degradation_scatter(self, report: Optional[dict] = None,
                                 save_path: Optional[str] = None):
        """
        Scatter plot: baseline rank vs recent rank.
        Mirrors on the diagonal = stable. Below = degrading.
        """
        import matplotlib.pyplot as plt

        if self.baseline_ranks is None:
            self.build_baseline()
        if report is None:
            report = self.detect_degradation()

        # Recent ranks: median of last 5 frames
        recent_ranks = np.array([
            np.median(self.mirror_histories[mid].ranks[-5:])
            for mid in range(self.N_MIRRORS)
        ])

        flagged = report['all_flagged']

        fig, ax = plt.subplots(figsize=(8, 8))

        # Normal mirrors
        normal = [i for i in range(self.N_MIRRORS) if i not in flagged]
        ax.scatter(self.baseline_ranks[normal], recent_ranks[normal],
                   alpha=0.4, s=20, color='steelblue', label='Normal')

        # Flagged mirrors
        flagged_list = list(flagged)
        if flagged_list:
            ax.scatter(self.baseline_ranks[flagged_list],
                       recent_ranks[flagged_list],
                       alpha=0.8, s=60, color='red',
                       marker='x', linewidths=2, label='Flagged')

            # Label the worst offenders
            for mid in flagged_list[:5]:
                ax.annotate(f'{mid}',
                            (self.baseline_ranks[mid], recent_ranks[mid]),
                            fontsize=8, fontweight='bold', color='red',
                            xytext=(5, 5), textcoords='offset points')

        # Diagonal = no change
        ax.plot([1, 249], [1, 249], 'k--', alpha=0.3, label='No change')

        ax.set_xlabel('Baseline Rank (median)')
        ax.set_ylabel('Recent Rank (median last 5 frames)')
        ax.set_title('Mirror Degradation: Baseline vs Recent Ranking')
        ax.legend()
        ax.set_xlim(0, 250)
        ax.set_ylim(0, 250)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()

    # ─────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────

    def save_results(self, output_path: str):
        """Save all results to npz for later analysis."""
        n_frames = len(self.frame_results)

        rank_matrix = np.zeros((n_frames, self.N_MIRRORS))
        composite_matrix = np.zeros((n_frames, self.N_MIRRORS))
        zscore_matrix = np.zeros((n_frames, self.N_MIRRORS))
        timestamps = []

        for i, result in enumerate(self.frame_results):
            rank_matrix[i] = result.ranks
            composite_matrix[i] = result.composite_score
            zscore_matrix[i] = result.z_scores
            timestamps.append(result.timestamp.isoformat())

        np.savez_compressed(
            output_path,
            rank_matrix=rank_matrix,
            composite_matrix=composite_matrix,
            zscore_matrix=zscore_matrix,
            timestamps=np.array(timestamps),
            baseline_ranks=self.baseline_ranks if self.baseline_ranks is not None else np.array([]),
            baseline_rank_std=self.baseline_rank_std if self.baseline_rank_std is not None else np.array([]),
        )
        print(f"Results saved to {output_path}")


# ─────────────────────────────────────────────
# Usage example
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Example usage — replace with your actual extractor classes.
    """
    from glob import glob

    # ── Configuration ──
    MIRROR_POINTS_JSON = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/points_IRCam.json"
    IMAGE_DIR = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/data/images_for_analysis/"
    OUTPUT_DIR = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/data/mon_results/"

    # ── Import your extractors ──
    from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor
    from MirrorFeatureExtractor.mirror_feature_extractor import MirrorFeatureExtractor

    mirror_extractor = SimpleMirrorExtractor(MIRROR_POINTS_JSON)
    feature_extractor = MirrorFeatureExtractor()

    # ── Collect images ──
    # Top camera images (frontal view with visible mirrors)
    image_paths = sorted(glob(os.path.join(IMAGE_DIR, "IRCamM1T*.jpg")))
    print(f"Found {len(image_paths)} top-camera images")

    # ── Initialize tracker ──
    tracker = MirrorRankingTracker(
         mirror_points_json=MIRROR_POINTS_JSON,
         mirror_extractor=mirror_extractor,
         feature_extractor=feature_extractor,
     )

    # ── Process all images ──
    tracker.process_images(image_paths)

    # ── Build baseline from first 20 "good" images ──
    tracker.build_baseline(n_frames=10)

    # ── Detect degradation ──
    report = tracker.detect_degradation(
         rank_drop_threshold=3.0,   # 3 sigma drop = flagged
         trend_window=5,            # look at last 5 frames
         min_consecutive_days=3,    # 3+ consecutive bad frames
     )
    tracker.print_report(report)

    # ── Quick check: what does the latest frame look like? ──
    summary = tracker.get_frame_summary(-1)  # last frame
    print(f"\nLatest frame ({summary['timestamp']}):")
    print(f"  Best mirrors:  {[m['mirror_id'] for m in summary['best_5']]}")
    print(f"  Worst mirrors: {[m['mirror_id'] for m in summary['worst_5']]}")

    # ── Visualize ──
    tracker.plot_ranking_history(
         mirror_ids=[report['rank_drop'][0]['mirror_id']] if report['rank_drop'] else [0, 50, 100],
         save_path=os.path.join(OUTPUT_DIR, "ranking_history.png")
     )
    tracker.plot_degradation_scatter(
         report=report,
         save_path=os.path.join(OUTPUT_DIR, "degradation_scatter.png")
    )
    tracker.plot_rank_heatmap(
         save_path=os.path.join(OUTPUT_DIR, "rank_heatmap.png")
     )

    # ── Save for later ──
    tracker.save_results(os.path.join(OUTPUT_DIR, "mirror_tracking.npz"))