# MirrorExtractor/simple_mirror_extractor.py

import json
import cv2
import numpy as np
import os

RGB_TO_GRAY = np.array([0.299, 0.587, 0.114])


def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using a weighted average.

    Args:
        img_rgb: Input image array of shape (H, W, 3) in RGB format.

    Returns:
        Grayscale image of shape (H, W) as uint8.
    """
    return np.dot(img_rgb[..., :3], RGB_TO_GRAY).astype(np.uint8)


class SimpleMirrorExtractor:
    """Extract individual mirror panels from a telescope camera image.

    Mirror polygon coordinates are loaded from a JSON file with the structure:
        {"mirror_ids": {"id_001": [[x1, y1], [x2, y2], ...], ...}}

    Args:
        mirror_points_path: Path to the JSON file with mirror polygon coordinates.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        KeyError: If the JSON file does not contain the expected structure.
    """

    def __init__(self, mirror_points_path: str) -> None:
        if not os.path.exists(mirror_points_path):
            raise FileNotFoundError(
                f"Mirror points file not found: {mirror_points_path}"
            )
        self.mirror_points_path = mirror_points_path
        with open(mirror_points_path, "r") as f:
            self.points: dict = json.load(f)

    def __repr__(self) -> str:
        n = len(self.points.get("mirror_ids", {}))
        return (
            f"SimpleMirrorExtractor("
            f"mirror_points_path='{self.mirror_points_path}')"
        )

    def get_point_coords(self, mirror_id: int) -> list:
        """Return polygon vertices for a given mirror ID.

        Args:
            mirror_id: Integer mirror identifier (e.g. 15 → key 'id_015').

        Returns:
            List of [x, y] coordinate pairs.

        Raises:
            KeyError: If the mirror ID is not present in the loaded data.
        """
        key = f"id_{mirror_id:03d}"
        if key not in self.points["mirror_ids"]:
            raise KeyError(
                f"Mirror '{key}' not found. "
                f"Available IDs: {list(self.points['mirror_ids'].keys())[:5]} ..."
            )
        return self.points["mirror_ids"][key]

    def _build_mask(self, shape: tuple, pts: np.ndarray) -> np.ndarray:
        """Create a binary polygon mask.

        Args:
            shape: Image shape (H, W) or (H, W, C).
            pts: Polygon vertices array of shape (N, 1, 2).

        Returns:
            Binary mask of shape (H, W) as uint8 (0 or 255).
        """
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], color=255)
        return mask

    def extract_mirror(self, img: np.ndarray, mirror_id: int) -> np.ndarray:
        """Extract and return a cropped grayscale patch for a single mirror.

        The mirror polygon is used to mask the image; the result is then
        cropped to the bounding rectangle of the polygon.

        Args:
            img: Input image as a NumPy array (H, W, 3), expected in RGB.
            mirror_id: Integer mirror identifier.

        Returns:
            Cropped grayscale image (H', W') as uint8.
        """
        points = self.get_point_coords(mirror_id)
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        mask = self._build_mask(img.shape, pts)
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        x, y, w, h = cv2.boundingRect(pts)
        cropped = masked_img[y : y + h, x : x + w]

        return rgb_to_gray(cropped)


