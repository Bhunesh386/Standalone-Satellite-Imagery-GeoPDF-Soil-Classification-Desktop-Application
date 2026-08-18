"""
Soil Categorical Legend and Palette Manager.
Manages class IDs, display names, hex/RGB palettes, and preview swatches.
"""

import os
import json
from typing import Dict, Any, List, Tuple
import numpy as np


class SoilLegendManager:
    """Loads and formats soil classification categorical color legends."""

    def __init__(self, legend_path: str = "labels/class_legend.json"):
        self.legend_path = legend_path
        self.classes_data = self._load_legend()

    def _load_legend(self) -> List[Dict[str, Any]]:
        """Loads legend JSON or falls back to defaults."""
        if os.path.exists(self.legend_path):
            try:
                with open(self.legend_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("classes", [])
            except Exception as e:
                print(f"Warning: Failed to load legend from {self.legend_path}: {e}")

        # Default fallback
        return [
            {"id": 0, "name": "NoData / Masked", "color_rgb": [0, 0, 0], "hex": "#000000", "description": "Unclassified/NoData"},
            {"id": 1, "name": "Alluvial Soil", "color_rgb": [230, 184, 0], "hex": "#E6B800", "description": "Fertile river silt"},
            {"id": 2, "name": "Black Soil (Vertisol)", "color_rgb": [54, 43, 40], "hex": "#362B28", "description": "Moisture-retentive clay"},
            {"id": 3, "name": "Red Soil", "color_rgb": [192, 57, 43], "hex": "#C0392B", "description": "Iron-oxide rich"},
            {"id": 4, "name": "Lateritic Soil", "color_rgb": [211, 84, 0], "hex": "#D35400", "description": "Leached aluminosilicate"},
            {"id": 5, "name": "Sandy / Desert Soil", "color_rgb": [243, 156, 18], "hex": "#F39C12", "description": "Arid coarse sand"}
        ]

    def get_color_table(self) -> Dict[int, Tuple[int, int, int, int]]:
        """Returns dict of class_id -> (R, G, B, A)."""
        table = {}
        for c in self.classes_data:
            rgb = c.get("color_rgb", [0, 0, 0])
            alpha = 0 if c["id"] == 0 else 255
            table[c["id"]] = (rgb[0], rgb[1], rgb[2], alpha)
        return table

    def get_class_names(self) -> Dict[int, str]:
        """Returns dict of class_id -> name."""
        return {c["id"]: c["name"] for c in self.classes_data}

    def colorize_class_map(self, class_map: np.ndarray) -> np.ndarray:
        """
        Converts uint8 class map (H, W) into an RGB image (H, W, 3).
        """
        h, w = class_map.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

        for c in self.classes_data:
            mask = (class_map == c["id"])
            rgb = c.get("color_rgb", [0, 0, 0])
            rgb_img[mask] = rgb

        return rgb_img

    def to_json(self, out_path: str):
        """Saves current legend to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"classes": self.classes_data}, f, indent=2)
