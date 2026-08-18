"""
Batched Inference Engine for Geospatial Soil Segmentation.
Coordinates tiling, PyTorch batch forward passes, uncertainty estimation,
overlap stitching, and exclusion mask application.
"""

import time
from typing import Tuple, Dict, Any, Optional, Callable
import numpy as np
import torch
import torch.nn.functional as F

from models.architecture import DualHeadUNet, load_model
from geospatial.preprocess import preprocess_raster_for_model
from geospatial.tile import generate_tiles
from geospatial.stitch import TileStitcher


class SoilInferenceEngine:
    """Manages local batched segmentation inference with dual heads."""

    def __init__(
        self,
        weights_path: str = "models/weights.pt",
        in_channels: int = 4,
        num_classes: int = 6,
        tile_size: int = 512,
        tile_overlap: int = 32,
        blend_method: str = "cosine",
        batch_size: int = 4,
        device: str = "auto"
    ):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.blend_method = blend_method
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = load_model(
            weights_path=weights_path,
            in_channels=in_channels,
            num_classes=num_classes,
            device=self.device
        )

    def run_inference(
        self,
        raster: np.ndarray,
        exclusion_mask: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end segmentation on input raster.
        
        Args:
            raster: (C, H, W) or (H, W, C) input array.
            exclusion_mask: (H, W) binary mask where 1 indicates masked/excluded area.
            progress_callback: Optional callback fn(percent: float, message: str).
            
        Returns:
            Dict containing:
                - 'class_map': (H, W) uint8 classified array [0..5]
                - 'confidence_map': (H, W) float32 confidence map [0..1]
                - 'prob_map': (num_classes, H, W) float32 probabilities
                - 'metrics': dictionary of execution timing and pixel counts
        """
        t0 = time.time()

        if progress_callback:
            progress_callback(5.0, "Preprocessing and channel normalization...")

        # 1. Preprocessing
        norm_raster, nodata_mask, prep_stats = preprocess_raster_for_model(
            raster=raster,
            target_channels=self.model.in_channels,
            normalize_method="percentile"
        )
        _, h, w = norm_raster.shape

        t_prep = time.time() - t0

        if progress_callback:
            progress_callback(15.0, f"Generating overlapping tiles ({self.tile_size}x{self.tile_size})...")

        # 2. Tiling
        tile_tuples = generate_tiles(
            raster=norm_raster,
            tile_size=self.tile_size,
            overlap=self.tile_overlap
        )
        total_tiles = len(tile_tuples)

        # 3. Stitcher setup
        stitcher = TileStitcher(
            full_shape=(h, w),
            num_classes=self.num_classes,
            tile_size=self.tile_size,
            blend_method=self.blend_method
        )

        # 4. Batched inference
        t_infer_start = time.time()
        num_batches = int(np.ceil(total_tiles / self.batch_size))

        self.model.eval()
        with torch.no_grad():
            for b_idx in range(num_batches):
                start_i = b_idx * self.batch_size
                end_i = min(start_i + self.batch_size, total_tiles)
                batch_items = tile_tuples[start_i:end_i]

                # Stack batch tensor
                batch_arrays = [item[0] for item in batch_items]
                batch_tensor = torch.from_numpy(np.stack(batch_arrays, axis=0)).to(self.device)

                # Forward pass
                logits, confs = self.model(batch_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                confs_np = confs.cpu().numpy()

                # Add each tile prediction to stitcher
                for local_i, (_, window) in enumerate(batch_items):
                    stitcher.add_tile_prediction(
                        probs_tile=probs[local_i],
                        conf_tile=confs_np[local_i],
                        window=window
                    )

                # Update progress
                pct = 20.0 + (end_i / total_tiles) * 60.0
                if progress_callback:
                    progress_callback(pct, f"Inference processing: tile {end_i}/{total_tiles} ({pct:.0f}%)...")

        t_infer = time.time() - t_infer_start

        if progress_callback:
            progress_callback(85.0, "Stitching prediction grid with spatial blending...")

        # 5. Finalize stitching
        class_map, confidence_map, prob_map = stitcher.finalize()

        # 6. Apply NoData and Exclusion Masks
        if progress_callback:
            progress_callback(92.0, "Applying exclusion masks and boundary filters...")

        # Mask invalid / nodata pixels to class 0
        class_map[nodata_mask] = 0
        confidence_map[nodata_mask] = 0.0

        # Burn user exclusion mask if provided
        if exclusion_mask is not None:
            ex_mask_bool = (exclusion_mask > 0)
            if ex_mask_bool.shape == class_map.shape:
                class_map[ex_mask_bool] = 0
                confidence_map[ex_mask_bool] = 0.0

        t_total = time.time() - t0

        if progress_callback:
            progress_callback(100.0, "Inference completed successfully!")

        return {
            "class_map": class_map,
            "confidence_map": confidence_map,
            "prob_map": prob_map,
            "metrics": {
                "total_tiles": total_tiles,
                "batch_size": self.batch_size,
                "device": self.device,
                "time_preprocessing_s": round(t_prep, 3),
                "time_inference_s": round(t_infer, 3),
                "time_total_s": round(t_total, 3),
                "dimensions": [w, h]
            }
        }
