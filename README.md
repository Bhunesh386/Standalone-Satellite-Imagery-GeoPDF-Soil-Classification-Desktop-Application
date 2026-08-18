# Standalone Satellite-Imagery & GeoPDF Soil Classification Desktop Application

A fully local, offline, native desktop software system built with **PyQt6 / PySide6**, **PyTorch**, **Rasterio**, **PyMuPDF**, and **ReportLab** to perform per-pixel soil classification, geospatial uncertainty mapping, and automated reporting on GeoTIFF and GeoPDF files.

---

## 🌟 Key Features

1. **100% Offline & Native:** Runs entirely locally on user hardware with zero external cloud or web dependencies.
2. **Multi-Format Ingestion:** Supports `.tif`, `.tiff` multi-spectral satellite imagery and modern `.pdf` GeoPDF documents with automatic signature and integrity validation.
3. **GeoPDF Inspector & Layer-Role Matrix:**
   - Detects CRS, page dimensions, coordinate viewports, and OCG layer trees.
   - Assignable layer roles: `Model Input`, `Exclusion Mask` (e.g. roads, neatlines, furniture), `Reference Label`, and `Ignore`.
   - Multi-page visual thumbnail selector and non-blocking RGB fallback detection.
4. **Three GeoPDF Processing Modes:**
   - **Mode A (Raster):** Extracts rendered page at target DPI, crops neatline viewport to strip map furniture, and calculates affine geotransforms.
   - **Mode B (Vector):** Extracts vector paths and polylines, converting coordinates into georeferenced Shapely geometries.
   - **Mode C (Hybrid - Recommended):** Base raster imagery as model input combined with vector exclusion masks.
5. **Sliding-Window Tiling & Blending Engine:**
   - Splits large scenes into overlapping tiles (e.g. 512×512 with 32px overlap).
   - Smooth 2D Cosine / Hann, Gaussian, or Linear blending filters to eliminate boundary seams.
6. **Dual-Head PyTorch U-Net Model:**
   - **Head 1 (Soil Class Segmentation):** Pixel-level categorical probabilities over 6 classes (NoData, Alluvial, Black/Vertisol, Red, Lateritic, Sandy/Desert).
   - **Head 2 (Confidence Estimation):** Calibrated pixel-wise certainty map combining entropy and softmax probability.
   - Includes pretrained synthetic weight generator for immediate out-of-the-box execution.
7. **Comprehensive Deliverables Package:**
   - `classified_soil_map.tif`: 8-bit single-band GeoTIFF with embedded RGBA color table, CRS, Affine transform, NoData=0, and class metadata tags.
   - `soil_confidence.tif`: Float32 single-band inference confidence heatmap.
   - `classified_soil_map.pdf`: Publication-grade georeferenced GeoPDF report with neatline, scale bar, north arrow, and categorical legend box.
   - `classification_report.json`: Full audit metrics including class area in $km^2$ / $ha$, percentages, confidence statistics, and runtimes.
   - `preview.png` & `legend.json`.
8. **Modern Dark GIS Desktop GUI:**
   - Step 1: Ingestion & Drag-and-Drop Dropzone
   - Step 2: GeoPDF Inspector & Layer Hierarchy Matrix
   - Step 3: Processing Grid, CRS, Resolution & Export Options
   - Step 4: Interactive Viewport Preview with Pan/Zoom & Coordinate Tracker
   - Step 5: Results Dashboard with Tri-View Comparison, Statistics Tables, and Output Shortcuts

---

## 🏗️ Architecture & Module Structure

```text
build_geospatial_soil_classifier/
├── app.py                      # Main desktop GUI & CLI entrypoint
├── config.yaml                 # Grid, model, and export configuration
├── requirements.txt            # Local Python dependencies
├── models/
│   ├── architecture.py         # Dual-Head U-Net PyTorch implementation
│   └── weights.pt              # Saved model weights
├── ingestion/
│   ├── detect_format.py        # Magic byte & file format detector
│   ├── inspect_geopdf.py       # GeoPDF metadata, CRS, and layer tree extractor
│   ├── render_geopdf.py        # DPI-aware PDF viewport renderer & thumbnail generator
│   ├── extract_geopdf_raster.py# Mode A neatline raster extractor & georeferencer
│   └── extract_geopdf_vector.py# Mode B & C vector path and geometry extractor
├── geospatial/
│   ├── crs.py                  # PyProj CRS reprojection & auto-UTM zone calculator
│   ├── validate.py             # Raster integrity and metadata validator
│   ├── preprocess.py           # Radiometric scaling, percentile normalization & masking
│   ├── tile.py                 # Sliding-window tiling generator
│   ├── rasterize.py            # Vector layer and road exclusion rasterizer
│   ├── stitch.py               # Overlap stitching & 2D spatial blending filters
│   └── export.py               # GeoTIFF writer with color tables & tags
├── inference/
│   └── predict.py              # Batched segmentation inference pipeline
├── reporting/
│   ├── legend.py               # Categorical color palette and legend management
│   ├── report.py               # JSON audit report generator (areas, stats, runtimes)
│   └── geopdf_export.py        # ReportLab GeoPDF map sheet generator
├── labels/
│   └── class_legend.json       # Color definitions & soil geotechnical descriptions
├── gui/
│   ├── main_window.py          # Main Qt desktop window & multi-step workflow coordinator
│   ├── qt_compat.py            # PyQt6 / PySide6 abstraction bridge
│   ├── qt_figure.py            # Custom high-performance Matplotlib Qt widget
│   ├── theme.py                # Dark GIS theme and CSS stylesheets
│   ├── ingestion_panel.py      # Step 1: Drag-and-drop ingestion
│   ├── layer_matrix_panel.py   # Step 2: GeoPDF layer-role matrix
│   ├── config_panel.py         # Step 3: Processing & export settings
│   ├── preview_canvas.py       # Step 4: Interactive spatial viewport
│   ├── execution_worker.py     # QThread asynchronous background worker
│   └── results_viewer.py       # Step 5: Results & analytics dashboard
├── samples/
│   ├── generate_sample_data.py # Sample GeoTIFF & GeoPDF generator
│   ├── sample_geotiff.tif      # 4-band synthetic satellite scene (UTM 43N)
│   └── sample_geopdf.pdf       # Multi-page GeoPDF with neatline & vector layers
└── tests/
    ├── test_pipeline.py        # Automated backend unit tests
    └── test_gui.py             # Automated GUI integration tests
```

---

## 🚀 Quickstart Guide

### 1. Launch Interactive Desktop GUI
```bash
python3 app.py
```

### 2. Preload a File Directly into GUI
```bash
python3 app.py --input samples/sample_geotiff.tif
# or
python3 app.py --input samples/sample_geopdf.pdf
```

### 3. Run in Headless CLI Batch Mode
```bash
python3 app.py --cli --input samples/sample_geotiff.tif --output output_batch/ --crs EPSG:32643
```

### 4. Run Automated Test Suite
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

---

## 🎨 Soil Classification Taxonomy

| ID | Soil Class | Hex Color | Description | Fertility | Permeability |
|:---|:---|:---:|:---|:---:|:---:|
| **0** | **NoData / Masked** | `#000000` | Unclassified, road network, furniture, or neatline border | N/A | N/A |
| **1** | **Alluvial Soil** | `#E6B800` | Transported river sediment; loam, silt, and fine clay | Very High | Moderate |
| **2** | **Black Soil (Vertisol)** | `#362B28` | Moisture-retentive montmorillonite clay (Chernozem/Vertisol) | High | Low / Impermeable |
| **3** | **Red Soil** | `#C0392B` | Iron-oxide rich porous metamorphic soil | Medium-Low | High |
| **4** | **Lateritic Soil** | `#D35400` | Leached residual aluminosilicate tropical soil | Low / Acidic | Moderate-High |
| **5** | **Sandy / Desert Soil** | `#F39C12` | Arid quartz sand with minimal organic matter | Very Low | Very High |