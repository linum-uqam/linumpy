# Mosaic Grid Format (OME-Zarr)


---

## Overview

Mosaic grids in linumpy are stored in the **OME-Zarr** format, a cloud-optimized, chunked array format designed for large microscopy datasets. The format supports multi-resolution pyramids, metadata, and efficient partial reads.

---

## File Structure

### Directory Layout

```
mosaic_grid_3d_z00.ome.zarr/
├── .zattrs                 # Root attributes (OME metadata)
├── .zgroup                 # Zarr group marker
├── 0/                      # Full resolution level
│   ├── .zarray             # Array metadata
│   └── <chunks>/           # Data chunks
├── 1/                      # 2x downsampled (optional)
│   ├── .zarray
│   └── <chunks>/
├── 2/                      # 4x downsampled (optional)
│   └── ...
└── ...
```

### Resolution Levels

Two pyramid creation modes are available:

#### Traditional Power-of-2 Pyramids (Mosaic Grids)

Used during preprocessing for intermediate mosaic grids:
- **Level 0**: Full resolution
- **Level 1**: 2x downsampled
- **Level 2**: 4x downsampled
- etc.

The number of levels is controlled by `--n_levels` parameter during creation.

#### Analysis-Optimized Pyramids (Final 3D Volume)

For the final stacked 3D volume, specific analysis-friendly resolutions are used:

| Level | Resolution | Use Case |
|-------|------------|----------|
| 0 | 10 µm | High-resolution cellular analysis |
| 1 | 25 µm | Standard analysis resolution |
| 2 | 50 µm | Overview, atlas registration |
| 3 | 100 µm | Quick visualization, large-scale |

This is controlled by the `--pyramid_resolutions` parameter in `linum_stack_slices_motor.py` or the `pyramid_resolutions` parameter in the Nextflow workflow.

**Note:** Only resolutions ≥ the base processing resolution are included. For example, processing at 25 µm will create levels at 25, 50, and 100 µm.

---

## Array Dimensions

### 3D Mosaic Grid

```
Shape: (Z, X, Y)
```

| Dimension | Description |
|-----------|-------------|
| Z | Depth/axial dimension |
| X | First lateral dimension |
| Y | Second lateral dimension |

### 2D Mosaic Grid (AIP)

```
Shape: (X, Y)
```

---

## Metadata

### OME Metadata (`.zattrs`)

```json
{
    "multiscales": [{
        "version": "0.4",
        "name": "mosaic_grid_3d_z00",
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [{
                    "type": "scale",
                    "scale": [1.5, 10.0, 10.0]
                }]
            },
            {
                "path": "1",
                "coordinateTransformations": [{
                    "type": "scale",
                    "scale": [1.5, 20.0, 20.0]
                }]
            }
        ]
    }]
}
```

### Key Metadata Fields

| Field | Description |
|-------|-------------|
| `axes` | Dimension names, types, and units |
| `datasets` | Resolution levels with paths and transforms |
| `coordinateTransformations` | Pixel-to-physical coordinate mapping |
| `scale` | Resolution per dimension (µm/pixel) |

---

## Reading OME-Zarr Files

### Using linumpy

```python
from linumpy.io.zarr import read_omezarr

# Read image and resolution
image, resolution = read_omezarr("mosaic_grid_3d_z00.ome.zarr")

# image: dask.array.Array with shape (Z, X, Y)
# resolution: tuple (res_z, res_x, res_y) in mm/pixel

print(f"Shape: {image.shape}")
print(f"Resolution: {resolution} mm/pixel")
print(f"Chunks: {image.chunks}")
```

### Using zarr directly

```python
import zarr

# Open store
store = zarr.open("mosaic_grid_3d_z00.ome.zarr", mode='r')

# Read full resolution
data = store['0'][:]

# Read downsampled level
data_2x = store['1'][:]

# Access metadata
import json
with open("mosaic_grid_3d_z00.ome.zarr/.zattrs") as f:
    metadata = json.load(f)
```

### Using napari

```bash
napari mosaic_grid_3d_z00.ome.zarr
```

---

## Writing OME-Zarr Files

### Using linumpy

```python
from linumpy.io.zarr import save_omezarr
import dask.array as da
import numpy as np

# Create sample data
data = da.from_array(np.random.rand(100, 512, 512), chunks=(10, 128, 128))
resolution = (0.0015, 0.010, 0.010)  # mm/pixel

# Save
save_omezarr(data, "output.ome.zarr", resolution, chunks=data.chunks)
```

### With Multiple Resolution Levels

```python
from linumpy.io.zarr import OmeZarrWriter, AnalysisOmeZarrWriter

# Traditional power-of-2 pyramid (2x downsampling per level)
writer = OmeZarrWriter(
    "output.ome.zarr",
    shape=(100, 512, 512),
    chunks=(10, 128, 128),
    dtype=np.float32
)
writer[:] = data[:]
writer.finalize(resolution, n_levels=5)  # Creates levels 0-5

# Analysis-optimized pyramid (specific resolutions in µm)
writer = AnalysisOmeZarrWriter(
    "output.ome.zarr",
    shape=(100, 512, 512),
    chunks=(10, 128, 128),
    dtype=np.float32
)
writer[:] = data[:]
writer.finalize(resolution, [10, 25, 50, 100])  # Creates 10, 25, 50, 100 µm levels
# or simply: writer.finalize(resolution)  # uses default [10, 25, 50, 100]
```

---

## Chunking and Sharding

### Chunk Size

Chunks determine how data is stored and accessed. Optimal chunk sizes balance:
- Read/write performance
- Compression efficiency
- Memory usage

Typical values:
```
3D: (16, 128, 128) to (64, 256, 256)
2D: (256, 256) to (1024, 1024)
```

### Sharding (Zarr v3)

Sharding groups multiple chunks into larger files, reducing filesystem overhead:

```
sharding_factor = 4  # 4x4 chunks per shard
```

Controlled by `--sharding_factor` parameter in `linum_create_mosaic_grid_3d.py`.

---

## Data Types

### Common Types

| Type | Description | Typical Use |
|------|-------------|-------------|
| `uint8` | 0-255 | Display, compressed |
| `uint16` | 0-65535 | Raw microscopy data |
| `float32` | 32-bit float | Processed data |
| `float64` | 64-bit float | High-precision processing |

### Type Conversion

```python
# During reading
image = image.astype(np.float32)

# During writing (specify dtype)
save_omezarr(data.astype(np.uint16), "output.ome.zarr", resolution, ...)
```

---

## Naming Convention

### Mosaic Grid Files

```
mosaic_grid_3d_z{slice_id}.ome.zarr
```

- `mosaic_grid_3d`: Indicates 3D mosaic grid
- `z{slice_id}`: Two-digit slice identifier (e.g., `z00`, `z01`, `z15`)
- `.ome.zarr`: OME-Zarr format extension

### Examples

```
mosaic_grid_3d_z00.ome.zarr  # First slice
mosaic_grid_3d_z01.ome.zarr  # Second slice
mosaic_grid_3d_z15.ome.zarr  # 16th slice
```

### Processed Files

```
slice_z{slice_id}_{process}.ome.zarr
```

Examples:
```
slice_z00_stitch_3d.ome.zarr
slice_z00_axial_corr.ome.zarr
slice_z00_normalize.ome.zarr
```

---

## Viewing OME-Zarr Files

### Napari (Recommended)

```bash
pip install napari[all]
napari mosaic_grid_3d_z00.ome.zarr
```

### Neuroglancer

OME-Zarr files can be viewed in Neuroglancer if hosted on a web server.

### Using linumpy Scripts

```bash
# View OME-Zarr
linum_view_omezarr.py mosaic_grid_3d_z00.ome.zarr

# Take screenshot
linum_screenshot_omezarr.py mosaic_grid_3d_z00.ome.zarr screenshot.png
```

---

## Compression

### Supported Codecs

| Codec | Description | Use Case |
|-------|-------------|----------|
| `blosc` | Fast, general purpose | Default |
| `zstd` | High compression ratio | Archival |
| `lz4` | Very fast | Real-time |
| `gzip` | Widely compatible | Exchange |

### Default Configuration

linumpy uses `blosc` with `lz4` compressor by default.

---

## Troubleshooting

### File Won't Open

```bash
# Check file structure
ls -la mosaic_grid_3d_z00.ome.zarr/

# Verify .zattrs exists
cat mosaic_grid_3d_z00.ome.zarr/.zattrs
```

### Memory Issues

```python
# Use dask for lazy loading
image, res = read_omezarr("large_file.ome.zarr")
# image is a dask array - not loaded into memory

# Load only a subset
subset = image[0:10, 100:200, 100:200].compute()
```

### Wrong Dimensions

Check the axes metadata in `.zattrs` to verify dimension order.

---

## Related Tools

| Tool | Purpose |
|------|---------|
| `linum_create_mosaic_grid_3d.py` | Create mosaic from raw tiles |
| `linum_resample_mosaic_grid.py` | Resample to different resolution |
| `linum_convert_omezarr_to_nifti.py` | Convert to NIfTI format |
| `linum_view_omezarr.py` | Interactive viewer |
| `linum_screenshot_omezarr.py` | Generate preview images |
