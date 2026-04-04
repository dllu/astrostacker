# Astrostacker

Wide-angle astrophotography stacking pipeline for tripod-mounted RAW sequences with mixed static foreground and moving sky.

It currently does the following:

- decodes RAW files with `rawpy`
- scans RAW Bayer frames once to learn persistent hot pixels and repairs RAW-domain defects before demosaic
- extracts Fuji lens metadata with `exiv2`
- applies lens shading and radial distortion correction
- segments sky vs foreground with SAM3 when available, with a heuristic fallback
- detects stars and estimates a homography against a reference frame
- fuses sky and foreground separately
- writes debug PNGs plus a linear DNG-style TIFF output

## Setup

To get the segmentation model that separates the foreground from the night sky, you'll need to get the weights from https://huggingface.co/facebook/sam3.

```bash
uv sync
```

## Example

```bash
uv run python -m astrostacker.cli \
  --output /tmp/astrostacker-example/out.dng \
  --debug-dir /tmp/astrostacker-example/debug \
  --preview-scale 24 \
  --segmentation-downsample 4 \
  --dilation-radius 21 \
  --blur-radius 9 \
  example/2026-03-28-04-35-44_DSCF7462_131e0a50692721d760e8a1472570b4de3c3dd01a.raf \
  example/2026-03-28-04-35-54_DSCF7463_05e5f5e084d4b42941f25118e7dcbbff89cec181.raf
```

If you want to force a specific SAM3 checkpoint:

```bash
uv run python -m astrostacker.cli \
  --sam3-checkpoint /home/dllu/proj/sam3-weights/sam3.pt \
  --output /tmp/astrostacker-example/out.dng \
  --debug-dir /tmp/astrostacker-example/debug \
  --segmentation-downsample 4 \
  example/2026-03-28-04-35-44_DSCF7462_131e0a50692721d760e8a1472570b4de3c3dd01a.raf \
  example/2026-03-28-04-35-54_DSCF7463_05e5f5e084d4b42941f25118e7dcbbff89cec181.raf
```

## Outputs

- `out.dng`: stacked linear output
- `debug/sky_mask_preview.png`: cleaned sky mask
- `debug/foreground_mask_preview.png`: complementary foreground mask
- `debug/sky_weight_preview.png`: softened sky weight map
- `debug/hot_pixels_persistent_preview.png`: learned persistent hot-pixel map preview
- `debug/stars_*.png`: detected star overlays
- `debug/enhanced_stars_*.png`: high-pass star detection previews
- `debug/stacked_preview.png`: tone-mapped stack preview
