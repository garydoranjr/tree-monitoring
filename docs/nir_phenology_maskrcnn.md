# Using the Near-Infrared Band for Canopy Phenology with Mask R-CNN

This note summarizes prior work on incorporating the near-infrared
(NIR) band into convolutional tree-crown detection and phenology
models, the strategies for adapting RGB-pretrained networks to
additional spectral bands, and how those choices are implemented in
this repository's Planet Mask R-CNN pipeline.

## Why NIR matters for canopy phenology

Healthy, turgid leaves reflect strongly in the NIR (~750–900 nm) while
absorbing in the red, producing the steep "red edge" that underlies
NDVI and most vegetation indices. Phenological transitions in a
tropical canopy change exactly this signal: new leaf flush raises NIR
reflectance and NDVI; senescence and deciduousness lower it; and mass
flowering can sharply alter visible reflectance (often brightening or
reddening the crown) while NIR responds to the loss or covering of
photosynthetic leaf area. The visible bands alone confound several of
these states (a pale flowering crown and a sunlit bare crown can look
similar in RGB), whereas the red/NIR contrast separates leaf-on from
leaf-off conditions much more robustly. This is why the project's
existing observability and NDVI work (`scripts/calculate_ndvi.py`,
`scripts/crown_ndvi_scores.py`) relies on the NIR band, and why
feeding NIR directly to the detector — rather than only RGB — is
expected to improve phenology-event classification.

The Planet AnalyticMS imagery used here is natively 4-band, ordered
`[Blue(1), Green(2), Red(3), NIR(4)]` (uint16 surface DN), so the NIR
band is already available at no extra acquisition cost; the only change
needed is to stop discarding it.

## Prior work: NIR / multispectral input for CNN crown models

Mask R-CNN and related instance-segmentation networks have become a
standard tool for individual-tree-crown delineation, and several
studies show that adding NIR or using color-infrared (CIR) composites
improves detection over RGB alone:

- Color-infrared (NIR–Red–Green) input to Mask R-CNN for tree
  detection in plantations achieved high F1 scores (~0.92 in
  coniferous, ~0.85 in deciduous, ~0.83 in mixed stands), with the
  network shown to exploit multi-band information for separating
  individual trees ([Hao et al., 2021, ISPRS J. Photogramm. Remote
  Sens.](https://sciencedirect.com/science/article/pii/S0924271621001611)).
- Combining visible and NIR (and lidar) bands for instance
  segmentation with Mask R-CNN and DETR improved crown delineation
  relative to RGB-only baselines, demonstrating the value of early
  fusion of spectral channels ([Sun et al., 2023, ISPRS Open J.
  Photogramm. Remote
  Sens.](https://www.sciencedirect.com/science/article/pii/S266739322300008X)).
- Mask R-CNN delineates tropical-forest crowns accurately from
  high-resolution RGB ([Ball et al., 2023, Remote Sensing in Ecology
  and Conservation](https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.332)),
  establishing the RGB baseline that 4-band input aims to extend.

The general finding is that vegetation tasks benefit from the NIR band
because it carries most of the leaf-physiology signal, and that
detectors can use it directly when the input layer is widened to accept
it.

## Two strategies for adding NIR to a pretrained RGB model

ImageNet/COCO-pretrained backbones expect a 3-channel input. There are
two common ways to introduce NIR:

1. **Band substitution (stay 3-channel).** Replace one visible band
   (commonly blue, which is the least informative for vegetation) with
   NIR, forming an `[NIR, Green, Red]`-style CIR composite. The network
   and all pretrained weights are used unchanged. This is the simplest
   option and works with the existing RGB chip/PNG pipeline, but it
   discards the blue band and forces NIR to reuse a filter trained for a
   visible band.

2. **True 4-channel input (widen the first conv).** Replace the first
   convolution so it accepts 4 input channels, keeping all RGB
   information and adding NIR as a genuine fourth channel. This requires
   modifying the architecture and initializing the new channel's
   weights, but retains the full spectral content and lets the network
   learn an NIR-specific filter. This is the approach used by the
   project's existing `scripts/train_planet_image_segformer_4b.py`.

**This repository uses strategy (2)** for Mask R-CNN, for consistency
with the SegFormer-4b model and to keep all four bands. Strategy (1)
remains available implicitly because the 3-band RGB pipeline is
preserved (`--bands 3`, the default).

## First-conv weight initialization

When widening the first convolution from 3 to 4 input channels, the new
NIR channel's weights must be initialized. Strategies reported in the
literature include:

- **Copy a visible channel** (e.g. duplicate the pretrained red-channel
  filter into the NIR channel), exploiting the rough correlation between
  red and NIR filter responses on vegetation.
- **Average / inflate** the pretrained RGB filters across the channel
  dimension and replicate the mean into the new channel.
- **Random initialization** of the new channel while keeping pretrained
  RGB weights.
- **Zero initialization** of the new channel, so that at the start of
  training the 4-band model is numerically identical to the pretrained
  3-band model and the NIR contribution is learned gradually from a
  warm start.

These options and their trade-offs for multispectral CNNs are discussed
in, e.g., [Lin et al., 2024, "Impact of architecture on robustness and
interpretability of multispectral deep neural
networks"](https://arxiv.org/pdf/2309.12463) and broader transfer-from-RGB
literature.

**This repository uses zero-initialization** for the NIR channel (RGB
channels copy the pretrained COCO weights), matching
`modify_model_for_4bands` in `scripts/train_planet_image_segformer_4b.py`.
This gives a stable warm start: the detector behaves exactly like the
pretrained RGB model on the first step and only deviates as the NIR
filter learns to contribute.

## Normalization and input scaling

Raw Planet bands are uint16 digital numbers with very different dynamic
ranges per band. The pipeline applies a **per-band 0–99.9th-percentile
stretch to [0, 1]** at load time, then a per-channel mean/std
normalization. For the 4-band models the ImageNet RGB statistics are
reused for the NIR channel (mean `0.485`, std `0.229` — the red-channel
values), so the NIR channel is normalized on the same scale as the
visible bands:

- `image_mean = [0.485, 0.456, 0.406, 0.485]`
- `image_std  = [0.229, 0.224, 0.225, 0.229]`

In Mask R-CNN these are set on the detection transform
(`model.transform.image_mean` / `image_std`), which normalizes each
channel and therefore requires one value per band.

## Implementation in this repository

Two scripts gained an opt-in 4-band mode while preserving the existing
3-band RGB behavior as the default:

- `scripts/apply_drone_labels_coreg.py` — `-b/--bands 4` discovers the
  raw `*4band.tif` Planet scenes (instead of `*rgb.tif`) and writes a
  4-band uint16 GeoTIFF training chip (`{scene}.tif`, CRS/transform/
  nodata preserved) alongside the usual RGB QA `{scene}.png`,
  `{scene}.mask.png`, and optional `.drone.png` / `.ocm.png`.
  Coregistration is matched on the Red band in both modes (band 3 for
  4-band, band 1 for RGB) so alignment is unchanged.
- `scripts/train_planet_image_maskrcnn.py` — `--bands 4` loads the
  4-band GeoTIFF chips (per-band percentile stretch, channel-aware
  ColorJitter), replaces the ResNet-50 backbone's `conv1` with a
  4-input-channel convolution (RGB weights copied, NIR zero-init), and
  extends the detection-transform normalization to four bands.

The design mirrors the existing 4-band SegFormer model
(`scripts/train_planet_image_segformer_4b.py`), which serves as the
reference implementation for 4-band loading, augmentation, and
first-layer surgery.

## References

- Hao, Z., et al. (2021). Automated tree-crown and height detection in
  a young forest plantation using Mask R-CNN. *ISPRS Journal of
  Photogrammetry and Remote Sensing.*
  <https://sciencedirect.com/science/article/pii/S0924271621001611>
- Sun, Y., et al. (2023). Towards complete tree crown delineation by
  instance segmentation with Mask R-CNN and DETR using UAV-based
  multispectral imagery and lidar data. *ISPRS Open Journal of
  Photogrammetry and Remote Sensing.*
  <https://www.sciencedirect.com/science/article/pii/S266739322300008X>
- Ball, J. G. C., et al. (2023). Accurate delineation of individual
  tree crowns in tropical forests from aerial RGB imagery using Mask
  R-CNN. *Remote Sensing in Ecology and Conservation.*
  <https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.332>
- Lin, S., et al. (2024). Impact of architecture on robustness and
  interpretability of multispectral deep neural networks. *arXiv.*
  <https://arxiv.org/pdf/2309.12463>
