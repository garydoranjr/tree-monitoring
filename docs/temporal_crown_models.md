# Models for classifying known crowns from a Planet time series with cloud masking

This note surveys models that classify tree crowns from a **time series** of
images, skipping cloudy, shadowed or hazy observations in both training and
evaluation. It ends with the model chosen for a prototype
([crown_temporal_model.md](crown_temporal_model.md)).

## Why a temporal crown model

Every Planet model in this repo sees one image at a time: the Mask R-CNN, the
DINOv3 segmenter and the SegFormers. The "temporal CV" of
[planet_temporal_generalization.md](planet_temporal_generalization.md) is
blocked cross-validation over dates, not a temporal model. Its job is
interpolation between drone flights, and there the persistence baseline beats
the Mask R-CNN out to 30 days. Copying the drone labels of the nearest training
flight is all the persistence baseline does.

The crowns are already mapped (the 2022-09-29 improved crown map, 2,280
crowns). The detection problem can therefore be recast as **classifying known
crowns from their own Planet series**: for each crown and Planet date, predict
the flowering-or-deciduous `event`, using every clear observation of that crown
before and after the date.

## Requirements

- **Irregular, gappy dates.** About 2,100 PlanetScope scenes fall between 2020
  and 2026, from 0 to several per day, with long cloudy gaps in the wet season.
- **Masking at two levels:**
  - pixels inside a crown that OmniCloudMask (OCM) calls cloud, thin cloud or
    shadow, `scripts/cloud_mask_planet.py`;
  - whole crown-dates that are mostly obscured.

  Neither may enter the input or the loss.
- **Data.** 4-band B/G/R/NIR TOA at 3 m. A crown of 50 m² covers only about 5
  pixels, so a texture-based CNN gets little to work with.
- **Labels.** Labels exist only on drone-flight dates: crown × flight tables
  `crown_classifications*.nc4` (README section 3). Within ±2 days of each flight
  there are Planet scenes to attach them to.

## Candidates

| model | input unit | how clouds and gaps are handled | fit here |
|---|---|---|---|
| **PSE + TAE** (Garnot et al. 2020) and **L-TAE** (Garnot & Landrieu 2020) | one object (parcel) per date, as an unordered set of pixels | Each date is embedded from a pixel set, so cloudy pixels can simply be left out. Dates are positioned by days since the first observation, not by index, "to account for inconsistent temporal sampling" (Garnot et al. 2020, p. 5). Invalid dates can be masked out of attention. | **Best.** A crown is a parcel. The model is small and trains from scratch, so it needs no pretraining on PlanetScope bands. |
| U-TAE / PaPs (Garnot & Landrieu 2021) | dense image series (PASTIS) | Padded dates are masked for the whole image | Dense segmentation, one map per series rather than per date. It has no per-pixel cloud mask and would need retraining on 3 m 4-band data. |
| TSViT (Tarasiou et al. 2023) | dense image series | No per-pixel validity mask | Same objections as U-TAE |
| Presto (Tseng et al. 2023) | per-pixel time series | Masks per timestep and per band (`mask` argument) | Built on **monthly** Sentinel-1/2 timesteps, which loses the sub-weekly flowering signal |
| Galileo (Tseng et al. 2025) | multimodal space-time tokens | Token masking during pretraining | 24 monthly timesteps at 10 m |
| AnySat (Astruc et al. 2025) | multi-resolution, multimodal | JEPA pretraining. It handles any resolution, but its GeoPlex pretraining does not include PlanetScope. | Possible encoder later, if labels prove too few |
| OlmoEarth (2025) | multimodal | Bandset masking during pretraining | Pretrained on Sentinel-1/2 and Landsat-8, not PlanetScope |
| Prithvi-EO-2.0 (Szwarcman et al. 2024) | HLS image series | MAE pretraining | 30 m HLS with a few fixed timesteps |
| TESSERA (2025), AlphaEarth (Brown et al. 2025) | per-pixel annual embeddings | Trained to be invariant to which clear observations are sampled. TESSERA samples 40 valid timesteps per year (p. 17). | Annual, 10 m, Sentinel or Landsat, so too coarse in time and space. The augmentation idea is borrowed: random date dropout. |
| Frozen DINOv3 crown embedding per date + temporal attention | crown | The same date masking as PSE-LTAE | A follow-up variant. The repo already loads DINOv3 (`train_planet_image_dinov3.py`). |

The PlanetScope crown-phenology literature mostly fits curves to vegetation-index
series of individual crowns rather than training a classifier:
- Wu et al. 2020 (temperate autumn phenology);
- the Sahel NDVI study (RSE 2025);
- the spring white-flower study (ISPRS 2025).

Chen et al. 2024 is the closest deep model: a 3D CNN plus LSTM on 3 m Planet
series for crowns delineated from the UAV.

At BCI, Lee et al. 2023 classified flowering from the drone series and trimmed
each crown by a 2 m buffer against misalignment between dates, as Park et al.
2019 did. The pixel extraction here does the same.

## Choice

**PSE with a temporal transformer, one output per date** (call it PSE-LTAE):
- **Pixels.** On every date, each crown becomes the set of its OCM-clear pixels
  inside a buffered crown polygon.
- **Dates.** A crown-date with too few clear pixels is masked out of the
  attention and out of the loss.
- **Output.** The original L-TAE pools the whole series into one embedding. Here
  every date gets its own output instead, so the model predicts the event on
  each Planet date from the crown's clear observations around it.

Two departures from Garnot et al. 2020:
1. **Pixel sampling.** They draw one pixel set per parcel and reuse it on every
   date. Here the set is drawn per date, from that date's clear pixels only.
2. **Padding.** They repeat pixels to pad small parcels. Here padding is masked
   out of the pooling, which they recommend for unbiased estimates.

Not done yet:
- **Haze** is caught only by the crown-visibility random forest
  ([observability_methodology.md](observability_methodology.md)), which is not
  wired in.
- **Terrain and illumination shade** is not masked anywhere.

## References

Library ids are Scholium ids.

- Garnot, Landrieu, Giordano & Chehata 2020, *Satellite image time series
  classification with pixel-set encoders and temporal self-attention*, CVPR,
  doi:10.1109/cvpr42600.2020.01234. Code: github.com/VSainteuf/pytorch-psetae.
- Garnot & Landrieu 2020, *Lightweight temporal self-attention for classifying
  satellite image time series*, arXiv:2007.00586.
- Garnot & Landrieu 2021, *Panoptic segmentation of satellite image time series
  with convolutional temporal attention networks*, ICCV,
  doi:10.1109/iccv48922.2021.00483.
- Tarasiou, Chavez & Zafeiriou 2023, *ViTs for SITS*, CVPR,
  doi:10.1109/cvpr52729.2023.01004.
- Tseng et al. 2023, *Lightweight, pre-trained transformers for remote sensing
  timeseries* (Presto), arXiv:2304.14065.
- Tseng et al. 2025, *Galileo*, arXiv:2502.09356.
- Astruc et al. 2025, *AnySat*, arXiv:2412.14123.
- *OlmoEarth*, arXiv:2511.13655.
- Szwarcman et al. 2024, *Prithvi-EO-2.0*, arXiv:2412.02732.
- *TESSERA*, arXiv:2506.20380.
- Brown et al. 2025, *AlphaEarth Foundations*, arXiv:2507.22291.
- Chen et al. 2024, *Using time-series imagery and 3DLSTM model to classify
  individual tree species*, Int. J. Digital Earth,
  doi:10.1080/17538947.2024.2308728.
- Lee et al. 2023, ISPRS J., doi:10.1016/j.isprsjprs.2023.05.022.
- Wu et al. 2020, ISPRS J., doi:10.1016/j.isprsjprs.2020.10.017.
- Wang et al. 2021 (STI-ACSS), RSE, doi:10.1016/j.rse.2021.112604.
- Wang et al. 2022 (OCBGF), RSE, doi:10.1016/j.rse.2022.113136.
- Ball et al. 2024, doi:10.2139/ssrn.4875595.
