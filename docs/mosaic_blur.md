# Effective resolution and motion blur of the 50ha drone mosaics

Script: [`scripts/measure_mosaic_blur.py`](../scripts/measure_mosaic_blur.py).
Outputs (October 2026 run): `/Volumes/Earth03/flower/figs/202610_updates/`
(`mosaic_blur.csv`, `blur_exif.csv`, `blur_validation.csv`, `blur_*.png/.pdf`);
spectra cache in `/Volumes/Earth03/flower/results/mosaic_blur/`.

## Question

The phantom (Phantom 4 Pro) and mavic (Mavic 3M) mosaics could differ in
effective resolution through motion blur. The phantom camera was thought to
have loosened over the years, which would make its blur grow over time. This
measures, for every **global-alignment** mosaic:
- the point-spread function (PSF) width in cm;
- the blur's dominant direction;
- the length of the motion that would produce that blur.

Local-alignment mosaics are left out because their warping could read as blur.

## Result

- **Effective resolution:** about 7.1–7.4 cm PSF FWHM, or about 1.6 pixels,
  for every series. Median isotropic FWHM:
  - phantom 24782016: 7.13 cm (1.58 px);
  - phantom C3KW2X: 7.27 cm (1.61 px);
  - mavic: 7.37 cm (1.56 px).
- **Mavic vs phantom:** the mavic figure is slightly larger in cm only because
  its GSD is coarser (median 4.73 vs 4.51 cm). Across mavic flights, FWHM
  tracks GSD (Spearman ρ = 0.73), so its resolution is set by flight height
  rather than motion.
- **No sign of the phantom camera loosening, 2018–2023:**
  - Isotropic FWHM against date: ρ = −0.14 (p = 0.17); a fit with exposure
    time gives −0.06 cm/yr.
  - Measured motion length minus EXIF-predicted forward motion, against date:
    ρ = −0.09 (p = 0.42).
  - The early-2018 flights are the blurriest phantom mosaics: 8.9 cm on
    2018-04-25 and 9.2 cm on 2018-05-23, both with 1/100 s exposures.
- **Phantom blur is forward motion during the exposure:**
  - It runs along the flight lines: axial-mean azimuth 5° against north–south
    lines, mean resultant length R = 0.70.
  - Its length matches EXIF flight speed × median exposure time (r = 0.75;
    median measured/predicted = 0.95; median 3.9 cm, about 0.9 px).
  - The major-axis FWHM grows by about 0.065 cm per ms of exposure.
- **Mavic** has a larger directional component: a median motion length of
  5.9 cm along azimuth 16° (R = 0.97). Its flight lines and exposure times are
  not recorded locally (`metadata_all.csv` has neither), so this can't yet be
  tied to forward motion. GNSS correction (PPK, RTK, none) makes no
  difference.
- **C3KW2X** phantom mosaics (2023-10 to 2024-03, no EXIF) blur along about
  143° instead of north–south, with a median motion length of 3.0 cm.
  - The 2024 release reprocessed every date, and its pixel-scale gradient
    energy (Tenengrad) is about half that of 24782016 at the same FWHM.
  - So this may come from processing or from a changed flight plan.

## Method

**Tiles.** A fixed grid of 60 tiles, each 20.48 m square (10 × 6 centres),
inside the phantom 50ha grid, inset 30 m from its edge. Each tile is read at
native resolution from every mosaic, with no resampling: 90 + 16 phantom and
96 mavic. Tiles are converted to luminance.
- Mavic tiles with any alpha = 0 pixel are dropped.
- Phantom has no alpha band (band 4 is height), and deep shadows clip to
  RGB = 0. So a phantom tile is dropped only if more than 0.5 % of it is
  black.
- All 60 tiles were usable in all 202 mosaics.

**Spectra.** For each tile: subtract the mean, apply a 2D Hann window, take
|FFT|², and bin it.
- Bins are in physical frequency: radial bins of 0.25 cycles/m up to
  9.5 cycles/m (below every mosaic's Nyquist limit), and 12 orientation bins
  over 180°.
- Using physical frequency lets mosaics with different GSDs (4.1–5.2 cm) share
  bins without resampling.
- The noise floor is the mean power beyond 0.9 × Nyquist.

**Isotropic width.** Blur multiplies the spectrum by |MTF|², so on the same
patch of ground the scene cancels in the ratio of two mosaics' spectra.
- For a Gaussian-equivalent PSF with covariance Σ:
  ln[P_i(f)/P_ref(f)] = c − 4π² fᵀ(Σ_i − Σ_ref) f. This is linear in ΔΣ.
- The free offset c absorbs differences in contrast, illumination and sensor
  gain. Spectra ignore shifts, so the ~0.5 m offsets between global mosaics
  don't matter.
- P_ref is the per-tile, per-bin median over all 202 mosaics.
- ΔΣ is fitted per tile by weighted least squares:
  - only bins between 0.75 cycles/m and where either spectrum falls to 4× its
    noise floor;
  - iterated so that only bins with predicted attenuation below e³ are used,
    where the Gaussian model holds.
- Tiles are combined by median, with a 500-draw bootstrap over tiles.
- **Absolute anchor:** fitting P = A f^−α exp(−4π²σ²f²) + N to the radial
  reference spectrum gives σ_ref = 3.09 ± 0.10 cm (FWHM 7.28 cm), with
  α = 1.54.
- Each mosaic's isotropic σ² = σ_ref² + tr(ΔΣ)/2. Fitting the same model to
  each mosaic's own spectrum agrees (`blur_anchor_check`).
- The absolute anchor depends on the power-law scene model. Differences
  between mosaics do not.

**Direction and motion length.** The ratio method can't be used for
direction. The anisotropy of the median reference is unknown, and relative to
a reference that mixes both drones, any difference between the drones shows
up as perpendicular blur in each.
- Instead, each tile's own spectrum is split by frequency dependence. At each
  radial bin, the second angular harmonic of ln P is regressed on
  [1, 4π²f²] over 0.5–6 cycles/m:
  - the f² term is the deviatoric part of the blur covariance
    [[a, b], [b, −a]];
  - the frequency-flat term is the canopy texture's own anisotropy.
- **Why this works:** in the phantom mosaics the texture term is within ±15°
  of the solar azimuth at mid-flight for 76 % of mosaics. The blur term is
  within ±15° of the sun for only 10 %, and within ±15° of the flight line for
  58 % (81 % within ±30°) (`blur_exif_crosscheck` (c)).
- A first version measured direction against an all-dates reference. Its blur
  direction ran perpendicular to the sun and swung with the season.
- Major/minor σ² = isotropic σ² ± √(a² + b²).
- Motion length L = √(24·√(a² + b²)), because a linear motion of length L adds
  L²/12 of variance along its direction only.

**Validation** (`validate`, mosaic 2019-04-24): the tiles were blurred with
known kernels, re-quantized and refitted.
- Isotropic Gaussian σ of 0.5–3 px is recovered to within 0.1 cm.
- Motion of 1–4 px (5–20 cm) is recovered with lengths within +5 to +20 %
  and azimuths within 1–8°. At 1–2 px and 30°, the azimuth is 13–22° off.
- Longer motion is underestimated, because the box blur's first spectral zero
  enters the fit band. The real mosaics show 2–9 cm.
- Spurious anisotropy from isotropic blur stays at or below 1 cm up to
  σ = 1.5 px.

**EXIF cross-check** (24782016 only; there is no EXIF for C3KW2X or mavic).
Per mission:
- median `ExposureTime`;
- flight speed, as total distance ÷ total time over straight runs of
  consecutive GPS fixes (timestamps are only to 1 s);
- flight-line azimuth;
- solar position at mid-flight (EXIF clock taken as UTC−5).

**Mavic flight metadata** (`stri/globus/metadata_all.csv`) gives each
flight's GNSS correction, processing comments and native GSD, which are
joined into `mosaic_blur.csv`.

## Caveats

- Wind moving the canopy during an exposure counts as blur. Phenology changes
  the texture. Both are reduced, but not removed, by taking medians over 60
  tiles and using the scene-cancelling ratio.
- Each tile mixes source photos blended by the mosaicking software.
- The Gaussian-equivalent width summarizes the PSF's second moment, not its
  shape. Pixel-scale sharpening or smoothing above 9.5 cycles/m is outside the
  fit; this is why C3KW2X's Tenengrad is low while its FWHM is normal.

## Reproduce

```
python scripts/measure_mosaic_blur.py tiles
python scripts/measure_mosaic_blur.py spectra     # ~45 min from Earth03, restartable
python scripts/measure_mosaic_blur.py validate
python scripts/measure_mosaic_blur.py exif
python scripts/measure_mosaic_blur.py fit
python scripts/measure_mosaic_blur.py plot
```

## Figure captions

**`blur_resolution_timeseries`** — Effective resolution of each 50ha
global-alignment drone mosaic, as the full width at half maximum (FWHM, cm)
of a Gaussian-equivalent point-spread function. Each mosaic has two points:
the major axis (filled) and the minor axis (open), joined by a vertical line.
Series: phantom 24782016 (blue circles, 2018-04-04 – 2023-10-24), phantom
C3KW2X (green triangles, 2023-10-31 – 2024-03-18) and mavic (red squares,
2024-03-06 – 2026-01-20). Each value is the median over 60 tiles of 20.48 m.
The isotropic width comes from the tile power spectrum's ratio to the median
spectrum of all 202 mosaics, put in absolute terms by a power-law ×
Gaussian-MTF fit to that median (σ = 3.09 cm). The major/minor split comes
from the part of each mosaic's own spectral anisotropy that grows as f².
Dashed lines show twice the mosaic GSD (the Nyquist-limited resolution). The
dotted vertical line marks the 24782016 → C3KW2X release boundary.

**`blur_anisotropy`** — Directional (motion) blur of each global-alignment
mosaic. Top: equivalent linear-motion length (cm, √24 × the half-difference
of the PSF variances along the major and minor axes), with 95 % bootstrap
intervals over the 60 tiles, for phantom 24782016 (blue circles,
2018-04-04 – 2023-10-24), phantom C3KW2X (green triangles,
2023-10-31 – 2024-03-18) and mavic (red squares, 2024-03-06 – 2026-01-20).
The black line is the forward motion predicted from the 24782016 EXIF (flight
speed × median exposure time per mission). Bottom: azimuth of the blur major
axis (degrees from north, modulo 180). Points are filled where the motion
length's lower bound exceeds half a pixel. The black ticks at 0° are the
24782016 flight-line azimuths from EXIF GPS tracks.

**`blur_exif_crosscheck`** — Phantom 24782016 blur against flight parameters
from the raw-image EXIF, one point per mosaic (2018-04-04 – 2023-10-24,
coloured by year). (a) Measured motion length (cm) against predicted forward
motion (flight speed × median exposure time, cm); the dashed line is 1:1.
(b) Major-axis PSF FWHM (cm) against median exposure time (ms). (c)
Histograms (10° bins) of three azimuth differences:
- the frequency-flat canopy-texture anisotropy minus the solar azimuth at
  mid-flight (orange);
- the f²-scaling blur anisotropy minus the solar azimuth (grey);
- the blur anisotropy minus the flight-line azimuth (blue).

**`blur_spectra_examples`** — Spectral evidence for blur differences, with
the sharpest and blurriest mosaic of each drone by isotropic FWHM. Left: the
median over 60 tiles of the log ratio of each mosaic's angle-averaged power
spectrum to the all-mosaic median spectrum, against squared spatial frequency
(cycles² m⁻²), offset to zero at 0.75 cycles/m. Thick lines are phantom
24782016 (blue) and mavic (red), solid for the sharpest and dashed for the
blurriest. The thin lines are the fitted Gaussian Δσ², which is a straight
line on this axis. Right: the same 8 m crop from each of the four mosaics at
native resolution, each with a 1–99 % per-crop stretch.

**`blur_anchor_check`** — Checks on the absolute scale. (a) The radial
reference spectrum (black, median over all 202 mosaics and 60 tiles, power
normalized to the 0.25–1 cycles/m band) with the fitted model
A f^−α exp(−4π²σ²f²) + N (red). (b) Isotropic σ (cm) from the spectral-ratio
method against σ from fitting the same model to each mosaic's own spectrum.
Series: phantom 24782016 (blue circles), phantom C3KW2X (green triangles) and
mavic (red squares); the dashed line is 1:1. (c) Median tile Tenengrad
(gradient energy divided by tile variance) against PSF FWHM in pixels, same
series.

**`blur_mavic_metadata`** — Mavic (2024-03-06 – 2026-01-20) blur against
flight metadata from `stri/globus/metadata_all.csv`. (a) Isotropic PSF FWHM
(cm) and (b) motion length (cm) against mosaic GSD (cm), coloured by GNSS
correction (PPK, RTK or none). Open circles mark flights whose processing
comments note illumination artifacts, an incomplete mission or a distorted
DSM.

**`blur_validation`** — Recovery of known synthetic blur applied to the 60
tiles of the phantom 2019-04-24 global mosaic (GSD 4.51 cm). The blurred
tiles were re-quantized to integers and refitted against the unblurred tiles.
(a) Recovered isotropic Δσ against injected Gaussian σ (0.5–3 px). (b)
Recovered against injected linear-motion length, where injected length is
the discrete kernel's own second moment, at azimuths 0°, 30°, 90° and 135°.
(c) Recovered minus injected azimuth. Dashed lines are 1:1 (a, b) or zero
error (c). The shaded band (over 4.5 px) is where the motion kernel's first
spectral zero falls inside the 0.5–6 cycles/m fit band and the Gaussian
model fails.
