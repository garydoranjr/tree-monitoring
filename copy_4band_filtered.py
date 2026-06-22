"""
Mirror the RGB-filtered chip subset to the 4-band chip set.

For every scene prefix present in the filtered RGB directory, copy the
matching `<prefix>_4band.*` chips from the full 4-band stretch-stats
directory into the parallel `_filt` directory. The single
`coreg_log.json` is also copied through.
"""

import os
import shutil

SRC_4BAND = "/Volumes/Earth03/flower/20260608_full_label_application_x4_coreg_4band_stretch_stats"
REF_RGB_FILT = "/Volumes/Earth03/flower/20260608_full_label_application_x4_coreg_rgb_stretch_stats_filt"
DST_4BAND_FILT = "/Volumes/Earth03/flower/20260608_full_label_application_x4_coreg_4band_stretch_stats_filt"

LOG_NAME = "coreg_log.json"

os.makedirs(DST_4BAND_FILT, exist_ok=True)


def rgb_prefix(name: str) -> str:
    """RGB chips are `<prefix>.<suffix>` -- strip from first dot."""
    return name.split(".", 1)[0]


def fourband_prefix(name: str) -> str | None:
    """4-band chips are `<prefix>_4band.<suffix>`; return None if unmatched."""
    marker = "_4band."
    idx = name.find(marker)
    if idx == -1:
        return None
    return name[:idx]


# Build the set of allowed scene prefixes from the reference RGB dir.
allowed_prefixes = set()
for name in os.listdir(REF_RGB_FILT):
    if name == LOG_NAME:
        continue
    allowed_prefixes.add(rgb_prefix(name))

print(f"Reference rgb_filt dir : {REF_RGB_FILT}")
print(f"Source 4band dir       : {SRC_4BAND}")
print(f"Destination 4band_filt : {DST_4BAND_FILT}")
print(f"Allowed prefixes       : {len(allowed_prefixes)}")

copied = 0
skipped = 0
matched_prefixes = set()

for name in sorted(os.listdir(SRC_4BAND)):
    src = os.path.join(SRC_4BAND, name)
    if not os.path.isfile(src):
        continue

    if name == LOG_NAME:
        shutil.copy2(src, os.path.join(DST_4BAND_FILT, name))
        copied += 1
        print(f"  Copied (log): {name}")
        continue

    prefix = fourband_prefix(name)
    if prefix is None:
        skipped += 1
        continue

    if prefix in allowed_prefixes:
        shutil.copy2(src, os.path.join(DST_4BAND_FILT, name))
        copied += 1
        matched_prefixes.add(prefix)
    else:
        skipped += 1

unmatched = allowed_prefixes - matched_prefixes
print()
print(f"Copied                            : {copied}")
print(f"Skipped (no prefix match)         : {skipped}")
print(f"Unique 4-band prefixes copied     : {len(matched_prefixes)}")
print(f"rgb_filt prefixes with no 4-band  : {len(unmatched)}")
if unmatched:
    print("  (first 10):")
    for p in sorted(unmatched)[:10]:
        print(f"    {p}")
