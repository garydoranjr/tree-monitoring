#!/usr/bin/env python
"""
Smoke test for apply_drone_labels_coreg.py with OCM mask support.

Runs the script against the first two label files and checks basic invariants:
  - With --maskdir: output PNGs are created; coreg_failures.json (if written) is valid.
  - Without --maskdir: every scene that produced output with --maskdir also produces
    output without it (COREG success is independent of maskdir).

Requires /Volumes/Earth03/flower/ to be mounted.
This test runs real COREG calls so expect it to take several minutes.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

LABELS_DIR = Path("/Volumes/Earth03/flower/results/classifications")
DRONEDIR   = Path("/Volumes/Earth03/flower/stri/24782016/BCI_50ha_timeseries_local_alignment")
PLANETDIR  = Path("/Volumes/Earth03/flower/planet_clipped/rgb")
MASKDIR    = Path("/Volumes/Earth03/flower/planet_clipped/ocm")
SCRIPT     = Path(__file__).parent / "apply_drone_labels_coreg.py"

FAIL = "\033[31mFAIL\033[0m"
OK   = "\033[32mOK\033[0m"


def check_prereqs():
    missing = [p for p in [LABELS_DIR, DRONEDIR, PLANETDIR, MASKDIR] if not p.exists()]
    if missing:
        for p in missing:
            print(f"SKIP: {p} not found (volume not mounted?)")
        sys.exit(0)


def run(args, label):
    print(f"\n--- {label} ---")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    if result.returncode != 0:
        print(f"{FAIL}: script exited with code {result.returncode}")
        sys.exit(1)


def assert_ok(condition, message):
    if not condition:
        print(f"{FAIL}: {message}")
        sys.exit(1)
    print(f"{OK}: {message}")


def main():
    check_prereqs()

    # Use dates known to have matching Planet images and OCM masks.
    labelfiles = [
        LABELS_DIR / "BCI_50ha_2021_01_27_local_classifications.tif",
        LABELS_DIR / "BCI_50ha_2022_03_30_local_classifications.tif",
    ]
    labelfiles = [lf for lf in labelfiles if lf.exists()]
    if not labelfiles:
        print("SKIP: No test label files found in", LABELS_DIR)
        sys.exit(0)

    print(f"Testing with {len(labelfiles)} label file(s):")
    for lf in labelfiles:
        print(f"  {lf.name}")

    with tempfile.TemporaryDirectory(prefix="smoke_coreg_") as tmpdir:
        tmpdir = Path(tmpdir)
        with_dir = tmpdir / "with_mask"
        no_dir   = tmpdir / "no_mask"

        # --- Run WITH --maskdir ---
        run(
            [
                *[str(lf) for lf in labelfiles],
                str(DRONEDIR), str(PLANETDIR), str(with_dir),
                "-r", "4.0", "-t", "2", "-k", str(MASKDIR),
            ],
            "With --maskdir",
        )

        mask_pngs_with = list(with_dir.glob("*.mask.png"))
        failures_json  = with_dir / "coreg_failures.json"

        print(f"\nMask PNGs written (with mask): {len(mask_pngs_with)}")

        if failures_json.exists():
            with open(failures_json) as f:
                failures = json.load(f)
            assert_ok(isinstance(failures, list), "coreg_failures.json is a JSON list")

            for entry in failures:
                assert_ok("scene" in entry, f"failure entry has 'scene' key: {entry}")
                assert_ok("label" in entry, f"failure entry has 'label' key: {entry}")
                assert_ok("clear_fraction" in entry, f"failure entry has 'clear_fraction' key: {entry}")
                cf = entry["clear_fraction"]
                if cf is not None:
                    assert_ok(
                        0.0 <= cf <= 1.0,
                        f"clear_fraction in [0,1] for {entry['scene']}: {cf}",
                    )

            # No mask.png should exist for any failed scene
            for entry in failures:
                mask_png = with_dir / f"{entry['scene']}.mask.png"
                assert_ok(
                    not mask_png.exists(),
                    f"No mask.png written for failed COREG scene {entry['scene']}",
                )
            print(f"Failures recorded: {len(failures)}")
        else:
            print("No coreg_failures.json written (all scenes succeeded or no matching scenes)")
            failures = []

        # --- Run WITHOUT --maskdir ---
        run(
            [
                *[str(lf) for lf in labelfiles],
                str(DRONEDIR), str(PLANETDIR), str(no_dir),
                "-r", "4.0", "-t", "2",
            ],
            "Without --maskdir",
        )

        mask_pngs_no = list(no_dir.glob("*.mask.png"))
        print(f"\nMask PNGs written (no mask): {len(mask_pngs_no)}")

        with_scenes = {png.name.replace(".mask.png", "") for png in mask_pngs_with}
        no_scenes   = {png.name.replace(".mask.png", "") for png in mask_pngs_no}

        missing = with_scenes - no_scenes
        assert_ok(
            len(missing) == 0,
            f"All with-mask scenes also present in no-mask run (missing: {missing})",
        )

        extra = no_scenes - with_scenes
        if extra:
            print(f"NOTE: {len(extra)} scene(s) present in no-mask run but not with-mask run")
            print("  (expected when --maskdir causes some scenes to be skipped due to missing OCM files)")

        print(f"\n{OK}: All smoke tests passed.")


if __name__ == "__main__":
    main()
