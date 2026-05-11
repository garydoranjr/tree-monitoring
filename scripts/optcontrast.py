"""
optcontrast.py

Mapping from matplotlib perceptual colormaps to their optimal single contrast color.

The optimal color is chosen via the geometric-mean-luminance strategy: given a
colormap whose colors span relative luminances [L_min, L_max], the contrast color
that maximises the *minimum* WCAG contrast ratio across all colormap values has
relative luminance:

    L_opt = sqrt((L_min + 0.05) * (L_max + 0.05)) - 0.05

A vivid hue is then chosen at that luminance (full HSV saturation, value adjusted
to hit L_opt). The max achievable minimum contrast ratio is also recorded for
reference.
"""

# Mapping: colormap name -> dict with contrast color info
#
# Keys:
#   'hex'        : CSS hex string for the contrast color
#   'rgb'        : (R, G, B) tuple, values in [0, 1]
#   'rgb255'     : (R, G, B) tuple, values in [0, 255]
#   'min_cr'     : maximum achievable minimum WCAG contrast ratio (float)
#   'L_opt'      : optimal relative luminance of the contrast color (float)

OPTIMAL_CONTRAST: dict[str, dict] = {
    "viridis": {
        "hex":    "#F20000",
        "rgb":    (0.951, 0.000, 0.000),
        "rgb255": (242,   0,     0),
        "min_cr": 3.47,
        "L_opt":  0.190,
    },
    "plasma": {
        "hex":    "#C65F00",
        "rgb":    (0.777, 0.373, 0.000),
        "rgb255": (198,   95,    0),
        "min_cr": 3.61,
        "L_opt":  0.202,
    },
    "inferno": {
        "hex":    "#7B7600",
        "rgb":    (0.484, 0.465, 0.000),
        "rgb255": (123,   119,   0),
        "min_cr": 4.46,
        "L_opt":  0.174,
    },
    "magma": {
        "hex":    "#0066FF",
        "rgb":    (0.000, 0.400, 1.000),
        "rgb255": (0,     102,   255),
        "min_cr": 4.46,
        "L_opt":  0.173,
    },
    "cividis": {
        "hex":    "#038C00",
        "rgb":    (0.011, 0.549, 0.000),
        "rgb255": (3,     140,   0),
        "min_cr": 3.54,
        "L_opt":  0.187,
    },
}


def get_contrast_color(cmap_name: str, fmt: str = "rgb") -> tuple | str:
    """
    Return the optimal contrast color for a given colormap.

    Parameters
    ----------
    cmap_name : str
        One of 'viridis', 'plasma', 'inferno', 'magma', 'cividis'.
    fmt : str
        Output format: 'rgb' (0-1 floats), 'rgb255' (0-255 ints), or 'hex'.

    Returns
    -------
    tuple or str
        The contrast color in the requested format.
    """
    entry = OPTIMAL_CONTRAST[cmap_name]
    if fmt not in ("rgb", "rgb255", "hex"):
        raise ValueError(f"Unknown format '{fmt}'. Choose 'rgb', 'rgb255', or 'hex'.")
    return entry[fmt]


if __name__ == "__main__":
    print(f"{'Colormap':<10}  {'Hex':>8}  {'RGB (0-1)':>28}  {'Min CR':>8}")
    print("-" * 62)
    for name, info in OPTIMAL_CONTRAST.items():
        r, g, b = info["rgb"]
        print(
            f"{name:<10}  {info['hex']:>8}  "
            f"({r:.3f}, {g:.3f}, {b:.3f})  "
            f"{info['min_cr']:>6.2f}:1"
        )
