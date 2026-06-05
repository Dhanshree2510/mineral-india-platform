# src/pipeline/features.py
# ─────────────────────────────────────────────────────────────────
# PURPOSE: Compute spectral indices from raw Sentinel-2 bands
#
# QGIS EQUIVALENT:
#   Raster → Raster Calculator → type formula → output raster
#   e.g. NDVI = ("B8@1" - "B4@1") / ("B8@1" + "B4@1")
#
# We do all of that in numpy — same math, no GUI needed.
# Output: a single multi-band GeoTIFF where each band = one index
# ─────────────────────────────────────────────────────────────────

import numpy as np
import rasterio
import os
import yaml
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def load_sentinel2(filepath: str) -> tuple:
    """
    Loads Sentinel-2 GeoTIFF and returns bands as float32 arrays.

    WHY float32?
    Raw values are uint16 (0–10000).
    Division in index formulas needs float — integer division
    truncates decimals and gives wrong results.
    e.g. (2209 - 794) / (2209 + 794) as int = 1415/3003 = 0 ← WRONG
                                              as float = 0.471 ← RIGHT

    Returns:
        bands: dict of {band_name: 2D numpy array}
        profile: rasterio profile (used to write output GeoTIFF)
    """
    log.info(f"Loading: {filepath}")
    with rasterio.open(filepath) as src:
        profile = src.profile.copy()
        raw     = src.read().astype(np.float32)

    # Band order matches what we downloaded:
    # Index 0=B2, 1=B3, 2=B4, 3=B8, 4=B11, 5=B12
    bands = {
        "B2_blue":  raw[0],
        "B3_green": raw[1],
        "B4_red":   raw[2],
        "B8_nir":   raw[3],
        "B11_swir1": raw[4],
        "B12_swir2": raw[5],
    }

    # Replace zeros with NaN — zero = masked/cloud pixel
    # NaN propagates through math correctly (result = NaN not 0)
    for name in bands:
        bands[name][bands[name] == 0] = np.nan

    log.info(f"Loaded {len(bands)} bands | shape: {raw[0].shape}")
    return bands, profile


def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Divides two arrays, returning NaN where denominator = 0.

    Standard division crashes with ZeroDivisionError if any
    pixel has b=0. np.where handles this safely.

    QGIS Raster Calculator does this automatically.
    In numpy we must do it manually.
    """
    return np.where(
        np.abs(b) > 1e-10,  # denominator is non-zero
        a / b,               # normal division
        np.nan               # NaN where b is ~zero
    )


def compute_indices(bands: dict) -> dict:
    """
    Computes all spectral indices from raw bands.

    Each index is a single 2D array (same size as input bands).
    Values typically range from -1 to +1 for ratio indices,
    or 0 to ~5 for simple ratios.

    WHY THESE INDICES?
    These are not arbitrary — each one was designed by geologists
    to highlight a specific mineral or surface property.
    """

    B2  = bands["B2_blue"]
    B3  = bands["B3_green"]
    B4  = bands["B4_red"]
    B8  = bands["B8_nir"]
    B11 = bands["B11_swir1"]
    B12 = bands["B12_swir2"]

    indices = {}

    # ── 1. NDVI — Normalized Difference Vegetation Index ─────────
    # Range: -1 to +1
    # High NDVI (>0.4)  = dense vegetation (forest)
    # Low NDVI  (<0.1)  = bare rock, mining pits, dry soil
    #
    # WHY USEFUL FOR MINERALS:
    # Iron ore outcrops = very low NDVI (bare rock, no plants)
    # Mining pits       = negative NDVI (no vegetation at all)
    # Acts as a MASK — where NDVI is low, rock is exposed
    #
    # Formula: (NIR - Red) / (NIR + Red)
    indices["NDVI"] = safe_divide(B8 - B4, B8 + B4)

    # ── 2. Iron Oxide Ratio ───────────────────────────────────────
    # Range: typically 0.5 to 3.0
    # HIGH value = iron oxide minerals (hematite, goethite, limonite)
    # These are the surface expression of iron ore deposits
    #
    # WHY IT WORKS:
    # Iron oxides strongly ABSORB blue light (they look red/brown)
    # and REFLECT red light. So Red/Blue is HIGH for iron oxides.
    #
    # In Keonjhar: look for pixels where this ratio > 2.0
    # Those are your gossan / iron ore outcrop zones.
    #
    # Formula: Red / Blue
    indices["iron_oxide"] = safe_divide(B4, B2)

    # ── 3. Ferrous Iron Index ─────────────────────────────────────
    # Range: typically 0.3 to 2.5
    # HIGH = ferrous iron minerals (magnetite, olivine, pyroxene)
    # LOW  = non-ferrous rocks (granite, limestone)
    #
    # Magnetite = primary iron ore mineral → HIGH ferrous index
    # Surrounding granite/gneiss → LOW ferrous index
    #
    # Formula: SWIR1 / NIR
    indices["ferrous_iron"] = safe_divide(B11, B8)

    # ── 4. Clay Ratio ─────────────────────────────────────────────
    # Range: typically 0.5 to 2.0
    # HIGH = clay minerals (kaolinite, illite, smectite)
    # Clay alteration = hydrothermal activity = near ore deposits
    #
    # WHY: Hot fluids from magma alter surrounding rock to clay.
    # Finding clay = finding where hydrothermal fluids flowed.
    # Copper, gold, and some iron deposits have clay halos.
    #
    # Formula: SWIR1 / SWIR2
    indices["clay_ratio"] = safe_divide(B11, B12)

    # ── 5. Laterite Index ─────────────────────────────────────────
    # Range: 0 to 1
    # HIGH = laterite surfaces (tropical weathered rock)
    # Laterite = source rock for bauxite (aluminium ore)
    # Also found over weathered iron ore in Odisha
    #
    # Formula: Red / NIR (inverse of NDVI essentially)
    indices["laterite"] = safe_divide(B4, B8)

    # ── 6. NDWI — Normalized Difference Water Index ───────────────
    # Range: -1 to +1
    # HIGH (>0)  = water bodies (rivers, ponds, mining pits)
    # LOW  (<0)  = land surface
    #
    # WHY USEFUL:
    # Open pit mines often fill with water (bright NDWI)
    # Drainage patterns around deposits
    # Also helps mask out water before training
    #
    # Formula: (Green - NIR) / (Green + NIR)
    indices["NDWI"] = safe_divide(B3 - B8, B3 + B8)

    # ── 7. Bare Soil Index (BSI) ──────────────────────────────────
    # Range: -1 to +1
    # HIGH = bare soil / rock (no vegetation, no water)
    # Bare rock = exposed geology = where minerals are accessible
    #
    # Formula: ((SWIR1 + Red) - (NIR + Blue)) /
    #           ((SWIR1 + Red) + (NIR + Blue))
    numerator   = (B11 + B4) - (B8 + B2)
    denominator = (B11 + B4) + (B8 + B2)
    indices["BSI"] = safe_divide(numerator, denominator)

    # ── 8. SWIR Ratio (mineralogy discriminator) ──────────────────
    # Range: typically 0.5 to 3.0
    # Different minerals have different SWIR1/SWIR2 values:
    #   Kaolinite (clay)   → ratio ~1.4
    #   Alunite (sulfate)  → ratio ~0.9
    #   Calcite (carbonate)→ ratio ~1.2
    #   Iron ore (hematite)→ ratio ~1.6
    # This is ASTER bands 5/8 equivalent in Sentinel-2
    #
    # Formula: SWIR1 / SWIR2 (same as clay but interpreted differently)
    # We keep this separate from clay_ratio because when combined
    # with other indices, the pattern helps discriminate more minerals
    indices["swir_ratio"] = safe_divide(B11, B12)

    log.info(f"Computed {len(indices)} spectral indices:")
    for name, arr in indices.items():
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            log.info(f"  {name:<20} min={valid.min():6.3f}  "
                     f"max={valid.max():6.3f}  mean={valid.mean():6.3f}")

    return indices


def save_indices(indices: dict,
                 reference_profile: dict,
                 output_path: str) -> None:
    """
    Saves all computed indices as a multi-band GeoTIFF.

    One band per index, in the order of the dict.
    The output file can be opened directly in QGIS.

    Band order in output:
      1: NDVI
      2: iron_oxide
      3: ferrous_iron
      4: clay_ratio
      5: laterite
      6: NDWI
      7: BSI
      8: swir_ratio

    QGIS: Add Raster Layer → open this file → Style each band
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    index_names = list(indices.keys())
    arrays = np.stack(
        [indices[name] for name in index_names], axis=0
    )  # shape: (n_indices, H, W)

    profile = reference_profile.copy()
    profile.update(
        count=len(indices),
        dtype=np.float32,     # indices are float, not uint16
        compress="lzw",
        nodata=np.nan
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arrays.astype(np.float32))
        # Write band descriptions so QGIS shows names
        for i, name in enumerate(index_names, start=1):
            dst.update_tags(i, name=name)

    log.info(f"✅ Saved {len(indices)} indices → {output_path}")
    log.info(f"   Band order: {index_names}")


def visualize_indices(indices: dict, output_dir: str) -> None:
    """
    Creates visualization plots for all indices.

    Uses diverging colormaps where appropriate:
    - RdYlGn for NDVI (red=bare, green=vegetation)
    - hot    for iron_oxide (dark=low, bright=high iron)
    - RdBu   for NDWI (blue=water, red=dry land)

    These match standard remote sensing conventions.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Colormap per index — geologically meaningful
    cmaps = {
        "NDVI":         ("RdYlGn", -0.3, 0.8),
        "iron_oxide":   ("hot",     0.5, 3.0),
        "ferrous_iron": ("YlOrRd",  0.3, 2.0),
        "clay_ratio":   ("PuOr",    0.5, 2.0),
        "laterite":     ("OrRd",    0.0, 1.5),
        "NDWI":         ("RdBu",   -0.5, 0.5),
        "BSI":          ("copper", -0.3, 0.5),
        "swir_ratio":   ("plasma",  0.5, 2.5),
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        "Spectral Indices — Keonjhar District, Odisha\n"
        "Each index highlights different mineral/surface properties",
        fontsize=14
    )

    for ax, (name, arr) in zip(axes.flat, indices.items()):
        cmap, vmin, vmax = cmaps.get(name, ("viridis", None, None))
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = os.path.join(output_dir, "04_spectral_indices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info(f"Saved: {out}")
    plt.show()

    # ── Iron oxide zoomed in — most important for iron ore ────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(
        indices["iron_oxide"],
        cmap="hot", vmin=0.5, vmax=3.0
    )
    ax.set_title(
        "Iron Oxide Ratio — Keonjhar\n"
        "Bright zones = high iron oxide = potential iron ore",
        fontsize=13
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Iron Oxide Ratio (B4/B2)")
    out = os.path.join(output_dir, "05_iron_oxide_zoomed.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    log.info(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    config = yaml.safe_load(open("configs/config.yaml"))

    input_path  = os.path.join(config["paths"]["raw"],
                               "sentinel2_keonjhar.tif")
    output_path = os.path.join(config["paths"]["processed"],
                               "indices_keonjhar.tif")
    viz_dir     = "outputs/verification"

    # 1. Load bands
    bands, profile = load_sentinel2(input_path)

    # 2. Compute indices
    indices = compute_indices(bands)

    # 3. Save as GeoTIFF
    save_indices(indices, profile, output_path)

    # 4. Visualize
    visualize_indices(indices, viz_dir)

    print(f"\n✅ Feature engineering complete.")
    print(f"   Indices GeoTIFF → {output_path}")
    print(f"   Plots          → {viz_dir}/")
    print(f"\nNext: open {output_path} in QGIS")
    print(f"  → Add Raster Layer → Style each band with its colormap")
    print(f"  → Compare iron_oxide band with known mine locations")