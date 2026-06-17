# scripts/aster_via_gee.py
# ─────────────────────────────────────────────────────────────────
# Downloads ASTER SWIR bands directly from Google Earth Engine.
# GEE has the full ASTER mission archive (2000–2024).
# No earthaccess, no ordering, no file format issues.
#
# GEE collection: ASTER/AST_L1T_003
# Contains raw radiance for all VNIR + SWIR bands.
# We compute a median composite across all valid scenes.
#
# SWIR bands in GEE naming:
#   B4 → 1656nm  B5 → 2167nm  B6 → 2209nm
#   B7 → 2262nm  B8 → 2336nm  B9 → 2400nm
# ─────────────────────────────────────────────────────────────────

import ee
import geedim
import rasterio
import numpy as np
import yaml
import os
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def build_aster_composite(geometry: ee.Geometry) -> ee.Image:
    log.info("Searching ASTER L1T collection on GEE...")

    collection = (
        ee.ImageCollection("ASTER/AST_L1T_003")
        .filterBounds(geometry)
        .filterDate("2000-03-01", "2024-11-28")
        .filter(ee.Filter.lt("CLOUDCOVER", 20))
    )

    count = collection.size().getInfo()
    log.info(f"ASTER scenes found: {count}")

    if count == 0:
        raise ValueError("No ASTER scenes found. Try increasing CLOUDCOVER to 30.")

    # ── FIXED: zero-padded band names ────────────────────────────
    swir_bands = ["B04", "B05", "B06", "B07", "B08", "B09"]

    composite = (
        collection
        .select(swir_bands)
        .median()
        .toUint16()
    )

    log.info(f"Composite built with SWIR bands: {swir_bands}")
    return composite


def download_aster_gee(composite: ee.Image,
                        geometry: ee.Geometry,
                        config: dict,
                        output_path: str,study_area) -> None:
    log.info(f"Downloading ASTER SWIR composite → {output_path}")
    log.info("geedim will auto-tile to stay within GEE limits...")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ── FIXED: use new geedim accessor instead of MaskedImage ────
    gd_image = geedim.MaskedImage(composite)

    gd_image.download(
        output_path,
        crs=config[study_area]["crs"],
        scale=30,
        region=geometry,
        dtype="uint16",
        overwrite=True
    )

    size_mb = os.path.getsize(output_path) / 1e6
    log.info(f"✅ Downloaded: {output_path} ({size_mb:.1f} MB)")


def verify_and_plot(filepath: str, output_dir: str,study_area:str) -> None:
    """
    Verifies the downloaded file and creates mineral ratio maps.
    """
    config = yaml.safe_load(open("config.yaml"))
    os.makedirs(output_dir, exist_ok=True)

    BAND_LABELS = [
        "B4 — 1656nm\nBaseline SWIR",
        "B5 — 2167nm\nKaolinite absorption",
        "B6 — 2209nm\nHematite absorption",
        "B7 — 2262nm\nAlunite absorption",
        "B8 — 2336nm\nCalcite absorption",
        "B9 — 2400nm\nSerpentine"
    ]

    with rasterio.open(filepath) as src:
        data = src.read().astype(np.float32)
        log.info(f"\n── ASTER SWIR FILE SUMMARY ───────────────")
        log.info(f"Bands  : {src.count}")
        log.info(f"Size   : {src.width} × {src.height} px")
        log.info(f"CRS    : {src.crs}")
        log.info(f"Scale  : 30m")

    # Band stats
    log.info(f"\n── BAND STATISTICS ───────────────────────")
    for i, label in enumerate(BAND_LABELS):
        band  = data[i]
        valid = band[band > 0]
        if len(valid):
            log.info(f"  {label.split(chr(10))[0]:<20} "
                     f"mean={valid.mean():.0f}  "
                     f"max={valid.max():.0f}")

    # ── Plot all 6 SWIR bands ─────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"ASTER SWIR Composite — {config[study_area]['name']} (via GEE)\n"
        "Median of all cloud-free scenes 2000–2024",
        fontsize=14
    )
    cmaps = ["gray", "YlOrRd", "hot", "PuOr", "Blues", "Greens"]

    for i, (ax, label, cmap) in enumerate(
        zip(axes.flat, BAND_LABELS, cmaps)
    ):
        band  = data[i]
        valid = band[band > 0]
        if not len(valid):
            ax.text(0.5, 0.5, "No data", ha="center")
            continue
        vmin = np.percentile(valid, 2)
        vmax = np.percentile(valid, 98)
        im   = ax.imshow(band, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    out = os.path.join(output_dir, f"06_aster_swir_{config[study_area]['mineral']}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info(f"Saved: {out}")
    plt.show()

    # ── Hematite ratio: B6/B5 ────────────────────────────────────
    # > 1.0 = hematite dominant = iron ore signature
    # < 1.0 = kaolinite dominant = clay alteration
    b5 = data[1].copy()
    b6 = data[2].copy()
    b5[b5 == 0] = np.nan
    hematite = np.where(b5 > 10, b6 / b5, np.nan)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(hematite, cmap="RdBu_r", vmin=0.85, vmax=1.15)
    ax.set_title(
        "ASTER Hematite Ratio (B6 / B5)\n"
        "RED  > 1.0 = hematite dominant = iron ore\n"
        "BLUE < 1.0 = kaolinite dominant = clay/alteration",
        fontsize=12
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="B6 / B5")
    out = os.path.join(output_dir, f"07_hematite_ratio_{config[study_area]['mineral']}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    log.info(f"Saved: {out}")
    plt.show()

    # ── Clay alteration: B5/B4 ────────────────────────────────────
    # High = strong kaolinite absorption = hydrothermal alteration
    b4 = data[0].copy()
    b4[b4 == 0] = np.nan
    clay = np.where(b4 > 10, b5 / b4, np.nan)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(clay, cmap="YlOrBr", vmin=0.8, vmax=1.3)
    ax.set_title(
        "ASTER Clay Alteration (B5 / B4)\n"
        "DARK = strong kaolinite absorption = hydrothermal alteration zones",
        fontsize=12
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="B5 / B4")
    out = os.path.join(output_dir, f"08_clay_alteration_{config[study_area]['mineral']}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    log.info(f"Saved: {out}")
    plt.show()

    log.info("\n✅ ASTER verification complete")


def aster_via_gee_download(study_area):
    config = yaml.safe_load(open("config.yaml"))

    ee.Initialize(project=config["project"]["gee_project"])

    bbox = config[study_area]["bbox"]
    geometry = ee.Geometry.Rectangle([
        bbox["lon_min"], bbox["lat_min"],
        bbox["lon_max"], bbox["lat_max"]
    ])  
    area_name = config[study_area]['name']
    mineral = config[study_area]['mineral']


    output_path = os.path.join(
        config["paths"]["raw"], f"aster_swir_{mineral}_{area_name}.tif"
    )

    # Build composite
    composite = build_aster_composite(geometry)

    # Download
    download_aster_gee(composite, geometry, config, output_path,study_area)

    # Verify + plot
    verify_and_plot(output_path, "outputs/verification",study_area)

    print(f"\n✅ ASTER SWIR ready: {output_path}")
    