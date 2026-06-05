# src/ingestion/gee_sentinel2.py  ── FINAL VERSION using geedim
# ─────────────────────────────────────────────────────────────────
# Why geedim?
#   geemap.ee_export_image uses getDownloadURL which has a hard
#   32MB + 10000px limit on COMPUTED images (composites).
#   geedim splits the image into small tiles internally,
#   downloads them in parallel, and merges automatically.
#   No Google Drive needed. No manual tiling.
# ─────────────────────────────────────────────────────────────────

import ee
import geedim
import geedim.mask
import yaml
import os
import logging
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def initialize_gee(project_id):
    """
    Initializes GEE. geedim uses the same ee credentials
    so no extra authentication needed.
    """
    ee.Initialize(project=project_id)
    log.info(f"GEE Initialized with project: {project_id}")


def build_collection(config, geometry):
    """
    Builds a cloud-free Sentinel-2 median composite.
    Returns a plain ee.Image — geedim wraps it later.
    """
    s2 = config["sentinel2"]
    bands = [b for b in s2["bands"] if b != "SCL"]

    def mask_clouds(img):
        scl  = img.select("SCL")
        mask = (scl.neq(3).And(scl.neq(8))
                          .And(scl.neq(9))
                          .And(scl.neq(10)))
        return img.updateMask(mask)

    collection = (
        ee.ImageCollection(s2["collection"])
        .filterBounds(geometry)
        .filterDate(s2["date_start"], s2["date_end"])
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",
                             s2["cloud_threshold"]))
    )

    count = collection.size().getInfo()
    log.info(f"Scenes found after filtering: {count}")

    if count == 0:
        raise ValueError(
            "No scenes found.\n"
            "Try: increase cloud_threshold to 20 in config.yaml"
        )

    composite = (
        collection
        .map(mask_clouds)
        .select(bands)
        .median()
        # Cast to uint16 — Sentinel-2 reflectance values are
        # 0–10000 (integer). Storing as float32 triples the size.
        .toUint16()
    )

    log.info(f"Composite built | bands: {bands}")
    return composite


def download_with_geedim(image: ee.Image,
                          geometry: ee.Geometry,
                          config: dict,
                          output_path: str) -> None:
    """
    Downloads a GEE image using geedim's automatic tiling.

    HOW GEEDIM WORKS:
    1. It wraps your ee.Image in a geedim.MaskedImage
    2. Internally splits the region into tiles that fit
       within GEE's per-request limit
    3. Downloads all tiles in parallel threads
    4. Merges them into one GeoTIFF automatically

    No Drive. No manual tiling. No size errors.

    Args:
        image:       the ee.Image composite to download
        geometry:    ee.Geometry defining the region
        config:      our config dict
        output_path: where to save the final GeoTIFF
    """
    s2  = config["sentinel2"]
    crs = config["study_area"]["crs"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    log.info(f"Wrapping image with geedim...")

    # geedim.MaskedImage wraps any ee.Image for download
    # mask_shadows=False because we already masked in the composite
    gd_image = geedim.MaskedImage(image)

    log.info(f"Starting download → {output_path}")
    log.info("geedim will auto-tile and show a progress bar...")

    # .download() handles everything:
    #   crs       = target coordinate system
    #   scale     = pixel size in metres
    #   region    = geographic extent
    #   dtype     = output data type (uint16 saves space vs float32)
    #   overwrite = replace existing file if present
    gd_image.download(
        output_path,
        crs=crs,
        scale=s2["scale"],
        region=geometry,
        dtype="uint16",
        overwrite=True
    )

    log.info(f"Download complete: {output_path}")


def verify_download(filepath: str) -> bool:
    """
    Verifies the downloaded GeoTIFF is valid and has real data.
    Equivalent to loading in QGIS and checking the attribute table.
    """
    if not os.path.exists(filepath):
        log.error(f"File missing: {filepath}")
        return False

    size_mb = os.path.getsize(filepath) / 1e6
    if size_mb < 0.1:
        log.error(f"File too small ({size_mb:.2f} MB) — likely empty")
        return False

    try:
        with rasterio.open(filepath) as src:
            import numpy as np
            b1 = src.read(1)
            log.info(f"✅ Verified: {filepath}")
            log.info(f"   Size on disk : {size_mb:.1f} MB")
            log.info(f"   Bands        : {src.count}")
            log.info(f"   Pixels       : {src.width} × {src.height}")
            log.info(f"   CRS          : {src.crs}")
            log.info(f"   Bounds       : {src.bounds}")
            log.info(f"   Value range  : {b1.min()} – {b1.max()}")
            return int(b1.max()) > 0
    except Exception as e:
        log.error(f"File corrupted: {e}")
        return False


if __name__ == "__main__":
    # ── Load config ───────────────────────────────────────────────
    config = load_config()

    # ── Connect to GEE ────────────────────────────────────────────
    initialize_gee(config["project"]["gee_project"])

    # ── Define study area ─────────────────────────────────────────
    bbox = config["study_area"]["bbox"]
    geometry = ee.Geometry.Rectangle([
        bbox["lon_min"], bbox["lat_min"],
        bbox["lon_max"], bbox["lat_max"]
    ])
    log.info(f"Study area: {bbox}")

    # ── Build composite ───────────────────────────────────────────
    composite = build_collection(config, geometry)

    # ── Download ──────────────────────────────────────────────────
    output = os.path.join(
        config["paths"]["raw"], "sentinel2_keonjhar.tif"
    )

    download_with_geedim(composite, geometry, config, output)

    # ── Verify ────────────────────────────────────────────────────
    if verify_download(output):
        print(f"\n✅ Sentinel-2 data ready at: {output}")
    else:
        print("\n❌ Verify failed — check logs above")
        print("   Fallback: try exporting to Google Drive instead")
        print("   Run: python src/ingestion/export_to_drive.py")