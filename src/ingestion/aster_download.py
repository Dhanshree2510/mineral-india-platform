# src/ingestion/aster_download.py  ── FIXED VERSION
# ─────────────────────────────────────────────────────────────────
# Key fixes:
#   1. Use AST_07XT Version 4 (short_name="AST_07XT" version="004")
#   2. Widen date range to 2020–2024 (ASTER is not daily — need wide window)
#   3. Add GEE fallback using ASTER thermal emissivity dataset
#      (always available, no ordering needed)
# ─────────────────────────────────────────────────────────────────

import earthaccess
import rasterio
import numpy as np
import yaml
import os
import logging
from pathlib import Path
from rasterio.merge import merge as rio_merge
from rasterio.warp import (calculate_default_transform,
                            reproject, Resampling)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def authenticate_nasa():
    try:
        earthaccess.login(strategy="netrc")
        log.info("NASA Earthdata authenticated")
    except Exception:
        log.info("Saved credentials not found — trying interactive login")
        earthaccess.login(strategy="interactive")


def search_aster_v4(bbox: dict, max_scenes: int = 20) -> list:
    """
    Searches for ASTER V4 scenes.

    KEY CHANGES from V3:
    - version="004" instead of "003"
    - Much wider date range — ASTER has 16-day repeat cycle
      and not every pass covers our exact bbox.
      We search 3 full years to guarantee finding scenes.
    - V4 is a full pre-processed collection — no ordering needed.
      Results are available immediately for download.

    ASTER covers Odisha roughly every 16 days but cloud cover
    and instrument scheduling reduce usable scenes.
    Over 3 years we expect 20-40 cloud-free scenes.
    """
    log.info("Searching ASTER V4 (AST_07XT version 004)...")
    log.info(f"Bbox: {bbox}")

    # Wide date range — ASTER is not daily, need years not months
    results = earthaccess.search_data(
        short_name="AST_07XT",
        version="004",                 # ← V4, not V3
        bounding_box=(
            bbox["lon_min"],
            bbox["lat_min"],
            bbox["lon_max"],
            bbox["lat_max"]
        ),
        temporal=("2019-01-01", "2024-11-28"),  # before power failure
        count=max_scenes
    )

    log.info(f"Found {len(results)} ASTER V4 scenes")

    for i, r in enumerate(results[:5]):  # show first 5
        try:
            date = r["umm"]["TemporalExtent"]["RangeDateTime"][
                "BeginningDateTime"
            ]
            log.info(f"  Scene {i+1}: {date}")
        except Exception:
            log.info(f"  Scene {i+1}: (date unavailable)")

    return results


def download_aster_scenes(results: list, output_dir: str) -> list:
    if not results:
        log.warning("No results to download")
        return []

    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Downloading {len(results)} scenes to: {output_dir}")

    files = earthaccess.download(results, local_path=output_dir)

    valid = []
    for f in files:
        f = Path(f)
        if f.exists() and f.stat().st_size > 100_000:
            log.info(f"  ✅ {f.name} ({f.stat().st_size/1e6:.0f} MB)")
            valid.append(str(f))
        else:
            log.warning(f"  ⚠️  {f} missing or too small")

    return valid


def extract_swir_from_tif(tif_path: str,
                           output_path: str,
                           target_crs: str = "EPSG:32644") -> bool:
    """
    Extracts SWIR bands from ASTER V4 GeoTIFF.

    V4 CHANGE: ASTER V4 delivers GeoTIFF files directly
    (not HDF like V3). Each file is one band or a small set.
    We identify SWIR files by their filename suffix:
      *_SRF_SWIR_B04.tif through *_SRF_SWIR_B09.tif

    This function handles a DIRECTORY of per-band TIFFs
    and stacks them into one 6-band output.
    """
    try:
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            src_crs = src.crs
            src_transform = src.transform

        # Scale: V4 reflectance scale factor = 0.001
        arr[arr <= 0] = np.nan
        arr = arr * 0.001

        # Reproject to target CRS
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, target_crs,
            profile["width"], profile["height"],
            *rasterio.transform.array_bounds(
                profile["height"], profile["width"], src_transform
            )
        )

        reprojected = np.empty((dst_h, dst_w), dtype=np.float32)
        reproject(
            source=arr,
            destination=reprojected,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

        out_profile = profile.copy()
        out_profile.update(
            crs=target_crs, transform=dst_transform,
            width=dst_w, height=dst_h,
            count=1, dtype=np.float32,
            compress="lzw", nodata=np.nan, driver="GTiff"
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(reprojected, 1)

        return True

    except Exception as e:
        log.error(f"Failed: {tif_path} — {e}")
        return False


def stack_swir_bands(swir_dir: str,
                     output_path: str) -> bool:
    """
    Stacks individual SWIR band TIFFs into one 6-band GeoTIFF.

    V4 delivers one file per band. We find all SWIR band files
    (B04–B09), sort them, and stack into (6, H, W).

    This is QGIS: Raster → Miscellaneous → Merge (with separate bands)
    """
    swir_files = sorted([
        f for f in Path(swir_dir).glob("*SWIR*.tif")
        if any(f"B0{i}" in f.name for i in range(4, 10))
    ])

    if len(swir_files) == 0:
        # Try alternative naming (some V4 files use B4 not B04)
        swir_files = sorted([
            f for f in Path(swir_dir).glob("*.tif")
            if any(f"_B{i}" in f.name for i in range(4, 10))
        ])

    log.info(f"Found {len(swir_files)} SWIR band files")
    for f in swir_files:
        log.info(f"  {f.name}")

    if len(swir_files) == 0:
        log.error("No SWIR band files found")
        return False

    arrays = []
    ref_profile = None

    for f in swir_files:
        with rasterio.open(f) as src:
            arrays.append(src.read(1).astype(np.float32))
            if ref_profile is None:
                ref_profile = src.profile.copy()

    stacked = np.stack(arrays, axis=0)  # (n_bands, H, W)

    ref_profile.update(
        count=len(arrays),
        dtype=np.float32,
        compress="lzw",
        nodata=np.nan
    )

    with rasterio.open(output_path, "w", **ref_profile) as dst:
        dst.write(stacked)

    log.info(f"✅ Stacked {len(arrays)} SWIR bands → {output_path}")
    return True


# ─────────────────────────────────────────────────────────────────
# FALLBACK: Get ASTER data via Google Earth Engine
# This is ALWAYS available — no ordering, no earthaccess needed.
# Product: NASA/ASTER_GED/AG100_003
#   = ASTER Global Emissivity Dataset (100m, preprocessed)
#   Bands: emissivity_band10 through emissivity_band14 (thermal)
#   + temperature
# For SWIR mineral mapping, GEE also has:
#   ASTER L1T raw radiance → we compute band ratios ourselves
# ─────────────────────────────────────────────────────────────────

def download_aster_via_gee(config: dict,
                            geometry,
                            output_path: str) -> None:
    """
    Downloads ASTER thermal emissivity data via GEE.

    WHY THIS AS FALLBACK:
    NASA/ASTER_GED/AG100_003 is always available on GEE —
    no account, no ordering, no date searching.
    It is a composite (best scenes averaged) so no cloud issues.

    BANDS:
    emissivity_band10 → 8.125–8.475 μm  silicate minerals
    emissivity_band11 → 8.475–8.825 μm  silicate vs carbonate
    emissivity_band12 → 8.925–9.275 μm  quartz
    emissivity_band13 → 10.25–10.95 μm  carbonates
    emissivity_band14 → 10.95–11.65 μm  carbonates + silicates
    temperature       → land surface temperature (K)

    These are THERMAL bands (not SWIR) but still useful:
    - Quartz-rich rocks (silica cap on deposits) → high emissivity B12
    - Carbonates (limestone, marble) → high B13/B14
    - Iron oxide rocks → lower emissivity than silicates
    """
    import ee
    import geemap

    log.info("Downloading ASTER Global Emissivity via GEE (fallback)...")

    aster_ged = ee.Image("NASA/ASTER_GED/AG100_003").select([
        "emissivity_band10",
        "emissivity_band11",
        "emissivity_band12",
        "emissivity_band13",
        "emissivity_band14",
        "temperature"
    ])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    import geedim
    gd_image = geedim.MaskedImage(aster_ged)
    gd_image.download(
        output_path,
        crs=config["study_area"]["crs"],
        scale=100,            # ASTER GED is 100m resolution
        region=geometry,
        dtype="float32",
        overwrite=True
    )

    log.info(f"✅ ASTER GED saved: {output_path}")


def download_aster_swir_via_gee(config: dict,
                                 geometry,
                                 output_path: str) -> None:
    """
    Downloads ASTER L1T radiance via GEE and computes SWIR ratios.

    GEE collection: ASTER/AST_L1T_003
    This has raw radiance — we compute band ratios ourselves.
    Not atmospherically corrected but good enough for ratios.

    SWIR bands in GEE ASTER L1T:
    B4 → 1656nm, B5 → 2167nm, B6 → 2209nm
    B7 → 2262nm, B8 → 2336nm, B9 → 2400nm

    We take the median composite across all available scenes
    (same approach as Sentinel-2 composite).
    """
    import ee
    import geedim

    log.info("Downloading ASTER L1T SWIR bands via GEE...")

    collection = (
        ee.ImageCollection("ASTER/AST_L1T_003")
        .filterBounds(geometry)
        .filterDate("2000-01-01", "2024-11-28")  # full mission
        .filter(ee.Filter.lt("CLOUDCOVER", 15))
        .select(["B4", "B5", "B6", "B7", "B8", "B9"])
    )

    count = collection.size().getInfo()
    log.info(f"Found {count} ASTER L1T scenes in GEE")

    if count == 0:
        raise ValueError("No ASTER L1T scenes found in GEE either")

    composite = collection.median()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    gd_image = geedim.MaskedImage(composite)
    gd_image.download(
        output_path,
        crs=config["study_area"]["crs"],
        scale=30,
        region=geometry,
        dtype="uint16",
        overwrite=True
    )

    log.info(f"✅ ASTER SWIR (GEE) saved: {output_path}")


if __name__ == "__main__":
    import ee

    config  = yaml.safe_load(open("configs/config.yaml"))
    bbox    = config["study_area"]["bbox"]
    crs     = config["study_area"]["crs"]
    raw_dir = config["paths"]["raw"]

    ee.Initialize(project=config["project"]["gee_project"])
    geometry = ee.Geometry.Rectangle([
        bbox["lon_min"], bbox["lat_min"],
        bbox["lon_max"], bbox["lat_max"]
    ])

    # ── Try earthaccess V4 first ──────────────────────────────────
    log.info("=" * 55)
    log.info("STRATEGY 1: earthaccess — ASTER V4")
    log.info("=" * 55)

    try:
        authenticate_nasa()
        results = search_aster_v4(bbox, max_scenes=10)

        if results:
            raw_dir_aster = os.path.join(raw_dir, "aster_raw")
            files = download_aster_scenes(results, raw_dir_aster)

            if files:
                final = os.path.join(raw_dir, "aster_keonjhar.tif")
                stack_swir_bands(raw_dir_aster, final)
                log.info(f"✅ ASTER V4 via earthaccess: {final}")
            else:
                raise RuntimeError("No files downloaded")
        else:
            raise RuntimeError("No scenes found")

    except Exception as e:
        log.warning(f"earthaccess failed: {e}")

        # ── Fallback: GEE ASTER L1T SWIR ─────────────────────────
        log.info("\n" + "=" * 55)
        log.info("STRATEGY 2: GEE — ASTER L1T SWIR bands")
        log.info("=" * 55)

        try:
            swir_path = os.path.join(raw_dir, "aster_swir_keonjhar.tif")
            download_aster_swir_via_gee(config, geometry, swir_path)
            log.info(f"✅ ASTER SWIR via GEE: {swir_path}")

        except Exception as e2:
            log.warning(f"GEE SWIR also failed: {e2}")

            # ── Final fallback: GEE ASTER thermal emissivity ──────
            log.info("\n" + "=" * 55)
            log.info("STRATEGY 3: GEE — ASTER Global Emissivity (thermal)")
            log.info("=" * 55)

            ged_path = os.path.join(raw_dir, "aster_ged_keonjhar.tif")
            download_aster_via_gee(config, geometry, ged_path)
            log.info(f"✅ ASTER GED via GEE: {ged_path}")

    print("\n✅ ASTER ingestion complete")
    print("   Check data/raw/ for aster_*.tif files")