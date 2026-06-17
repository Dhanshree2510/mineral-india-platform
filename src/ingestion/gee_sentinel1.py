# src/ingestion/gee_sentinel1.py  ── FIXED
# src/ingestion/gee_sentinel1.py  ── FIXED
import ee
import geedim
import yaml
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def load_config(path="config.yaml"):   # ← Fix 3
    with open(path) as f:
        return yaml.safe_load(f)


def initialize_gee(project_id):
    ee.Initialize(project=project_id)
    log.info(f"GEE Initialized for Sentinel-1: {project_id}")


def build_sentinel1_composite(geometry, config):
    """
    Builds a Sentinel-1 SAR median composite.
    VV = surface roughness / rock texture
    VH = volume scattering (vegetation)
    VV/VH ratio = geology discriminator
    """
    s2_config = config["sentinel2"]

    log.info("Filtering Sentinel-1 GRD Collection...")
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(s2_config["date_start"], s2_config["date_end"])
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains(
            "transmitterReceiverPolarisation", "VV"
        ))
        .filter(ee.Filter.listContains(
            "transmitterReceiverPolarisation", "VH"
        ))
        .select(["VV", "VH"])
    )

    count = collection.size().getInfo()
    log.info(f"Found {count} Sentinel-1 SAR scenes")

    if count == 0:
        raise ValueError(
            "No Sentinel-1 scenes found. "
            "Check bbox or widen date range."
        )

    composite = collection.median()

    vv    = composite.select("VV")
    vh    = composite.select("VH")
    ratio = vv.subtract(vh).rename("VV_VH_ratio")

    return composite.addBands(ratio)


def download_s1_gee(composite, geometry, config, output_path,study_area):
    """
    Downloads SAR composite at native 10m using geedim.
    """
    crs = config[study_area]["crs"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    log.info(f"Downloading Sentinel-1 → {output_path}")

    # ── Fix 1: use .gd accessor ───────────────────────────────────
    gd_image = geedim.MaskedImage(composite)

    gd_image.download(
        output_path,
        crs=crs,
        scale=10,
        region=geometry,
        dtype="float32",
        overwrite=True
    )

    size_mb = os.path.getsize(output_path) / 1e6
    log.info(f"✅ Sentinel-1 saved: {output_path} ({size_mb:.1f} MB)")


def gee_sentinel1_download(study_area):
    from pyproj import Transformer
    import rasterio

    config = load_config()

    ref_raster = os.path.join(config["paths"]["raw"],
                               f"sentinel2_{config[study_area]['name']}_{config[study_area]['mineral']}.tif")
    output     = os.path.join(config["paths"]["raw"],
                               f"sentinel1_{config[study_area]['name']}_{config[study_area]['mineral']}.tif")

    if not os.path.exists(ref_raster):
        raise FileNotFoundError(
            f"Reference Sentinel-2 missing: {ref_raster}\n"
            "Run gee_sentinel2.py first."
        )

    # ── Fix 2: initialize GEE BEFORE using any ee.* calls ────────
    initialize_gee(config["project"]["gee_project"])

    # Derive bbox from actual Sentinel-2 file bounds
    with rasterio.open(ref_raster) as src:
        true_crs = src.crs.to_string()
        bounds   = src.bounds

    transformer = Transformer.from_crs(
        true_crs, "EPSG:4326", always_xy=True
    )
    lon_min, lat_min = transformer.transform(bounds.left,  bounds.bottom)
    lon_max, lat_max = transformer.transform(bounds.right, bounds.top)

    geometry = ee.Geometry.Rectangle([
        float(min(lon_min, lon_max)), float(min(lat_min, lat_max)),
        float(max(lon_min, lon_max)), float(max(lat_min, lat_max))
    ])

    log.info(f"Derived bbox: [{lon_min:.4f}, {lat_min:.4f}, "
             f"{lon_max:.4f}, {lat_max:.4f}]")

    composite = build_sentinel1_composite(geometry, config)
    download_s1_gee(composite, geometry, config, output,study_area)

    print(f"\n✅ Sentinel-1 ready: {output}")