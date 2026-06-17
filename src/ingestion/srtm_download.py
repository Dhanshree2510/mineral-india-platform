# src/ingestion/srtm_download.py  ── FIXED
import ee
import geedim
import yaml
import os
import logging
import rasterio
import numpy as np
from scipy.ndimage import sobel, uniform_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def load_config(path="config.yaml"):    # ← Fix 1
    with open(path) as f:
        return yaml.safe_load(f)


def initialize_gee(project_id):
    ee.Initialize(project=project_id)
    log.info(f"GEE Initialized for SRTM: {project_id}")


def compute_local_twi(dem_array, pixel_size_m=30.0):
    """
    Computes TWI using scipy — no richdem needed.
    TWI = ln(flow_accumulation / tan(slope))
    High TWI = drainage zone = minerals accumulate here.
    """
    dz_dx = sobel(dem_array, axis=1) / (8.0 * pixel_size_m)
    dz_dy = sobel(dem_array, axis=0) / (8.0 * pixel_size_m)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

    slope_rad = np.clip(slope_rad,
                        np.radians(0.01),
                        np.radians(89.9))

    flow_acc = uniform_filter(dem_array, size=5).astype(np.float32)
    flow_acc = np.abs(flow_acc - dem_array) + 1.0

    twi = np.log(flow_acc / (np.tan(slope_rad) + 1e-6))

    log.info(f"TWI range: {twi.min():.2f} – {twi.max():.2f}")
    return np.degrees(slope_rad), twi


def build_terrain_stack(geometry, config, output_path):
    """
    Downloads SRTM terrain bands from GEE and appends local TWI.

    Output bands (5 total):
      1: elevation   (metres)
      2: slope       (degrees)
      3: aspect      (degrees, 0=north)
      4: hillshade   (0-255, visual)
      5: TWI         (dimensionless, high=drainage zone)
    """
    srtm_config = config["srtm"]
    crs         = config["study_area"]["crs"]
    scale       = srtm_config["scale"]

    log.info("Fetching SRTM from GEE...")
    dem_image = ee.Image(srtm_config["collection"])
    terrain   = ee.Terrain.products(dem_image).select(
        ["elevation", "slope", "aspect", "hillshade"]
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path.replace(".tif", "_base.tif")

    log.info("Downloading 4-band terrain base via geedim...")

    # ── Fix 3: use MaskedImage (works on your geedim version) ─────
    gd_image = geedim.MaskedImage(terrain)

    gd_image.download(
        tmp_path,
        crs=crs,
        scale=scale,
        region=geometry,
        dtype="float32",
        overwrite=True
    )

    # Read downloaded raster
    with rasterio.open(tmp_path) as src:
        profile = src.profile.copy()
        data    = src.read()          # shape: (4, H, W)

    # Compute TWI locally from elevation band
    elevation  = data[0]
    _, twi     = compute_local_twi(elevation, pixel_size_m=float(scale))

    # Stack: elevation, slope, aspect, hillshade, TWI
    final = np.vstack([data, twi[np.newaxis, :, :]])

    profile.update(count=5, dtype="float32", compress="lzw")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(final.astype(np.float32))
        for i, name in enumerate(
            ["elevation", "slope", "aspect", "hillshade", "TWI"], 1
        ):
            dst.update_tags(i, name=name)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    size_mb = os.path.getsize(output_path) / 1e6
    log.info(f"✅ Terrain stack saved: {output_path} ({size_mb:.1f} MB)")
    log.info("   Bands: elevation, slope, aspect, hillshade, TWI")


def srtm_download(study_area):
    from pyproj import Transformer

    config = load_config()
    area_name = config[study_area]["name"]
    mineral = config[study_area]["mineral"]
    ref_raster = os.path.join(config["paths"]["raw"],
                               f"sentinel2_{mineral}_{area_name}.tif")
    output     = os.path.join(config["paths"]["raw"],
                               f"srtm_terrain_{mineral}_{area_name}.tif")

    if not os.path.exists(ref_raster):
        raise FileNotFoundError(
            f"Sentinel-2 reference missing: {ref_raster}\n"
            "Run gee_sentinel2.py first."
        )

    # ── Fix 2: initialize GEE FIRST before any ee.* usage ────────
    initialize_gee(config["project"]["gee_project"])

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

    log.info(f"Bbox from S2 file: [{lon_min:.4f}, {lat_min:.4f}, "
             f"{lon_max:.4f}, {lat_max:.4f}]")

    build_terrain_stack(geometry, config, output)

    print(f"\n✅ SRTM terrain ready: {output}")