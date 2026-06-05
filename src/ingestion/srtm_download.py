# src/ingestion/srtm_download.py
# ─────────────────────────────────────────────────────────────────
# PURPOSE: Download SRTM Elevation data and compute terrain derivatives.
# WHY TERRAIN MATTERS FOR GEOLOGY:
#   1. Lithological Control: Different rock units weather differently. Resistant 
#      iron-ore formations often form prominent high-elevation ridges.
#   2. Structural Lineaments: Fault lines and fracture zones present distinct 
#      topographic signatures (linear valleys, drainage offsets, steep scarps).
#   3. Hydrological Accumulation (TWI): Weathered mineral deposits and secondary 
#      enrichments tend to accumulate in drainage pathways and breaks-in-slope.
# ─────────────────────────────────────────────────────────────────
import ee
import geedim
import yaml
import os
import logging
import rasterio
import numpy as np
from scipy.ndimage import sobel, uniform_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.INFO if hasattr(logging, 'INFO') else 20
logger = logging.getLogger(__name__)

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def initialize_gee(project_id):
    ee.Initialize(project=project_id)
    logger.info(f"GEE Initialized for SRTM with project: {project_id}")

def compute_local_twi(dem_array, pixel_size_m=30.0):
    """
    Computes the Topographic Wetness Index (TWI) using pure numpy/scipy.
    Formula: TWI = ln( alpha / tan(beta) )
      where alpha = Upslope contributing area (approximated via neighborhood smoothing)
            beta  = Slope gradient in radians
    
    High TWI values indicate valleys, depressions, and natural drainage channels.
    """
    logger.info("Computing local slope and aspect gradients via Sobel kernels...")
    
    # Horn's Method (1981) via Sobel filters matching standard GDAL/QGIS specifications
    dz_dx = sobel(dem_array, axis=1) / (8.0 * pixel_size_m)
    dz_dy = sobel(dem_array, axis=0) / (8.0 * pixel_size_m)
    
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    
    # Force minimal slope bounding to prevent division-by-zero errors in tangent math
    slope_rad = np.clip(slope_rad, np.radians(0.01), np.radians(89.9))
    
    logger.info("Approximating local flow accumulation routing...")
    # Uniform smoothing filter effectively acts as a catchment convergence metric
    flow_acc = uniform_filter(dem_array, size=5).astype(np.float32)
    flow_acc = np.abs(flow_acc - dem_array) + 1.0 
    
    logger.info("Evaluating Topographic Wetness Index array...")
    twi = np.log(flow_acc / (np.tan(slope_rad) + 1e-6))
    
    return slope_deg, twi

def build_terrain_stack(geometry, config, output_path):
    """
    Extracts GEE terrain properties and layers them with local secondary indices.
    """
    srtm_config = config["srtm"]
    crs = config["study_area"]["crs"]
    scale = srtm_config["scale"] # Standard SRTM resolution is 30m
    
    logger.info(f"Fetching GEE SRTM Elevation Model...")
    dem_image = ee.Image(srtm_config["collection"])
    
    # Extract structural terrain products directly via GEE server side
    terrain_products = ee.Terrain.products(dem_image)
    
    # Capture elevation, slope, aspect, and hillshade
    selected_terrain = terrain_products.select(["elevation", "slope", "aspect", "hillshade"])
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Intermediate extraction via geedim to compute TWI locally on the array
    logger.info("Downloading intermediate DEM layers via geedim client...")
    tmp_path = output_path.replace(".tif", "_base.tif")
    
    gd_image = geedim.MaskedImage(selected_terrain)
    gd_image.download(
        tmp_path,
        crs=crs,
        scale=scale,
        region=geometry,
        dtype="float32",
        overwrite=True
    )
    
    # Read downloaded raster array into system memory
    with rasterio.open(tmp_path) as src:
        profile = src.profile.copy()
        raster_data = src.read() # Shape: (4, Height, Width)
        
    # Isolate elevation band array
    elevation_matrix = raster_data[0]
    
    # Execute structural TWI calculations
    _, twi_matrix = compute_local_twi(elevation_matrix, pixel_size_m=float(scale))
    
    # Append the newly calculated TWI matrix to create a 5-band raster architecture
    final_raster_stack = np.vstack([raster_data, np.expand_dims(twi_matrix, axis=0)])
    
    # Update raster IO metadata profile properties
    profile.update(
        count=5,
        dtype="float32",
        compress="lzw"
    )
    
    logger.info(f"Writing complete 5-band composite terrain stack -> {output_path}")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(final_raster_stack.astype(np.float32))
        
        # Tag bands with functional titles for clean scannability in QGIS / Python loaders
        band_names = ["elevation", "slope", "aspect", "hillshade", "TWI"]
        for idx, name in enumerate(band_names, start=1):
            dst.update_tags(idx, name=name)
            
    # Clean up intermediate artifact file safely
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        
    size_mb = os.path.getsize(output_path) / 1e6
    logger.info(f"✅ Terrain extraction finalized successfully: {size_mb:.1f} MB")

if __name__ == "__main__":
    config = load_config()
    initialize_gee(config["project"]["gee_project"])
    
    bbox = config["study_area"]["bbox"]
    geometry = ee.Geometry.Rectangle([
        bbox["lon_min"], bbox["lat_min"],
        bbox["lon_max"], bbox["lat_max"]
    ])
    
    output_file = os.path.join(config["paths"]["raw"], "srtm_terrain_keonjhar.tif")
    build_terrain_stack(geometry, config, output_file)