# src/ingestion/gee_sentinel2.py
# ─────────────────────────────────────────────────────────────────
# PURPOSE: Clean download of Sentinel-2 surface reflectance
#          with explicit coordinate header locking via geedim.
# ─────────────────────────────────────────────────────────────────
import ee
import geedim
import yaml
import os
import logging
import rasterio
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def initialize_gee(project_id):
    ee.Initialize(project=project_id)
    log.info(f"GEE Initialized successfully: {project_id}")

def build_collection(config, geometry):
    s2 = config["sentinel2"]
    bands = [b for b in s2["bands"] if b != "SCL"]
    
    collection = (ee.ImageCollection(s2["collection"])
                  .filterBounds(geometry)
                  .filterDate(s2["date_start"], s2["date_end"])
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", s2["cloud_threshold"])))
    
    count = collection.size().getInfo()
    log.info(f"Scenes discovered in target footprint: {count}")
    if count == 0:
        raise ValueError("No low-cloud scenes found. Widen date windows.")
        
    def mask_clouds(img):
        scl = img.select("SCL")
        mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(mask)
        
    composite = collection.map(mask_clouds).select(bands).median().toUint16()
    return composite


def gee_sentinel2_download(study_area):    
    config = load_config()
    initialize_gee(config["project"]["gee_project"])
    
    mineral = config[study_area]["mineral"]
    area_name    = config[study_area]["name"]
    
    bbox = config[study_area]["bbox"]
    crs_target = config[study_area]["crs"]
    output_path = os.path.join(config["paths"]["raw"], f"sentinel2_{mineral}_{area_name}.tif")
    
    # Explicitly build the search geometry from the config lat/lon degrees
    geometry = ee.Geometry.Rectangle([
        bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]
    ])
    
    composite = build_collection(config, geometry)
    
    log.info(f"Starting geedim download wrapper targeting {crs_target}...")
    gd_image = geedim.MaskedImage(composite)
    
    # CRITICAL FIX: Force overwrite to destroy the corrupted offset file
    gd_image.download(
        output_path, 
        crs=crs_target, 
        scale=config["sentinel2"]["scale"], 
        region=geometry, 
        dtype="uint16", 
        overwrite=True
    )
    
    # ── Immediate Header Verification ──
    with rasterio.open(output_path) as src:
        log.info(f"✅ New Sentinel-2 file locked down successfully!")
        log.info(f"   -> True Grid Bounds: {dict(zip(['left', 'bottom', 'right', 'top'], src.bounds))}")
        log.info(f"   -> CRS: {src.crs}")