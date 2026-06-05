# src/ingestion/gee_sentinel1.py
# ─────────────────────────────────────────────────────────────────
# PURPOSE: Download Sentinel-1 SAR imagery over study area via GEE
# WHY SAR (Radar) MATTERS FOR GEOLOGY:
#   1. Active sensor: Sends microwave pulses and measures backscatter.
#   2. Cloud proof: Radar wavelengths penetrate tropical cloud decks entirely.
#   3. Structural mapping: Highly sensitive to surface roughness, slope orientation,
#      and structural lineaments (faults/fractures) where ore bodies often trap.
# ─────────────────────────────────────────────────────────────────
import ee
import geedim
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.INFO if hasattr(logging, 'INFO') else 20
logger = logging.getLogger(__name__)

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def initialize_gee(project_id):
    ee.Initialize(project=project_id)
    logger.info(f"GEE Initialized for Sentinel-1 with project: {project_id}")

def build_sentinel1_composite(geometry, config):
    """
    Builds a Sentinel-1 SAR median composite.
    
    WHAT ARE VV AND VH POLARIZATIONS?
    SAR sends polarized radar waves. 
    - VV (Vertical send, Vertical receive): Highly sensitive to surface roughness 
      and micro-topography (rock type textures).
    - VH (Vertical send, Horizontal receive): Sensitive to volume scattering 
      (vegetation canopy, loose soil breakdown).
    - VV/VH Ratio: Helps mathematically cancel out vegetation noise to expose 
      underlying rock geology variations.
    """
    s2_config = config["sentinel2"] # Align temporal window with Sentinel-2 data
    
    logger.info("Filtering Sentinel-1 GRD Collection...")
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(s2_config["date_start"], s2_config["date_end"])
        # IW (Interferometric Wide Swath) is the standard high-res ground mode
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        # Filter for both polarization channels
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )
    
    count = collection.size().getInfo()
    logger.info(f"Found {count} Sentinel-1 SAR scenes within date range.")
    if count == 0:
        raise ValueError("No Sentinel-1 scenes found. Check your bounding box or date ranges.")
        
    # Temporal median composite significantly collapses radar speckle noise
    composite = collection.median()
    
    # Calculate Cross-Polarization Ratio (VV - VH in dB scale is equivalent to division)
    vv = composite.select("VV")
    vh = composite.select("VH")
    ratio = vv.subtract(vh).rename("VV_VH_ratio")
    
    # Return the 3-band radar stacked image
    return composite.addBands(ratio)

def download_s1_gee(composite, geometry, config, output_path):
    """
    Downloads the SAR composite at native 10m resolution using geedim.
    geedim will auto-tile and merge seamlessly without hitting pixel caps.
    """
    crs = config["study_area"]["crs"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    logger.info(f"Preparing geedim download wrapper for Sentinel-1...")
    
    # ── FIXED HERE ──────────────────────────────────────────────────
    # composite.gd already outputs a geedim.MaskedImage instance!
    gd_image = geedim.MaskedImage(composite) 
    # ────────────────────────────────────────────────────────────────
    
    logger.info(f"Starting native 10m Sentinel-1 download -> {output_path}")
    gd_image.download(
        output_path,
        crs=crs,
        scale=10,        # Sentinel-1 native resolution is 10 meters
        region=geometry,
        dtype="float32", # Radar backscatter coefficient values are decibel floats (dB)
        overwrite=True
    )
    
    size_mb = os.path.getsize(output_path) / 1e6
    logger.info(f"✅ Download complete: {output_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    config = load_config()
    initialize_gee(config["project"]["gee_project"])
    
    bbox = config["study_area"]["bbox"]
    geometry = ee.Geometry.Rectangle([
        bbox["lon_min"], bbox["lat_min"],
        bbox["lon_max"], bbox["lat_max"]
    ])
    
    output_file = os.path.join(config["paths"]["raw"], "sentinel1_keonjhar.tif")
    
    s1_composite = build_sentinel1_composite(geometry, config)
    download_s1_gee(s1_composite, geometry, config, output_file)