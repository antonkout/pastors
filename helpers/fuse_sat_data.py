"""
Fuse Dragonette hyperspectral with Sentinel-2 NIR/SWIR bands.
Includes pan-sharpening for 20m bands to reduce blocky artifacts.

Pan-sharpening method: Brovey-like intensity modulation using 
spectrally similar Dragonette bands as the high-resolution source.
"""

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from scipy.ndimage import uniform_filter


def pansharpen_band(lowres_band, highres_reference, filter_size=4):
    """
    Pan-sharpen a low-resolution band using a high-resolution reference.
    
    Uses ratio-based sharpening: preserves spectral values while adding
    spatial detail from the reference band.
    
    Parameters
    ----------
    lowres_band : np.ndarray
        The resampled low-resolution band (already at target resolution)
    highres_reference : np.ndarray
        High-resolution band to extract spatial detail from
    filter_size : int
        Size of smoothing filter (roughly lowres/highres ratio)
    
    Returns
    -------
    np.ndarray
        Pan-sharpened band
    """
    # Smooth the high-res reference to match low-res spatial frequency
    highres_smoothed = uniform_filter(highres_reference.astype(np.float32), size=filter_size)
    
    # Avoid division by zero
    highres_smoothed = np.where(highres_smoothed == 0, 1e-8, highres_smoothed)
    
    # Calculate detail ratio (high-freq component)
    detail_ratio = highres_reference / highres_smoothed
    
    # Apply detail to low-res band
    sharpened = lowres_band * detail_ratio
    
    # Preserve original mean to avoid brightness shift
    sharpened = sharpened * (np.nanmean(lowres_band) / (np.nanmean(sharpened) + 1e-8))
    
    return sharpened.astype(np.float32)


def fuse_dragonette_sentinel2_pansharpened(
    dragonette_path: str,
    sentinel2_path: str,
    output_path: str,
    s2_bands_to_extract: dict = None
):
    """
    Fuse Dragonette with Sentinel-2 bands, pan-sharpening the 20m bands.
    
    Parameters
    ----------
    dragonette_path : str
        Path to Dragonette hyperspectral GeoTIFF (23 bands)
    sentinel2_path : str
        Path to stacked Sentinel-2 GeoTIFF
    output_path : str
        Path for output fused GeoTIFF
    s2_bands_to_extract : dict
        Band config: {index: {'name': str, 'gsd': int, 'sharpen_ref': int or None}}
    """
    
    # Default band configuration for your 11-band S2 stack
    # sharpen_ref = Dragonette band index (0-based) to use for pan-sharpening
    if s2_bands_to_extract is None:
        s2_bands_to_extract = {
            7: {
                'name': 'B8_842nm',
                'gsd': 10,
                'sharpen_ref': None  # 10m native, no sharpening needed
            },
            8: {
                'name': 'B8A_865nm',
                'gsd': 20,
                'sharpen_ref': 22  # Use Dragonette 799nm (closest to 865nm)
            },
            10: {
                'name': 'B11_1614nm',
                'gsd': 20,
                'sharpen_ref': 22  # Use Dragonette 799nm (best proxy for SWIR structure)
            },
            11: {
                'name': 'B12_2202nm',
                'gsd': 20,
                'sharpen_ref': 22  # Use Dragonette 799nm
            }
        }
    
    # 1. Open Dragonette as reference
    print("Reading Dragonette...")
    with rasterio.open(dragonette_path) as drag:
        dragonette_data = drag.read().astype(np.float32)
        target_crs = drag.crs
        target_transform = drag.transform
        target_height, target_width = drag.height, drag.width
        target_res = target_transform[0]
        
        print(f"  Shape: {dragonette_data.shape}")
        print(f"  Resolution: {target_res:.2f}m")
    
    # 2. Open Sentinel-2 and process each band
    print(f"\nProcessing Sentinel-2 bands...")
    s2_processed = []
    s2_descriptions = []
    
    with rasterio.open(sentinel2_path) as s2:
        s2_res = s2.transform[0]
        print(f"  S2 has {s2.count} bands, resolution: {s2_res:.1f}m")
        
        for band_idx, config in s2_bands_to_extract.items():
            band_name = config['name']
            native_gsd = config['gsd']
            sharpen_ref = config['sharpen_ref']
            
            print(f"\n  Processing band {band_idx} ({band_name}, native {native_gsd}m)...")
            
            # Resample to target grid
            resampled = np.empty((target_height, target_width), dtype=np.float32)
            
            reproject(
                source=rasterio.band(s2, band_idx),
                destination=resampled,
                src_transform=s2.transform,
                src_crs=s2.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
            
            # Pan-sharpen if needed
            if sharpen_ref is not None:
                print(f"    Pan-sharpening using Dragonette band {sharpen_ref + 1}...")
                
                # Get the reference band from Dragonette
                ref_band = dragonette_data[sharpen_ref]
                
                # Calculate filter size based on resolution ratio
                filter_size = int(np.ceil(native_gsd / target_res))
                
                # Apply pan-sharpening
                resampled = pansharpen_band(resampled, ref_band, filter_size=filter_size)
                print(f"    Done (filter_size={filter_size})")
            else:
                print(f"    No sharpening needed (10m native)")
            
            s2_processed.append(resampled)
            s2_descriptions.append(band_name)
    
    # 3. Stack all bands
    s2_stack = np.stack(s2_processed, axis=0)
    fused = np.vstack([dragonette_data, s2_stack])
    
    print(f"\nFused shape: {fused.shape}")
    print(f"  Dragonette: {dragonette_data.shape[0]} bands (503-799nm)")
    print(f"  Sentinel-2: {s2_stack.shape[0]} bands ({', '.join(s2_descriptions)})")
    
    # 4. Create band descriptions
    dragonette_wavelengths = [
        503, 510, 519, 535, 549, 570,
        584, 600, 614,
        635, 649, 660, 669, 679, 690, 699,
        711, 722, 734, 750, 764, 782,
        799
    ]
    descriptions = [f"Drag_{wl}nm" for wl in dragonette_wavelengths]
    descriptions += [f"S2_{name}" for name in s2_descriptions]
    
    # 5. Write output
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': target_width,
        'height': target_height,
        'count': fused.shape[0],
        'crs': target_crs,
        'transform': target_transform,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256
    }
    
    print(f"\nWriting to {output_path}...")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(fused)
        for i, desc in enumerate(descriptions, start=1):
            dst.set_band_description(i, desc)
    
    print("Done!")
    print(f"Output: {fused.shape[0]} bands, {target_width}x{target_height} pixels")
    
    return output_path


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    
    # --- EDIT THESE PATHS ---
    dragonette_path = "/Users/antonkout/Documents/doctorado/data/satellite/dragonette.tif"
    sentinel2_path = "/Users/antonkout/Documents/doctorado/data/satellite/sentinel-2_06112025.tif"
    output_path = "/Users/antonkout/Documents/doctorado/data/satellite/fused_satellite.tif"
    
    # Run fusion with pan-sharpening
    fuse_dragonette_sentinel2_pansharpened(
        dragonette_path=dragonette_path,
        sentinel2_path=sentinel2_path,
        output_path=output_path
    )