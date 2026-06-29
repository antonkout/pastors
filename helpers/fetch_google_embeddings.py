"""
Fetch Google Satellite Embedding (AlphaEarth) for a given year + AOI from
Earth Engine and save a native-resolution (10 m) 64-band GeoTIFF locally.

Collection: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL  (bands A00..A63, 10 m, annual).

Requires a one-time Earth Engine authentication (`earthengine authenticate`).
After that this runs unattended. The image is downloaded via getDownloadURL in
band chunks (to stay under the request size limit) and stitched with rasterio.
"""

import os
import tempfile

import ee
import requests
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask


def clip_raster_to_aoi(raster_path, aoi_path, output_path=None, nodata=np.nan):
    """
    Mask a raster to an AOI polygon: pixels outside the polygon become `nodata`.
    Overwrites in place if `output_path` is None. Band descriptions preserved.
    """
    gdf = gpd.read_file(aoi_path)
    with rasterio.open(raster_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        descs = src.descriptions
        inside = geometry_mask(
            gdf.to_crs(src.crs).geometry.values,
            out_shape=(src.height, src.width),
            transform=src.transform, invert=True,   # True = inside AOI
        )
    data[:, ~inside] = nodata
    profile.update(dtype='float32', nodata=nodata)

    out = output_path or raster_path
    with rasterio.open(out, 'w', **profile) as dst:
        dst.write(data)
        for i, d in enumerate(descs, 1):
            if d:
                dst.set_band_description(i, d)
    n_out = int((~inside).sum())
    print(f"  clipped to AOI: {n_out:,} pixels set to nodata -> {out}")
    return out


def _init_ee(project=None):
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:
        # Fall back to interactive auth only if no stored credentials exist.
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()


def fetch_embeddings(aoi_path, year, output_path,
                     scale=10, crs=None, band_chunk=16, project=None,
                     clip=True):
    """
    Download the annual satellite embedding for `year` over the AOI bbox.

    Parameters
    ----------
    aoi_path : str
        Path to a vector AOI (GeoJSON/shapefile/...). Its bounding box is used.
    year : int
        Embedding year (e.g. 2024 for Oman, 2025 for Kenya). 2017-2025 available.
    output_path : str
        Destination GeoTIFF (64 bands, float32), e.g. data/satellite/embeddings_oman.tif
    scale : int
        Pixel size in metres (native = 10).
    crs : str or None
        Output CRS (e.g. 'EPSG:32640'). If None, the AOI's UTM zone is used.
    band_chunk : int
        Bands per download request (32 -> two requests for 64 bands).
    project : str or None
        Earth Engine Cloud project, if your account needs one.

    Returns
    -------
    output_path : str
    """
    _init_ee(project)

    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        raise ValueError(f"AOI {aoi_path} has no CRS.")
    # Output CRS: provided, else the AOI's own projected CRS, else its UTM zone.
    if crs is None:
        crs = str(gdf.crs) if gdf.crs.is_projected else str(gdf.estimate_utm_crs())

    # EE region as a lon/lat bbox (EPSG:4326).
    minx, miny, maxx, maxy = gdf.to_crs(4326).total_bounds
    region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

    col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    img = (col.filterDate(f'{year}-01-01', f'{year}-12-31')
              .filterBounds(region)
              .mosaic())
    band_names = img.bandNames().getInfo()
    print(f"  {len(band_names)} bands, year {year}, crs {crs}, scale {scale} m")

    arrays, profile = [], None
    for i in range(0, len(band_names), band_chunk):
        sub = img.select(band_names[i:i + band_chunk])
        url = sub.getDownloadURL({
            'scale': scale, 'crs': crs, 'region': region, 'format': 'GEO_TIFF',
        })
        print(f"    downloading bands {i}-{i + band_chunk - 1} ...")
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        tmp = tempfile.mktemp(suffix='.tif')
        with open(tmp, 'wb') as f:
            f.write(r.content)
        with rasterio.open(tmp) as s:
            arrays.append(s.read())
            profile = s.profile
        os.remove(tmp)

    data = np.vstack(arrays).astype(np.float32)
    profile.update(count=data.shape[0], dtype='float32', compress='lzw')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)
        for bi, nm in enumerate(band_names, 1):
            dst.set_band_description(bi, f'Emb_{nm}')

    print(f"  wrote {output_path}  ({data.shape[0]} bands, "
          f"{profile['width']}x{profile['height']})")

    # Clip to the actual AOI polygon (the download covers its bbox).
    if clip:
        clip_raster_to_aoi(output_path, aoi_path)
    return output_path


if __name__ == "__main__":
    base = "./data/satellite"
    # year per region
    fetch_embeddings("./data/geospatial/oman_aoi.geojson", 2024,
                     f"{base}/embeddings_oman.tif")
    fetch_embeddings("./data/geospatial/kenya_aoi.geojson", 2025,
                     f"{base}/embeddings_kenya.tif")
