"""Build a feature stack from a multispectral satellite raster.

Appends NDVI, RE_NDVI, Blue/Red, Blue/NIR ratios and the bare-soil /
sparse-vegetation indices BSI, SAVI, MSAVI to the original bands. Band positions are looked up by `common_name` in a STAC-style
metadata JSON (eo:bands).
"""
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def _read_band_indices(metadata_path):
    """Return {common_name: 1-indexed band number} from STAC eo:bands."""
    with open(metadata_path) as f:
        md = json.load(f)
    bands = None
    for a in md.get('assets', {}).values():
        if isinstance(a, dict) and 'eo:bands' in a:
            bands = a['eo:bands']
            break
    if bands is None:
        bands = md.get('eo:bands')
    if not bands:
        raise ValueError(f"No eo:bands in {metadata_path}")
    return {b['common_name']: i + 1
            for i, b in enumerate(bands) if b.get('common_name')}


def build_feature_stack(sat_path, metadata_path, out_path=None, embeddings_path=None,
                        embeddings_resampling='nearest'):
    """
    Stack original satellite bands + spectral indices into one GeoTIFF.

    Indices computed if the required bands exist (per metadata):
      - NDVI            : (NIR - Red)   / (NIR + Red)
      - RE_NDVI         : (NIR - RedEdge) / (NIR + RedEdge)
      - Blue_Red_Ratio  : Blue / Red
      - Blue_NIR_Ratio  : Blue / NIR
      - BSI             : ((SWIR+Red) - (NIR+Blue)) / ((SWIR+Red) + (NIR+Blue))
      - SAVI            : 1.5 * (NIR - Red) / (NIR + Red + 0.5)
      - MSAVI           : (2*NIR+1 - sqrt((2*NIR+1)^2 - 8*(NIR-Red))) / 2

    Parameters
    ----------
    sat_path : str
        Multispectral satellite GeoTIFF.
    metadata_path : str
        STAC-style JSON with `eo:bands` listing each band's `common_name`.
    out_path : str
        Output GeoTIFF (float32, LZW).

    Returns
    -------
    n_bands : int
        Total band count in the output.
    names : list[str]
        Feature names in band order.
    """
    cn = _read_band_indices(metadata_path)
    print(f"  bands from metadata: {cn}")

    with rasterio.open(sat_path) as src:
        sat = src.read().astype(np.float32)
        meta = src.meta.copy()
    n_sat = sat.shape[0]

    feats = [sat]
    names = [f'Band_{i+1}' for i in range(n_sat)]

    def b(name):
        idx = cn.get(name)
        return sat[idx - 1] if idx else None

    blue, red, re_, nir = b('blue'), b('red'), b('rededge'), b('nir')
    swir = b('swir16')
    eps = 1e-8

    if nir is not None and red is not None:
        feats.append(np.where((nir + red) > 0,
                              (nir - red) / (nir + red + eps), 0)[None])
        names.append('NDVI')
    if nir is not None and re_ is not None:
        feats.append(np.where((nir + re_) > 0,
                              (nir - re_) / (nir + re_ + eps), 0)[None])
        names.append('RE_NDVI')
    if blue is not None and red is not None:
        feats.append(np.where(red > 0, blue / (red + eps), 0)[None])
        names.append('Blue_Red_Ratio')
    if blue is not None and nir is not None:
        feats.append(np.where(nir > 0, blue / (nir + eps), 0)[None])
        names.append('Blue_NIR_Ratio')

    # Bare-soil / sparse-vegetation indices (use S2 reflectance bands, scaled 0-1).
    # BSI  : bare-soil discriminator (SWIR/Red vs NIR/Blue contrast).
    # SAVI : soil-adjusted NDVI (L=0.5) - less soil-background bias than NDVI.
    # MSAVI: self-adjusting SAVI for sparse canopy over bright soil (no fixed L).
    if swir is not None and red is not None and nir is not None and blue is not None:
        num = (swir + red) - (nir + blue)
        den = (swir + red) + (nir + blue)
        feats.append(np.where(den != 0, num / (den + eps), 0)[None])
        names.append('BSI')
    if nir is not None and red is not None:
        feats.append((1.5 * (nir - red) / (nir + red + 0.5))[None])
        names.append('SAVI')
        msavi = (2 * nir + 1 - np.sqrt(np.maximum((2 * nir + 1) ** 2
                                                  - 8 * (nir - red), 0))) / 2
        feats.append(msavi[None])
        names.append('MSAVI')

    # Append native-resolution embedding bands, resampled onto the satellite
    # grid. 'nearest' keeps hard 10 m steps (blocky); 'bilinear' smooths across
    # the 10 m cells (softer, fewer square artifacts). Same memory either way.
    if embeddings_path is not None:
        rs = getattr(Resampling, embeddings_resampling)
        H, W = sat.shape[1], sat.shape[2]
        with rasterio.open(embeddings_path) as emb:
            edata = emb.read()
            enames = [emb.descriptions[i] or f'A{i:02d}' for i in range(emb.count)]
            edst = np.empty((emb.count, H, W), dtype=np.float32)
            reproject(
                source=edata, destination=edst,
                src_transform=emb.transform, src_crs=emb.crs,
                dst_transform=meta['transform'], dst_crs=meta['crs'],
                src_nodata=emb.nodata, dst_nodata=emb.nodata,
                resampling=rs,
            )
        feats.append(edst)
        names += [n if n.startswith('Emb_') else f'Emb_{n}' for n in enames]
        print(f"  + {len(enames)} embedding bands ({embeddings_resampling})")

    stacked = np.concatenate(feats, axis=0).astype(np.float32)
    meta.update(count=stacked.shape[0], dtype='float32', compress='lzw')

    if out_path:                       # write only if a path is given
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(stacked)
            for i, name in enumerate(names, 1):
                dst.set_band_description(i, name)
        print(f"✓ {out_path}")
    print(f"  features ({len(names)}): {names}")
    return stacked, names, meta