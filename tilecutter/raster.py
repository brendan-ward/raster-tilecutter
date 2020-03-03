"""Raster file processing functions"""

from collections import defaultdict
import math
import os
import numpy as np
import rasterio
from rasterio.warp import transform_bounds


EPSILON = 1.0e-10

# Bounds of web mercator in geographic coordinates
WEB_MERCATOR_BOUNDS = (
    -180 + EPSILON,  # w
    -85.051129,  # s
    180 - EPSILON,  # e
    85.051129,  # n
)


def get_geo_bounds(src):
    """Calculate the geographic bounds of the data source, and clip by max
    extent of Web Mercator projection.

    Parameters
    ----------
    src : rasterio.DatasetReader

    Returns
    -------
    bounds tuple: west, south, east, north
    """

    w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    # clip to limits of Web Mercator projection
    return (
        max(w, WEB_MERCATOR_BOUNDS[0]),
        max(s, WEB_MERCATOR_BOUNDS[1]),
        min(e, WEB_MERCATOR_BOUNDS[2]),
        min(n, WEB_MERCATOR_BOUNDS[3]),
    )


def get_mbtiles_meta(src, center_zoom=0):
    """Get mbtiles metadata specific to this data source

    Parameters
    ----------
    src : rasterio.DatasetReader
    center_zoom : int, optional (default 0)
        zoom to use for centerpoint

    Returns
    -------
    dict with name, bounds, and center
    """

    center_long, center_lat = src.lnglat()

    return {
        "name": os.path.split(src.name)[-1],
        "bounds": ",".join("{0:4f}".format(v) for v in get_geo_bounds(src)),
        "center": "{0:4f},{1:4f},{2}".format(center_long, center_lat, center_zoom),
    }


def get_default_max_zoom(src):
    """Calculate a default max zoom based on raster dimensions.

    Derived from rio-mbtiles.

    Parameters
    ----------
    src : rasterio.DatasetReader

    Returns
    -------
    int
    """

    # TODO: modify this to best preserve pixel size
    w, s, e, n = get_geo_bounds(src)
    zw = int(round(math.log(360.0 / (e - w), 2.0)))
    zh = int(round(math.log(170.1022 / (n - s), 2.0)))
    return max(zw, zh)


def to_indexed_tif(infilename, outfilename, values):
    """Converts the input tif to uint8 indexed data.  Input tif must be a single-band
    image.

    Each value in the output is assigned the index of that value within values.

    Nodata is explicitly set to the size of the values list.

    Parameters
    ----------
    infilename : input tif filename
    outfilename : output tif filename
    values : list-like of values
    """

    nodata_value = len(values)
    if nodata_value > 255:
        raise ValueError("There must be less than 255 values to create an indexed tif")

    with rasterio.open(infilename) as src:
        if src.count > 1:
            raise ValueError("Input must be a single-band image")

        data = src.read(1)
        out_data = np.empty(shape=data.shape, dtype="uint8")
        out_data.fill(nodata_value)

        for index, value in enumerate(values):
            out_data[data == value] = index

        meta = src.meta.copy()
        meta.update({"dtype": "uint8", "nodata": nodata_value})
        with rasterio.open(outfilename, "w", **meta) as out:
            out.write(out_data, 1)


def unique_to_indexed(arr):
    """
    Convert an array to indexed values.
    Parameters
    ----------
    arr: ndarray, assumed to be integer type

    Returns
    -------
    (indexed array, unique values)
    """

    unique = np.unique(arr)
    if isinstance(arr, np.ma.masked_array):
        out = np.ma.copy(arr)
        unique = unique.compressed()
    else:
        out = np.copy(arr)

    for i in range(0, unique.size):
        out[arr == unique[i]] = i

    return out, unique


def has_matching_attributes(rasters, attribute):
    """Validate that all input rasters have the same value for the attribute

    Parameters
    ----------
    rasters : iterable of rasterio.DatasetReaders
    attribute : string
        name of attribute to test

    Returns
    -------

    bool: True if all rasters match, False otherwise
    """

    # atts = ("crs", "transform", "width", "height")
    # att_values = defaultdict(set)
    value = None
    for src in rasters:
        next_value = str(getattr(src, att))
        if value is None:
            value = next_value
        elif value != next_value:
            return False

    return True
