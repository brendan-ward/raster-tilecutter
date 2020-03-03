from functools import partial
import os
import math
import json
from tempfile import TemporaryDirectory

from affine import Affine
import mercantile
import numpy as np
import rasterio

from tilecutter.rgb import hex_to_rgb
from tilecutter.png import to_smallest_png, to_paletted_png
from tilecutter.raster import get_geo_bounds, get_default_max_zoom, to_indexed_tif
from tilecutter.tiles import read_tiles


def tif_to_tiles(
    infilename,
    outpath,
    min_zoom,
    max_zoom,
    tile_size=256,
    tile_renderer=to_smallest_png,
):
    """Convert a tif to image tiles, rendered according to tile_renderer.

    By default, tiles are rendered as data using the smallest PNG image type.

    Images will be stored in subdirectories under path:
    <outpath>/<zoom>/<x>/<y>.png

    Note: tile x,y,z coordinates follow the XYZ scheme to match their numbering in an mbtiles file.

    Parameters
    ----------
    infilename : path to input GeoTIFF file
    path : root path of output tiles
    min_zoom : int, optional (default: 0)
    max_zoom : int, optional (default: None, which means it will automatically be calculated from extent)
    tile_size : int, optional (default: 256)
    tile_renderer : function, optional (default: to_smallest_png)
        function that takes as input the data array for the tile and returns a PNG
    """

    with rasterio.open(infilename) as src:

        for tile, data in read_tiles(
            src, min_zoom=min_zoom, max_zoom=max_zoom, tile_size=tile_size
        ):
            # Only write non-empty tiles
            if not np.all(data == src.nodata):

                # flip tile Y to match xyz scheme
                # TODO: should this be in path below?
                tiley = int(math.pow(2, tile.z)) - tile.y - 1

                outfilename = "{path}/{z}/{x}/{y}.png".format(
                    path=outpath, z=tile.z, x=tile.x, y=tile.y
                )
                outdir = os.path.dirname(outfilename)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                with open(outfilename, "wb") as out:
                    out.write(tile_renderer(data))


def render_tif_to_tiles(
    infilename, outpath, colormap, min_zoom, max_zoom, tile_size=256
):
    """Convert a tif to image tiles, rendered according to the colormap.

    The tif is first converted into an indexed image (if necessary) that matches the number of colors in the colormap,
    and all values not in the colormap are masked out.

    Images will be stored in subdirectories under path:
    <outpath>/<zoom>/<x>/<y>.png

    Note: tile x,y,z coordinates follow the XYZ scheme to match their numbering in an mbtiles file.

    Parameters
    ----------
    infilename : path to input GeoTIFF file
    path : root path of output tiles
    colormap : dict of values to hex color codes
    min_zoom : int, optional (default: 0)
    max_zoom : int, optional (default: None, which means it will automatically be calculated from extent)
    """

    # palette is created as a series of r,g,b values.  Positions correspond to the index
    # of each value in the image
    values = sorted(colormap.keys())
    palette = np.array([hex_to_rgb(colormap[value]) for value in values], dtype="uint8")

    with TemporaryDirectory() as tmpdir:
        with rasterio.Env() as env:
            with rasterio.open(infilename) as src:
                if src.count > 1:
                    raise ValueError("tif must be single band")

                # Convert the image to indexed, if necessary
                unique_values = np.unique(src.read(1, masked=True))
                unique_values = [v for v in unique_values if v is not np.ma.masked]

                if len(set(unique_values).difference(values)):
                    # convert the image to indexed
                    print("Converting tif to indexed tif")
                    indexedfilename = os.path.join(tmpdir, "indexed.tif")
                    to_indexed_tif(infilename, indexedfilename, values)

                else:
                    indexedfilename = infilename

            paletted_renderer = partial(
                to_paletted_png, palette=palette, nodata=src.nodata
            )
            tif_to_tiles(
                indexedfilename,
                outpath,
                min_zoom,
                max_zoom,
                tile_size,
                tile_renderer=paletted_renderer,
            )
