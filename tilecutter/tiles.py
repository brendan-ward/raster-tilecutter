from affine import Affine
import mercantile
import numpy as np
from progress.bar import IncrementalBar as Bar
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from tilecutter.raster import get_geo_bounds, get_default_max_zoom


def read_tiles(src, min_zoom=0, max_zoom=None, tile_size=256, resampling="nearest"):
    """This function is a generator that reads all tiles
    that overlap with the extent of src between min_zoom and max_zoom.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Input dataset, opened for reading
    min_zoom : int, optional (default 0)
    max_zoom : int, optional (default None)
        If None, max_zoom will be calculated based on the extent of src
    tile_size : int, optional (default 256)
        length and width of tile
    resampling : str, optional (default 'nearest')
        Must be a supported value of rasterio.enums.Resampling

    Yields
    ------
    tile (mercantile.Tile), tile data (of shape (tile_size, tile_size))
    """

    def _read_tile(vrt, tile, tile_size=256):
        """Read a tile of data from the VRT.

        If the tile bounds fall outside the vrt bounds, we have to calculate
        offsets and widths ourselves (because WarpedVRT does not allow boundless reads)
        and paste the data that were read into an otherwise blank tile
        (filled with Nodata value).

        Parameters
        ----------
        vrt : rasterio.WarpedVRT
            WarpedVRT initialized from the data source.  Example:
                with WarpedVRT(
                    src,
                    crs="EPSG:3857",
                    nodata=src.nodata,
                    resampling=Resampling.nearest,
                    width=tile_size,
                    height=tile_size,
                ) as vrt
        tile : mercantile.Tile
            Tile object describing z, x, y coordinates
        tile_size : int, optional (default 256)
            length and width of tile

        Returns
        -------
        numpy ndarray (2D) or None
            None is returned if tile is completely filled with vrt.nodata values
        """

        tile_bounds = mercantile.xy_bounds(*tile)
        window = vrt.window(*tile_bounds)

        dst_transform = vrt.window_transform(window)
        scaling = Affine.scale(window.width / tile_size, window.height / tile_size)
        dst_transform *= scaling

        x_res = abs(dst_transform.a)
        y_res = abs(dst_transform.e)

        left_offset = max(int(round((vrt.bounds[0] - tile_bounds[0]) / x_res, 0)), 0)
        right_offset = max(int(round((tile_bounds[2] - vrt.bounds[2]) / x_res, 0)), 0)

        bottom_offset = max(int(round((vrt.bounds[1] - tile_bounds[1]) / y_res, 0)), 0)
        top_offset = max(int(round((tile_bounds[3] - vrt.bounds[3]) / y_res, 0)), 0)

        width = tile_size - left_offset - right_offset
        height = tile_size - top_offset - bottom_offset

        data = vrt.read(out_shape=(1, height, width), window=window)

        if np.all(data == vrt.nodata):
            return None

        if width != tile_size or height != tile_size:
            # Create a blank tile (filled with nodata) and paste in data
            out = np.empty((1, tile_size, tile_size), dtype=vrt.dtypes[0])
            out.fill(vrt.nodata)
            out[
                0,
                top_offset : top_offset + data.shape[1],
                left_offset : left_offset + data.shape[2],
            ] = data
            data = out

        return data[0]

    # Note: transform, width, height are omitted; these are automatically calculated by GDAL
    with WarpedVRT(
        src,
        crs="EPSG:3857",
        nodata=src.nodata,
        resampling=Resampling[resampling],
        warp_mem_limit=1024,
        warp_extras={"NUM_THREADS": "ALL_CPUS"},
    ) as vrt:

        if max_zoom is None:
            max_zoom = get_default_max_zoom(src)

        for zoom in range(min_zoom, max_zoom + 1):

            tiles = list(mercantile.tiles(*get_geo_bounds(src), zoom))

            for tile in Bar(
                f"Zoom {zoom}: ",
                max=len(tiles),
                suffix="%(percent)d%% [time: %(elapsed_td)s]",
            ).iter(tiles):
                data = _read_tile(vrt, tile, tile_size)
                yield tile, data

