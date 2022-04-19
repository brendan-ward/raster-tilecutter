"""PNG processing functions"""


from io import BytesIO

import numpy as np
from numpy.ma.core import is_masked
from PIL import Image
import rasterio
from rasterio.dtypes import get_minimum_dtype

from tilecutter.rgb import to_rgb_array, to_rgba_array


MAX_VALUE = {"P": 255, "L": 255, "RGB": 16777215, "RGBA": 4294967295}


def get_smallest_image_type(arr):
    """Determine the smallest image type that will fit the data type of
    the input array.

    Parameters
    ----------
    arr : numpy array

    Raises
    ------
    NotImplementedError
        raised if array data type is not uint8, uint16, or uint32

    Returns
    -------
    PIL Image type: one of "L", "RGB", "RGBA"
    """

    target_dtype = get_minimum_dtype(arr)

    if target_dtype == "uint8":
        return "L"
    elif target_dtype == "uint16":
        return "RGB"
    elif target_dtype == "uint32":
        if arr.max() <= MAX_VALUE["RGB"]:
            return "RGB"
        return "RGBA"
    else:
        raise NotImplementedError(
            "data type is not yet supported: {}".format(target_dtype)
        )


def to_smallest_png(arr, image_type=None):
    """
    Convert an array to PNG, using the smallest PNG bit depth that
    will contain the data range: 8, 24, or 32.

    If the input is a masked array, the maximum value of the data type
    will be used to fill nodata.

    You can pre-fill nodata with a different value, but if you use a value well
    outside your value range, this may force use of a larger output PNG bit depth
    than is ideal.

    This can produce paletted PNGs if image_type=P; the values will be stored
    into the blue value of the palette.  This is helpful if the image needs to
    be decoded as RGB type in the map client.

    Parameters
    ----------
    arr: input array or masked array, must have dtype of uint8, uint16, or uint32

    Returns
    -------
    PNG bytes
    """

    if arr.dtype.kind not in ("u", "i"):
        raise ValueError("Input array must be integer type")

    if image_type is None:
        image_type = get_smallest_image_type(arr)

    else:
        if not image_type in ("P", "L", "RGB", "RGBA"):
            raise ValueError("Image type must be one of: P, L, RGB, RGBA")

    if image_type == "P":
        values = np.arange(0, arr.max() + 1, dtype="uint8")
        zeros = np.zeros_like(values)
        # store values into blue value of palette
        palette = np.vstack([zeros, zeros, values]).T
        return to_paletted_png(arr.astype("uint8"), palette)

    # IMPORTANT: browsers seem to distort alpha values, so RGBA is not valid for encoding data that must be read out
    # with exactly the same values
    if image_type == "RGBA":
        raise ValueError(
            "RGBA is not currently supported due to variable decoding in browsers"
        )

    if is_masked(arr):
        # If it is masked, fill it with appropriate nodata value
        image_type_max = MAX_VALUE[image_type]
        if arr.max() < image_type_max:
            arr = arr.filled(image_type_max)
        else:
            raise ValueError(
                "Max of value range must be max of data type - 1"
                "to allow max of data type to be nodata value"
            )

    if image_type == "L":
        image_data = arr.astype("uint8")

    elif image_type == "RGB":
        image_data = to_rgb_array(np.asarray(arr))

    elif image_type == "RGBA":
        image_data = to_rgba_array(np.asarray(arr))

    else:
        raise NotImplementedError("values require an image type that is not supported")

    img = Image.frombuffer(
        image_type, (arr.shape[1], arr.shape[0]), image_data, "raw", image_type, 0, 1
    )

    buf = BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)  # rewind to beginning of buffer
    return buf.read()


def to_paletted_png(arr, palette, nodata=None):
    """
    Render an array as a paletted PNG.

    Parameters
    ----------
    arr : input array or masked array, must have dtype of uint8
    palette : numpy array of 8 bit tuples [(r, g, b), ...], where the index corresponds to the value in the image
    nodata : nodata value, will be set as transparent in the image (optional, default: None)

    Returns
    -------
    PNG bytes
    """

    if arr.dtype.kind not in ("u", "i"):
        raise ValueError("Input array must be integer type")

    # TODO: validate palette: must be of a small enough size and have rgb tuples

    nodata_index = None
    if is_masked(arr) or nodata is not None:
        # If it is masked, fill with index at end of palette
        nodata_index = len(palette)

        # add nodata color to palette (set as transparent below)
        palette = np.append(palette, (0, 0, 0))

        if is_masked(arr):
            arr = arr.filled(nodata_index)
        else:
            arr[arr == nodata] = nodata_index

    img = Image.frombuffer("P", (arr.shape[1], arr.shape[0]), arr, "raw", "P", 0, 1)
    # palette must be a list of [r, g, b, r, g, b, ...]  values
    img.putpalette(palette.flatten().tolist(), "RGB")

    if nodata_index is not None:
        img.info["transparency"] = nodata_index

    buf = BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)  # rewind to beginning of buffer
    return buf.read()
