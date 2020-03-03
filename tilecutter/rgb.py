"""RGB and RGBA processing functions"""


def hex_to_rgb(color):
    """Convert a hex color code to an 8 bit rgb tuple.

    Parameters
    ----------
    color : string, must be in #112233 syntax

    Returns
    -------
    tuple : (red, green, blue) as 8 bit numbers

    """

    if not len(color) == 7 and color[0] == "#":
        raise ValueError("Color must be in #112233 format")

    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def to_rgb_array(arr):
    """
    Convert 2D integer values to RGB triples .

    Parameters
    ----------
    arr: array of input values to split into rgb, will be cast to uint32

    Returns
    -------
    array of RGB triples: [[[R, G, B] ...]]    (uint8)
    """

    # from: http://stackoverflow.com/questions/19432423/convert-array-of-single-integer-pixels-to-rgb-triplets-in-python
    bgr = arr.astype("uint32").view("uint8").reshape(arr.shape + (4,))[..., :3]
    # flip innermost array to get rgb triples
    return bgr[..., ::-1].copy()


# Note: this currently cannot be decoded properly, apparently because browsers can mess with gamma and alpha:
# https://stackoverflow.com/questions/27767914/why-is-a-canvas-drawn-with-an-image-a-slightly-different-color-than-the-image-it
def to_rgba_array(arr):
    """
    Convert 2D integer values to RGBA tuples .

    Parameters
    ----------
    arr: array of input values to split into rgba, will be cast to uint32

    Returns
    -------
    array of RGBA tuples: [[[R, G, B, A] ...]]    (uint8)
    """

    # from: http://stackoverflow.com/questions/19432423/convert-array-of-single-integer-pixels-to-rgb-triplets-in-python
    abgr = arr.astype("uint32").view("uint8").reshape(arr.shape + (4,))
    # flip innermost array to get rgba
    return abgr[..., ::-1].copy()
