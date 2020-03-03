def get_dtype(max_value):
    """
    Calculate the appropriate dtype that will contain the max value

    Parameters
    ----------
    max_value : number
        max encoded value

    Returns
    -------
    numpy dtype string
    """

    if max_value <= 255:
        return "uint8"
    if max_value <= 65535:
        return "uint16"
    if max_value <= 4294967295:
        return "uint32"

    raise Exception("value is too large for uint32 / rgba")


def get_nodata_value(max_value):
    """
    Calculate the appropriate nodata value based on max of value range for the
    output image type

    Parameters
    ----------
    max_value : number
        max encoded value

    Returns
    -------
    int, nodata value
    """

    if max_value < 255:  # uint8 / grayscale
        return 255
    if max_value < 65535:  # uint16 / RGB
        return 65535
    if max_value < 16777215:  # uint32 / RGB
        return 16777215
    if max_value < 4294967295:  # uint32 / RGBA
        return 4294967295

    raise Exception("value is too large for uint32 / rgba")
