import typing

import numpy as np


def box2tuple(box: np.ndarray) -> typing.Tuple[typing.Tuple[int, int], ...]:
    """
    Transform 2-D numpy array to tuple
    Args:
        box: box vertices coordinates as a numpy array of shape (4, 2) or (4, 1, 2)

    Returns: tuple of tuple pairs of xy coordinates

    """
    return tuple(map(tuple, box.reshape(-1, 2).tolist()))
