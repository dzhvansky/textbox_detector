"""
Calculate different parameters of geometric figures
"""


import typing

import numpy as np

from textbox_detector.utils.basic_geometry import point_distance, intersect_point


def check_normal_boxes(boxes: np.ndarray) -> None:
    """
    Check boxes coordinates are normal (coordinates should be between -0.5 and 1.5)
    Args:
        box: boxes / single box in normal coordinates (xy bounds from 0 to 1)
    Returns: None,  raises error if coordinates are not normal (xy bounds from 0 to 1)
    """
    if (boxes < - 0.5).any() or (boxes > 1.5).any():
        raise ValueError("Box coordinates should be greater than 0. and less than 1.")
    return None


def get_box_params(
        box: np.ndarray(shape=(4, 2), dtype=float)
) -> typing.Tuple[float, float, float]:
    """
    Calculate width, height and angle in degrees for (rotated) rectangle
    Args:
        box: counterclockwise sorted 4 points coordinates (x, y) for box
    Returns: width, height, angle
    """
    x0, y0 = box[0, 0], box[0, 1]
    x1, y1 = box[1, 0], box[1, 1]
    x2, y2 = box[2, 0], box[2, 1]

    width = point_distance([x0, y0], [x1, y1])
    height = point_distance([x1, y1], [x2, y2])
    angle = np.arctan((y1 - y0) / (x1 - x0 + 1e-10)) / np.pi * 180

    return width, height, angle


def boundary_limits(box: np.ndarray(shape=(4, 2), dtype=float),
                    max_x: float = 1., max_y: float = 1.) -> typing.Dict[str, float]:
    """
    Calculate minimum distance between vertices of the box and borders of the image
    Args:
        box: 4 points normal coordinates (x, y) for box
        max_x: max x value (1. for normalized coords, image width for absolute coords)
        max_y: max y value (1. for normalized coords, image height for absolute coords)
    Returns: dictionary with left, right, top, bottom distance
    """
    angle = get_box_params(box)[2]

    left = np.array([[0., 0.], [0., max_y]])
    right = np.array([[max_x, 0.], [max_x, max_y]])
    top = np.array([[0., 0.], [max_x, 0.]])
    bottom = np.array([[0., max_y], [max_x, max_y]])

    sign = np.array(
        [(box[:, 0] >= 0.).all(), (box[:, 0] <= max_x).all(), (box[:, 1] >= 0.).all(), (box[:, 1] <= max_y).all()]
    ).astype(int)
    sign = sign - (sign == 0).astype(int)
    sign = {k: v for k, v in zip(['left', 'right', 'top', 'bottom'], sign)}

    limits = {}
    if angle >= 0:
        limits['left'] = min(point_distance(box[0, :], intersect_point(box[[0, 1], :], top)),
                             sign['left'] * point_distance(box[3, :], intersect_point(box[[2, 3], :], left)))
        limits['right'] = min(sign['right'] * point_distance(box[1, :], intersect_point(box[[0, 1], :], right)),
                              point_distance(box[2, :], intersect_point(box[[2, 3], :], bottom)))
        limits['top'] = min(sign['top'] * point_distance(box[0, :], intersect_point(box[[0, 3], :], top)),
                            point_distance(box[1, :], intersect_point(box[[1, 2], :], right)))
        limits['bottom'] = min(point_distance(box[3, :], intersect_point(box[[0, 3], :], left)),
                               sign['bottom'] * point_distance(box[2, :], intersect_point(box[[1, 2], :], bottom)))
    else:
        limits['left'] = min(sign['left'] * point_distance(box[0, :], intersect_point(box[[0, 1], :], left)),
                             point_distance(box[3, :], intersect_point(box[[2, 3], :], bottom)))
        limits['right'] = min(point_distance(box[1, :], intersect_point(box[[0, 1], :], top)),
                              sign['right'] * point_distance(box[2, :], intersect_point(box[[2, 3], :], right)))
        limits['top'] = min(point_distance(box[0, :], intersect_point(box[[0, 3], :], left)),
                            sign['top'] * point_distance(box[1, :], intersect_point(box[[1, 2], :], top)))
        limits['bottom'] = min(sign['bottom'] * point_distance(box[3, :], intersect_point(box[[0, 3], :], bottom)),
                               point_distance(box[2, :], intersect_point(box[[1, 2], :], right)))

    return limits
