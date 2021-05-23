"""
Module for basic geometric functions (points, polygons, rectangles, quadrilaterals)
"""

import typing

import cv2
import numpy as np
import shapely.geometry as geometry


def point_distance(
    point_a: np.ndarray(shape=(2,), dtype=float),
    point_b: np.ndarray(shape=(2,), dtype=float),
) -> float:
    """
    Euclidean distance between two points
    Args:
        point_a: np.array([x0,y0]) for point "a"
        point_b: np.array([x1,y1]) for point "b"
    Returns: distance between points
    """
    return geometry.Point(point_a).distance(geometry.Point(point_b))


def intersect_point(
    line_a: np.ndarray(shape=(2, 2), dtype=float),
    line_b: np.ndarray(shape=(2, 2), dtype=float),
    continuation: bool = True,
    eps: float = 1e-6,
) -> np.ndarray(shape=(2,), dtype=float):
    """
    Get intersection point of two line segments
    Args:
        line_a: np.array([[x0,y0], [x1,y1]]) for line "a"
        line_b: np.array([[x0,y0], [x1,y1]]) for line "b"
        continuation: get line continuations intersection (True/False)
        eps: epsilon value
    Returns:line intersection point [x, y]
    """

    if continuation:
        a, b = line_a[1] - line_a[0], line_b[1] - line_b[0]
        if abs(a[0] * b[1] - a[1] * b[0]) < eps:
            intersect = np.array([np.nan, np.nan])
        else:
            t, _ = np.linalg.solve(np.vstack([a, b]).T, line_b[0] - line_a[0])
            intersect = (1 - t) * line_a[0] + t * line_a[1]
    else:
        point = geometry.LineString(line_a.tolist()).intersection(
            geometry.LineString(line_b.tolist())
        )
        intersect = (
            np.array([point.x, point.y])
            if not isinstance(point, geometry.LineString)
            else np.array([np.nan, np.nan])
        )

    return intersect


def line2polygon(points: np.ndarray, eps: float = 1e-8):
    """
    Transforms a segment into a polygon of minimal area
    (to add bias to the points (x0, y0))
    Args:
        points: np.array([[x1, y1], [x2, y2], ...])
        eps: bias for the first point
    Returns: 2d array from original and added points
    """
    x0, y0 = points[0, 0], points[0, 1]
    add_points = np.array([[x0, y0 + eps], [x0 + eps, y0]])
    return np.vstack([points, add_points])


def rotate_polygon(
    polygon: np.ndarray, angle: float, center: typing.Tuple[float, float]
) -> np.ndarray:
    """
        Rotate polygon/box xy coordinates around the given center
        Args:
            polygon: xy coordinates of shape (4, 2) as a numpy array
            angle: angle to rotate (counterclockwise)
            center: center xy coordinates
    ​
        Returns: rotated box/polygon coordinates
    ​
    """
    moments = cv2.getRotationMatrix2D(center, -angle, 1.0)
    # Prepare the vectors to be transformed
    vectors = np.hstack([polygon.reshape(-1, 2), np.ones((polygon.shape[0], 1))]).T
    # Perform the actual rotation and return the coordinates
    return np.rint(moments @ vectors).T.astype(polygon.dtype).reshape(polygon.shape)
