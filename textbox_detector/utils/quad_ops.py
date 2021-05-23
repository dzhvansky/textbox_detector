"""
Different operations with quadrilaterals
"""

import typing

import numpy as np
import shapely.geometry as geometry

from textbox_detector.utils.basic_geometry import point_distance, line2polygon, rotate_polygon
from textbox_detector.utils.figure_param import check_normal_boxes, boundary_limits, get_box_params


def normalize_quads(quads: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Transform quadrilaterals coordinates from image sized to interval [0:1]
    Args:
        quads: quadrilaterals of possible shape (N_quads, 4, 2) or (N_quads, 4, 1, 2)
        width:  image width
        height: image height
    Returns:
    """
    norm_quads = quads.astype(float) / np.array([width, height]) if len(quads) > 0 else quads.copy()
    return norm_quads.reshape(-1, 4, 2)


def restore_quads(quads: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Transform quadrilaterals coordinates from interval [0:1] to image sized
    Args:
        quads: quadrilaterals of possible shape (4,2) or (4,1,2)
        width:  image width
        height: image height
    Returns:
    """
    check_normal_boxes(quads)
    restored_quads = np.rint(quads * np.array([width, height])) if len(quads) > 0 else quads.copy()
    return restored_quads.reshape(-1, 4, 1, 2).astype(np.int32)


def sort_norm_quads(quads: np.ndarray,
                    mode: str = 'vertical',
                    angle: typing.Optional[float] = None
                    ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Sort normalized quadrilaterals
    Args:
        quads: numpy array of quadrilaterals of shape (4,2)
        mode: horizontal, vertical, skewed
        angle: rotation angle in degrees for "skewed" mode
    Returns: index and sort parameter values
    """
    check_normal_boxes(quads)

    if mode == "vertical":
        sort_param = quads[:, :, 1].mean(axis=1).reshape(-1)
    elif mode == "horizontal":
        sort_param = quads[:, :, 0].mean(axis=1).reshape(-1)
    elif mode == "skewed" and angle:
        sort_param = (quads[:, :, 1].mean(axis=1)
                      - np.tan(angle / 180 * np.pi) * quads[:, :, 0].mean(axis=1)).reshape(-1)
    elif mode == "skewed" and not angle:
        return sort_norm_quads(quads, 'vertical')
    else:
        raise Exception("Choose mode from [mean, horizontal, vertical, skewed]")
    index = np.argsort(sort_param)
    return index, sort_param[index]


def circ_quad(point_set: np.ndarray, angle: typing.Optional[float] = None) -> np.ndarray:
    """
    Describes a rectangle around points
    Args:
        point_set: np.array([[x1, y1], [x2, y2], ...]) or np.array of shape (N_points, 1, 2)
        angle: optional, fixed rotation angle for rectangle (in degrees, between 90 and -90)
    Returns: the coordinates of the vertices of the rectangle, starting from the top left counterclockwise
    """
    original_dtype = point_set.dtype
    points = point_set.copy().reshape(-1, 2)

    if points.shape[0] < 3 or \
            isinstance(geometry.Polygon(points).minimum_rotated_rectangle, geometry.linestring.LineString):
        points = line2polygon(points)

    if not angle:
        quad = np.asarray(geometry.Polygon(points).minimum_rotated_rectangle.exterior.coords.xy).T[:4, :]
        # get top left vertices index
        top_left_point = np.array([np.min(quad[:, 0]), np.min(quad[:, 1])])
        top_left_idx = np.argsort([point_distance(top_left_point, point) for point in quad])[0]
        # set correct vertices order
        box = np.vstack([quad[top_left_idx:, :], quad[:top_left_idx, :]])

    else:
        # rotate points to the given angle
        center = points[:, 0].mean(), points[:, 1].mean()
        aligned = rotate_polygon(points, angle=-angle, center=center)
        # describe the rectangle
        min_x, max_x = aligned[:, 0].min(), aligned[:, 0].max()
        min_y, max_y = aligned[:, 1].min(), aligned[:, 1].max()
        aligned_rect = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
        # rotate the rectangle backwards
        box = rotate_polygon(aligned_rect, angle=angle, center=center)

    if np.issubdtype(original_dtype, np.integer):
        box = np.round(box)

    return box.astype(original_dtype)


def circ_quads(point_sets: np.ndarray, angle: typing.Optional[float] = None) -> np.ndarray:
    """
    Describes a rectangles around sets of points
    Args:
        points: np.array([[x1, y1], [x2, y2], ...])
        angle: optional, fixed rotation angle for rectangle (in degrees, between 90 and -90)
    Returns: the coordinates of the vertices of the rectangle, starting from the top left counterclockwise
    """
    return np.asarray([circ_quad(point_set, angle) for point_set in point_sets])


def get_box_groups(quadrilaterals: np.ndarray,
                   height_threshold: float = 0.025,
                   width_threshold: float = 0.075,
                   angle: typing.Optional[float] = None
                   ) -> typing.List[typing.List[int]]:
    """
    Group boxes by vertical levels
    Args:
        quadrilaterals: normalized quadrilaterals
        height_threshold: min height delta between box vertical levels
        width_threshold: min width delta between box vertical levels
        angle: rotation angle in degrees for "skewed" mode
    Returns: list of box group index lists
    """
    check_normal_boxes(quadrilaterals)
    n_boxes = quadrilaterals.shape[0]
    ver_idx, ver_levels = sort_norm_quads(quadrilaterals, mode='skewed', angle=angle)

    # группируем боксы по вертикальным уровням
    ver_groups = list()
    ver_group_index = [0]
    for i in range(1, n_boxes):
        if ver_levels[i] <= ver_levels[i - 1] + height_threshold:
            ver_group_index.append(i)
        else:
            ver_groups.append(ver_group_index)
            ver_group_index = [i]
    ver_groups.append(ver_group_index)

    groups = list()
    # группируем боксы по горизонтальным уровням
    for ver_group_idx in ver_groups:
        ver_group_idx = ver_idx[ver_group_idx]
        group_size = len(ver_group_idx)
        if group_size == 1:
            groups.append(ver_group_idx.tolist())
        else:
            ver_group = quadrilaterals[ver_group_idx]
            hor_idx, _ = sort_norm_quads(ver_group, mode='horizontal')
            left_borders = np.mean(ver_group[hor_idx][:, [0, 3], 0], axis=1).squeeze()
            right_borders = np.max(ver_group[hor_idx][:, [1, 2], 0], axis=1).squeeze()
            right_border = right_borders[0]
            hor_group_idx = [0]
            for j in range(1, group_size):
                if left_borders[j] <= right_border + width_threshold:
                    hor_group_idx.append(j)
                else:
                    groups.append(ver_group_idx[hor_idx[hor_group_idx]].tolist())
                    hor_group_idx = [j]
                right_border = right_borders[j]
            groups.append(ver_group_idx[hor_idx[hor_group_idx]].tolist())

    return groups


def expand_box(box: np.ndarray(shape=(4, 2)),
               left: float = 0., right: float = 0., top: float = 0., bottom: float = 0.,
               img_width: float = 1., img_height: float = 1.,
               relative: bool = True, border_limits: bool = True) -> np.ndarray(shape=(4, 2)):
    """
    Extend the bounds of a quad in normal/absolute coordinates
    Args:
        box: counterclockwise sorted coordinates of the corners of the quad
        left: width fraction of box (relative mode) or image (absolute mode) to expand left
        right: width fraction of box (relative mode) or image (absolute mode) to expand right
        top: width fraction of box (relative mode) or image (absolute mode) to expand top
        bottom: width fraction of box (relative mode) or image (absolute mode) to expand bottom
        img_width: image width (default -- normal coords = 1.)
        img_height: image height (default -- normal coords = 1.)
        relative: bool type of coordinates to expand, True - relative to box size, False - normal absolute
        border_limits: bool for box restriction to the image borders
​
    Returns: the coordinates of the corners of the extended quad
    Examples:
        >>> expand_box(box, **{'left': 0.01, 'right': 0.01, 'top': 0.005, 'bottom': 0.005}, relative=False)
    """
    new_box = np.copy(box).astype(float).reshape(-1, 2)

    box_width, box_height, angle = get_box_params(new_box)
    radian_angle = angle / 180 * np.pi
    sin, cos = np.sin(radian_angle), np.cos(radian_angle)

    if not relative:
        box_width, box_height = img_width, img_height

    # сначала расширяем по вертикали (т.к. расширение меньше), затем расширяем по горизонтали
    if top != 0. or bottom != 0.:
        expand_top = top * box_height
        expand_bottom = bottom * box_height
        top_bottom_coeff = np.array([-sin, cos])
        if border_limits:
            limits = boundary_limits(new_box, max_x=img_width, max_y=img_height)
            expand_top = min(expand_top, limits['top'])
            expand_bottom = min(expand_bottom, limits['bottom'])
        new_box[[0, 1], :] = new_box[[0, 1], :] - expand_top * top_bottom_coeff
        new_box[[2, 3], :] = new_box[[2, 3], :] + expand_bottom * top_bottom_coeff

    if left != 0. or right != 0.:
        expand_left = left * box_width
        expand_right = right * box_width
        left_right_coeff = np.array([cos, sin])
        if border_limits:
            limits = boundary_limits(new_box, max_x=img_width, max_y=img_height)
            expand_left = min(expand_left, limits['left'])
            expand_right = min(expand_right, limits['right'])
        new_box[[0, 3], :] = new_box[[0, 3], :] - expand_left * left_right_coeff
        new_box[[1, 2], :] = new_box[[1, 2], :] + expand_right * left_right_coeff

    if np.issubdtype(box.dtype, np.integer):
        new_box = np.round(new_box)

    return new_box.astype(box.dtype).reshape(box.shape)


def expand_boxes(boxes: np.ndarray, expand: typing.Dict[str, float],
                 img_width: float = 1., img_height: float = 1.,
                 relative: bool = True, border_limits: bool = True) -> np.ndarray:
    """
    Extend the bounds of a quads in normal coordinates (xy bounds from 0 to 1)
    Args:
        boxes: numpy array of counterclockwise sorted coordinates of the corners of the quads
        expand: a dictionary of 4-sided extension ('left', 'right', 'top', 'bottom') in normal coordinates
        img_width: image width (default -- normal coords = 1.)
        img_height: image height (default -- normal coords = 1.)
        relative: bool type of coordinates to expand, True - relative to box size, False - normal absolute
        border_limits: bool for box restriction to the image borders
​
    Returns: the coordinates of the corners of the extended quads
    Examples:
        >>> expand_boxes(boxes, {'left': 0.01, 'right': 0.01, 'top': 0.005, 'bottom': 0.005}, relative=False)
    """
    return np.asarray([expand_box(box, **expand, img_width=img_width, img_height=img_height,
                                  relative=relative, border_limits=border_limits) for box in boxes])


def revert_boxes(boxes: np.ndarray, width: int, height: int, angle: int) -> np.ndarray:
    """
    Revert boxes coordinates for rotated image part
    Args:
        boxes: text boxes in original image coordinates of shape (N, 4, 1, 2)
        width: width of image part
        height: height of image part
        angle: (clockwise) angle of original image part

    Returns: restored (rotated) boxes coordinates for image part

    """
    if angle in [90, 270]:
        reverted_boxes = boxes[..., ::-1]
    else:
        reverted_boxes = boxes.copy()

    if angle == 90:
        reverted_boxes[..., 0] = width - reverted_boxes[..., 0]
    elif angle == 180:
        reverted_boxes[..., 0] = width - reverted_boxes[..., 0]
        reverted_boxes[..., 1] = height - reverted_boxes[..., 1]
    elif angle == 270:
        reverted_boxes[..., 1] = height - reverted_boxes[..., 1]
    else:
        raise ValueError("Angle value must be from the list [90, 180, 270]")

    return reverted_boxes


def restore_boxes(boxes: np.ndarray, x_restore: float, y_restore: float, x_bias: int, y_bias: int,
                  part_width: int, part_height: int, angle: int) -> np.ndarray:
    """
    Restore pattern based detected boxes in original image coordinates
    Args:
        boxes: text boxes in cropped image coordinates of shape (N, 4, 1, 2)
        x_restore: coefficient for x-axis image restore
        y_restore: coefficient for y-axis image restore
        x_bias: x-axis bias for cropped image
        y_bias: y-axis bias for cropped image
        part_width: width of image part
        part_height: height of image part
        angle: (clockwise) angle of original image part
    Returns: restored boxes in original image coordinates
    """
    orig_dtype = boxes.dtype

    if len(boxes) > 0:
        # меняем координаты коробок в случае, если угол поворота отличен от нуля
        if angle != 0:
            boxes = revert_boxes(boxes, part_width, part_height, angle)
        # восстанавливаем координаты коробок в координатах исходной картинки
        boxes = boxes * np.array([x_restore, y_restore]) + np.array([x_bias, y_bias])
        # в нормальных кооординатах исходной картинки
    return np.round(boxes).astype(orig_dtype)
