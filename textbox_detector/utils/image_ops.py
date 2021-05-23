"""
Image transformations (rescaling, denoising, thresholding, deskewing, rotations)
"""

import typing

import cv2
import numpy as np

from textbox_detector.utils import quad_ops, figure_param


def get_sorted_contours(
    opening: np.ndarray, external_only: bool = True
) -> typing.List[np.ndarray]:
    """
    Find and sort contours in the image
    Args:
        opening: binarized image (height, width)
        external_only: bool - discard inner contours only or leave all contours
    Returns: list of contour coordinates reverse sorted by contour area
    """
    group_mode = cv2.RETR_EXTERNAL if external_only else cv2.RETR_LIST
    cnts, _ = cv2.findContours(255 - opening, group_mode, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(cnts, key=cv2.contourArea, reverse=True)


def rescale(
    image: np.ndarray,
    coeff: typing.Optional[float] = None,
    new_shape: typing.Tuple[int, int] = None,
) -> np.ndarray:
    """
    Rescale image
    Args:
        image: open cv image
        coeff: scale coefficient (the same for width and height)
        new_shape: new shape of the image - (width, height) tuple
    Returns: rescaled image
    """
    src = image.copy()
    h, w = src.shape[:2]

    if coeff and coeff != 1:
        h_new = round(h * coeff)
        w_new = round(w * coeff)
    elif new_shape:
        w_new, h_new = new_shape
    else:
        return src

    if w_new * h_new > w * h:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA

    return cv2.resize(src, (w_new, h_new), interpolation=interpolation)


def remove_noise(image: np.ndarray, method: str, kernel_size: int) -> np.ndarray:
    """
    Removes noise with Blur
    Options:
    - Median Blur (median)
    - Gaussian Blur (gaussian)
    Parameters:
    - image
    - method: 'median' or 'gaussian'
    """
    if method == "median":
        return cv2.medianBlur(image, kernel_size)
    elif method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    else:
        raise Exception('Type of blur should be from ["median", "gaussian"]')


def grayscale(image: np.ndarray) -> np.ndarray:
    """
    Returns image in shades of gray
    Args:
        image: original image
    Returns: gray scaled image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_by_height(image, height: int):
    """
    Resize image by height
    Args:
        image: image as 3d array
        height: new image height

    Returns: resized image

    """
    h, w = image.shape[:2]
    width = round(height / h * w)
    return rescale(image, new_shape=(width, height))


def resize_by_width(image, width: int):
    """
    Resize image by width
    Args:
        image: image as 3d array
        width: new image width

    Returns: resized image

    """
    h, w = image.shape[:2]
    height = round(width / w * h)
    return rescale(image, new_shape=(width, height))


def crop_image(
    gray: np.ndarray,
    part_height: int,
    part_width: int,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    angle: int = 0,
) -> typing.Tuple[np.ndarray, typing.Dict]:
    """
    Crop image for pattern based textbox detection
    Args:
        gray: normalized grayscaled image
        part_height: height of cropped target image
        part_width: width of cropped target image
        x_start: proportion of image width to start cropping
        x_end: proportion of image width to finish cropping
        y_start: proportion of image height to start cropping
        y_end: proportion of image height to finish cropping
        angle: (clockwise) angle of area part. Possible values - 0, 90, 180, 270
    Returns: cropped image and parameters dictionary for original coordinates restoring
    """
    height, width = gray.shape[:2]
    x_start_px, x_end_px = round(width * x_start), round(width * x_end)
    y_start_px, y_end_px = round(height * y_start), round(height * y_end)
    img_part = gray[y_start_px:y_end_px, x_start_px:x_end_px]
    img_part = rescale(img_part, new_shape=(part_width, part_height))

    if angle == 0:
        pass
    elif angle == 90:
        img_part = cv2.rotate(img_part, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        img_part = cv2.rotate(img_part, cv2.ROTATE_180)
    elif angle == 270:
        img_part = cv2.rotate(img_part, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise ValueError("Angle value must be from the list [0, 90, 180, 270]")

    restore_params = {
        "x_restore": (x_end_px - x_start_px) / part_width,
        "y_restore": (y_end_px - y_start_px) / part_height,
        "x_bias": x_start_px,
        "y_bias": y_start_px,
        "part_width": part_width,
        "part_height": part_height,
        "angle": angle,
    }

    return img_part, restore_params


def process_gray_image(
    image: np.ndarray,
    block_size: int,
    thresh_weight: int,
    struct_kernel_size_x: int,
    struct_kernel_size_y: int,
    **kwargs
) -> np.ndarray:
    """
    Gray image preprocessing for pattern based textbox detection
    Args:
        image: grayscaled image as a numpy array
        block_size: block size for cv2 adaptive threshold
        thresh_weight: weight for cv2 adaptive threshold
        struct_kernel_size: square kernel size for opening morphological transformation
        **kwargs
    Returns:
    """
    # предобработка изображения: adaptive threshold + morphology
    thresh = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        thresh_weight,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (struct_kernel_size_x, struct_kernel_size_y)
    )
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)


def crop_rotated_rect(
    image: np.ndarray, rect: np.ndarray(shape=(4, 2), dtype=np.uint8)
) -> np.ndarray:
    """

    Args:
        img: original image as a np.array
        rect: rectangle to crop of the shape (4,2), x and y coordinates along 0-axis

    Returns: cropped rectangle area

    """
    src = image.copy()
    height, width = src.shape[0], src.shape[1]

    w, h, angle = figure_param.get_box_params(rect)
    center = rect[:, 0].mean(), rect[:, 1].mean()
    size = round(w), round(h)

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rotated = cv2.warpAffine(src, M, (width, height))

    return cv2.getRectSubPix(img_rotated, size, center)


def get_filled_areas(
    image: np.ndarray,
    indent_factor: typing.Optional[int] = 50,
    upscale_size: int = 2048,
    scale_coeff: float = 0.1,
    canny_min: int = 40,
    canny_max: int = 200,
    max_n_boxes: int = 5,
    struct_kernel_size_x: int = 8,
    struct_kernel_size_y: int = 8,
    min_area: float = 0.05,
    page_angle: typing.Optional[float] = None,
) -> typing.List[np.ndarray]:
    """
    Get filled page areas as list of separate images boxes to crop
    Args:
        image: image as a np.ndarray
        indent_factor: factor to compute indent size (image size portion) - Optional
        upscale_size: size for image scaling (width-height)
        scale_coeff: scale coefficient for computation speed (less then 1. for downscaling)
        canny_min: Canny threshold minimum value
        canny_max: Canny threshold maximum value
        max_n_boxes: max number of filled areas in the image
        struct_kernel_size_x: size of structure_kernel (x dimensions)
        struct_kernel_size_y: size of structure_kernel (y dimensions)
        min_area: minimal area to find
        page_angle: optional - precomputed angle of filled area(-s) to crop

    Returns: list of coordinates of filled image areas

    """
    src = image.copy()
    height, width = src.shape[:2]

    # приводим картинку к стандартному размеру (увеличиваем размер, чтобы корректно детектировать контуры)
    standard_coeff = ((upscale_size ** 2) / src.shape[0] / src.shape[1]) ** 0.5
    scaled = rescale(image, coeff=standard_coeff)

    # обрезаем края, чтобы убрать внешний контур
    if indent_factor:
        indent = max(1, int((height + width) / 2 // indent_factor))
        scaled = scaled[indent:-indent, indent:-indent]
    else:
        indent = 0

    # предобработка изображения для поиска больших контуров с наполнением (текстом) внутри
    gray = grayscale(rescale(scaled, coeff=scale_coeff))
    edged = cv2.Canny(gray, canny_min, canny_max)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (struct_kernel_size_x, struct_kernel_size_y)
    )
    processed_img = cv2.morphologyEx(255 - edged, cv2.MORPH_OPEN, kernel, iterations=1)

    # поиск достаточно больших контуров
    cnts = get_sorted_contours(processed_img)[:max_n_boxes]
    boxes = (
        quad_ops.circ_quads(np.asarray(cnts) / scale_coeff, angle=page_angle) + indent
    ) / standard_coeff
    box_params = [figure_param.get_box_params(b) for b in boxes]
    box_areas = np.asarray([p[0] / width * p[1] / height for p in box_params])
    verified_idx = np.where(box_areas > min_area)[0]
    boxes = np.asarray(boxes)[verified_idx]

    return boxes
