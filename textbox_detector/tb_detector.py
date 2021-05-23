import logging
import os
import typing

import cv2
import numpy as np

from textbox_detector.data import data_structures
from textbox_detector.utils import quad_ops, figure_param, image_ops, utils

logger = logging.getLogger(os.path.basename(__name__))

try:
    import matplotlib.pyplot as plt
except ImportError:
    logger.warning("Matplotlib is not installed. Debug is not available")


class TextBoxDetector:
    def __init__(
        self,
        area_tags: typing.List[str],
        preproc_params: typing.Dict,
        crop_params: typing.Dict,
        detection_params: typing.Dict,
        merge_params: typing.Dict,
        filter_params: typing.Dict,
        expand_params: typing.Dict,
        boxes_tagging: typing.Dict[str, typing.Optional[typing.Callable]],
        calc_thresh_weight: typing.Dict[str, typing.Optional[typing.Callable]],
        debug: bool = False,
    ):

        self.area_tags = area_tags
        self.preproc_params = preproc_params
        self.crop_params = crop_params
        self.detection_params = detection_params
        self.merge_params = merge_params
        self.filter_params = filter_params
        self.expand_params = expand_params
        self.boxes_tagging = boxes_tagging
        self.calc_thresh_weight = calc_thresh_weight
        self.debug = debug

        if self.debug:
            try:
                _ = plt.figure()
            except NameError:
                logger.warning("Matplotlib is not installed. Debug is not available")
                self.debug = False

    @staticmethod
    def _detect_single_boxes(
        img: np.ndarray,
        n_boxes: int,
        min_height: int,
        max_height: int,
        max_width: float,
        max_area: float,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Find series-number text related contours and boxes
        Args:
            img: binarized image as numpy array
            n_boxes: number of top area contours and corresponding boxes
            min_height: minimal box height
            max_height: maximal box height
            max_width: maximal box width
            max_area: maximal box area
        Returns: filtered contours and corresponding boxes
        """
        h, w = img.shape[:2]
        cnts = image_ops.get_sorted_contours(img)
        # описываем боксы вокруг контуров, фильтруем контуры/боксы по набору параметров
        cnts_filt = []
        boxes_filt = []
        for cnt in cnts[:n_boxes]:
            box = quad_ops.circ_quad(cnt)
            norm_box = np.squeeze(quad_ops.normalize_quads(box, w, h), axis=0)
            width, height, _ = figure_param.get_box_params(norm_box)
            area = width * height
            if (
                min_height <= height <= max_height
                and width < max_width
                and area < max_area
            ):
                cnts_filt.append(cnt)
                boxes_filt.append(norm_box)

        return np.asarray(cnts_filt), np.asarray(boxes_filt)

    @staticmethod
    def _merge_single_boxes(
        norm_boxes: np.ndarray,
        cnts: np.ndarray,
        height_lim: float,
        width_lim: float,
        angle: float = 0.0,
    ) -> np.ndarray:
        """
        Get merged series-number boxes from single boxes and contours
        Args:
            norm_boxes: single boxes in normal coordinates
            cnts: contours from single boxes
            height_lim: height merge limit
            width_lim: width merge limit
            angle: raw box merging angle

        Returns: array of merged series-number boxes (shape = (N_boxes, 4, 1, 2))

        """
        if len(norm_boxes) > 0:
            merge_idx = quad_ops.get_box_groups(
                norm_boxes,
                height_threshold=height_lim,
                width_threshold=width_lim,
                angle=angle,
            )
        else:
            merge_idx = []

        merged_boxes = []
        for group_idx in merge_idx:
            merged_cnt = np.vstack(cnts[group_idx])
            merged_boxes.append(quad_ops.circ_quad(merged_cnt))

        return np.asarray(merged_boxes)

    @staticmethod
    def _filter_boxes(
        boxes: np.ndarray,
        min_width: float,
        max_width: float = 1.0,
        height_delta: float = 1.0,
        angle_delta: float = 90.0,
    ) -> np.ndarray:
        """
        Filter boxes by set of params
        Args:
            boxes: boxes in normal coordinates
            min_width: minimal box width
            max_width: maximal box width
            height_delta: delta for median box height
            angle_delta: delta for median box angle

        Returns: filtered boxes

        """
        filt_boxes = []
        boxes_params = []
        for box in boxes:
            width, height, angle = figure_param.get_box_params(box)
            if min_width <= width <= max_width:
                boxes_params.append((width, height, angle))
                filt_boxes.append(box)
        median_height = np.median([p[1] for p in boxes_params]).item() if boxes_params else None
        median_angle = np.median([p[2] for p in boxes_params]).item() if boxes_params else None

        final_boxes = []
        for fbox, (w, h, a) in zip(filt_boxes, boxes_params):
            if (
                median_height - height_delta <= h <= median_height + height_delta
                and median_angle - angle_delta <= a <= median_angle + angle_delta
            ):
                final_boxes.append(fbox)
        return np.asarray(final_boxes)

    def _draw_image(
        self,
        title: str,
        img: np.ndarray,
        boxes: np.ndarray,
        add_boxes: typing.Optional[np.ndarray] = None,
    ) -> None:
        """
        Draw image and boxes
        Args:
            title: image title
            img: image as a 3-D or 2-D numpy array
            boxes: boxes to draw as a numpy array
            add_boxes: additional boxes to draw

        Returns:

        """
        if img.ndim == 2:
            pic = np.clip(np.stack((img,) * 3, axis=-1), a_min=150, a_max=255)
        else:
            pic = img.copy()

        if add_boxes is not None:
            for b in add_boxes:
                cv2.drawContours(pic, [b], 0, (0, 255, 0), 2)

        for b in boxes:
            cv2.drawContours(pic, [b], 0, (255, 0, 0), 3)

        plt.figure(figsize=(10, 10))
        plt.imshow(pic)
        plt.title(title)
        plt.show()

    def _process_area(
        self, image: np.ndarray, tag: str
    ) -> typing.Tuple[np.ndarray, typing.List[str], typing.List[float]]:
        """
        Process cropped image area with text boxes to find
        Args:
            image: grayscaled and rotated image part
            tag: area tag
            debug: bool - draw results

        Returns: text boxes, tags and scores

        """
        img_width = image.shape[1]
        img_height = image.shape[0]

        preproc_params = self.preproc_params[tag]
        calc_thresh_weight = self.calc_thresh_weight[tag]
        if calc_thresh_weight:
            preproc_params.update({"thresh_weight": calc_thresh_weight(image)})
        # adaptive threshold + morphological translation
        binarized = image_ops.process_gray_image(image, **preproc_params)

        # находим отдельные контуры с заданными параметрами
        detect_params = self.detection_params[tag]
        cnts, single_boxes = self._detect_single_boxes(binarized, **detect_params)
        # объединяем близко расположенные коробки
        merge_params = self.merge_params[tag]
        merged_boxes = self._merge_single_boxes(single_boxes, cnts, **merge_params)
        norm_boxes = quad_ops.normalize_quads(merged_boxes, img_width, img_height)
        # фильтруем объединенные коробки
        filter_params = self.filter_params[tag]
        filtered_boxes = self._filter_boxes(norm_boxes, **filter_params)

        # тэггируем и окончательно фильтруем коробки
        boxes_tagging = self.boxes_tagging[tag]
        if boxes_tagging:
            filtered_boxes, tags, scores = boxes_tagging(filtered_boxes)
        else:
            tags = len(filtered_boxes) * [""]
            scores = len(filtered_boxes) * [0.0]
        # восстанавливаем коробки в координатах исходной части картинки
        boxes = quad_ops.restore_quads(filtered_boxes, img_width, img_height)

        # расширяем коробки
        boxes = quad_ops.expand_boxes(
            boxes,
            self.expand_params["relative"],
            img_width=img_width,
            img_height=img_height,
            relative=True,
        )
        boxes = quad_ops.expand_boxes(
            boxes,
            self.expand_params["absolute"],
            img_width=img_width,
            img_height=img_height,
            relative=False,
        )

        if self.debug:
            self._draw_image(
                tag,
                binarized,
                boxes=quad_ops.restore_quads(filtered_boxes, img_width, img_height),
                add_boxes=quad_ops.restore_quads(single_boxes, img_width, img_height),
            )

        return boxes, tags, scores

    def process_document(self, image: np.ndarray) -> typing.Dict[str, typing.List]:
        # нормируем картинку по размеру
        scale_width = self.preproc_params["scale_width"]
        scale_coeff = scale_width / image.shape[1]
        resized = image_ops.resize_by_width(image, scale_width)
        # переводим в серый цвет и чистим от шума
        gray = image_ops.grayscale(resized)
        gauss_kernel = self.preproc_params["denoise_kernel_size"]
        denoised = image_ops.remove_noise(
            gray, method="gaussian", kernel_size=gauss_kernel
        )

        text_boxes = {}
        for tag in self.area_tags:
            # вырезаем области для поиска коробок
            img_part, restore_params = image_ops.crop_image(
                denoised, **self.crop_params[tag]
            )
            boxes, tags, scores = self._process_area(img_part, tag)
            # восстанавливаем коробки в координатах нормированной картинки
            boxes = quad_ops.circ_quads(quad_ops.restore_boxes(boxes, **restore_params))
            # восстанавливаем коробки в координатах исходной картинки
            boxes = np.round(boxes / scale_coeff).astype(boxes.dtype)

            text_boxes[tag] = [
                data_structures.TextBox(box=utils.box2tuple(b), tag=t, detect_score=s)
                for b, t, s in zip(boxes, tags, scores)
            ]

        if self.debug:
            boxes = [np.asarray(b.box) for tb in text_boxes.values() for b in tb]
            self._draw_image("document", image, boxes=np.asarray(boxes))

        return text_boxes
