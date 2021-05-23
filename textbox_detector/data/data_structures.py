"""
Module for different data structures
"""

import typing
from pydantic import BaseModel

CornerCoordinates = typing.Tuple[
    typing.Tuple[int, int],
    typing.Tuple[int, int],
    typing.Tuple[int, int],
    typing.Tuple[int, int],
]

Rect = typing.Tuple[int, int, int, int]

DEFAULT_CORNER_COORD = ((0, 0), (0, 0), (0, 0), (0, 0))


class TextBox(BaseModel):
    box: CornerCoordinates = DEFAULT_CORNER_COORD
    tag: str = ""
    detect_score: float = 0.0
    text: str = ""
    ocr_score: float = 0.0

    class Config:
        arbitrary_types_allowed = True
