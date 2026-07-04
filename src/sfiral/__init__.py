"""
Sfiralium-Core: Topological Computing Framework
Реализация принципа Сфирали: зеркальная антисимметрия, S-петля и фрактальная вложенность.

Основано на работах О.С. Басаргина.
"""

from .neurons import SINLayer, FSINLayer
from .geometry import H2Algebra, H3Algebra, SfiralGeometry

__version__ = "0.1.0"
__author__ = "O.S. Basargin, S. Chernenko"
__license__ = "MIT (with author's terms)"

__all__ = [
    "SINLayer",
    "FSINLayer",
    "H2Algebra",
    "H3Algebra",
    "SfiralGeometry",
]
