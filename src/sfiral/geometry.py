import torch

class H2Algebra:
    """
    Алгебра двойной переменной H2 (z = x + jy, j^2 = 1).
    Основа для вычисления бинглов и моделирования S-петли.
    """
    @staticmethod
    def multiply(z1, z2):
        """Умножение двойных чисел: (x1 + j*y1)(x2 + j*y2)"""
        x1, y1 = z1[..., 0], z1[..., 1]
        x2, y2 = z2[..., 0], z2[..., 1]
        real = x1 * x2 + y1 * y2
        imag = x1 * y2 + y1 * x2
        return torch.stack([real, imag], dim=-1)

    @staticmethod
    def norm_squared(z):
        """Квадрат модуля (инвариант Сфирали): x^2 - y^2"""
        return z[..., 0]**2 - z[..., 1]**2
