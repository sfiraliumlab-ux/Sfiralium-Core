"""
Математическое ядро Сфирали: алгебры H2, H3 и геометрические операции.
"""

import torch
import torch.nn as nn
import math


class H2Algebra:
    """
    Алгебра двойной переменной H2 (z = x + jy, j² = 1).
    Основа для вычисления бинглов и моделирования S-петли.
    """
    
    @staticmethod
    def multiply(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Умножение двойных чисел: (x1 + j·y1)(x2 + j·y2) = (x1·x2 + y1·y2) + j·(x1·y2 + y1·x2)
        z1, z2: тензоры формы (..., 2), где [..., 0] = x, [..., 1] = y
        """
        x1, y1 = z1[..., 0], z1[..., 1]
        x2, y2 = z2[..., 0], z2[..., 1]
        real = x1 * x2 + y1 * y2
        imag = x1 * y2 + y1 * x2
        return torch.stack([real, imag], dim=-1)

    @staticmethod
    def conjugate(z: torch.Tensor) -> torch.Tensor:
        """Зеркальное сопряжение: z* = x - jy"""
        return torch.stack([z[..., 0], -z[..., 1]], dim=-1)

    @staticmethod
    def norm_squared(z: torch.Tensor) -> torch.Tensor:
        """Инвариант Сфирали: |z|² = x² - y²"""
        return z[..., 0] ** 2 - z[..., 1] ** 2

    @staticmethod
    def s_loop(z: torch.Tensor, psi: torch.Tensor = None) -> torch.Tensor:
        """
        S-петля: плавный фазовый переход через световой конус.
        psi: фазовый параметр (по умолчанию вычисляется автоматически)
        """
        if psi is None:
            norm_sq = H2Algebra.norm_squared(z)
            psi = torch.tanh(norm_sq)
        
        # Модуляция амплитуды через S-петлю
        scale = torch.sigmoid(psi)
        return z * scale.unsqueeze(-1)


class H3Algebra:
    """
    Алгебра тройных чисел H3 = R ⊕ R ⊕ R.
    Кубическая метрика, бинглы и тринглы.
    """
    
    @staticmethod
    def cubic_metric(v: torch.Tensor) -> torch.Tensor:
        """
        Кубическая метрика: M(v) = |x1·x2·x3|^(1/3)
        v: тензор формы (..., 3)
        """
        return torch.abs(v[..., 0] * v[..., 1] * v[..., 2]) ** (1 / 3)

    @staticmethod
    def bingle(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Бингл — обобщённый угол между двумя векторами в H3.
        Аналог скалярного произведения для кубической метрики.
        """
        # Нормировка по кубической метрике
        m1 = H3Algebra.cubic_metric(v1)
        m2 = H3Algebra.cubic_metric(v2)
        
        # Бингл через кубическую форму
        cubic_form = torch.abs(
            v1[..., 0] * v2[..., 1] * v2[..., 2] +
            v1[..., 1] * v2[..., 2] * v2[..., 0] +
            v1[..., 2] * v2[..., 0] * v2[..., 1]
        ) ** (1 / 3)
        
        return torch.acos(torch.clamp(cubic_form / (m1 * m2 + 1e-8), -1, 1))

    @staticmethod
    def tringle(v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor) -> torch.Tensor:
        """
        Трингл — трёхгранный фазовый инвариант.
        Площадь геодезического треугольника на индикатрисе.
        """
        # Нормировка векторов
        m1 = H3Algebra.cubic_metric(v1).unsqueeze(-1)
        m2 = H3Algebra.cubic_metric(v2).unsqueeze(-1)
        m3 = H3Algebra.cubic_metric(v3).unsqueeze(-1)
        
        v1_n = v1 / (m1 + 1e-8)
        v2_n = v2 / (m2 + 1e-8)
        v3_n = v3 / (m3 + 1e-8)
        
        # Трингл через смешанное кубическое произведение
        mixed = torch.abs(
            v1_n[..., 0] * v2_n[..., 1] * v3_n[..., 2] +
            v1_n[..., 1] * v2_n[..., 2] * v3_n[..., 0] +
            v1_n[..., 2] * v2_n[..., 0] * v3_n[..., 1]
        )
        
        return torch.acos(torch.clamp(mixed, 0, 1))


class SfiralGeometry(nn.Module):
    """
    Геометрический модуль Сфирали: параметризация витков и S-петли.
    """
    
    def __init__(self, radius: float = 1.0, height: float = 1.0, kz: float = 1.0):
        super().__init__()
        self.radius = nn.Parameter(torch.tensor(radius))
        self.height = nn.Parameter(torch.tensor(height))
        self.kz = nn.Parameter(torch.tensor(kz))
    
    def coil_minus(self, theta: torch.Tensor) -> torch.Tensor:
        """Первый виток V⁻"""
        x = self.radius * torch.cos(theta)
        y = self.radius * torch.sin(theta)
        z = self.kz * theta
        return torch.stack([x, y, z], dim=-1)
    
    def coil_plus(self, theta: torch.Tensor) -> torch.Tensor:
        """Второй виток V⁺ (зеркально-антисимметричный)"""
        x = -self.radius * torch.cos(theta)
        y = -self.radius * torch.sin(theta)
        z = self.height - self.kz * theta
        return torch.stack([x, y, z], dim=-1)
    
    def s_loop(self, theta: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        S-петля: интерполяция между витками.
        alpha: весовая функция α(s), где α² + β² = 1
        """
        beta = torch.sqrt(torch.clamp(1 - alpha ** 2, min=0))
        v_minus = self.coil_minus(theta)
        v_plus = self.coil_plus(theta)
        return alpha.unsqueeze(-1) * v_minus + beta.unsqueeze(-1) * v_plus
    
    def forward(self, theta: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Полная траектория Сфирали"""
        return self.s_loop(theta, alpha)
