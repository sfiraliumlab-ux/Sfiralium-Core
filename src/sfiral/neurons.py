"""
Сфиральные Искусственные Нейроны (СИН и ФСИН).
Архитектура на основе зеркальной антисимметрии, S-петли и фрактальной вложенности.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SINLayer(nn.Module):
    """
    Сфиральный Искусственный Нейрон (СИН).
    
    Архитектура:
    - V⁻ ветвь: прямое преобразование
    - V⁺ ветвь: зеркально-антисимметричное преобразование
    - S-петля: фазовая интеграция через параметризуемый гейт
    
    Преимущества:
    - Топологическая защита от шумов
    - Балансировка противоположных сигналов
    - Минимизация потерь информации
    """
    
    def __init__(self, in_features: int, out_features: int, s_gate_dim: int = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Ветвь V⁻ (прошлое / актуальное состояние)
        self.branch_minus = nn.Linear(in_features, out_features)
        
        # Ветвь V⁺ (будущее / потенциальное состояние) — антисимметрия
        self.branch_plus = nn.Linear(in_features, out_features)
        
        # S-петля: параметризуемый гейт фазового перехода
        gate_in = s_gate_dim if s_gate_dim else in_features
        self.s_gate = nn.Sequential(
            nn.Linear(gate_in, out_features),
            nn.Sigmoid()  # α ∈ [0, 1], β = √(1 - α²)
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов с учётом зеркальной антисимметрии"""
        nn.init.xavier_uniform_(self.branch_minus.weight)
        # V⁺ инициализируется как зеркальное отражение V⁻
        with torch.no_grad():
            self.branch_plus.weight.copy_(-self.branch_minus.weight.clone())
            self.branch_plus.bias.copy_(-self.branch_minus.bias.clone())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через СИН.
        
        Args:
            x: входной тензор формы (batch, in_features)
        
        Returns:
            выходной тензор формы (batch, out_features)
        """
        # 1. Обработка в антисимметричных ветвях
        v_minus = self.branch_minus(x)
        v_plus = self.branch_plus(x)
        
        # 2. S-петля: вычисление фазового веса
        alpha = self.s_gate(x)
        beta = torch.sqrt(torch.clamp(1.0 - alpha ** 2, min=1e-7))
        
        # 3. Интеграция через S-петлю
        # y = α·V⁻ + β·V⁺
        output = alpha * v_minus + beta * v_plus
        
        return output
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


class FSINLayer(nn.Module):
    """
    Фрактальный Сфиральный Искусственный Нейрон (ФСИН).
    
    Рекурсивная вложенность базовых СИН для многомасштабного анализа.
    
    Args:
        in_features: размерность входа
        out_features: размерность выхода
        fractal_depth: глубина фрактала (количество уровней вложенности)
        scale_factor: коэффициент масштабирования между уровнями
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 fractal_depth: int = 3, scale_factor: float = 0.7):
        super().__init__()
        self.fractal_depth = fractal_depth
        self.scale_factor = scale_factor
        
        # Создаём каскад СИН-слоёв
        self.layers = nn.ModuleList()
        current_dim = in_features
        
        for i in range(fractal_depth):
            # На последнем уровне — выходная размерность
            next_dim = out_features if i == fractal_depth - 1 else current_dim
            self.layers.append(SINLayer(current_dim, next_dim))
            current_dim = next_dim
        
        # Фрактальные весовые коэффициенты
        self.fractal_weights = nn.Parameter(
            torch.tensor([scale_factor ** i for i in range(fractal_depth)])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через фрактальную структуру.
        Каждый уровень анализирует данные с разной степенью детализации.
        """
        h = x
        outputs = []
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            outputs.append(h * self.fractal_weights[i])
        
        # Интеграция всех уровней
        return sum(outputs)
    
    def extra_repr(self) -> str:
        return (f'fractal_depth={self.fractal_depth}, '
                f'scale_factor={self.scale_factor}')


class SfiralActivation(nn.Module):
    """
    Функция активации на основе S-петли.
    Использует сигмоиду от нормы двойного числа для плавного фазового перехода.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # S-образная активация через гиперболический тангенс
        return torch.tanh(x / self.temperature)


class TopologicalLoss(nn.Module):
    """
    Топологический регуляризатор: штрафует сеть за нарушение зеркальной антисимметрии.
    """
    
    def __init__(self, lambda_reg: float = 0.01):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(self, layer: SINLayer) -> torch.Tensor:
        """
        Вычисляет топологический лосс для СИН-слоя.
        Минимизирует разницу между весами V⁻ и -V⁺.
        """
        weight_diff = layer.branch_minus.weight + layer.branch_plus.weight
        bias_diff = layer.branch_minus.bias + layer.branch_plus.bias
        
        loss = (torch.mean(weight_diff ** 2) + torch.mean(bias_diff ** 2))
        return self.lambda_reg * loss
