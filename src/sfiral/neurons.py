import torch
import torch.nn as nn
import torch.nn.functional as F

class SINLayer(nn.Module):
    """
    Сфиральный Искусственный Нейрон (СИН).
    Архитектура основана на двух зеркально-антисимметричных ветвях (V-, V+) 
    и S-образном модуле интеграции (фазовый переход).
    """
    def __init__(self, in_features, out_features, s_gate_dim=None):
        super(SINLayer, self).__init__()
        # Ветвь V- (Прошлая фаза / Актуальное состояние)
        self.branch_neg = nn.Linear(in_features, out_features)
        # Ветвь V+ (Грядущая фаза / Потенциальное состояние) - Антисимметрия
        self.branch_pos = nn.Linear(in_features, out_features)
        
        # S-петля: параметризуемый гейт фазового перехода
        # Определяет баланс между ветвями в зависимости от входного контекста
        gate_in = s_gate_dim if s_gate_dim else in_features
        self.s_gate = nn.Sequential(
            nn.Linear(gate_in, out_features),
            nn.Sigmoid() # Ограничивает альфа и бета от 0 до 1
        )
        
    def forward(self, x):
        # 1. Обработка в антисимметричных ветвях
        v_neg = self.branch_neg(x)
        v_pos = self.branch_pos(x) 
        
        # 2. S-петля: вычисление фазового веса (альфа/бета)
        # В классической Сфирали: R(s) = alpha(s)*r- + beta(s)*r+, где alpha^2 + beta^2 = 1
        alpha = self.s_gate(x)
        beta = torch.sqrt(torch.clamp(1.0 - alpha**2, min=1e-7))
        
        # 3. Интеграция через S-петлю (Топологическая защита от шума)
        # Антисимметричное сложение компенсирует локальные флуктуации
        out = (alpha * v_neg) + (beta * v_pos)
        
        return out

class FSINLayer(nn.Module):
    """
    Фрактальный Сфиральный Искусственный Нейрон (ФСИН).
    Рекурсивная вложенность базовых СИН для многомасштабного анализа.
    """
    def __init__(self, in_features, out_features, fractal_depth=2):
        super(FSINLayer, self).__init__()
        self.depth = fractal_depth
        self.layers = nn.ModuleList()
        
        # Создаем фрактальную структуру (упрощенная рекурсия для первого коммита)
        current_dim = in_features
        for i in range(fractal_depth):
            next_dim = out_features if i == fractal_depth - 1 else current_dim
            self.layers.append(SINLayer(current_dim, next_dim))
            current_dim = next_dim
            
    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
            # Здесь в будущем можно добавить резонансную обратную связь между уровнями фрактала
        return h
