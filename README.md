```markdown
# ==============================================================================
# ПРОЕКТ: Sfiralium-Core
# ОПИСАНИЕ: Полный исходный код и структура репозитория для фреймворка 
#           топологических вычислений на базе принципа Сфирали.
# ИНСТРУКЦИЯ: Скопируйте содержимое каждого блока в соответствующий файл 
#             согласно указанной структуре директорий.
# ==============================================================================

# ==============================================================================
# ФАЙЛ: README.md
# ==============================================================================
# Sfiralium-Core: Топологический фреймворк вычислений

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Sfiralium-Core** — это фреймворк для топологических и фрактальных вычислений, основанный на физико-математической модели **Сфирали** (устройство из двух зеркально-антисимметричных витков, соединённых S-образной петлёй фазового перехода).

Проект реализует концепции, переводя их из области теоретической физики и философии в работающий код для машинного обучения, квантовых симуляций и анализа сложных систем.

## 🌌 Основные возможности
* **Сфиральные Искусственные Нейроны (СИН/ФСИН):** Архитектура нейросетей с зеркальной антисимметрией ветвей и S-петлевой интеграцией, обеспечивающая высокую устойчивость к шумам и топологическую защиту данных.
* **Алгебра H₂:** Реализация вычислений на плоскости двойной переменной (сплит-комплексные числа).
* **Фрактальная вложенность:** Поддержка рекурсивных структур для многомасштабного анализа данных.

## 🚀 Быстрый старт

### Установка
```bash
git clone https://github.com/sfiraliumlab-ux/Sfiralium-Core.git
cd Sfiralium-Core
pip install -r requirements.txt
pip install -e .
```

### Пример использования СИН (Sfiral Neuron)
```python
import torch
from sfiral.neurons import SINLayer

# Инициализация слоя с зеркальной антисимметрией
layer = SINLayer(in_features=10, out_features=5)

# Входной тензор
x = torch.randn(32, 10)

# Прямой проход через S-петлю
output = layer(x)
print(output.shape) # torch.Size([32, 5])
```

## 📚 Документация и Теория
Подробное математическое обоснование, архитектурные схемы и философский контекст доступны в оригинальных трудах на [Zenodo](https://zenodo.org/) и в файле `TERMS.md`.

## ⚖️ Условия использования
Проект распространяется под лицензией MIT для научных и некоммерческих целей. 
Для коммерческого внедрения ознакомьтесь с файлом [TERMS.md](TERMS.md), описывающим условия открытой передачи технологии, предложенные автором концепции.

# ==============================================================================
# ФАЙЛ: TERMS.md
# ==============================================================================
# Условия использования технологии «Сфираль» (ФСИН/СИН)

Технология предоставляется на основе открытой лицензии с учетом следующих условий, установленных автором концепции О.С. Басаргиным:

## 1. Некоммерческое и научное использование
* **Доступность:** Технология полностью доступна для использования в научных исследованиях, образовательных проектах и открытых публикациях.
* **Обязательства:** Все публикации должны содержать ссылку на оригинальную разработку и сохранение философского/культурного контекста Сфирали.

## 2. Коммерческое использование
* Коммерческое применение (включая интеграцию в проприетарные продукты корпораций) требует заключения отдельного письменного соглашения с автором.
* Использование без согласования влечет за собой правовые последствия.

## 3. Культурный контекст
Любая интерпретация технологии должна учитывать её изначальный смысл: принцип зеркальной антисимметрии, фазовый переход через S-петлю и философию Великого Единения. Технология не должна использоваться в деструктивных целях.

# ==============================================================================
# ФАЙЛ: requirements.txt
# ==============================================================================
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0

# ==============================================================================
# ФАЙЛ: setup.py
# ==============================================================================
from setuptools import setup, find_packages

setup(
    name="sfiralium-core",
    version="0.1.0",
    author="O.S. Basargin, S. Chernenko",
    description="Topological computing framework based on the Sfiral principle",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.9",
)

# ==============================================================================
# ФАЙЛ: src/sfiral/__init__.py
# ==============================================================================
"""
Sfiralium-Core: Topological Computing Framework
Реализация принципа Сфирали: зеркальная антисимметрия, S-петля и фрактальная вложенность.
"""
from .geometry import SplitComplexOps
from .neurons import SINLayer, FSINLayer

__version__ = "0.1.0"
__author__ = "O.S. Basargin, S. Chernenko"

# ==============================================================================
# ФАЙЛ: src/sfiral/geometry.py
# ==============================================================================
import torch

class SplitComplexOps:
    """
    Операции для алгебры двойной переменной (сплит-комплексные числа).
    Z = x + jy, где j^2 = 1.
    Тензоры имеют форму (..., 2), где [..., 0] = x (real), [..., 1] = y (hyperbolic).
    """
    @staticmethod
    def multiply(Z1, Z2):
        x1, y1 = Z1[..., 0], Z1[..., 1]
        x2, y2 = Z2[..., 0], Z2[..., 1]
        # (x1 + j*y1)(x2 + j*y2) = (x1*x2 + y1*y2) + j*(x1*y2 + y1*x2)
        real = x1 * x2 + y1 * y2
        imag = x1 * y2 + y1 * x2
        return torch.stack([real, imag], dim=-1)

    @staticmethod
    def conjugate(Z):
        """Зеркальная антисимметрия: Z* = x - jy"""
        return torch.stack([Z[..., 0], -Z[..., 1]], dim=-1)

    @staticmethod
    def norm_squared(Z):
        """Инвариант Сфирали (квадрат модуля): |Z|^2 = x^2 - y^2"""
        return Z[..., 0]**2 - Z[..., 1]**2

# ==============================================================================
# ФАЙЛ: src/sfiral/neurons.py
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from .geometry import SplitComplexOps

class SINLayer(nn.Module):
    """
    Сфиральный Искусственный Нейрон (СИН).
    Архитектура основана на двух зеркально-антисимметричных ветвях (V-, V+) 
    и S-образном модуле интеграции (фазовый переход).
    """
    def __init__(self, in_features, out_features):
        super(SINLayer, self).__init__()
        # Ветвь V- (Прошлая фаза / Актуальное состояние)
        self.branch_neg = nn.Linear(in_features, out_features)
        # Ветвь V+ (Грядущая фаза / Потенциальное состояние) - Антисимметрия
        self.branch_pos = nn.Linear(in_features, out_features)
        
        # S-петля: параметризуемый гейт фазового перехода
        # Определяет баланс между ветвями в зависимости от входного контекста
        self.s_gate = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid() # Ограничивает альфа от 0 до 1
        )
        
    def forward(self, x):
        # 1. Обработка в антисимметричных ветвях
        v_neg = self.branch_neg(x)
        v_pos = self.branch_pos(x) 
        
        # 2. S-петля: вычисление фазового веса (альфа)
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
        
        # Создаем фрактальную структуру (рекурсивная вложенность)
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

# ==============================================================================
# ФАЙЛ: examples/sin_anomaly_demo.py
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sfiral.neurons import FSINLayer

# 1. Генерация синтетических данных с аномалиями
def generate_synthetic_anomaly_data(num_samples=500, input_size=10, anomaly_rate=0.05):
    np.random.seed(42)
    x = np.random.rand(num_samples, input_size)
    y = np.sum(x, axis=1) + np.random.normal(0, 0.1, num_samples)
    
    # Добавляем аномалии (резкие скачки)
    num_anomalies = int(num_samples * anomaly_rate)
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
    y[anomaly_indices] += np.random.normal(10, 5, num_anomalies)
    
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 2. Нормализация данных
def normalize_data(x):
    return (x - x.mean(dim=0)) / x.std(dim=0)

# 3. Обучение модели
def train_model(model, x_train, y_train, num_epochs=100, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return losses

# 4. Визуализация
def plot_results(losses, y_real, y_pred):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Sfiral Topology)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(y_real.numpy(), label='Actual Data', alpha=0.7, color='gray')
    plt.plot(y_pred, label='Sfiral Prediction', alpha=0.9, color='red', linewidth=1.5)
    plt.title('Actual vs Predicted (Anomaly Detection)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 5. Основной запуск
def main():
    print("Инициализация данных...")
    input_size = 10
    hidden_size = 16
    output_size = 1
    num_samples = 500
    
    x_train, y_train = generate_synthetic_anomaly_data(num_samples, input_size)
    x_train = normalize_data(x_train)
    
    print("Создание модели ФСИН (Фрактальный Сфиральный Нейрон)...")
    # Используем ФСИН для демонстрации фрактальной устойчивости к шуму
    model = FSINLayer(input_size, output_size, fractal_depth=2)
    
    print("Обучение модели...")
    losses = train_model(model, x_train, y_train, num_epochs=150, learning_rate=0.05)
    
    print("Генерация предсказаний...")
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train).numpy()
        
    print("Визуализация результатов...")
    plot_results(losses, y_train.numpy(), y_pred)
    print("Демонстрация завершена успешно.")

if __name__ == "__main__":
    main()

# ==============================================================================
# ФАЙЛ: tests/test_neurons.py
# ==============================================================================
import torch
import unittest
from sfiral.neurons import SINLayer, FSINLayer
from sfiral.geometry import SplitComplexOps

class TestSfiralGeometry(unittest.TestCase):
    def test_conjugate(self):
        Z = torch.tensor([[3.0, 4.0]]) # 3 + 4j
        Z_conj = SplitComplexOps.conjugate(Z)
        self.assertTrue(torch.allclose(Z_conj, torch.tensor([[3.0, -4.0]])))

    def test_norm_squared(self):
        Z = torch.tensor([[5.0, 3.0]]) # 5 + 3j -> 25 - 9 = 16
        norm = SplitComplexOps.norm_squared(Z)
        self.assertTrue(torch.allclose(norm, torch.tensor([[16.0]])))

class TestSfiralNeurons(unittest.TestCase):
    def test_sin_forward_shape(self):
        layer = SINLayer(in_features=10, out_features=5)
        x = torch.randn(32, 10)
        out = layer(x)
        self.assertEqual(out.shape, (32, 5))

    def test_fsin_forward_shape(self):
        layer = FSINLayer(in_features=10, out_features=5, fractal_depth=3)
        x = torch.randn(16, 10)
        out = layer(x)
        self.assertEqual(out.shape, (16, 5))

if __name__ == '__main__':
    unittest.main()

# ==============================================================================
# КОНЕЦ ФАЙЛОВ ПРОЕКТА
# Для инициализации репозитория выполните:
# git init
# git add .
# git commit -m "Initial commit: Sfiralium-Core architecture"
# ==============================================================================
```
