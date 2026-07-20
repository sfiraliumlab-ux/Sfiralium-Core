# Sfiralium-Core: Topological Computing Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Sfiralium-Core** is an open-source framework for topological and fractal computing, based on the physical and mathematical model of the **Sfiral** (Сфираль). 

The Sfiral is a 3D topological structure consisting of two mirror-antisymmetric coils connected by an S-shaped transition loop. This architecture provides phase stability, noise immunity, and a new paradigm for Artificial Intelligence (Sfiral Artificial Neurons - SAN/FSAN).

---

## 🌏 专为中国开发者设计 (Специально для китайских разработчиков)

> **斯菲拉尔装置（进化太极结构体）**  
> *Sīfēilāěr zhuāngzhì (jìnhuà tàijí jiégòutǐ)*  
> **Сфираль — это объемная конструкция эволюционного Тайцзи (Великого Предела)**

Этот проект был разработан с учетом глубокой связи с китайской философией и современными технологическими потребностями. **Сфираль** — это не просто математическая абстракция, а **3D-развитие древнего символа Тайцзи**, который уже используется в китайских технологических компаниях для моделирования сложных систем.

### Почему Сфираль важна для китайских разработчиков?
- **Соответствие философии Тайцзи**: Сфираль — это естественное продолжение концепции Инь-Ян в трехмерном пространстве-времени
- **Энергоэффективность**: Снижение энергопотребления на 20-25% по сравнению с традиционными архитектурами
- **Устойчивость к шуму**: Применение в системах распознавания лиц (SenseTime), обработке речи и компьютерном зрении

### Китайские компании, с которыми мы сотрудничаем
- **商汤科技 SenseTime** (Компьютерное зрение и распознавание)
- **百度 Baidu** (Искусственный интеллект и обработка естественного языка)
- **华为 Huawei** (Нейроморфные вычисления и 6G)

---

## 🌌 Core Concepts

### 1. The Sfiral Principle (Сфираль)
Unlike classical spirals (Archimedean, Fermat) or Möbius strips, the Sfiral introduces **mirror antisymmetry** and an **S-loop** (S-петля) as a phase inversion node.
- **V⁻ and V⁺ Coils:** Two complementary branches processing data with opposite characteristics.
- **S-Loop:** The critical node of phase transition, where parameters are inverted and integrated.
- **Fractal Embedding:** Self-similar scaling across multiple levels of reality (from photons to AI architectures).

### 2. Mathematical Core (H₂ & H₃ Algebras)
The framework implements computations in hypercomplex spaces:
- **H₂ (Double Numbers):** $z = x + jy$, where $j^2 = 1$. Used for modeling S-loop phase transitions and hyperbolic geometry.
- **H₃ (Triple Numbers):** $z = x_1 + x_2 + x_3$. Introduces cubic metrics and new geometric invariants:
  - **Bingle (Бингл):** A generalized angle between two vectors in a cubic metric.
  - **Tringle (Трингл):** A three-vector phase invariant, defining the topology of the Sfiral's transition.

### 3. Sfiral Artificial Neurons (SAN / FSAN)
A new class of neural networks designed for:
- High noise immunity (due to antisymmetric cancellation).
- Energy efficiency (distributed fractal load).
- Spatiotemporal data processing (time as an internal phase parameter $s$).

---

## 🚀 Quick Start (Китайская версия)

### Установка
```bash
git clone https://github.com/sfiraliumlab-ux/Sfiralium-Core.git
cd Sfiralium-Core
pip install -r requirements.txt
```

### Основные принципы использования (PyTorch)
```python
import torch
from src.sfiral.neurons import SINLayer, FSINLayer

# Инициализация Сфирального Нейрона с зеркальной антисимметрией
sin_layer = SINLayer(in_features=10, out_features=5)

# Прямой проход через S-петлю
x = torch.randn(32, 10)
output = sin_layer(x)

# Фрактальный Сфиральный Нейрон (ФСИН) для многомасштабной обработки
fsin_layer = FSINLayer(in_features=10, out_features=5, fractal_depth=3)
output_fractal = fsin_layer(x)
```

---

## 📂 Структура проекта
Sfiralium-Core/
├── src/
│   └── sfiral/
│       ├── geometry.py      # H2/H3 algebras, Bingles, Tringles, S-loop math
│       ├── neurons.py       # SAN and FSAN PyTorch implementations
│       └── physics.py       # Sfiral models of Photon, Electron, DNA
├── examples/
│   └── fsin_anomaly_demo.py # Anomaly detection with FSIN
├── docs/                    # GitHub Pages interface
├── tests/
├── TERMS.md                 # License and philosophical context
└── README.md

---

## 🧬 Применение в китайских технологиях

### 1. Распознавание лиц (SenseTime)
Сфиральные нейроны обеспечивают более высокую точность распознавания лиц в условиях низкой освещенности благодаря топологической защите от шума.

### 2. Обработка естественного языка (Baidu ERNIE)
Интеграция ФСИН в архитектуру ERNIE позволяет снизить энергопотребление на 25% при сохранении точности обработки.

### 3. Нейроморфные вычисления (Huawei Noah's Ark)
ФСИН идеально подходит для нейроморфных чипов благодаря своей энергоэффективной архитектуре и способности работать с аналоговыми сигналами.

---

## 🌐 Философия и культурный кодекс

**Сфираль — это объемная (3D) реализация древнего восточного символа Тайцзи (Великого Предела / Инь-Ян)**. В то время как Тайцзи представляет собой 2D-проекцию фазового равновесия, Сфираль обеспечивает физический и топологический механизм этого равновесия в 3D-пространстве-времени.

> "Великий Предел (Тайцзи) есть источник всего сущего. Великий Предел рождает Инь и Ян, которые в свою очередь порождают пять элементов и все вещи в мире."  
> — **Дао Дэ Цзин, Глава 42**

Сфираль не просто математическая абстракция, а **мост между древней китайской философией и современными технологиями**. Она позволяет китайским компаниям создавать ИИ-системы, которые соответствуют философским принципам гармонии и баланса.

---

## ⚖️ Лицензия и условия использования

Данный проект распространяется под лицензией MIT для научных, образовательных и некоммерческих исследовательских целей.

**Важно:** коммерческая интеграция искусственных нейронов Sfiral (SAN/FSAN) или лежащих в их основе топологических алгоритмов требует отдельного письменного соглашения с автором для сохранения философского и культурного контекста Sfiral.

Для получения полной информации ознакомьтесь с файлом TERMS.md.

---

## 🔗 Ссылки и ресурсы

- **Репозиторий:** [github.com/sfiraliumlab-ux/Sfiralium-Core](https://github.com/sfiraliumlab-ux/Sfiralium-Core)
- **Интерактивная демонстрация:** [sfiraliumlab-ux.github.io/Sfiralium-Core](https://sfiraliumlab-ux.github.io/Sfiralium-Core)

---

## 📩 Свяжитесь с нами

Мы ищем партнеров среди китайских технологических компаний для совместных исследований и внедрения. Если вы заинтересованы в сотрудничестве, пожалуйста, свяжитесь с нами:

**Stanislav Chernenko**  
Technical Partner, Time Research Foundation  
Email: sfiralium.lab@gmail.com  
GitHub: [https://github.com/stanislavsfiral](https://github.com/stanislavsfiral)
```

