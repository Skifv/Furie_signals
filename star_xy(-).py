import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Параметры
N = 16  # Количество складываемых гармоник
T = 100e-6  # Период, 100 мкс

a = 0  # Начало диапазона
b = a + T  # Конец диапазона

# Флаги отображения
show_graph_x = True  # Отображение графика x(t)
show_graph_y = True  # Отображение графика y(t)
show_graph_xy = True # Отображение графика y(x)
print_amplitudes_x = True  # Печать амплитуд и фаз для x(t)
print_amplitudes_y = True  # Печать амплитуд и фаз для y(t)

# Точки для x(t)
points_x = [
    (a, -0.5),
    (a + T / 5, 0),
    (a + 2 * T / 5, 0.5),
    (a + 3 * T / 5, -0.8),
    (a + 4 * T / 5, 0.8),
    (b, -0.5)
]

# Точки для y(t)
points_y = [
    (a, -0.91),
    (a + T / 5, 0.58),
    (a + 2 * T / 5, -0.91),
    (a + 3 * T / 5, 0),
    (a + 4 * T / 5, 0),
    (b, -0.91)
]

# Функция для построения кусочно-линейного сигнала
def piecewise_linear_signal(t, points):
    for i in range(len(points) - 1):
        t1, y1 = points[i]
        t2, y2 = points[i + 1]
        if t1 <= t <= t2:
            k = (y2 - y1) / (t2 - t1)
            b = y1 - k * t1
            return k * t + b
    return 0

# Коэффициенты Фурье
def a_k(k, T, func, points):
    result, _ = quad(lambda t: func(t, points) * np.cos(2 * np.pi * k * t / T), a, b)
    return 2 / T * result

def b_k(k, T, func, points):
    result, _ = quad(lambda t: func(t, points) * np.sin(2 * np.pi * k * t / T), a, b)
    return 2 / T * result

# Вычисление амплитуды и фазы
def calculate_amplitudes_and_phases(N, T, func, points):
    amplitudes = []
    phases = []
    for k in range(1, N + 1):
        a = a_k(k, T, func, points)
        b = b_k(k, T, func, points)
        amplitude = np.sqrt(a**2 + b**2)
        phase = np.arctan2(b, a) * (180 / np.pi)
        amplitudes.append(amplitude)
        phases.append(phase)
    return amplitudes, phases

# Восстановление сигнала из ряда Фурье
def fourier_series(t, a0, amplitudes, phases, N, T):
    result = a0 + sum(
        amplitudes[k - 1] * np.cos(2 * np.pi * k * t / T - np.radians(phases[k - 1]))
        for k in range(1, N + 1)
    )
    return result

# Печать гармоник
def print_harmonics(a0, amplitudes, phases, label):
    print(f"Гармоники для {label}:")
    print(f"  Постоянная компонента: A_0 = {a0:.3f}")
    for k, (amplitude, phase) in enumerate(zip(amplitudes, phases), start=1):
        print(f"  Гармоника {k}: A = {amplitude:.3f}, φ = {phase:.0f}°")

# Построение графиков
def plot_signal(t_vals, original_signal, approx_signal, label):
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals * 1e6, original_signal, linestyle='--', label=f"Оригинал {label}")
    #plt.plot(t_vals * 1e6, approx_signal, label=f"Фурье: {label}")
    plt.xlabel("Время [мкс]")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid()
    plt.title(f"График сигнала {label}")
    plt.show()

# Построение графика зависимости y(x)
def plot_xy(original_x, original_y, approx_x, approx_y):
    plt.figure(figsize=(10, 6))
    plt.plot(original_x, original_y, linestyle='--', label="Оригинал: y(x)")
    #plt.plot(approx_x, approx_y, label=f"Фурье: y(x), N={N}")
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    plt.axis("equal")  # Сделать оси одинаковыми, чтобы фигура не была искажена
    plt.legend()
    plt.grid()
    plt.title("График зависимости y(x)")
    plt.show()

# Подготовка данных
a0_x = a_k(0, T, piecewise_linear_signal, points_x) / 2
amplitudes_x, phases_x = calculate_amplitudes_and_phases(N, T, piecewise_linear_signal, points_x)

a0_y = a_k(0, T, piecewise_linear_signal, points_y) / 2
amplitudes_y, phases_y = calculate_amplitudes_and_phases(N, T, piecewise_linear_signal, points_y)

# Печать гармоник
if print_amplitudes_x:
    print_harmonics(a0_x, amplitudes_x, phases_x, "x(t)")
if print_amplitudes_y:
    print_harmonics(a0_y, amplitudes_y, phases_y, "y(t)")

# Построение графиков x(t), y(t)
t_vals = np.linspace(a, b, 1000)
original_x = [piecewise_linear_signal(t, points_x) for t in t_vals]
original_y = [piecewise_linear_signal(t, points_y) for t in t_vals]

approx_x = [fourier_series(t, a0_x, amplitudes_x, phases_x, N, T) for t in t_vals]
approx_y = [fourier_series(t, a0_y, amplitudes_y, phases_y, N, T) for t in t_vals]

# Построение временных графиков
if show_graph_x:
    plot_signal(t_vals, original_x, approx_x, "x(t)")
if show_graph_y:
    plot_signal(t_vals, original_y, approx_y, "y(t)")

# Построение графика зависимости y(x)
if show_graph_xy:
    plot_xy(original_x, original_y, approx_x, approx_y)
