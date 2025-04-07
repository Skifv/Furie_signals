import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import sys

# Проверяем, передан ли хотя бы один аргумент
if len(sys.argv) > 1:
    number = sys.argv[-1]  # Последний аргумент, количество складываемых гармоник
    try:
        # Преобразуем в число
        N = int(number)
    except ValueError:
        print(f"Ошибка: '{number}' не является числом.")
        exit(1)
else:
    print("Аргументы не переданы.")
    exit(1)


# Параметры
T = 100e-6  # Период, 2π для параметризации кривой
w = 2 * np.pi / T

a = 0 # Начало диапазона
b = a + T # Конец диапазона

# Флаги отображения
show_graph_x = True  # Отображение графика x(t)
show_graph_y = True  # Отображение графика y(t)
show_graph_xy = True # Отображение графика y(x)
print_amplitudes_x = True  # Печать амплитуд и фаз для x(t)
print_amplitudes_y = True  # Печать амплитуд и фаз для y(t)

# Параметрические уравнения для кривой сердца
def x(t):
    return 16 * np.sin(w * t)**3

def y(t):
    return 13 * np.cos(w * t) - 5 * np.cos(2 * w * t) - 2 * np.cos(3 * w * t) - np.cos(4 * w * t)

# Коэффициенты Фурье для x(t) и y(t)
def a_k(k, T, func):
    result, _ = quad(lambda t: func(t) * np.cos(2 * np.pi * k * t / T), a, b)
    return 2 / T * result

def b_k(k, T, func):
    result, _ = quad(lambda t: func(t) * np.sin(2 * np.pi * k * t / T), a, b)
    return 2 / T * result

# Вычисление амплитуды и фазы для x(t) и y(t)
def calculate_amplitudes_and_phases(N, T, func):
    amplitudes = []
    phases = []
    for k in range(1, N + 1):
        a = a_k(k, T, func)
        b = b_k(k, T, func)
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
def print_harmonics(a0, amplitudes, phases, real_phases, label):
    print(f"Гармоники для {label}:")
    print(f"  Постоянная компонента: A_0 = {a0:.3f}")
    for k, (amplitude, phase, real_phase) in enumerate(zip(amplitudes, phases, real_phases), start=1):
        print(f"  Гармоника {k}: A = {amplitude:.3f}, φ_theor = {phase:.0f}°, φ_real = {real_phase}°")


def get_real_phases(phases):
    # Округляем элементы массива до целых чисел
    res = np.round(phases).astype(int)

    # Преобразуем первый элемент
    res[0] = (90 - res[0]) % 360

    phi_0 = res[0]
        
    # Преобразуем остальные элементы    
    for k in range(1, len(res)):
        res[k] = (90 - res[k] - (k + 1) * phi_0) % 360
    
    return res

# Построение графиков
def plot_signal(t_vals, original_signal, approx_signal, label):
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals * 1e6, original_signal, linestyle='--', label=f"Оригинал {label}")
    plt.plot(t_vals * 1e6, approx_signal, label=f"Фурье: {label}")
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
    plt.plot(approx_x, approx_y, label=f"Фурье: y(x), N={N}")
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    plt.legend()
    plt.axis("equal")  # Сделать оси одинаковыми, чтобы фигура не была искажена
    plt.grid()
    plt.title("График зависимости y(x)")
    plt.show()

# Подготовка данных
a0_x = a_k(0, T, x) / 2
amplitudes_x, phases_x = calculate_amplitudes_and_phases(N, T, x)
real_phases_x = get_real_phases(phases_x)

a0_y = a_k(0, T, y) / 2
amplitudes_y, phases_y = calculate_amplitudes_and_phases(N, T, y)
real_phases_y = get_real_phases(phases_y)

# Печать гармоник
if print_amplitudes_x:
    print_harmonics(a0_x, amplitudes_x, phases_x, real_phases_x, "x(t)")
if print_amplitudes_y:
    print_harmonics(a0_y, amplitudes_y, phases_y, real_phases_y, "y(t)")

# Построение графиков x(t), y(t)
t_vals = np.linspace(a, b, 1000)
original_x = [x(t) for t in t_vals]
original_y = [y(t) for t in t_vals]

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
