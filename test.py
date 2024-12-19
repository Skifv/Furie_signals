import numpy as np

def get_real_phases(phases):
    # Округляем элементы массива до целых чисел
    res = np.round(phases).astype(int)
    
    # Преобразуем первый элемент
    res[0] = (res[0] - 90) % 360
    
    # Преобразуем остальные элементы
    for k in range(1, len(res)):
        res[k] = (res[k] - 90 - 90 * (k - 1)) % 360
    
    return res
