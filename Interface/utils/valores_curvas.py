import numpy as np
from scipy import signal


def calculo_maximo(data, time):
    """
    Calcula el mayor valor de la data entrante, y devuelve el tiempo y la posición en la que se encuentra este mayor
    :param data: Data en la que se busca el max
    :param time: Array de tiempo, corresponde al tiempo de cada dato de data
    :return: maximo valor, tiempo en el que ocurre, y la posición del array
    """
    input_data = np.array(data)
    pos_max = signal.find_peaks_cwt(input_data, np.arange(1, 150))[0]
    time_max = np.array(time)[pos_max]
    max_element = np.round(input_data[pos_max], 2)

    return max_element, time_max, pos_max


def calculo_pendiente(data, time):
    """
    Devuelve la pendiente inicial, considerando el rise time de la curva com el momento donde la data sube desde el 10%
    del máximo al 90% del máximo
    :param data: Data de entrada, eje X
    :param time: Data de tiempo, eje Y
    :return: valor de la pendiente, tiempo inicial, tiempo final, valor inicial, valor final
    """
    max_data, time_max, pos_max = calculo_maximo(data, time)
    por10 = max_data*0.1
    por90 = max_data*0.9
    first = 0
    last = 0
    for valor in np.array(data):
        if valor > por10 and first == 0:
            first = valor
            continue
        if valor < por90:
            last = valor
            continue
        else:
            break
    pos_first = np.where(np.array(data) == first)[0][0]
    pos_last = np.where(np.array(data) == last)[0][0]
    time_first = np.array(time).astype(np.float)[pos_first]
    time_last = np.array(time).astype(np.float)[pos_last]
    try:
        p = np.round((last-first)/(time_last-time_first), 2)
    except ZeroDivisionError:
        p = 0
    return p, time_first, time_last, first, last


def calculo_area_curva(data, time):
    """
    Entregado el tiempo y la data, calcula el área de la curva utilizando trampz de Numpy
    :param data: Array con los valores de los promedios de las intensidades
    :param time: Array con los valores de tiempo de las imágenes
    :return: area calculada, redondeada al segundo decimal
    """
    input_data = np.array(data)
    input_time = np.array(time)
    input_time = input_data.astype(np.float)
    area = np.round(np.trapz(input_data, input_time), 2)
    return area
