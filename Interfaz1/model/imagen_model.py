import cv2
import numpy as np

import matplotlib.pyplot as plt


class Imagen:

    def __init__(self, image_size, view_size, wl, ww, mouse_px, mouse_mm,
                 zoom, angle, image_position, compression, thickness, location,
                 mri, field_strenght, image_acq_time,
                 pixel_array):

        self.image_size = image_size
        self.view_size = view_size
        self.wl = wl
        self.ww = ww
        self.mouse_px = mouse_px
        self.mouse_mm = mouse_mm

        self.zoom = zoom
        self.angle = angle
        self.image_position = image_position
        self.compression = compression
        self.thickness = thickness
        self.location = location

        self.mri = mri
        self.field_strenght = field_strenght
        self.image_acq_time = image_acq_time

        self.pixel_array = pixel_array
        self.predict = []

        self.border = []

        self.radio = []
        self.endocardio = []
        self.epicardio = []
        self.sangre = []

        self.m = {'I': 0.0, 'II': 0.0, 'III': 0.0, 'IV': 0.0, 'init': ''}
        self.end_part = {"1": [], "2": [], "3": [], "4": []}
        self.epi_part = {"1": [], "2": [], "3": [], "4": []}
        self.san_part = {"1": [], "2": [], "3": [], "4": []}

    def arreglar_pad(self):
        imagen = self.pixel_array
        predict = self.predict
        epicardio = self.epicardio
        endocardio = self.endocardio
        sangre = self.sangre
        (x, y) = imagen.shape
        (xp, yp) = predict.shape
        qx = int((xp - x) / 2)
        qy = int((yp - y) / 2)
        self.predict = predict[qx:xp - qx, qy:yp - qy]
        self.epicardio = np.array(epicardio)[qx:xp - qx, qy:yp - qy]
        self.endocardio = np.array(endocardio)[qx:xp - qx, qy:yp - qy]
        self.sangre = np.array(sangre)[qx:xp - qx, qy:yp - qy]
        self.detectar_centro()

    def separacion_miocardio(self):
        """
        Detecta borde del predict, su centro y separa el miocardio entre epicardio y endocardio
        :return:
        """
        self.deteccion_bordes()
        self.detectar_separacion()
        pass

    def deteccion_bordes(self):
        """
        Detecta los bordes de la segmentacion encontrada usando Canny
        :return: guarda los resultados en self.border
        """
        img_augmented = self.predict * 5
        border = cv2.Canny(img_augmented, 3, 7)
        self.border = border

    def detectar_centro(self):
        """
        Detectar el centro del miocardio. Para esto se calcula el promedio de las posiciones del pool de sangre
        :return: guarda el centro en self.radio
        """
        sum_x = 0
        sum_y = 0
        cant_px = 0
        for i in range(len(self.predict)):
            for j in range(len(self.predict[i])):
                if self.predict[i][j] == 2:
                    sum_x += i
                    sum_y += j
                    cant_px += 1
        try:
            x = np.rint(sum_x / cant_px)
        except ZeroDivisionError:
            x = 0
        try:
            y = np.rint(sum_y / cant_px)
        except ZeroDivisionError:
            y = 0
        self.radio = [x, y]

    def detectar_separacion(self):
        """
        Detectar que parte del miocardio es parte del endocardio y cual del epicardio
        Buscar valores 1 de self.predict
        - Si más cercano es 0: ENDOCARDIO
        - Si más cercano es 2: EPICARDIO
        - Si ambos cercanos: EPICARDIO
        :return: guarda los resutlados en self.epicardio y self.endocardio
        """
        for i in range(len(self.predict)):
            x_end = []
            x_epi = []
            x_san = []
            for j in range(len(self.predict[i])):
                is0 = False
                is2 = False
                if self.predict[i][j] == 1:
                    dif = 1
                    while not is0 and not is2:
                        is0, is2 = self.mas_cercano(i, j, dif)
                        if is2 and not is0:
                            x_end.append(1)
                            x_epi.append(0)
                            x_san.append(0)
                        elif is0:
                            x_end.append(0)
                            x_epi.append(1)
                            x_san.append(0)
                        dif += 1
                elif self.predict[i][j] == 2:
                    x_san.append(1)
                    x_end.append(0)
                    x_epi.append(0)
                else:
                    x_epi.append(0)
                    x_end.append(0)
                    x_san.append(0)
            self.endocardio.append(np.array(x_end))
            self.epicardio.append(np.array(x_epi))
            self.sangre.append(np.array(x_san))

    def mas_cercano(self, pos_x, pos_y, dist):
        """
        Encuentra el punto más cercano de interes al punto de análisis. Se debe encontrar fuera del miocardio (0) o
        sangre (2). Se buscar en cuadrados, con centro en el punto
        :param pos_x: posicion x del punto
        :param pos_y: posicion y del punto
        :param dist: distancia desde el punto hacia donde analizar
        :return: is0 True si detecto miocardio, False caso contrario. is2 True si detecto sangre, False, caso contrario
        """
        is0 = False
        is2 = False
        # Fijar pos_y-dist
        if pos_y - dist >= 0:
            for i in range(dist):
                if pos_x - i >= 0:
                    if self.predict[pos_x - i][pos_y - dist] == 0:
                        is0 = True
                    elif self.predict[pos_x - i][pos_y - dist] == 2:
                        is2 = True
                if pos_x + i < len(self.predict):
                    if self.predict[pos_x + i][pos_y - dist] == 0:
                        is0 = True
                    elif self.predict[pos_x + i][pos_y - dist] == 2:
                        is2 = True
        # Fijar pos_y + dist
        if pos_y + dist < len(self.predict[0]):
            for i in range(dist):
                if pos_x - i >= 0:
                    if self.predict[pos_x - i][pos_y + dist] == 0:
                        is0 = True
                    elif self.predict[pos_x - i][pos_y + dist] == 2:
                        is2 = True
                if pos_x + i < len(self.predict):
                    if self.predict[pos_x + i][pos_y + dist] == 0:
                        is0 = True
                    elif self.predict[pos_x + i][pos_y + dist] == 2:
                        is2 = True
        # Fijar pos_x-dist
        if pos_x - dist >= 0:
            for i in range(dist):
                if pos_y - i >= 0:
                    if self.predict[pos_x - dist][pos_y - i] == 0:
                        is0 = True
                    elif self.predict[pos_x - dist][pos_y - i] == 2:
                        is2 = True
                if pos_y + i < len(self.predict[i]):
                    if self.predict[pos_x - dist][pos_y + i] == 0:
                        is0 = True
                    elif self.predict[pos_x - dist][pos_y + i] == 2:
                        is2 = True
        # Fijar pos_x + dist
        if pos_x + dist >= len(self.predict):
            for i in range(dist):
                if pos_y - i >= 0:
                    if self.predict[pos_x + dist][pos_y - i] == 0:
                        is0 = True
                    elif self.predict[pos_x + dist][pos_y - i] == 2:
                        is2 = True
                if pos_y + i < len(self.predict[i]):
                    if self.predict[pos_x + dist][pos_y + i] == 0:
                        is0 = True
                    elif self.predict[pos_x + dist][pos_y + i] == 2:
                        is2 = True
        return is0, is2

    def get_prom_epi(self, parte):
        """
        Calcula el promedio de intensidad de pixeles del Epicardio en la zona la imagen
        :param parte: a que cantidad de data se refiere
            0: toda la zona del epicardio
            1: subdiv1
            2: subdiv2
            3: subdiv3
            4: subdiv4
        :return: float valor del promedio de intensidad
        """
        sum_in = 0
        total = 0
        if parte == 0:
            for i in range(len(self.epicardio)):
                for j in range(len(self.epicardio[i])):
                    if self.epicardio[i][j] == 1:
                        sum_in += self.pixel_array[i][j]
                        total += 1
        else:
            data = self.epi_part[str(parte)]
            for i in range(len(data)):
                for j in range(len(data[i])):
                    if data[i][j] == 1:
                        sum_in += self.pixel_array[i][j]
                        total += 1
        try:
            prom = sum_in / total
        except ZeroDivisionError:
            prom = 0
        return prom

    def get_prom_end(self, parte):
        """
        Calcula el promedio de intensidad de pixeles del Endocardio en la zona la imagen
        :param parte: a que cantidad de data se refiere
            0: toda la zona del epicardio
            1: subdiv1
            2: subdiv2
            3: subdiv3
            4: subdiv4
        :return: float valor del promedio de intensidad
        """
        sum_in = 0
        total = 0
        if parte == 0:
            for i in range(len(self.endocardio)):
                for j in range(len(self.endocardio[i])):
                    if self.endocardio[i][j] == 1:
                        sum_in += self.pixel_array[i][j]
                        total += 1
        else:
            data = self.end_part[str(parte)]
            for i in range(len(data)):
                for j in range(len(data[i])):
                    if data[i][j] == 1:
                        sum_in += self.pixel_array[i][j]
                        total += 1
        try:
            prom = sum_in/total
        except ZeroDivisionError:
            prom = 0
        return prom

    def get_prom_blood(self, parte):
        """
        Calcula el promedio de intensidad de pixeles de zona de sangre de la imagen
        :param parte: a que cantidad de data se refiere
            0: toda la zona del epicardio
            1: subdiv1
            2: subdiv2
            3: subdiv3
            4: subdiv4
        :return: float valor del promedio de intensidad
        """
        sum_in = 0
        total = 0
        if parte == 0:
            for i in range(len(self.pixel_array)):
                for j in range(len(self.pixel_array[i])):
                    if self.sangre[i][j] == 1:
                        sum_in += self.pixel_array[i][j]
                        total += 1
        else:
            data = self.san_part[str(parte)]
            for i in range(len(data)):
                for j in range(len(data[i])):
                    if data[i][j] == 1:
                        sum_in += self.pixel_array[i][j]
                        total += 1
        try:
            prom = sum_in / total
        except ZeroDivisionError:
            prom = 0
        return prom

    def dividir_miocardio(self, punto):
        """
        Recibiendo el punto ingresado por el usario, divide el miocardio en las 4 secciones necesarias, utlizando el
        sistema de cuadrantes, que complementan las funciones llamadas
        :param punto: punto ingresado por el usuario
        :return: void, las secciones quedan guardadas como parámetro de la imagen.
        """
        x_p = int(punto[0])
        y_p = int(punto[1])
        p = self.calculo_m(x_p, y_p)
        if p != 0 and np.isfinite(p):
            cuadrante = self.detectar_cuadrante(punto)  #  Da el cuadrante en el que se encuentra I, II, III, IV
            self.m[cuadrante] = p
            self.m["init"] = cuadrante
            self.completar_pendientes()
            self.partir_puntos()
        else:
            cuadrante = self.detectar_cuadrante(punto)  # Da el eje en el que se encuentra I-II, II-III, III-IV, I-IV
            self.m["init"] = cuadrante
            self.partir_puntos()


    def detectar_cuadrante(self, p):
        """
        Dado un punto p, calcula el cuadrante al que pertenece, tomando en consideración que el (0,0) del sistema
        creado es el punto de radio del pool de sangre. En caso de que toque un eje, se devuelve un par con los
        cuadrantes que toca.
        :param p: punto ingresado para analizar (x, y)
        :return: String con el valor del cuadrante donde se encuentra
        """
        x = p[0]
        y = p[1]
        if x > self.radio[0]:
            if y > self.radio[1]:
                return 'IV'
            elif y < self.radio[1]:
                return 'I'
            else:
                return 'I-IV'
        elif x < self.radio[0]:
            if y > self.radio[1]:
                return 'III'
            elif y < self.radio[1]:
                return 'II'
            else:
                return 'II-III'
        else:
            if y > self.radio[1]:
                return 'III-IV'
            elif y <= self.radio[1]:
                return 'I-II'

    def completar_pendientes(self):
        """
        Se llama a esta función solo si es que el punto ingresado tiene una pendiente distinta a 0 y finita.
        Tomando el valor de pendiente del init, procede a calcular el resto de las penditentes usando m1*m2=-1, ya que
        deben ser pendientes perpendiculares
        :return:
        """
        init = self.m["init"]
        p = self.m[init]
        p_cont = -1 / p
        if init == "I" or init == "III":
            self.m["I"] = p
            self.m["II"] = p_cont
            self.m["III"] = p
            self.m["IV"] = p_cont
        elif init == "II" or init == "IV":
            self.m["I"] = p_cont
            self.m["II"] = p
            self.m["III"] = p_cont
            self.m["IV"] = p

    def calculo_m(self, x, y):
        """
        Calulo de pendiente con respecto al centro del pool de sangre. Si es que se divide por cero, se devuelve el
        infinito de nunpy
        :param x: valor en el eje X
        :param y: valor en el eje Y
        :return: valor de pendiente calculada
        """
        try:
            if x < int(self.radio[0]):
                p = (y - int(self.radio[1])) / (int(self.radio[0]) - x)
            else:
                p = (int(self.radio[1] - y)) / (x - int(self.radio[0]))
        except ZeroDivisionError:
            p = np.inf
        return p

    def partir_puntos(self):
        """
        Se recorren los puntos de predict, endocadio y epicardio, se encuentran los puntos de interes y luefo se ingresa
        en la zona que le corresponde. Se usan las funciones ingresar_punto si es que el punto de toque ingresado se
        encuentra en la zona de cuadrantes, o sino ingresar_por_eje, en caso de estar en uno de los ejes.
        Se identifican según si la pendiente sigue siendo 0
        :return: void, llena el diccionario con las 4 zonas divididas
        """
        for i in range(len(self.predict)):
            end1 = []
            end2 = []
            end3 = []
            end4 = []
            epi1 = []
            epi2 = []
            epi3 = []
            epi4 = []
            san1 = []
            san2 = []
            san3 = []
            san4 = []
            for j in range(len(self.predict[i])):
                #  Buscar puntos de sangre
                if self.sangre[i][j] == 1:
                    pos_c = self.detectar_cuadrante((i, j))
                    pend = self.calculo_m(i, j)
                    if self.m["I"] == 0.0:
                        self.ingresar_por_eje(pos_c, san1, san2, san3, san4)
                    else:
                        self.ingresar_punto(pos_c, pend, san1, san2, san3, san4)
                else:
                    self.ingresar_punto('0', 0, san1, san2, san3, san4)

                #  Buscar punto de Epicardio
                if self.epicardio[i][j] == 1:
                    pos_c = self.detectar_cuadrante((i, j))
                    pend = self.calculo_m(i, j)
                    if self.m["I"] == 0.0:
                        self.ingresar_por_eje(pos_c, epi1, epi2, epi3, epi4)
                    else:
                        self.ingresar_punto(pos_c, pend, epi1, epi2, epi3, epi4)
                else:
                    self.ingresar_punto('0', 0, epi1, epi2, epi3, epi4)

                #  Buscar punto de Endocardio
                if self.endocardio[i][j] == 1:
                    pos_c = self.detectar_cuadrante((i, j))
                    pend = self.calculo_m(i, j)
                    if self.m["I"] == 0.0:
                        self.ingresar_por_eje(pos_c, end1, end2, end3, end4)
                    else:
                        self.ingresar_punto(pos_c, pend, end1, end2, end3, end4)
                else:
                    self.ingresar_punto('0', 0, end1, end2, end3, end4)
            self.san_part["1"].append(san1)
            self.san_part["2"].append(san2)
            self.san_part["3"].append(san3)
            self.san_part["4"].append(san4)
            self.epi_part["1"].append(epi1)
            self.epi_part["2"].append(epi2)
            self.epi_part["3"].append(epi3)
            self.epi_part["4"].append(epi4)
            self.end_part["1"].append(end1)
            self.end_part["2"].append(end2)
            self.end_part["3"].append(end3)
            self.end_part["4"].append(end4)

    def ingresar_punto(self, pos, pend, a1, a2, a3, a4):
        """
        Dado el punto, se ingresa el punto al array que le corresponde como un 1, mientras que en los otros se ingresa
        un 0, para mantener la forma de imagen y luego poder ingesarla
        :param pos: posicion con respecto al cuadrante
        :param pend: pendiente del punto, sirve para comparar y ver a cual segmento va
        :param a1: primer array de forma
        :param a2: segundo array de forma
        :param a3: tercer array de forma
        :param a4: cuarto array de forma
        :return: void, se finaliza con el punto agregado donde corresponde, y el resto con 0
        """
        if pos == '0':
            a1.append(0); a2.append(0); a3.append(0); a4.append(0)
        else:
            if self.m["init"] == 'I':
                if pos == 'I':
                    if pend < self.m["I"]:
                        a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                    else:
                        a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'II':
                    if pend < self.m["II"]:
                        a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                    else:
                        a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'III':
                    if pend < self.m["III"]:
                        a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                    else:
                        a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'IV':
                    if pend < self.m["IV"]:
                        a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                    else:
                        a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'I-II':
                    a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'II-III':
                    a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'III-IV':
                    a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'I-IV':
                    a1.append(1); a2.append(0); a3.append(0); a4.append(0)
            elif self.m["init"] == 'II':
                if pos == 'I':
                    if pend < self.m["I"]:
                        a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                    else:
                        a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'II':
                    if pend < self.m["II"]:
                        a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                    else:
                        a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'III':
                    if pend < self.m["III"]:
                        a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                    else:
                        a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'IV':
                    if pend < self.m["IV"]:
                        a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                    else:
                        a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'I-II':
                    a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'II-III':
                    a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'III-IV':
                    a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'I-IV':
                    a1.append(0); a2.append(1); a3.append(0); a4.append(0)
            elif self.m["init"] == 'III':
                if pos == 'I':
                    if pend < self.m["I"]:
                        a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                    else:
                        a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'II':
                    if pend < self.m["II"]:
                        a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                    else:
                        a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'III':
                    if pend < self.m["III"]:
                        a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                    else:
                        a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'IV':
                    if pend < self.m["IV"]:
                        a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                    else:
                        a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'I-II':
                    a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'II-III':
                    a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'III-IV':
                    a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'I-IV':
                    a1.append(0); a2.append(0); a3.append(1); a4.append(0)
            elif self.m["init"] == 'IV':
                if pos == 'I':
                    if pend < self.m["I"]:
                        a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                    else:
                        a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'II':
                    if pend < self.m["II"]:
                        a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                    else:
                        a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'III':
                    if pend < self.m["III"]:
                        a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                    else:
                        a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'IV':
                    if pend < self.m["IV"]:
                        a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                    else:
                        a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'I-II':
                    a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'II-III':
                    a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'III-IV':
                    a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'I-IV':
                    a1.append(0); a2.append(0); a3.append(0); a4.append(1)

    def ingresar_por_eje(self, pos, a1, a2, a3, a4):
        """
        Caso en el que el punto tocado sea parte de algún eje. En este caso, el punto a ingresar, será parte de un
        cuadrante directamente, ya que las secciones son cuadrantes
        :param pos: pos del punto a ingresar
        :param a1: primera posc de array
        :param a2: segunda posc de array
        :param a3: tercero posc de array
        :param a4: cuarto posc de array
        :return: void, llena los array con sus correspondientes
        """
        if pos == '0':
            a1.append(0); a2.append(0); a3.append(0); a4.append(0)
        else:
            if self.m["init"] == 'I-II':
                if pos == 'I':
                    a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'II':
                    a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'III':
                    a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'IV':
                    a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'I-II':
                    a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'II-III':
                    a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'III-IV':
                    a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'I-IV':
                    a1.append(1); a2.append(0); a3.append(0); a4.append(0)
            elif self.m["init"] == 'II-III':
                if pos == 'I':
                    a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'II':
                    a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'III':
                    a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'IV':
                    a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'I-II':
                    a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'II-III':
                    a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'III-IV':
                    a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'I-IV':
                    a1.append(0); a2.append(1); a3.append(0); a4.append(0)
            elif self.m["init"] == 'III-IV':
                if pos == 'I':
                    a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'II':
                    a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'III':
                    a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'IV':
                    a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'I-II':
                    a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'II-III':
                    a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'III-IV':
                    a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'I-IV':
                    a1.append(0); a2.append(0); a3.append(1); a4.append(0)
            elif self.m["init"] == 'I-IV':
                if pos == 'I':
                    a1.append(0); a2.append(0); a3.append(0); a4.append(1)
                elif pos == 'II':
                    a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'III':
                    a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'IV':
                    a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'I-II':
                    a1.append(0); a2.append(0); a3.append(1); a4.append(0)
                elif pos == 'II-III':
                    a1.append(0); a2.append(1); a3.append(0); a4.append(0)
                elif pos == 'III-IV':
                    a1.append(1); a2.append(0); a3.append(0); a4.append(0)
                elif pos == 'I-IV':
                    a1.append(0); a2.append(0); a3.append(0); a4.append(1)
