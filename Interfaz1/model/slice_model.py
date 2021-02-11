import skimage

from utils.dicom_utils import refactor_dicom_file

import PIL.Image
import PIL.ImageTk

import numpy as np
import cv2


class SliceModel:

    def __init__(self, slice_loc):
        self.slice_loc = slice_loc
        self.imgs = []
        self.inicio = 0
        self.fin = 0
        self.pos_actual = 0

        #  Datos de curvas, promedios de intensidades
        self.data_sangre = {'total': [], 'sub1': [], 'sub2': [], 'sub3': [], 'sub4': []}
        self.data_epicardio = {'total': [], 'sub1': [], 'sub2': [], 'sub3': [], 'sub4': []}
        self.data_endocardio = {'total': [], 'sub1': [], 'sub2': [], 'sub3': [], 'sub4': []}

        self.data_tiempo = []   # Comun para todas las curvas

        self.ww = 0
        self.wl = 0

    def agregar_img_slice(self, imagen):
        self.imgs.append(imagen)
        self.ordenar_lista()
        self.fin = len(self.imgs)-1

    def ordenar_lista(self):
        self.imgs.sort(key=lambda x: x.image_acq_time)
        self.ww = self.imgs[0].ww
        self.wl = self.imgs[0].wl

    def cantidad_imgs(self):
        return len(self.imgs)

    def quitar_primera(self):
        self.imgs = self.imgs[1:]
        self.fin = self.cantidad_imgs() - 1

    def agregar_predict_img(self, predict):
        for i in range(len(predict)):
            self.imgs[i].predict = predict[i]
            self.imgs[i].separacion_miocardio()
            self.imgs[i].arreglar_pad()
        self.data_sangre['total'] = self.calculo_curva(tipo=0, zona=1)
        self.data_epicardio['total'] = self.calculo_curva(tipo=0, zona=2)
        self.data_endocardio['total'] = self.calculo_curva(tipo=0, zona=3)
        self.data_tiempo = self.arrays_tiempo()

    def current_img(self, w, h):
        img_actual = self.imgs[self.pos_actual]
        img_array = img_actual.pixel_array
        img_array = refactor_dicom_file(img_array, self.ww, self.wl)
        img_array = cv2.resize(img_array, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        img_array = PIL.Image.frombytes('L', (img_array.shape[1], img_array.shape[0]), img_array.astype('b').tostring())

        left_button = True
        if self.pos_actual == 0:
            left_button = False
        right_button = True
        if self.pos_actual == (self.cantidad_imgs()-1):
            right_button = False
        return img_array, self.pos_actual, left_button, right_button

    def current_predict(self, zona):
        """
        Entrega el predict de la zona pedida
        :param zona: Zona pedida
            0: Toda
            1: Sub1
            2: Sub2
            3: Sub3
            4: Sub4
        :return:
        """
        img_actual = self.imgs[self.pos_actual]
        if zona == 0:
            sangre = img_actual.sangre
            epicardio = img_actual.epicardio
            endocardio = img_actual.endocardio
        else:
            sangre = img_actual.san_part[str(zona)]
            epicardio = img_actual.epi_part[str(zona)]
            endocardio = img_actual.end_part[str(zona)]
        return sangre, epicardio, endocardio

    def aumentar_pos(self):
        if self.pos_actual == (self.cantidad_imgs()-1):
            pass
        else:
            self.pos_actual += 1

    def disminuir_pos(self):
        if self.pos_actual == 0:
            pass
        else:
            self.pos_actual -= 1

    def calculo_curva(self, tipo, zona):
        """
        Calcula el promedio de intensidades de las imágenes del array a analiazar
        :param tipo: los datos con los que se trabajara
            0: toda la zona
            1: subdivisión 1
            2: subdivisión 2
            3: subdivisión 3
            4: subdivisión 4
        :param zona: zona en la que se trabajara
            1: sangre
            2: epicardio
            3: endocardio
        :return: par de arrays data vs tiempo
        """
        data = []
        for image in self.imgs:
            intensidad = 0
            if zona == 1:
                intensidad = image.get_prom_blood(tipo)
            elif zona == 2:
                intensidad = image.get_prom_epi(tipo)
            elif zona == 3:
                intensidad = image.get_prom_end(tipo)
            data.append(intensidad)
        return data

    def arrays_tiempo(self):
        time = []
        for image in self.imgs:
            time.append(image.image_acq_time)
        return time

    def set_init_current(self):
        self.inicio = self.pos_actual

    def set_fin_current(self):
        self.fin = self.pos_actual

    def calcular_division_miocardio(self, punto):
        """
        Al entregarle el punto ingresado por el usuario, se dispone a recorrer las imágenes para calcular las cuatro
        divisiones del miocardio que se necesitan. Además, guarda el promedio de intensidades de curva en los
        parámetros de la clase
        :param punto: punto (x, y) ingresados por el usuario
        :return: void, quedan los datos guardados en la imagen, además de los parametros de intensidad de curva
        """
        for imagen in self.imgs:
            imagen.dividir_miocardio(punto)
        self.data_sangre["sub1"] = self.calculo_curva(1, 1)
        self.data_sangre["sub2"] = self.calculo_curva(2, 1)
        self.data_sangre["sub3"] = self.calculo_curva(3, 1)
        self.data_sangre["sub4"] = self.calculo_curva(4, 1)
        self.data_epicardio["sub1"] = self.calculo_curva(1, 2)
        self.data_epicardio["sub2"] = self.calculo_curva(2, 2)
        self.data_epicardio["sub3"] = self.calculo_curva(3, 2)
        self.data_epicardio["sub4"] = self.calculo_curva(4, 2)
        self.data_endocardio["sub1"] = self.calculo_curva(1, 3)
        self.data_endocardio["sub2"] = self.calculo_curva(2, 3)
        self.data_endocardio["sub3"] = self.calculo_curva(3, 3)
        self.data_endocardio["sub4"] = self.calculo_curva(4, 3)



def predict_none(predict):
    for i in predict:
        for j in i:
            if j == 0:
                j = None
    return predict


def border_none(border):
    for i in border:
        for j in i:
            if j == 0:
                j = None
    return border


def epi_none(epi):
    for i in epi:
        for j in i:
            if j == 0:
                j = None
    return epi


def endo_none(endo):
    for i in endo:
        for j in i:
            if j == 0:
                j = None
    return endo

