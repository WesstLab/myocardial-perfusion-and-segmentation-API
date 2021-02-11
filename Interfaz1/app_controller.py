from tkinter import *

from model.paq_img_model import PaqImgModel
from model.patient_data import Patient
from res import colors, dim, string
import numpy as np
import os

import pickle

from screen.results_screen import ResultScreen
from screen.upload_image import UploadImage
from segmentation.model.input_data.input_app import InputDataApp
from segmentation.model.model.models.u_net_fine_tuning import U_net_fine_tune as U_net


def predict_ztxy(model, img_array):
    n_timesteps = img_array.shape[1]
    img_length = img_array.shape[2]
    img_flatten = img_array.reshape(-1, img_length, img_length)
    img_flatten = img_flatten[..., np.newaxis]
    results_flatten = model.predict(img_flatten)
    results_array = results_flatten.reshape(
        -1, n_timesteps, img_length, img_length)
    return results_array


class AppController:

    def __init__(self):

        self.img_rest = PaqImgModel("Rest")
        self.img_stress = PaqImgModel("Stress")

        self.dir_img_rest = ""
        self.dir_img_stress = ""

        self.patient = Patient()

        # Inicio de Pantalla
        self.pantalla = Tk()

        # Configuración de Pantalla
        self.pantalla.title(string.STR_TITULO)
        self.pantalla.resizable(False, False)  # No deja redimencionar
        self.pantalla.config(bg=colors.background)

        # Frames
        self.frame_datos = Frame(self.pantalla)
        self.frame_datos.config(bg=colors.background,
                                width=dim.WIDTH_FRAME_DATOS,
                                height=dim.HEIGHT_FRAME_DATOS,
                                relief="ridge",
                                borderwidth=2,
                                highlightbackground="black")

        self.frame_img = Frame(self.pantalla)
        self.frame_img.config(#bg=colors.background,
                              width=dim.WIDTH_FRAME_IMG,
                              height=dim.HEIGHT_FRAME_IMG,
                              relief="ridge",
                              borderwidth=2,
                              highlightbackground="black")

        self.frame_opt = Frame(self.pantalla)
        self.frame_opt.config(bg=colors.background,
                              width=dim.WIDTH_FRAME_OPT,
                              height=dim.HEIGHT_FRAME_OPT,
                              relief="ridge",
                              borderwidth=2,
                              highlightbackground="black")

        frame_ul = UploadImage(frame_lateral=self.frame_datos, frame_opciones=self.frame_opt,
                               frame_principal=self.frame_img, parent=self)
        frame_ul.start()

        self.frame_datos.grid(row=0, column=0, rowspan=2)
        self.frame_img.grid(row=0, column=1)
        self.frame_opt.grid(row=1, column=1)

        self.popup = None

    def start(self):
        while True:
            try:
                self.pantalla.update_idletasks()
                self.pantalla.update()
            except UnicodeDecodeError:
                print("Caught Scroll Error")
        #  self.pantalla.mainloop()

    def procces_rest_img(self, path_img):
        """
        Esta función guarda las imágenes dcm subidas a la lista correspondiente

        Devuleve boolean, si es que puede mostrar el botón del proceso siguiente, que es el encargado de hacer el
        procesamiento de la imagen

        Parámetros:
        img -- lista de rutas de las imágenes
        """
        self.dir_img_rest = path_img
        print(path_img)
        # self.predict_img()
        return True

    def procces_stress_img(self, path_img):
        """
        Esta función guarda las imágenes dcm subidas a la lista correspondiente

        Devuleve boolean, si es que puede mostrar el botón del proceso siguiente, que es el encargado de hacer el
        procesamiento de la imagen

        Parámetros:
        img -- lista de rutas de las imágenes
        """
        self.dir_img_stress = path_img
        print(path_img)
        return True

    def all_img(self):
        if self.img_rest.esta_vacio() or self.img_stress.esta_vacio():
            return False
        else:
            return True

    def predict_img(self):
        """
        asd
        print(os.getcwd() + '\\segmentation\\checkpoint\\model')
        checkpoint_path = "C:\\Users\\Seba\\Desktop\\ProyectoE\\" \
                          "Interfaz1/segmentation/checkpoint/model"

        """
        print(os.getcwd() + '\\segmentation\\checkpoint\\model')
        checkpoint_path = "\\Users\\Seba\\Desktop\\ProyectoE\\" \
                          "Interfaz1\\segmentation\\checkpoint\\model"
        model = U_net()
        model.saver.restore(model.sess, checkpoint_path)

        dataset_stress = InputDataApp(self.dir_img_stress)
        data_extract_stress, frames_drop_stress = dataset_stress.get_data()
        pred_stress = predict_ztxy(model, data_extract_stress)
        if frames_drop_stress == 0:
            borrar = False
        else:
            borrar = True
        self.img_stress.agregar_predict(pred_stress, borrar)
        pass
        dataset_rest = InputDataApp(self.dir_img_rest)
        data_extract_rest, frames_drop_rest = dataset_rest.get_data()
        pred_rest = predict_ztxy(model, data_extract_rest)
        if frames_drop_rest == 0:
            borrar = False
        else:
            borrar = True
        self.img_rest.agregar_predict(pred_rest, borrar)

    def init_result_screen(self):
        pass

    def process_img(self):
        self.process_popup("Procesando imágenes. \n Esto puede tardar varios minutos...")
        print("processing")
        self.predict_img()
        file_pk = open('paq_stress.obj', 'wb')
        pickle.dump(self.img_stress, file_pk)
        file_pk.close()
        file_pk2 = open('paq_rest.obj', 'wb')
        pickle.dump(self.img_rest, file_pk2)
        file_pk2.close()

        self.to_result_screen()
        self.close_popup()
        pass

    def clean_pantalla(self):
        for widget in self.frame_datos.winfo_children():
            widget.destroy()
        for widget in self.frame_opt.winfo_children():
            widget.destroy()
        for widget in self.frame_img.winfo_children():
            widget.destroy()

    def to_result_screen(self):
        self.clean_pantalla()
        res_screen = ResultScreen(self.frame_datos, self.frame_opt, self.frame_img, self)
        res_screen.start()

    def clean_pantalla_pack(self):
        for widget in self.frame_datos.winfo_children():
            widget.destroy()
        for widget in self.frame_opt.winfo_children():
            widget.destroy()
        for widget in self.frame_img.winfo_children():
            widget.destroy()

    def nuevo_paciente(self):
        self.clean_pantalla_pack()
        frame_ul = UploadImage(frame_lateral=self.frame_datos, frame_opciones=self.frame_opt,
                               frame_principal=self.frame_img, parent=self)
        frame_ul.start()

        self.img_rest = PaqImgModel("Rest")
        self.img_stress = PaqImgModel("Stress")

        self.dir_img_rest = ""
        self.dir_img_stress = ""

        self.patient = Patient()

    def process_popup(self, text):
        self.popup = Toplevel()
        label = Label(self.popup, text=text)
        label.pack()

    def close_popup(self):
        self.popup.destroy()
