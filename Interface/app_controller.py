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

        # Screen Startup
        self.screen = Tk()

        # Display Configuration
        self.screen.title(string.STR_TITULO)
        self.screen.resizable(False, False)  # Does not allow to resize
        self.screen.config(bg=colors.background)

        # Frames
        self.frame_data = Frame(self.screen)
        self.frame_data.config(bg=colors.background,
                               width=dim.WIDTH_FRAME_DATA,
                               height=dim.HEIGHT_FRAME_DATA,
                               relief="ridge",
                               borderwidth=2,
                               highlightbackground="black")

        self.frame_img = Frame(self.screen)
        self.frame_img.config(width=dim.WIDTH_FRAME_IMG,
                              height=dim.HEIGHT_FRAME_IMG,
                              relief="ridge",
                              borderwidth=2,
                              highlightbackground="black")

        self.frame_opt = Frame(self.screen)
        self.frame_opt.config(bg=colors.background,
                              width=dim.WIDTH_FRAME_OPT,
                              height=dim.HEIGHT_FRAME_OPT,
                              relief="ridge",
                              borderwidth=2,
                              highlightbackground="black")

        frame_ul = UploadImage(frame_lateral=self.frame_data, frame_opciones=self.frame_opt,
                               frame_principal=self.frame_img, parent=self)
        frame_ul.start()

        self.frame_data.grid(row=0, column=0, rowspan=2)
        self.frame_img.grid(row=0, column=1)
        self.frame_opt.grid(row=1, column=1)

        self.popup = None

    def start(self):
        while True:
            try:
                self.screen.update_idletasks()
                self.screen.update()
            except UnicodeDecodeError:
                print("Caught Scroll Error")
        #  self.screen.mainloop()

    def process_rest_img(self, path_img):
        """
        This function saves the dcm images uploaded to the corresponding list.

        Returns boolean, if it can display the button of the next process, which is responsible for
        doing the image processing

        Parameters:
        img -- list of image paths
        """
        self.dir_img_rest = path_img
        print(path_img)
        # self.predict_img()
        return True

    def process_stress_img(self, path_img):
        """
        This function saves the dcm images uploaded to the corresponding list.

        Returns boolean, if it can display the button of the next process, which is in charge of doing the
        processing of the image

        Parameters:
        img -- list of image paths
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
        
        print(os.getcwd() + '\\segmentation\\checkpoint\\model')
        checkpoint_path = os.getcwd() + '\\segmentation\\checkpoint\\model'

        model = U_net()
        model.saver.restore(model.sess, checkpoint_path)

        dataset_stress = InputDataApp(self.dir_img_stress)
        data_extract_stress, frames_drop_stress = dataset_stress.get_data()
        pred_stress = predict_ztxy(model, data_extract_stress)
        if frames_drop_stress == 0:
            delete = False
        else:
            delete = True
        self.img_stress.agregar_predict(pred_stress, delete)
        pass
        dataset_rest = InputDataApp(self.dir_img_rest)
        data_extract_rest, frames_drop_rest = dataset_rest.get_data()
        pred_rest = predict_ztxy(model, data_extract_rest)
        if frames_drop_rest == 0:
            delete = False
        else:
            delete = True
        self.img_rest.agregar_predict(pred_rest, delete)

    def init_result_screen(self):
        pass

    def process_img(self):
        self.process_popup("Processing images. \n This can take several minutes...")
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

    def clean_screen(self):
        for widget in self.frame_data.winfo_children():
            widget.destroy()
        for widget in self.frame_opt.winfo_children():
            widget.destroy()
        for widget in self.frame_img.winfo_children():
            widget.destroy()

    def to_result_screen(self):
        self.clean_screen()
        res_screen = ResultScreen(self.frame_data, self.frame_opt, self.frame_img, self)
        res_screen.start()

    def clean_screen_pack(self):
        for widget in self.frame_data.winfo_children():
            widget.destroy()
        for widget in self.frame_opt.winfo_children():
            widget.destroy()
        for widget in self.frame_img.winfo_children():
            widget.destroy()

    def new_patient(self):
        self.clean_screen_pack()
        frame_ul = UploadImage(frame_lateral=self.frame_data, frame_opciones=self.frame_opt,
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
