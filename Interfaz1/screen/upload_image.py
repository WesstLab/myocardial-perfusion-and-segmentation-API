from os import listdir, path as pt
from tkinter import *
from tkinter import filedialog
import tkinter.font as font
import PIL.Image
import PIL.ImageTk
import pydicom

from model.imagen_model import Imagen
from res import dim, string, colors
from utils.dicom_utils import refactor_dicom_file
from utils.tooltip_utils import CreateToolTip


class UploadImage:

    def __init__(self, frame_lateral, frame_opciones, frame_principal, parent):
        self.frame_lateral = frame_lateral
        self.frame_opciones = frame_opciones
        self.frame_principal = frame_principal
        self.parent = parent

        self.frame_rest = Frame(self.frame_principal)
        self.frame_stress = Frame(self.frame_principal)
        self.titulo_rest = None
        self.titulo_stress = None
        self.canvas_rest = None
        self.canvas_stress = None


    def start(self):
        test_incial_elementos = Label(self.frame_lateral,
                                      text=string.TXT_UL_INIT,
                                      bg=colors.background,
                                      fg="#000000",
                                      justify="center",
                                      font=('', 18))
        test_incial_elementos.place(x=dim.WIDTH_UI_FD_TXT_INICIAL,
                                    y=dim.HEIGHT_UI_FD_TXT_INICIAL,
                                    anchor="center")
       # button_font = font.Font(family='Helvitica', size=10)

        button_add_img_rest = Button(self.frame_lateral,
                                     text=string.TXT_UL_REST,
                                     command=self.subir_img_rest,
                                     bg='#0052cc',
                                     activebackground='#0052cc',
                                     activeforeground='#aaffaa',
                                     fg='#ffffff',
                                     borderwidth=1,
                                     width=15, height=2)

        button_add_img_rest.place(x=dim.WIDTH_UI_FD_BT_REST,
                                  y=dim.HEIGHT_UI_FD_BT_REST,
                                  anchor="center")
        CreateToolTip(button_add_img_rest, "Click to add the image package "
                                                                     "exam at rest")
        button_add_img_stress = Button(self.frame_lateral,
                                       text=string.TXT_UL_STRESS,
                                       command=self.subir_img_stress,
                                       bg='#0052cc',
                                       activebackground='#0052cc',
                                       activeforeground='#aaffaa',
                                       fg='#ffffff',
                                       borderwidth=1,
                                       width=15, height=2)
        button_add_img_stress.place(x=dim.WIDTH_UI_FD_BT_STRESS,
                                    y=dim.HEIGHT_UI_FD_BT_STRESS,
                                    anchor="center")
        CreateToolTip(button_add_img_stress, "Click to add the image package "
                                             "exam in stress")
        self.init_imgs_system()

        boton_prueba = Button(self.frame_lateral,
                              text="Use Latest Data",
                              command=self.ir_a_res,
                              bg='#0052cc',
                              activebackground='#0052cc',
                              activeforeground='#aaffaa',
                              fg='#ffffff',
                              borderwidth=1,
                              width=15, height=2)
        boton_prueba.place(x=dim.WIDTH_UI_FD_TXT_INICIAL,
                           y=300,
                           anchor="center")
        CreateToolTip(boton_prueba, "Go to results of last test entered")

    def subir_img_rest(self):
        print("START UPLOAD IMG REST")
        path_imgs = filedialog.askdirectory(initialdir="/",
                                            title=string.FileDialog_Rest)
        self.parent.dir_img_rest = path_imgs
        self.parent.img_rest.reiniciar_paq()
        self.process_path_img(path_imgs, self.parent.img_rest)
        cantidad_img = self.parent.img_rest.cantidad_imagenes()
        self.titulo_rest.config(text="Images in rest: " + str(cantidad_img) + " total")
        self.chek_button()
        self.print_img(1)

    def subir_img_stress(self):
        print("START UPLOAD IMG STRESS")
        path_imgs = filedialog.askdirectory(initialdir="/",
                                            title=string.FileDialog_Stress)
        self.parent.dir_img_stress = path_imgs
        self.parent.img_stress.reiniciar_paq()
        self.process_path_img(path_imgs, self.parent.img_stress)
        cantidad_img = self.parent.img_stress.cantidad_imagenes()
        self.titulo_stress.config(text="Images in stress: " + str(cantidad_img) + " total")
        self.chek_button()
        self.print_img(2)


    def process_path_img(self, path, paq):
        lista_file = listdir(path)
        lista_file.sort()
        primero = True
        for file in lista_file:
            filename, file_extension = pt.splitext(file)
            if file_extension == ".dcm":
                archivo = pydicom.dcmread(path+"/"+file)
                if primero:
                    self.agregar_paciente(archivo, paq)
                    primero = False
                imagen = Imagen(
                    image_size=str(archivo.Rows) + 'x' + str(archivo.Columns),
                    view_size="",
                    wl=archivo.WindowCenter,
                    ww=archivo.WindowWidth,
                    mouse_mm="",
                    mouse_px="",
                    zoom="",
                    angle="",
                    image_position=archivo.ImagePositionPatient,
                    compression="",
                    thickness=archivo.SliceThickness,
                    location=archivo.SliceLocation,
                    mri="",
                    field_strenght=archivo.MagneticFieldStrength,
                    image_acq_time=archivo.AcquisitionTime,
                    pixel_array=archivo.pixel_array
                )
                paq.agregar_img(imagen)
        paq.get_array_slice()

    def chek_button(self):
        if self.parent.all_img():
            self.poner_boton_procesar()
        else:
            self.quitar_boton_procesar()

    def poner_boton_procesar(self):
        button_procesar = Button(self.frame_opciones,
                                 text=string.TXT_UL_PROCESS,
                                 command=self.parent.process_img,
                                 bg='#0052cc',
                                 activebackground='#0052cc',
                                 activeforeground='#aaffaa',
                                 fg='#ffffff',
                                 borderwidth=1,
                                 width=15, height=2)
        button_procesar.place(x=dim.WIDTH_UI_BT_PROCESS,
                              y=dim.HEIGHT_UI_BT_PROCESS,
                              anchor="center")
        CreateToolTip(button_procesar, "Click to start process \n It may take several minutes")

    def quitar_boton_procesar(self):
        for widget in self.frame_opciones.winfo_children():
            widget.destroy()

    def print_img(self, posc):
        if posc == 1:
            img = self.parent.img_rest.contenido[0].imgs[0].pixel_array

            img = refactor_dicom_file(img,
                                      self.parent.img_rest.contenido[0].imgs[0].ww,
                                      self.parent.img_rest.contenido[0].imgs[0].wl)
            img = PIL.Image.frombytes('L', (img.shape[1], img.shape[0]),
                                            img.astype('b').tostring())
            img = PIL.ImageTk.PhotoImage(img)
            self.canvas_rest.create_image(256, 256, image=img)
            self.canvas_rest.photo_reference = img

        elif posc == 2:
            img = self.parent.img_stress.contenido[0].imgs[0].pixel_array
            img = refactor_dicom_file(img,
                                      self.parent.img_stress.contenido[0].imgs[0].ww,
                                      self.parent.img_stress.contenido[0].imgs[0].wl)
            img = PIL.Image.frombytes('L', (img.shape[1], img.shape[0]),
                                      img.astype('b').tostring())
            img = PIL.ImageTk.PhotoImage(img)
            self.canvas_stress.create_image(256, 256, image=img)
            self.canvas_stress.photo_reference = img
        pass

    def imagen_sobre_canvas(self, parent, arr, x, y):
        image = Label(parent)
        img_creada = PIL.Image.frombytes('L', (arr.shape[1], arr.shape[0]), arr.astype('b').tostring())
        img_creada = PIL.ImageTk.PhotoImage(img_creada)
        image.place(x=x,
                    y=y,
                    width=dim.WIDTH_UI_RES_IMG_REST,
                    height=int((256 * dim.WIDTH_IMG_UL / 208)))
        image.config(image=img_creada)
        image.photo_reference = img_creada

    def ir_a_res(self):
        self.frame_stress.destroy()
        self.frame_rest.destroy()
        self.parent.to_result_screen()
        pass

    def agregar_paciente(self, archivo, paq):
        if paq.tipo == "Rest":
            self.parent.patient.r_name = str(archivo.PatientID)
            self.parent.patient.r_series_desc = str(archivo.SeriesDescription)
            self.parent.patient.r_series_id = str(archivo.SeriesNumber)
            self.parent.patient.r_study_desc = str(archivo.StudyDescription)
        elif paq.tipo == "Stress":
            self.parent.patient.s_name = str(archivo.PatientID)
            self.parent.patient.s_series_desc = str(archivo.SeriesDescription)
            self.parent.patient.s_series_id = str(archivo.SeriesNumber)
            self.parent.patient.s_study_desc = str(archivo.StudyDescription)

    def init_imgs_system(self):
        self.frame_rest.config(bg=colors.background,
                               width=dim.WIDTH_UI_RES_IMG_REST,
                               height=dim.HEIGHT_FRAME_IMG,
                               relief="ridge",
                               borderwidth=dim.BORDER_UI_IMG,
                               highlightbackground="black")
        self.frame_stress.config(bg=colors.background,
                                 width=dim.WIDTH_UI_RES_IMG_REST,
                                 height=dim.HEIGHT_FRAME_IMG,
                                 relief="ridge",
                                 borderwidth=dim.BORDER_UI_IMG,
                                 highlightbackground="black")

        f_titulo_rest = Frame(self.frame_rest)
        f_titulo_rest.config(width=int(dim.WIDTH_TEXT_TITLE_IMG),
                             height=int(dim.HEIGHT_TEXT_TITLE_IMG))
        f_titulo_rest.pack_propagate(0)
        self.titulo_rest = Label(f_titulo_rest)
        self.titulo_rest.config(text="Images in rest: No images",
                                bg=colors.background,
                                fg="#000000",
                                relief="ridge",
                                borderwidth=2,
                                justify="center")
        self.titulo_rest.pack(fill=BOTH, expand=1)

        f_titulo_stress = Frame(self.frame_stress)
        f_titulo_stress.config(width=int(dim.WIDTH_TEXT_TITLE_IMG),
                               height=int(dim.HEIGHT_TEXT_TITLE_IMG))
        f_titulo_stress.pack_propagate(0)
        self.titulo_stress = Label(f_titulo_stress)
        self.titulo_stress.config(text="Images in stress: No images",
                                  bg=colors.background,
                                  fg="#000000",
                                  relief="ridge",
                                  borderwidth=2,
                                  justify="center")
        self.titulo_stress.pack(fill=BOTH, expand=1)

        f_canvas_rest = Frame(self.frame_rest)
        f_canvas_rest.config(width=int(dim.WIDTH_TEXT_TITLE_IMG),
                             height=int(dim.HEIGHT_CANVAS_IMG))
        f_canvas_rest.pack_propagate(0)
        self.canvas_rest = Canvas(f_canvas_rest)
        self.canvas_rest.pack(fill=BOTH, expand=1)

        f_canvas_stress = Frame(self.frame_stress)
        f_canvas_stress.config(width=int(dim.WIDTH_UI_RES_IMG_REST),
                               height=int(dim.HEIGHT_CANVAS_IMG))
        f_canvas_stress.pack_propagate(0)
        self.canvas_stress = Canvas(f_canvas_stress)
        self.canvas_stress.pack(fill=BOTH, expand=1)

        f_titulo_rest.grid(row=0, column=1, columnspan=2)
        f_titulo_stress.grid(row=0, column=1, columnspan=2)
        f_canvas_stress.grid(row=1, column=1, columnspan=2)
        f_canvas_rest.grid(row=1, column=1, columnspan=2)

        self.frame_rest.grid(row=0, column=0)
        self.frame_stress.grid(row=0, column=1)
