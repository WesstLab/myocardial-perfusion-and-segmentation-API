from tkinter import *

from model.exportacion_model import ExportPDF
from res import dim, string, colors
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from PIL import Image, ImageTk


import pickle

from utils.dicom_utils import refactor_dicom_file
from utils.tkinter_img import img2rgba
from utils.tooltip_utils import CreateToolTip
from utils.valores_curvas import *


class ResultScreen:

    def __init__(self, frame_lateral, frame_opciones, frame_principal, parent):
        self.frame_lateral = frame_lateral
        self.frame_opciones = frame_opciones
        self.frame_princial = frame_principal
        self.parent = parent

        #  FRAME OPCIONES
        self.boton_export = Button(self.frame_opciones,bg='#0052cc',activebackground='#0052cc',activeforeground='#aaffaa',fg='#ffffff',borderwidth=1,width=15,height=2)
        self.boton_particion = Button(self.frame_opciones,bg='#0052cc',activebackground='#0052cc',activeforeground='#aaffaa',fg='#ffffff',borderwidth=1,width=15,height=2)
        self.boton_volver = Button(self.frame_opciones,bg='#0052cc',activebackground='#0052cc',activeforeground='#aaffaa',fg='#ffffff',borderwidth=1,width=15,height=2)
        self.b_export_ttip = CreateToolTip(self.boton_export, string.tt_export)
        self.b_particion_ttip = CreateToolTip(self.boton_particion, string.tt_particion)
        self.b_volver_ttip = CreateToolTip(self.boton_volver, string.tt_volver)

        #  FRAME LATERAL
        self.f_titulo_lat = Frame(self.frame_lateral)
        self.titulo_lat = Label(self.f_titulo_lat)
        self.f_paciente = Frame(self.frame_lateral)
        self.paciente = Label(self.f_paciente)
        self.f_study_desc = Frame(self.frame_lateral)
        self.study_desc = Label(self.f_study_desc)
        self.f_serie_desc = Frame(self.frame_lateral)
        self.serie_desc = Label(self.f_serie_desc)
        self.f_serie_id = Frame(self.frame_lateral)
        self.serie_id = Label(self.f_serie_id)
        self.f_espacio_lateral_1 = Frame(self.frame_lateral)
        self.espacio_lateral_1 = Label(self.f_espacio_lateral_1)
        self.f_espacio_lateral_2 = Frame(self.frame_lateral)
        self.espacio_lateral_2 = Label(self.f_espacio_lateral_2)
        self.f_titulo_curva_stress = Frame(self.frame_lateral)
        self.titulo_curva_stress = Label(self.f_titulo_curva_stress)
        self.f_titulo_curva_rest = Frame(self.frame_lateral)
        self.titulo_curva_rest = Label(self.f_titulo_curva_rest)


        self.f_canvas_stress = Frame(self.frame_lateral)
        self.figure_stress = Figure(figsize=None, dpi=100)
        self.plot_stress = self.figure_stress.add_subplot(111)
        self.plot_stress_res = None
        self.plot_stress_axis = None
        self.canvas_stress = FigureCanvasTkAgg(self.figure_stress, self.f_canvas_stress)

        self.f_canvas_rest = Frame(self.frame_lateral)
        self.figure_rest = Figure(dpi=100)
        self.plot_rest = self.figure_rest.add_subplot(111)
        self.plot_rest_res = None
        self.plot_rest_axis = None
        self.canvas_rest = FigureCanvasTkAgg(self.figure_rest, self.f_canvas_rest)

        self.f_button_epi = Frame(self.frame_lateral)
        self.button_epi = Button(self.f_button_epi,bg='#0052cc',activebackground='#0052cc',activeforeground='#aaffaa',fg='#ffffff',borderwidth=1,width=15,height=2)
        CreateToolTip(self.f_button_epi, string.tt_bt_epi)
        self.f_button_end = Frame(self.frame_lateral)
        self.button_end = Button(self.f_button_end,bg='#0052cc',activebackground='#0052cc',activeforeground='#aaffaa',fg='#ffffff',borderwidth=1,width=15,height=2)
        CreateToolTip(self.f_button_end, string.tt_bt_end)
        self.f_button_san = Frame(self.frame_lateral)
        self.button_san = Button(self.f_button_san,bg='#0052cc',activebackground='#0052cc',activeforeground='#aaffaa',fg='#ffffff',borderwidth=1,width=15,height=2)
        CreateToolTip(self.f_button_san, string.tt_bt_san)

        #  FRAME IMAGENES
        self.f_titulo_img_res = Frame(self.frame_princial)
        self.titulo_img_res = Label(self.f_titulo_img_res)
        self.f_titulo_img_stress = Frame(self.frame_princial)
        self.titulo_img_stress = Label(self.f_titulo_img_stress)
        self.f_img_res = Frame(self.frame_princial)
        self.img_res = Canvas(self.f_img_res)
        self.f_img_stress = Frame(self.frame_princial)
        self.img_stress = Canvas(self.f_img_stress)

        self.zona_select = 1

        self.f_label_slice = Frame(self.frame_princial)
        self.label_slice = Label(self.f_label_slice)
        self.f_valor_slice = Frame(self.frame_princial)
        self.valor_slice = ttk.Combobox(self.f_valor_slice)
        CreateToolTip(self.f_valor_slice, string.tt_slice)

        self.f_label_ww = Frame(self.frame_princial)
        self.label_ww = Label(self.f_label_ww)
        self.f_valor_ww = Frame(self.frame_princial)
        self.valor_ww = Entry(self.f_valor_ww)
        CreateToolTip(self.f_valor_ww, string.tt_ww)
        self.f_label_wl = Frame(self.frame_princial)
        self.label_wl = Label(self.f_label_wl)
        self.f_valor_wl = Frame(self.frame_princial)
        self.valor_wl = Entry(self.f_valor_wl)
        CreateToolTip(self.f_valor_wl, string.tt_wl)
        self.f_boton_recarga = Frame(self.frame_princial)
        self.boton_recarga = Button(self.f_boton_recarga,bg='#0052cc',activebackground='#0052cc',activeforeground='#aaffaa',fg='#ffffff',borderwidth=1,width=15,height=2)
        CreateToolTip(self.f_boton_recarga, string.tt_recargar)
        self.f_init_stress = Frame(self.frame_princial)
        self.init_stress = Button(self.f_init_stress)
        self.f_fin_stress = Frame(self.frame_princial)
        self.fin_stress = Button(self.f_fin_stress)
        self.f_init_rest = Frame(self.frame_princial)
        self.init_rest = Button(self.f_init_rest)
        self.f_fin_rest = Frame(self.frame_princial)
        self.fin_rest = Button(self.f_fin_rest)
        self.f_izq_rest = Frame(self.frame_princial)
        self.izq_rest = Button(self.f_izq_rest)
        self.f_der_rest = Frame(self.frame_princial)
        self.der_rest = Button(self.f_der_rest)
        self.f_izq_stress = Frame(self.frame_princial)
        self.izq_stress = Button(self.f_izq_stress)
        self.f_der_stress = Frame(self.frame_princial)
        self.der_stress = Button(self.f_der_stress)

        self.f_info_res_left = Frame(self.frame_princial)
        self.info_res_left = Label(self.f_info_res_left)
        self.f_info_res_center = Frame(self.frame_princial)
        self.info_res_center = Label(self.f_info_res_center)
        self.f_info_res_right = Frame(self.frame_princial)
        self.info_res_right = Label(self.f_info_res_right)
        self.f_info_stress_left = Frame(self.frame_princial)
        self.info_stress_left = Label(self.f_info_stress_left)
        self.f_info_stress_center = Frame(self.frame_princial)
        self.info_stress_center = Label(self.f_info_stress_center)
        self.f_info_stress_right = Frame(self.frame_princial)
        self.info_stress_right = Label(self.f_info_stress_right)

        self.f_label_div = Frame(self.frame_princial)
        self.label_div = Label(self.f_label_div)
        self.f_valor_div = Frame(self.frame_princial)
        self.valor_div = ttk.Combobox(self.f_valor_div)
        CreateToolTip(self.valor_div, string.tt_valor_div)

        self.f_esp_1 = Frame(self.frame_princial)
        self.esp_1 = Label(self.f_esp_1)
        self.f_esp_2 = Frame(self.frame_princial)
        self.esp_2 = Label(self.f_esp_2)
        self.f_esp_3 = Frame(self.frame_princial)
        self.esp_3 = Label(self.f_esp_3)
        self.f_esp_4 = Frame(self.frame_princial)
        self.esp_4 = Label(self.f_esp_4)
        self.f_dif_1 = Frame(self.frame_princial)
        self.dif_1 = Label(self.f_dif_1)
        self.f_dif_2 = Frame(self.frame_princial)
        self.dif_2 = Label(self.f_dif_2)
        self.f_dif_3 = Frame(self.frame_princial)
        self.dif_3 = Label(self.f_dif_3)
        self.f_dif_4 = Frame(self.frame_princial)
        self.dif_4 = Label(self.f_dif_4)
        self.f_dif_5 = Frame(self.frame_princial)
        self.dif_5 = Label(self.f_dif_5)
        self.f_dif_6 = Frame(self.frame_princial)
        self.dif_6 = Label(self.f_dif_6)

        self.f_dif_7 = Frame(self.frame_princial)
        self.dif_7 = Label(self.f_dif_7)
        self.f_dif_8 = Frame(self.frame_princial)
        self.dif_8 = Label(self.f_dif_8)
        self.f_dif_9 = Frame(self.frame_princial)
        self.dif_9 = Label(self.f_dif_8)

        #  Labels de Tabla de Resultados
        self.f_tabla_resultados = Frame(self.frame_princial)
        self.resultados_stress = None
        self.resultados_rest = None
        self.res_area = None
        self.area_stress = None
        self.area_rest = None
        self.res_peak = None
        self.res_peak_rest = None
        self.res_peak_stress = None
        self.res_pend = None
        self.res_pend_rest = None
        self.res_pend_stress = None
        self.res_ratio = None
        self.res_ratio_value = None

        self.slice_select_rest = 0
        self.slice_select_stress = 0
        self.pos_actual_img = 0
        self.photo_reference_rest = None
        self.photo_reference_stress = None

        self.habilitar_punto_rest = 0
        self.habilitar_punto_stress = 0
        self.habilitado_para_div = 0

        #  Imagenes mostradas en Canvas
        self.predict_rest = None
        self.predict_stress = None
        self.imagen_rest = None
        self.imagen_stress = None

        self.punto_stress = None
        self.punto_stress_canvas = ""
        self.punto_rest_canvas = ""
        self.punto_rest = None
        self.subdiv = 'total'

    def start(self):
        self.iniciar_opciones()
        self.iniciar_lateral()
        self.iniciar_principal()

        self.iniciar_data()

    def iniciar_opciones(self):
        # BOTONES OPT
        self.boton_export.config(text=string.boton_export,
                                 command=self.exportacion_pdf)
        self.boton_export.place(x=dim.pos_x_buttons_opt,
                                y=dim.pos_y_buttons_opt,
                                anchor="center")
        self.boton_particion.config(text=string.boton_particion,
                                    command=self.particionar)
        self.boton_particion.place(x=dim.pos_x_buttons_opt * 3,
                                   y=dim.pos_y_buttons_opt,
                                   anchor="center")
        self.boton_volver.config(text=string.boton_volver,
                                 command=self.volver_upload)
        self.boton_volver.place(x=dim.pos_x_buttons_opt * 5,
                                y=dim.pos_y_buttons_opt,
                                anchor="center")

    def iniciar_lateral(self):
        # INFO FRAME
        self.f_titulo_lat.config(width=dim.width_res_opt_title,
                                 height=dim.height_res_opt_title)
        # bordes para prueba de frame ,relief="solid", bd=1)
        self.f_titulo_lat.pack_propagate(0)
        self.titulo_lat.config(text=string.txt_lateral_title,
                               font=('', 14)) # original font=('',14)
        self.titulo_lat.pack(fill=BOTH, expand=1)


        self.f_paciente.config(width=dim.width_res_opt_title,
                               height=dim.height_res_opt_paciente)
        self.f_paciente.pack_propagate(0)
        self.paciente.config(text=string.txt_info_paciente,
                             anchor=W)
        self.paciente.pack(fill=BOTH, expand=1)

        self.f_study_desc.config(width=dim.width_res_opt_title,
                                 height=dim.height_res_opt_study)
        self.f_study_desc.pack_propagate(0)
        self.study_desc.config(text=string.txt_info_study_desc,
                               anchor=W)
        self.study_desc.pack(fill=BOTH, expand=1)

        self.f_serie_desc.config(width=dim.width_res_opt_title,
                                 height=dim.height_res_opt_series_desc)
        self.f_serie_desc.pack_propagate(0)
        self.serie_desc.config(text=string.txt_info_series_desc,
                               anchor=W)
        self.serie_desc.pack(fill=BOTH, expand=1)

        self.f_serie_id.config(width=dim.width_res_opt_title,
                               height=dim.height_res_opt_series_id)
        self.f_serie_id.pack_propagate(0)
        self.serie_id.config(text=string.txt_info_series_id,
                             anchor=W)
        self.serie_id.pack(fill=BOTH, expand=1)

        self.f_espacio_lateral_1.config(width=dim.width_res_opt_title,
                                        height=dim.height_res_opt_espacio)
        self.f_espacio_lateral_1.pack_propagate(0)
        self.espacio_lateral_1.pack(fill=BOTH, expand=1)

        #  Botones pada modificar curvas, ahora solo un boton para ambas
        self.f_button_san.config(width=dim.width_res_opt_curva_button,
                                 height=dim.height_res_opt_curva_button)
        self.f_button_san.pack_propagate(0)
        self.button_san.config(text=string.txt_botones_curva_sangre,
                               font=('', 9), #original font=('', 10)
                               command=lambda: self.curva_print(1))
        self.button_san.pack(side=LEFT, fill=BOTH, expand=1)

        self.f_button_epi.config(width=dim.width_res_opt_curva_button,
                                 height=dim.height_res_opt_curva_button)
        self.f_button_epi.pack_propagate(0)
        self.button_epi.config(text=string.txt_botones_curva_epi,
                               font=('', 9), #original font=('', 10)
                               command=lambda: self.curva_print(2))
        self.button_epi.pack(fill=BOTH, expand=1)

        self.f_button_end.config(width=dim.width_res_opt_curva_button,
                                 height=dim.height_res_opt_curva_button)
        self.f_button_end.pack_propagate(0)
        self.button_end.config(text=string.txt_botones_curva_endo,
                               font=('', 9), #original font=('', 10)
                               command=lambda: self.curva_print(3))
        self.button_end.pack(fill=BOTH, expand=1)

        self.f_titulo_curva_rest.config(width=dim.width_res_opt_title,
                                        height=dim.height_res_opt_curva_title)
        self.f_titulo_curva_rest.pack_propagate(0)
        self.titulo_curva_rest.config(text=string.txt_title_curva_rest,
                                      font=('', 15))
        self.titulo_curva_rest.pack(fill=BOTH, expand=1)


        self.f_canvas_rest.config(width=dim.width_res_opt_title,
                                  height=dim.height_res_opt_curva_grafico,
                                  borderwidth=1,
                                  relief="solid")
        self.f_canvas_rest.pack_propagate(0)
        self.canvas_rest.draw()
        self.canvas_rest.get_tk_widget().pack(fill=BOTH, expand=1)

        self.f_espacio_lateral_2.config(width=dim.width_res_opt_title,
                                        height=dim.height_res_opt_espacio)
        self.f_espacio_lateral_2.pack_propagate(0)
        self.espacio_lateral_2.pack(fill=BOTH, expand=1)

        self.f_titulo_curva_stress.config(width=dim.width_res_opt_title,
                                          height=dim.height_res_opt_curva_title)
        self.f_titulo_curva_stress.pack_propagate(0)
        self.titulo_curva_stress.config(text=string.txt_title_curva_stress,
                                        font=('', 15))
        self.titulo_curva_stress.pack(fill=BOTH, expand=1)

        self.f_canvas_stress.config(width=dim.width_res_opt_title,
                                    height=dim.height_res_opt_curva_grafico,
                                    borderwidth=1,
                                    relief="solid")
        self.f_canvas_stress.pack_propagate(0)
        self.canvas_stress.draw()
        self.canvas_stress.get_tk_widget().pack(fill=BOTH, expand=1)

        self.frame_lateral.grid_propagate(False)
        self.f_titulo_lat.grid(row=1, column=1, columnspan=3)
        self.f_paciente.grid(row=2, column=1, columnspan=3)
        self.f_study_desc.grid(row=3, column=1, columnspan=3)
        self.f_serie_desc.grid(row=4, column=1, columnspan=3)
        self.f_serie_id.grid(row=5, column=1, columnspan=3)
        self.f_espacio_lateral_1.grid(row=6, column=1, columnspan=3)
        self.f_button_san.grid(row=7, column=1)
        self.f_button_epi.grid(row=7, column=2)
        self.f_button_end.grid(row=7, column=3)
        self.f_titulo_curva_rest.grid(row=8, column=1, columnspan=3)
        self.f_canvas_rest.grid(row=9, column=1, columnspan=3)
        self.f_espacio_lateral_2.grid(row=10, column=1, columnspan=3)
        self.f_titulo_curva_stress.grid(row=11, column=1, columnspan=3)
        self.f_canvas_stress.grid(row=12, column=1, columnspan=3)

    def iniciar_principal(self):
        self.f_esp_1.config(width=dim.w_espacio_1,
                            height=dim.h_espacio_1)
        self.f_esp_1.pack_propagate(0)
        #  self.f_esp_1.config(borderwidth=2, relief="solid")
        self.esp_1.pack(fill=BOTH, expand=1)

        #  --------------------

        self.f_dif_1.config(width=dim.w_dif_1,
                            height=dim.h_dif_1)
        self.f_dif_1.pack_propagate(0)
        #  self.f_dif_1.config(borderwidth=2, relief="solid")
        self.dif_1.pack(fill=BOTH, expand=1)

        self.f_titulo_img_res.config(width=dim.width_title_img,
                                     height=dim.height_title_img)
        self.f_titulo_img_res.pack_propagate(0)
        self.titulo_img_res.config(text=string.txt_titulo_img_rest,
                                   font=('', 15))
        self.titulo_img_res.config(borderwidth=2, relief="solid")
        self.titulo_img_res.pack(fill=BOTH, expand=1)

        self.f_img_res.config(width=dim.width_img_res_screen,
                              height=dim.height_img_res_screen)
        self.f_img_res.pack_propagate(0)
        self.img_res.config(borderwidth=2, relief="solid")
        self.img_res.pack(fill=BOTH, expand=1)

        self.f_dif_2.config(width=dim.w_dif_2,
                            height=dim.h_dif_2)
        self.f_dif_2.pack_propagate(0)
        #  self.dif_2.config(borderwidth=2, relief="solid")
        self.dif_2.pack(fill=BOTH, expand=1)

        self.f_izq_rest.config(width=dim.width_button_left_right,
                               height=dim.height_button_left_right)
        self.f_izq_rest.pack_propagate(0)
        self.izq_rest.config(text="<",
                             command=lambda: self.mov_img(1, 1))
        self.izq_rest.pack(fill=BOTH, expand=1)

        self.f_info_res_left.config(width=dim.width_button_init_fin,
                                    height=dim.height_button_init_fin)
        self.f_info_res_left.pack_propagate(0)
        self.info_res_left.pack(fill=BOTH, expand=1)
        self.f_info_res_center.config(width=dim.w_dif_4,
                                      height=dim.h_dif_4)
        self.f_info_res_center.pack_propagate(0)
        self.info_res_center.config(borderwidth=2, relief="solid")
        self.info_res_center.pack(fill=BOTH, expand=1)
        self.f_info_res_right.config(width=dim.width_button_init_fin,
                                     height=dim.height_button_init_fin)
        self.f_info_res_right.pack_propagate(0)
        self.info_res_right.pack(fill=BOTH, expand=1)

        self.f_der_rest.config(width=dim.width_button_left_right,
                               height=dim.height_button_left_right)
        self.f_der_rest.pack_propagate(0)
        self.der_rest.config(text=">",
                             command=lambda: self.mov_img(1, 2))
        self.der_rest.pack(fill=BOTH, expand=1)

        self.f_dif_3.config(width=dim.w_dif_3,
                            height=dim.h_dif_3)
        self.f_dif_3.pack_propagate(0)
        #  self.dif_3.config(borderwidth=2, relief="solid")
        self.dif_3.pack(fill=BOTH, expand=1)

        self.f_init_rest.config(width=dim.width_button_init_fin,
                                height=dim.height_button_init_fin)
        self.f_init_rest.pack_propagate(0)
        self.init_rest.config(text="init",
                              command=lambda: self.set_init_fin(1, 1))
        self.init_rest.pack(fill=BOTH, expand=1)

        self.f_dif_4.config(width=dim.w_dif_4,
                            height=dim.h_dif_4)
        self.f_dif_4.pack_propagate(0)
        #  self.dif_4.config(borderwidth=2, relief="solid")
        self.dif_4.pack(fill=BOTH, expand=1)
        self.f_fin_rest.config(width=dim.width_button_init_fin,
                               height=dim.height_button_init_fin)
        self.f_fin_rest.pack_propagate(0)
        self.fin_rest.config(text="fin",
                             command=lambda: self.set_init_fin(2, 1))
        self.fin_rest.pack(fill=BOTH, expand=1)

        self.f_dif_5.config(width=dim.w_dif_5,
                            height=dim.h_dif_5)
        self.f_dif_5.pack_propagate(0)
        #  self.dif_5.config(borderwidth=2, relief="solid")
        self.dif_5.pack(fill=BOTH, expand=1)

        #  --------------------

        self.f_esp_2.config(width=dim.w_espacio_2,
                            height=dim.h_espacio_2)
        self.f_esp_2.pack_propagate(0)
        #  self.esp_2.config(borderwidth=2, relief="solid")
        self.esp_2.pack(fill=BOTH, expand=1)

        #  --------------------

        self.f_titulo_img_stress.config(width=dim.width_title_img,
                                        height=dim.height_title_img)
        self.f_titulo_img_stress.pack_propagate(0)
        self.titulo_img_stress.config(text=string.txt_titulo_img_stress,
                                      font=('', 15),
                                      borderwidth=2,
                                      relief="solid")
        self.titulo_img_stress.pack(fill=BOTH, expand=1)

        self.f_img_stress.config(width=dim.width_img_res_screen,
                                 height=dim.height_img_res_screen)
        self.f_img_stress.pack_propagate(0)
        self.img_stress.config(borderwidth=2, relief="solid")
        self.img_stress.pack(fill=BOTH, expand=1)

        self.f_izq_stress.config(width=dim.width_button_left_right,
                                 height=dim.height_button_left_right)
        self.f_izq_stress.pack_propagate(0)
        self.izq_stress.config(text="<",
                               command=lambda: self.mov_img(2, 1))
        self.izq_stress.pack(fill=BOTH, expand=1)

        self.f_info_stress_left.config(width=dim.width_button_init_fin,
                                       height=dim.height_button_init_fin)
        self.f_info_stress_left.pack_propagate(0)
        self.info_stress_left.pack(fill=BOTH, expand=1)
        self.f_info_stress_center.config(width=dim.w_dif_4,
                                         height=dim.h_dif_4)
        self.f_info_stress_center.pack_propagate(0)
        self.info_stress_center.config(borderwidth=2, relief="solid")
        self.info_stress_center.pack(fill=BOTH, expand=1)
        self.f_info_stress_right.config(width=dim.width_button_init_fin,
                                        height=dim.height_button_init_fin)
        self.f_info_stress_right.pack_propagate(0)
        self.info_stress_right.pack(fill=BOTH, expand=1)

        self.f_der_stress.config(width=dim.width_button_left_right,
                                 height=dim.height_button_left_right)
        self.f_der_stress.pack_propagate(0)
        self.der_stress.config(text=">",
                               command=lambda: self.mov_img(2, 2))
        self.der_stress.pack(fill=BOTH, expand=1)

        self.f_init_stress.config(width=dim.width_button_init_fin,
                                  height=dim.height_button_init_fin)
        self.f_init_stress.pack_propagate(0)
        self.init_stress.config(text="init",
                                command=lambda: self.set_init_fin(2, 1))
        self.init_stress.pack(fill=BOTH, expand=1)
        self.f_dif_6.config(width=dim.w_dif_4,
                            height=dim.h_dif_4)
        self.f_dif_6.pack_propagate(0)
        self.dif_6.pack(fill=BOTH, expand=1)
        self.f_fin_stress.config(width=dim.width_button_init_fin,
                                 height=dim.height_button_init_fin)
        self.f_fin_stress.pack_propagate(0)
        self.fin_stress.config(text="end",
                               command=lambda: self.set_init_fin(2, 2))
        self.fin_stress.pack(fill=BOTH, expand=1)

        #  --------------------

        self.f_esp_3.config(width=dim.w_espacio_3,
                            height=dim.h_espacio_3)
        self.f_esp_3.pack_propagate(0)
        #  self.esp_3.config(borderwidth=2, relief="solid")
        self.esp_3.pack(fill=BOTH, expand=1)

        #  --------------------

        self.f_dif_7.config(width=dim.w_dif_7,
                            height=dim.h_dif_7)
        self.f_dif_7.pack_propagate(0)
        self.dif_7.pack(fill=BOTH, expand=1)

        self.f_label_slice.config(width=dim.width_label_valor_img,
                                  height=dim.height_label_valor_img)
        self.f_label_slice.pack_propagate(0)
        self.label_slice.config(text=string.txt_valor_slice,
                                anchor=W)
        self.label_slice.pack(fill=BOTH, expand=1)
        self.f_valor_slice.config(width=dim.width_valor_img,
                                  height=dim.height_valor_img)
        self.f_valor_slice.pack_propagate(0)
        self.valor_slice.pack(fill=BOTH, expand=1)

        self.f_dif_8.config(width=dim.w_dif_8,
                            height=dim.h_dif_8)
        self.f_dif_8.pack_propagate(0)
        self.dif_8.pack(fill=BOTH, expand=1)

        self.f_label_ww.config(width=dim.width_label_valor_img,
                               height=dim.height_label_valor_img)
        self.f_label_ww.pack_propagate(0)
        self.label_ww.config(text=string.txt_valor_ww,
                             anchor=W)
        self.label_ww.pack(fill=BOTH, expand=1)
        self.f_valor_ww.config(width=dim.width_valor_img,
                               height=dim.height_valor_img)
        self.f_valor_ww.pack_propagate(0)
        self.valor_ww.config(borderwidth=2,
                             relief="solid")
        self.valor_ww.pack(fill=BOTH, expand=1)

        self.f_label_wl.config(width=dim.width_label_valor_img,
                               height=dim.height_label_valor_img)
        self.f_label_wl.pack_propagate(0)
        self.label_wl.config(text=string.txt_valor_wl,
                             anchor=W)
        self.label_wl.pack(fill=BOTH, expand=1)
        self.f_valor_wl.config(width=dim.width_valor_img,
                               height=dim.height_valor_img)
        self.f_valor_wl.pack_propagate(0)
        self.valor_wl.config(borderwidth=2,
                             relief="solid")
        self.valor_wl.pack(fill=BOTH, expand=1)
        self.f_boton_recarga.config(width=dim.width_valor_img,
                                    height=dim.height_valor_img)
        self.f_boton_recarga.pack_propagate(0)
        self.boton_recarga.config(text="Reload",
                                  command=self.recarga_img)
        self.boton_recarga.pack(fill=BOTH, expand=1)

        self.f_dif_9.config(width=dim.w_dif_9,
                            height=dim.h_dif_9)
        self.f_dif_9.pack_propagate(0)
        self.dif_9.pack(fill=BOTH, expand=1)

        self.f_tabla_resultados.config(width=240,
                                       height=175)
        self.f_tabla_resultados.pack_propagate(0)
        self.init_result_frame()

        #  --------------------


        self.f_esp_4.config(width=dim.w_espacio_4,
                            height=dim.h_espacio_4)
        self.f_esp_4.pack_propagate(0)
        self.esp_4.pack(fill=BOTH, expand=1)

        self.frame_princial.grid_propagate(False)
        self.f_esp_1.grid(row=0, column=0, rowspan=14)

        self.f_dif_1.grid(row=0, column=1, columnspan=5)
        self.f_titulo_img_res.grid(row=1, column=1, columnspan=5)
        self.f_img_res.grid(row=2, column=1, columnspan=5, rowspan=7)
        self.f_dif_2.grid(row=9, column=1, columnspan=5)
        self.f_izq_rest.grid(row=10, column=1, sticky=NSEW)
        self.f_info_res_left.grid(row=10, column=2)
        self.f_info_res_center.grid(row=10, column=3)
        self.f_info_res_right.grid(row=10, column=4)
        self.f_der_rest.grid(row=10, column=5, sticky=NSEW)
        self.f_dif_3.grid(row=11, column=1, columnspan=5)
        self.f_dif_4.grid(row=12, column=3)
        self.f_dif_5.grid(row=13, column=1, columnspan=5)

        self.f_esp_2.grid(row=0, column=6, rowspan=14)

        self.f_titulo_img_stress.grid(row=1, column=7, columnspan=5)
        self.f_img_stress.grid(row=2, column=7, columnspan=5, rowspan=7)
        self.f_izq_stress.grid(row=10, column=7, sticky=NSEW)
        self.f_info_stress_left.grid(row=10, column=8)
        self.f_info_stress_center.grid(row=10, column=9)
        self.f_info_stress_right.grid(row=10, column=10)
        self.f_der_stress.grid(row=10, column=11, sticky=NSEW)
        self.f_dif_6.grid(row=12, column=9)

        self.f_esp_3.grid(row=0, column=12, rowspan=14)
        self.f_dif_7.grid(row=2, column=13, columnspan=2)
        self.f_label_slice.grid(row=3, column=13)
        self.f_valor_slice.grid(row=3, column=14)
        self.f_dif_8.grid(row=4, column=13, columnspan=2)
        self.f_label_ww.grid(row=5, column=13)
        self.f_valor_ww.grid(row=5, column=14)
        self.f_label_wl.grid(row=6, column=13)
        self.f_valor_wl.grid(row=6, column=14)
        self.f_boton_recarga.grid(row=7, column=14)
        # self.f_dif_9.grid(row=8, column=13, columnspan=2)
        self.f_label_div.grid(row=8, column=13)
        self.f_valor_div.grid(row=8, column=14)
        self.f_tabla_resultados.grid(row=10, column=12, columnspan=4, rowspan=13)

        self.f_esp_4.grid(row=0, column=15, rowspan=14)
        pass

    def init_result_frame(self):
        w = 80
        h = 35

        self.f_tabla_resultados.config(relief=SOLID, borderwidth=1)

        frame_esquina = Frame(self.f_tabla_resultados)
        frame_esquina.config(width=w, height=h,
                             relief=SOLID, borderwidth=1)
        frame_esquina.pack_propagate(0)

        frame_title_stress = Frame(self.f_tabla_resultados)
        frame_title_stress.config(width=w, height=h,
                                  relief=SOLID, borderwidth=1)
        frame_title_stress.pack_propagate(0)
        CreateToolTip(frame_title_stress, string.tt_titulo_res_stress)
        self.resultados_stress = Label(frame_title_stress)
        self.resultados_stress.config(text="Stress curve")
        self.resultados_stress.pack(fill=BOTH, expand=1)

        frame_title_rest = Frame(self.f_tabla_resultados)
        frame_title_rest.config(width=w, height=h,
                                relief=SOLID, borderwidth=1)
        frame_title_rest.pack_propagate(0)
        CreateToolTip(frame_title_rest, string.tt_titulo_res_rest)
        self.resultados_rest = Label(frame_title_rest)
        self.resultados_rest.config(text="Rest curve")
        self.resultados_rest.pack(fill=BOTH, expand=1)

        frame_title_area = Frame(self.f_tabla_resultados)
        frame_title_area.config(width=w, height=h,
                                relief=SOLID, borderwidth=1)
        frame_title_area.pack_propagate(0)
        CreateToolTip(frame_title_area, string.tt_res_area)
        self.res_area = Label(frame_title_area)
        self.res_area.config(text="Area")
        self.res_area.pack(fill=BOTH, expand=1)

        frame_title_peak = Frame(self.f_tabla_resultados)
        frame_title_peak.config(width=w, height=h,
                                relief=SOLID, borderwidth=1)
        frame_title_peak.pack_propagate(0)
        CreateToolTip(frame_title_peak, string.tt_res_peak)
        self.res_peak = Label(frame_title_peak)
        self.res_peak.config(text="Peak")
        self.res_peak.pack(fill=BOTH, expand=1)

        frame_title_pend = Frame(self.f_tabla_resultados)
        frame_title_pend.config(width=w, height=h,
                                relief=SOLID, borderwidth=1)
        frame_title_pend.pack_propagate(0)
        CreateToolTip(frame_title_pend, string.tt_res_pendiente)
        self.res_pend = Label(frame_title_pend)
        self.res_pend.config(text="Slope")
        self.res_pend.pack(fill=BOTH, expand=1)

        frame_title_ratio = Frame(self.f_tabla_resultados)
        frame_title_ratio.config(width=w, height=h,
                                 relief=SOLID, borderwidth=1)
        frame_title_ratio.pack_propagate(0)
        CreateToolTip(frame_title_ratio, string.tt_res_coef)
        self.res_ratio = Label(frame_title_ratio)
        self.res_ratio.config(text="Coefficient")
        self.res_ratio.pack(fill=BOTH, expand=1)

        frame_area_rest = Frame(self.f_tabla_resultados)
        frame_area_rest.config(width=w, height=h,
                               relief=SOLID, borderwidth=1)
        frame_area_rest.pack_propagate(0)
        self.area_rest = Label(frame_area_rest)
        self.area_rest.pack(fill=BOTH, expand=1)

        frame_area_stress = Frame(self.f_tabla_resultados)
        frame_area_stress.config(width=w, height=h,
                                 relief=SOLID, borderwidth=1)
        frame_area_stress.pack_propagate(0)
        self.area_stress = Label(frame_area_stress)
        self.area_stress.pack(fill=BOTH, expand=1)

        frame_peak_rest = Frame(self.f_tabla_resultados)
        frame_peak_rest.config(width=w, height=h,
                               relief=SOLID, borderwidth=1)
        frame_peak_rest.pack_propagate(0)
        self.res_peak_rest = Label(frame_peak_rest)
        self.res_peak_rest.pack(fill=BOTH, expand=1)

        frame_peak_stress = Frame(self.f_tabla_resultados)
        frame_peak_stress.config(width=w, height=h,
                                 relief=SOLID, borderwidth=1)
        frame_peak_stress.pack_propagate(0)
        self.res_peak_stress = Label(frame_peak_stress)
        self.res_peak_stress.pack(fill=BOTH, expand=1)

        frame_pend_rest = Frame(self.f_tabla_resultados)
        frame_pend_rest.config(width=w, height=h,
                               relief=SOLID, borderwidth=1)
        frame_pend_rest.pack_propagate(0)
        self.res_pend_rest = Label(frame_pend_rest)
        self.res_pend_rest.pack(fill=BOTH, expand=1)

        frame_pend_stress = Frame(self.f_tabla_resultados)
        frame_pend_stress.config(width=w, height=h,
                                 relief=SOLID, borderwidth=1)
        frame_pend_stress.pack_propagate(0)
        self.res_pend_stress = Label(frame_pend_stress)
        self.res_pend_stress.pack(fill=BOTH, expand=1)

        frame_value_ratio = Frame(self.f_tabla_resultados)
        frame_value_ratio.config(width=int(w*2), height=h,
                                 relief=SOLID, borderwidth=1)
        frame_value_ratio.pack_propagate(0)
        self.res_ratio_value = Label(frame_value_ratio)
        self.res_ratio_value.pack(fill=BOTH, expand=1)

        frame_esquina.grid(row=0, column=0)
        frame_title_rest.grid(row=0, column=1)
        frame_title_stress.grid(row=0, column=2)
        frame_title_area.grid(row=1, column=0)
        frame_area_rest.grid(row=1, column=1)
        frame_area_stress.grid(row=1, column=2)
        frame_title_peak.grid(row=2, column=0)
        frame_peak_rest.grid(row=2, column=1)
        frame_peak_stress.grid(row=2, column=2)
        frame_title_pend.grid(row=3, column=0)
        frame_pend_rest.grid(row=3, column=1)
        frame_pend_stress.grid(row=3, column=2)
        frame_title_ratio.grid(row=4, column=0)
        frame_value_ratio.grid(row=4, column=1, columnspan=2)
        pass

    def iniciar_data(self):

        #  Carga de datos de paciente
        name_paciente = self.parent.patient.s_name
        series_id_patient = self.parent.patient.s_series_id
        series_des_patient = self.parent.patient.s_series_desc
        study_patient = self.parent.patient.s_study_desc
        self.paciente.config(text=(self.paciente.cget("text")+name_paciente))
        self.serie_id.config(text=(self.serie_id.cget("text") + series_id_patient))
        self.serie_desc.config(text=(self.serie_desc.cget("text") + series_des_patient))
        self.study_desc.config(text=(self.study_desc.cget("text") + study_patient))

        #  Carga de datos antiguos
        file1 = open('paq_rest.obj', 'rb')
        self.parent.img_rest = pickle.load(file1)
        file2 = open('paq_stress.obj', 'rb')
        self.parent.img_stress = pickle.load(file2)

        #  Carga de Valores de Imagenes : WW-WL-SLICE
        valores_slice = self.parent.img_stress.get_array_slice()
        self.valor_slice['values'] = valores_slice
        self.valor_slice.bind("<<ComboboxSelected>>", self.sle_cbox)
        self.valor_slice.current(0)
        self.valor_wl.insert(0, str(self.parent.img_stress.contenido[self.slice_select_stress].wl))
        self.valor_ww.insert(0, str(self.parent.img_stress.contenido[self.slice_select_stress].ww))

        #  Carga de Imágenes de Visualización
        self.imprimir_imagenes(tipo=0)
        self.print_img_prediccion(tipo=0)

        #  Carga de funciones para toque de Canvas - Imágenes
        self.img_res.bind("<Button-1>", self.pressed_rest)
        self.img_stress.bind("<Button-1>", self.pressed_stress)

        #  Carga de datos de curvas
        self.curva_print(1)
        pass

    def pressed_rest(self, event):
        if self.habilitar_punto_rest == 1:
            if self.punto_rest_canvas == "":
                self.punto_rest_canvas = self.img_res.create_oval(event.x-3,
                                                                  event.y-3,
                                                                  event.x+3,
                                                                  event.y+3,
                                                                  fill=colors.punto)
            else:
                self.img_res.delete(self.punto_rest_canvas)
                self.punto_rest_canvas = self.img_res.create_oval(event.x-3,
                                                                  event.y-3,
                                                                  event.x+3,
                                                                  event.y+3,
                                                                  fill=colors.punto)
            self.punto_rest = [int(event.x/2), int(event.y/2)]
            if self.punto_rest is not None and self.punto_stress is not None:
                self.boton_export.config(state=NORMAL)
        else:
            pass

    def pressed_stress(self, event):
        if self.habilitar_punto_stress == 1:
            print(self.punto_stress_canvas)
            if self.punto_stress_canvas == "":
                self.punto_stress_canvas = self.img_stress.create_oval(event.x - 3,
                                                                       event.y - 3,
                                                                       event.x + 3,
                                                                       event.y + 3,
                                                                       fill=colors.punto)
            else:
                self.img_stress.delete(self.punto_stress_canvas)
                self.punto_stress_canvas = self.img_stress.create_oval(event.x - 3,
                                                                       event.y - 3,
                                                                       event.x + 3,
                                                                       event.y + 3,
                                                                       fill=colors.punto)
            self.punto_stress = [int(event.x/2), int(event.y/2)]
            if self.punto_rest is not None and self.punto_stress is not None:
                self.boton_export.config(state=NORMAL)
        else:
            pass

    def recarga_img(self):
        ww_obt = self.valor_ww.get()
        wl_obt = self.valor_wl.get()
        self.parent.img_stress.contenido[self.slice_select_stress].wl = int(wl_obt)
        self.parent.img_rest.contenido[self.slice_select_rest].wl = int(wl_obt)
        self.parent.img_stress.contenido[self.slice_select_stress].ww = int(ww_obt)
        self.parent.img_rest.contenido[self.slice_select_rest].ww = int(ww_obt)
        self.imprimir_imagenes(tipo=0)

    def sle_cbox(self, event):
        valor_select = str(self.valor_slice.get()).split(':')[0]
        self.slice_select_rest = int(valor_select) - 1
        self.slice_select_stress = int(valor_select) - 1
        self.imprimir_imagenes(tipo=0)
        self.curva_print(1)
        pass

    def mov_img(self, paquete, direccion):
        """
        Mueve las imágenes de la visualización. Método llamado por los botones de movimiento
        :param paquete: paquete de imagenes a mover
            1: rest
            2: stress
        :param direccion: dirección de movimiento
            1: Left
            2: Right
        :return: llama a imprimir_imagen, segun sea el caso
        """
        if paquete == 1:
            if direccion == 1:
                self.parent.img_rest.contenido[int(self.slice_select_rest)].disminuir_pos()
                self.imprimir_imagenes(tipo=1)
            elif direccion == 2:
                self.parent.img_rest.contenido[int(self.slice_select_rest)].aumentar_pos()
                self.imprimir_imagenes(tipo=1)
        elif paquete == 2:
            if direccion == 1:
                self.parent.img_stress.contenido[int(self.slice_select_stress)].disminuir_pos()
                self.imprimir_imagenes(tipo=2)
            elif direccion == 2:
                self.parent.img_stress.contenido[int(self.slice_select_stress)].aumentar_pos()
                self.imprimir_imagenes(tipo=2)

    def set_init_fin(self, paquete, opcion):
        """
        Set el valor del frame inicial y final de una imagen. Ayuda a descartar valores
        :param paquete: paquete de imagenes a mover
            1: rest
            2: stress
        :param opcion: Opcion a modificar
            1: Init
            2: Fin
        :return:
        """
        if paquete == 1:
            if opcion == 1:
                self.parent.img_rest.contenido[int(self.slice_select_rest)].set_init_current()
            elif opcion == 2:
                self.parent.img_rest.contenido[int(self.slice_select_rest)].set_fin_current()
            self.curva_print(1)
        elif paquete == 2:
            if opcion == 1:
                self.parent.img_stress.contenido[int(self.slice_select_rest)].set_init_current()
            elif opcion == 2:
                self.parent.img_stress.contenido[int(self.slice_select_rest)].set_fin_current()
            self.curva_print(1)

    def imprimir_imagenes(self, tipo=0):
        """
        Muestra las imagenes en la zona del canvas correspondientes. Puede mostrar 1 de las dos imagenes, o actualizar
        ambas, dependiendo de los parámetros ingresados
        :param tipo: Determina cual de las imagenes se desplegará
            0: Ambas Imágenes
            1: Muestra img Rest
            2: Muestra img Stress
        :return: imágenes mostradas en el canvas
        """
        if tipo == 0 or tipo == 1:
            #  Se imprime rest
            x, y = self.parent.img_rest.contenido[self.slice_select_rest].imgs[0].pixel_array.shape
            height = int(dim.height_img_res_screen)
            width = int(height*y/x)
            img, pos_actual, l_button, r_button = self.parent.img_rest.contenido[
                int(self.slice_select_rest)].current_img(width, height)
            img = ImageTk.PhotoImage(img)
            self.imagen_rest = img
            self.img_res.create_image(dim.w_mid_img, dim.h_mid_img, image=self.imagen_rest)
            cantidad_img = self.parent.img_rest.contenido[self.slice_select_rest].cantidad_imgs()
            self.info_res_center.config(text="Image " + str(pos_actual+1) + " of " + str(cantidad_img))
            if l_button:
                self.izq_rest.config(state=NORMAL)
            else:
                self.izq_rest.config(state=DISABLED)
            if r_button:
                self.der_rest.config(state=NORMAL)
            else:
                self.der_rest.config(state=DISABLED)
        if tipo == 0 or tipo == 2:
            #  Se imprime stress
            x, y = self.parent.img_stress.contenido[self.slice_select_stress].imgs[0].pixel_array.shape
            height = int(dim.height_img_res_screen)
            width = int(height * y / x)
            img, pos_actual, l_button, r_button = self.parent.img_stress.contenido[
                int(self.slice_select_stress)].current_img(width, height)
            img = ImageTk.PhotoImage(img)
            self.imagen_stress = img
            self.img_stress.create_image(dim.w_mid_img, dim.h_mid_img, image=self.imagen_stress)
            cantidad_img = self.parent.img_stress.contenido[self.slice_select_stress].cantidad_imgs()
            self.info_stress_center.config(text="Image " + str(pos_actual + 1) + " of " + str(cantidad_img))
            if l_button:
                self.izq_stress.config(state=NORMAL)
            else:
                self.izq_stress.config(state=DISABLED)
            if r_button:
                self.der_stress.config(state=NORMAL)
            else:
                self.der_stress.config(state=DISABLED)
        self.print_img_prediccion(tipo=tipo)

    def exportacion_pdf(self):
        if self.habilitado_para_div == 0 or self.habilitado_para_div == 2:
            self.expotar_data_pdf()
        elif self.habilitado_para_div == 1:
            self.habilitado_para_div = 2
            self.boton_particion.config(text="Complete Section")
            self.boton_export.config(text=string.boton_export)
            self.boton_volver.config(text=string.boton_volver)
            self.b_particion_ttip.change_text(string.tt_particion)
            self.b_export_ttip.change_text(string.tt_export)
            self.b_volver_ttip.change_text(string.tt_volver)
            self.punto_tocado()

    def particionar(self):
        if self.habilitado_para_div == 0:
            print("Partition")
            self.habilitado_para_div = 1
            self.habilitar_punto_stress = 1
            self.habilitar_punto_rest = 1
            self.boton_volver.config(text="Cancel")
            self.boton_export.config(text="Confirm",
                                     state=DISABLED)
            self.boton_particion.config(text="Delete Points")
            self.b_particion_ttip.change_text(string.tt_borrar_puntos)
            self.b_export_ttip.change_text(string.tt_confirmar)
            self.b_volver_ttip.change_text(string.tt_cancelar)
        elif self.habilitado_para_div == 1:
            self.img_stress.delete(self.punto_stress_canvas)
            self.img_res.delete(self.punto_rest_canvas)
            self.punto_stress = None
            self.punto_rest = None
        elif self.habilitado_para_div == 2:
            self.habilitado_para_div = 0
            self.boton_particion.config(text=string.boton_particion)


    def volver_upload(self):
        if self.habilitado_para_div == 0 or self.habilitado_para_div ==  2:
            print("Volver UpLoad")
            self.parent.nuevo_paciente()
        elif self.habilitado_para_div == 1:
            self.habilitado_para_div = 0
            self.habilitar_punto_stress = 0
            self.habilitar_punto_rest = 0
            self.img_res.delete(self.punto_rest_canvas)
            self.img_stress.delete(self.punto_stress_canvas)
            self.boton_particion.config(text=string.boton_particion)
            self.boton_export.config(text=string.boton_export,
                                     state=NORMAL)
            self.boton_volver.config(text=string.boton_volver)
            self.b_particion_ttip.change_text(string.tt_particion)
            self.b_export_ttip.change_text(string.tt_export)
            self.b_volver_ttip.change_text(string.tt_volver)

    def punto_tocado(self):
        self.parent.img_rest.calculo_division(self.punto_rest)
        self.parent.img_stress.calculo_division(self.punto_stress)

        self.f_valor_div.config(width=dim.width_valor_img,
                                height=dim.height_valor_img)
        self.f_valor_div.pack_propagate(0)
        self.f_label_div.config(width=dim.width_label_valor_img,
                                height=dim.height_label_valor_img)
        self.f_label_div.pack_propagate(0)

        self.label_div.config(text="Partition",
                              anchor=W)
        self.label_div.pack(fill=BOTH, expand=1)
        self.valor_div.pack(fill=BOTH, expand=1)
        self.valor_div.config(values=[1, 2, 3, 4])
        self.valor_div.bind("<<ComboboxSelected>>", self.cambio_particion)
        self.valor_div.current(0)
        self.f_dif_9.pack_forget()
        self.f_label_div.grid(row=8, column=13)
        self.f_valor_div.grid(row=8, column=14)
        self.cambio_particion(event=None)
        pass

    def cambio_particion(self, event):
        particion = self.valor_div.get()
        self.subdiv = 'sub' + str(particion)
        self.curva_print(1)


    def curva_print(self, zona):
        """
        Despliega la curva de perfusión de Stress. Dependiendo del caso
        :param zona: Zona que se quiere observar
            1: Sangre
            2: Epicardio
            3: Endocardio
        :return: Curva impresa en el canvas
        """
        init_rest = self.parent.img_rest.contenido[self.slice_select_rest].inicio
        fin_rest = self.parent.img_rest.contenido[self.slice_select_rest].fin
        init_stress = self.parent.img_stress.contenido[self.slice_select_stress].inicio
        fin_stress = self.parent.img_stress.contenido[self.slice_select_stress].fin
        time_r = self.parent.img_rest.contenido[self.slice_select_rest].data_tiempo
        time_rest = range(len(time_r))
        time_s = self.parent.img_stress.contenido[self.slice_select_stress].data_tiempo
        time_stress = range(len(time_s))
        data_rest = []
        data_stress = []
        if zona == 1:
            self.zona_select = 1
            self.button_san.config(state=DISABLED)
            self.button_epi.config(state=NORMAL)
            self.button_end.config(state=NORMAL)
            data_rest = self.parent.img_rest.contenido[self.slice_select_rest].data_sangre[self.subdiv]
            data_stress = self.parent.img_stress.contenido[self.slice_select_stress].data_sangre[self.subdiv]
        elif zona == 2:
            self.zona_select = 2
            self.button_san.config(state=NORMAL)
            self.button_epi.config(state=DISABLED)
            self.button_end.config(state=NORMAL)
            data_rest = self.parent.img_rest.contenido[self.slice_select_rest].data_epicardio[self.subdiv]
            data_stress = self.parent.img_stress.contenido[self.slice_select_stress].data_epicardio[self.subdiv]
        elif zona == 3:
            self.zona_select = 3
            self.button_san.config(state=NORMAL)
            self.button_epi.config(state=NORMAL)
            self.button_end.config(state=DISABLED)
            data_rest = self.parent.img_rest.contenido[self.slice_select_rest].data_endocardio[self.subdiv]
            data_stress = self.parent.img_stress.contenido[self.slice_select_stress].data_endocardio[self.subdiv]
        if self.plot_rest_res is None:
            self.plot_rest_res, = self.plot_rest.plot(time_rest[init_rest:fin_rest],
                                                      data_rest[init_rest:fin_rest])
            self.plot_stress_res, = self.plot_stress.plot(time_stress[init_stress:fin_stress],
                                                          data_stress[init_stress:fin_stress])
            self.canvas_stress.figure.axes[0].xaxis.set_visible(False)
            self.canvas_rest.figure.axes[0].xaxis.set_visible(False)
        else:

            self.plot_rest_res.set_xdata(time_rest[init_rest:fin_rest])
            self.plot_rest_res.set_ydata(data_rest[init_rest:fin_rest])
            self.plot_stress_res.set_xdata(time_stress[init_stress:fin_stress])
            self.plot_stress_res.set_ydata(data_stress[init_stress:fin_stress])


            self.canvas_rest.figure.axes[0].set_ylim(min(data_rest[init_rest:fin_rest]),
                                                     max(data_rest[init_rest:fin_rest]))
            # self.canvas_rest.figure.axes[0].set_xlim(min(time_rest[init_rest:fin_rest]),
            #                                          max(time_rest[init_rest:fin_rest]))
            self.canvas_rest.draw()
            self.canvas_stress.figure.axes[0].set_ylim(min(data_stress[init_stress:fin_stress]),
                                                       max(data_stress[init_stress:fin_stress]))
            # self.canvas_stress.figure.axes[0].set_xlim(min(time_stress[init_stress:fin_stress]),
            #                                            max(time_stress[init_stress:fin_stress]))
            self.canvas_stress.draw()
        self.print_img_prediccion(tipo=0)
        self.rellenar_tabla(time_rest[init_rest:fin_rest],
                            data_rest[init_rest:fin_rest],
                            time_stress[init_stress:fin_stress],
                            data_stress[init_stress:fin_stress])


    def print_img_prediccion(self, tipo):
        """
        Imprime las preducciones, según la zona de interés
        :param tipo: Que tipo de imágen se desplegará
            0: Ambas Imágenes
            1: Muestra img Rest
            2: Muestra img Stress
        :return: Imagen sobre canvas
        """
        zona = self.zona_select
        if tipo == 0 or tipo == 1:
            #  Imprimer Rest
            x, y = self.parent.img_rest.contenido[self.slice_select_rest].imgs[0].pixel_array.shape
            height = int(dim.height_img_res_screen)
            width = int(height * y / x)
            img_rgba = self.get_predict(tipo, zona, width, height)
            img_rgba = Image.frombytes('RGBA', (img_rgba.shape[1], img_rgba.shape[0]), img_rgba.astype('b').tostring())
            img_rgba = ImageTk.PhotoImage(img_rgba)
            self.predict_rest = img_rgba
            self.img_res.create_image(dim.w_mid_img, dim.h_mid_img, image=self.predict_rest)

            pass
        if tipo == 0 or tipo == 2:
            #  Imprimer Stress
            x, y = self.parent.img_stress.contenido[self.slice_select_stress].imgs[0].pixel_array.shape
            height = int(dim.height_img_res_screen)
            width = int(height * y / x)
            img_rgba = self.get_predict(tipo, zona, width, height)
            img_rgba = Image.frombytes('RGBA', (img_rgba.shape[1], img_rgba.shape[0]), img_rgba.astype('b').tostring())
            img_rgba = ImageTk.PhotoImage(img_rgba)
            self.predict_stress = img_rgba
            self.img_stress.create_image(dim.w_mid_img, dim.h_mid_img, image=self.predict_stress)
            pass

    def get_predict(self, tipo, zona, w, h):
        """
        Entrega el array de la imagen en RGBA de la zona que se quiere mostrar
        :param tipo: Que tipo de imágen se desplegará
            0: Ambas Imágenes
            1: Muestra img Rest
            2: Muestra img Stress
        :param zona: Zona que se quiere observar
            1: Sangre
            2: Epicardio
            3: Endocardio
        :param w: Width que se quiere la imagen
        :param h: Height que se quiere la imagen
        :return: Array RGBA
        """
        img_rgba = None
        if tipo == 0 or tipo == 1:
            # Devolver imagen REST
            if self.subdiv == 'total':
                imgs_array = self.parent.img_rest.contenido[
                    int(self.slice_select_rest)].current_predict(0)[zona - 1]
            else:
                parte = int(self.subdiv[len(self.subdiv)-1])
                imgs_array = self.parent.img_rest.contenido[
                    int(self.slice_select_rest)].current_predict(parte)[zona - 1]
            img_rgba = img2rgba(imgs_array, zona, w, h)
        if tipo == 0 or tipo == 2:
            # Devolver imagen STRESS
            if self.subdiv == 'total':
                imgs_array = self.parent.img_stress.contenido[
                    int(self.slice_select_stress)].current_predict(0)[zona - 1]
            else:
                parte = int(self.subdiv[len(self.subdiv) - 1])
                imgs_array = self.parent.img_stress.contenido[
                    int(self.slice_select_stress)].current_predict(parte)[zona - 1]
            img_rgba = img2rgba(imgs_array, zona, w, h)
        return img_rgba

    def rellenar_tabla(self, time_rest, data_rest, time_stress, data_stress):
        self.area_rest.config(text=str(calculo_area_curva(data_rest, time_rest)))
        self.area_stress.config(text=str(calculo_area_curva(data_stress, time_stress)))
        self.res_peak_rest.config(text=str(calculo_maximo(data_rest, time_rest)[0]))
        self.res_peak_stress.config(text=str(calculo_maximo(data_stress, time_stress)[0]))
        self.res_pend_rest.config(text=str(calculo_pendiente(data_rest, time_rest)[0]))
        self.res_pend_stress.config(text=str(calculo_pendiente(data_stress, time_stress)[0]))
        self.res_ratio_value.config(text=str(np.round(calculo_pendiente(data_stress, time_stress)[0]
                                             / calculo_pendiente(data_rest, time_rest)[0], 2)))

    def expotar_data_pdf(self):
        pdf = ExportPDF(self.parent)
        print(pdf.generar_reporte())
        print("Export PDF")
