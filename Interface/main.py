from tkinter import *
from tkinter import filedialog

import pydicom

from model import imagen_model as image

# General Constants
background = "#EDEDEC"
width_sc = 1280
height_sc = 720

# General variables
cantImg = 0
res = []


def process_img(names):
    res_procc = []
    for name in names:
        dataset = pydicom.dcmread(name)
        imagen = image.Imagen(dataset.pixel_array,
                              "",
                              1,
                              dataset[0x29, 0x1008],
                              dataset.StudyDate,
                              dataset.StudyTime,
                              dataset.PatientID,
                              "")
        res_procc.append(imagen)
    return res_procc


def clean_pantalla():
    for widget in frame_datos.winfo_children():
        widget.place_forget()
    for widget in frame_opt.winfo_children():
        widget.place_forget()
    for widget in frame_img.winfo_children():
        widget.place_forget()


def toma_datos():
    clean_pantalla()
    textNoHayElementos.place(x=int(width_sc / 10),
                             y=50,
                             anchor="center")
    botonAgregarImg.place(x=int(width_sc / 10),
                          y=100,
                          anchor="center")


def frame_imagenes():
    clean_pantalla()
    print(len(res))
    print(res[0].id_foto)
    botonResultados.place(x=int(width_sc * 4 / 20),
                          y=height_sc / 10,
                          anchor="center")
    botonSeccionar.place(x=int(width_sc * 8 / 20),
                         y=height_sc / 10,
                         anchor="center")
    botonVolverEmpezar.place(x=int(width_sc * 12 / 20),
                             y=height_sc / 10,
                             anchor="center")


def procesar_img():
    clean_pantalla()
    botonProcesar.place(x=int(width_sc * 4 / 10),
                        y=30,
                        anchor="center")


def seccionar():
    print("SECTIONING")


def resutlados():
    print("RESULTS")


def subir_img():
    global res
    img = filedialog.askopenfilenames(initialdir="/",
                                      title="Select file")
    textCantidadImg['text'] = len(img)
    res = process_img(img)
    frame_lista_img()


def frame_lista_img():
    clean_pantalla()
    textDeseaAnalizar.place(x=width_sc * 4 / 10,
                            y=30,
                            anchor="center")
    textListaCant.place(x=int(width_sc / 10),
                        y=50,
                        anchor="center")
    textCantidadImg.place(x=int(width_sc / 10),
                          y=80,
                          anchor="center")
    botonSiProceso.place(x=width_sc * 4 / 15,
                         y=60,
                         anchor="center")
    botonNoProceso.place(x=width_sc * 8 / 15,
                         y=60,
                         anchor="center")


# Screen Startup
pantalla = Tk()

# Display Configuration
pantalla.title("Myocardial Perfusion Universidad de Chile")
pantalla.resizable(False, False)  # Does not let you resize
pantalla.config(bg=background)

# Frames
frame_datos = Frame(pantalla)
frame_datos.config(bg=background,
                   width=width_sc / 5,
                   height=height_sc,
                   relief="ridge",
                   borderwidth=2,
                   highlightbackground="black")

frame_img = Frame(pantalla)
frame_img.config(bg=background,
                 width=width_sc * 4 / 5,
                 height=height_sc * 4 / 5,
                 relief="ridge",
                 borderwidth=2,
                 highlightbackground="black")

frame_opt = Frame(pantalla)
frame_opt.config(bg=background,
                 width=width_sc * 4 / 5,
                 height=height_sc / 5,
                 relief="ridge",
                 borderwidth=2,
                 highlightbackground="black")

# Frame Toma de Datos
textNoHayElementos = Label(frame_datos,
                           text="Select the images \n you wish to analyze",
                           bg=background,
                           fg="#000000",
                           justify="center")
textNoHayElementos.place(x=int(width_sc / 10),
                         y=50,
                         anchor="center")
botonAgregarImg = Button(frame_datos,
                         text="Upload Images",
                         command=subir_img,
                         bg=background,
                         borderwidth=0,
                         fg=background)
botonAgregarImg.place(x=int(width_sc / 10),
                      y=100,
                      anchor="center")

# Image List Frame
textDeseaAnalizar = Label(frame_opt,
                          text="Do you want to start image analysis?",
                          bg=background,
                          fg="#000000",
                          justify="center")
textListaCant = Label(frame_datos,
                      text="Selected images:",
                      bg=background,
                      fg="#000000",
                      justify="center")
textCantidadImg = Label(frame_datos,
                        text="",
                        bg=background,
                        fg="#000000",
                        justify="center")
botonSiProceso = Button(frame_opt,
                        text="Yes",
                        command=procesar_img,
                        bg=background,
                        borderwidth=0,
                        fg=background)
botonNoProceso = Button(frame_opt,
                        text="No",
                        command=toma_datos,
                        bg=background,
                        borderwidth=0,
                        fg=background)

# Process frame
botonProcesar = Button(frame_opt,
                       text="PASS",
                       command=frame_imagenes,
                       bg=background,
                       borderwidth=0,
                       fg=background)

# Sample Image Frame
botonResultados = Button(frame_opt,
                         text="Results",
                         command=resutlados,
                         bg=background,
                         borderwidth=0,
                         fg=background)
botonSeccionar = Button(frame_opt,
                        text="Generate sections",
                        command=seccionar,
                        bg=background,
                        borderwidth=0,
                        fg=background)
botonVolverEmpezar = Button(frame_opt,
                            text="Other Analysis",
                            command=toma_datos,
                            bg=background,
                            borderwidth=0,
                            fg=background)

frame_datos.grid(row=0, column=0, rowspan=2)
frame_img.grid(row=0, column=1)
frame_opt.grid(row=1, column=1)
pantalla.mainloop()
