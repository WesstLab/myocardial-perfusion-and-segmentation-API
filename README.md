# myocardial-perfusion-and-segmentation-API

API prototype for myocardial magnetic resonance image visualization and segmentation and perfusion curve analysis this prototype has been developed in the Python( 3.6) programming language using tkinter, opencv, scikit, tensorflow and other libraries with the distributor Anaconda. 

# Demo 

# Installation

## Environment installation

Next, we will explain how to install the environment used to use the program, from installing anaconda to implementing the virtual environment with the environment.yml file:

1.	Install the anaconda distributor.
[![](https://assets-cdn.anaconda.com/assets/company/anaconda-logo.png?mtime=20200723150109&focal=none)](https://www.anaconda.com/products/individual#Downloads)

2.	Once installed, you must enter the Anaconda navigator.

3.	go to **"Environments"**, then to **"root"**, press the arrow and press open terminal to create a virtual environment with the following command in the terminal:
```bash
conda env create --file environment.yml
```
With this, the virtual environment is ready to run the application

## Trained model

The trained model is the algorithm that processes the heart images and delivers the results. To integrate the trained model to the application, you must:

1.	Download model from the following link: https://drive.google.com/file/d/1SHUHxrZHm8eR9kIf9HXl9Lh9I7bf799r/view?usp=sharing
2.	Place the model in the **"segmentation"** folder of the project.

With that the application should work and process the images correctly.

# Notes

.To run the application run test.py
.The **"Export results"** button is currently unavailable.
