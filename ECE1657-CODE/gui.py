import tkinter as tk
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from helper import *
from test import *
from graphs import *
import main
from sympy.parsing.sympy_parser import (parse_expr,standard_transformations, implicit_multiplication_application)
from sympy import sympify


def p():
    """Convert the value for Fahrenheit to Celsius and insert the
    result into lbl_result.
    """
    
    i1= ent_i1.get()
    i2= ent_i2.get()
    xstart = ent_xstart.get()
    xend = ent_xend.get()
    ystart = ent_ystart.get() 
    yend = ent_yend.get()
    J1= ent_J1.get()
    J2=ent_J2.get()
    alpha = ent_alpha.get()
    error = ent_error.get()
    
    J1raw =  sym.sympify(J1)
    J2raw =  sym.sympify(J2)

    #J2raw =  parse_expr(J2, transformations=(standard_transformations + (implicit_multiplication_application,)))

    br = main.BR_play(float(i1), float(i2), np.linspace(int(xstart),int(xend), 1000), \
    np.linspace(int(ystart),int(yend), 1000), J1raw, J2raw, float(alpha), float(error))
    
    gp = main.Gradient_play(float(i1), float(i2), np.linspace(float(xstart),float(xend), 1000), \
    np.linspace(float(ystart),float(yend), 1000), J1raw, J2raw, float(alpha), float(error))
    
    lbl_br_res["text"] = "{}".format(br[2])
    lbl_gp_res["text"] = "{}".format(gp[2])

window = tk.Tk()
window.title("Convex Continuous Game Solver")

frm_entry = tk.Frame(master=window)

frm_entry = tk.Frame(master=window)

ent_i1 = tk.Entry(master=frm_entry, width=15)
lbl_i1 = tk.Label(master=frm_entry, text="i1")

ent_i2 = tk.Entry(master=frm_entry, width=15)
lbl_i2 = tk.Label(master=frm_entry, text="i2")

ent_xstart = tk.Entry(master=frm_entry, width=5)
lbl_x = tk.Label(master=frm_entry, text=" <= x <= ")
ent_xend = tk.Entry(master=frm_entry, width=5)

ent_ystart = tk.Entry(master=frm_entry, width=5)
lbl_y = tk.Label(master=frm_entry, text=" <= y <=")
ent_yend = tk.Entry(master=frm_entry, width=5)

ent_J1 = tk.Entry(master=frm_entry, width=15)
lbl_J1 = tk.Label(master=frm_entry, text="J1")

ent_J2 = tk.Entry(master=frm_entry, width=15)
lbl_J2 = tk.Label(master=frm_entry, text="J2")

ent_alpha = tk.Entry(master=frm_entry, width=15)
lbl_alpha = tk.Label(master=frm_entry, text="\u03B1")

ent_error = tk.Entry(master=frm_entry, width=15)
lbl_error = tk.Label(master=frm_entry, text="e")


lbl_i1.grid(row=0, column=0, sticky="w")
ent_i1.grid(row=0, column=1, sticky="e")

lbl_i2.grid(row=1, column=0, sticky="w")
ent_i2.grid(row=1, column=1, sticky="e")

ent_xstart.grid(row=2, column=0, sticky="e")
lbl_x.grid(row=2, column=1, sticky="w")
ent_xend.grid(row=2, column=1, sticky="e")

ent_ystart.grid(row=3, column=0, sticky="e")
lbl_y.grid(row=3, column=1, sticky="w")
ent_yend.grid(row=3, column=1, sticky="e")

lbl_J1.grid(row=4, column=0, sticky="w")
ent_J1.grid(row=4, column=1, sticky="e")

lbl_J2.grid(row=5, column=0, sticky="w")
ent_J2.grid(row=5, column=1, sticky="e")

lbl_alpha.grid(row=6, column=0, sticky="w")
ent_alpha.grid(row=6, column=1, sticky="e")

lbl_error.grid(row=7, column=0, sticky="w")
ent_error.grid(row=7, column=1, sticky="e")


btn_convert = tk.Button(
    master=window,
    text="Compute u*",
    command=p  # <--- Add this line
)

lbl_br_res = tk.Label(master=window, text="BR")
lbl_gp_res = tk.Label(master=window, text="GP")

frm_entry.grid(row=0, column=0, padx=10)
btn_convert.grid(row=0, column=1, pady=10)
lbl_br_res.grid(row=0, column=2, padx=10)
lbl_gp_res.grid(row=1, column=2, padx=10)

window.mainloop()
