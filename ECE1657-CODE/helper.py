import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from helper import *
from test import *
from graphs import *
import main


## Cost functions
def example1():
    u1, u2= sym.symbols('u1 u2')
    J1 = 2*(u1)**2 - 2*(u1) - u1*u2
    J2 = (u2)**2 - (1/2)*u2 - u1*u2
    return (J1, J2)

def example2():
    u1, u2= sym.symbols('u1 u2')
    J1 = (2)*(u1)**2 - (1/4)*(u1) - u1*u2
    J2 = (2/3)*(u2)**2 - (3/4)*(u2) - u2*u1
    return (J1, J2)

def example3():
    u1, u2= sym.symbols('u1 u2')
    J1 = (2)*(u1)**3 + (3/8)*(u1) - u1*u2
    J2 = (2/3)*(u2)**3 - (7/8)*(u2) - u2*u1
    return (J1, J2)
    
def example4():
    u1, u2= sym.symbols('u1 u2')
    J1 = (2)*(u1)**4 + (5/8)*(u1) - u1*u2
    J2 = (2/3)*(u2)**4 - (7/8)*(u2) - u2*u1
    return (J1, J2)

def example5():
    u1, u2= sym.symbols('u1 u2')
    J1 = 2*(u1)**5 - 2*(u1) - u1*u2
    J2 = (u2)**5 - (1/2)*u2 - u1*u2
    return (J1, J2)

'''

#additional functions not yet tested

def example6():
    u1, u2= sym.symbols('u1 u2')
    J1 = (-1/2)*(u2)**2 +2*(u1)**2 +2*u1*u2 - (7/2)*(u1) - (5/4)*(u2)
    J2 = (1/2)*(u2)**2 -2*(u1)**2 -2*u1*u2 + (7/2)*(u1) + (5/4)*(u2)
    return (J1, J2)
    
def example7():
    u1, u2= sym.symbols('u1 u2')
    J1 = 100*(u1)**2 - 2*(u1) - u1*u2
    J2 = 100*(u2)**2 - (1/2)*u2 - u1*u2
    return (J1, J2)

def example8():
    u1, u2= sym.symbols('u1 u2')
    J1 = (1/100)*(u1)**2 - 2*(u1) - u1*u2
    J2 = (1/100)*(u2)**2 - (1/2)*u2 - u1*u2
    return (J1, J2)
'''

