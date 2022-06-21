import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from helper import *
from test import *
from graphs import *
import main

u1, u2= sym.symbols('u1 u2')
J1 = 2*(u1)**2 - 2*(u1) - u1*u2
J2 = (u2)**2 - (1/2)*u2 - u1*u2
diff = sym.Derivative(J1, u1)
exact = diff.doit()
J1real = sym.lambdify([u1,u2], J1)
test = "this is a test"
#print(J1real(1,1))



'''
-functions to code
--> number of iterations per alpha v
--> number of iterations per error v
--> total computation time per alpha v 
--> total computation time per error v

--> convergence of u1* and u2* for br and gr v
--> convergence of NE for br and gr v

--> diff initial conditions, different lambdas v <-- NOT DIFFERENT LAMBDAS YET


--> show limitations of gradient, ie a sharp function, vs a smoother one,
--> functions need to be strictly convex

--> make ui
--> make gradient decent decrease alpha if stuck / make modified version
--> test BR with large alphas to compare with gradient descent
--> maybe test different polynomial orders (make sure they are convex on the correct spots!!!

--> maybe have live drwaing of thing
--> note, if things get out of hand, numpy gives error cus of alpha, tri alpha >.45
--> can comment on accuracy
-->--< br has issues with converging faster, talk about stop condition and choice
-->--< talk about how alpha has to be greater than the error, or else stop condition acheived
--> numerically, br limited by choice of x maybe???
--> talk about numerical issues with implementing gradient descent, and how that may af
--> some issue with initial conditions, same shit as before

--> interesting to study, trying to take something continuous and adapt for a discrete program
-->--> vectorization, ie matlab
'''

##Iterations per alpha



## Notes
#need to make the update rule for this
#need to make the gradient descent

'''
final plt lists:
- alpha over time
- have NE convergence / minimixations
- 
'''