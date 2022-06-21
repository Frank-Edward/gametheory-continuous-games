#####################################################################
## ECE1657 Project ##################################################
#####################################################################

##imports
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from helper import *
from test import *
from graphs import *
import main


##Iterative Algorithms

def BR(u, u1, u2, J):
    J = J(u1, u2)
    return u[J.argmin()]
    
def BR_play(i1, i2, x, y, J1raw, J2raw, alpha, error):
    x1 = i1
    x2 = i2
    count = 0
    u1, u2= sym.symbols('u1 u2')

    J1 = sym.lambdify([u1,u2], J1raw)
    J2 = sym.lambdify([u1,u2], J2raw)


    while(True):
        x1new = x1 + alpha*(BR(x, x, x2, J1)-x1)
        x2new = x2 + alpha*(BR(y, x1, y, J2)-x2)
        #x1new = BR(x, x, x2, J1)
        #x2new = BR(y, x1, y, J2)
        count += 1
        if (abs(x1new - x1) < error) and (abs(x2new - x2) < error):
            x1, x2 = x1new, x2new
            break
        else:
            x1, x2 = x1new, x2new
        if count>300000:
            return (None, None, None)
    return (x1,x2, count)

def Gradient_play(i1, i2, x, y, J1, J2, alpha, error):
    
    x1 = i1
    x2 = i2
    count = 0
    
    u1, u2 = sym.symbols('u1 u2')
    diff1 = sym.Derivative(J1, u1)
    diff1 = diff1.doit()
    gradJ1 = sym.lambdify([u1,u2], diff1)
    diff2 = sym.Derivative(J2, u2)
    diff2 = diff2.doit()
    gradJ2 = sym.lambdify([u1,u2], diff2)
    
    while(True):
        x1new = x1 - alpha*gradJ1(x1, x2)
        x2new = x2 - alpha*gradJ2(x1, x2)
        count += 1
        #print(x1new, x2new)
        if (abs(x1new - x1) < error) and (abs(x2new - x2) < error):
            x1, x2 = x1new, x2new
            break
        else:
            x1, x2 = max(x1new, x[0]), max(x2new,y[0])
            x1, x2 = min(x1, x[-1]), min(x2,y[-1])
        if count>300000:
            return (None, None, None)
    return (x1,x2, count)

def BR_play_values(i1, i2, x, y, J1raw, J2raw, alpha, error):
    x1 = i1
    x2 = i2
    count = 0
    u1, u2= sym.symbols('u1 u2')

    J1 = sym.lambdify([u1,u2], J1raw)
    J2 = sym.lambdify([u1,u2], J2raw)

    x1_matrix = np.array([])
    x2_matrix = np.array([])
    while(True):
        
        x1new = x1 + alpha*(BR(x, x, x2, J1)-x1)
        x2new = x2 + alpha*(BR(y, x1, y, J2)-x2)
        #print(x1new, x2new)
        count += 1
        if (abs(x1new - x1) < error) and (abs(x2new - x2) < error):
            x1, x2 = x1new, x2new
            x1_matrix = np.append(x1_matrix, [x1])
            x2_matrix = np.append(x2_matrix, [x2])
            break
        else:
            x1, x2 = x1new, x2new
            x1_matrix = np.append(x1_matrix, [x1])
            x2_matrix = np.append(x2_matrix, [x2])
        if count>100000:
            return (x1_matrix,x2_matrix, count)
    return (x1_matrix,x2_matrix, count)

def Gradient_play_values(i1, i2, x, y, J1, J2, alpha, error):
    
    x1 = i1
    x2 = i2
    count = 0
    
    u1, u2 = sym.symbols('u1 u2')
    diff1 = sym.Derivative(J1, u1)
    diff1 = diff1.doit()
    gradJ1 = sym.lambdify([u1,u2], diff1)
    diff2 = sym.Derivative(J2, u2)
    diff2 = diff2.doit()
    gradJ2 = sym.lambdify([u1,u2], diff2)
    x1_matrix = np.array([])
    x2_matrix = np.array([])
    while(True):
        x1new = x1 - alpha*gradJ1(x1, x2)
        x2new = x2 - alpha*gradJ2(x1, x2)
        #print(x1new, x2new)
        count += 1
        if (abs(x1new - x1) < error) and (abs(x2new - x2) < error):
            x1, x2 = x1new, x2new
            x1_matrix = np.append(x1_matrix, [x1])
            x2_matrix = np.append(x2_matrix, [x2])
            break
        else:
            x1, x2 = x1new, x2new
            #x1, x2 = max(x1new, x[0]), max(x2new,y[0])
            #x1, x2 = min(x1, x[-1]), min(x2,y[-1])
            x1_matrix = np.append(x1_matrix, [x1])
            x2_matrix = np.append(x2_matrix, [x2])
        if count>100000:
            return (x1_matrix,x2_matrix, count)
    return (x1_matrix,x2_matrix, count)


##Testing

if __name__ == '__main__':
    
    J1raw, J2raw = example1()
    x = np.linspace(0,1,10000)
    y = np.linspace(0,1,10000)
    print("br2",BR_play(0, 0, x, y, J1raw, J2raw, 0.01, 0.000001))
    print("gp2", Gradient_play(0.5, 0.5, x, y, J1raw, J2raw, 0.01, 0.000001))
    
    J1raw, J2raw = example2()
    x = np.linspace(0.2,0.8,10000)
    y = np.linspace(0.2,0.8,10000)
    print("br2",BR_play(0.5, 0.5, x, y, J1raw, J2raw, 0.01, 0.000001))
    print("gp2", Gradient_play(0.5, 0.5, x, y, J1raw, J2raw, 0.01, 0.000001))
    
    J1raw, J2raw = example3()
    x = np.linspace(0.2,0.8,10000)
    y = np.linspace(0.2,0.8,10000)
    print("br3",BR_play(0.5, 0.5, x, y, J1raw, J2raw, 0.01, 0.000001))
    print("gp3", Gradient_play(0.5, 0.5, x, y, J1raw, J2raw, 0.01, 0.000001))
    
    J1raw, J2raw = example4()
    x = np.linspace(0.2,0.8,10000)
    y = np.linspace(0.2,0.8,10000)
    print("br4",BR_play(0.5, 0.5, x, y, J1raw, J2raw, 0.01, 0.000001))
    print("gp4", Gradient_play(0.5, 0.5, x, y, J1raw, J2raw, 0.01, 0.000001))
    
    '''
    print("### Special Graph ###")
    J1raw2, J2raw2 = example2()
    J1raw3, J2raw3 = example3()
    J1raw4, J2raw4 = example4()
    x = np.linspace(0.2,0.8,1000)
    y = np.linspace(0.2,0.8,1000)    
    J1mat = [J1raw2,J1raw3,J1raw4]
    J2mat = [J2raw2,J2raw3,J2raw4]
    special_u_convergence_vs_iterations(0.5,0.5,x,y,J1mat,J2mat, 0.01, 0.00001)
    '''
    '''
    print("### Starting Example 1 ###")
    J1raw, J2raw = example1()
    example = 1
    x = np.linspace(0,1,1000)
    y = np.linspace(0,1,1000)
    convergence_vs_alpha(0.5,0.5,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001, example)
    convergence_vs_error(0.5,0.5,x,y,J1raw,J2raw,0.01,[0.0000001,0.0000005,0.000001,0.000005,0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],example)
    u_convergence_vs_iterations(0.5,0.5,x,y,J1raw,J2raw, 0.01, 0.00001,example) 
    J_convergence_vs_iterations(0.5,0.5,x,y,J1raw,J2raw, 0.01, 0.000001,example)
    iterations_vs_alpha(0.5,0.5,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001,example)
    iterations_vs_error(0.5,0.5,x,y,J1raw,J2raw,0.01,[0.0000001,0.0000005, 0.000001,0.000005,0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],example)
    time_vs_alpha(0.5,0.5,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001,example)
    time_vs_error(0.5,0.5,x,y,J1raw,J2raw, 0.01 , [0.0000001,0.0000005, 0.000001,0.000005, 0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],example)
    iterations_vs_ics(np.linspace(0.2,0.8,101),np.linspace(0.2,0.8,101),x,y,J1raw,J2raw,0.01, 0.00001,1)
    time_vs_ics(np.linspace(0.2,0.8,101),np.linspace(0.2,0.8,101),x,y,J1raw,J2raw,0.01, 0.00001,1)
    '''
    '''
    print("### Starting Example 2 ###")
    J1raw, J2raw = example2()
    x = np.linspace(0.2,0.8,1000)
    y = np.linspace(0.2,0.8,1000)    
    iterations_vs_ics(np.linspace(0.2,0.8,101),np.linspace(0.2,0.8,101),x,y,J1raw,J2raw,0.01, 0.00001,2)
    time_vs_ics(np.linspace(0.2,0.8,101),np.linspace(0.2,0.8,101),x,y,J1raw,J2raw,0.01, 0.00001,2)
    
    J1raw, J2raw = example3()
    x = np.linspace(0.2,0.8,1000)
    y = np.linspace(0.2,0.8,1000)    
    iterations_vs_ics(np.linspace(0.2,0.8,101),np.linspace(0.2,0.8,101),x,y,J1raw,J2raw,0.01, 0.00001,3)
    time_vs_ics(np.linspace(0.2,0.8,101),np.linspace(0.2,0.8,101),x,y,J1raw,J2raw,0.01, 0.00001,3)
    
    J1raw, J2raw = example4()
    x = np.linspace(0.2,0.8,1000)
    y = np.linspace(0.2,0.8,1000)    
    iterations_vs_ics(np.linspace(0.2,0.8,101),np.linspace(0.2,0.8,101),x,y,J1raw,J2raw,0.01, 0.00001,4)
    time_vs_ics(np.linspace(0.2,0.8,101),np.linspace(0.2,0.8,101),x,y,J1raw,J2raw,0.01, 0.00001,4)
    '''
    '''
    print("### Starting Example 2 ###")
    J1raw, J2raw = example2()
    example = 2
    x = np.linspace(0.2,0.8,1000)
    y = np.linspace(0.2,0.8,1000)
    convergence_vs_alpha(0.5,0.5,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001, example)
    convergence_vs_error(0.5,0.5,x,y,J1raw,J2raw,0.01,[0.0000001,0.0000005,0.000001,0.000005,0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],example)
    u_convergence_vs_iterations(0.5,0.5,x,y,J1raw,J2raw, 0.01, 0.00001,example) 
    J_convergence_vs_iterations(0.5,0.5,x,y,J1raw,J2raw, 0.01, 0.000001,example)
    iterations_vs_alpha(0.5,0.5,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001,example)
    iterations_vs_error(0.5,0.5,x,y,J1raw,J2raw,0.01,[0.0000001,0.0000005, 0.000001,0.000005,0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],example)
    time_vs_alpha(0.5,0.5,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001,example)
    time_vs_error(0.5,0.5,x,y,J1raw,J2raw, 0.01 , [0.0000001,0.0000005, 0.000001,0.000005, 0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],example)
    
    print("### Starting Example 3 ###")
    J1raw, J2raw = example3()
    example = 3
    x = np.linspace(0.2,0.8,1000)
    y = np.linspace(0.2,0.8,1000)
    convergence_vs_alpha(0.5,0.5,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001, example)
    convergence_vs_error(0.5,0.5,x,y,J1raw,J2raw,0.01,[0.0000001,0.0000005,0.000001,0.000005,0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],example)
    u_convergence_vs_iterations(0.5,0.5,x,y,J1raw,J2raw, 0.01, 0.00001,example) 
    J_convergence_vs_iterations(0.5,0.5,x,y,J1raw,J2raw, 0.01, 0.000001,example)
    iterations_vs_alpha(0.5,0.5,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001,example)
    iterations_vs_error(0.5,0.5,x,y,J1raw,J2raw,0.01,[0.0000001,0.0000005, 0.000001,0.000005,0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],example)
    time_vs_alpha(0.5,0.5,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001,example)
    time_vs_error(0.5,0.5,x,y,J1raw,J2raw, 0.01 , [0.0000001,0.0000005, 0.000001,0.000005, 0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],example)
    
    print("### Starting Example 4 ###")
    J1raw, J2raw = example4()
    example = 4
    x = np.linspace(0.2,0.8,1000)
    y = np.linspace(0.2,0.8,1000)
    convergence_vs_alpha(0.5,0.5,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001, example)
    convergence_vs_error(0.5,0.5,x,y,J1raw,J2raw,0.01,[0.0000001,0.0000005,0.000001,0.000005,0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],example)
    u_convergence_vs_iterations(0.5,0.5,x,y,J1raw,J2raw, 0.01, 0.00001,example) 
    J_convergence_vs_iterations(0.5,0.5,x,y,J1raw,J2raw, 0.01, 0.000001,example)
    iterations_vs_alpha(0.5,0.5,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001,example)
    iterations_vs_error(0.5,0.5,x,y,J1raw,J2raw,0.01,[0.0000001,0.0000005, 0.000001,0.000005,0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],example)
    time_vs_alpha(0.5,0.5,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001,example)
    time_vs_error(0.5,0.5,x,y,J1raw,J2raw, 0.01 , [0.0000001,0.0000005, 0.000001,0.000005, 0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],example)
    '''
    
    
    
    '''
    
    
    J1raw, J2raw = example3()
    print(BR_play(0.5, 0.5, x, y, J1raw, J2raw, 0.00001, 0.0000001))
    print(Gradient_play(0.5, 0.5, x, y, J1raw, J2raw, 0.00001, 0.0000001))
    
    J1raw, J2raw = example4()
    print(BR_play(0.5, 0.5, x, y, J1raw, J2raw,  0.00001, 0.0000001))
    print(Gradient_play(0.5, 0.5, x, y, J1raw, J2raw,  0.00001, 0.0000001))
    
    count=0
    J1raw, J2raw = example4()
    for alpha in np.linspace(0.000001,0.01,20):
        count+=1
        example = "test-z{}".format(count) #"test-{}".format(str(alpha).replace(".", "-"))
        u_convergence_vs_iterations(0.5,0.5,x,y,J1raw,J2raw, alpha, alpha/100,example)
        print("done", example)
    '''
    '''
    J1raw, J2raw = example1()
    x = np.linspace(0,1,1000)
    y = np.linspace(0,1,1000)
    print(BR_play(0, 0, x, y, J1raw, J2raw, 0.01, 0.000001))
    print(Gradient_play(0, 0, x, y, J1raw, J2raw, 0.01, 0.000001))
    convergence_vs_alpha(0,0,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001,1)
    convergence_vs_error(0,0,x,y,J1raw,J2raw, 0.01 , [0.0000001,0.0000005, 0.000001,0.000005, 0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],1)
    u_convergence_vs_iterations(0,0,x,y,J1raw,J2raw, 0.01, 0.00001,1) 
    J_convergence_vs_iterations(0,0,x,y,J1raw,J2raw, 0.01, 0.000001,1)
    iterations_vs_alpha(0,0,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001,1)
    iterations_vs_error(0,0,x,y,J1raw,J2raw, 0.01 , [0.0000001,0.0000005, 0.000001,0.000005, 0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],1)
    time_vs_alpha(0,0,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001,1)
    time_vs_error(0,0,x,y,J1raw,J2raw, 0.01 , [0.0000001,0.0000005, 0.000001,0.000005, 0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009],1)
    '''
    
    
    
    '''
    J1raw, J2raw = example1()
    x = np.linspace(0,1,1000)
    y = np.linspace(0,1,1000)
    #print(BR_play(0, 0, x, y, J1raw, J2raw, 0.01, 0.000001))
    #print(Gradient_play(0, 0, x, y, J1raw, J2raw, 0.01, 0.000001))
    #Gradient_play(1, 1, x, y, J1raw, J2raw, 0.001, 0.0001)
    
    ##convergence vs alpha
    convergence_vs_alpha(0,0,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001)
    
    ##convergence vs error
    convergence_vs_error(0,0,x,y,J1raw,J2raw, 0.01 , [0.0000001,0.0000005, 0.000001,0.000005, 0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009])

    ##u convergence vs iterations
    u_convergence_vs_iterations(0,0,x,y,J1raw,J2raw, 0.01, 0.00001) 
    
    ##J convergence vs iterations
    J_convergence_vs_iterations(0,0,x,y,J1raw,J2raw, 0.01, 0.000001)
    
    ##iterations vs alpha
    iterations_vs_alpha(0,0,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001)
    
    ##iterations vs errorJ2raw
    iterations_vs_error(0,0,x,y,J1raw,J2raw, 0.01 , [0.0000001,0.0000005, 0.000001,0.000005, 0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009])
    
    ##time vs alpha
    time_vs_alpha(0,0,x,y,J1raw,J2raw, np.linspace(0.0001,0.03,1000) , 0.000001)
    
    ##time vs errorJ2raw
    time_vs_error(0,0,x,y,J1raw,J2raw, 0.01 , [0.0000001,0.0000005, 0.000001,0.000005, 0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009])
    '''
    ##iterations vs ics
    #iterations_vs_ics(np.linspace(0,1,101),np.linspace(0,1,101),x,y,J1raw,J2raw,0.01, 0.00001)
    
    ##iterations vs ics
    #time_vs_ics(np.linspace(0,1,101),np.linspace(0,1,101),x,y,J1raw,J2raw,0.01, 0.00001)
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    print(test)
    print(x)
    '''
    '''x = np.linspace(0,1,1000)
    y = np.linspace(0,1,1000)

    best_response_result = BR_play(0, 0, x, y, J1, J2, 0.01, 0.00001)
    gradient_response_result = Gradient_play(0, 0, x, y, gradJ1, gradJ2, 0.01, 0.00001)
    print(test)
    
    
    
    alpha = np.linspace(0.0001, 0.01, 100)
    alpha_test_count(0, 0, x, y, J1, J2, gradJ1, gradJ2, alpha, 0.00001)'''

'''
y = np.linspace(0,1,10)
ax = plt.gca()
for i in y:
    x = np.linspace(0,1,1000)
    J = J1(x,i)
    B = BR(x, x, i, J1)
    vline_color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(x,J, color =vline_color)
    plt.axvline(x=B, color=vline_color)
plt.show()

'''
'''
p = numpy.poly1d([1, 0, 1])
>>> print p
   2
1 x + 1
>>> q = p.deriv()
>>> print q
2 x
>>> q(5)
10

    sym.init_printing()
    x = sym.symbols("x")
    f = x**2
    dx = sym.Derivative(f)
    dx = dx.doit()
'''