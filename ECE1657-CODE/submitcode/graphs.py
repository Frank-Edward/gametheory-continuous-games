#####################################################################
## Graphs ###########################################################
#####################################################################
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from helper import *
from test import *
from graphs import *
import main


def convergence_vs_alpha(x1,x2,x,y,J1,J2,alpha_matrix, error,example):
    brconv1 = np.array([])
    gpconv1 = np.array([])
    brconv2 = np.array([])
    gpconv2 = np.array([])
    for alpha in alpha_matrix:
        br = main.BR_play(x1, x2, x, y, J1, J2, alpha, error)
        gp = main.Gradient_play(x1, x2, x, y, J1, J2, alpha, error)
        
        brconv1 = np.append(brconv1, [br[0]])
        brconv2 = np.append(brconv2, [br[1]])
        gpconv1 = np.append(gpconv1, [gp[0]])
        gpconv2 = np.append(gpconv2, [gp[1]])

        print("Done {}".format(alpha))
    plt.figure()
    plt.plot(alpha_matrix,brconv1, color = 'b', ls = '-')
    plt.plot(alpha_matrix,brconv2, color = 'b', ls = '--')
    plt.plot(alpha_matrix,gpconv1, color = 'r', ls = '-')
    plt.plot(alpha_matrix,gpconv2, color = 'r', ls = '--')
    plt.xlabel("Learning Rate (\u03B1)")
    plt.ylabel("u")
    plt.legend(["u1 Best-Response Play","u2 Best-Response Play", "u1 Gradient Play","u2 Gradient Play"])
    plt.title("Convergence for u vs Learning Rate (\u03B1) for Example {}".format(example))
    plt.savefig("{}-1-cva".format(example),  dpi = 300)
    plt.close()
    #plt.show()
    
def convergence_vs_error(x1,x2,x,y,J1,J2,alpha, error_matrix, example):
    brconv1 = np.array([])
    gpconv1 = np.array([])
    brconv2 = np.array([])
    gpconv2 = np.array([])
    for error in error_matrix:
        br = main.BR_play(x1, x2, x, y, J1, J2, alpha, error)
        gp = main.Gradient_play(x1, x2, x, y, J1, J2, alpha, error)
        
        brconv1 = np.append(brconv1, [br[0]])
        brconv2 = np.append(brconv2, [br[1]])
        gpconv1 = np.append(gpconv1, [gp[0]])
        gpconv2 = np.append(gpconv2, [gp[1]])

        print("Done {}".format(alpha))
    plt.figure()
    plt.plot(error_matrix,brconv1, color = 'b', ls = '-')
    plt.plot(error_matrix,brconv2, color = 'b', ls = '--')
    plt.plot(error_matrix,gpconv1, color = 'r', ls = '-')
    plt.plot(error_matrix,gpconv2, color = 'r', ls = '--')
    plt.xlabel("Convergence Error")
    plt.ylabel("u")
    plt.legend(["u1 Best-Response Play","u2 Best-Response Play", "u1 Gradient Play","u2 Gradient Play"])
    plt.title("Convergence for u vs Convergence Error for Example {}".format(example))
    plt.savefig("{}-2-cve".format(example),  dpi = 300)
    plt.close()
    #plt.show()
    
def u_convergence_vs_iterations(x1,x2,x,y,J1,J2,alpha, error,example):
    brcount = np.array([])
    gpcount = np.array([])
    br = main.BR_play_values(x1, x2, x, y, J1, J2, alpha, error)
    gp = main.Gradient_play_values(x1, x2, x, y, J1, J2, alpha, error)
    if br[2] == gp[2]:
        brplot1 = br[0]
        brplot2 = br[1]
        gpplot1 = gp[0]
        gpplot2 = gp[1]
    
    elif br[2]< gp[2]:
        brplot1 = np.append(br[0] , br[0][-1]*np.ones(gp[2]-br[2]))
        brplot2 = np.append(br[1] , br[1][-1]*np.ones(gp[2]-br[2]))
        gpplot1 = gp[0]
        gpplot2 = gp[1]
    else:
        brplot1 = br[0]
        brplot2 = br[1]
        gpplot1 = np.append(gp[0] , gp[0][-1]*np.ones(br[2]-gp[2]))
        gpplot2 = np.append(gp[1] , gp[1][-1]*np.ones(br[2]-gp[2]))
    plt.figure()
    plt.plot(brplot1, color = 'b', ls = '-')
    plt.plot(brplot2, color = 'b', ls = '--')
    plt.plot(gpplot1, color = 'r', ls = '-')
    plt.plot(gpplot2, color = 'r', ls = '--')
    plt.xlabel("Iteration")
    plt.ylabel("u")
    plt.legend(["u1 Best-Response Play","u2 Best-Response Play", "u1 Gradient Play","u2 Gradient Play"])
    plt.title("Convergence for u vs Iteration number for Example {}".format(example))
    plt.savefig("{}-3-ucvi".format(example), dpi=300)
    plt.close()
    #plt.show()
    
def J_convergence_vs_iterations(x1,x2,x,y,J1,J2,alpha, error,example):
    brcount = np.array([])
    gpcount = np.array([])
    br = main.BR_play_values(x1, x2, x, y, J1, J2, alpha, error)
    gp = main.Gradient_play_values(x1, x2, x, y, J1, J2, alpha, error)
    if br[2] == gp[2]:
        brplot1 = br[0]
        brplot2 = br[1]
        gpplot1 = gp[0]
        gpplot2 = gp[1]
    
    elif br[2]< gp[2]:
        brplot1 = np.append(br[0] , br[0][-1]*np.ones(gp[2]-br[2]))
        brplot2 = np.append(br[1] , br[1][-1]*np.ones(gp[2]-br[2]))
        gpplot1 = gp[0]
        gpplot2 = gp[1]
    else:
        brplot1 = br[0]
        brplot2 = br[1]
        gpplot1 = np.append(gp[0] , gp[0][-1]*np.ones(br[2]-gp[2]))
        gpplot2 = np.append(gp[1] , gp[1][-1]*np.ones(br[2]-gp[2]))
    
    u1, u2= sym.symbols('u1 u2')

    J1func = sym.lambdify([u1,u2], J1)
    J2func = sym.lambdify([u1,u2], J2)
    brJ1 = J1func(brplot1,brplot2)
    brJ2 = J2func(brplot1,brplot2)
    gpJ1 = J1func(gpplot1,gpplot2)
    gpJ2 = J2func(gpplot1,gpplot2)
    plt.figure()
    plt.plot(brJ1, color = 'b', ls = '-')
    plt.plot(brJ2, color = 'b', ls = '--')
    plt.plot(gpJ1, color = 'r', ls = '-')
    plt.plot(gpJ2, color = 'r', ls = '--')
    plt.xlabel("Iteration")
    plt.ylabel("J")
    plt.legend(["J1 Best-Response Play","J2 Best-Response Play", "J1 Gradient Play","J2 Gradient Play"])
    plt.title("Convergence for J vs Iteration number for Example {}".format(example))
    plt.savefig("{}-4-jcvi".format(example), dpi = 300)
    plt.close()
    #plt.show()

def iterations_vs_alpha(x1,x2,x,y,J1,J2,alpha_matrix, error,example):
    brcount = np.array([])
    gpcount = np.array([])
    for alpha in alpha_matrix:
        br = main.BR_play(x1, x2, x, y, J1, J2, alpha, error)
        gp = main.Gradient_play(x1, x2, x, y, J1, J2, alpha, error)
        
        brcount = np.append(brcount, [br[2]])
        gpcount = np.append(gpcount, [gp[2]])
        
        print("Done {}".format(alpha))
    plt.figure()
    plt.plot(alpha_matrix, brcount, color = 'b')
    plt.plot(alpha_matrix, gpcount, color = 'r')
    plt.xlabel("Learning Rate (\u03B1)")
    plt.ylabel("Iterations")
    plt.legend(["Best-Response Play", "Gradient Play"])
    plt.title("Iterations vs Learning Rate (\u03B1) for Example {}".format(example))
    plt.savefig("{}-5-iva".format(example),  dpi = 300)
    plt.close()
    #plt.show()


def iterations_vs_error(x1,x2,x,y,J1,J2,alpha, error_matrix,example):
    brcount = np.array([])
    gpcount = np.array([])
    for error in error_matrix:
        br = main.BR_play(x1, x2, x, y, J1, J2, alpha, error)
        gp = main.Gradient_play(x1, x2, x, y, J1, J2, alpha, error)
        
        brcount = np.append(brcount, [br[2]])
        gpcount = np.append(gpcount, [gp[2]])
        
        print("Done {}".format(alpha))
    plt.figure()
    plt.plot(error_matrix, brcount, color = 'b')
    plt.plot(error_matrix, gpcount, color = 'r')
    plt.xlabel("Convergence Error")
    plt.ylabel("Iterations")
    plt.legend(["Best-Response Play", "Gradient Play"])
    plt.title("Iterations vs Convergence Error for Example {}".format(example))
    plt.savefig("{}-6-ive".format(example), dpi = 300)
    plt.close()
    #plt.show()

def time_vs_alpha(x1,x2,x,y,J1,J2,alpha_matrix, error,example):
    brtime = np.array([])
    gptime = np.array([])
    for alpha in alpha_matrix:
        time_br_start = time.time()
        br = main.BR_play(x1, x2, x, y, J1, J2, alpha, error)
        time_br_end = time.time()
        time_gp_start = time.time()
        gp = main.Gradient_play(x1, x2, x, y, J1, J2, alpha, error)
        time_gp_end = time.time()
        brtime = np.append(brtime, [time_br_end-time_br_start])
        gptime = np.append(gptime, [time_gp_end-time_gp_start])
        
        print("Done {}".format(alpha))
    plt.figure()
    plt.plot(alpha_matrix, brtime, color = 'b')
    plt.plot(alpha_matrix, gptime, color = 'r')
    plt.xlabel("Learning Rate (\u03B1)")
    plt.ylabel("Time (s)")
    plt.legend(["Best-Response Play", "Gradient Play"])
    plt.title("Time vs Learning Rate(\u03B1) for Example {}".format(example))
    plt.savefig("{}-7-tva".format(example), dpi=300)
    plt.close()
    #plt.show()


def time_vs_error(x1,x2,x,y,J1,J2,alpha, error_matrix,example):
    brtime = np.array([])
    gptime = np.array([])
    for error in error_matrix:
        time_br_start = time.time()
        br = main.BR_play(x1, x2, x, y, J1, J2, alpha, error)
        time_br_end = time.time()
        time_gp_start = time.time()
        gp = main.Gradient_play(x1, x2, x, y, J1, J2, alpha, error)
        time_gp_end = time.time()
        brtime = np.append(brtime, [time_br_end-time_br_start])
        gptime = np.append(gptime, [time_gp_end-time_gp_start])
        
        print("Done {}".format(alpha))
    plt.figure()
    plt.plot(error_matrix, brtime, color = 'b')
    plt.plot(error_matrix, gptime, color = 'r')
    plt.xlabel("Convergence Error")
    plt.ylabel("Time (s)")
    plt.legend(["Best-Response Play", "Gradient Play"])
    plt.title("Time vs Convergence Error for Example {}".format(example))
    plt.savefig("{}-8-tve".format(example), dpi=300)
    plt.close()
    #plt.show()




#################################
def iterations_vs_ics(x1_matrix,x2_matrix,x,y,J1,J2,alpha, error, example):
    brcount = np.zeros((len(x1_matrix),len(x2_matrix)))
    gpcount = np.zeros((len(x1_matrix),len(x2_matrix)))
    for i in range(len(x1_matrix)):
        for j in range(len(x2_matrix)): 
            br = main.BR_play(x1_matrix[i],x2_matrix[j], x, y, J1, J2, alpha, error)
            gp = main.Gradient_play(x1_matrix[i],x2_matrix[j], x, y, J1, J2, alpha, error)
        
            brcount[i][j] = br[2]
            gpcount[i][j] = gp[2]
        
        print("Done {}".format(i))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("Iterations vs Initial conditions for Example {}".format(example))

    X, Y = np.meshgrid(x1_matrix, x2_matrix)
    ax.plot_surface(X, Y, brcount, color='b', alpha=0.5)
    ax.plot_surface(X,Y, gpcount, color = 'r', alpha=0.5)
    
    ax.plot_wireframe(X, Y, brcount,rcount=10, ccount = 10, color = 'b')
    ax.plot_wireframe(X,Y, gpcount,rcount=10, ccount = 10, color = 'r')
    #ax.plot_trisurf(x1_matrix, x2_matrix, brcount, color = 'b')
    #ax.plot_trisurf(x1_matrix, x2_matrix, gpcount, color = 'r')
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_zlabel("Iterations")
    plt.show()

def time_vs_ics(x1_matrix,x2_matrix,x,y,J1,J2,alpha, error, example):
    brcount = np.zeros((len(x1_matrix),len(x2_matrix)))
    gpcount = np.zeros((len(x1_matrix),len(x2_matrix)))
    for i in range(len(x1_matrix)):
        for j in range(len(x2_matrix)): 
            time_br_start = time.time()
            br = main.BR_play(x1_matrix[i], x2_matrix[j], x, y, J1, J2, alpha, error)
            time_br_end = time.time()
            time_gp_start = time.time()
            gp = main.Gradient_play(x1_matrix[i], x2_matrix[j], x, y, J1, J2, alpha, error)
            time_gp_end = time.time()
            
        
            brcount[i][j] = time_br_end-time_br_start
            gpcount[i][j] = time_gp_end-time_gp_start
        
        print("Done {}".format(i))
    fig = plt.figure()

    ax = plt.axes(projection='3d')
    ax.set_title("Time vs Initial conditions for Example {}".format(example))

    X, Y = np.meshgrid(x1_matrix, x2_matrix)
    ax.plot_surface(X, Y, brcount, color='b', alpha=0.5)
    ax.plot_surface(X,Y, gpcount, color = 'r', alpha=0.5)
    
    ax.plot_wireframe(X, Y, brcount,rcount=10, ccount = 10, color = 'b')
    ax.plot_wireframe(X,Y, gpcount,rcount=10, ccount = 10, color = 'r')
    #ax.plot_trisurf(x1_matrix, x2_matrix, brcount, color = 'b')
    #ax.plot_trisurf(x1_matrix, x2_matrix, gpcount, color = 'r')
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_zlabel("Time (s)")
    plt.show()

################################################################################

def special_u_convergence_vs_iterations(x1,x2,x,y,J1,J2, alpha, error):
    plt.figure()
    colorb = ['b','g','c']
    colorg = ['r','m','y']
    for i in range(len(J1)):

        brcount = np.array([])
        gpcount = np.array([])
        J1in = J1[i]
        J2in = J2[i]
        br = main.BR_play_values(x1, x2, x, y, J1in, J2in, alpha, error)
        gp = main.Gradient_play_values(x1, x2, x, y, J1in, J2in, alpha, error)
        if br[2] == gp[2]:
            brplot1 = br[0]
            brplot2 = br[1]
            gpplot1 = gp[0]
            gpplot2 = gp[1]
        elif br[2]< gp[2]:
            brplot1 = np.append(br[0] , br[0][-1]*np.ones(gp[2]-br[2]))
            brplot2 = np.append(br[1] , br[1][-1]*np.ones(gp[2]-br[2]))
            gpplot1 = gp[0]
            gpplot2 = gp[1]
        else:
            brplot1 = br[0]
            brplot2 = br[1]
            gpplot1 = np.append(gp[0] , gp[0][-1]*np.ones(br[2]-gp[2]))
            gpplot2 = np.append(gp[1] , gp[1][-1]*np.ones(br[2]-gp[2]))
        plt.plot(brplot1, color = colorb[i], ls = '-')
        plt.plot(brplot2, color = colorb[i], ls = '--')
        plt.plot(gpplot1, color = colorg[i], ls = '-')
        plt.plot(gpplot2, color = colorg[i], ls = '--')
    plt.xlabel("Iteration")
    plt.ylabel("u")
    plt.legend(["u1 Best-Response Play ^2","u2 Best-Response Play ^2", "u1 Gradient Play ^2","u2 Gradient Play ^2",\
    "u1 Best-Response Play ^3","u2 Best-Response Play ^3", "u1 Gradient Play ^3","u2 Gradient Play ^3",\
    "u1 Best-Response Play ^4","u2 Best-Response Play ^4", "u1 Gradient Play ^4","u2 Gradient Play ^4"])
    plt.title("Convergence for u vs Iteration number for Examples 2,3,4")
    plt.savefig("0--special-ucvi", dpi=300)
    plt.close()
    #plt.show()