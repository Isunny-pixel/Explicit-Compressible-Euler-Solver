# -*- coding: utf-8 -*-
"""
Computer Project 1: Explicit Compressible Euler Solver




"""
#<_________________Import_____________>
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time, os, sys, io
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from os import system, name
plt.rcParams.update({'font.size': 15})

#<_________________ DEFINED FUNCTION CODE _____________>


def read_mesh(string):
    
    data = np.loadtxt(string)
    i_max, j_max = int(data[0][0]), int(data[0][1])
    grid_XY = np.moveaxis(np.transpose(np.reshape(data[1:], (j_max, i_max, 2))), 0, -1)

    return grid_XY,i_max,j_max

 #<_____________INITIAL PARAMETER SETTING_____________________>   

def setting_intial_data():
    Iter_parameter = 10000  # (Change Here)
    flag=int(0)
    M_inf = 1.4  # Change Mach here
    convergence_limit = 1e-6
    c = 0.2  #CFL VALUE
    
    
    print("NOTE: Mach and other setting can be set inside the code")
    print("NOTE: \n Mach is set as {} \n CFL is set as {} \n Convergence limit is set as {} \n ".format(M_inf,c, convergence_limit))
    
    while(flag !=1 or flag !=2):
        print("Enter Mode of FLux:")
        print("Press '1' For AUSM")
        print("Press '2' For Van Leer")
        
        flag=int(input("Enter Input (1 or 2)"))
        
        if flag==1:
            AUSM_SWITCH = True  # Turn on AUSM or off 
            break
        elif flag==2:
            AUSM_SWITCH = False  # Turn on AUSM or off 
            break
        else:
            print(" \n !!!  Error in input. Please try again !!! \n")
        

    
    return Iter_parameter,AUSM_SWITCH,M_inf, convergence_limit, c 

def setting_constants_data():
    #Assuming air as a calorically perfect gas
    R = 287
    gamma = 1.4
    C_v = R / (gamma - 1)
    C_p = gamma * C_v
    return R,gamma,C_v,C_p

def setting_parameter_data():
    
    Pressure_infinity = 1013.25
    Temperature_infinity = 300
    Density_infinity = Pressure_infinity / (R * Temperature_infinity)
    A_infinity = pow(gamma * R * Temperature_infinity, 0.5)
    U_infinity = M_inf * A_infinity
    V_infinity = 0
    h_infinity = 0.5 * (U_infinity ** 2 + V_infinity ** 2) + C_p * Temperature_infinity
    
    return Pressure_infinity,Temperature_infinity,Density_infinity,A_infinity,U_infinity,V_infinity,h_infinity

#Residual histogram setting
def setting_resiual_histo_data():
    histogram_converg_array = []
    residual_initial_norm_0 = 0
    residual_initial_norm_i = 0
    counter_var = 0
    
    return histogram_converg_array,residual_initial_norm_0,residual_initial_norm_i,counter_var



#<_____________ Calculation Functions _____________________>   

 
def Area_Normal():
    X_para_nx = grid_XY[:, 1:, 1] - grid_XY[:, :-1, 1]
    X_para_ny = grid_XY[:, :-1, 0] - grid_XY[:, 1:, 0]
    X_para_area = pow(X_para_nx ** 2 + X_para_ny ** 2, 0.5)
    
    Y_para_ny = grid_XY[1:, :, 0] - grid_XY[:-1, :, 0]
    Y_para_nx = grid_XY[:-1, :, 1] - grid_XY[1:, :, 1]
    Y_para_area = pow(Y_para_nx ** 2 + Y_para_ny ** 2, 0.5)
    return X_para_nx,X_para_ny,X_para_area,Y_para_ny, Y_para_nx,Y_para_area

#boundary condition assigning
def Boundary_Cond_assign_func(M_para_flow, P_para_flow, T_para_flow, U_para_flow, V_para_flow):
    if M_para_flow > 1:
        # Inflow 
        P_para_flow[0, :], T_para_flow[0, :], U_para_flow[0, :], V_para_flow[0, :] = Pressure_infinity, Temperature_infinity, U_infinity, V_infinity
        # Outflow
        P_para_flow[-1, :], T_para_flow[-1, :], U_para_flow[-1, :], V_para_flow[-1, :] = P_para_flow[-2, :], T_para_flow[-2, :], \
                                                                     U_para_flow[-2, :], V_para_flow[-2, :]
    elif (M_para_flow >= 0) or (M_para_flow <= 1):
        # Inflow 
        P_para_flow[0, :], T_para_flow[0, :], U_para_flow[0, :], V_para_flow[0, :] = P_para_flow[1, :], Temperature_infinity, U_infinity, V_infinity
        # Outflow 
        P_para_flow[-1, :], T_para_flow[-1, :], U_para_flow[-1, :], V_para_flow[-1, :] = Pressure_infinity, T_para_flow[-2, :], \
                                                                     U_para_flow[-2, :], V_para_flow[-2, :]
    else:
        print('Value Error')
    return P_para_flow, T_para_flow, U_para_flow, V_para_flow
#Generate Array of parameter
def array_generator():

    h_temp = (0.5 * (U_infinity ** 2 + V_infinity ** 2) + C_v * Temperature_infinity)
    P = np.zeros((i_max + 1, j_max + 1)) + Pressure_infinity
    T = np.zeros((i_max + 1, j_max + 1)) + Temperature_infinity
    a = np.zeros((i_max + 1, j_max + 1)) + A_infinity
    Rho = np.zeros((i_max + 1, j_max + 1)) + Density_infinity
    Rho_up = np.zeros((i_max - 1, j_max - 1)) + Density_infinity
    u = np.zeros((i_max + 1, j_max + 1)) + U_infinity
    u_up = np.zeros((i_max - 1, j_max - 1)) + Density_infinity * U_infinity
    v = np.zeros((i_max + 1, j_max + 1)) + V_infinity
    v_up = np.zeros((i_max - 1, j_max - 1)) + Density_infinity * V_infinity
    h = np.zeros((i_max + 1, j_max + 1)) + h_infinity
    h_up = np.zeros((i_max - 1, j_max - 1)) + Density_infinity * h_temp

    return h_temp, P,T,a,Rho,Rho_up,u,u_up,v,v_up,h,h_up

#<_____________ Data Save Load Functions _____________________>   


def storing_function():
    #Storing Parameters
    np.savetxt('Density_sol.txt', 0.25 * (Rho[1:, 1:] + Rho[:-1, 1:] + Rho[:-1, :-1] + Rho[1:, :-1]))
    np.savetxt('Pressure_sol.txt',0.25 * (P[1:, 1:] + P[:-1, 1:] + P[:-1, :-1] + P[1:, :-1]))
    np.savetxt('Temperature_sol.txt', 0.25 * (T[1:, 1:] + T[:-1, 1:] + T[:-1, :-1] + T[1:, :-1]))
    np.savetxt('U_vector_sol.txt', 0.25 * (u[1:, 1:] + u[:-1, 1:] + u[:-1, :-1] + u[1:, :-1]))
    np.savetxt('V_vector_sol.txt', 0.25 * (v[1:, 1:] + v[:-1, 1:] + v[:-1, :-1] + v[1:, :-1]))
    np.savetxt('a_final.txt', 0.25 * (a[1:, 1:] + a[:-1, 1:] + a[:-1, :-1] + a[1:, :-1]))
    np.savetxt('h_final.txt', 0.25 * (h[1:, 1:] + h[:-1, 1:] + h[:-1, :-1] + h[1:, :-1]))

def loading_function():
    #Loading Parameters
    Rho_final=np.loadtxt('Density_sol.txt')
    P_final=np.loadtxt("Pressure_sol.txt")
    T_final=np.loadtxt("Temperature_sol.txt")
    U_final=np.loadtxt("U_vector_sol.txt")
    V_final=np.loadtxt("V_vector_sol.txt")
    a_final=np.loadtxt("a_final.txt")
    h_final=np.loadtxt("h_final.txt")
    
    return Rho_final,P_final,T_final,U_final,V_final,a_final,h_final



def Plottting_func_Resiual():
    if AUSM_SWITCH:
        flux_flag = 'AUSM based'
    else:
        flux_flag = 'Van Leer based'
    
    plt.figure(figsize=(14, 11))

    plt.semilogy(histogram_converg_array,color='r')
    plt.title('Convergence Plot for given Mach={}, Flux Type={}'.format(M_inf, flux_flag))
    plt.xlabel('Iteration number', fontsize=16)
    plt.ylabel('Residual', fontsize=16)
    plt.grid()
    plt.savefig("Convergence_mach_{}_flux_{}.png".format(M_inf, flux_flag), dpi=250)

def Plottting_func_Pressure():
    
    if AUSM_SWITCH:
        flux_flag = 'AUSM based'
    else:
        flux_flag = 'Van Leer based'
    
    
    plt.figure(figsize=(14, 11))
    ax = plt.axes(projection='3d')
    bb = ax.plot_surface(grid_XY[:, :, 0], grid_XY[:, :, 1], P_final, rstride=1, cstride=1, cmap='jet', edgecolor='none')
    cc = plt.colorbar(bb, ax=ax, shrink=0.5, aspect=5)
    cc.set_label('Pressure Value in (Pa)')
    ax.set_title('Pressure Value Plot, at mach={}, FLux Type={}'.format(M_inf, flux_flag))
    ax.set_xlabel('X_coord')
    ax.set_ylabel('Y_coord')
    ax.set_zlabel('$Pressure$')
    ax.view_init(45,45+90)
    plt.savefig("Pressure_plot_mach_{}_flux_{}.png".format(M_inf, flux_flag), dpi=250)
    
    plt.figure(figsize=(14, 11))
    qq = plt.contour(grid_XY[:, :, 0], grid_XY[:, :, 1], P_final, 40, cmap='jet')
    plt.title('Pressure Contour, at M = {}, FLux Type : {}'.format(M_inf, flux_flag))
    cc = plt.colorbar(qq)
    cc.set_label('Pressure Value in (Pa)')
    plt.xlabel('X_coord')
    plt.ylabel('Y_coord')
    plt.savefig("Pressure_Contour_plot_mach_{}_flux_{}.png".format(M_inf, flux_flag), dpi=250)

def Plottting_func_v_velocity():

    if AUSM_SWITCH:
        flux_flag = 'AUSM based'
    else:
        flux_flag = 'Van Leer based'
    
    
    
    plt.figure(figsize=(14, 11))
    ax = plt.axes(projection='3d')
    bb = ax.plot_surface(grid_XY[:, :, 0], grid_XY[:, :, 1], V_final, rstride=1, cstride=1, cmap='jet', edgecolor='none')
    cc = plt.colorbar(bb, ax=ax, shrink=0.5, aspect=5)
    cc.set_label('v Value in (m/s)')
    ax.set_title('v Value plot, at M = {}, FLux Type : {}'.format(M_inf, flux_flag))
    ax.set_xlabel('X_coord')
    ax.set_ylabel('Y_coord')
    ax.set_zlabel('V_parameter')
    ax.view_init(45,45+90)
    plt.savefig("v_Velocity_plot_mach_{}_flux_{}.png".format(M_inf, flux_flag), dpi=250)
    
    plt.figure(figsize=(14, 11))
    qq = plt.contour(grid_XY[:, :, 0], grid_XY[:, :, 1], V_final, 40, cmap='jet')
    plt.title('v component Velocity Value (Contour line), at M = {}, Flux type : {}'.format(M_inf, flux_flag))
    cc = plt.colorbar(qq)
    cc.set_label('v component Velocity Value in (m/s)')
    plt.xlabel('X_coord')
    plt.ylabel('Y_coord')
    plt.savefig("v_Velocity_Contour_mach_{}_flux_{}.png".format(M_inf, flux_flag), dpi=250)
 
def Plottting_func_u_velocity():    

    if AUSM_SWITCH:
        flux_flag = 'AUSM based'
    else:
        flux_flag = 'Van Leer based'
    
    

    ax = plt.axes(projection='3d')
    bb = ax.plot_surface(grid_XY[:, :, 0], grid_XY[:, :, 1], U_final, rstride=1, cstride=1, cmap='jet', edgecolor='none')
    cc = plt.colorbar(bb, ax=ax, shrink=0.5, aspect=5)
    cc.set_label('u component Velocity Value in (m/s)')
    ax.set_title('u component Velocity Value Plot, at M = {}, Flux type : {}'.format(M_inf, flux_flag))
    ax.set_xlabel('X_coord')
    ax.set_ylabel('Y_coord')
    ax.set_zlabel('U_parameter')
    ax.view_init(45,45+90)
    
    plt.savefig("u_Velocity_plot_mach_{}_flux_{}.png".format(M_inf, flux_flag), dpi=250)
    
    plt.figure(figsize=(14, 11))
    qq = plt.contour(grid_XY[:, :, 0], grid_XY[:, :, 1], U_final, 40, cmap='jet')
    plt.title('u component Velocity Contour, at M = {}, Flux type : {}'.format(M_inf, flux_flag))
    cc = plt.colorbar(qq)
    cc.set_label('u component Velocity Value in (m/s)')
    plt.xlabel('X_coord')
    plt.ylabel('Y_coord')
    plt.savefig("u_Velocity_Con_mach_{}_flux_{}.png".format(M_inf, flux_flag), dpi=250)    
    
def Plottting_func_vector_velocity():    

    if AUSM_SWITCH:
        flux_flag = 'AUSM based'
    else:
        flux_flag = 'Van Leer based'
        

    plt.figure(figsize=(14, 11))
    M = np.sqrt(U_final**2 + V_final**2)
    qq = plt.quiver(grid_XY[:, :, 0], grid_XY[:, :, 1], U_final, V_final, M, linewidth=0.1, cmap='jet')
    cc = plt.colorbar(qq)
    cc.set_label('Velocity Value in (m/s)')
    plt.title('Velocity Vector Plot, at M = {}, Flux type : {}'.format(M_inf, flux_flag))
    plt.xlabel('X_coord', fontsize=20)
    plt.ylabel('Y_coord', fontsize=20)
    plt.savefig("Velocity_vector_mach_{}_flux_{}.png".format(M_inf, flux_flag), dpi=250)    

def Plottting_func_mach():    

    if AUSM_SWITCH:
        flux_flag = 'AUSM-based'
    else:
        flux_flag = 'Van Leer -based'

    plt.figure(figsize=(14, 11))
    ax = plt.axes(projection='3d')
    bb = ax.plot_surface(grid_XY[:, :, 0], grid_XY[:, :, 1], U_final/a_final, rstride=1, cstride=1, cmap='jet', edgecolor='none')
    cc = plt.colorbar(bb, ax=ax, shrink=0.5, aspect=5)
    cc.set_label('Mach Value')
    ax.set_title('Mach Value Plot, for given M = {}, Flux type : {}'.format(M_inf, flux_flag))
    ax.set_xlabel('X_coord')
    ax.set_ylabel('Y_coord')
    ax.set_zlabel('M_parameter')
    ax.view_init(45,45+90)
    plt.savefig("Mach_plot_value_mach_{}_flux_{}.png".format(M_inf, flux_flag), dpi=250)
    plt.figure(figsize=(14, 11))
    qq = plt.contour(grid_XY[:, :, 0], grid_XY[:, :, 1], U_final/a_final, 40, cmap='jet')
    plt.title('Mach Contour, for given M = {}, Flux type : {}'.format(M_inf, flux_flag))
    cc = plt.colorbar(qq)
    cc.set_label('Mach Value')
    plt.xlabel('X_coord')
    plt.ylabel('Y_coord')
    plt.savefig("Mach_based_Conture_mach_{}_flux_{}.png".format(flux_flag, M_inf), dpi=250)
 

def loop_func(h_temp, P,T,a,Rho,Rho_up,u,u_up,v,v_up,h,h_up,counter_var):
    
    while True:
        startTime=time.time()
        #_ Boundary Conditions #_####
    
        # Enforce Flow Boundary
        
        
        # Wall Boundary and Tangency
        P[:, 0], P[:, -1] = P[:, 1], P[:, -2]
    
        u[1:-1, 0] = u[1:-1, 1] - (2 * u[1:-1, 1]
                                   + v[1:-1, 1] * (Y_para_ny / Y_para_area)[:, 1]) * (Y_para_nx / Y_para_area)[:, 1]
        v[1:-1, 0] = v[1:-1, 1] - (2 * u[1:-1, 1] * (Y_para_nx / Y_para_area)[:, 1]
                                   + v[1:-1, 1] * (Y_para_ny / Y_para_area)[:, 1]**2)
    
        u[1:-1, -1] = u[1:-1, -2] - (2 * u[1:-1, -2]
                                     + v[1:-1, -2] * (Y_para_ny / Y_para_area)[:, -2]) * (Y_para_nx / Y_para_area)[:, -2]
        v[1:-1, -1] = v[1:-1, -2] - (2 * u[1:-1, -2] * (Y_para_nx / Y_para_area)[:, -2]
                                     + v[1:-1, -2] * (Y_para_ny / Y_para_area)[:, -2]**2)
    
        #_####  X-Flux #_#####
    
        u_para_temp = (u[:-1, 1:-1] * X_para_nx + v[:-1, 1:-1] * X_para_ny) / X_para_area
        M_para_temp = u_para_temp / a[:-1, 1:-1]
        Para_alpha = 0.5 * (1 + np.sign(M_para_temp))
        Para_beta = -np.maximum(0, 1 - np.floor(np.abs(M_para_temp)))
    
        XP_u = np.abs(u_para_temp) + a[:-1, 1:-1]
        XP_d = Para_alpha * (1 + Para_beta) - 0.25 * Para_beta * pow(1 + M_para_temp, 2) * (2 - M_para_temp)
        XP_c = Para_alpha * (1 + Para_beta) * M_para_temp - 0.25 * Para_beta * pow(1 + M_para_temp, 2)
    
    
        u_para_temp = (u[1:, 1:-1] * X_para_nx + v[1:, 1:-1] * X_para_ny) / X_para_area
        M_para_temp = u_para_temp / a[1:, 1:-1]
        Para_alpha = 0.5 * (1 - np.sign(M_para_temp))
        Para_beta = -np.maximum(0, 1 - np.floor(np.abs(M_para_temp)))
        
        XN_para_u = np.abs(u_para_temp) + a[1:, 1:-1]
        XN_para_d = Para_alpha * (1 + Para_beta) - 0.25 * Para_beta * ((1 - M_para_temp) ** 2) * (2 + M_para_temp)
        XN_para_c = Para_alpha * (1 + Para_beta) * M_para_temp + 0.25 * Para_beta * pow(1 - M_para_temp, 2)
    
        #AUSM SWITCH VALUE 
        if AUSM_SWITCH:
            XP_c = np.maximum(0, XP_c + XN_para_c)
            XN_para_c = np.minimum(0, XP_c + XN_para_c)
    
        X_flux_rho = (Rho[:-1, 1:-1] * a[:-1, 1:-1] * XP_c 
                      + Rho[1:, 1:-1] * a[1:, 1:-1] * XN_para_c) * X_para_area
        X_flux_u = (Rho[:-1, 1:-1] * a[:-1, 1:-1] * XP_c * u[:-1, 1:-1] 
                    + Rho[1:, 1:-1] * a[1:, 1:-1] * XN_para_c * u[1:, 1:-1]) * X_para_area \
                   + XP_d * P[:-1, 1:-1] * X_para_nx + XN_para_d * P[1:, 1:-1] * X_para_nx
        X_flux_v = (Rho[:-1, 1:-1] * a[:-1, 1:-1] * XP_c * v[:-1, 1:-1] 
                    + Rho[1:, 1:-1] * a[1:, 1:-1] * XN_para_c * v[1:, 1:-1]) * X_para_area \
                   + XP_d * P[:-1, 1:-1] * X_para_ny + XN_para_d * P[1:, 1:-1] * X_para_ny
        X_flux_h = (Rho[:-1, 1:-1] * a[:-1, 1:-1] * XP_c * h[:-1, 1:-1] 
                    + Rho[1:, 1:-1] * a[1:, 1:-1] * XN_para_c * h[1:, 1:-1]) * X_para_area
    
        #_####  Y-Flux #_#####
    
        u_para_temp = (u[1:-1, :-1] * Y_para_nx + v[1:-1, :-1] * Y_para_ny) / Y_para_area
        M_para_temp = u_para_temp / a[1:-1, :-1]
        Para_alpha = 0.5 * (1 + np.sign(M_para_temp))
        Para_beta = -np.maximum(0, 1 - np.floor(np.abs(M_para_temp)))
        
        YP_para_u = np.abs(u_para_temp) + a[1:-1, :-1]
        YP_para_d = Para_alpha * (1 + Para_beta) - 0.25 * Para_beta * ((1 + M_para_temp) ** 2) * (2 - M_para_temp)
        YP_para_c = Para_alpha * (1 + Para_beta) * M_para_temp - 0.25 * Para_beta * (1 + M_para_temp) ** 2
    
    
        u_para_temp = (u[1:-1, 1:] * Y_para_nx + v[1:-1, 1:] * Y_para_ny) / Y_para_area
        M_para_temp = u_para_temp / a[1:-1, 1:]
        Para_alpha = 0.5 * (1 - np.sign(M_para_temp))
        Para_beta = -np.maximum(0, 1 - np.floor(np.abs(M_para_temp)))
        
        YN_para_u = np.abs(u_para_temp) + a[1:-1, 1:]
        YN_para_d = Para_alpha * (1 + Para_beta) - 0.25 * Para_beta * ((1 - M_para_temp) ** 2) * (2 + M_para_temp)
        YN_para_c = Para_alpha * (1 + Para_beta) * M_para_temp + 0.25 * Para_beta * (1 - M_para_temp) ** 2
    
        # Estimate Flux for Quantities
        if AUSM_SWITCH:
            YP_para_c = np.maximum(0, YP_para_c + YN_para_c)
            YN_para_c = np.minimum(0, YP_para_c + YN_para_c)
    
        Y_flux_para_rho = (Rho[1:-1, :-1] * a[1:-1, :-1] * YP_para_c 
                      + Rho[1:-1, 1:] * a[1:-1, 1:] * YN_para_c) * Y_para_area
        Y_flux_para_rho[:, 0], Y_flux_para_rho[:, -1] = 0, 0
    
        Y_flux_para_u = (Rho[1:-1, :-1] * a[1:-1, :-1] * YP_para_c * u[1:-1, :-1] 
                    + Rho[1:-1, 1:] * a[1:-1, 1:] * YN_para_c * u[1:-1, 1:]) * Y_para_area \
                   + YP_para_d * P[1:-1, :-1] * Y_para_nx + YN_para_d * P[1:-1, 1:] * Y_para_nx
        Y_flux_para_u[:, 0] = P[1:-1, 1] * Y_para_nx[:, 0] * (YP_para_d[:, 0] + YN_para_d[:, 0])
        Y_flux_para_u[:, -1] = P[1:-1, -2] * Y_para_nx[:, -1] * (YP_para_d[:, -1] + YN_para_d[:, -1])
    
        Y_flux_para_v = (Rho[1:-1, :-1] * a[1:-1, :-1] * YP_para_c * v[1:-1, :-1] 
                    + Rho[1:-1, 1:] * a[1:-1, 1:] * YN_para_c * v[1:-1, 1:]) * Y_para_area \
                   + YP_para_d * P[1:-1, :-1] * Y_para_ny + YN_para_d * P[1:-1, 1:] * Y_para_ny
        Y_flux_para_v[:, 0] = P[1:-1, 1] * Y_para_ny[:, 0] * (YP_para_d[:, 0] + YN_para_d[:, 0])
        Y_flux_para_v[:, -1] = P[1:-1, -2] * Y_para_ny[:, -1] * (YP_para_d[:, -1] + YN_para_d[:, -1])
    
        Y_flux_para_h = (Rho[1:-1, :-1] * a[1:-1, :-1] * YP_para_c * h[1:-1, :-1] 
                    + Rho[1:-1, 1:] * a[1:-1, 1:] * YN_para_c * h[1:-1, 1:]) * Y_para_area
        Y_flux_para_h[:, 0], Y_flux_para_h[:, -1] = 0, 0
    
        #_##  Residual Estimator #_####
        res_para_rho = X_flux_rho[1:, :] - X_flux_rho[:-1, :] + Y_flux_para_rho[:, 1:] - Y_flux_para_rho[:, :-1]
        res_para_U = X_flux_u[1:, :] - X_flux_u[:-1, :] + Y_flux_para_u[:, 1:] - Y_flux_para_u[:, :-1]
        res_para_V = X_flux_v[1:, :] - X_flux_v[:-1, :] + Y_flux_para_v[:, 1:] - Y_flux_para_v[:, :-1]
        res_para_H = X_flux_h[1:, :] - X_flux_h[:-1, :] + Y_flux_para_h[:, 1:] - Y_flux_para_h[:, :-1]
    
        residual_initial_norm_i = np.sqrt(np.sum(pow(res_para_rho / (Density_infinity * U_infinity), 2)
                                       + pow(res_para_U / (Density_infinity * U_infinity ** 2), 2)
                                       + pow(res_para_V / (Density_infinity * U_infinity ** 2), 2)
                                       + pow(res_para_H / (Density_infinity * U_infinity * h_infinity), 2)))
    
        #_#  Solution Updating #_#####
        updateing_para_value = c / (X_para_area[:-1, :] * XN_para_u[:-1, :] + X_para_area[1:, :] * XP_u[1:, :] +
                        Y_para_area[:, :-1] * YN_para_u[:, :-1] + Y_para_area[:, 1:] * YP_para_u[:, 1:])
    
        # Estimate Update in paramter 
        Rho_up = Rho_up - updateing_para_value * res_para_rho
        u_up = u_up - updateing_para_value * res_para_U
        v_up = v_up - updateing_para_value * res_para_V
        h_up = h_up - updateing_para_value * res_para_H
    
        # Paramater update assigning
        Rho[1:-1, 1:-1] = Rho_up
        u[1:-1, 1:-1] = u_up / Rho_up
        v[1:-1, 1:-1] = v_up / Rho_up
        T[1:-1, 1:-1] = (gamma - 1) * (h_up / Rho_up - 0.5 * (u[1:-1, 1:-1] ** 2 + v[1:-1, 1:-1] ** 2)) / R
        a[1:-1, 1:-1] = (gamma * R * np.abs(T[1:-1, 1:-1])) ** 0.5
        P[1:-1, 1:-1] = Rho_up * R * T[1:-1, 1:-1]
        h[1:-1, 1:-1] = C_p * T[1:-1, 1:-1] + 0.5 * (u[1:-1, 1:-1] ** 2 + v[1:-1, 1:-1] ** 2)
    
        # loop ending condition of convergence
        if counter_var == 0:
            residual_initial_norm_0 = residual_initial_norm_i #Assigning first residual
    
        if ((counter_var != 0) and (residual_initial_norm_i / residual_initial_norm_0 < convergence_limit)) or (counter_var > Iter_parameter):
            break
        
        histogram_converg_array.append(residual_initial_norm_i / residual_initial_norm_0)
        print("Convergence factor  {}".format(residual_initial_norm_i / residual_initial_norm_0))
        counter_var += 1
        
        totalTime = round(time.time() - startTime, 4)
        print('Iterations Completed in : %s seconds' % (totalTime))
    

#<_________________ MAIN CODE _____________>

print("-----------PROGRAM BOOTING----------------------")



# PARAMTERS Initializer
Iter_parameter,AUSM_SWITCH,M_inf, convergence_limit, c=setting_intial_data()
R,gamma,C_v,C_p=setting_constants_data()
Pressure_infinity,Temperature_infinity,Density_infinity,A_infinity,U_infinity,V_infinity,h_infinity=setting_parameter_data()


# Loading Grid Dataset Initializer
string='E:/Whatsapp data/whatsapp data/computational aerodynamics/bumpgrid.dat' #FILE LOCATION ENTER HERE
grid_XY,i_max,j_max=read_mesh(string)

#Area Normal Initializer
X_para_nx,X_para_ny,X_para_area,Y_para_ny, Y_para_nx,Y_para_area=Area_Normal()

#Create Array of paramters
h_temp, P,T,a,Rho,Rho_up,u,u_up,v,v_up,h,h_up=array_generator()


histogram_converg_array,residual_initial_norm_0,residual_initial_norm_i,counter_var=setting_resiual_histo_data()


#Starts time to measure overall time of convergence given threshold of error. 
startTime1=time.time()


###########################################


loop_func(h_temp, P,T,a,Rho,Rho_up,u,u_up,v,v_up,h,h_up,counter_var)


###########################################


totalTime1 = round(time.time() - startTime1, 4)
print('Total Iterations Completed in : %s seconds' % (totalTime1))


#Formating of data 

u[1:-1, 0] = u[1:-1, 1] - (2 * u[1:-1, 1] + v[1:-1, 1] * (Y_para_ny / Y_para_area)[:, 1]) * (Y_para_nx / Y_para_area)[:, 1]
v[1:-1, 0] = v[1:-1, 1] - (2 * u[1:-1, 1] * (Y_para_nx / Y_para_area)[:, 1] + v[1:-1, 1] * (Y_para_ny / Y_para_area)[:, 1]**2)
u[1:-1, -1] = u[1:-1, -2] - (2 * u[1:-1, -2] + v[1:-1, -2] * (Y_para_ny / Y_para_area)[:, -2]) * (Y_para_nx / Y_para_area)[:, -2]
v[1:-1, -1] = v[1:-1, -2] - (2 * u[1:-1, -2] * (Y_para_nx / Y_para_area)[:, -2] + v[1:-1, -2] * (Y_para_ny / Y_para_area)[:, -2]**2)

#Save parameter
storing_function()

#Load Paramter , can be used to preload graphs if calculation done
Rho_final,P_final,T_final,U_final,V_final,a_final,h_final=loading_function()

#Plot functions
Plottting_func_Resiual()
Plottting_func_Pressure()
Plottting_func_u_velocity()
Plottting_func_v_velocity()
Plottting_func_vector_velocity()
Plottting_func_mach()


print("-----------PROGRAM COMPLETED----------------------")