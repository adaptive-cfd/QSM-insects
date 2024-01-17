#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: nico
"""
##%kinematics
import os
import csv
os.environ['QTA_QP_PLATFORM'] = 'wayland'
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib import animation
import functools 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import insect_tools as it 
from debug import writeArraytoFile

"""
all the parameters we need to describe a wing in space
global -> body -> stroke -> wing 
"""
#helper functions

##########################################################################################################################

def parse_wing_file(wing_file, scale, pivot_index) -> np.ndarray:
    #open the file in read mode     
    file_to_read = open(wing_file, 'r')
    #cswwriter activation 
    csvreader = csv.reader(file_to_read)
    wingPoints = []
    for row in csvreader:
        #each row is a list/array
        current_row = [-float(row[1]), float(row[0]), 0.0]
        wingPoints.append(current_row)
    #connect the last point to first one 
    wingPoints.append(wingPoints[0].copy())
    #converts to a numpy array for vector operation/computation
    wingPoints = np.array(wingPoints)

    #shift wing points at the first point 
    pivot = wingPoints[pivot_index]
    wingPoints -= pivot
    file_to_read.close() 
    return wingPoints * scale 

def getChordLength(wingPoints, y_coordinate):
    #get the division in wing segments (front and back)
    split_index = 16
    righthand_section = wingPoints[:split_index]
    lefthand_section = wingPoints[split_index:]

    #interpolate righthand section 
    righthand_section_interpolation = interp1d(righthand_section[:, 1], righthand_section[:, 0], fill_value='extrapolate')
  
    #interpolate lefthand section
    lefthand_section_interpolation = interp1d(lefthand_section[:, 1], lefthand_section[:, 0], fill_value='extrapolate') 
    
    #generate the chord as a function of y coordinate
    chord_length = abs(righthand_section_interpolation(y_coordinate) - lefthand_section_interpolation(y_coordinate))
    return chord_length

def convert_from_wing_reference_frame_to_stroke_plane(points, parameters, invert=False):
    #points passed into this fxn must be in the wing reference frame x(w) y(w) z(w)
    #phi, alpha, theta
    phi = parameters[4] #rad
    alpha = parameters[5] #rad
    theta = parameters[6] #rad
    if invert: 
         phi = -phi 
         alpha = -alpha 
    rotationMatrix = np.matmul(
                                    np.matmul(
                                            it.Ry(alpha),
                                            it.Rz(theta)
                                    ),
                                    it.Rx(phi)
                                )
    rotationMatrixTrans = np.transpose(rotationMatrix)  
    strokePoints = np.zeros((points.shape[0], 3))
    for point in range(points.shape[0]): 
        x_s = np.matmul(rotationMatrixTrans, points[point])
        strokePoints[point, :] = x_s
    return strokePoints, rotationMatrix, rotationMatrixTrans


def convert_from_stroke_plane_to_body_reference_frame(points, parameters, invert=False):
    #points must be in stroke plane x(s) y(s) z(s)
    eta = parameters[3] #rad
    flip_angle = 0 
    if invert:
         flip_angle = np.pi
    rotationMatrix = np.matmul(
                                it.Rx(flip_angle),
                                it.Ry(eta)
                                )
    rotationMatrixTrans = np.transpose(rotationMatrix) 
    bodyPoints = np.zeros((points.shape[0], 3))
    for point in range(points.shape[0]): 
        x_b = np.matmul(rotationMatrixTrans, points[point])
        bodyPoints[point,:] = x_b
    return bodyPoints, rotationMatrix, rotationMatrixTrans

def convert_from_body_reference_frame_to_global_reference_frame(points, parameters):
    #points passed into this fxn must be in the body reference frame x(b) y(b) z(b)
    #phi, alpha, theta
    psi = parameters [0] #rad
    beta = parameters [1] #rad
    gamma = parameters [2] #rad
    rotationMatrix = np.matmul(
                                    np.matmul(
                                            it.Rx(psi),
                                            it.Ry(beta)
                                    ),
                                    it.Rz(gamma)
                                )
    rotationMatrixTrans = np.transpose(rotationMatrix)
    globalPoints = np.zeros((points.shape[0], 3))
    for point in range(points.shape[0]): 
        x_g = np.matmul(rotationMatrixTrans, points[point])
        globalPoints[point, :] = x_g 
    return globalPoints, rotationMatrix, rotationMatrixTrans

def generate_rot_wing(wingRotationMatrix, bodyRotationMatrixTrans, strokeRotationMatrixTrans, phi, phi_dt, alpha, alpha_dt, theta, theta_dt): 
    phiMatrixTrans = np.transpose(it.Rx(phi)) #np.transpose(getRotationMatrix('x', phi))
    alphaMatrixTrans = np.transpose(it.Ry(alpha)) #np.transpose(getRotationMatrix('y', alpha))
    thetaMatrixTrans = np.transpose(it.Rz(theta)) #np.transpose(getRotationMatrix('z', theta))
    vector_phi_dt = np.array([[phi_dt], [0], [0]])
    vector_alpha_dt = np.array([[0], [alpha_dt], [0]])
    vector_theta_dt = np.array([[0], [0], [theta_dt]])
    rot_wing_s = np.matmul(phiMatrixTrans, (vector_phi_dt+np.matmul(thetaMatrixTrans, (vector_theta_dt+np.matmul(alphaMatrixTrans, vector_alpha_dt)))))
    rot_wing_b = np.matmul(strokeRotationMatrixTrans, rot_wing_s)
    rot_wing_w = np.matmul(wingRotationMatrix, rot_wing_s)
    rot_wing_g = np.matmul(bodyRotationMatrixTrans, rot_wing_b)
    return rot_wing_g, rot_wing_b, rot_wing_w #these are all (3x1) vectors 

def generate_u_wing_g_position(rot_wing_g, y_wing_g):
    # #omega x point
    #both input vectors have to be reshaped to (1,3) to meet the requirements of np.cross (last axis of both vectors -> 2 or 3). to that end either reshape(1,3) or flatten() kommen in frage
    rot_wing_g = rot_wing_g.reshape(1,3)  
    y_wing_g = y_wing_g.reshape(1,3)
    u_wing_g_position = np.cross(rot_wing_g, y_wing_g)
    return u_wing_g_position

def generate_u_wing_w(u_wing_g, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix):
    rotationMatrix = np.matmul(np.matmul(bodyRotationMatrix, strokeRotationMatrix), wingRotationMatrix)
    u_wing_w = np.matmul(rotationMatrix, u_wing_g)
    return u_wing_w

def getWindDirectioninWingReferenceFrame(u_wind_g, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix): 
    u_wind_b = np.matmul(bodyRotationMatrix, u_wind_g.reshape(3,1))
    u_wind_s = np.matmul(strokeRotationMatrix, u_wind_b)
    u_wind_w = np.matmul(wingRotationMatrix, u_wind_s)
    return u_wind_w

def getAoA(x_wing_g, e_u_wing_g):
    AoA = np.arccos(np.dot(x_wing_g, e_u_wing_g))
    return AoA

# def getAoA(drag_vector, x_wing_g):
#     #should be in the wing reference frame
#     AoA = np.arctan2(np.linalg.norm(np.cross(x_wing_g, drag_vector)), np.dot(-drag_vector, x_wing_g.reshape(3,1))) #rad
#     return AoA  
        
def load_kinematics_data(file='kinematics_data_for_QSM.csv'): 
    t = np.zeros(shape=(1000, 1)) 
    alpha = np.zeros(shape=(1000, 1)) 
    phi = np.zeros(shape=(1000, 1)) 
    theta = np.zeros(shape=(1000, 1)) 
    alpha_dt = np.zeros(shape=(1000, 1)) 
    phi_dt = np.zeros(shape=(1000, 1)) 
    theta_dt = np.zeros(shape=(1000, 1)) 

    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=';') 
        c = 0 
        for line in reader:
            if c>= 2:     
                timeStep, alphas, phis, thetas, alphas_dt, phis_dt, thetas_dt  = line
                t[c-2, :] = float(timeStep)
                alpha[c-2, :] = float(alphas)
                phi[c-2, :] = float(phis)
                theta[c-2, :] = float(thetas)
                alpha_dt[c-2, :] = float(alphas_dt)
                phi_dt[c-2, :] = float(phis_dt)
                theta_dt[c-2, :] = float(thetas_dt)
            c += 1
    return t, np.radians(alpha), np.radians(phi), np.radians(theta), np.radians(alpha_dt), np.radians(phi_dt), np.radians(theta_dt)

def generateSequence (wingPoints, wingtip_index, pivot_index, start_time=0, number_of_timesteps=360, frequency=1, useCFDData=True):
    u_wind_g = np.array([0, 0, 0])

    timeline, alphas, phis, thetas, alphas_dt, phis_dt, thetas_dt = load_kinematics_data('kinematics_data_for_QSM.csv')
    # timeline, alphas, phis, thetas, alphas_dt, phis_dt, thetas_dt = it.load_t_file('kinematics_data_for_QSM.csv')
        
    alphas_interp = interp1d(timeline.flatten(), alphas.flatten(), fill_value='extrapolate')
    phis_interp = interp1d(timeline.flatten(), phis.flatten(), fill_value='extrapolate')
    thetas_interp = interp1d(timeline.flatten(), thetas.flatten(), fill_value='extrapolate')

    timeline = np.linspace(0, 1, 101)

    strokePointsSequence = np.zeros((timeline.shape[0], wingPoints.shape[0], 3))
    bodyPointsSequence = np.zeros((timeline.shape[0], wingPoints.shape[0], 3))
    globalPointsSequence = np.zeros((timeline.shape[0], wingPoints.shape[0], 3))

    rots_wing_b = np.zeros((timeline.shape[0], 3, 1))
    rots_wing_w = np.zeros((timeline.shape[0], 3, 1))
    rots_wing_g = np.zeros((timeline.shape[0], 3, 1))

    us_wing_w = np.zeros((timeline.shape[0], 3, 1))
    us_wing_g = np.zeros((timeline.shape[0], 3, 1))

    us_wind_w = np.zeros((timeline.shape[0], 3, 1))
    AoA = np.zeros((timeline.shape[0], 1))
    dragVectors_wing_g = np.zeros((timeline.shape[0], 3))
    liftVectors = np.zeros((timeline.shape[0], 3))
    e_liftVectors = np.zeros((timeline.shape[0], 3))

    delta_t = timeline[1] - timeline[0]

    for timeStep in range(timeline.shape[0]):
        t = timeline[timeStep]
        #parameter array: psi [0], beta[1], gamma[2], eta[3], phi[4], alpha[5], theta[6]
        alphas_dt = (alphas_interp(t+delta_t) - alphas_interp(t)) / delta_t
        phis_dt = (phis_interp(t+delta_t) - phis_interp(t)) / delta_t
        thetas_dt = (thetas_interp(t+delta_t) - thetas_interp(t)) / delta_t

        parameters = [0, 0, 0, -80*np.pi/180, phis_interp(t), alphas_interp(t), thetas_interp(t)] # 7 angles in radians! #without alphas[timeStep] any rotation around any y axis through an angle of pi/2 gives an error! 
        parameters_dt = [0, 0, 0, 0, phis_dt, alphas_dt, thetas_dt]

        strokePoints, wingRotationMatrix, wingRotationMatrixTrans = convert_from_wing_reference_frame_to_stroke_plane(wingPoints, parameters)
        bodyPoints, strokeRotationMatrix, strokeRotationMatrixTrans = convert_from_stroke_plane_to_body_reference_frame(strokePoints, parameters)
        globalPoints, bodyRotationMatrix, bodyRotationMatrixTrans = convert_from_body_reference_frame_to_global_reference_frame(bodyPoints, parameters)
        
        strokePointsSequence[timeStep, :] = strokePoints
        bodyPointsSequence[timeStep, :] = bodyPoints
        globalPointsSequence[timeStep, :] = globalPoints

        x_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[1], [0], [0]]))
        y_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[0], [1], [0]]))

        rot_wing_g, rot_wing_b, rot_wing_w = generate_rot_wing(wingRotationMatrix, bodyRotationMatrixTrans, strokeRotationMatrixTrans, parameters[4], parameters_dt[4], parameters[5], 
                                    parameters_dt[5], parameters[6], parameters_dt[6])
        
        rots_wing_b[timeStep, :] = rot_wing_b
        rots_wing_w[timeStep, :] = rot_wing_w
        rots_wing_g[timeStep, :] = rot_wing_g

        u_wing_g = generate_u_wing_g_position(rot_wing_g, y_wing_g)

        u_flight_g = u_wing_g + u_wind_g

        u_wing_w = generate_u_wing_w(u_wing_g.reshape(3,1), bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix)
        us_wing_g[timeStep, :] = u_wing_g.reshape(3,1)
        us_wing_w[timeStep, :] = u_wing_w
        u_wind_w = getWindDirectioninWingReferenceFrame(u_flight_g, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix) - np.array(u_wing_w.reshape(3,1))
        us_wind_w[timeStep, :] = u_wind_w

        u_wing_g_magnitude = np.sqrt(u_wing_g[0, 0]**2 + u_wing_g[0, 1]**2 + u_wing_g[0, 2]**2)
        if u_wing_g_magnitude != 0:  
            e_u_wing_g = u_wing_g/u_wing_g_magnitude
        else:
            e_u_wing_g = u_wing_g 
        dragVector_wing_g = -e_u_wing_g
        dragVectors_wing_g[timeStep, :] = dragVector_wing_g
    
        #lift 
        liftVector = np.cross(e_u_wing_g, y_wing_g.flatten())
        liftVector = liftVector*np.sign(alphas_interp(t))
        liftVectors[timeStep, :] = liftVector

        aoa = getAoA(x_wing_g.reshape(1,3), e_u_wing_g.reshape(3,1)) #use this one for getAoA thru arccos 
        # aoa = getAoA(dragVector_wing_g, x_wing_g.flatten()) #use this one for getAoA thru arctan 
        AoA[timeStep, :] = aoa
        liftVector_magnitude = np.sqrt(liftVector[0, 0]**2 + liftVector[0, 1]**2 + liftVector[0, 2]**2)
        if liftVector_magnitude != 0: 
            e_liftVector = liftVector / liftVector_magnitude
        else:
            e_liftVector = liftVector
        e_liftVectors[timeStep, :] = e_liftVector
        
        

    #validation of our u_wing_g: 
    #left and right derivative: 
    delta_t = timeline[1] - timeline[0]
    verifying_us_wing_g = np.zeros((timeline.shape[0], wingPoints.shape[0], 3))
    for timeStep in range(timeline.shape[0]):
        currentGlobalPoint = globalPointsSequence[timeStep]
        leftGlobalPoint = globalPointsSequence[timeStep-1]
        rightGlobalPoint = globalPointsSequence[(timeStep+1)%len(timeline)]
        LHD = (currentGlobalPoint - leftGlobalPoint) / delta_t
        RHD = (rightGlobalPoint - currentGlobalPoint) / delta_t
        verifying_us_wing_g[timeStep, :] = (LHD+RHD)/2
    verifying_us_wing_g = verifying_us_wing_g  
    return timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, dragVectors_wing_g, e_liftVectors

def animationPlot(ax, alphas, pointsSequence, us_wing_g, AoA, wingtip_index, pivot_index, Fl_BEM, Fd_BEM, dragVectors_wing_g, e_liftVectors, timeStep): 
    #get point set by timeStep number
    points = pointsSequence[timeStep] #pointsSequence can either be global, body, stroke 
    #clear the current axis 
    ax.cla()

    # extract the x, y and z coordinates 
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    #axis limit
    a = 4

    trajectory = np.array(pointsSequence)[:, wingtip_index]
    #3D trajectory 
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='k', linestyle='dashed', linewidth=0.5)

    #x-y plane trajectory 
    ax.plot(trajectory[:, 0], trajectory[:, 1], 0*trajectory[:, 2]-a, color='k', linestyle='dashed', linewidth=0.5)
    
    #z-y plane prajectory 
    ax.plot(0*trajectory[:, 0]-a, trajectory[:, 1], trajectory[:, 2], color='k', linestyle='dashed', linewidth=0.5)

    #x-z plane prajectory 
    ax.plot(trajectory[:, 0], 0*trajectory[:, 1]+a, trajectory[:, 2], color='k', linestyle='dashed', linewidth=0.5)
    
    # #use axis to plot surface 
    ax.plot_trisurf(X, Y, Z, edgecolors='black') 
    ax.add_collection3d(Poly3DCollection(verts=[points]))

    #shadows
    #x-y plane shadow
    XY_plane_shadow = np.vstack((X, Y, -a*np.ones_like(Z))).transpose()
    ax.add_collection3d(Poly3DCollection(verts=[XY_plane_shadow], color='#d3d3d3'))
    #y-z plane shadow
    YZ_plane_shadow = np.vstack((-a*np.ones_like(X), Y, Z)).transpose()
    ax.add_collection3d(Poly3DCollection(verts=[YZ_plane_shadow], color='#d3d3d3'))
    #x-z plane shadow
    XZ_plane_shadow = np.vstack(((X, a*np.ones_like(Y), Z))).transpose()
    ax.add_collection3d(Poly3DCollection(verts=[XZ_plane_shadow], color='#d3d3d3'))
    
    #plotting the vector x, y, z, u, v, w
    u_wing_g = us_wing_g[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], u_wing_g[0], u_wing_g[1], u_wing_g[2], color='orange', label=r'$\overrightarrow{u}^{(g)}_w$' )
    dragVector_wing_g = dragVectors_wing_g[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], dragVector_wing_g[0], dragVector_wing_g[1], dragVector_wing_g[2], color='green', label=r'$\overrightarrow{d}^{(g)}_w$' )
    #lift 
    liftVector = e_liftVectors[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], liftVector[0], liftVector[1], liftVector[2])
    ax.legend()
    
    #set the axis limits
    ax.set_xlim([-a, a])
    ax.set_ylim([-a, a])
    ax.set_zlim([-a, a])

    #set the axis labels 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Timestep: {timeStep} \n⍺: {np.round(np.degrees(alphas[timeStep]), 2)} \nAoA: {np.round(np.degrees(AoA[timeStep]), 2)} \nFl_BEM: {np.round(Fl_BEM[timeStep], 4)} \nFd_BEM: {np.round(Fd_BEM[timeStep], 4)}')
    
def generatePlotsForKinematicsSequence(timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index, Fl_BEM, Fd_BEM): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    anim = animation.FuncAnimation(fig, functools.partial(animationPlot, ax, alphas, globalPointsSequence, us_wing_g, AoA,wingtip_index, pivot_index, Fl_BEM, Fd_BEM, dragVectors_wing_g, e_liftVectors), frames=len(timeline), repeat=True)
    #anim.save('u&d_vectors.gif') 
    plt.show() 

def kinematics(): 
    #file import: get shape 
    wing_file = 'drosophilaMelanogasterWing.csv'
    pivot_index = -8
    wingPoints = parse_wing_file(wing_file, 0.001, pivot_index)
    wingtip_index = 17
    #create figure 
    timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, dragVectors_wing_g, e_liftVectors = generateSequence(wingPoints, wingtip_index, pivot_index, frequency=10, number_of_timesteps=360, useCFDData=True)
    #generatePlotsForKinematicsSequence(timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, wingtip_index, pivot_index)
    return timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index
kinematics()
############################################################################################################################################################################################
##% dynamics

def getAerodynamicCoefficients(x0, AoA): 
    deg2rad = np.pi/180.0 
    rad2deg = 180.0/np.pi
    
    AoA = rad2deg*AoA
    
    cl = x0[0] + x0[1]*np.sin( deg2rad*(2.13*AoA - 7.20) )
    cd = x0[2] + x0[3]*np.cos( deg2rad*(2.04*AoA - 9.82) )
    
    return cl, cd

def load_forces_data(file = 'forces_data_for_QSM.csv'):
    t = [] 
    Fx = [] 
    Fy = [] 
    Fz = [] 

    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=';') 
        c = 0 
        for line in reader:
            if c>= 2:     
                timeStep, fx, fy, fz = line
                t.append(float(timeStep))
                Fx.append(float(fx)) 
                Fy.append(float(fy))
                Fz.append(float(fz))
            c += 1
    return np.array(t), np.array(Fx), np.array(Fy), np.array(Fz)

############################################################################################################################################################################################
##%% main 

from scipy.integrate import trapz
import scipy.optimize as opt
import time

def F(x, timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index, Fx_CFD_interp, Fy_CFD_interp, Fz_CFD_interp, Fx_CFD, Fy_CFD, Fz_CFD, show_plots=False):
    cl, cd = getAerodynamicCoefficients(x, np.array(AoA))
    if show_plots: 
        # plt.plot(timeline, np.degrees(phis), label='ɸ')
        # plt.plot(timeline, np.degrees(alphas), label ='⍺')
        # plt.plot(timeline, np.degrees(thetas), label='Θ')
        # plt.legend()
        # plt.show()

        plt.plot(timeline, np.degrees(AoA), label='AoA')
        plt.xlabel('t/T')
        plt.legend()
        plt.show()
        

    min_y = np.min(wingPoints[:, 1])
    max_y = np.max(wingPoints[:, 1])
    y_space = np.linspace(min_y, max_y, 100)
    c = getChordLength(wingPoints, y_space)
    c_norm = c / np.max(c)
    c_norm_interpolation = interp1d(y_space, c_norm)

    def Cr2(r): 
        return c_norm_interpolation(r) * r**2
    # fxn evaluated at the intervals 

    F_r = Cr2(y_space)
    I = trapz(F_r, y_space)

    planar_rot_w_squared = rots_wing_w[:, 0]**2 + rots_wing_w[:, 2]**2 
    rho = 1.225
    cr = c_norm
    br = y_space[1]-y_space[0]
    Fl_BEM_magnitude = np.zeros(timeline.shape[0])
    Fd_BEM_magnitude = np.zeros(timeline.shape[0])

    #calculation of the magnitude of the lift force for each blade 
    for i in range(y_space.shape[0]):
        Fl_BEM_magnitude += 0.5*rho*cl.reshape(101,)*planar_rot_w_squared.reshape(101,)*((y_space[i]-y_space[0])**2)*cr[i]*br 
        Fd_BEM_magnitude += 0.5*rho*cd.reshape(101,)*planar_rot_w_squared.reshape(101,)*((y_space[i]-y_space[0])**2)*cr[i]*br 
    
    Fl_BEM = np.zeros((timeline.shape[0], 3))
    Fd_BEM = np.zeros((timeline.shape[0], 3))
    for i in range(timeline.shape[0]):
        Fl_BEM[i,:] = (Fl_BEM_magnitude[i] * e_liftVectors[i])
        Fd_BEM[i,:] = (Fd_BEM_magnitude[i] * dragVectors_wing_g[i])

    Fx_QSM = Fl_BEM[:, 0]+Fd_BEM[:, 0]
    Fy_QSM = Fl_BEM[:, 1]+Fd_BEM[:, 1]
    Fz_QSM = Fl_BEM[:, 2]+Fd_BEM[:, 2]

    K_num = np.linalg.norm(Fx_QSM-Fx_CFD_interp(timeline)) + np.linalg.norm(Fz_QSM-Fz_CFD_interp(timeline))
    K_den = np.linalg.norm(Fx_CFD_interp(timeline) + np.linalg.norm(Fz_CFD_interp(timeline)))
    if K_den != 0: 
        K = K_num/K_den
    else:
        K = K_num
    if show_plots:
        graphAoA = np.linspace(-9, 90, 100)*(np.pi/180)
        gCl, gCd = getAerodynamicCoefficients(x, graphAoA)
        fig, ax = plt.subplots()
        ax.plot(np.degrees(graphAoA), gCl, label='Cl')
        ax.plot(np.degrees(graphAoA), gCd, label='Cd')
        ax.set_xlabel('AoA[°]')
        plt.legend()
        plt.show()

        plt.plot(timeline[:], Fx_QSM, label='Fx_QSM', color='red')
        plt.plot(timeline[:], Fz_QSM, label='Fz_QSM', color='blue')
        plt.plot(timeline[:], Fx_CFD_interp(timeline), label='Fx_CFD', linestyle = 'dashed', color='red')
        plt.plot(timeline[:], Fz_CFD_interp(timeline), label='Fz_CFD', linestyle = 'dashed', color='blue')
        plt.xlabel('t/T')
        plt.ylabel('Force')
        plt.title(f'Fx_QSM/Fx_CFD = {np.round(np.linalg.norm(Fx_QSM)/np.linalg.norm(Fx_CFD_interp(timeline)), 3)}; Fz_QSM/Fz_CFD = {np.round(np.linalg.norm(Fz_QSM)/np.linalg.norm(Fz_CFD_interp(timeline)), 3)}')
        plt.legend()
        plt.show()

        generatePlotsForKinematicsSequence(timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index, Fl_BEM, Fd_BEM)
    return K 

###optimization 
def main():
    timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index = kinematics()
    t, Fx_CFD, Fy_CFD, Fz_CFD = load_forces_data('forces_data_for_QSM.csv')
    # t, Fx_CFD, Fy_CFD, Fz_CFD = it.load_t_file('forces_data_for_QSM.csv')
    Fx_CFD_interp = interp1d(t, Fx_CFD, fill_value='extrapolate')
    Fy_CFD_interp = interp1d(t, Fy_CFD, fill_value='extrapolate')
    Fz_CFD_interp = interp1d(t, Fz_CFD, fill_value='extrapolate')
    t, alpha_CFD, phi_CFD, theta_CFD, alpha_dot_CFD, phi_dot_CFD, theta_dot_CFD = load_kinematics_data() 
    x_0 = [0.03433548, -0.01193863,  0.0338657,  -0.023361]
    bounds = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]
    optimize = True
    if optimize:
        start = time.time()
        optimization = opt.differential_evolution(F, args=(timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index, Fx_CFD_interp, Fy_CFD_interp, Fz_CFD_interp, Fx_CFD, Fy_CFD, Fz_CFD), bounds=bounds, x0=x_0, maxiter=20)
        x_final = optimization.x
        K_final = optimization.fun
        print('completed in:', round(time.time() - start, 3), ' seconds')
    else:
        x_final = [0.03433548, -0.01193863,  0.0338657,  -0.023361]
        K_final = 0.09349021020747196

    print('x0_final: ', x_final, 'K_final: ', K_final)
    F(x_final, timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index, Fx_CFD_interp, Fy_CFD_interp, Fz_CFD_interp, Fx_CFD, Fy_CFD, Fz_CFD, True)
main()