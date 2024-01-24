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
#global variables:
def load_kinematics_data(file): 
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

def load_forces_data(file):
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

wing_file = 'drosophilaMelanogasterWing.csv'
pivot_index = -8
wingPoints = parse_wing_file(wing_file, 0.001, pivot_index)
wingtip_index = 17

# spanwise normalization   
min_y = np.min(wingPoints[:, 1])
max_y = np.max(wingPoints[:, 1])
diff = max_y-min_y
wingPoints_norm = wingPoints/diff
min_y = np.min(wingPoints_norm[:, 1])
max_y = np.max(wingPoints_norm[:, 1])
R = max_y - min_y

u_wind_g = np.array([0, 0, 0])

timeline, alphas, phis, thetas, alphas_dt, phis_dt, thetas_dt = load_kinematics_data('kinematics_data_for_QSM.csv')

# kinematics_cfd = it.load_t_file('kinematics_musca_intact.t')
# timeline = kinematics_cfd[:,0].flatten()
# alphas = kinematics_cfd[:,8].flatten()
# phis = kinematics_cfd[:,9].flatten()
# thetas = kinematics_cfd[:,10].flatten()

# timeline, xc_body_g_x, xc_body_g_y, xc_body_g_z, psi, beta, gamma, eta_stroke, alphas, phis, thetas, alpha_r, phi_r, theta_r, rot_rel_l_w_x, rot_rel_l_w_y, rot_rel_l_w_z, rot_rel_r_w_x, rot_rel_r_w_y, rot_rel_r_w_z, rot_dt_l_w_x, rot_dt_l_w_y, rot_dt_l_w_z, rot_dt_r_w_x, rot_dt_r_w_y, rot_dt_r_w_z = it.load_t_file('kinematics_musca_intact.t')  
alphas_interp = interp1d(timeline.flatten(), alphas.flatten(), fill_value='extrapolate')
phis_interp = interp1d(timeline.flatten(), phis.flatten(), fill_value='extrapolate')
thetas_interp = interp1d(timeline.flatten(), thetas.flatten(), fill_value='extrapolate')

# alphas_dt_interp = interp1d(timeline.flatten(), alphas_dt.flatten(), fill_value='extrapolate')
# phis_dt_interp = interp1d(timeline.flatten(), phis_dt.flatten(), fill_value='extrapolate')
# thetas_dt_interp = interp1d(timeline.flatten(), thetas_dt.flatten(), fill_value='extrapolate')


timeline = np.linspace(0, 1, 101)

alphas_dt_sequence = np.zeros((timeline.shape[0]))
phis_dt_sequence = np.zeros((timeline.shape[0]))
thetas_dt_sequence = np.zeros((timeline.shape[0]))

strokePointsSequence = np.zeros((timeline.shape[0], wingPoints.shape[0], 3))
bodyPointsSequence = np.zeros((timeline.shape[0], wingPoints.shape[0], 3))
globalPointsSequence = np.zeros((timeline.shape[0], wingPoints.shape[0], 3))

rots_wing_b = np.zeros((timeline.shape[0], 3, 1))
rots_wing_w = np.zeros((timeline.shape[0], 3, 1))
rots_wing_g = np.zeros((timeline.shape[0], 3, 1))
planar_rots_wing_g = np.zeros((timeline.shape[0], 3, 1))

us_wing_w = np.zeros((timeline.shape[0], 3, 1))
us_wing_g = np.zeros((timeline.shape[0], 3, 1))
us_wing_g_magnitude = np.zeros((timeline.shape[0]))

us_wind_w = np.zeros((timeline.shape[0], 3, 1))
AoA = np.zeros((timeline.shape[0], 1))
e_dragVectors_wing_g = np.zeros((timeline.shape[0], 3))
liftVectors = np.zeros((timeline.shape[0], 3))
e_liftVectors = np.zeros((timeline.shape[0], 3))

y_wing_g_sequence = np.zeros((timeline.shape[0], 3))
z_wing_g_sequence = np.zeros((timeline.shape[0], 3))

delta_t = timeline[1] - timeline[0]

t, Fx_CFD, Fy_CFD, Fz_CFD = load_forces_data('forces_data_for_QSM.csv')

# forces_CFD = it.load_t_file('forces.t')
# t = forces_CFD[:, 0]
# Fx_CFD = forces_CFD[:, 1]
# Fy_CFD = forces_CFD[:, 2]
# Fz_CFD = forces_CFD[:, 3]

Fx_CFD_interp = interp1d(t, Fx_CFD, fill_value='extrapolate')
Fy_CFD_interp = interp1d(t, Fy_CFD, fill_value='extrapolate')
Fz_CFD_interp = interp1d(t, Fz_CFD, fill_value='extrapolate')
t, alpha_CFD, phi_CFD, theta_CFD, alpha_dot_CFD, phi_dot_CFD, theta_dot_CFD = load_kinematics_data('kinematics_data_for_QSM.csv')  

Fl = np.zeros((timeline.shape[0], 3))
Fd = np.zeros((timeline.shape[0], 3))
Frot = np.zeros((timeline.shape[0], 3))

Fl_magnitude = np.zeros(timeline.shape[0])
Fd_magnitude = np.zeros(timeline.shape[0])
Frot_magnitude = np.zeros(timeline.shape[0])


def getChordLength(wingPoints, y_coordinate):
    #get the division in wing segments (leading and trailing)
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
    rot_wing_w = np.matmul(wingRotationMatrix, rot_wing_s)
    rot_wing_b = np.matmul(strokeRotationMatrixTrans, rot_wing_s)
    rot_wing_g = np.matmul(bodyRotationMatrixTrans, rot_wing_b)
    planar_rot_wing_g = np.matmul(bodyRotationMatrixTrans, np.matmul(strokeRotationMatrixTrans, np.matmul(phiMatrixTrans, (vector_phi_dt+np.matmul(thetaMatrixTrans, (vector_theta_dt))))))
    return rot_wing_g, rot_wing_b, rot_wing_w, planar_rot_wing_g #these are all (3x1) vectors 

def generate_u_wing_g_position(rot_wing_g, y_wing_g):
    # #omega x point
    #both input vectors have to be reshaped to (1,3) to meet the requirements of np.cross (last axis of both vectors -> 2 or 3). to that end either reshape(1,3) or flatten() kommen in frage
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

def generateSequence (start_time=0, number_of_timesteps=360, frequency=1, useCFDData=True):
    
    for timeStep in range(timeline.shape[0]):
        global strokePointsSequence 
        t = timeline[timeStep]
        # parameter array: psi [0], beta[1], gamma[2], eta[3], phi[4], alpha[5], theta[6]
        # alphas_dt = alphas_dt_interp(t)
        # phis_dt = phis_dt_interp(t)
        # thetas_dt = thetas_dt_interp(t)

        alphas_dt = (alphas_interp(t+delta_t) - alphas_interp(t)) / delta_t
        phis_dt = (phis_interp(t+delta_t) - phis_interp(t)) / delta_t
        thetas_dt = (thetas_interp(t+delta_t) - thetas_interp(t)) / delta_t

        alphas_dt_sequence[timeStep] = alphas_dt
        phis_dt_sequence[timeStep] = phis_dt
        thetas_dt_sequence[timeStep] = thetas_dt

        parameters = [0, 0, 0, -80*np.pi/180, phis_interp(t), alphas_interp(t), thetas_interp(t)] # 7 angles in radians! #without alphas[timeStep] any rotation around any y axis through an angle of pi/2 gives an error! 
        parameters_dt = [0, 0, 0, 0, phis_dt, alphas_dt, thetas_dt]
    
        strokePoints, wingRotationMatrix, wingRotationMatrixTrans = convert_from_wing_reference_frame_to_stroke_plane(wingPoints, parameters)
        bodyPoints, strokeRotationMatrix, strokeRotationMatrixTrans = convert_from_stroke_plane_to_body_reference_frame(strokePoints, parameters)
        globalPoints, bodyRotationMatrix, bodyRotationMatrixTrans = convert_from_body_reference_frame_to_global_reference_frame(bodyPoints, parameters)
        
        strokePointsSequence[timeStep, :] = strokePoints
        bodyPointsSequence[timeStep, :] = bodyPoints
        globalPointsSequence[timeStep, :] = globalPoints

        # y_wing_g coincides with the tip only if R is normalized. 
        x_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[1], [0], [0]]))
        y_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[0], [1], [0]]))
        z_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[0], [0], [1]]))
    
        y_wing_g_sequence[timeStep, :] = y_wing_g.flatten()
        z_wing_g_sequence[timeStep, :] = z_wing_g.flatten()
        
        rot_wing_g, rot_wing_b, rot_wing_w, planar_rot_wing_g = generate_rot_wing(wingRotationMatrix, bodyRotationMatrixTrans, strokeRotationMatrixTrans, parameters[4], parameters_dt[4], parameters[5], 
                                    parameters_dt[5], parameters[6], parameters_dt[6])
        
        rots_wing_b[timeStep, :] = rot_wing_b
        rots_wing_w[timeStep, :] = rot_wing_w
        rots_wing_g[timeStep, :] = rot_wing_g
        planar_rots_wing_g[timeStep, :] = planar_rot_wing_g

        u_wing_g = generate_u_wing_g_position(rot_wing_g.reshape(1,3), y_wing_g.reshape(1,3))

        u_flight_g = u_wing_g + u_wind_g

        u_wing_w = generate_u_wing_w(u_wing_g.reshape(3,1), bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix)
        us_wing_g[timeStep, :] = u_wing_g.reshape(3,1)
        us_wing_w[timeStep, :] = u_wing_w
        u_wind_w = getWindDirectioninWingReferenceFrame(u_flight_g, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix) - np.array(u_wing_w.reshape(3,1))
        us_wind_w[timeStep, :] = u_wind_w

        # u_wing_g_magnitude = np.sqrt(u_wing_g[0, 0]**2 + u_wing_g[0, 1]**2 + u_wing_g[0, 2]**2)
        u_wing_g_magnitude = np.linalg.norm(u_wing_g)
        us_wing_g_magnitude[timeStep] = u_wing_g_magnitude

        if u_wing_g_magnitude != 0:  
            e_u_wing_g = u_wing_g/u_wing_g_magnitude
        else:
            e_u_wing_g = u_wing_g 
        e_dragVector_wing_g = -e_u_wing_g
        e_dragVectors_wing_g[timeStep, :] = e_dragVector_wing_g
    
        #lift 
        liftVector = np.cross(e_u_wing_g, y_wing_g.flatten())
        liftVector = liftVector*np.sign(alphas_interp(t))
        liftVectors[timeStep, :] = liftVector

        aoa = getAoA(x_wing_g.reshape(1,3), e_u_wing_g.reshape(3,1)) #use this one for getAoA thru arccos 
        # aoa = getAoA(e_dragVector_wing_g, x_wing_g.flatten()) #use this one for getAoA thru arctan 
        AoA[timeStep, :] = aoa
        liftVector_magnitude = np.sqrt(liftVector[0, 0]**2 + liftVector[0, 1]**2 + liftVector[0, 2]**2)
        if liftVector_magnitude != 0: 
            e_liftVector = liftVector / liftVector_magnitude
        else:
            e_liftVector = liftVector
        e_liftVectors[timeStep, :] = e_liftVector

    #validation of our u_wing_g: 
    #left and right derivative: 
    verifying_us_wing_g = np.zeros((timeline.shape[0], wingPoints.shape[0], 3))
    for timeStep in range(timeline.shape[0]):
        currentGlobalPoint = globalPointsSequence[timeStep]
        leftGlobalPoint = globalPointsSequence[timeStep-1]
        rightGlobalPoint = globalPointsSequence[(timeStep+1)%len(timeline)]
        LHD = (currentGlobalPoint - leftGlobalPoint) / delta_t
        RHD = (rightGlobalPoint - currentGlobalPoint) / delta_t
        verifying_us_wing_g[timeStep, :] = (LHD+RHD)/2
    verifying_us_wing_g = verifying_us_wing_g  

def animationPlot(ax, timeStep): 
    #get point set by timeStep number
    points = globalPointsSequence[timeStep] #pointsSequence can either be global, body, stroke 
    #clear the current axis 
    ax.cla()

    # extract the x, y and z coordinates 
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    #axis limit
    a = 4

    trajectory = np.array(globalPointsSequence)[:, wingtip_index]
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
    e_dragVector_wing_g = e_dragVectors_wing_g[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], e_dragVector_wing_g[0], e_dragVector_wing_g[1], e_dragVector_wing_g[2], color='green', label=r'$\overrightarrow{d}^{(g)}_w$' )
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
    ax.set_title(f'Timestep: {timeStep} \n⍺: {np.round(np.degrees(alphas[timeStep]), 2)} \nAoA: {np.round(np.degrees(AoA[timeStep]), 2)} \nFl: {np.round(Fl[timeStep], 4)} \nFd: {np.round(Fd[timeStep], 4)}')
    
def generatePlotsForKinematicsSequence():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    anim = animation.FuncAnimation(fig, functools.partial(animationPlot, ax), frames=len(timeline), repeat=True)
    #anim.save('u&d_vectors.gif') 
    plt.show() 

def kinematics(): 
    #create figure 
    generateSequence(frequency=10, number_of_timesteps=360, useCFDData=True)
    #generatePlotsForKinematicsSequence(timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, wingtip_index, pivot_index)
############################################################################################################################################################################################
##% dynamics

def getAerodynamicCoefficients(x0, AoA): 
    deg2rad = np.pi/180.0 
    rad2deg = 180.0/np.pi
    
    AoA = rad2deg*AoA
    
    cl = x0[0] + x0[1]*np.sin( deg2rad*(2.13*AoA - 7.20) )
    cd = x0[2] + x0[3]*np.cos( deg2rad*(2.04*AoA - 9.82) )
    crot = x0[3]
    return cl, cd, crot

############################################################################################################################################################################################
##%% main 

from scipy.integrate import trapz
import scipy.optimize as opt
import time


def F(x, show_plots=False):
    global Fl_magnitude, Fd_magnitude, Frot_magnitude, planar_rots_wing_g
    cl, cd, crot = getAerodynamicCoefficients(x, np.array(AoA))
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
        

    # chord calculation 
    y_space = np.linspace(min_y, max_y, 100)
    c = getChordLength(wingPoints_norm, y_space) 
    # c_mean = np.mean(c)

    # planar_rot_wing_g_squared = rots_wing_g[:, 0]**2 + rots_wing_g[:, 2]**2 #planar angular velocity 𝛀(φ, Θ)

    rho = 1.225
    dr = y_space[1]-y_space[0]
    # c = c / c_mean
    nt = timeline.shape[0]
    # rots_wing_g_magnitude = np.linalg.norm(rots_wing_g, axis=1)

    Fl_magnitude = np.zeros(timeline.shape[0])
    Fd_magnitude = np.zeros(timeline.shape[0])
    Frot_magnitude = np.zeros(timeline.shape[0])
    planar_rots_wing_g = planar_rots_wing_g.reshape(101,3)
    planar = np.zeros(timeline.shape[0])

    #calculation of the magnitude of the lift/drag force for each blade. each force is then summed up for each timestep and a (101,) array is returned.
    #each row represents a timestep and the value contained therein the total Fl/Fd for that time
    #this loop does the following: it loops over y_space (100, 1000, 10000 however many points the user sets) and for every point it computes the value 
    #for all timesteps for that point and ONLY then it moves on to the next point, computes all timesteps and so on, until it's done looping over y_space
    for i in range(y_space.shape[0]):
        r = y_space[i]-y_space[0]
        y_blade_g = r*y_wing_g_sequence #(101,3)
        blade_planar_us_wing_g = np.cross(planar_rots_wing_g, y_blade_g, axis=1)
        blade_planar_us_wing_g_magnitude = np.linalg.norm(blade_planar_us_wing_g, axis=1)

        # Fl_magnitude += 0.5*rho*cl.reshape(nt,)*(us_wing_g_magnitude**2)*c[i]*dr 
        # Fd_magnitude += 0.5*rho*cd.reshape(nt,)*(us_wing_g_magnitude**2)*c[i]*dr 
        # Frot_magnitude += rho*crot*us_wing_g_magnitude*alphas_dt_sequence*(c[i]**2)*dr
        # Frot_magnitude += rho*crot*us_wing_g_magnitude*alphas_dt_interp(timeline)*(c[i]**2)*dr
        
        # Frot_magnitude += rho*crot*blade_planar_us_wing_g_magnitude*alphas_dt_sequence*(c[i]**2)*dr
        Fl_magnitude += 0.5*rho*cl.reshape(nt,)*(blade_planar_us_wing_g_magnitude**2)*c[i]*dr 
        Fd_magnitude += 0.5*rho*cd.reshape(nt,)*(blade_planar_us_wing_g_magnitude**2)*c[i]*dr 
        # planar += (blade_planar_us_wing_g_magnitude**2)*c[i]*dr 
        
        # Frot_magnitude += rho*crot*(us_wing_g_magnitude)*rots_wing_g_magnitude.reshape(nt,)*(c_mean**2)*R*(y_space[i]-y_space[0])*(c_hat[i]**2)*dr # based on Sane 2002 page 1091: r_hat = y_space[i]-y_space[0] ; dr_hat = dr 

    # writeArraytoFile(Fl_magnitude, 'Fl_magnitude_num.txt')
    # writeArraytoFile(Fd_magnitude, 'Fd_magnitude_num.txt')
    # writeArraytoFile(planar , 'planar_num.txt')
    
    # # # vector calculation of the lift and drag forces. arrays of the form (101, 3) 
    for i in range(timeline.shape[0]):
        Fl[i,:] = (Fl_magnitude[i] * e_liftVectors[i])
        Fd[i,:] = (Fd_magnitude[i] * e_dragVectors_wing_g[i])
        Frot[i,:] = (Frot_magnitude[i] * z_wing_g_sequence[i])

    Fx_QSM = Fl[:, 0]+Fd[:, 0]+Frot[:, 0]
    Fy_QSM = Fl[:, 1]+Fd[:, 1]+Frot[:, 1]
    Fz_QSM = Fl[:, 2]+Fd[:, 2]+Frot[:, 2]

    K_num = np.linalg.norm(Fx_QSM-Fx_CFD_interp(timeline)) + np.linalg.norm(Fz_QSM-Fz_CFD_interp(timeline))
    K_den = np.linalg.norm(Fx_CFD_interp(timeline)) + np.linalg.norm(Fz_CFD_interp(timeline))
    if K_den != 0: 
        K = K_num/K_den
    else:
        K = K_num
    if show_plots:
        graphAoA = np.linspace(-9, 90, 100)*(np.pi/180)
        gCl, gCd, gCrot = getAerodynamicCoefficients(x, graphAoA)
        fig, ax = plt.subplots()
        ax.plot(np.degrees(graphAoA), gCl, label='Cl')
        ax.plot(np.degrees(graphAoA), gCd, label='Cd')
        # ax.plot(np.degrees(graphAoA), gCrot*np.ones_like(gCl), label='Crot')
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

        generatePlotsForKinematicsSequence()
    return K 

###optimization 
def main():
    kinematics()
    x_0 = [0.225, 1.58,  1.92,  -1.55]
    bounds = [(-3, 3), (-3, 3), (-3, 3), (-3, 3)]
    optimize = True
    if optimize:
        start = time.time()
        optimization = opt.differential_evolution(F, bounds=bounds, x0=x_0, maxiter=20)
        x_final = optimization.x
        K_final = optimization.fun
        print('completed in:', round(time.time() - start, 3), ' seconds')
    else:
        x_final = [0.03433548, -0.01193863,  0.0338657,  -0.023361]
        K_final = 0.09349021020747196

    print('x0_final: ', x_final, 'K_final: ', K_final)
    F(x_final, True)

# import cProfile
# import pstats
# import io
# profile = cProfile.Profile()
# profile.enable()
main()
# profile.disable()
# s = io.StringIO()
# ps = pstats.Stats(profile, stream=s).sort_stats('cumulative') # tottime
# ps.print_stats()
# with open('profile.txt', 'w+') as f:
#     f.write(s.getvalue())