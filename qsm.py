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

#helper functions

#function to load the required forces data
def load_forces_data(file):
    #the items below are lists which must be transformed to np.array upon returning them
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
 
def parse_wing_file(wing_file, scale) -> np.ndarray:
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
    hinge_index = np.argmin(wingPoints[:, 1])
    wingtip_index = np.argmax(wingPoints[:, 1])
    #shift wing points at the first point 
    pivot = wingPoints[hinge_index]
    wingPoints -= pivot
    file_to_read.close() 
    return wingPoints * scale, hinge_index, wingtip_index

#global variables:

#wing's an array of discrete points.
wing_file = 'drosophilaMelanogasterWing.csv'
wingPoints, hinge_index, wingtip_index = parse_wing_file(wing_file, 0.001)

# #print wing to check positioning and indexing
# plt.scatter(wingPoints[:, 0], wingPoints[:, 1])
# i = 0
# for wingPoint in wingPoints:
#     plt.text(wingPoint[0], wingPoint[1], str(i))
#     i += 1
# plt.xlim([-4,4])
# plt.ylim([-4,4])
# plt.show()
# exit()


# spanwise normalization   
min_y = np.min(wingPoints[:, 1])
max_y = np.max(wingPoints[:, 1])
R_nonnormalized = max_y-min_y #unnormalized wing radius
e_wingPoints = wingPoints/R_nonnormalized 
min_y = np.min(e_wingPoints[:, 1])
max_y = np.max(e_wingPoints[:, 1])
e_R = max_y - min_y #e_R = 1



#load kinematics data by means of load_t_file function from insect_tools library 
kinematics_cfd = it.load_t_file('kinematics_musca_intact.t')
timeline = kinematics_cfd[:,0].flatten()
alphas = kinematics_cfd[:,8].flatten()
phis = kinematics_cfd[:,9].flatten()
thetas = kinematics_cfd[:,10].flatten()

#interpolate alpha, phi and theta  with respect to the original timeline
alphas_interp = interp1d(timeline.flatten(), alphas.flatten(), fill_value='extrapolate')
phis_interp = interp1d(timeline.flatten(), phis.flatten(), fill_value='extrapolate')
thetas_interp = interp1d(timeline.flatten(), thetas.flatten(), fill_value='extrapolate')

#timeline downsizing 
timeline = np.linspace(0, 1, 101)
nt = timeline.shape[0] #number of timesteps

#here all of the required variable arrays are created to match the size of the timeline 
#since for every timestep each variable must be computed. this will happen in 'generateSequence'  
alphas_dt_sequence = np.zeros((nt))
phis_dt_sequence = np.zeros((nt))
thetas_dt_sequence = np.zeros((nt))

strokePointsSequence = np.zeros((nt, wingPoints.shape[0], 3))
bodyPointsSequence = np.zeros((nt, wingPoints.shape[0], 3))
globalPointsSequence = np.zeros((nt, wingPoints.shape[0], 3))

rots_wing_b = np.zeros((nt, 3, 1))
rots_wing_w = np.zeros((nt, 3, 1))
rots_wing_g = np.zeros((nt, 3, 1))
planar_rots_wing_g = np.zeros((nt, 3, 1))

us_wing_w = np.zeros((nt, 3, 1))
us_wing_g = np.zeros((nt, 3, 1))
us_wing_g_magnitude = np.zeros((nt))

us_wind_w = np.zeros((nt, 3, 1))

AoA = np.zeros((nt, 1))
e_dragVectors_wing_g = np.zeros((nt, 3))
liftVectors = np.zeros((nt, 3))
e_liftVectors = np.zeros((nt, 3))

y_wing_g_sequence = np.zeros((nt, 3))
z_wing_g_sequence = np.zeros((nt, 3))

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

Fl = np.zeros((nt, 3))
Fd = np.zeros((nt, 3))
Frot = np.zeros((nt, 3))

Fl_magnitude = np.zeros(nt)
Fd_magnitude = np.zeros(nt)
Frot_magnitude = np.zeros(nt)

#this function calculates the chord length by splitting into 2 segments (LE and TE segment) and then interpolating along the y-axis 
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

#the convert_from_*_reference_frame_to_* functions convert points from one reference frame to another
#they take the points and the parameter list (angles) as arguments. 
#the function first calculates the rotation matrix and its transpose, and then multiplies each point with the tranpose since 
#by convention in this code we derotate as we start out with wing points and they must be converted down to global points. 
#this function returns the converted points as well as the rotation matrix and its tranpose 

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

#generate rot wing calculates the angular velocity of the wing in all reference frames, as well as the planar angular velocity {𝛀(φ,Θ)} 
#which will later be used to calculate the forces on the wing.  planar angular velocity {𝛀(φ,Θ)} comes from the decomposition of the motion
#into 'translational' and rotational components, with the rotational component beig defined as ⍺ (the one around the y-axis in our convention)
#this velocity is obtained by setting ⍺ to 0, as can be seen below
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

#since the absolute linear velocity of the wing depends both on time and on the position along the wing
#this function calculates only its position dependency
def generate_u_wing_g_position(rot_wing_g, y_wing_g):
    # #omega x point
    #both input vectors have to be reshaped to (1,3) to meet the requirements of np.cross (last axis of both vectors -> 2 or 3). to that end either reshape(1,3) or flatten() kommen in frage
    u_wing_g_position = np.cross(rot_wing_g, y_wing_g)
    return u_wing_g_position

#this function calculates the linear velocity of the wing in the wing reference frame
def generate_u_wing_w(u_wing_g, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix):
    rotationMatrix = np.matmul(np.matmul(bodyRotationMatrix, strokeRotationMatrix), wingRotationMatrix)
    u_wing_w = np.matmul(rotationMatrix, u_wing_g)
    return u_wing_w

def getWindDirectioninWingReferenceFrame(u_wind_g, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix): 
    u_wind_b = np.matmul(bodyRotationMatrix, u_wind_g.reshape(3,1))
    u_wind_s = np.matmul(strokeRotationMatrix, u_wind_b)
    u_wind_w = np.matmul(wingRotationMatrix, u_wind_s)
    return u_wind_w

#in this code AoA is defined as the arccos of the dot product between the unit vector along x direction and the unit vector of the absolute linear velocity
def getAoA(x_wing_g, e_u_wing_g):
    AoA = np.arccos(np.dot(x_wing_g, e_u_wing_g))
    return AoA

#alternative definition of AoA 
# def getAoA(drag_vector, x_wing_g):
#     #should be in the wing reference frame
#     AoA = np.arctan2(np.linalg.norm(np.cross(x_wing_g, drag_vector)), np.dot(-drag_vector, x_wing_g.reshape(3,1))) #rad
#     return AoA  

#this function computes the value of each variable for each timestep and stores them in arrays 
def generateSequence (start_time=0, number_of_timesteps=360, frequency=1, useCFDData=True):
    for timeStep in range(nt):
        global strokePointsSequence 
        t = timeline[timeStep]
        # parameter array: psi [0], beta[1], gamma[2], eta[3], phi[4], alpha[5], theta[6]
        # alphas_dt = alphas_dt_interp(t)
        # phis_dt = phis_dt_interp(t)
        # thetas_dt = thetas_dt_interp(t)

        #here the time derivatives of the angles are calculated by means of 1st order approximations
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

        #these are all the absolute unit vectors of the wing 
        #y_wing_g coincides with the tip only if R is normalized. 
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
        us_wing_g[timeStep, :] = u_wing_g.reshape(3,1)

        u_wing_w = generate_u_wing_w(u_wing_g.reshape(3,1), bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix)
        us_wing_w[timeStep, :] = u_wing_w

        u_wind_g = np.array([0, 0, 0]) #absolute wind velocity 
        #absolute mean flow velocity is defined as the (linear) sum of the absolute winG velocity and the absolute winD velocity 
        u_flight_g = u_wing_g + u_wind_g

        u_wind_w = getWindDirectioninWingReferenceFrame(u_flight_g, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix) - np.array(u_wing_w.reshape(3,1))
        us_wind_w[timeStep, :] = u_wind_w


        u_wing_g_magnitude = np.linalg.norm(u_wing_g)
        us_wing_g_magnitude[timeStep] = u_wing_g_magnitude

        if u_wing_g_magnitude != 0:  
            e_u_wing_g = u_wing_g/u_wing_g_magnitude
        else:
            e_u_wing_g = u_wing_g 
        e_dragVector_wing_g = -e_u_wing_g
        e_dragVectors_wing_g[timeStep, :] = e_dragVector_wing_g
    
        #lift. lift vector is multiplied with the sign of alpha to have their signs match 
        liftVector = np.cross(e_u_wing_g, y_wing_g.flatten())
        liftVector = liftVector*np.sign(alphas_interp(t)) # 
        liftVectors[timeStep, :] = liftVector

        aoa = getAoA(x_wing_g.reshape(1,3), e_u_wing_g.reshape(3,1)) #use this one for getAoA with arccos 
        # aoa = getAoA(e_dragVector_wing_g, x_wing_g.flatten()) #use this one for getAoA with arctan 
        AoA[timeStep, :] = aoa
        liftVector_magnitude = np.sqrt(liftVector[0, 0]**2 + liftVector[0, 1]**2 + liftVector[0, 2]**2)
        if liftVector_magnitude != 0: 
            e_liftVector = liftVector / liftVector_magnitude
        else:
            e_liftVector = liftVector
        e_liftVectors[timeStep, :] = e_liftVector

    #validation of our u_wing_g by means of a first order approximation
    #left and right derivative: 
    verifying_us_wing_g = np.zeros((nt, wingPoints.shape[0], 3))
    for timeStep in range(nt):
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

# run the live animation of the wing 
def generatePlotsForKinematicsSequence():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    anim = animation.FuncAnimation(fig, functools.partial(animationPlot, ax), frames=len(timeline), repeat=True)
    #anim.save('u&d_vectors.gif') 
    plt.show() 

def kinematics(): 
    #create figure 
    generateSequence(frequency=10, number_of_timesteps=360, useCFDData=True)
    #generatePlotsForKinematicsSequence(timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, wingtip_index, hinge_index)
############################################################################################################################################################################################
##% dynamics

def getAerodynamicCoefficients(x0, AoA): 
    deg2rad = np.pi/180.0 
    rad2deg = 180.0/np.pi
    
    AoA = rad2deg*AoA
    
    # Cl and Cd definitions from Dickinson 1999
    cl = x0[0] + x0[1]*np.sin( deg2rad*(2.13*AoA - 7.20) )
    cd = x0[2] + x0[3]*np.cos( deg2rad*(2.04*AoA - 9.82) )
    crot = x0[3]
    return cl, cd, crot

############################################################################################################################################################################################
##%% main 

from scipy.integrate import trapz, simpson
import scipy.optimize as opt
import time

#cost function which tells us how far off our QSM values are from the CFD ones
def F(x, show_plots=False):
    #global variable must be imported in order to modify them locally
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
    c = getChordLength(e_wingPoints, y_space) 
    rho = 1.225

    # #START OF ANALYTICAL VERSION
    # #to do the analytical run comment out lines 545 - 548 and 556 - 567 !!! 
    # #computation following Nakata 2015 eqns. 2.4a-c
    # c_interpolation = interp1d(y_space, c) #we create a function that interpolates our chord (c) w respect to our span (y_space)

    # #the following comes from defining lift/drag in the following way: dFl = 0.5*rho*cl*v^2*c*dr -> where v = linear velocity, c = chord length, dr = chord width
    # #v can be broken into 𝛀(φ,Θ)*r  (cf. lines 245-248). plugging that into our equation we get: dFl = 0.5*rho*cl*𝛀^2(φ,Θ)*r^2*c*dr (lift in each blade)
    # #integrating both sides, and pulling constants out of integrand on RHS: Fl = 0.5*rho*cl*𝛀^2(φ,Θ)*∫c*r^2*dr 
    # #our function def Cr2 then calculates the product of c and r^2 ; I (second moment of area) performs the integration of the product 
    # #drag is pretty much the same except that instead of cl we use cd: Fd = 0.5*rho*cd*𝛀^2(φ,Θ)*∫c*r^2*dr
    # #and the rotational force is defined as follows: Frot = 0.5*rho*crot*𝛀(φ,Θ)*∫c^2*r*dr

    # def Cr2(r): 
    #     return c_interpolation(r) * r**2
    # def C2r(r):
    #     return (c_interpolation(r)**2) * r
    
    # Ild = simpson(Cr2(y_space), y_space) #second moment of area for lift/drag calculations
    # Irot = simpson(C2r(y_space), y_space) #second moment of area for rotational force calculation 

    # planar_rots_wing_g_norm = np.linalg.norm(planar_rots_wing_g, axis=1)
    # rho = 1.225
    # Fl_magnitude = 0.5*rho*cl*(planar_rots_wing_g_norm**2)*Ild
    # Fd_magnitude = 0.5*rho*cd*(planar_rots_wing_g_norm**2)*Ild
    # Frot_magnitude = rho*crot*planar_rots_wing_g_norm.reshape(101,)*alphas_dt_sequence*Irot
    # #END OF ANALYTICAL VERSION 

    #START OF NUMERICAL VERSION
    #computation following Nakata 2015 eqns. 2.4a-c
    #thickness of each blade 
    dr = y_space[1]-y_space[0]

    Fl_magnitude = np.zeros(nt)
    Fd_magnitude = np.zeros(nt)
    Frot_magnitude = np.zeros(nt)
    planar_rots_wing_g = planar_rots_wing_g.reshape(101,3)

    # the numerical computation follows the same original equations as the analytical one, but instead of performing the integral, we calculate the forces
    # in each blade and then sum them up: dFl = 0.5*rho*cl*𝛀^2(φ,Θ)*r^2*c*dr, dFd = 0.5*rho*cd*𝛀^2(φ,Θ)*r^2*c*dr, dFrot = 0.5*rho*crot*𝛀(φ,Θ)*r*c^2*dr
    # calculation of the magnitude of the lift/drag/rotational force for each blade. each force is then summed up for each timestep and a (101,) array is returned.
    # each row represents a timestep and the value contained therein the total Fl/Fd for that time
    # this loop does the following: it loops over y_space (100, 1000, 10000 however many points the user sets) and for every point it computes the value 
    # for all timesteps for that point and ONLY then it moves on to the next point, computes all timesteps and so on, until it's done looping over y_space
    for i in range(y_space.shape[0]):
        #as previously discussed (cf. 245 - 248), for the numerical calculations of Fd, Fl, Frot we must use the absolute angular velocity that solely depends on phi and theta {𝛀(φ,Θ)} -> absolute planar angular velocity
        #now, since we are looping over the span (y_space) here, we have to calculate the absolute planar linear velocity for each blade by computing the cross product 
        #of the absolute planar angular velocity and the absolute radius of each blade (y_blade_g). 
        r = y_space[i]-y_space[0]
        y_blade_g = r*y_wing_g_sequence #(101,3)
        blade_planar_us_wing_g = np.cross(planar_rots_wing_g, y_blade_g, axis=1)
        blade_planar_us_wing_g_magnitude = np.linalg.norm(blade_planar_us_wing_g, axis=1)
        
        Frot_magnitude += rho*crot*blade_planar_us_wing_g_magnitude*alphas_dt_sequence*(c[i]**2)*dr
        Fl_magnitude += 0.5*rho*cl.reshape(nt,)*(blade_planar_us_wing_g_magnitude**2)*c[i]*dr
        Fd_magnitude += 0.5*rho*cd.reshape(nt,)*(blade_planar_us_wing_g_magnitude**2)*c[i]*dr
    #END OF NUMERICAL VERSION 

    # # # vector calculation of Fl, Fd, Frot. arrays of the form (101, 3) 
    for i in range(nt):
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
        plt.plot(timeline[:], -Fy_QSM, label='Fy_QSM', color='green')
        plt.plot(timeline[:], Fz_QSM, label='Fz_QSM', color='blue')
        plt.plot(timeline[:], Fx_CFD_interp(timeline), label='Fx_CFD', linestyle = 'dashed', color='red')
        plt.plot(timeline[:], Fy_CFD_interp(timeline), label='Fy_CFD', linestyle = 'dashed', color='green')
        plt.plot(timeline[:], Fz_CFD_interp(timeline), label='Fz_CFD', linestyle = 'dashed', color='blue')
        plt.xlabel('t/T')
        plt.ylabel('Force')
        plt.title(f'Fx_QSM/Fx_CFD = {np.round(np.linalg.norm(Fx_QSM)/np.linalg.norm(Fx_CFD_interp(timeline)), 3)}; Fz_QSM/Fz_CFD = {np.round(np.linalg.norm(Fz_QSM)/np.linalg.norm(Fz_CFD_interp(timeline)), 3)}')
        plt.legend()
        plt.show()

        generatePlotsForKinematicsSequence()
    return K 

#optimization by means of opt.differential_evolution which calculates the global minimum of our cost function (def F) and tells us 
#for what x_0 values/input this minimum is attained  
def main():
    kinematics()
    x_0 = [1.39072943, -0.46943661,  1.38872827, -0.94982812]
    bounds = [(-3, 3), (-3, 3), (-3, 3), (-3, 3)]
    optimize = True
    if optimize:
        start = time.time()
        optimization = opt.differential_evolution(F, bounds=bounds, x0=x_0, maxiter=20)
        x0_final = optimization.x
        K_final = optimization.fun
        print('completed in:', round(time.time() - start, 3), ' seconds')
    else:
        x0_final = [0.03433548, -0.01193863,  0.0338657,  -0.023361]
        K_final = 0.09349021020747196

    print('x0_final: ', x0_final, 'K_final: ', K_final)
    F(x0_final, True)

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
# with open('debug/profile.txt', 'w+') as f:
#     f.write(s.getvalue())