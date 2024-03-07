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
from python_tools_master import insect_tools as it 
from python_tools_master import wabbit_tools as wt
from debug import writeArraytoFile

#global variables:
isLeft = wt.get_ini_parameter('cfd_run/PARAMS.ini', 'Insects', 'LeftWing', dtype=bool)
wingShape = wt.get_ini_parameter('cfd_run/PARAMS.ini', 'Insects', 'WingShape', dtype=str)
if 'from_file' in wingShape:
    wingShape_file = os.path.join('cfd_run', wingShape.replace('from_file::', ''))
time_max = wt.get_ini_parameter('cfd_run/PARAMS.ini', 'Time', 'time_max', dtype=float)
kinematics_file = wt.get_ini_parameter('cfd_run/PARAMS.ini', 'Insects', 'FlappingMotion_right', dtype=str)
if 'from_file' in kinematics_file:
    kinematics_file = os.path.join('cfd_run', kinematics_file.replace('from_file::', ''))

xc, yc = it.visualize_wing_shape_file(wingShape_file)
zc = np.zeros_like(xc)
wingPoints = np.vstack([xc, yc, zc])
wingPoints = np.transpose(wingPoints)
hinge_index = np.argmin(wingPoints[:, 1])
wingtip_index = np.argmax(wingPoints[:, 1])

# #print wing to check positioning and indexing
# plt.figure()
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
R = max_y-min_y #nonnormalized wing radius
e_wingPoints = wingPoints/R #normalization of the wing points
min_y = np.min(e_wingPoints[:, 1])
max_y = np.max(e_wingPoints[:, 1])
e_R = max_y - min_y #e_R = 1

#load kinematics data by means of eval_angles_kinematics_file function from insect_tools library 
timeline, phis, alphas, thetas = it.eval_angles_kinematics_file(fname=kinematics_file, time=np.linspace(0.0, 1.0, 101)[:-1])
phis = np.radians(phis)
alphas = np.radians(alphas)
thetas = np.radians(thetas)

#timeline reassignment. when the timeline from the function eval_angles_kinematics_file is used 
#the last point of the timeline is the same as the first one (so timeline[0] = timeline[-1])
#to solve this we can either do a variable reassigment or remove the last entry in timeline. the second method
#leaves you with one less point in the array. the first method is preferred. 
timeline = np.linspace(0, 1, timeline.shape[0])
nt = timeline.shape[0] #number of timesteps

#here all of the required variable arrays are created to match the size of the timeline 
#since for every timestep each variable must be computed. this will happen in 'generateSequence'  
alphas_dt_sequence = np.zeros((nt))
phis_dt_sequence = np.zeros((nt))
thetas_dt_sequence = np.zeros((nt))

alphas_dt_dt_sequence = np.zeros((nt))
phis_dt_dt_sequence = np.zeros((nt))
thetas_dt_dt_sequence = np.zeros((nt))

strokePointsSequence = np.zeros((nt, wingPoints.shape[0], 3))
bodyPointsSequence = np.zeros((nt, wingPoints.shape[0], 3))
globalPointsSequence = np.zeros((nt, wingPoints.shape[0], 3))

rots_wing_b = np.zeros((nt, 3, 1))
rots_wing_s = np.zeros((nt, 3, 1))
rots_wing_w = np.zeros((nt, 3, 1))
rots_wing_g = np.zeros((nt, 3, 1))

planar_rots_wing_s = np.zeros((nt, 3, 1))
planar_rots_wing_w = np.zeros((nt, 3, 1))
planar_rots_wing_g = np.zeros((nt, 3, 1))

planar_rot_acc_wing_g = np.zeros((nt, 3, 1))
planar_rot_acc_wing_w = np.zeros((nt, 3, 1))
planar_rot_acc_wing_s = np.zeros((nt, 3, 1))

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

z_wing_w_sequence = np.zeros((nt, 3))

Zalpha = np.zeros((nt, 3))
e_Fam = np.zeros((nt, 3))

wingRotationMatrix_sequence = np.zeros((nt, 3, 3))
wingRotationMatrixTrans_sequence = np.zeros((nt, 3, 3))
strokeRotationMatrix_sequence = np.zeros((nt, 3, 3))
strokeRotationMatrixTrans_sequence = np.zeros((nt, 3, 3))
bodyRotationMatrix_sequence = np.zeros((nt, 3, 3))
bodyRotationMatrixTrans_sequence = np.zeros((nt, 3, 3))

r = np.zeros((nt, 3))

delta_t = timeline[1] - timeline[0]

forces_CFD = it.load_t_file('cfd_run/forces_rightwing.t', T0=[1.0,2.0])
t = forces_CFD[:, 0]-1.0
Fx_CFD = forces_CFD[:, 1]
Fy_CFD = forces_CFD[:, 2]
Fz_CFD = forces_CFD[:, 3]

moments_CFD = it.load_t_file('cfd_run/moments_rightwing.t', T0=[1.0, 2.0])
#no need to read in the time again as it's the same from the forces file
Mx_CFD = moments_CFD[:, 1]
My_CFD = moments_CFD[:, 2]
Mz_CFD = moments_CFD[:, 3]
# M_CFD = moments_CFD[:, 1:4]

if isLeft == False: 
    Fy_CFD = -Fy_CFD
    My_CFD = -My_CFD
    print('The data correspond to the right wing. They will be adjusted to follow the left wing convention in this code.')
else: 
    print('The data correspond to the left wing')

if np.round(forces_CFD[-1, 0],3) != time_max: 
    raise ValueError('CFD cycle number does not match that the actual run. Check your PARAMS, forces and moments files\n')

print('The number of cycles is ', time_max, '. The forces and moments data were however only sampled for ', np.round(t[-1]), ' cycle(s)') #a cycle is defined as 1 downstroke + 1 upstroke ; cycle duration is 1.0 seconds. 

Fx_CFD_interp = interp1d(t, Fx_CFD, fill_value='extrapolate')
Fy_CFD_interp = interp1d(t, Fy_CFD, fill_value='extrapolate')
Fz_CFD_interp = interp1d(t, Fz_CFD, fill_value='extrapolate')

Mx_CFD_interp = interp1d(t, Mx_CFD,fill_value='extrapolate')
My_CFD_interp = interp1d(t, My_CFD,fill_value='extrapolate')
Mz_CFD_interp = interp1d(t, Mz_CFD,fill_value='extrapolate')

Fl = np.zeros((nt, 3))
Fd = np.zeros((nt, 3))
Frot = np.zeros((nt, 3))
Fam = np.zeros((nt, 3))

Fl_magnitude = np.zeros(nt)
Fd_magnitude = np.zeros(nt)
Frot_magnitude = np.zeros(nt)
Fam_magnitude = np.zeros(nt)


#this function calculates the chord length by splitting into 2 segments (LE and TE segment) and then interpolating along the y-axis 
def getChordLength(wingPoints, y_coordinate):
    #get the division in wing segments (leading and trailing)
    split_index = wingtip_index
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
    planar_rot_wing_s = np.matmul(phiMatrixTrans, (vector_phi_dt+np.matmul(thetaMatrixTrans, (vector_theta_dt))))
    planar_rot_wing_w = np.matmul(wingRotationMatrix, np.matmul(phiMatrixTrans, (vector_phi_dt+np.matmul(thetaMatrixTrans, (vector_theta_dt)))))
    planar_rot_wing_g = np.matmul(bodyRotationMatrixTrans, np.matmul(strokeRotationMatrixTrans, np.matmul(phiMatrixTrans, (vector_phi_dt+np.matmul(thetaMatrixTrans, (vector_theta_dt))))))
    return rot_wing_g, rot_wing_b, rot_wing_s, rot_wing_w, planar_rot_wing_g, planar_rot_wing_s, planar_rot_wing_w #these are all (3x1) vectors 

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

def getLever(F, M): 
#to find the lever we need to solve for r: M = r x F. however since no inverse of the cross product exists, we have to use the vector triple product
#what we get is: r = M x F/norm(F)^2 + t*F ; where * is the dot product and t is any constant. for this equation to be valid, F and M must be orthogonal to each other, 
#such that F*M = 0
    for i in range(nt): 
        F_norm = (np.linalg.norm(F, axis=1))
        if F_norm[i] != 0:
            r[i, :] = np.cross(M[i, :], F[i, :])/F_norm[i]**2 + 1*F[i, :] #here i take t = 1 
        else: 
            r[i, :] = 0 
    return r 

#alternative definition of AoA 
# def getAoA(drag_vector, x_wing_g):
#     #should be in the wing reference frame
#     AoA = np.arctan2(np.linalg.norm(np.cross(x_wing_g, drag_vector)), np.dot(-drag_vector, x_wing_g.reshape(3,1))) #rad
#     return AoA  

#this function computes the value of each variable for each timestep and stores them in arrays 
def generateSequence():
    for timeStep in range(nt):
        global strokePointsSequence, planar_rots_wing_w, planar_rot_position_w
        # parameter array: psi [0], beta[1], gamma[2], eta[3], phi[4], alpha[5], theta[6]
        # alphas_dt = alphas_dt_interp(t)
        # phis_dt = phis_dt_interp(t)
        # thetas_dt = thetas_dt_interp(t)

        #here the 1st time derivatives of the angles are calculated by means of 2nd order central difference approximations
        alphas_dt = (alphas[(timeStep+1)%nt] - alphas[timeStep-1]) / (2*delta_t) #here we compute the modulus of (timestep+1) and nt to prevent overflowing. central difference 
        phis_dt = (phis[(timeStep+1)%nt] - phis[timeStep-1]) / (2*delta_t)
        thetas_dt = (thetas[(timeStep+1)%nt] - thetas[timeStep-1]) / (2*delta_t)
        
        # phis_dt = (phis[(timeStep+1)%nt] - phis[timeStep]) / (delta_t) #1dt order forward difference approximation of 1st derivative of phi
        # thetas_dt = (thetas[(timeStep+1)%nt] - thetas[timeStep]) / (delta_t)

        parameters = [0, 0, 0, -80*np.pi/180, phis[timeStep], alphas[timeStep], thetas[timeStep]] # 7 angles in radians! #without alphas[timeStep] any rotation around any y axis through an angle of pi/2 gives an error! 
        parameters_dt = [0, 0, 0, 0, phis_dt, alphas_dt, thetas_dt]
        
        strokePoints, wingRotationMatrix, wingRotationMatrixTrans = convert_from_wing_reference_frame_to_stroke_plane(wingPoints, parameters)
        bodyPoints, strokeRotationMatrix, strokeRotationMatrixTrans = convert_from_stroke_plane_to_body_reference_frame(strokePoints, parameters)
        globalPoints, bodyRotationMatrix, bodyRotationMatrixTrans = convert_from_body_reference_frame_to_global_reference_frame(bodyPoints, parameters)
        
        strokePointsSequence[timeStep, :] = strokePoints
        bodyPointsSequence[timeStep, :] = bodyPoints
        globalPointsSequence[timeStep, :] = globalPoints

        wingRotationMatrix_sequence[timeStep, :] = wingRotationMatrix
        wingRotationMatrixTrans_sequence[timeStep, :] = wingRotationMatrixTrans
        strokeRotationMatrix_sequence[timeStep, :] = strokeRotationMatrix
        strokeRotationMatrixTrans_sequence[timeStep, :] = strokeRotationMatrixTrans
        bodyRotationMatrix_sequence[timeStep, :] = bodyRotationMatrix
        bodyRotationMatrixTrans_sequence[timeStep, :] = bodyRotationMatrixTrans

        #these are all the absolute unit vectors of the wing 
        #y_wing_g coincides with the tip only if R is normalized. 
        x_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[1], [0], [0]]))
        y_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[0], [1], [0]]))
        z_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[0], [0], [1]]))

        y_wing_g_sequence[timeStep, :] = y_wing_g.flatten()
        z_wing_g_sequence[timeStep, :] = z_wing_g.flatten()

        z_wing_w = np.array([[0], [0], [1]])
        z_wing_w_sequence[timeStep, :] = z_wing_w.reshape(3,)

        rot_wing_g, rot_wing_b, rot_wing_s, rot_wing_w, planar_rot_wing_g, planar_rot_wing_s, planar_rot_wing_w = generate_rot_wing(wingRotationMatrix, bodyRotationMatrixTrans, strokeRotationMatrixTrans, parameters[4], parameters_dt[4], parameters[5], 
                                    parameters_dt[5], parameters[6], parameters_dt[6])
        
        rots_wing_b[timeStep, :] = rot_wing_b
        rots_wing_s[timeStep, :] = rot_wing_s
        rots_wing_w[timeStep, :] = rot_wing_w
        rots_wing_g[timeStep, :] = rot_wing_g

        planar_rots_wing_s[timeStep, :] = planar_rot_wing_s
        planar_rots_wing_w[timeStep, :] = planar_rot_wing_w
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
        liftVector = liftVector*np.sign(alphas[timeStep]) # 
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

        alphas_dt_sequence[timeStep] = alphas_dt
        phis_dt_sequence[timeStep] = phis_dt
        thetas_dt_sequence[timeStep] = thetas_dt

        alphas_dt_dt = (alphas[(timeStep+1)%nt] - 2*alphas[timeStep] + alphas[timeStep-1]) / (delta_t**2) #2nd order central difference approximation of 2nd derivative of alpha
        phis_dt_dt = (phis[(timeStep+1)%nt] - 2*phis[timeStep] + phis[timeStep-1]) / (delta_t**2) #2nd order central difference approximation of 2nd derivative of phi
        thetas_dt_dt = (thetas[(timeStep+1)%nt] - 2*thetas[timeStep] + thetas[timeStep-1]) / (delta_t**2) #2nd order central difference approximation of 2nd derivative of theta

        # phis_dt_dt = (phis_dt_sequence[(timeStep+1)%nt] - phis_dt_sequence[timeStep-1]) / (2*delta_t) #2nd order central difference approximation of 1st derivative of phi_dt
        # thetas_dt_dt = (thetas_dt_sequence[(timeStep+1)%nt] - thetas_dt_sequence[timeStep-1]) / (2*delta_t) #2nd order central difference approximation of 1st derivative of theta_dt

       
        alphas_dt_dt_sequence[timeStep] = alphas_dt_dt
        phis_dt_dt_sequence[timeStep] = phis_dt_dt
        thetas_dt_dt_sequence[timeStep] = thetas_dt_dt
        
        planar_rot_acc_wing_s[timeStep, :] = [[phis_dt_dt_sequence[timeStep]], 
                                              [-thetas_dt_dt_sequence[timeStep]*np.sin(phis[timeStep]) - thetas_dt_sequence[timeStep]*np.cos(phis[timeStep])*phis_dt_sequence[timeStep]],
                                                [thetas_dt_dt_sequence[timeStep]*np.cos(phis[timeStep]) + thetas_dt_sequence[timeStep]*np.sin(phis[timeStep])*phis_dt_sequence[timeStep]]]
        planar_rot_acc_wing_w[timeStep, :] = np.matmul(wingRotationMatrix, planar_rot_acc_wing_s[timeStep])
        
        # planar_rot_acc_wing_w[timeStep, :] = ((planar_rots_wing_w[(timeStep+1)%nt] - planar_rots_wing_w[timeStep-1])) / (2*delta_t)
        # planar_rot_acc_wing_w[timeStep, :] = ((planar_rots_wing_w[(timeStep+1)%nt] - planar_rots_wing_w[timeStep])) / (delta_t)

        #validation of our u_wing_g by means of a first order approximation
        #left and right derivative: 
        verifying_us_wing_g = np.zeros((nt, wingPoints.shape[0], 3))
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
    a = 2

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
    
    #velocity at wingtip 
    u_wing_g = us_wing_g[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], u_wing_g[0], u_wing_g[1], u_wing_g[2], color='orange', label=r'$\overrightarrow{u}^{(g)}_w$' )
    
    #drag
    e_dragVector_wing_g = e_dragVectors_wing_g[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], e_dragVector_wing_g[0], e_dragVector_wing_g[1], e_dragVector_wing_g[2], color='green', label=r'$\overrightarrow{e_D}^{(g)}_w$' )
    
    #lift 
    liftVector = e_liftVectors[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], liftVector[0], liftVector[1], liftVector[2], color='blue', label=r'$\overrightarrow{e_L}^{(g)}_w$')
    
    #z_wing_g 
    z_wing_g_sequence_plot = z_wing_g_sequence[timeStep]*np.sign(alphas[timeStep])
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], z_wing_g_sequence_plot[0], z_wing_g_sequence_plot[1], z_wing_g_sequence_plot[2], color='red', label=r'$\overrightarrow{e_{F_{am}}}^{(g)}_w$')

    # #e_Fam 
    # e_Fam_plot = e_Fam[timeStep]
    # ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], e_Fam_plot[0], e_Fam_plot[1], e_Fam_plot[2], color='red', label=r'$\overrightarrow{e_{F_{am}}}^{(g)}_w$')

    ax.legend()
    
    #set the axis limits
    ax.set_xlim([-a, a])
    ax.set_ylim([-a, a])
    ax.set_zlim([-a, a])

    #set the axis labels 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Timestep: {timeStep}, ⍺: {np.round(np.degrees(alphas[timeStep]), 2)}, AoA: {np.round(np.degrees(AoA[timeStep]), 2)} \nFl: {np.round(Fl[timeStep], 4)} \nFd: {np.round(Fd[timeStep], 4)} \nFrot: {np.round(Frot[timeStep], 4)} \nFam: {np.round(Fam[timeStep], 4)}')

# run the live animation of the wing 
def generatePlotsForKinematicsSequence():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    anim = animation.FuncAnimation(fig, functools.partial(animationPlot, ax), frames=len(timeline), repeat=True)
    #anim.save('u&d_vectors.gif') 
    plt.show() 

def kinematics(): 
    #create figure 
    generateSequence()
    #generatePlotsForKinematicsSequence(timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, wingtip_index, hinge_index)
############################################################################################################################################################################################
##% dynamics

def getAerodynamicCoefficients(x0, AoA): 
    deg2rad = np.pi/180.0 
    rad2deg = 180.0/np.pi
    
    AoA = rad2deg*AoA
    
    # Cl and Cd definitions from Dickinson 1999
    Cl = x0[0] + x0[1]*np.sin( deg2rad*(2.13*AoA - 7.20) )
    Cd = x0[2] + x0[3]*np.cos( deg2rad*(2.04*AoA - 9.82) )
    Crot = x0[3]
    Cam1 = x0[4]
    Cam2 = x0[5]
    Cam3 = x0[6]
    return Cl, Cd, Crot, Cam1, Cam2, Cam3

############################################################################################################################################################################################
##%% main 

from scipy.integrate import trapz, simpson
import scipy.optimize as opt
import time

#cost function which tells us how far off our QSM values are from the CFD ones
def cost(x, numerical=False, nb=1000, show_plots=False):
    #global variable must be imported in order to modify them locally
    global Fl_magnitude, Fd_magnitude, Frot_magnitude, Fam_magnitude, planar_rots_wing_g, y_wing_g_sequence, Fam

    Cl, Cd, Crot, Cam1, Cam2, Cam3 = getAerodynamicCoefficients(x, np.array(AoA))

    # chord calculation 
    y_space = np.linspace(min_y, max_y, 100)
    c = getChordLength(e_wingPoints, y_space)

    # plt.plot(c,y_space)
    # plt.show()
    # exit()

    rho = 1.225

    #both Cl and Cd are of shape (nt, 1), this however poses a dimensional issue when the magnitude of the lift/drag force is to be multiplied
    #with their corresponding vectors. to fix this, we reshape Cl and Cd to be of shape (nt,)
    Cl = Cl.reshape(nt,) 
    Cd = Cd.reshape(nt,)
    _planar_rots_wing_g = planar_rots_wing_g.reshape(nt,3)
    _planar_rots_wing_w = planar_rots_wing_w.reshape(nt,3)
    _planar_rot_acc_wing_g = planar_rot_acc_wing_g.reshape(nt,3)
    _planar_rot_acc_wing_w = planar_rot_acc_wing_w.reshape(nt,3) 
    _alphas_dt_dt_sequence = alphas_dt_dt_sequence

    if numerical:
        #START OF NUMERICAL VERSION
        #computation following Nakata 2015 eqns. 2.4a-c
        #thickness of each blade 
        dr = y_space[1]-y_space[0]

        Fl_magnitude = np.zeros(nt)
        Fd_magnitude = np.zeros(nt)
        Frot_magnitude = np.zeros(nt)
        Fam_magnitude = np.zeros(nt)

        # the numerical computation follows the same original equations as the analytical one, but instead of performing the integral, we calculate the forces
        # in each blade and then sum them up: dFl = 0.5*rho*Cl*𝛀^2(φ,Θ)*r^2*c*dr, dFd = 0.5*rho*Cd*𝛀^2(φ,Θ)*r^2*c*dr, dFrot = 0.5*rho*Crot*𝛀(φ,Θ)*r*c^2*dr
        # calculation of the magnitude of the lift/drag/rotational force for each blade. each force is then summed up for each timestep and a (nt,) array is returned.
        # each row represents a timestep and the value contained therein the total Fl/Fd/Frot for that timestep
        # this loop does the following: it loops over y_space (100, 1000, 10000 however many points the user sets) and for every point it computes the value 
        # for all timesteps for that point and ONLY then it moves on to the next point, computes all timesteps and so on, until it's done looping over y_space
        for i in range(y_space.shape[0]):
            #as previously discussed (cf. 245 - 248), for the numerical calculations of Fd, Fl, Frot we must use the absolute angular velocity that solely depends on phi and theta {𝛀(φ,Θ)} -> absolute planar angular velocity
            #now, since we are looping over the span (y_space) here, we have to calculate the absolute planar linear velocity for each blade by computing the cross product 
            #of the absolute planar angular velocity and the absolute radius of each blade (y_blade_g). 
            r = y_space[i] - y_space[0]
            y_blade_g = r*y_wing_g_sequence #(nt,3)
            blade_planar_us_wing_g = np.cross(_planar_rots_wing_g, y_blade_g, axis=1)
            blade_planar_us_wing_g_magnitude = np.linalg.norm(blade_planar_us_wing_g, axis=1)
            
            Fl_magnitude += 0.5*rho*Cl*(blade_planar_us_wing_g_magnitude**2)*c[i]*dr
            Fd_magnitude += 0.5*rho*Cd*(blade_planar_us_wing_g_magnitude**2)*c[i]*dr
            Frot_magnitude += rho*Crot*blade_planar_us_wing_g_magnitude*alphas_dt_sequence*(c[i]**2)*dr
            Fam_magnitude += Cam1*rho*np.pi/4*(_planar_rot_acc_wing_w[:, 2])*r*c[i]**2*dr + Cam2*rho*np.pi/4*(alphas_dt_sequence*_planar_rots_wing_w[:, 0])*r*c[i]**2*dr + Cam3*rho*np.pi/16*_alphas_dt_dt_sequence*c[i]**3*dr

            # Fam_magnitude += Cam1*_planar_rot_acc_wing_w[:, 2] + Cam2*alphas_dt_sequence*_planar_rots_wing_w[:, 0] + Cam3*alphas_dt_dt_sequence
            # Fam_magnitude += Cam1*rho*np.pi/4*(_planar_rot_acc_wing_w[:, 2] + alphas_dt_sequence*_planar_rots_wing_w[:, 0])*r*c[i]**2*dr + Cam2*rho*np.pi/16*_alphas_dt_dt_sequence*c[i]**3*dr
            # Fam_magnitude += rho*np.pi/4*(_planar_rot_acc_wing_g[:, 2] + alphas_dt_sequence*_planar_rots_wing_g[:, 0])*r*c[i]**2*dr + rho*np.pi/16*_alphas_dt_dt_sequence*c[i]**3*dr
        #END OF NUMERICAL VERSION   
                    
    else: 
        #START OF ANALYTICAL VERSION
        #computation following Nakata 2015 eqns. 2.4a-c
        c_interpolation = interp1d(y_space, c) #we create a function that interpolates our chord (c) w respect to our span (y_space)

        #the following comes from defining lift/drag in the following way: dFl = 0.5*rho*Cl*v^2*c*dr -> where v = linear velocity, c = chord length, dr = chord width
        #v can be broken into 𝛀(φ,Θ)*r  (cf. lines 245-248). plugging that into our equation we get: dFl = 0.5*rho*Cl*𝛀^2(φ,Θ)*r^2*c*dr (lift in each blade)
        #integrating both sides, and pulling constants out of integrand on RHS: Fl = 0.5*rho*Cl*𝛀^2(φ,Θ)*∫c*r^2*dr 
        #our function def Cr2 then calculates the product of c and r^2 ; I (second moment of area) performs the integration of the product 
        #drag is pretty much the same except that instead of Cl we use Cd: Fd = 0.5*rho*Cd*𝛀^2(φ,Θ)*∫c*r^2*dr
        #and the rotational force is defined as follows: Frot = 0.5*rho*Crot*𝛀(φ,Θ)*∫c^2*r*dr
        def Cr2(r): 
            return c_interpolation(r) * r**2
        def C2r(r):
            return (c_interpolation(r)**2) * r
        
        Ild = simpson(Cr2(y_space), y_space) #second moment of area for lift/drag calculations
        Irot = simpson(C2r(y_space), y_space) #second moment of area for rotational force calculation 

        planar_rots_wing_g_magnitude = np.linalg.norm(planar_rots_wing_g, axis=1)
        planar_rots_wing_g_magnitude = planar_rots_wing_g_magnitude.reshape(nt,) #here we reshape to fix dimensionality issues as planar_rots_wing_g_magnitude is of shape (nt, 1) and it should be of shape (nt,)
        rho = 1.225
        Fl_magnitude = 0.5*rho*Cl*(planar_rots_wing_g_magnitude**2)*Ild
        Fd_magnitude = 0.5*rho*Cd*(planar_rots_wing_g_magnitude**2)*Ild
        Frot_magnitude = rho*Crot*planar_rots_wing_g_magnitude*alphas_dt_sequence*Irot
        #END OF ANALYTICAL VERSION 

    # vector calculation of Fl, Fd, Frot, Fam. arrays of the form (nt, 3) 
    for i in range(nt):
        Fl[i,:] = (Fl_magnitude[i] * e_liftVectors[i])
        Fd[i,:] = (Fd_magnitude[i] * e_dragVectors_wing_g[i])
        Frot[i,:] = (Frot_magnitude[i] * z_wing_g_sequence[i])
        Fam[i, :] = (Fam_magnitude[i] * z_wing_g_sequence[i]*np.sign(alphas[i]))
        # Fam[i,:] = (Fam_magnitude[i] * z_wing_w_sequence[i]*np.sign(alphas[i]))
        # Fam[i, :] = np.matmul(bodyRotationMatrixTrans_sequence[i], np.matmul(strokeRotationMatrixTrans_sequence[i], np.matmul(wingRotationMatrixTrans_sequence[i], Fam[i])))
        # e_Fam[i, :] = (Fam[i]/Fam_magnitude[i])
    
    # writeArraytoFile(Fam_magnitude, 'debug/F_am_w_mag.txt')
    # writeArraytoFile(Fam_magnitude1, 'debug/Fam_mag1.txt')
    # writeArraytoFile(Fam_magnitude2, 'debug/Fam_mag2.txt')
    # writeArraytoFile(alphas_dt_dt_sequence, 'debug/alphas_dt_dt2.txt')
    # writeArraytoFile(phis, 'debug/phis.txt')
    # writeArraytoFile(thetas, 'debug/thetas.txt')
    # writeArraytoFile(planar_rot_acc_wing_w, 'debug/planar_rot_acc_wing_w(2nd_order_2).txt')
    # writeArraytoFile(planar_rot_acc_wing_s, 'debug/planar_rot_acc_wing_s(1st_order).txt')
    # writeArraytoFile(phis_dt_sequence, 'debug/phis_dt_forward.txt')
    # writeArraytoFile(thetas_dt_sequence, 'debug/thetas_dt_forward.txt')
    # print(thetas_dt_dt_sequence)
    # exit()
    
    Fx_QSM = Fl[:, 0] + Fd[:, 0] + Frot[:, 0] + Fam[:, 0]
    Fy_QSM = Fl[:, 1] + Fd[:, 1] + Frot[:, 1] + Fam[:, 1]
    Fz_QSM = Fl[:, 2] + Fd[:, 2] + Frot[:, 2] + Fam[:, 2]

    K_num = np.linalg.norm(Fx_QSM-Fx_CFD_interp(timeline)) + np.linalg.norm(Fz_QSM-Fz_CFD_interp(timeline)) + np.linalg.norm(Fy_QSM+Fy_CFD_interp(timeline))
    K_den = np.linalg.norm(Fx_CFD_interp(timeline)) + np.linalg.norm(Fz_CFD_interp(timeline)) + np.linalg.norm(-Fy_CFD_interp(timeline))
     
    if K_den != 0: 
        K = K_num/K_den
    else:
        K = K_num

    F_QSM = np.vstack((Fx_QSM, Fy_QSM, Fz_QSM)).transpose()
    M_CFD = np.vstack((Mx_CFD_interp(timeline), My_CFD_interp(timeline), Mz_CFD_interp(timeline))).transpose()

    F_QSM = np.array(F_QSM)
    M_CFD = np.array(M_CFD)

    r = getLever(F_QSM, M_CFD)

    if show_plots:
        # plt.plot(timeline, np.degrees(phis), label='ɸ')
        # plt.plot(timeline, np.degrees(alphas), label ='⍺')
        # plt.plot(timeline, np.degrees(thetas), label='Θ')
        # plt.legend()
        # plt.show()

        #AoA
        plt.plot(timeline, np.degrees(AoA), label='AoA', color = '#F1680F')
        plt.xlabel('t/T')
        plt.legend()
        plt.show()

        #coefficients
        graphAoA = np.linspace(-9, 90, 100)*(np.pi/180)
        gCl, gCd, gCrot, gCam1, gCam2, gCam3 = getAerodynamicCoefficients(x, graphAoA)
        fig, ax = plt.subplots()
        ax.plot(np.degrees(graphAoA), gCl, label='Cl', color='#0F95F1')
        ax.plot(np.degrees(graphAoA), gCd, label='Cd', color='#F1AC0F')
        # ax.plot(np.degrees(graphAoA), gCrot*np.ones_like(gCl), label='Crot')
        ax.set_xlabel('AoA[°]')
        plt.legend()
        plt.show()

        #forces
        plt.plot(timeline[:], Fx_QSM, label='Fx_QSM', color='red')
        plt.plot(timeline[:], Fy_QSM, label='Fy_QSM', color='green')
        plt.plot(timeline[:], Fz_QSM, label='Fz_QSM', color='blue')
        plt.plot(timeline[:], Fx_CFD_interp(timeline), label='Fx_CFD', linestyle = 'dashed', color='red')
        plt.plot(timeline[:], Fy_CFD_interp(timeline), label='Fy_CFD', linestyle = 'dashed', color='green')
        plt.plot(timeline[:], Fz_CFD_interp(timeline), label='Fz_CFD', linestyle = 'dashed', color='blue')
        plt.xlabel('t/T [s]')
        plt.ylabel('Force [mN]')
        plt.title(f'Fx_QSM/Fx_CFD = {np.round(np.linalg.norm(Fx_QSM)/np.linalg.norm(Fx_CFD_interp(timeline)), 3)}; Fz_QSM/Fz_CFD = {np.round(np.linalg.norm(Fz_QSM)/np.linalg.norm(Fz_CFD_interp(timeline)), 3)}')
        plt.legend()
        # plt.savefig('qsm', dpi=2000)
        plt.show()

        #vertical forces
        # plt.plot(timeline, Fl[:, 2], label = 'Vertical lift force', color='gold')
        plt.plot(timeline, Frot[:, 2], label = 'Vertical rotational force', color='orange')
        # plt.plot(timeline, Fd[:, 2], label = 'Vertical drag force', color='lightgreen')
        plt.plot(timeline, Fam[:, 2], label = 'Vertical added mass force', color='red')
        plt.plot(timeline, Fz_QSM, label = 'Vertical QSM force', color='blue')
        plt.plot(timeline, Fz_CFD_interp(timeline), label = 'Vertical CFD force', color='purple')
        plt.xlabel('t/T [s]')
        plt.ylabel('Force [mN]')
        plt.legend()
        # plt.savefig('debug/vertical_forces_no_Fam', dpi=2000)
        plt.show()

        #lever
        plt.plot(timeline, r[:, 0], color='#C00891', label='Lever x-component')
        plt.plot(timeline, r[:, 1], color='#0F2AEE', label='Lever y-component')
        plt.plot(timeline, r[:, 2], color='#0FEE8C', label='Lever z-component')
        # plt.plot(timeline, np.linalg.norm(r, axis=1), color='#08C046', label='Lever magnitude')
        plt.xlabel('t/T [s]')
        plt.ylabel('Lever [mm]')
        plt.legend()
        # plt.savefig('debug/r', dpi=2000)
        plt.show()


        #moments
        plt.plot(timeline[:], Mx_CFD_interp(timeline), label='Mx_CFD', linestyle = 'dashed', color='red')
        plt.plot(timeline[:], My_CFD_interp(timeline), label='My_CFD', linestyle = 'dashed', color='green')
        plt.plot(timeline[:], Mz_CFD_interp(timeline), label='Mz_CFD', linestyle = 'dashed', color='blue')
        plt.xlabel('t/T [s]')
        plt.ylabel('Moment [mN*m]')
        plt.legend()
        # plt.savefig('debug/moments', dpi=2000)
        plt.show()

        generatePlotsForKinematicsSequence()
    return K

#optimization by means of opt.differential_evolution which calculates the global minimum of our cost function (def F) and tells us 
#for what x_0 values/input this minimum is attained  

#optimizing using scipy.optimize.minimize which is faster
def main():
    kinematics()
    x_0 = [0.225, 1.58,  1.92, -1.55, 1, 1, 1] #initial definition of x0 following Dickinson 1999
    bounds = [(-3, 3), (-3, 3), (-3, 3), (-3, 3), (-3, 3), (-3, 3), (-3, 3)]
    optimize = True
    nb = 1000 #nb: number of blades 
    numerical = True 
    if optimize:
        start = time.time()
        optimization = opt.minimize(cost, args=(numerical, nb), bounds=bounds, x0=x_0)
        x0_final = optimization.x
        K_final = optimization.fun
        if numerical:
            print('Computing using the numerical approach')
        else: 
            print('Computing using the analytical approach')
        print('Computing for: ' + str(nb) + ' blades')
        print('Completed in:', round(time.time() - start, 3), 'seconds')
    else:
        x0_final = [0.225, 1.58,  1.92, -1.55]
        K_final = ''
        if numerical:
            print('Computing using the numerical approach')
        else: 
            print('Computing using the analytical approach')
        print('Computing for: ' + str(nb) + ' blades')
        cost(x0_final, numerical, nb, show_plots=True)
    print('x0_final:', np.round(x0_final, 5), '\nK_final:', K_final)
    cost(x0_final, show_plots=True)

# #optimizing using scipy.optimize.differential_evolution which is considerably slower than scipy.optimize.minimize
# #the results also fluctuate quite a bit using this optimizer.
# def main():
#     kinematics()
#     x_0 = [0.225, 1.58,  1.92, -1.55] #initial definition of x0 following Dickinson 1999
#     bounds = [(-3, 3), (-3, 3), (-3, 3), (-3, 3)]
#     nb = 100 #nb: number of blades 
#     numerical = False 
#     optimize = True
#     if optimize:
#         start = time.time()
#         optimization = opt.differential_evolution(cost, args=(numerical, nb), bounds=bounds, x0=x_0, maxiter=100)
#         x0_final = optimization.x
#         K_final = optimization.fun
#         if numerical:
#             print('Computing using the numerical approach')
#         else: 
#             print('Computing using the analytical approach')
#         print('Computing for: ' + str(nb) + ' blades')
#         print('Completed in:', round(time.time() - start, 3), 'seconds')
#     else:
#         x0_final = [1.76254482, -1.06909505,  1.12313521, -0.72540114]
#         K_final = 0.5108267902800643
#     print('x0_final:', np.round(x0_final, 5), '\nK_final:', K_final)
#     cost(x0_final, show_plots=False)

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