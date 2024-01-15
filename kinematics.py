#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: nico
"""
import os
os.environ['QTA_QP_PLATFORM'] = 'wayland'
import numpy as np 
import csv 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib import animation
import functools 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import insect_tools
import debug as db 


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
                                            insect_tools.Ry(alpha),
                                            insect_tools.Rz(theta)
                                    ),
                                    insect_tools.Rx(phi)
                                )
    rotationMatrixTrans = np.transpose(rotationMatrix)  
    strokePoints = np.zeros((points.shape[0], 3))
    for point in range(points.shape[0]): 
        x_s = np.matmul(rotationMatrixTrans, points[point])
        strokePoints[point, :] = x_s
   #print('\n\nrotation matrix wing to stroke', rotationMatrix, '\n\n tranpose of rotation matrix wing to stroke', rotationMatrixTrans)
    return strokePoints, [rotationMatrix, rotationMatrixTrans]


def convert_from_stroke_plane_to_body_reference_frame(points, parameters, invert=False):
    #points must be in stroke plane x(s) y(s) z(s)
    #Rx(pi/0) * R (eta)
    eta = parameters[3] #rad
    flip_angle = 0 
    if invert:
         flip_angle = np.pi
    rotationMatrix = np.matmul(
                                insect_tools.Rx(flip_angle),
                                insect_tools.Ry(eta)
                                )
    rotationMatrixTrans = np.transpose(rotationMatrix) 
    bodyPoints = np.zeros((points.shape[0], 3))
    for point in range(points.shape[0]): 
        x_b = np.matmul(rotationMatrixTrans, points[point])
        bodyPoints[point,:] = x_b
    return bodyPoints, [rotationMatrix, rotationMatrixTrans]

def convert_from_body_reference_frame_to_global_reference_frame(points, parameters):
    #points passed into this fxn must be in the body reference frame x(b) y(b) z(b)
    #phi, alpha, theta
    psi = parameters [0] #rad
    beta = parameters [1] #rad
    gamma = parameters [2] #rad
    rotationMatrix = np.matmul(
                                    np.matmul(
                                            insect_tools.Rx(psi),
                                            insect_tools.Ry(beta)
                                    ),
                                    insect_tools.Rz(gamma)
                                )
    rotationMatrixTrans = np.transpose(rotationMatrix)
    globalPoints = np.zeros((points.shape[0], 3))
    for point in range(points.shape[0]): 
        x_g = np.matmul(rotationMatrixTrans, points[point])
        globalPoints[point, :] = x_g 
    return globalPoints, [rotationMatrix, rotationMatrixTrans]

def plot(points, title=''):
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    #no axis so create one
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #use class axis to plot

   # ax.plot_trisurf(X, Y, Z, color ='#969f65', )
    ax.scatter(X, Y, Z)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    #plt.show

def create_wing_angles(timeline, frequency):
    phis = 40*np.cos(timeline)

    y1 = 45
    y2 = -30

    # add phase shift
    firstcycle = len(timeline)//frequency
    phase_shift_angle = (np.pi/2+np.pi/10)
    phase_shift_angle_norm =(phase_shift_angle/(2*np.pi))
    phaseid = int(phase_shift_angle_norm * firstcycle)

    #initial feathering angle 
    alphas = np.zeros_like(timeline)+y1

    #when timeline is greater than 5*pi/6 use the ramp fxn 
    margin = np.pi/8
    m = ((y2-y1) / (margin))
    x = timeline[timeline>=(np.pi-margin)]
    x1 =np.pi-margin
    alphas[timeline>=(np.pi-margin)] = y1 + m*(x-x1)

    #when timeline greater than pi use y2
    alphas[timeline>=np.pi] = y2


    #when timeline is greater than 2*pi-pi/6 use the ramp fxn 
    m = ((y1 - y2)/(margin))
    x = timeline[timeline>=(2*np.pi-margin)]
    x1 =2*np.pi - margin
    alphas[timeline>=(2*np.pi-margin)] = y2 + m*(x-x1)

    # apply phase shift 
    alphas_after = alphas[phaseid:]
    alphas_before = alphas[:phaseid]
    alphas = np.hstack([alphas_after, alphas_before])

    thetas = 3*np.sin(3*timeline+np.pi/180)+4*np.sin(2*timeline-2*np.pi/180)-10
    return np.radians(phis), np.radians(alphas), np.radians(thetas)

def create_wing_angles_dt(phis, alphas, thetas, timeline):
    dt = timeline[1]-timeline[0]

    phis_dt = np.zeros((timeline.shape[0], 3)) #rad/s
    alphas_dt = np.zeros((timeline.shape[0], 3)) #rad/s
    thetas_dt = np.zeros((timeline.shape[0], 3)) #rad/s

    for timeStep in range(timeline.shape[0]): 
        d_phi = phis[timeStep] - phis[timeStep-1]
        d_alpha = alphas[timeStep] - alphas[timeStep-1]
        d_theta = thetas[timeStep] - thetas[timeStep-1]

        phi_dt = d_phi/dt #rad/s
        alpha_dt = d_alpha/dt #rad/s
        theta_dt = d_theta/dt #rad/s
        phis_dt[timeStep, :] = phi_dt
        alphas_dt[timeStep, :] = alpha_dt
        thetas_dt[timeStep, :] = theta_dt 
    return phis_dt, alphas_dt, thetas_dt

def generate_rot_wing(wingRotationMatrix, bodyRotationMatrixTrans, strokeRotationMatrixTrans, phi, phi_dt, alpha, alpha_dt, theta, theta_dt): 
    phiMatrixTrans = np.transpose(insect_tools.Rx(phi)) #np.transpose(getRotationMatrix('x', phi))
    alphaMatrixTrans = np.transpose(insect_tools.Ry(alpha)) #np.transpose(getRotationMatrix('y', alpha))
    thetaMatrixTrans = np.transpose(insect_tools.Rz(theta)) #np.transpose(getRotationMatrix('z', theta))
    vector_phi_dt = np.array([[phi_dt], [0], [0]])
    vector_alpha_dt = np.array([[0], [alpha_dt], [0]])
    vector_theta_dt = np.array([[0], [0], [theta_dt]])
    rot_wing_s = np.matmul(phiMatrixTrans, (vector_phi_dt+np.matmul(thetaMatrixTrans, (vector_theta_dt+np.matmul(alphaMatrixTrans, vector_alpha_dt)))))
    rot_wing_b = np.matmul(strokeRotationMatrixTrans, rot_wing_s)
    rot_wing_w = np.matmul(wingRotationMatrix, rot_wing_s)
    rot_wing_g = np.matmul(bodyRotationMatrixTrans, rot_wing_b)
    return rot_wing_g, rot_wing_b, rot_wing_w

def generate_u_wing_g_position(rot_wing_g, y_wing_g):
    # #omega x point
    # #keep in mind that omega is a COLUMN vector and wing_point is a ROW vector. omega has to be converted to a ROW vector in order to calculate the x product
    rot_wing_g = rot_wing_g.flatten() #either we flatten rot_wing_g or we reshape y_wing_g.reshape(3, 1)
    u_wing_g_position = np.cross(rot_wing_g, y_wing_g)
    return u_wing_g_position

def generate_u_wing_w(u_wing_g, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix):
    rotationMatrix = np.matmul(np.matmul(bodyRotationMatrix, strokeRotationMatrix), wingRotationMatrix)
    u_wing_w = np.matmul(rotationMatrix, u_wing_g)
    return u_wing_w

def getWindDirectioninWingReferenceFrame(u_wind_g, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix): 
    u_wind_b = np.matmul(bodyRotationMatrix, u_wind_g.reshape(3,1))
    u_wind_s = np.matmul(strokeRotationMatrix, u_wind_b.reshape(3,1))
    u_wind_w = np.matmul(wingRotationMatrix, u_wind_s.reshape(3,1))
    return u_wind_w

def getAoA(x_wing_g, e_u_wing_g):
    AoA = np.arccos(np.dot(x_wing_g, e_u_wing_g))
    return AoA

# def getAoA(drag_vector, x_wing_g):
#     #should be in the wing reference frame
#     AoA = np.arctan2(np.linalg.norm(np.cross(x_wing_g, drag_vector)), np.dot(-drag_vector, x_wing_g)) #rad
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

    if useCFDData:
        timeline, alphas, phis, thetas, alphas_dt, phis_dt, thetas_dt = load_kinematics_data()
        # #plots to check angles 
        # plt.plot(timeline, np.degrees(phis), label='ɸ')
        # plt.plot(timeline, np.degrees(alphas), label ='⍺')
        # plt.plot(timeline, np.degrees(thetas), label='Θ')
        # plt.legend()
        # plt.show()
    else: 
        #timeline
        timeline = np.mod((frequency*np.linspace(start_time, 2*np.pi, number_of_timesteps+1)[:-1]), 2*np.pi)
        #print('\nthis is the length of our timeline:', len(timeline))
        #angles
        phis, alphas, thetas = create_wing_angles(timeline, frequency=1)
        # # plots to check angles 
        # plt.plot(timeline, np.degrees(phis), timeline, np.degrees(alphas), timeline, np.degrees(thetas))
        # plt.show()

        #angles_dt
        phis_dt, alphas_dt, thetas_dt = create_wing_angles_dt(phis, alphas, thetas, timeline)

    print('alpha:', alphas.shape, '\n', 'timeline:', timeline.shape)
    print('alpha:', type(alphas), '\n', 'timeline:', type(timeline))
    db.writeArraytoFile(alphas, 'alphas.txt')
    db.writeArraytoFile(timeline, 'timeline.txt')
    alphas_interp = interp1d(timeline, alphas, fill_value='extrapolate')
    phis_interp = interp1d(timeline, phis, fill_value='extrapolate')
    thetas_interp = interp1d(timeline, thetas, fill_value='extrapolate')

    timeline = np.linspace(0, 1, 101)

    strokePointsSequence = np.zeros((timeline.shape[0], 3))
    bodyPointsSequence = np.zeros((timeline.shape[0], 3))
    globalPointsSequence = np.zeros((timeline.shape[0], 3))

    rots_wing_b = np.zeros((timeline.shape[0], 3))
    rots_wing_w = np.zeros((timeline.shape[0], 3))
    rots_wing_g = np.zeros((timeline.shape[0], 3))

    us_wing_w = np.zeros((timeline.shape[0], 3))
    us_wing_g = np.zeros((timeline.shape[0], 3))

    us_wind_w = np.zeros((timeline.shape[0], 3))
    AoA = np.zeros((timeline.shape[0], 3))
    u_wing_g_vectors = np.zeros((timeline.shape[0], 3)) 
    dragVectors_wing_g = np.zeros((timeline.shape[0], 3))
    liftVectors = np.zeros((timeline.shape[0], 3))
    e_liftVectors = np.zeros((timeline.shape[0], 3))

    delta_t = timeline[1] - timeline[0]
    #print(delta_t)

    for timeStep in range(timeline.shape[0]):
        t = timeline[timeStep]
        #parameter array: psi [0], beta[1], gamma[2], eta[3], phi[4], alpha[5], theta[6]
        alphas_dt = (alphas_interp(t+delta_t) - alphas_interp(t)) / delta_t
        phis_dt = (phis_interp(t+delta_t) - phis_interp(t)) / delta_t
        thetas_dt = (thetas_interp(t+delta_t) - thetas_interp(t)) / delta_t

        parameters = [0, 0, 0, -80*np.pi/180, phis_interp(t), alphas_interp(t), thetas_interp(t)] # 7 angles in radians! #without alphas[timeStep] any rotation around any y axis through an angle of pi/2 gives an error! 
        parameters_dt = [0, 0, 0, 0, phis_dt, alphas_dt, thetas_dt]
        strokePoints, [wingRotationMatrix, wingRotationMatrixTrans] = convert_from_wing_reference_frame_to_stroke_plane(wingPoints, parameters)
        bodyPoints, [strokeRotationMatrix, strokeRotationMatrixTrans] = convert_from_stroke_plane_to_body_reference_frame(strokePoints, parameters)
        globalPoints, [bodyRotationMatrix, bodyRotationMatrixTrans] = convert_from_body_reference_frame_to_global_reference_frame(bodyPoints, parameters)
        #print('\n\nthese are the global points before\n\n', globalPoints)
        # print('\n\nthis is the stroke rotation matrix: ', strokeRotationMatrix)
        # print('\n\nthis is the transpose of stroke rotation matrix : ', strokeRotationMatrixTrans)
        strokePointsSequence[timeStep, :] = strokePoints
        bodyPointsSequence[timeStep, :] = bodyPoints
        globalPointsSequence[timeStep, :] = globalPoints

        x_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[1], [0], [0]]))
        y_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[0], [1], [0]]))
        #print('x_wing_g:', x_wing_g, '\n', 'y_wing_g:', y_wing_g)

        rot_wing_g, rot_wing_b, rot_wing_w = generate_rot_wing(wingRotationMatrix, bodyRotationMatrixTrans, strokeRotationMatrixTrans, parameters[4], parameters_dt[4], parameters[5], 
                                    parameters_dt[5], parameters[6], parameters_dt[6])
        
        rots_wing_b[timeStep, :] = rot_wing_b
        rots_wing_w[timeStep, :] = rot_wing_w
        rots_wing_g[timeStep, :] = rot_wing_g

        u_wing_g = generate_u_wing_g_position(rot_wing_g, y_wing_g.flatten())

        vWtotG = u_wing_g + u_wind_g
        
        # print('\n\n\nomega w w', rot_wing_w)
        # print('\n\n\n points', wingPoints)
        u_wing_w = generate_u_wing_w(u_wing_g, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix)
        us_wing_g[timeStep, :] = u_wing_g
        us_wing_w[timeStep, :] = u_wing_w

        u_wind_w = getWindDirectioninWingReferenceFrame(vWtotG, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix).flatten() - np.array(u_wing_w)
        us_wind_w[timeStep, :] = u_wind_w
        
        #print('AoA:', np.degrees(aoa[wingtip_index]))
        #print('alpha:', np.degrees(parameters[5]))
        #print('theta:', np.degrees(parameters[6]))

        #velocity vector here 
        u_wing_g_vector = u_wing_g.flatten()
        # u_wing_g_vector = np.sum(us_wing_g[timeStep], axis=0)
        u_wing_g_vectors [timeStep, :] = u_wing_g_vector
        u_wing_g_magnitude = np.sqrt(u_wing_g_vector[0]**2 + u_wing_g_vector[1]**2 + u_wing_g_vector[2]**2)
        if u_wing_g_magnitude != 0:  
            e_u_wing_g = u_wing_g_vector/u_wing_g_magnitude
        else:
            e_u_wing_g = u_wing_g_vector 
        dragVector_wing_g = -e_u_wing_g
        dragVectors_wing_g[timeStep, :] = dragVector_wing_g
    
        #lift 
        #R = globalPointsSequence[timeStep][wingtip_index] - globalPointsSequence[timeStep][pivot_index]
        #ax.quiver(X[pivot_index], Y[pivot_index], Z[pivot_index], R[0], R[1], R[2], color='red')
        liftVector = np.cross(e_u_wing_g, y_wing_g.flatten())
        # print('lift vector before', liftVector)
        # print('alphas', alphas[timeStep], timeStep)
        liftVector = liftVector*np.sign(alphas_interp(t))
        liftVectors[timeStep, :] = liftVector
        # if liftVector[2] < 0:
        #     liftVector[2] = -liftVector[2] 

        # if (60 <= np.degrees(alphas[timeStep]) <= 62): 
        #     pass
        # elif (-38 <= np.degrees((alphas[timeStep])) <= -36):
        #     pass
        # else:
        #     liftVector *= 0 
        #     dragVector_wing_g *= 0 
       #aoa = getAoA(dragVector_wing_g, x_wing_g.flatten())
        aoa = getAoA(x_wing_g.flatten(), e_u_wing_g)
        AoA.append(aoa)
        liftVector_magnitude = np.sqrt(liftVector[0]**2 + liftVector[1]**2 + liftVector[2]**2)
        if liftVector_magnitude != 0: 
            e_liftVector = liftVector / liftVector_magnitude
        else:
            e_liftVector = liftVector
        e_liftVectors[timeStep, :] = e_liftVector
        
        

    #validation of our u_wing_g: 
    #left and right derivative: 
    delta_t = timeline[1] - timeline[0]
    verifying_us_wing_g = np.zeros((timeline.shape[0], 3))
    for timeStep in range(timeline.shape[0]):
        currentGlobalPoint = globalPointsSequence[timeStep]
        leftGlobalPoint = globalPointsSequence[timeStep-1]
        rightGlobalPoint = globalPointsSequence[(timeStep+1)%len(timeline)]
        LHD = (currentGlobalPoint - leftGlobalPoint) / delta_t
        RHD = (rightGlobalPoint - currentGlobalPoint) / delta_t
        verifying_us_wing_g[timeStep, :] = (LHD+RHD)/2
    verifying_us_wing_g = verifying_us_wing_g


        
    # # plot to check rot_wing_w's components 
    # rots_wing_w = np.array(rots_wing_w)
    # plt.plot(timeline[:-1], rots_wing_w[:-1, 0], label='rot_wing_w x')
    # plt.plot(timeline[:-1], rots_wing_w[:-1, 1], label='rot_wing_w y')
    # plt.plot(timeline[:-1], rots_wing_w[:-1, 2], label='rot_wing_w z')
    # plt.legend()
    # plt.show()
    # rots_wing_b = np.array(rots_wing_b)
    # plt.plot(timeline[:-1], rots_wing_b[:-1, 0], label='rot_wing_b x')
    # plt.plot(timeline[:-1], rots_wing_b[:-1, 1], label='rot_wing_b y')
    # plt.plot(timeline[:-1], rots_wing_b[:-1, 2], label='rot_wing_b z')
    # plt.legend()
    # plt.show()
    
    
    return timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, u_wing_g_vectors, dragVectors_wing_g, e_liftVectors
    #for each timestep in timeline : extract orientation of the points by performing appropriate rotations  

def animationPlot(ax, alphas, pointsSequence, us_wing_g, AoA, wingtip_index, pivot_index, Fl, Fd, u_wing_g_vectors, dragVectors_wing_g, e_liftVectors, timeStep): 
    #get point set by timeStep number
    points = pointsSequence[timeStep] #pointsSequence can either be global, body, stroke 
    #clear the current axis 
    ax.cla()
    # print('these are the global points after: ', points)
    #extract the x, y and z coordinates 
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
    
    # # print('x', X)
    # # print('y', Y)
    # # print('z', Z)
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
    u_wing_g_vector = u_wing_g_vectors[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], u_wing_g_vector[0], u_wing_g_vector[1], u_wing_g_vector[2], color='orange', label=r'$\overrightarrow{u}^{(g)}_w$' )
    # print('u_wing_g vector', u_wing_g_vector)
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
    ax.set_title(f'Timestep: {timeStep} \n⍺: {np.round(np.degrees(alphas[timeStep]), 2)} \nAoA: {np.round(np.degrees(AoA[timeStep]), 2)} \nFl: {np.round(Fl[timeStep], 4)} \nFd: {np.round(Fd[timeStep], 4)}')
    
def generatePlotsForKinematicsSequence(timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, u_wing_g_vectors, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index, Fl, Fd): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    anim = animation.FuncAnimation(fig, functools.partial(animationPlot, ax, alphas, globalPointsSequence, us_wing_g, AoA,wingtip_index, pivot_index, Fl, Fd, u_wing_g_vectors, dragVectors_wing_g, e_liftVectors), frames=len(timeline), repeat=True)
    #anim.save('u&d_vectors.gif') 
    plt.show() 

def kinematics(): 
    ##########################################################################################################################

    #file import: get shape 
    wing_file = 'drosophilaMelanogasterWing.csv'
    pivot_index = -8
    wingPoints = parse_wing_file(wing_file, 0.001, pivot_index)
    wingtip_index = 17

    # #print wing to check positioning!
    # plt.scatter(wingPoints[:, 0], wingPoints[:, 1])
    # i = 0
    # for wingpoint in wingPoints:
    #     plt.text(wingpoint[0], wingpoint[1], str(i))
    #     i += 1
    # plt.xlim([-4,4])
    # plt.ylim([-4,4])
    # plt.show()

    # #run some checks! 
    # parameters = [0, 0, 0, np.pi/2, 0, 0, 0]
    # print('wing points', wingPoints)
    # plot(wingPoints, 'wing')
    # plt.show()

    # strokePoints, [strokeRotationMatrix, strokeRotationMatrixTrans] = convert_from_wing_reference_frame_to_stroke_plane(wingPoints, parameters)
    # print('stroke points', strokePoints)
    # plot(strokePoints, 'stroke')
    # plt.show()

    # bodyPoints, [bodyRotationMatrix, bodyRotationMatrixTrans] = convert_from_stroke_plane_to_body_reference_frame(strokePoints, parameters)
    # print('body points', bodyPoints)
    # plot(bodyPoints, 'body')
    # plt.show()

    # globalPoints, [globalRotationMatrix, globalRotationMatrixTrans]= convert_from_body_reference_frame_to_global_reference_frame(bodyPoints, parameters)
    # print('global points', globalPoints)
    # plot(globalPoints, 'global')
    # plt.show()

    #creation figure 
    timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, u_wing_g_vectors, dragVectors_wing_g, e_liftVectors = generateSequence(wingPoints, wingtip_index, pivot_index, frequency=10, number_of_timesteps=360, useCFDData=True)
    #generatePlotsForKinematicsSequence(timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, wingtip_index, pivot_index)
    return timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, u_wing_g_vectors, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index

kinematics()