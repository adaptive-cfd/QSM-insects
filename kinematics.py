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


"""
all the parameters we need to describe a wing in space
global -> body -> stroke -> wing 
"""
#helper function

##########################################################################################################################

def parse_wing_file(wing_file, scale, pivot_index) -> np.ndarray:
    #open the file in read mode     
    file_to_read = open(wing_file, 'r')
    #cswwriter activation 
    csvreader = csv.reader(file_to_read)
    wing_points = []
    for row in csvreader:
        #each row is a list/array
        current_row = [-float(row[1]), float(row[0]), 0.0]
        wing_points.append(current_row)
    #connect the last point to first one 
    wing_points.append(wing_points[0].copy())
    #converts to a numpy array for vector operation/computation
    wing_points = np.array(wing_points)

    #shift wing points at the first point 
    pivot = wing_points[pivot_index]
    wing_points -= pivot
    file_to_read.close() 
    return wing_points * scale 

def getChordLength(wing_points, y_coordinate):
    #get the division in wing segments (front and back)
    split_index = 16
    righthand_section = wing_points[:split_index]
    lefthand_section = wing_points[split_index:]

    #interpolate righthand section 
    righthand_section_interpolation = interp1d(righthand_section[:, 1], righthand_section[:, 0], fill_value='extrapolate')
  
    #interpolate lefthand section
    lefthand_section_interpolation = interp1d(lefthand_section[:, 1], lefthand_section[:, 0], fill_value='extrapolate') 
    
    #generate the chord as a function of y coordinate
    chord_length = abs(righthand_section_interpolation(y_coordinate) - lefthand_section_interpolation(y_coordinate))
    return chord_length

def getRotationMatrix(axis, angle):
        """
        angle must be in radians
        """
        if axis == 'x':
            matrix=[
                [1, 0, 0],
                [0, np.cos(angle), np.sin(angle)],
                [0, -np.sin(angle), np.cos(angle)]            
            ]  
        elif axis == 'y':
            matrix=[
                [np.cos(angle), 0, -np.sin(angle)],
                [0, 1, 0],
                [np.sin(angle), 0, np.cos(angle)]            
            ]  
        elif axis == 'z':
            matrix=[
                [np.cos(angle), np.sin(angle), 0],
                [-np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]            
            ]
        return np.array(matrix) #use np.matrix see what gives?? - nope. np.matrix only returns a 2D array (2d matrix). our matrices are 3+ D

def convert_from_wing_reference_frame_to_stroke_plane(points, parameters, invert=False):
    #points passed into this fxn must be in the wing reference frame x(w) y(w) z(w)
    #phi, alpha, theta
    phi = parameters [4]
    alpha = parameters [5]
    theta = parameters [6]
    if invert: 
         phi = -phi 
         alpha = -alpha 
    rotationMatrix = np.matmul(
                                    np.matmul(
                                            getRotationMatrix('y', alpha),
                                            getRotationMatrix('z', theta)
                                    ),
                                    getRotationMatrix('x', phi)
                                )
    rotationMatrix_T = np.transpose(rotationMatrix)  #replace inv with transpose 
    stroke_points = []
    for point in range(len(points)): 
        x_s = np.matmul(rotationMatrix_T, points[point])
        stroke_points.append(x_s)
   #print('\n\nrotation matrix wing to stroke', rotationMatrix, '\n\n tranpose of rotation matrix wing to stroke', rotationMatrix_T)
    return np.array(stroke_points), [np.array(rotationMatrix), np.array(rotationMatrix_T)]


def convert_from_stroke_plane_to_body_reference_frame(points, parameters, invert=False):
    #points must be in stroke plane x(s) y(s) z(s)
    #Rx(pi/0) * R (eta)
    eta = parameters[3]
    flip_angle = 0 
    if invert:
         flip_angle = np.pi
    rotationMatrix = np.matmul(
                                getRotationMatrix('x', flip_angle),
                                getRotationMatrix('y', eta)
                                )
    rotationMatrix_T = np.transpose(rotationMatrix) 
    body_points = [] 
    for point in range(len(points)): 
        x_b = np.matmul(rotationMatrix_T, points[point])
        body_points.append(x_b)
    return np.array(body_points), [rotationMatrix, rotationMatrix_T]

def convert_from_body_reference_frame_to_global_reference_frame(points, parameters):
    #points passed into this fxn must be in the body reference frame x(b) y(b) z(b)
    #phi, alpha, theta
    psi = parameters [0]
    beta = parameters [1]
    gamma = parameters [2]
    rotationMatrix = np.matmul(
                                    np.matmul(
                                            getRotationMatrix('x', psi),
                                            getRotationMatrix('y', beta)
                                    ),
                                    getRotationMatrix('z', gamma)
                                )
    rotationMatrix_T = np.transpose(rotationMatrix)
    global_points = [] 
    for point in range(len(points)): 
        x_g = np.matmul(rotationMatrix_T, points[point])
        global_points.append(x_g)
    return np.array(global_points), [rotationMatrix, rotationMatrix_T]

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

def create_wing_angles_dot(phis, alphas, thetas, timeline):
    dt = timeline[1]-timeline[0]

    phis_dot = []
    alphas_dot = []
    thetas_dot = [] 

    for timeStep in range(len(timeline)): 
        d_phi = phis[timeStep] - phis[timeStep-1]
        d_alpha = alphas[timeStep] - alphas[timeStep-1]
        d_theta = thetas[timeStep] - thetas[timeStep-1]

        dphi_dt = d_phi/dt
        dalpha_dt = d_alpha/dt
        dtheta_dt = d_theta/dt
        phis_dot.append(dphi_dt)
        alphas_dot.append(dalpha_dt)
        thetas_dot.append(dtheta_dt)
    return np.array(phis_dot), np.array(alphas_dot), np.array(thetas_dot)

def generate_omegaW(wing_rotationMatrix, stroke_rotationMatrix, stroke_rotationMatrix_T, phi, phi_dot, alpha, alpha_dot, theta, theta_dot): 
    phiRot_T = np.transpose(getRotationMatrix('x', phi))
    alphaRot_T = np.transpose(getRotationMatrix('y', alpha))
    thetaRot_T = np.transpose(getRotationMatrix('z', theta))
    vector_phi_dot = np.array([[phi_dot], [0], [0]])
    vector_alpha_dot = np.array([[0], [alpha_dot], [0]])
    vector_theta_dot = np.array([[0], [0], [theta_dot]])
    omegaW_s = np.matmul(phiRot_T, (vector_phi_dot+np.matmul(thetaRot_T, (vector_theta_dot+np.matmul(alphaRot_T, vector_alpha_dot)))))
    omegaW_b = np.matmul(stroke_rotationMatrix_T, omegaW_s)
    omegaW_w = np.matmul(wing_rotationMatrix, omegaW_s)
    return omegaW_b, omegaW_w

def generate_uW_w_position(omegaW_b, yWing):
    # #omega x point
    # #keep in mind that omega is a COLUMN vector and wing_point is a ROW vector. omega has to be converted to a ROW vector in order to calculate the x product
    # uW_w_position = []
    # #since uW_w depends on both position and time then we need to account for both. here we account for the position dependance and in generateSequence we take
    # #the time dependance into account 
    # for wing_point in wing_points:    
    #     omegaW_w = omegaW_w.flatten()
    #     result = np.cross(omegaW_w, wing_point) #either we flatten omegaW_w or we reshape wing_point.reshape(3, 1)
    #     uW_w_position.append(result)
    omegaW_b = omegaW_b.flatten() #either we flatten omegaW_w or we reshape yWing.reshape(3, 1)
    uW_w_position = np.cross(omegaW_b, yWing)
    return uW_w_position

def generate_uW_g(uW_w, body_rotationMatrix_T, stroke_rotationMatrix_T, wing_rotationMatrix_T):
    rotationMatrix = np.matmul(np.matmul(body_rotationMatrix_T, stroke_rotationMatrix_T), wing_rotationMatrix_T)
    # uW_g = [] 
    # for point_velocity in uW_w:
    #     #print('\npoint velocity:\n', point_velocity) 
    #     u_W_g = np.matmul(rotationMatrix, point_velocity)
    #     # u_W_g = np.matmul(stroke_rotationMatrix_T, u_W_g)
    #     # u_W_g = np.matmul(body_rotationMatrix_T, u_W_g)
    #     #print('\nu w g:\n', u_W_g)
    #     uW_g.append(u_W_g)
    # # print('\n uW_w:\n', np.array(uW_w))
    # # print('\n uW_g: \n', np.array(uW_g))
    uW_g = np.matmul(rotationMatrix, uW_w)
    return uW_g

def orthogonal_vector(normal, R):
    return np.cross(normal, R)

def windDirection_from_global_to_body(vw_g, body_rotationMatrix):
    return np.matmul(body_rotationMatrix, vw_g.reshape(3,1))

def windDirection_from_body_to_stroke(vw_g, stroke_rotationMatrix):
    return np.matmul(stroke_rotationMatrix, vw_g.reshape(3,1))

def windDirection_from_stroke_to_wing(vw_g, wing_rotationMatrix):
    return np.matmul(wing_rotationMatrix, vw_g.reshape(3,1))

def getWindDirectioninWingReferenceFrame(vw_g, body_rotationMatrix, stroke_rotationMatrix, wing_rotationMatrix): 
    vw_b = windDirection_from_global_to_body(vw_g, body_rotationMatrix)
    vw_s = windDirection_from_body_to_stroke(vw_g, stroke_rotationMatrix)
    vw_w = windDirection_from_stroke_to_wing(vw_g, wing_rotationMatrix) 
    return vw_w

def getAoA(drag_vector, xWing):
    #should be in the wing reference frame
    aoa = np.arctan2(np.linalg.norm(np.cross(xWing, drag_vector)), np.dot(-drag_vector, xWing))
    return aoa  
        
def load_kinematics_data(file='kinematics_data_for_QSM.csv'): 
    t = [] 
    alpha = [] 
    phi = []
    theta = []
    alpha_dot = [] 
    phi_dot = []
    theta_dot = [] 

    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=';') 
        c = 0 
        for line in reader:
            if c>= 2:     
                timeStep, alphas, phis, thetas, alphas_dot, phis_dot, thetas_dot  = line
                t.append(float(timeStep))
                alpha.append(float(alphas))
                phi.append(float(phis))
                theta.append(float(thetas))
                alpha_dot.append(float(alphas_dot))
                phi_dot.append(float(phis_dot))
                theta_dot.append(float(thetas_dot))
            c += 1
    return np.array(t), np.radians(alpha), np.radians(phi), np.radians(theta), np.radians(alpha_dot), np.radians(phi_dot), np.radians(theta_dot)

def generateSequence (wing_points, wingtip_index, pivot_index, start_time=0, number_of_timesteps=360, frequency=1, useCFDData=True):
    vw_g = np.array([0, 0, 0])

    if useCFDData:
        timeline, alphas, phis, thetas, alphas_dot, phis_dot, thetas_dot = load_kinematics_data()
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

        #angles_dot
        phis_dot, alphas_dot, thetas_dot = create_wing_angles_dot(phis, alphas, thetas, timeline)

    alphas_interp = interp1d(timeline, alphas, fill_value='extrapolate')
    phis_interp = interp1d(timeline, phis, fill_value='extrapolate')
    thetas_interp = interp1d(timeline, thetas, fill_value='extrapolate')

    timeline = np.linspace(0, 1, 101)

    stroke_points_sequence = []
    body_points_sequence = []
    global_points_sequence = []

    omegasW_b = []
    omegasW_w = []

    usW_w = []
    usW_G = []

    vw_w = []
    AoA = [] 
    uW_g_vectors = [] 
    dW_g_vectors = []
    lift_vectors = [] 
    lift_vectors_norm = []

    delta_t = timeline[1] - timeline[0]
    print(delta_t)

    for timeStep in range(len(timeline)):
        t = timeline[timeStep]
        #parameter array: psi [0], beta[1], gamma[2], eta[3], phi[4], alpha[5], theta[6]
        alphas_dot = (alphas_interp(t+delta_t) - alphas_interp(t)) / delta_t
        phis_dot = (phis_interp(t+delta_t) - phis_interp(t)) / delta_t
        thetas_dot = (thetas_interp(t+delta_t) - thetas_interp(t)) / delta_t

        parameters = [0, 0, 0, -80*np.pi/180, phis_interp(t), alphas_interp(t), thetas_interp(t)] # 7 angles in radians! #without alphas[timeStep] any rotation around any y axis through an angle of pi/2 gives an error! 
        parameters_dot = [0, 0, 0, 0, phis_dot, alphas_dot, thetas_dot]
        stroke_points, [wing_rotationMatrix, wing_rotationMatrix_T] = convert_from_wing_reference_frame_to_stroke_plane(wing_points, parameters)
        body_points, [stroke_rotationMatrix, stroke_rotationMatrix_T] = convert_from_stroke_plane_to_body_reference_frame(stroke_points, parameters)
        global_points, [body_rotationMatrix, body_rotationMatrix_T] = convert_from_body_reference_frame_to_global_reference_frame(body_points, parameters)
        #print('\n\nthese are the global points before\n\n', global_points)
        # print('\n\nthis is the stroke rotation matrix: ', stroke_rotationMatrix)
        # print('\n\nthis is the transpose of stroke rotation matrix : ', stroke_rotationMatrix_T)
        stroke_points_sequence.append(stroke_points)
        body_points_sequence.append(body_points)
        global_points_sequence.append(global_points)

        xWing = np.matmul((np.matmul(stroke_rotationMatrix_T, wing_rotationMatrix_T)), np.array([[1], [0], [0]]))
        yWing = np.matmul((np.matmul(stroke_rotationMatrix_T, wing_rotationMatrix_T)), np.array([[0], [1], [0]]))
        print('xWing:', xWing, '\n', 'yWing:', yWing)

        omegaW_b, omegaW_w = generate_omegaW(wing_rotationMatrix, stroke_rotationMatrix, stroke_rotationMatrix_T, parameters[4], parameters_dot[4], parameters[5], 
                                    parameters_dot[5], parameters[6], parameters_dot[6])
        omegasW_b.append(omegaW_b)
        omegasW_w.append(omegaW_w)

        uWw = generate_uW_w_position(omegaW_b, yWing.flatten())
        vww = getWindDirectioninWingReferenceFrame(vw_g, body_rotationMatrix, stroke_rotationMatrix, wing_rotationMatrix).flatten() - np.array(uWw)
        
        # print('\n\n\nomega w w', omegaW_w)
        # print('\n\n\n points', wing_points)
        uWg = generate_uW_g(uWw, body_rotationMatrix_T, stroke_rotationMatrix_T, wing_rotationMatrix_T) #here its suppose to be transpose wing rotation matrix but we have to swap bc this code is all swaped 
        usW_G.append(uWg)
        usW_w.append(uWw)
        vw_w.append(vww)
        
        #print('AoA:', np.degrees(aoa[wingtip_index]))
        #print('alpha:', np.degrees(parameters[5]))
        #print('theta:', np.degrees(parameters[6]))

        #velocity vector here 
        uW_g_vector = uWw.flatten()
        # uW_g_vector = np.sum(usW_G[timeStep], axis=0)
        uW_g_vectors.append(uW_g_vector)
        uW_g_magnitude = np.sqrt(uW_g_vector[0]**2 + uW_g_vector[1]**2 + uW_g_vector[2]**2)
        if uW_g_magnitude != 0:  
            uW_g_vector_normalized = uW_g_vector/uW_g_magnitude
        else:
            uW_g_vector_normalized = uW_g_vector 
        dW_g_vector = -uW_g_vector_normalized
        dW_g_vectors.append(dW_g_vector)
    
        #lift 
        #R = global_points_sequence[timeStep][wingtip_index] - global_points_sequence[timeStep][pivot_index]
        #ax.quiver(X[pivot_index], Y[pivot_index], Z[pivot_index], R[0], R[1], R[2], color='red')
        lift_vector = orthogonal_vector(uW_g_vector_normalized, yWing.flatten())
        # print('lift vector before', lift_vector)
        # print('alphas', alphas[timeStep], timeStep)
        lift_vector = lift_vector*np.sign(alphas_interp(t))
        lift_vectors.append(lift_vector)
        # if lift_vector[2] < 0:
        #     lift_vector[2] = -lift_vector[2] 

        # if (60 <= np.degrees(alphas[timeStep]) <= 62): 
        #     pass
        # elif (-38 <= np.degrees((alphas[timeStep])) <= -36):
        #     pass
        # else:
        #     lift_vector *= 0 
        #     dW_g_vector *= 0 
        aoa = getAoA(dW_g_vector, xWing.flatten())
        AoA.append(aoa)
        lift_vector_mag = np.sqrt(lift_vector[0]**2 + lift_vector[1]**2 + lift_vector[2]**2)
        if lift_vector_mag != 0: 
            lift_vector_norm = lift_vector / lift_vector_mag
        else:
            lift_vector_norm = lift_vector
        lift_vectors_norm.append(lift_vector_norm)
        
        

    #validation of our uW_g: 
    #left and right derivative: 
    delta_t = timeline[1] - timeline[0]
    verifying_usW_g = []
    for timeStep in range(len(timeline)):
        currentGlobalPoint = global_points_sequence[timeStep]
        leftGlobalPoint = global_points_sequence[timeStep-1]
        rightGlobalPoint = global_points_sequence[(timeStep+1)%len(timeline)]
        LHD = (currentGlobalPoint - leftGlobalPoint) / delta_t
        RHD = (rightGlobalPoint - currentGlobalPoint) / delta_t
        verifying_usW_g.append((LHD+RHD)/2)
    verifying_usW_g = np.array(verifying_usW_g)


        
    # # plot to check omegaW_w's components 
    # omegasW_w = np.array(omegasW_w)
    # plt.plot(timeline[:-1], omegasW_w[:-1, 0], label='omegaW_w x')
    # plt.plot(timeline[:-1], omegasW_w[:-1, 1], label='omegaW_w y')
    # plt.plot(timeline[:-1], omegasW_w[:-1, 2], label='omegaW_w z')
    # plt.legend()
    # plt.show()
    # omegasW_b = np.array(omegasW_b)
    # plt.plot(timeline[:-1], omegasW_b[:-1, 0], label='omegaW_b x')
    # plt.plot(timeline[:-1], omegasW_b[:-1, 1], label='omegaW_b y')
    # plt.plot(timeline[:-1], omegasW_b[:-1, 2], label='omegaW_b z')
    # plt.legend()
    # plt.show()
    
    
    return timeline, global_points_sequence, body_points_sequence, stroke_points_sequence, phis, alphas, thetas, omegasW_b, omegasW_w, usW_w, usW_G, verifying_usW_g, vw_w, AoA, uW_g_vectors, dW_g_vectors, lift_vectors_norm
    #for each timestep in timeline : extract orientation of the points by performing appropriate rotations  

def animationPlot(ax, alphas, points_sequence, usW_g, AoA, wingtip_index, pivot_index, Fl, Fd, uW_g_vectors, dW_g_vectors, lift_vectors_norm, timeStep): 
    #get point set by timeStep number
    points = points_sequence[timeStep] #points_sequence can either be global, body, stroke 
    #clear the current axis 
    ax.cla()
    # print('these are the global points after: ', points)
    #extract the x, y and z coordinates 
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    #axis limit
    a = 4

    trajectory = np.array(points_sequence)[:, wingtip_index]
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
    uW_g_vector = uW_g_vectors[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], uW_g_vector[0], uW_g_vector[1], uW_g_vector[2], color='orange', label=r'$\overrightarrow{u}^{(g)}_w$' )
    # print('uw_g vector', uW_g_vector)
    dW_g_vector = dW_g_vectors[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], dW_g_vector[0], dW_g_vector[1], dW_g_vector[2], color='green', label=r'$\overrightarrow{d}^{(g)}_w$' )
    #lift 
    lift_vector = lift_vectors_norm[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], lift_vector[0], lift_vector[1], lift_vector[2])
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
    
def generatePlotsForKinematicsSequence(timeline, global_points_sequence, body_points_sequence, stroke_points_sequence, wing_points, phis, alphas, thetas, omegasW_b, omegasW_w, usW_w, usW_g, verifying_usW_g, vw_w, AoA, uW_g_vectors, dW_g_vectors, lift_vectors_norm, wingtip_index, pivot_index, Fl, Fd): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    anim = animation.FuncAnimation(fig, functools.partial(animationPlot, ax, alphas, global_points_sequence, usW_g, AoA,wingtip_index, pivot_index, Fl, Fd, uW_g_vectors, dW_g_vectors, lift_vectors_norm), frames=len(timeline), repeat=True)
    #anim.save('u&d_vectors.gif') 
    plt.show() 

def kinematics(): 
    ##########################################################################################################################

    #file import: get shape 
    wing_file = 'drosophilaMelanogasterWing.csv'
    pivot_index = -8
    wing_points = parse_wing_file(wing_file, 0.001, pivot_index)
    wingtip_index = 17

    # #print wing to check positioning!
    # plt.scatter(wing_points[:, 0], wing_points[:, 1])
    # i = 0
    # for wingpoint in wing_points:
    #     plt.text(wingpoint[0], wingpoint[1], str(i))
    #     i += 1
    # plt.xlim([-4,4])
    # plt.ylim([-4,4])
    # plt.show()

    # #run some checks! 
    # parameters = [0, 0, 0, np.pi/2, 0, 0, 0]
    # print('wing points', wing_points)
    # plot(wing_points, 'wing')
    # plt.show()

    # stroke_points, [stroke_rotationMatrix, stroke_rotationMatrix_T] = convert_from_wing_reference_frame_to_stroke_plane(wing_points, parameters)
    # print('stroke points', stroke_points)
    # plot(stroke_points, 'stroke')
    # plt.show()

    # body_points, [body_rotationMatrix, body_rotationMatrix_T] = convert_from_stroke_plane_to_body_reference_frame(stroke_points, parameters)
    # print('body points', body_points)
    # plot(body_points, 'body')
    # plt.show()

    # global_points, [global_rotationMatrix, global_rotationMatrix_T]= convert_from_body_reference_frame_to_global_reference_frame(body_points, parameters)
    # print('global points', global_points)
    # plot(global_points, 'global')
    # plt.show()

    #creation figure 
    timeline, global_points_sequence, body_points_sequence, stroke_points_sequence, phis, alphas, thetas, omegasW_b, omegasW_w, usW_w, usW_g, verifying_usW_g, vw_w, AoA, uW_g_vectors, dW_g_vectors, lift_vectors_norm = generateSequence(wing_points, wingtip_index, pivot_index, frequency=10, number_of_timesteps=360, useCFDData=True)
    #generatePlotsForKinematicsSequence(timeline, global_points_sequence, body_points_sequence, stroke_points_sequence, phis, alphas, thetas, omegasW_b, omegasW_w, usW_w, usW_g, verifying_usW_g, vw_w, AoA, wingtip_index, pivot_index)
    return timeline, global_points_sequence, body_points_sequence, stroke_points_sequence, wing_points, phis, alphas, thetas, omegasW_b, omegasW_w, usW_w, usW_g, verifying_usW_g, vw_w, AoA, uW_g_vectors, dW_g_vectors, lift_vectors_norm, wingtip_index, pivot_index

kinematics()