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
from datetime import datetime

#different cfd runs: #'phi120.00_phim20.00_dTau0.05' #'phi129.76_phim10.34_dTau0.00'
cfd_run = 'phi120.00_phim20.00_dTau0.05'
def main(cfd_run, folder_name):

    #timestamp variable for saving figures with actual timestamp 
    now = datetime.now()
    rightnow = now.strftime(" %d-%m-%Y_%I:%M:%S")+".png"

    #global variables:

    isLeft = wt.get_ini_parameter(cfd_run+'/PARAMS.ini', 'Insects', 'LeftWing', dtype=bool)
    wingShape = wt.get_ini_parameter(cfd_run+'/PARAMS.ini', 'Insects', 'WingShape', dtype=str)
    if 'from_file' in wingShape:
        wingShape_file = os.path.join(cfd_run, wingShape.replace('from_file::', ''))
    time_max = wt.get_ini_parameter(cfd_run+'/PARAMS.ini', 'Time', 'time_max', dtype=float)
    kinematics_file = wt.get_ini_parameter(cfd_run+'/PARAMS.ini', 'Insects', 'FlappingMotion_right', dtype=str)
    if 'from_file' in kinematics_file:
        kinematics_file = os.path.join(cfd_run, kinematics_file.replace('from_file::', '')) 

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

    # #load kinematics data by means of eval_angles_kinematics_file function from insect_tools library 
    # timeline, phis, alphas, thetas = it.eval_angles_kinematics_file(fname=kinematics_file, time=np.linspace(0.0, 1.0, 101)[:-1]) #time=np.linspace(0.0, 1.0, 101))
    # phis = np.radians(phis)
    # alphas = np.radians(alphas)
    # thetas = np.radians(thetas)

    # #timeline reassignment. when the timeline from the function eval_angles_kinematics_file is used 
    # #the last point of the timeline is the same as the first one (so value[0] = value[-1]) which creates a redundancy because 
    # #in python value[-1] = value[last]. when calculating time derivatives this redundancy jumbles up the code
    # #to solve this we can either do a variable reassigment or remove the last entry in timeline. the second method
    # #leaves you with one less point in the array. the first method is preferred. 
    # timeline = np.linspace(0, 1, timeline.shape[0])
    # nt = timeline.shape[0] #number of timesteps

    # load kinematics data by means of load_t_file function from insect_tools library 
    kinematics_cfd = it.load_t_file(cfd_run+'/kinematics.t')
    timeline = kinematics_cfd[:,0].flatten()
    alphas_it = kinematics_cfd[:,11].flatten()
    phis_it = kinematics_cfd[:,12].flatten()
    thetas_it = kinematics_cfd[:,13].flatten()

    #interpolate alpha, phi and theta  with respect to the original timeline
    alphas_interp = interp1d(timeline.flatten(), alphas_it.flatten(), fill_value='extrapolate')
    phis_interp = interp1d(timeline.flatten(), phis_it.flatten(), fill_value='extrapolate')
    thetas_interp = interp1d(timeline.flatten(), thetas_it.flatten(), fill_value='extrapolate')

    #timeline downsizing 
    timeline = np.linspace(0, 1, 101)

    alphas = alphas_interp(timeline)[:-1]
    phis = phis_interp(timeline)[:-1]
    thetas = thetas_interp(timeline)[:-1]

    timeline = np.linspace(0, 1, 100)
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

    blade_acc_wing_w = np.zeros((nt, 3))

    us_wing_w = np.zeros((nt, 3, 1))
    us_wing_g = np.zeros((nt, 3, 1))
    us_wing_g_magnitude = np.zeros((nt))

    acc_wing_w = np.zeros((nt, 3, 1))
    acc_wing_g = np.zeros((nt, 3, 1))

    rot_acc_wing_g = np.zeros((nt, 3, 1))
    rot_acc_wing_w = np.zeros((nt, 3, 1))

    us_wind_w = np.zeros((nt, 3, 1))

    AoA = np.zeros((nt, 1))
    e_dragVectors_wing_g = np.zeros((nt, 3))
    liftVectors = np.zeros((nt, 3))
    e_liftVectors_g = np.zeros((nt, 3))

    y_wing_g_sequence = np.zeros((nt, 3))
    z_wing_g_sequence = np.zeros((nt, 3))

    y_wing_s_sequence = np.zeros((nt, 3))

    y_wing_w_sequence = np.zeros((nt, 3))
    z_wing_w_sequence = np.zeros((nt, 3))

    e_Fam = np.zeros((nt, 3))

    wingRotationMatrix_sequence = np.zeros((nt, 3, 3))
    wingRotationMatrixTrans_sequence = np.zeros((nt, 3, 3))
    strokeRotationMatrix_sequence = np.zeros((nt, 3, 3))
    strokeRotationMatrixTrans_sequence = np.zeros((nt, 3, 3))
    bodyRotationMatrix_sequence = np.zeros((nt, 3, 3))
    bodyRotationMatrixTrans_sequence = np.zeros((nt, 3, 3))

    rotationMatrix_g_to_w = np.zeros((nt, 3, 3))
    rotationMatrix_w_to_g = np.zeros((nt, 3, 3))

    lever = np.zeros((nt))
    lever_g = np.zeros((nt))
    lever_w = np.zeros((nt))

    delta_t = timeline[1] - timeline[0]

    # forces_CFD = it.load_t_file('cfd_run/forces_rightwing.t', T0=[1.0,2.0])
    forces_CFD = it.load_t_file(cfd_run+'/forces_rightwing.t', T0=[1.0, 2.0])
    t = forces_CFD[:, 0]-1.0
    Fx_CFD_g = forces_CFD[:, 1]
    Fy_CFD_g = forces_CFD[:, 2]
    Fz_CFD_g = forces_CFD[:, 3]

    # moments_CFD = it.load_t_file('cfd_run/moments_rightwing.t', T0=[1.0, 2.0])
    moments_CFD = it.load_t_file(cfd_run+'/moments_rightwing.t', T0=[1.0, 2.0])
    #no need to read in the time again as it's the same from the forces file
    Mx_CFD_g = moments_CFD[:, 1]
    My_CFD_g = moments_CFD[:, 2]
    Mz_CFD_g = moments_CFD[:, 3]
    # M_CFD = moments_CFD[:, 1:4]

    if isLeft == 0: 
        print('The parsed data correspond to the right wing.')
    else: 
        print('The parsed data correspond to the left wing.')

    if np.round(forces_CFD[-1, 0], 2) != time_max: 
        raise ValueError('CFD cycle number does not match that of the actual run. Check your PARAMS, forces and moments files\n')

    print('The number of cycles is ', time_max, '. The forces and moments data were however only sampled for ', np.round(t[-1]), ' cycle(s)') #a cycle is defined as 1 downstroke + 1 upstroke ; cycle duration is 1.0 seconds. 

    Fx_CFD_g_interp = interp1d(t, Fx_CFD_g, fill_value='extrapolate')
    Fy_CFD_g_interp = interp1d(t, Fy_CFD_g, fill_value='extrapolate')
    Fz_CFD_g_interp = interp1d(t, Fz_CFD_g, fill_value='extrapolate')
    F_CFD_g = np.vstack((Fx_CFD_g_interp(timeline), Fy_CFD_g_interp(timeline), Fz_CFD_g_interp(timeline))).transpose()

    Mx_CFD_g_interp = interp1d(t, Mx_CFD_g, fill_value='extrapolate')
    My_CFD_g_interp = interp1d(t, My_CFD_g, fill_value='extrapolate')
    Mz_CFD_g_interp = interp1d(t, Mz_CFD_g, fill_value='extrapolate')
    M_CFD_g = np.vstack((Mx_CFD_g_interp(timeline), My_CFD_g_interp(timeline), Mz_CFD_g_interp(timeline))).transpose()

    Fx_CFD_w = np.zeros((t.shape[0]))
    Fy_CFD_w = np.zeros((t.shape[0]))
    Fz_CFD_w = np.zeros((t.shape[0])) 
    F_CFD_w = np.zeros((nt, 3))
    Fz_CFD_w_vector = np.zeros((nt, 3))

    # Mx_CFD_w = np.zeros((t.shape[0]))
    # My_CFD_w = np.zeros((t.shape[0]))
    # Mz_CFD_w = np.zeros((t.shape[0])) 
    # Mx_CFD_w_vector = np.zeros((nt, 3))

    M_CFD_w = np.zeros((nt, 3))

    Mx_QSM_w = np.zeros((nt))
    Mx_QSM_g = np.zeros((nt))

    #gloabl reference frame
    Ftc = np.zeros((nt, 3))
    Ftd = np.zeros((nt, 3))
    Frc = np.zeros((nt, 3))
    Fam = np.zeros((nt, 3))
    Frd = np.zeros((nt, 3))
    Fwe = np.zeros((nt, 3))

    #wing reference frame
    Ftc_w = np.zeros((nt, 3))
    Ftd_w = np.zeros((nt, 3))
    Frc_w = np.zeros((nt, 3))
    Fam_w = np.zeros((nt, 3))
    Frd_w = np.zeros((nt, 3))
    Fwe_w = np.zeros((nt, 3))

    F_QSM_w = np.zeros((nt, 3))
    F_QSM_g = np.zeros((nt, 3))
    Fz_QSM_w_vector = np.zeros((nt, 3))

    F_QSM_gg = np.zeros((nt, 3))

    Ftc_magnitude = np.zeros(nt)
    Ftd_magnitude = np.zeros(nt)
    Frc_magnitude = np.zeros(nt)
    Fam_magnitude = np.zeros(nt)
    Frd_magnitude = np.zeros(nt)
    Fwe_magnitude = np.zeros(nt)

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

    def convert_from_wing_reference_frame_to_stroke_plane(points, parameters):
        #points passed into this fxn must be in the wing reference frame x(w) y(w) z(w)
        #phi, alpha, theta
        phi = parameters[4] #rad
        alpha = parameters[5] #rad
        theta = parameters[6] #rad
        if isLeft == 0: 
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

    def convert_from_stroke_plane_to_body_reference_frame(points, parameters):
        #points must be in stroke plane x(s) y(s) z(s)
        eta = parameters[3] #rad
        flip_angle = 0 
        if isLeft == 0:
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
        if isLeft == 0:
            phi = -phi
            phi_dt = -phi_dt
            alpha = -alpha
            alpha_dt = -alpha_dt
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
        # rotationMatrix = np.matmul(np.matmul(bodyRotationMatrix, strokeRotationMatrix), wingRotationMatrix)
        # rotationMatrix = np.matmul(wingRotationMatrix, np.matmul(strokeRotationMatrix, bodyRotationMatrix))
        # u_wing_w = np.matmul(rotationMatrix, u_wing_g)
        u_wing_w = np.matmul(wingRotationMatrix, np.matmul(strokeRotationMatrix, np.matmul(bodyRotationMatrix, u_wing_g)))
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

    # #alternative definition of AoA 
    # def getAoA(e_u_wing_g, x_wing_g):
    #     #should be in the wing reference frame
    #     AoA = np.arctan2(np.linalg.norm(np.cross(x_wing_g, e_u_wing_g)), np.dot(e_u_wing_g, x_wing_g.reshape(3,1))) #rad
    #     return AoA  

    # def getLever(F, M): 
    # #to find the lever we need to solve for r: M = r x F. however since no inverse of the cross product exists, we have to use the vector triple product
    # #what we get is: r = M x F/norm(F)^2 + t*F ; where * is the dot product and t is any constant. for this equation to be valid, F and M must be orthogonal to each other, 
    # #such that F*M = 0
    #     for i in range(nt): 
    #         F_norm = (np.linalg.norm(F, axis=1))
    #         if F_norm[i] != 0:
    #             lever[i, :] = np.cross(F[i, :], M[i, :])/F_norm[i]**2 + 0*F[i, :] #here I take t = 0
    #         else: 
    #             lever[i, :] = 0 
    #     return lever 

    def getLever(M, F): 
    #to find the lever we need to solve for r: M = r x F. however since no inverse of the cross product exists, we have to use the vector triple product
    #what we get is: r = M x F/norm(F)^2 + t*F ; where * is the dot product and t is any constant. for this equation to be valid, F and M must be orthogonal to each other, 
    #such that F*M = 0
        for i in range(nt): 
            F_norm = (np.linalg.norm(F, axis=1))
            if F_norm[i] != 0:
                lever[i, :] = np.cross(M[i, :], F[i, :])/F_norm[i]**2 + 0*F[i, :] #here I take t = 0
            else: 
                lever[i, :] = 0 
        return lever

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
        u_wing_g_plot = us_wing_g[timeStep]
        ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], u_wing_g_plot[0], u_wing_g_plot[1], u_wing_g_plot[2], color='orange', label=r'$\overrightarrow{u}^{(g)}_w$' )
        
        #drag
        e_dragVector_wing_g = e_dragVectors_wing_g[timeStep]
        ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], e_dragVector_wing_g[0], e_dragVector_wing_g[1], e_dragVector_wing_g[2], color='green', label=r'$\overrightarrow{e_D}^{(g)}_w$' )
        
        #lift 
        liftVector_g = e_liftVectors_g[timeStep]
        ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], liftVector_g[0], liftVector_g[1], liftVector_g[2], color='blue', label=r'$\overrightarrow{e_L}^{(g)}_w$')
        
        #z_wing_g 
        z_wing_g_sequence_plot = z_wing_g_sequence[timeStep]
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
        ax.set_title(f'Timestep: {timeStep}, ⍺: {np.round(np.degrees(alphas[timeStep]), 2)}, AoA: {np.round(np.degrees(AoA[timeStep]), 2)} \nFl: {np.round(Ftc[timeStep], 4)} \nFd: {np.round(Ftd[timeStep], 4)} \nFrot: {np.round(Frc[timeStep], 4)} \nFam: {np.round(Fam[timeStep], 4)}')

    # run the live animation of the wing 
    def generatePlotsForKinematicsSequence():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        anim = animation.FuncAnimation(fig, functools.partial(animationPlot, ax), frames=len(timeline), repeat=True)
        #anim.save('u&d_vectors.gif') 
        plt.show()

    #kinematics
    for timeStep in range(nt):
        #here the 1st time derivatives of the angles are calculated by means of 2nd order central difference approximations
        alphas_dt = (alphas[(timeStep+1)%nt] - alphas[timeStep-1]) / (2*delta_t) #here we compute the modulus of (timestep+1) and nt to prevent overflowing. central difference 
        phis_dt = (phis[(timeStep+1)%nt] - phis[timeStep-1]) / (2*delta_t)
        thetas_dt = (thetas[(timeStep+1)%nt] - thetas[timeStep-1]) / (2*delta_t)

        # #here the 1st time derivatives of the angles are calculated by means of 1st order forward difference approximations
        # alphas_dt = (phis[(timeStep+1)%nt] - phis[timeStep]) / (delta_t)
        # phis_dt = (phis[(timeStep+1)%nt] - phis[timeStep]) / (delta_t) #1st order forward difference approximation of 1st derivative of phi
        # thetas_dt = (thetas[(timeStep+1)%nt] - thetas[timeStep]) / (delta_t)

        # parameter array: psi [0], beta[1], gamma[2], eta[3], phi[4], alpha[5], theta[6]
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

        rotationMatrix_g_to_w[timeStep, :] = np.matmul(wingRotationMatrix, np.matmul(strokeRotationMatrix, bodyRotationMatrix))
        rotationMatrix_w_to_g[timeStep, :] = np.matmul(np.matmul(bodyRotationMatrixTrans, strokeRotationMatrixTrans), wingRotationMatrixTrans)

        #these are all the absolute unit vectors of the wing 
        #y_wing_g coincides with the tip only if R is normalized. 
        # x_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[1], [0], [0]]))
        # y_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[0], [1], [0]]))
        # z_wing_g = np.matmul((np.matmul(np.matmul(strokeRotationMatrixTrans, wingRotationMatrixTrans), bodyRotationMatrixTrans)), np.array([[0], [0], [1]]))
        x_wing_g = np.matmul(bodyRotationMatrixTrans, (np.matmul(strokeRotationMatrixTrans, (np.matmul(wingRotationMatrixTrans, np.array([[1], [0], [0]]))))))
        y_wing_g = np.matmul(bodyRotationMatrixTrans, (np.matmul(strokeRotationMatrixTrans, (np.matmul(wingRotationMatrixTrans, np.array([[0], [1], [0]]))))))
        z_wing_g = np.matmul(bodyRotationMatrixTrans, (np.matmul(strokeRotationMatrixTrans, (np.matmul(wingRotationMatrixTrans, np.array([[0], [0], [1]]))))))

        y_wing_s = np.matmul(wingRotationMatrixTrans, np.array([[0], [1], [0]]))

        y_wing_g_sequence[timeStep, :] = y_wing_g.flatten()
        z_wing_g_sequence[timeStep, :] = z_wing_g.flatten()

        y_wing_s_sequence[timeStep, :] = y_wing_s.reshape(3,)

        y_wing_w = np.array([[0], [1], [0]])
        z_wing_w = np.array([[0], [0], [1]])
        y_wing_w_sequence[timeStep, :] = y_wing_w.reshape(3,)
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

        # u_wind_g = np.array([0.0, 0.0, 0.0])
        u_wind_g = np.array([0.248, 0.0, 0.6]) #absolute wind velocity 

        u_wing_g = generate_u_wing_g_position(rot_wing_g.reshape(1,3), y_wing_g.reshape(1,3)) + u_wind_g
        us_wing_g[timeStep, :] = (u_wing_g).reshape(3,1) #remember to rename variables since u_wind_g has been introduced! 

        u_wing_w = generate_u_wing_w(u_wing_g.reshape(3,1), bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix)
        us_wing_w[timeStep, :] = u_wing_w

        #absolute mean flow velocity is defined as the (linear) sum of the absolute winG velocity and the absolute winD velocity 
        u_flight_g = u_wing_g

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
        liftVector_g = np.cross(e_u_wing_g, y_wing_g.flatten())
        if isLeft == 0:
            liftVector_g = liftVector_g*np.sign(-alphas[timeStep])
        else:
            liftVector_g = liftVector_g*np.sign(alphas[timeStep])
        liftVectors[timeStep, :] = liftVector_g

        aoa = getAoA(x_wing_g.reshape(1,3), e_u_wing_g.reshape(3,1)) #use this one for getAoA with arccos 
        # aoa = getAoA(e_u_wing_g, x_wing_g.flatten()) #use this one for getAoA with arctan 
        AoA[timeStep, :] = aoa
        liftVector_magnitude = np.sqrt(liftVector_g[0, 0]**2 + liftVector_g[0, 1]**2 + liftVector_g[0, 2]**2)
        if liftVector_magnitude != 0: 
            e_liftVector_g = liftVector_g / liftVector_magnitude
        else:
            e_liftVector_g = liftVector_g
        e_liftVectors_g[timeStep, :] = e_liftVector_g

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

        # #validation of our u_wing_g by means of a first order approximation
        # #left and right derivative: 
        # verifying_us_wing_g = np.zeros((nt, wingPoints.shape[0], 3))
        # currentGlobalPoint = globalPointsSequence[timeStep]
        # leftGlobalPoint = globalPointsSequence[timeStep-1]
        # rightGlobalPoint = globalPointsSequence[(timeStep+1)%len(timeline)]
        # LHD = (currentGlobalPoint - leftGlobalPoint) / delta_t
        # RHD = (rightGlobalPoint - currentGlobalPoint) / delta_t
        # verifying_us_wing_g[timeStep, :] = (LHD+RHD)/2
        # verifying_us_wing_g = verifying_us_wing_g
        
    #calculation of wingtip acceleration and angular acceleration in wing reference frame 
    for timeStep in range(nt):
        acc_wing_w[timeStep, :] = np.matmul(rotationMatrix_g_to_w[timeStep, :], (us_wing_g[(timeStep+1)%nt] - us_wing_g[timeStep-1])/(2*delta_t)) #acc_wing_w, central difference
        # acc_wing_w[timeStep, :] = np.matmul(rotationMatrix_g_to_w[timeStep, :], (us_wing_g[(timeStep+1)%nt] - us_wing_g[timeStep])/(delta_t)) #acc_wing_w, forward difference
        acc_wing_g[timeStep, :] = (us_wing_g[(timeStep+1)%nt] - us_wing_g[timeStep-1])/(2*delta_t)

        rot_acc_wing_g[timeStep, :] = (rots_wing_g[(timeStep+1)%nt] - rots_wing_g[timeStep-1]) / (2*delta_t) #central scheme
        # rot_acc_wing_g[timeStep, :] = (rots_wing_g[(timeStep+1)%nt] - rots_wing_g[timeStep]) / (delta_t) #forward scheme
        rot_acc_wing_w[timeStep, :] = np.matmul(rotationMatrix_g_to_w[timeStep, :], rot_acc_wing_g[timeStep, :])
        
    #calculation of wingtip planar velocity (planar since the contribution from alpha/spanwise component is not taken into account) in global and wing reference frames 
    rots_wing_g_magnitude = np.linalg.norm(rots_wing_g, axis=1).reshape(nt,)
    planar_rots_wing_g_magnitude = np.linalg.norm(planar_rots_wing_g, axis=1).reshape(nt,) #here we reshap[(timeStep+1)%nt]e to fix dimensionality issues as planar_rots_wing_g_magnitude is of shape (nt, 1) and it should be of shape (nt,)

    rots_wing_w_magnitude = np.linalg.norm(rots_wing_w, axis=1).reshape(nt,)
    planar_rots_wing_w_magnitude = np.linalg.norm(planar_rots_wing_w, axis=1).reshape(nt,)

    #computation of M_CFD_w and F_CFD_w
    for i in range(nt):
        M_CFD_w[i, :] = np.matmul(rotationMatrix_g_to_w[i, :], M_CFD_g[i, :])
        # M_CFD_w[i, :] = np.matmul(wingRotationMatrix_sequence[i, :], np.matmul(strokeRotationMatrix_sequence[i, :], np.matmul(bodyRotationMatrix_sequence[i, :], M_CFD_g[i, :])))
        # Mx_CFD_w_vector[i, :] = M_CFD_w[i, :]
        # Mx_CFD_w_vector[i, 1:3] = 0 
        
        F_CFD_w[i, :] = np.matmul(rotationMatrix_g_to_w[i, :], F_CFD_g[i, :])
        # Fz_CFD_w_vector[i, :] = F_CFD_w[i, :]
        # Fz_CFD_w_vector[i, 0:2] = 0

    # data_new = it.insectSimulation_postProcessing(cfd_run)
    # t_Mw = data_new[1961:3921, 0]-1
    # Mx_CFD_w = data_new[1961:3921, 4]
    # My_CFD_w = data_new[1961:3921, 5]
    # Mz_CFD_w = data_new[1961:3921, 6]

    # Mx_CFD_w_interp = interp1d(t_Mw, Mx_CFD_w, fill_value='extrapolate')
    # My_CFD_w_interp = interp1d(t_Mw, My_CFD_w, fill_value='extrapolate')
    # Mz_CFD_w_interp = interp1d(t_Mw, Mz_CFD_w, fill_value='extrapolate')

    # # M_CFD_w = np.vstack((Mx_CFD_w_interp(timeline), My_CFD_w_interp(timeline), Mz_CFD_w_interp(timeline))).transpose()
    # # M_CFD_w = np.vstack((Mx_CFD_w, My_CFD_w, Mz_CFD_w)).transpose()

    # #cfd moments in wing reference frame (insect tools)
    # plt.plot(timeline, Mx_CFD_w_interp(timeline), label='Mx_CFD_w_it', color='red')
    # plt.plot(timeline, My_CFD_w_interp(timeline), label='My_CFD_w_it', color='green')
    # plt.plot(timeline, Mz_CFD_w_interp(timeline), label='Mz_CFD_w_it',  color='blue')
    # plt.xlabel('t/T [s]')
    # plt.ylabel('Moment [mN*mm]')

    # #cfd moments in wing reference frame 
    # plt.plot(timeline, M_CFD_w[:, 0], label='Mx_CFD_w', color='yellow')
    # plt.plot(timeline, M_CFD_w[:, 1], label='My_CFD_w', color='pink')
    # plt.plot(timeline, M_CFD_w[:, 2], label='Mz_CFD_w',  color='purple')
    # plt.xlabel('t/T [s]')
    # plt.ylabel('Moment [mN*mm]')
    # plt.title('CFD moments in wing reference frame')
    # plt.legend()
    # plt.savefig('debug_images/cfd moments_w insect_tools vs own'+rightnow, dpi=300)
    # plt.show()
    # exit()

    ############################################################################################################################################################################################
    ##%% dynamics

    from scipy.integrate import trapz, simpson
    import scipy.optimize as opt
    import time

    def getAerodynamicCoefficients(x0, AoA): 
        deg2rad = np.pi/180.0 
        rad2deg = 180.0/np.pi
        
        AoA = rad2deg*AoA
        
        # Cl and Cd definitions from Dickinson 1999
        Cl = x0[0] + x0[1]*np.sin( deg2rad*(2.13*AoA - 7.20) )
        Cd = x0[2] + x0[3]*np.cos( deg2rad*(2.04*AoA - 9.82) )
        Crot = x0[4]
        Cam1 = x0[5]
        Cam2 = x0[6]
        Crd = x0[7]
        # Cwe = x0[8]
        return Cl, Cd, Crot, Cam1, Cam2, Crd #, Cwe

    #cost function which tells us how far off our QSM values are from the CFD ones for the forces
    def cost_forces(x, nb=1000, show_plots=False):
        #global variable must be imported in order to modify them locally
        nonlocal Ftc_magnitude, Ftd_magnitude, Frc_magnitude, Fam_magnitude, Frd_magnitude, Fam, AoA

        Cl, Cd, Crot, Cam1, Cam2, Crd = getAerodynamicCoefficients(x, np.array(AoA))

        # chord calculation 
        y_space = np.linspace(min_y, max_y, nb)
        c = getChordLength(e_wingPoints, y_space)

        # plt.plot(c,y_space)
        # plt.show()
        # exit()

        rho = 1.225

        #both Cl and Cd are of shape (nt, 1), this however poses a dimensional issue when the magnitude of the lift/drag force is to be multiplied
        #with their corresponding vectors. to fix this, we reshape Cl and Cd to be of shape (nt,)
        Cl = Cl.reshape(nt,) 
        Cd = Cd.reshape(nt,)
        AoA = AoA.reshape(nt,)

        #computation following Nakata 2015 eqns. 2.4a-c
        c_interpolation = interp1d(y_space, c) #we create a function that interpolates our chord (c) w respect to our span (y_space)

        #the following comes from defining lift/drag in the following way: dFl = 0.5*rho*Cl*v^2*c*dr -> where v = linear velocity, c = chord length, dr = chord width
        #v can be broken into 𝛀(φ,Θ)*r  (cf. lines 245-248). plugging that into our equation we get: dFl = 0.5*rho*Cl*𝛀^2(φ,Θ)*r^2*c*dr (lift in each blade)
        #integrating both sides, and pulling constants out of integrand on RHS: Ftc = 0.5*rho*Cl*𝛀^2(φ,Θ)*∫c*r^2*dr 
        #our function def Cr2 then calculates the product of c and r^2 ; I (second moment of area) performs the integration of the product 
        #drag is pretty much the same except that instead of Cl we use Cd: Ftd = 0.5*rho*Cd*𝛀^2(φ,Θ)*∫c*r^2*dr
        #and the rotational force is defined as follows: Frc = 0.5*rho*Crot*𝛀(φ,Θ)*∫c^2*r*dr
        def Cr2(r): 
            return c_interpolation(r) * r**2
        def C2r(r):
            return (c_interpolation(r)**2) * r
        def C2(r):
            return (c_interpolation(r)**2)
        def C3r3(r):
            return(np.sqrt((c_interpolation(r)**3)*(r**3)))
        
        Iam = simpson(C2(y_space), y_space)
        Iwe = simpson(C3r3(y_space), y_space)
        Ild = simpson(Cr2(y_space), y_space) #second moment of area for lift/drag calculations
        Irot = simpson(C2r(y_space), y_space) #second moment of area for rotational force calculation 
        
        # #calculation of forces not absorbing wing shape related and density of fluid terms into force coefficients
        # Ftc_magnitude = 0.5*rho*Cl*(planar_rots_wing_w_magnitude**2)*Ild #Nakata et al. 2015
        # Ftd_magnitude = 0.5*rho*Cd*(planar_rots_wing_w_magnitude**2)*Ild #Nakata et al. 2015
        # Frc_magnitude = rho*Crot*planar_rots_wing_w_magnitude*alphas_dt_sequence*Irot #Nakata et al. 2015
        # Fam_magnitude = -Cam1*rho*np.pi/4*Iam*acc_wing_w[:, 2] -Cam2*rho*np.pi/8*Iam*rot_acc_wing_w[:, 1] #Cai et al. 2021 #second term should be time derivative of rots_wing_w 
        # Frd_magnitude = -1/6*rho*Crd*np.abs(alphas_dt_sequence)*alphas_dt_sequence #Cai et al. 2021
        # #Fwe_magnitude = 1/2*rho*rots_wing_w_magnitude*np.sqrt(rots_wing_w_magnitude)*Iwe*Cwe 
        # #Fwe_magnitude = 1/2*rho*phis*np.sign(phis_dt_sequence)*np.sqrt(np.abs(phis_dt_sequence))*Iwe*Cwe

        #calculation of forces absorbing wing shape related and density of fluid terms into force coefficients
        Ftc_magnitude = Cl*(planar_rots_wing_w_magnitude**2)
        Ftd_magnitude = Cd*(planar_rots_wing_w_magnitude**2)
        Frc_magnitude = Crot*planar_rots_wing_w_magnitude*alphas_dt_sequence
        Fam_magnitude = Cam1*acc_wing_w[:, 2] + Cam2*rot_acc_wing_w[:, 1]
        Frd_magnitude = Crd*np.abs(alphas_dt_sequence)*alphas_dt_sequence
        # Fwe_magnitude = Cwe*rots_wing_w_magnitude*np.sqrt(rots_wing_w_magnitude)

        # vector calculation of Ftc, Ftd, Frc, Fam, Frd and Fwe arrays of the form (nt, 3).these vectors are in the global reference frame 
        for i in range(nt):
            Ftc[i, :] = (Ftc_magnitude[i] * e_liftVectors_g[i])
            Ftd[i, :] = (Ftd_magnitude[i] * e_dragVectors_wing_g[i])
            Frc[i, :] = (Frc_magnitude[i] * z_wing_g_sequence[i])
            Fam[i, :] = (Fam_magnitude[i] * z_wing_g_sequence[i])
            Frd[i, :] = (Frd_magnitude[i] * z_wing_g_sequence[i])
            Fwe[i, :] = (Fwe_magnitude[i] * z_wing_g_sequence[i])

        Fx_QSM = Ftc[:, 0] + Ftd[:, 0] + Frc[:, 0] + Fam[:, 0] + Frd[:, 0] + Fwe[:, 0]
        Fy_QSM = Ftc[:, 1] + Ftd[:, 1] + Frc[:, 1] + Fam[:, 1] + Frd[:, 1] + Fwe[:, 1]
        Fz_QSM = Ftc[:, 2] + Ftd[:, 2] + Frc[:, 2] + Fam[:, 2] + Frd[:, 2] + Fwe[:, 2]

        F_QSM_g[:] = Ftc + Ftd + Frc + Fam + Frd + Fwe  

        K_num = np.linalg.norm(Fx_QSM-Fx_CFD_g_interp(timeline)) + np.linalg.norm(Fz_QSM-Fz_CFD_g_interp(timeline)) #+ np.linalg.norm(Fy_QSM+Fy_CFD_g_interp(timeline))
        K_den = np.linalg.norm(Fx_CFD_g_interp(timeline)) + np.linalg.norm(Fz_CFD_g_interp(timeline)) #+ np.linalg.norm(-Fy_CFD_g_interp(timeline))
        
        if K_den != 0: 
            K = K_num/K_den
        else:
            K = K_num

        for i in range(nt):
            F_QSM_w[i, :] = np.matmul(rotationMatrix_g_to_w[i, :], F_QSM_g[i, :])
            # Ftc_w[i, :] = np.matmul(rotationMatrix_g_to_w[i, :], Ftc[i, :]) 
            # Ftd_w[i, :] = np.matmul(rotationMatrix_g_to_w[i, :], Ftd[i, :])
            # Frc_w[i, :] = (Frc_magnitude[i] * z_wing_w_sequence[i])
            # Fam_w[i, :] = (Fam_magnitude[i] * z_wing_w_sequence[i])
            # Frd_w[i, :] = (Frd_magnitude[i] * z_wing_w_sequence[i])
            # Fwe_w[i, :] = (Fwe_magnitude[i] * z_wing_w_sequence[i])

            # Fz_QSM_w_vector[i, :] = F_QSM_w[i, :] 
            # Fz_QSM_w_vector[i, 0:2] = 0
            # F_QSM_w[i, :] = np.matmul(wingRotationMatrix_sequence[i, :], np.matmul(strokeRotationMatrix_sequence[i, :], np.matmul(bodyRotationMatrix_sequence[i, :], F_QSM_g[i, :])))
            # F_QSM_gg[i, :] = np.matmul(bodyRotationMatrixTrans_sequence[i, :], np.matmul(strokeRotationMatrixTrans_sequence[i, :], np.matmul(wingRotationMatrixTrans_sequence[i, :], F_QSM_w[i, :])))
            # F_QSM_gg[i, :] = np.matmul(rotationMatrix_w_to_g[i, :], F_QSM_w[i, :])
            # F_QSM_w[i, :] = np.matmul(rotationMatrix_g_to_w[i, :], np.array([[1], [1], [1]]).reshape(3,))
            # # F_QSM_w[i, :] = np.matmul(wingRotationMatrix_sequence[i, :], np.matmul(strokeRotationMatrix_sequence[i, :], np.matmul(bodyRotationMatrix_sequence[i, :], np.array([[1], [1], [1]]).reshape(3,))))

        if show_plots:

            ##FIGURE 1
            fig, axes = plt.subplots(3, 2, figsize = (15, 15))

            #angles
            axes[0, 0].plot(timeline, np.degrees(phis), label='ɸ')
            axes[0, 0].plot(timeline, np.degrees(alphas), label ='⍺')
            axes[0, 0].plot(timeline, np.degrees(thetas), label='Θ')
            axes[0, 0].plot(timeline, np.degrees(AoA), label='AoA', color = 'purple')
            axes[0, 0].set_xlabel('t/T [s]')
            axes[0, 0].set_ylabel('[˚]')
            axes[0, 0].legend()

            #u_wing_w (tip velocity in wing reference frame )
            axes[0, 1].plot(timeline, us_wing_w[:, 0], label='u_x_wing_w')
            axes[0, 1].plot(timeline, us_wing_w[:, 1], label='u_y_wing_w')
            axes[0, 1].plot(timeline, us_wing_w[:, 2], label='u_z_wing_w')
            axes[0, 1].set_xlabel('t/T [s]')
            axes[0, 1].set_ylabel('[mm/s]')
            axes[0, 1].set_title('Tip velocity in wing reference frame')
            axes[0, 1].legend()

            #a_wing_w (tip acceleration in wing reference frame )
            axes[1, 0].plot(timeline, acc_wing_w[:, 0], label='a_x_wing_w')
            axes[1, 0].plot(timeline, acc_wing_w[:, 1], label='a_y_wing_w')
            axes[1, 0].plot(timeline, acc_wing_w[:, 2], label='a_z_wing_w')
            axes[1, 0].set_xlabel('t/T [s]')
            axes[1, 0].set_ylabel('[mm/s²]')
            axes[1, 0].set_title('Tip acceleration in wing reference frame')
            axes[1, 0].legend()

            #rot_wing_w (tip velocity in wing reference frame )
            axes[1, 1].plot(timeline, rots_wing_w[:, 0], label='rot_x_wing_w')
            axes[1, 1].plot(timeline, rots_wing_w[:, 1], label='rot_y_wing_w')
            axes[1, 1].plot(timeline, rots_wing_w[:, 2], label='rot_z_wing_w')
            axes[1, 1].set_xlabel('t/T [s]')
            axes[1, 1].set_ylabel('rad/s')
            axes[1, 1].set_title('Angular velocity in wing reference frame')
            axes[1, 1].legend()

            #rot_acc_wing_w (angular acceleration in wing reference frame )
            axes[2, 0].plot(timeline, rot_acc_wing_w[:, 0], label='rot_acc_x_wing_w')
            axes[2, 0].plot(timeline, rot_acc_wing_w[:, 1], label='rot_acc_y_wing_w')
            axes[2, 0].plot(timeline, rot_acc_wing_g[:, 2], label='rot_acc_z_wing_w')
            axes[2, 0].set_xlabel('t/T [s]')
            axes[2, 0].set_ylabel('[rad/s]²')
            axes[2, 0].set_title('Angular acceleration in wing reference frame')
            axes[2, 0].legend()

            #alphas_dt
            axes[2, 1].plot(timeline, alphas_dt_sequence)
            axes[2, 1].set_xlabel('t/T [s]')
            axes[2, 1].set_ylabel('[˚/s]')
            axes[2, 1].set_title('Time derivative of alpha')
            axes[2, 1].legend()

            plt.subplots_adjust(left=0.07, bottom=0.05, right=0.960, top=0.970, wspace=0.185, hspace=0.28)
            # plt.subplot_tool()
            plt.show()
            # plt.savefig(folder_name+'/figure1.png', dpi=300)
            
            ##FIGURE 2
            fig, axes = plt.subplots(2, 2, figsize = (15, 15))

            #coefficients
            graphAoA = np.linspace(-9, 90, 100)*(np.pi/180)
            gCl, gCd, gCrot, gCam1, gCam2, gCrd = getAerodynamicCoefficients(x, graphAoA)
            axes[0, 0].plot(np.degrees(graphAoA), gCl, label='Cl', color='#0F95F1')
            axes[0, 0].plot(np.degrees(graphAoA), gCd, label='Cd', color='#F1AC0F')
            # ax.plot(np.degrees(graphAoA), gCrot*np.ones_like(gCl), label='Crot')
            axes[0, 0].set_title('Lift and drag coeffficients') 
            axes[0, 0].set_xlabel('AoA[°]')
            axes[0, 0].set_ylabel('[]')
            axes[0, 0].legend(loc = 'upper right') 
            axes[0, 0].legend() 

            #vertical forces
            axes[0, 1].plot(timeline, Ftc[:, 2], label = 'Vertical lift force', color='gold')
            axes[0, 1].plot(timeline, Frc[:, 2], label = 'Vertical rotational force', color='orange')
            axes[0, 1].plot(timeline, Ftd[:, 2], label = 'Vertical drag force', color='lightgreen')
            axes[0, 1].plot(timeline, Fam[:, 2], label = 'Vertical added mass force', color='red')
            axes[0, 1].plot(timeline, Frd[:, 2], label = 'Vertical rotational drag force', color='green')
            # axes[0, 1].plot(timeline, Fwe[:, 2], label = 'Vertical wagner effect force')
            axes[0, 1].plot(timeline, Fz_QSM, label = 'Vertical QSM force', ls='-.', color='blue')
            axes[0, 1].plot(timeline, Fz_CFD_g_interp(timeline), label = 'Vertical CFD force', ls='--', color='purple')
            axes[0, 1].set_xlabel('t/T [s]')
            axes[0, 1].set_ylabel('Force [mN]')
            axes[0, 1].set_title('Vertical components of forces in global coordinate system')
            # plt.savefig('debug/vertical_forces_no_Fam', dpi=2000)
            axes[0, 1].legend()
        
            # #vertical forces_w
            # plt.figure() 
            # plt.plot(timeline, Ftc[:, 2], label = 'Vertical lift force_w', color='gold')
            # plt.plot(timeline, Frc_w[:, 2], label = 'Vertical rotational force_w', color='orange')
            # plt.plot(timeline, Ftd_w[:, 2], label = 'Vertical drag force_w', color='lightgreen')
            # plt.plot(timeline, Fam_w[:, 2], label = 'Vertical added mass force_w', color='red')
            # plt.plot(timeline, Frd_w[:, 2], label = 'Vertical rotational drag force_w', color='green')
            # # plt.plot(timeline, Fwe_w[:, 2], label = 'Vertical wagner effect force_w')
            # plt.plot(timeline, F_QSM_w[:, 2], label = 'Vertical QSM force_w' , color='blue')
            # plt.xlabel('t/T [s]')
            # plt.ylabel('Force [mN]')
            # plt.legend()
            # # plt.savefig('debug/vertical_forces_no_Fam', dpi=300)
            # plt.show()

            #qsm + cfd force components in wing reference frame
            axes[1, 0].plot(timeline, F_QSM_w[:, 0], label='Fx_QSM_w')
            axes[1, 0].plot(timeline, F_QSM_w[:, 1], label='Fy_QSM_w')
            axes[1, 0].plot(timeline, F_QSM_w[:, 2], label='Fz_QSM_w')
            axes[1, 0].plot(timeline, F_CFD_w[:, 0], label='Fx_CFD_w')
            axes[1, 0].plot(timeline, F_CFD_w[:, 1], label='Fy_CFD_w')
            axes[1, 0].plot(timeline, F_CFD_w[:, 2], label='Fz_CFD_w')
            axes[1, 0].set_xlabel('t/T [s]')
            axes[1, 0].set_ylabel('Force [mN]')
            axes[1, 0].set_title('QSM + CFD force components in wing reference frame')
            # plt.savefig('debug_images/QSM forces_w; '+cfd_run+rightnow, dpi=300)
            axes[1, 0].legend()

            #forces
            axes[1, 1].plot(timeline[:], Fx_QSM, label='Fx_QSM_g', color='red')
            axes[1, 1].plot(timeline[:], Fy_QSM, label='Fy_QSM_g', color='green')
            axes[1, 1].plot(timeline[:], Fz_QSM, label='Fz_QSM_g', color='blue')
            axes[1, 1].plot(timeline[:], Fx_CFD_g_interp(timeline), label='Fx_CFD_g', linestyle = 'dashed', color='red')
            axes[1, 1].plot(timeline[:], Fy_CFD_g_interp(timeline), label='Fy_CFD_g', linestyle = 'dashed', color='green')
            axes[1, 1].plot(timeline[:], Fz_CFD_g_interp(timeline), label='Fz_CFD_g', linestyle = 'dashed', color='blue')
            axes[1, 1].set_xlabel('t/T [s]')
            axes[1, 1].set_ylabel('Force [mN]')
            axes[1, 1].set_title(f'Fx_QSM_g/Fx_CFD_g = {np.round(np.linalg.norm(Fx_QSM)/np.linalg.norm(Fx_CFD_g_interp(timeline)), 3)}; Fz_QSM_g/Fz_CFD_g = {np.round(np.linalg.norm(Fz_QSM)/np.linalg.norm(Fz_CFD_g_interp(timeline)), 3)}')
            # plt.savefig('debug_images/forces; no Fwe; all _w; '+cfd_run+rightnow, dpi=300)
            axes[1, 1].legend(loc = 'upper right') 

            # #qsm force components in global reference frame
            # plt.figure()
            # plt.plot(timeline, F_QSM_g[:, 0], label='Fx_g')
            # plt.plot(timeline, F_QSM_g[:, 1], label='Fy_g')
            # plt.plot(timeline, F_QSM_g[:, 2], label='Fz_g')
            # plt.xlabel('t/T [s]')
            # plt.ylabel('Force [mN]')
            # plt.legend()
            # plt.show()

            plt.subplots_adjust(left=0.07, bottom=0.05, right=0.960, top=0.970, wspace=0.185, hspace=0.28)
            # plt.subplot_tool()
            plt.show()
            # plt.savefig(folder_name+'/figure2.png', dpi=300)

            # generatePlotsForKinematicsSequence()
        return K

    #optimization by means of opt.differential_evolution which calculates the global minimum of our cost function (def F) and tells us 
    #for what x_0 values/input this minimum is attained  

    #optimizing using scipy.optimize.minimize which is faster
    def force_optimization():
        x_0_forces =  [0.03161,  0.03312,  0.07465, -0.04036,  0.0634,  -0.04163, -0.00789,  0.01615]  #[30, 63,  0.0001, -378, 8, 13, 23.76, 448.9, 98.12] #initial definition of x0 following Dickinson 1999
        bounds = [(-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6)]
        optimize = True
        nb = 2000 #nb: number of blades 
        if optimize:
            start = time.time()
            optimization = opt.minimize(cost_forces, args=(nb,), bounds=bounds, x0=x_0_forces)
            x0_forces_optimized = optimization.x
            K_forces_optimized = optimization.fun
            print('Computing for: ' + str(nb) + ' blades')
            print('Completed in:', round(time.time() - start, 4), 'seconds')
        else:
            x0_forces_optimized = [0.225, 1.58,  1.92, -1.55, 1, 1, 1, 1]
            K_forces_optimized = ''
            print('Computing for: ' + str(nb) + ' blades')
            # cost_forces(x0_forces_optimized, nb, show_plots=True)
        print('x0_forces_optimized:', np.round(x0_forces_optimized, 5), '\nK_optimized_forces:', K_forces_optimized)
        cost_forces(x0_forces_optimized, show_plots=False)
        return x0_forces_optimized, K_forces_optimized

    # #optimizing using scipy.optimize.differential_evolution which is considerably slower than scipy.optimize.minimize
    # #the results also fluctuate quite a bit using this optimizer.
    # def force_optimization():
    #     kinematics()
    #     x_0_forces = [0.225, 1.58,  1.92, -1.55, 1, 1, 1, 1, 1] #initial definition of x0 following Dickinson 1999
    #     bounds = [(-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6)]
    #     nb = 1000 #nb: number of blades 
    #     optimize = True
    #     if optimize:
    #         start = time.time()
    #         optimization = opt.differential_evolution(cost_forces, args=(nb,), bounds=bounds, x0=x_0_forces, maxiter=100)
    #         x0_forces_optimized = optimization.x
    #         K_optimized = optimization.fun
    #         print('Computing for: ' + str(nb) + ' blades')
    #         print('Completed in:', round(time.time() - start, 3), 'seconds')
    #     else:
    #         x0_forces_optimized = [1.76254482, -1.06909505,  1.12313521, -0.72540114]
    #         K_optimized = 0.5108267902800643
    #     print('x0_forces_optimized:', np.round(x0_forces_optimized, 5), '\nK_optimized:', K_optimized)
    #     cost_forces(x0_forces_optimized, show_plots=True)

    # import cProfile
    # import pstats
    # import io
    # profile = cProfile.Profile()
    # profile.enable()
    x0_force_optimized, K0_forces_optimized = force_optimization()
    # profile.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(profile, stream=s).sort_stats('cumulative') # tottime
    # ps.print_stats()
    # with open('debug/profile.txt', 'w+') as f:
    #     f.write(s.getvalue())

    def cost_moments(x, show_plots=False):
        nonlocal M_CFD_w, F_QSM_w, F_QSM_g

        C_lever = x[0]
        
        lever_w[:] = M_CFD_w[:, 0]/F_CFD_w[:, 2]
        
        Mx_QSM_w_nonoptimized = np.average(lever_w)*F_QSM_w[:, 2]

        Mx_QSM_w[:] = C_lever*F_QSM_w[:, 2]
        
        # writeArraytoFile(Mx_QSM_w, 'debug/Mx_QSM_w; '+cfd_run+rightnow+'.txt')

        K2_num = np.linalg.norm(Mx_QSM_w - M_CFD_w[:,0]) 
        K2_den = np.linalg.norm(M_CFD_w[:,0])
        
        if K2_den != 0: 
            K2 = K2_num/K2_den
        else:
            K2 = K2_num

        if show_plots:
            ##FIGURE 3
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 15))

            #cfd vs qsm x-component of moment 
            # plt.figure()
            ax1.plot(timeline[:], Mx_QSM_w_nonoptimized[:],  label='Mx_QSM_w', color='red')
            ax1.plot(timeline[:], M_CFD_w[:, 0], label='Mx_CFD_w', color='blue')
            ax1.set_xlabel('t/T [s]')
            ax1.set_ylabel('Moment [mN*mm]')
            ax1.set_title('Mx_QSM_w (non-optimized) .vs. Mx_CFD_w ')
            ax1.legend()
            # plt.savefig('debug_images/Mx_w QSM (no optimizer) vs CFD; '+cfd_run+rightnow, dpi=300)

            #cfd vs qsm x-component of moment 
            ax2.plot(timeline[:], Mx_QSM_w, label='Mx_QSM_w', color='red')
            ax2.plot(timeline[:], M_CFD_w[:, 0], label='Mx_CFD_w', color='blue')
            ax2.set_xlabel('t/T [s]')
            ax2.set_ylabel('Moment [mN*mm]')
            ax2.set_title('Mx_QSM_w .vs. Mx_CFD_w ')
            ax2.legend()
            # plt.savefig('debug_images/Mx_w QSM vs CFD; '+cfd_run+rightnow, dpi=300)

            # #lever
            # plt.figure()
            # # plt.plot(timeline, lever[:, 0], color='#C00891', label='Lever x-component')
            # # plt.plot(timeline, lever[:, 1], color='#0F2AEE', label='Lever y-component')
            # # plt.plot(timeline, lever[:, 2], color='#0FEE8C', label='Lever z-component')
            # # plt.plot(timeline, np.linalg.norm(lever, axis=1), color='#08C046', label='Lever magnitude')
            # plt.plot(timeline, lever)
            # plt.xlabel('t/T [s]')
            # plt.ylabel('Lever [mm]')
            # plt.legend()
            # # plt.savefig('debug_images/lever; '+cfd_run+rightnow, dpi=300)
            # plt.show()

            # #cfd moments in wing reference frame (insect tools)
            # plt.figure()
            # plt.plot(t_Mw, Mx_CFD_w, label='Mx_CFD_w', color='red')
            # plt.plot(t_Mw, My_CFD_w, label='My_CFD_w', color='green')
            # plt.plot(t_Mw, Mz_CFD_w, label='Mz_CFD_w',  color='blue')
            # plt.xlabel('t/T [s]')
            # plt.ylabel('Moment [mN*mm]')
            # plt.legend()
            # plt.show()    

            # #cfd moments in wing reference frame (insect tools)
            # plt.figure()
            # # plt.plot(timeline, Mx_CFD_g_interp(timeline), label='Mx_CFD_w', color='red')
            # plt.plot(timeline, My_CFD_w_interp(timeline), label='My_CFD_w', color='green')
            # # plt.plot(timeline, Mz_CFD_w_interp(timeline), label='Mz_CFD_w',  color='blue')
            # plt.xlabel('t/T [s]')
            # plt.ylabel('Moment [mN*mm]')
            # plt.legend()
            # plt.show()

            # #cfd moments in global reference frame
            # plt.figure() 
            # plt.plot(timeline[:], Mx_CFD_g_interp(timeline), label='Mx_CFD', linestyle = 'dashed', color='red')
            # plt.plot(timeline[:], My_CFD_g_interp(timeline), label='My_CFD', linestyle = 'dashed', color='green')
            # plt.plot(timeline[:], Mz_CFD_g_interp(timeline), label='Mz_CFD', linestyle = 'dashed', color='blue')
            # plt.xlabel('t/T [s]')
            # plt.ylabel('Moment [mN*mm]')
            # plt.legend()
            # # plt.savefig('debug/moments', dpi=2000)
            # plt.show()

            # #cfd moments in wing reference frame
            # plt.figure() 
            # plt.plot(timeline, M_CFD_w[:, 0], label='Mx_CFD_w', color='red')
            # plt.plot(timeline, M_CFD_w[:, 1], label='My_CFD_w', color='green')
            # plt.plot(timeline, M_CFD_w[:, 2], label='Mz_CFD_w',  color='blue')
            # plt.xlabel('t/T [s]')
            # plt.ylabel('Moment [mN*mm]')
            # plt.title('CFD moments in wing reference frame')
            # plt.legend()
            # # plt.savefig('debug_images/CFD moments_w; '+cfd_run+rightnow, dpi=300)
            # plt.show()  

            plt.subplots_adjust(left=0.07, bottom=0.05, right=0.960, top=0.970, wspace=0.185, hspace=0.28)
            plt.subplot_tool()
            plt.show()
            # plt.savefig(folder_name+'/figure2.png', dpi=300)

        return K2

    #moment optimization
    def moment_optimization():
        x_0_moments = [1.0]
        bounds = [(-6, 6)]
        optimize = True
        if optimize:
            start = time.time()
            optimization = opt.minimize(cost_moments, bounds=bounds, x0=x_0_moments)
            x0_moment_optimized = optimization.x
            K_moment_optimized = optimization.fun
            print('Completed in:', round(time.time() - start, 4), 'seconds')
        else:
            x0_moment_optimized = [1.0]
            K_moment_optimized = ''
            # cost_moments(x0_moment_optimized, show_plots=True)
        print('x0_moment_optimized:', np.round(x0_moment_optimized, 5), '\nK_optimized_moments:', K_moment_optimized)
        cost_moments(x0_moment_optimized, show_plots=True)
        return x0_moment_optimized, K_moment_optimized

    x0_moment_optimized, K0_moment_optimized = moment_optimization()
    print('Lever value before optimization, calculated from M_CFD_w[:, 0]/F_CFD_w[:, 2] :', np.round(np.average(lever), 6))

    return np.append(x0_force_optimized, K0_forces_optimized), np.append(x0_moment_optimized, K0_moment_optimized)

main(cfd_run, 'post-processing')