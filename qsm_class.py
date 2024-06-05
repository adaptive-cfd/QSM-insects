#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Nicolas Munoz Giraldo, TU Berlin, as part of B.Sc. thesis under the supervision of Thomas Engels, CRCN, CNRS Aix-Marseille U & TU Berlin

"""
import numpy as np
import wabbit_tools as wt
import insect_tools
import os
from scipy.interpolate import interp1d
import time
import scipy.optimize as opt
import matplotlib.pyplot as plt
from numpy.linalg import norm
from insect_tools import Rx, Ry, Rz, vct

# use latex
plt.rcParams["text.usetex"] = True

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TODO:
#   - model for two wings
#   - body force model
#   - inertial power 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class QSM:
    def __init__(self, nt=300):
        self.nt = nt
        
        self.timeline = np.linspace(0, 1, nt, endpoint=False)
        self.delta_t = self.timeline[1] - self.timeline[0]    
        
        # here all of the required variable arrays are created to match the size of the timeline 
        # since for every timestep each variable must be computed. this will happen in 'generateSequence'  
        self.alphas_dt = np.zeros((nt))
        self.phis_dt   = np.zeros((nt))
        self.thetas_dt = np.zeros((nt))
        
        self.alphas_dtdt = np.zeros((nt))
        self.phis_dtdt   = np.zeros((nt))
        self.thetas_dtdt = np.zeros((nt))
               
        self.rots_wing_b = np.zeros((nt, 3, 1))
        self.rots_wing_s = np.zeros((nt, 3, 1))
        self.rots_wing_w = np.zeros((nt, 3, 1))
        self.rots_wing_g = np.zeros((nt, 3, 1))
        
        self.planar_rots_wing_s = np.zeros((nt, 3, 1))
        self.planar_rots_wing_w = np.zeros((nt, 3, 1))
        self.planar_rots_wing_g = np.zeros((nt, 3, 1))
                
        self.us_wing_w = np.zeros((nt, 3, 1))
        self.us_wing_g = np.zeros((nt, 3, 1))
        self.us_wing_g_magnitude = np.zeros((nt))
        
        self.acc_wing_w = np.zeros((nt, 3, 1))
        self.acc_wing_g = np.zeros((nt, 3, 1))
        
        self.rot_acc_wing_g = np.zeros((nt, 3, 1))
        self.rot_acc_wing_w = np.zeros((nt, 3, 1))
        
        self.us_wind_w = np.zeros((nt, 3, 1))
        
        self.AoA = np.zeros((nt, 1))
        self.e_dragVectors_wing_g = np.zeros((nt, 3))
        self.liftVectors = np.zeros((nt, 3))
        self.e_liftVectors_g = np.zeros((nt, 3))
        
        self.ey_wing_g = np.zeros((nt, 3))
        self.ez_wing_g = np.zeros((nt, 3))
        
        self.ey_wing_s = np.zeros((nt, 3))
        
        self.ey_wing_w = np.zeros((nt, 3))
        self.ez_wing_w = np.zeros((nt, 3))
        
        self.e_Fam = np.zeros((nt, 3))
        
        self.wingRotationMatrix = np.zeros((nt, 3, 3))
        self.wingRotationMatrixTrans = np.zeros((nt, 3, 3))
        self.strokeRotationMatrix = np.zeros((nt, 3, 3))
        self.strokeRotationMatrixTrans = np.zeros((nt, 3, 3))
        self.bodyRotationMatrix = np.zeros((nt, 3, 3))
        self.bodyRotationMatrixTrans = np.zeros((nt, 3, 3))
        
        self.rotationMatrix_g_to_w = np.zeros((nt, 3, 3))
        self.rotationMatrix_w_to_g = np.zeros((nt, 3, 3))
                        
        #global reference frame
        self.Ftc = np.zeros((nt, 3))
        self.Ftd = np.zeros((nt, 3))
        self.Frc = np.zeros((nt, 3))
        self.Fam = np.zeros((nt, 3))
        self.Frd = np.zeros((nt, 3))
        self.Fwe = np.zeros((nt, 3))
        
        #wing reference frame
        self.Ftc_w = np.zeros((nt, 3))
        self.Ftd_w = np.zeros((nt, 3))
        self.Frc_w = np.zeros((nt, 3))
        self.Fam_w = np.zeros((nt, 3))
        self.Frd_w = np.zeros((nt, 3))
        self.Fwe_w = np.zeros((nt, 3))
        
        self.F_QSM_w         = np.zeros((nt, 3))
        self.F_QSM_g         = np.zeros((nt, 3))
                
        self.Ftc_magnitude = np.zeros(nt)
        self.Ftd_magnitude = np.zeros(nt)
        self.Frc_magnitude = np.zeros(nt)
        self.Fam_magnitude = np.zeros(nt)
        self.Frd_magnitude = np.zeros(nt)
        self.Fwe_magnitude = np.zeros(nt)
        
        self.Mx_QSM_w = np.zeros(nt)
        self.My_QSM_w = np.zeros(nt)
        self.Mz_QSM_w = np.zeros(nt)
        
        # wing-geometry-dependent constants
        # they are computed (if desired) in setup_wing_shape
        self.Iam  = 1.0
        self.Iwe  = 1.0
        self.Ild  = 1.0
        self.Irot = 1.0
        
        self.wing = 'left'
        self.wingPoints = np.zeros((10,3))
        
    def parse_kinematics(self, params_file, kinematics_file, plot=True, wing='auto'):
        """
        Evaluate the kinematics given by kinematics_file and store the results
        in the class arrays. The kinematics file can be either: 
            - "kinematics.t", the output log file of a CFD run
            - an *.ini file, which is the CFD-code's descriptor file for kinematics


        what about eta etc? is not in the file

        """
                
        #generate rot wing calculates the angular velocity of the wing in all reference frames, as well as the planar angular velocity {𝛀(φ,Θ)} 
        #which will later be used to calculate the forces on the wing.  planar angular velocity {𝛀(φ,Θ)} comes from the decomposition of the motion
        #into 'translational' and rotational components, with the rotational component beig defined as ⍺ (the one around the y-axis in our convention)
        #this velocity is obtained by setting ⍺ to 0, as can be seen below
        def generate_rot_wing(wingRotationMatrix, bodyRotationMatrixTrans, strokeRotationMatrixTrans, phi, phi_dt, alpha, alpha_dt, theta, theta_dt, wing): 
            if wing == "right":
                phi = -phi
                phi_dt = -phi_dt
                alpha = -alpha
                alpha_dt = -alpha_dt
                
            phiMatrixTrans   = np.transpose(Rx(phi)) 
            alphaMatrixTrans = np.transpose(Ry(alpha)) 
            thetaMatrixTrans = np.transpose(Rz(theta))
            
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
        
       
        # since the absolute linear velocity of the wing depends both on time and on the position along the wing
        # this function calculates only its position dependency
        def generate_u_wing_g_position(rot_wing_g, ey_wing_g):
            # #omega x point
            #both input vectors have to be reshaped to (1,3) to meet the requirements of np.cross (last axis of both vectors -> 2 or 3). to that end either reshape(1,3) or flatten() kommen in frage
            u_wing_g_position = np.cross(rot_wing_g, ey_wing_g)
            return u_wing_g_position

        #this function calculates the linear velocity of the wing in the wing reference frame
        def generate_u_wing_w(u_wing_g, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix):
            u_wing_w = np.matmul(wingRotationMatrix, np.matmul(strokeRotationMatrix, np.matmul(bodyRotationMatrix, u_wing_g)))
            return u_wing_w

        def getWindDirectioninWingReferenceFrame(u_flight_g, bodyRotationMatrix, strokeRotationMatrix, wingRotationMatrix): 
            u_wind_b = np.matmul(bodyRotationMatrix, u_flight_g.reshape(3,1))
            u_wind_s = np.matmul(strokeRotationMatrix, u_wind_b)
            u_wind_w = np.matmul(wingRotationMatrix, u_wind_s)
            return u_wind_w

        #in this code AoA is defined as the arccos of the dot product between the unit vector along x direction and the unit vector of the absolute linear velocity
        def getAoA(ex_wing_g, e_u_wing_g):
            AoA = np.arccos(np.dot(ex_wing_g, e_u_wing_g))
            return AoA
        
        # left or right wing?
        if wing == 'auto':
            # read from PARAMS-file; this is the default. If we use QSM on a two (or four) winged simulation,
            # we create one QSM model for each wing.
            isLeft   = wt.get_ini_parameter(params_file, 'Insects', 'LeftWing', dtype=bool)
            isRight  = wt.get_ini_parameter(params_file, 'Insects', 'RightWing', dtype=bool)
            
            if isLeft:
                wing == "left"
            else:
                wing == "right"
            
            if isLeft and isRight:
                raise ValueError("This simulation included more than one wing, you need to create one QSM model per wing, pass wing=right and wing=left")
                
        self.wing = wing
            
        # insects cruising speed        
        self.u_infty_g = -np.asarray( wt.get_ini_parameter(params_file, 'ACM-new', 'u_mean_set', vector=True) ) # note sign change (wind vs body)
        
        if "kinematics.t" in kinematics_file:
            # load kinematics data by means of load_t_file function from insect_tools library 
            kinematics_cfd = insect_tools.load_t_file(kinematics_file)#, remove_outliers=True)#, interp=True, time_out=self.timeline)
            time_it   = kinematics_cfd[:, 0].flatten()
            psis_it   = kinematics_cfd[:, 4].flatten()
            betas_it  = kinematics_cfd[:, 5].flatten()
            gammas_it = kinematics_cfd[:, 6].flatten()
            etas_it   = kinematics_cfd[:, 7].flatten()
            
            if self.wing == "right": 
                # right wing
                alphas_it = kinematics_cfd[:,11].flatten()
                phis_it   = kinematics_cfd[:,12].flatten()
                thetas_it = kinematics_cfd[:,13].flatten()
                
                # As WABBIT also computes as stores angular velocity & acceleration
                # we can use this to verify the QSM computation done here
                self.debug_rotx_wing_g = kinematics_cfd[:, 17].flatten()
                self.debug_roty_wing_g = kinematics_cfd[:, 18].flatten()
                self.debug_rotz_wing_g = kinematics_cfd[:, 19].flatten()                
                self.debug_time = kinematics_cfd[:,0].flatten()                
                self.debug_rotx_dt_wing_g = kinematics_cfd[:, 23].flatten()
                self.debug_roty_dt_wing_g = kinematics_cfd[:, 24].flatten()
                self.debug_rotz_dt_wing_g = kinematics_cfd[:, 25].flatten()      
            else:
                # left wing
                alphas_it = kinematics_cfd[:,  8].flatten()
                phis_it   = kinematics_cfd[:,  9].flatten()
                thetas_it = kinematics_cfd[:, 10].flatten()                
                
                # As WABBIT also computes as stores angular velocity & acceleration
                # we can use this to verify the QSM computation done here
                self.debug_rotx_wing_g = kinematics_cfd[:, 14].flatten()
                self.debug_roty_wing_g = kinematics_cfd[:, 15].flatten()
                self.debug_rotz_wing_g = kinematics_cfd[:, 16].flatten()                
                self.debug_time = kinematics_cfd[:,0].flatten()                
                self.debug_rotx_dt_wing_g = kinematics_cfd[:, 20].flatten()
                self.debug_roty_dt_wing_g = kinematics_cfd[:, 21].flatten()
                self.debug_rotz_dt_wing_g = kinematics_cfd[:, 22].flatten()                
                
                
            #interpolate psi, beta, gamma, alpha, phi and theta  with respect to the original timeline
            psis_interp   = interp1d(time_it, psis_it  , fill_value='extrapolate')
            betas_interp  = interp1d(time_it, betas_it , fill_value='extrapolate')
            gammas_interp = interp1d(time_it, gammas_it, fill_value='extrapolate')
            etas_interp   = interp1d(time_it, etas_it  , fill_value='extrapolate')
            alphas_interp = interp1d(time_it, alphas_it, fill_value='extrapolate')
            phis_interp   = interp1d(time_it, phis_it  , fill_value='extrapolate')
            thetas_interp = interp1d(time_it, thetas_it, fill_value='extrapolate')
            
            self.psis   = psis_interp(self.timeline)
            self.betas  = betas_interp(self.timeline)
            self.gammas = gammas_interp(self.timeline)
            self.etas   = etas_interp(self.timeline)
            self.alphas = alphas_interp(self.timeline)
            self.phis   = phis_interp(self.timeline)
            self.thetas = thetas_interp(self.timeline)
            
        elif ".ini" in kinematics_file:
            # load kinematics data by means of eval_angles_kinematics_file function from insect_tools library 
            timeline, phis, alphas, thetas = insect_tools.eval_angles_kinematics_file(fname=kinematics_file, time=self.timeline)
            
            # convert to rad
            self.phis   = np.radians(phis)
            self.alphas = np.radians(alphas)
            self.thetas = np.radians(thetas)            
            
            yawpitchroll0 = wt.get_ini_parameter(params_file, 'Insects', 'yawpitchroll_0', vector=True)
            eta_stroke    = wt.get_ini_parameter(params_file, 'Insects', 'eta0', dtype=float )
            
            # yaw pitch roll and stroke plane angle are all constant in time
            self.psis   = np.zeros_like(self.phis) + yawpitchroll0[2]
            self.betas  = np.zeros_like(self.phis) + yawpitchroll0[1]
            self.gammas = np.zeros_like(self.phis) + yawpitchroll0[0]
            self.etas   = np.zeros_like(self.phis) + eta_stroke
        
        dt, nt = self.delta_t, self.nt
        
        # kinematics
        for timeStep in range(self.nt):
            #here the 1st time derivatives of the angles are calculated by means of 2nd order central difference approximations
            self.alphas_dt[timeStep] = (self.alphas[(timeStep+1)%nt] - self.alphas[timeStep-1]) / (2*dt) #here we compute the modulus of (timestep+1) and nt to prevent overflowing. central difference 
            self.phis_dt[timeStep]   = (self.phis[(timeStep+1)%nt]   -   self.phis[timeStep-1]) / (2*dt)
            self.thetas_dt[timeStep] = (self.thetas[(timeStep+1)%nt] - self.thetas[timeStep-1]) / (2*dt)
            
            # second order approx for second derivatives of input angles
            self.alphas_dtdt[timeStep] = (self.alphas[(timeStep+1)%nt] -2*self.alphas[timeStep] + self.alphas[timeStep-1]) / (dt**2)
            self.phis_dtdt[timeStep]   = (self.phis[(timeStep+1)%nt]   -2*self.phis[timeStep]   + self.phis[timeStep-1]  ) / (dt**2)
            self.thetas_dtdt[timeStep] = (self.thetas[(timeStep+1)%nt] -2*self.thetas[timeStep] + self.thetas[timeStep-1]) / (dt**2)

            alpha, phi, theta, eta, psi, beta, gamma = self.alphas[timeStep], self.phis[timeStep], self.thetas[timeStep], self.etas[timeStep], self.psis[timeStep], self.betas[timeStep], self.gammas[timeStep]
            alpha_dt, phi_dt, theta_dt = self.alphas_dt[timeStep], self.phis_dt[timeStep], self.thetas_dt[timeStep]
            
            M_wing   = insect_tools.M_wing(alpha, theta, phi, wing=self.wing)
            M_stroke = insect_tools.M_stroke(eta, self.wing)
            M_body   = insect_tools.M_body(psi, beta, gamma)
            M_g2w    = M_wing*M_stroke*M_body
            M_w2g    = M_g2w.T
            
            self.wingRotationMatrix[timeStep, :]        = M_wing
            self.strokeRotationMatrix[timeStep, :]      = M_stroke
            self.bodyRotationMatrix[timeStep, :]        = M_body
            self.wingRotationMatrixTrans[timeStep, :]   = np.transpose(M_wing)
            self.strokeRotationMatrixTrans[timeStep, :] = np.transpose(M_stroke)
            self.bodyRotationMatrixTrans[timeStep, :]   = np.transpose(M_body)
        
            self.rotationMatrix_g_to_w[timeStep, :]     = M_g2w
            self.rotationMatrix_w_to_g[timeStep, :]     = M_w2g
        
            # these are all the absolute unit vectors of the wing 
            # ey_wing_g coincides with the tip only if R is normalized. 
            ex_wing_g = M_w2g   * vct([1,0,0])
            ey_wing_g = M_w2g   * vct([0,1,0])
            ez_wing_g = M_w2g   * vct([0,0,1])        
            ey_wing_s = M_wing.T * vct([0,1,0])
            
        
            self.ey_wing_g[timeStep, :] = ey_wing_g.flatten()
            self.ez_wing_g[timeStep, :] = ez_wing_g.flatten()
            
            self.ey_wing_s[timeStep, :] = ey_wing_s.reshape(3,)
        
            ey_wing_w = np.array([[0], [1], [0]])
            ez_wing_w = np.array([[0], [0], [1]])
            self.ey_wing_w[timeStep, :] = ey_wing_w.reshape(3,)
            self.ez_wing_w[timeStep, :] = ez_wing_w.reshape(3,)
        
            rot_wing_g, rot_wing_b, rot_wing_s, rot_wing_w, planar_rot_wing_g, planar_rot_wing_s, planar_rot_wing_w = generate_rot_wing(
                self.wingRotationMatrix[timeStep, :], self.bodyRotationMatrixTrans[timeStep, :], self.strokeRotationMatrixTrans[timeStep, :], 
                phi, phi_dt, alpha, alpha_dt, theta, theta_dt, self.wing)
        
            self.rots_wing_b[timeStep, :] = rot_wing_b
            self.rots_wing_s[timeStep, :] = rot_wing_s
            self.rots_wing_w[timeStep, :] = rot_wing_w
            self.rots_wing_g[timeStep, :] = rot_wing_g
        
            self.planar_rots_wing_s[timeStep, :] = planar_rot_wing_s
            self.planar_rots_wing_w[timeStep, :] = planar_rot_wing_w
            self.planar_rots_wing_g[timeStep, :] = planar_rot_wing_g
        
            self.planar_rots_wing_w_magnitude = norm( self.planar_rots_wing_w, axis=1).reshape(nt,)
        
            u_wing_g = generate_u_wing_g_position(rot_wing_g.reshape(1,3), ey_wing_g.reshape(1,3)) + self.u_infty_g
            self.us_wing_g[timeStep, :] = (u_wing_g).reshape(3,1) #remember to rename variables since u_infty has been introduced! 
        
            u_wing_w = generate_u_wing_w(u_wing_g.reshape(3,1), self.bodyRotationMatrix[timeStep, :], self.strokeRotationMatrix[timeStep, :], self.wingRotationMatrix[timeStep, :])
            self.us_wing_w[timeStep, :] = u_wing_w
        
       
            u_wing_g_magnitude = norm(u_wing_g)
            self.us_wing_g_magnitude[timeStep] = u_wing_g_magnitude
        
            if u_wing_g_magnitude != 0:  
                e_u_wing_g = u_wing_g/u_wing_g_magnitude
            else:
                e_u_wing_g = u_wing_g 
            e_dragVector_wing_g = -e_u_wing_g
            self.e_dragVectors_wing_g[timeStep, :] = e_dragVector_wing_g
        
            # lift. lift vector is multiplied with the sign of alpha to have their signs match 
            liftVector_g = np.cross(e_u_wing_g, ey_wing_g.flatten())
            if self.wing == "right":
                liftVector_g = liftVector_g * np.sign(-alpha)
            else:
                liftVector_g = liftVector_g * np.sign(+alpha)
        
            aoa = getAoA(ex_wing_g.reshape(1,3), e_u_wing_g.reshape(3,1)) #use this one for getAoA with arccos 
            self.AoA[timeStep, :] = aoa
            
            
            liftVector_magnitude = np.sqrt(liftVector_g[0, 0]**2 + liftVector_g[0, 1]**2 + liftVector_g[0, 2]**2)
            if liftVector_magnitude != 0: 
                e_liftVector_g = liftVector_g / liftVector_magnitude
            else:
                e_liftVector_g = liftVector_g
            self.e_liftVectors_g[timeStep, :] = e_liftVector_g
        
            
        #calculation of wingtip acceleration and angular acceleration in wing reference frame 
        for timeStep in range(nt):
            # time derivative in inertial frame
            self.acc_wing_g[timeStep, :] = (self.us_wing_g[(timeStep+1)%nt] - self.us_wing_g[timeStep-1])/(2*dt)
            self.acc_wing_w[timeStep, :] = np.matmul(self.rotationMatrix_g_to_w[timeStep, :], self.acc_wing_g[timeStep, :])
        
            # time derivative in inertial frame
            self.rot_acc_wing_g[timeStep, :] = (self.rots_wing_g[(timeStep+1)%nt] - self.rots_wing_g[timeStep-1]) / (2*dt) #central scheme
            self.rot_acc_wing_w[timeStep, :] = np.matmul(self.rotationMatrix_g_to_w[timeStep, :], self.rot_acc_wing_g[timeStep, :])

        if plot:
            ## FIGURE 1
            fig, axes = plt.subplots(3, 2, figsize = (15, 15))

            # angles
            ax = axes[0,0]
            ax.plot(self.timeline, np.degrees(self.phis), label='$\\phi$')
            ax.plot(self.timeline, np.degrees(self.alphas), label ='$\\alpha$')
            ax.plot(self.timeline, np.degrees(self.thetas), label='$\\theta$')
            ax.plot(self.timeline, np.degrees(self.AoA), label='AoA', color = 'purple')
            ax.set_xlabel('$t/T$')
            ax.set_ylabel('(deg)')
            ax.legend() 
            
            # time derivatives of angles
            ax = axes[0,1]
            ax.plot(self.timeline, self.phis_dt, label='$\\dot\\phi$')
            ax.plot(self.timeline, self.alphas_dt, label='$\\dot\\alpha$')
            ax.plot(self.timeline, self.thetas_dt, label='$\\dot\\theta$')
            
            if self.wing == "right":
                ax.plot(self.timeline, np.sign(-self.alphas), 'k--', label='$\\mathrm{sign}(\\alpha)$' )
            else:
                ax.plot(self.timeline, np.sign(+self.alphas), 'k--', label='$\\mathrm{sign}(\\alpha)$' )
            ax.set_xlabel('$t/T$')
            ax.legend()

            # u_wing_w (tip velocity in wing reference frame )
            ax = axes[1,0]
            ax.plot(self.timeline, self.us_wing_w[:, 0], label='$u_{\\mathrm{wing},x}^{(w)}$')
            ax.plot(self.timeline, self.us_wing_w[:, 1], label='$u_{\\mathrm{wing},y}^{(w)}$')
            ax.plot(self.timeline, self.us_wing_w[:, 2], label='$u_{\\mathrm{wing},z}^{(w)}$')            
            ax.plot(self.timeline, self.us_wing_g_magnitude, 'k--', label='$\\left| \\underline{u}_{\\mathrm{wing}} \\right|$')
            
            ax.set_xlabel('$t/T$')
            ax.set_ylabel('[Rf]')
            ax.set_title('Tip velocity in wing reference frame, $\\mathrm{mean}\\left(\\left|\\underline{u}_{\\mathrm{wing}}\\right|\\right)=%2.2f$' % (np.mean(self.us_wing_g_magnitude)))
            ax.legend()

            #a_wing_w (tip acceleration in wing reference frame )
            ax = axes[1,1]
            ax.plot(self.timeline, self.acc_wing_w[:, 0], label='$\\dot{u}_{\\mathrm{wing},x}^{(w)}$')
            ax.plot(self.timeline, self.acc_wing_w[:, 1], label='$\\dot{u}_{\\mathrm{wing},y}^{(w)}$')
            ax.plot(self.timeline, self.acc_wing_w[:, 2], label='$\\dot{u}_{\\mathrm{wing},z}^{(w)}$')
            ax.set_xlabel('$t/T$')
            ax.set_ylabel('$Rf^2$')
            ax.set_title('Tip acceleration in wing reference frame')
            ax.legend()

            #rot_wing_w (tip velocity in wing reference frame )
            ax = axes[2,0]
            ax.plot(self.timeline, self.rots_wing_w[:, 0], label='$\\Omega_{\\mathrm{wing},x}^{(w)}$')
            ax.plot(self.timeline, self.rots_wing_w[:, 1], label='$\\Omega_{\\mathrm{wing},y}^{(w)}$')
            ax.plot(self.timeline, self.rots_wing_w[:, 2], label='$\\Omega_{\\mathrm{wing},z}^{(w)}$')
            ax.set_xlabel('$t/T$')
            ax.set_ylabel('rad/T')
            ax.set_title('Angular velocity in wing reference frame')
            ax.legend()

            #rot_acc_wing_w (angular acceleration in wing reference frame )
            ax = axes[2,1]
            ax.plot(self.timeline, self.rot_acc_wing_w[:, 0], label='$\\dot\\Omega_{\\mathrm{wing},x}^{(w)}$')
            ax.plot(self.timeline, self.rot_acc_wing_w[:, 1], label='$\\dot\\Omega_{\\mathrm{wing},y}^{(w)}$')
            ax.plot(self.timeline, self.rot_acc_wing_g[:, 2], label='$\\dot\\Omega_{\\mathrm{wing},z}^{(w)}$')
            ax.set_xlabel('$t/T$')
            ax.set_ylabel('[rad/T²]')
            ax.set_title('Angular acceleration in wing reference frame')
            ax.legend()



            plt.tight_layout()
            plt.draw()
            
            for ax in axes.flatten():
                insect_tools.indicate_strokes(ax=ax)
            

            
    def fit_to_CFD(self, cfd_run, paramsfile, T0=0.0, optimize=True):
        """
        Train the QSM model with a CFD run. This works only if you have initialized 
            * the kinematics with parse_kinematics_file
            * the wing shape with setup_wing_shape
        before calling this routine. 
        
        The routine reads the CFD data from the given directory cfd_run, and uses
        the given parameter file (often called PARAMS.ini or similar).
        
        We train the QSM model to a single wing currently. If you try to process a two-winged
        simulation, an error is raised.
        
        The optimized coefficients are stored in the QSM object.
        """
        
        self.time_max = wt.get_ini_parameter(paramsfile, 'Time', 'time_max', dtype=float)

        # setup the wing planform
        # NB: one can actually skip this, but not in the current implementation
        wingShape = wt.get_ini_parameter(paramsfile, 'Insects', 'WingShape', dtype=str)        
        if 'from_file' in wingShape:
            wingShape_file = os.path.join(cfd_run, wingShape.replace('from_file::', ''))
        else:
            # if the shape is unknown, setup_wing_shape returns a pseudo-wing shape contour
            # note for computing the QSM, a shape model is in fact optional.
            wingShape_file = 'none'
        
        self.setup_wing_shape(wingShape_file )
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ read CFD force/moments/power ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('The parsed data correspond to the %s wing.' % (self.wing))

           
        # figure out what cycle to use
        d = insect_tools.load_t_file(cfd_run+'/forces_'+self.wing+'wing.t', verbose=False)
        time_start, time_end = d[0,0], d[-1,0]
        print("CFD data t=[%f, %f]" % (time_start, time_end))
        print("QSM model uses t=[%f, %f]" % (T0, T0+1.0))
        
        # read in data from desired cycle (hence the shift by T0)
        # NOTE that load_t_file can remove outliers in the data, which turned out very useful for the 
        # musca domestica data (which have large jumps exactly at the reversals)
        forces_CFD  = insect_tools.load_t_file(cfd_run+'/forces_'+self.wing+'wing.t', interp=True, time_out=self.timeline+T0, remove_outliers=True)
        moments_CFD = insect_tools.load_t_file(cfd_run+'/moments_'+self.wing+'wing.t', interp=True, time_out=self.timeline+T0, remove_outliers=True)

        self.Fx_CFD_g = forces_CFD[:, 1]
        self.Fy_CFD_g = forces_CFD[:, 2]
        self.Fz_CFD_g = forces_CFD[:, 3]

        self.Mx_CFD_g = moments_CFD[:, 1]
        self.My_CFD_g = moments_CFD[:, 2]
        self.Mz_CFD_g = moments_CFD[:, 3]
        
        self.F_CFD_g = np.vstack((self.Fx_CFD_g, self.Fy_CFD_g, self.Fz_CFD_g)).transpose()
        self.M_CFD_g = np.vstack((self.Mx_CFD_g, self.My_CFD_g, self.Mz_CFD_g)).transpose()
               
        # computation of M_CFD_w and F_CFD_w
        self.F_CFD_w = np.zeros_like( self.F_CFD_g )
        self.M_CFD_w = np.zeros_like( self.M_CFD_g )
        
        for i in range(self.nt):
            self.M_CFD_w[i, :] = np.matmul(self.rotationMatrix_g_to_w[i, :], self.M_CFD_g[i, :])  
            self.F_CFD_w[i, :] = np.matmul(self.rotationMatrix_g_to_w[i, :], self.F_CFD_g[i, :])
            
        self.Fx_CFD_w = self.F_CFD_w[:,0]
        self.Fy_CFD_w = self.F_CFD_w[:,1]
        self.Fz_CFD_w = self.F_CFD_w[:,2]

        self.Mx_CFD_w = self.M_CFD_w[:,0]
        self.My_CFD_w = self.M_CFD_w[:,1]
        self.Mz_CFD_w = self.M_CFD_w[:,2]            
        
        # aerodynamic power. Can be read from t-file or computed from the dot-product of moment and angular 
        # velocity. This latter is done here: we do not store the power for each wing separately in the CFD run, and hence
        # computing it here is the better choice if possibly more than one wing has been simulated.
        self.P_CFD = -(  self.Mx_CFD_w*self.rots_wing_w[:,0,0] 
                       + self.My_CFD_w*self.rots_wing_w[:,1,0]  
                       + self.Mz_CFD_w*self.rots_wing_w[:,2,0])
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        def getAerodynamicCoefficients(x0, AoA): 
            deg2rad = np.pi/180.0 
            rad2deg = 180.0/np.pi
            
            AoA = rad2deg*AoA
            
            # Cl and Cd definitions from Dickinson 1999
            Cl   = x0[0] + x0[1]*np.sin( deg2rad*(2.13*AoA - 7.20) )
            Cd   = x0[2] + x0[3]*np.cos( deg2rad*(2.04*AoA - 9.82) )
            Crot = x0[4]
            Cam1 = x0[5]
            Cam2 = x0[6]
            Crd  = x0[7]
            # Cwe = x0[8]
            return Cl, Cd, Crot, Cam1, Cam2, Crd #, Cwe
        
        #%% force optimization

        # cost function which tells us how far off our QSM values are from the CFD ones for the forces
        def cost_forces( x, self, show_plots=False):            
            Cl, Cd, Crot, Cam1, Cam2, Crd = getAerodynamicCoefficients(x, self.AoA)

    
            rho = 1.225 # for future work, can also be set to 1.0 simply
            nt = self.nt

            # both Cl and Cd are of shape (nt, 1), this however poses a dimensional issue when the magnitude of the lift/drag force is to be multiplied
            # with their corresponding vectors. to fix this, we reshape Cl and Cd to be of shape (nt,)
            Cl = Cl.reshape(nt,) 
            Cd = Cd.reshape(nt,)
            
            # these are either precomputed from the wing shape or simply set to one.
            # The model can do without, as the optimizer simply adjusts the coefficients 
            # accordingly.
            Iam = self.Iam
            Iwe = self.Iwe
            Ild = self.Ild
            Irot = self.Irot
        
            
            # calculation of forces not absorbing wing shape related and density of fluid terms into force coefficients
            self.Ftc_magnitude = 0.5*rho*Cl*(self.planar_rots_wing_w_magnitude**2)*Ild #Nakata et al. 2015
            self.Ftd_magnitude = 0.5*rho*Cd*(self.planar_rots_wing_w_magnitude**2)*Ild #Nakata et al. 2015
            self.Frc_magnitude = rho*Crot*self.planar_rots_wing_w_magnitude*self.alphas_dt*Irot #Nakata et al. 2015
            self.Fam_magnitude = -Cam1*rho*np.pi/4*Iam*self.acc_wing_w[:, 2] -Cam2*rho*np.pi/8*Iam*self.rot_acc_wing_w[:, 1] #Cai et al. 2021 #second term should be time derivative of rots_wing_w 
            self.Frd_magnitude = -1/6*rho*Crd*np.abs(self.alphas_dt)*self.alphas_dt#Cai et al. 2021
 

            # vector calculation of Ftc, Ftd, Frc, Fam, Frd and Fwe arrays of the form (nt, 3).these vectors are in the global reference frame 
            for i in range(nt):
                self.Ftc[i, :] = (self.Ftc_magnitude[i] * self.e_liftVectors_g[i])
                self.Ftd[i, :] = (self.Ftd_magnitude[i] * self.e_dragVectors_wing_g[i])
                self.Frc[i, :] = (self.Frc_magnitude[i] * self.ez_wing_g[i])
                self.Fam[i, :] = (self.Fam_magnitude[i] * self.ez_wing_g[i])
                self.Frd[i, :] = (self.Frd_magnitude[i] * self.ez_wing_g[i])
                self.Fwe[i, :] = (self.Fwe_magnitude[i] * self.ez_wing_g[i])

            # total force generated by QSM
            self.Fx_QSM_g = self.Ftc[:, 0] + self.Ftd[:, 0] + self.Frc[:, 0] + self.Fam[:, 0] + self.Frd[:, 0] + self.Fwe[:, 0]
            self.Fy_QSM_g = self.Ftc[:, 1] + self.Ftd[:, 1] + self.Frc[:, 1] + self.Fam[:, 1] + self.Frd[:, 1] + self.Fwe[:, 1]
            self.Fz_QSM_g = self.Ftc[:, 2] + self.Ftd[:, 2] + self.Frc[:, 2] + self.Fam[:, 2] + self.Frd[:, 2] + self.Fwe[:, 2]

            self.F_QSM_g[:] = self.Ftc + self.Ftd + self.Frc + self.Fam + self.Frd + self.Fwe  

            K_forces_num = norm(self.Fx_QSM_g-self.Fx_CFD_g) + norm(self.Fz_QSM_g-self.Fz_CFD_g)
            K_forces_den = norm(self.Fx_CFD_g) + norm(self.Fz_CFD_g)
            
            if K_forces_den != 0: 
                K_forces = K_forces_num/K_forces_den
            else:
                K_forces = K_forces_num

            for i in range(nt):
                self.F_QSM_w[i, :] = np.matmul( self.rotationMatrix_g_to_w[i, :], self.F_QSM_g[i, :])

            if show_plots:

                ##FIGURE 2
                fig, axes = plt.subplots(2, 2, figsize = (15, 10))

                #coefficients
                graphAoA = np.linspace(-9, 90, 100)*(np.pi/180)
                gCl, gCd, gCrot, gCam1, gCam2, gCrd = getAerodynamicCoefficients(x, graphAoA)
                axes[0, 0].plot(np.degrees(graphAoA), gCl, label='Cl', color='#0F95F1')
                axes[0, 0].plot(np.degrees(graphAoA), gCd, label='Cd', color='#F1AC0F')
                # ax.plot(np.degrees(graphAoA), gCrot*np.ones_like(gCl), label='Crot')
                axes[0, 0].set_title('Lift and drag coeffficients') 
                axes[0, 0].set_xlabel('AoA[°]')
                axes[0, 0].set_ylabel('[-]')
                axes[0, 0].legend(loc = 'upper right') 

                #vertical forces
                axes[0, 1].plot(self.timeline, self.Ftc[:, 2], label = 'Vertical lift force', color='gold')
                axes[0, 1].plot(self.timeline, self.Frc[:, 2], label = 'Vertical rotational force', color='orange')
                axes[0, 1].plot(self.timeline, self.Ftd[:, 2], label = 'Vertical drag force', color='lightgreen')
                axes[0, 1].plot(self.timeline, self.Fam[:, 2], label = 'Vertical added mass force', color='red')
                axes[0, 1].plot(self.timeline, self.Frd[:, 2], label = 'Vertical rotational drag force', color='green')
                # axes[0, 1].plot(timeline, Fwe[:, 2], label = 'Vertical wagner effect force')
                axes[0, 1].plot(self.timeline, self.Fz_QSM_g, label = 'Vertical QSM force', ls='-.', color='blue')
                axes[0, 1].plot(self.timeline, self.Fz_CFD_g, label = 'Vertical CFD force', ls='--', color='purple')
                axes[0, 1].set_xlabel('$t/T$')
                axes[0, 1].set_ylabel('force')
                axes[0, 1].set_title('Vertical components of forces in global coordinate system')
                axes[0, 1].legend(loc = 'lower right')
            
                #qsm + cfd force components in wing reference frame
                axes[1, 0].plot(self.timeline, self.F_QSM_w[:, 0], label='Fx_QSM_w', c='r')
                axes[1, 0].plot(self.timeline, self.F_CFD_w[:, 0], ls='-.', label='Fx_CFD_w', c='r')
                axes[1, 0].plot(self.timeline, self.F_QSM_w[:, 1], label='Fy_QSM_w', c='g')
                axes[1, 0].plot(self.timeline, self.F_CFD_w[:, 1], ls='-.', label='Fy_CFD_w', c='g')
                axes[1, 0].plot(self.timeline, self.F_QSM_w[:, 2], label='Fz_QSM_w', c='b')
                axes[1, 0].plot(self.timeline, self.F_CFD_w[:, 2], ls='-.', label='Fz_CFD_w', c='b')
                axes[1, 0].set_xlabel('$t/T$')
                axes[1, 0].set_ylabel('force')
                axes[1, 0].set_title('QSM + CFD force components in wing reference frame')
                axes[1, 0].legend()

                #forces
                axes[1, 1].plot(self.timeline[:], self.Fx_QSM_g, label='Fx_QSM_g', color='red')
                axes[1, 1].plot(self.timeline[:], self.Fx_CFD_g, label='Fx_CFD_g', linestyle = 'dashed', color='red')
                axes[1, 1].plot(self.timeline[:], self.Fy_QSM_g, label='Fy_QSM_g', color='green')
                axes[1, 1].plot(self.timeline[:], self.Fy_CFD_g, label='Fy_CFD_g', linestyle = 'dashed', color='green')
                axes[1, 1].plot(self.timeline[:], self.Fz_QSM_g, label='Fz_QSM_g', color='blue')            
                axes[1, 1].plot(self.timeline[:], self.Fz_CFD_g, label='Fz_CFD_g', linestyle = 'dashed', color='blue')
                axes[1, 1].set_xlabel('$t/T$')
                axes[1, 1].set_ylabel('force')
                axes[1, 1].set_title( "Fi_QSM_g/Fi_CFD_g=(%2.2f, %2.2f, %2.2f) \nK=%2.3f" % (norm(self.Fx_QSM_g)/norm(self.Fx_CFD_g), 
                                                                                    norm(self.Fy_QSM_g)/norm(self.Fy_CFD_g), 
                                                                                    norm(self.Fz_QSM_g)/norm(self.Fz_CFD_g),
                                                                                    K_forces) ) 
                axes[1, 1].legend(loc = 'lower right') 


                for ax in axes.flatten()[1:]:
                    insect_tools.indicate_strokes(ax=ax)

                plt.tight_layout()
                plt.draw()
                
            return K_forces
        
        #----------------------------------------------------------------------
        # optimizing using scipy.optimize.minimize which is faster
        
        if optimize:
            start = time.time()
            
            bounds = [(-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6)]
            K_forces = 9e9
            
            # optimize 3 times from a different initial guess, use best solution found
            # NOTE: tests indicate the system always finds the same solution, so this could
            # be omitted. Kept for safety - we do less likely get stuck in local minima this way
            for itrial in range(3):
                x0_forces = np.random.rand(8)
                optimization = opt.minimize(cost_forces, args=(self, False), bounds=bounds, x0=x0_forces)
                
                x0_forces = optimization.x
                
                if optimization.fun < K_forces:
                    K_forces = optimization.fun
                    x0_best = x0_forces                
                    
            self.x0_forces = x0_best     
            self.K_forces = K_forces
            
            print('Completed in:', round(time.time() - start, 4), 'seconds')
            print('x0_optimized:', np.round(self.x0_forces, 5), '\nK_optimized_forces:', K_forces)
            
        cost_forces(self.x0_forces, self, show_plots=True)
        
        #%% moments optimization
        
        #cost_moments is defined in terms of the moments. this function will be optimized to find the lever (coordinates) that best matches (match) the QSM moments to their CFD counterparts  
        def cost_moments(x, self, show_plots=False):
            #here we define the the QSM moments as: M_QSM = [ C_lever_x_w*Fz_QSM_w, -C_lever_x_w*Fz_QSM_w, C_lever_x_w*Fy_QSM_w - C_lever_y_w*F_x_QSM_w ]
            #where C_lever_x_w and C_lever_y_w correspond to the spanwise and the chordwise locations of the lever in the wing reference frame. 
            #vector form: C_lever_w = [C_lever_x_w, C_lever_y_w, 0]

            C_lever_x_w = x[0]
            C_lever_y_w = x[1]            

            self.Mx_QSM_w[:] =  C_lever_y_w*self.F_QSM_w[:, 2]
            self.My_QSM_w[:] = -C_lever_x_w*self.F_QSM_w[:, 2]
            self.Mz_QSM_w[:] =  C_lever_x_w*self.F_QSM_w[:, 1] - C_lever_y_w*self.F_QSM_w[:, 0]

            K_moments_num = norm(self.Mx_QSM_w - self.M_CFD_w[:,0]) + norm(self.My_QSM_w - self.M_CFD_w[:,1]) + norm(self.Mz_QSM_w - self.M_CFD_w[:,2]) 
            K_moments_den = norm(self.M_CFD_w[:,0]) + norm(self.M_CFD_w[:,1]) + norm(self.M_CFD_w[:,2]) 
            
            if K_moments_den != 0: 
                K_moments = K_moments_num/K_moments_den
            else:
                K_moments = K_moments_num

            return K_moments

        # moment optimization
        if optimize:
            x0_moments = [1.0, 1.0]
            bounds = [(-6, 6), (-6, 6)]

            start = time.time()
            
            optimization = opt.minimize(cost_moments, args=(self, False), bounds=bounds, x0=x0_moments)
            
            self.x0_moments = optimization.x
            self.K_moments  = optimization.fun
            
            print('Completed in:', round(time.time() - start, 4), 'seconds')
            
            print('x0_moments_optimized:', np.round(self.x0_moments, 5), '\nK_moments_optimized:', self.K_moments)
        cost_moments(self.x0_moments, self, True)




        #%% power optimization
        
        #cost_power is defined in terms of the moments and power. this function will be optimized to find the lever (coordinates) that best matches (match) the QSM power to its CFD counterpart
        def cost_power(x, self, show_plots=False):
            #here we define the the QSM moments as: M_QSM = [ C_lever_x_w_power*Fz_QSM_w, -C_lever_x_w_power*Fz_QSM_w, C_lever_x_w_power*Fy_QSM_w - C_lever_y_w_power*F_x_QSM_w ]
            #where C_lever_x_w_power and C_lever_y_w_power correspond to the spanwise and the chordwise locations of the lever in the wing reference frame. 
            #vector form: C_lever_w = [C_lever_x_w_power, C_lever_y_w_power, 0]

            C_lever_x_w_power = x[0]
            C_lever_y_w_power = x[1]

            self.Mx_QSM_w_power =  C_lever_y_w_power*self.F_QSM_w[:, 2].flatten()
            self.My_QSM_w_power = -C_lever_x_w_power*self.F_QSM_w[:, 2].flatten()
            self.Mz_QSM_w_power =  C_lever_x_w_power*self.F_QSM_w[:, 1].flatten() - C_lever_y_w_power*self.F_QSM_w[:, 0].flatten()


            self.P_QSM_nonoptimized = -(self.Mx_QSM_w*self.rots_wing_w[:, 0].flatten()
                                      + self.My_QSM_w*self.rots_wing_w[:, 1].flatten() 
                                      + self.Mz_QSM_w*self.rots_wing_w[:, 2].flatten())

            self.P_QSM = -(self.Mx_QSM_w_power*self.rots_wing_w[:, 0].flatten() 
                         + self.My_QSM_w_power*self.rots_wing_w[:, 1].flatten() 
                         + self.Mz_QSM_w_power*self.rots_wing_w[:, 2].flatten())

            K_power_num = norm(self.P_QSM - self.P_CFD) 
            K_power_den = norm(self.P_CFD)

            if K_power_den != 0: 
                K_power = K_power_num/K_power_den
            else:
                K_power = K_power_num

            if show_plots:
                ##FIGURE 4
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 15))

                #cfd vs qsm moments
                ax1.plot(self.timeline, self.Mx_QSM_w, label='Mx_QSM_w', color='red')
                ax1.plot(self.timeline, self.M_CFD_w[:, 0], label='Mx_CFD_w', ls='-.', color='red')
                ax1.plot(self.timeline, self.My_QSM_w, label='My_QSM_w', color='blue')
                ax1.plot(self.timeline, self.M_CFD_w[:, 1], label='My_CFD_w', ls='-.', color='blue')
                ax1.plot(self.timeline, self.Mz_QSM_w, label='Mz_QSM_w', color='green')
                ax1.plot(self.timeline, self.M_CFD_w[:, 2], label='Mz_CFD_w', ls='-.', color='green')
                ax1.set_xlabel('$t/T$')
                ax1.set_ylabel('moment]')                
                ax1.set_title( "Mi_QSM_w/Mi_CFD_w=(%2.2f, %2.2f, %2.2f)" % (norm(self.Mx_QSM_w)/norm(self.Mx_CFD_w), 
                                                                            norm(self.My_QSM_w)/norm(self.My_CFD_w), 
                                                                            norm(self.Mz_QSM_w)/norm(self.Mz_CFD_w)) ) 
                ax1.legend()

                #optimized aerodynamic power
                ax2.plot(self.timeline, self.P_QSM_nonoptimized, label='P_QSM (non-optimized)', c='purple')
                ax2.plot(self.timeline, self.P_QSM, label='P_QSM (optimized)', color='b')
                ax2.plot(self.timeline, self.P_CFD, label='P_CFD', ls='-.', color='indigo')
                ax2.set_xlabel('$t/T$')
                ax2.set_ylabel('aerodynamic power')
                ax2.set_title("P_QSM/P_CFD=%2.2f K_power=%3.3f" % (norm(self.P_QSM)/norm(self.P_CFD), K_power) )
                ax2.legend()
                plt.tight_layout()
                plt.draw()
                
                insect_tools.indicate_strokes(ax=ax1)
                insect_tools.indicate_strokes(ax=ax2)
                
            return K_power

        # power optimization
        if optimize:            
            x0_power = [1.0, 1.0]
            bounds = [(-6, 6), (-6, 6)]

            start = time.time()
            optimization = opt.minimize(cost_power, args=(self, False), bounds=bounds, x0=x0_power)
            self.x0_power = optimization.x
            self.K_power = optimization.fun
            print('Completed in:', round(time.time() - start, 4), 'seconds')
            
            print('x0_power:', np.round(self.x0_power, 5), '\nK_power_optimized:', self.K_power)
            
        cost_power(self.x0_power, self, show_plots=True)
            

    def setup_wing_shape(self, wingShape_file, nb=1000):
        """
        Specifiy the wing shape (here, in the form of the wing contour).
        Note the code can run without this information, as the influence of
        the wing contour can also be taken into account by the optimized model
        coefficients (optimized using a reference CFD run).
        
        Shape data is read from an INI file.
        """
        
        print('Parsing wing contour: '+wingShape_file)
        
        if os.path.isfile(wingShape_file):
            xc, yc, area = insect_tools.wing_contour_from_file( wingShape_file )
        else:
            # if the shape is unknown, setup_wing_shape returns a pseudo-wing shape contour
            # note for computing the QSM, a shape model is in fact optional.
            xc, yc, area = np.asarray([0.0, 0.5, 0.0, -0.5, 0.0]), np.asarray([0.0, 0.5, 1.0, 0.5, 0.0]), 1.0
            
        zc = np.zeros_like(xc)
        
        self.wingPoints  = np.vstack([xc, yc, zc])
        self.wingPoints  = np.transpose(self.wingPoints)
        
        self.bodyPointsSequence   = np.zeros((self.nt, self.wingPoints.shape[0], 3))
        
        wingtip_index = np.argmax( self.wingPoints[:, 1] )
        
        # this function calculates the chord length by splitting into 2 segments (LE and TE segment) 
        # and then interpolating along the yw-axis 
        # It returns the chord length c as a function of yw (equivalent to r)
        def getChordLength(wingPoints, y_coordinate, wingtip_index):
            
            # get the division in wing segments (leading and trailing)
            split_index = wingtip_index
            righthand_section = wingPoints[:split_index]
            lefthand_section  = wingPoints[split_index:]

            #interpolate righthand section 
            righthand_section_interpolation = interp1d(righthand_section[:, 1], righthand_section[:, 0], fill_value='extrapolate')

            #interpolate lefthand section
            lefthand_section_interpolation = interp1d(lefthand_section[:, 1], lefthand_section[:, 0], fill_value='extrapolate') 
            
            #generate the chord as a function of y coordinate
            chord_length = abs(righthand_section_interpolation(y_coordinate) - lefthand_section_interpolation(y_coordinate))
            return chord_length
        
        
        
        min_y = np.min(self.wingPoints[:, 1])
        max_y = np.max(self.wingPoints[:, 1])
        
        # chord calculation 
        y_space = np.linspace(min_y, max_y, nb)
        c = getChordLength(self.wingPoints, y_space, wingtip_index)

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
        
        from scipy.integrate import simpson
        
        self.Iam  = simpson(C2(y_space), y_space)
        self.Iwe  = simpson(C3r3(y_space), y_space)
        self.Ild  = simpson(Cr2(y_space), y_space) #second moment of area for lift/drag calculations
        self.Irot = simpson(C2r(y_space), y_space) #second moment of area for rotational force calculation 
    
        self.hinge_index   = np.argmin( self.wingPoints[:, 1] )
        self.wingtip_index = np.argmax( self.wingPoints[:, 1] )
        
        # for kleemeier wing, integration did not work properly...
        if np.isnan(self.Iam): self.Iam = 1.0
        if np.isnan(self.Iwe): self.Iwe = 1.0
        if np.isnan(self.Ild): self.Ild = 1.0
        if np.isnan(self.Irot): self.Irot = 1.0
           
    
    def update_kinematics(self):
        return
    
    def eval_trainedQSM_given_kinematics(self):
        return
    
