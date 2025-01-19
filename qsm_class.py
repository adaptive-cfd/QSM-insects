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
import scipy
import time
import scipy.optimize as opt
import matplotlib.pyplot as plt
from insect_tools import Rx, Ry, Rz, vct

# use latex
plt.rcParams["text.usetex"] = True

deg2rad = np.pi/180.0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TODO:
#   - model for two wings
#   - body force model
#   - inertial power 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def vect(array):
    # return a column vector
    return array.reshape( array.size, 1)

def cross(a, b):
    # simplify the messy syntax a bit
    return np.cross( a.reshape(1,3), b.reshape(1,3)).reshape((3,1))

def norm(x):
    return( np.linalg.norm(x.flatten()) )

def normalize_vector( vector ):
    mag = norm( vector.flatten() )
    if mag >= 1e-16:
        vector /= mag
        
    return vector


def diff1(x, dt):
    # differentiate vector x with respect to time
    # use non-periodic, second order finite differences for that
    # note current version of qsm_class does require one-sided FD stencils
    # because concatenating many CFD runs does result in non-periodic data.
    import finite_differences
    
    D = finite_differences.D12(x.shape[0], dt)
    
    return np.matmul(D, x).reshape(x.shape)

def diff2(x, dt):
    # differentiate vector x with respect to time
    # use non-periodic, second order finite differences for that
    # note current version of qsm_class does require one-sided FD stencils
    # because concatenating many CFD runs does result in non-periodic data.
    import finite_differences
    
    D = finite_differences.D22(x.shape[0], dt)
    
    return np.matmul(D, x).reshape(x.shape)

class QSM:
    def __init__(self, nt=300, nruns=1, timeline=None, model_CL_CD='Dickinson'):
        self.nt = nt        
        self.model_CL_CD = model_CL_CD
        self.nruns = nruns
        
        # as a means of informing the user that they need to read CFD data before fitting (training)
        self.readAtLeastOneCFDrun = False
        
        if timeline is None:
            self.timeline = np.linspace(0, 1, nt, endpoint=False)
        else:
            self.timeline = timeline
        self.delta_t = self.timeline[1] - self.timeline[0]    
        
        self.alphas_dt = np.zeros((nt*nruns))
        self.phis_dt   = np.zeros((nt*nruns))
        self.thetas_dt = np.zeros((nt*nruns))
        
        self.alphas_dtdt = np.zeros((nt*nruns))
        self.phis_dtdt   = np.zeros((nt*nruns))
        self.thetas_dtdt = np.zeros((nt*nruns))
               
        self.rots_wing_b = np.zeros((nt*nruns,3))
        self.rots_wing_w = np.zeros((nt*nruns,3))
        self.rots_wing_g = np.zeros((nt*nruns,3))
        
        self.planar_rots_wing_w = np.zeros((nt*nruns,3))
        self.planar_rots_wing_g = np.zeros((nt*nruns,3))
        self.planar_rots_wing_mag = np.zeros((nt*nruns))
                
        self.u_tip_w = np.zeros((nt*nruns,3))
        self.u_tip_g = np.zeros((nt*nruns,3))
        self.u_tip_mag = np.zeros((nt*nruns))
        
        self.a_tip_w = np.zeros((nt*nruns,3))
        self.a_tip_g = np.zeros((nt*nruns,3))
        
        self.rot_acc_wing_g = np.zeros((nt*nruns,3))
        self.rot_acc_wing_w = np.zeros((nt*nruns,3))
                
        self.AoA = np.zeros((nt*nruns))
        self.e_drag_g = np.zeros((nt*nruns,3))
        self.e_lift_g = np.zeros((nt*nruns,3))        
        
        self.ey_wing_g = np.zeros((nt*nruns,3))
        self.ez_wing_g = np.zeros((nt*nruns,3))
        self.ex_wing_g = np.zeros((nt*nruns,3))                
                
        self.M_wing = np.zeros((nt*nruns, 3, 3))
        self.M_wing_inv = np.zeros((nt*nruns, 3, 3))
        self.M_stroke = np.zeros((nt*nruns, 3, 3))
        self.M_stroke_inv = np.zeros((nt*nruns, 3, 3))
        self.M_body = np.zeros((nt*nruns, 3, 3))
        self.M_body_inv = np.zeros((nt*nruns, 3, 3))
        
        self.M_g2w = np.zeros((nt*nruns, 3, 3))
        self.M_w2g = np.zeros((nt*nruns, 3, 3))
        self.M_b2w = np.zeros((nt*nruns, 3, 3))
        self.M_w2b = np.zeros((nt*nruns, 3, 3))

        self.Ftc = np.zeros((nt*nruns,3))
        self.Ftd = np.zeros((nt*nruns,3))
        self.Frc = np.zeros((nt*nruns,3))
        self.Fam = np.zeros((nt*nruns,3))
        self.Fam2 = np.zeros((nt*nruns,3))
        self.Frd = np.zeros((nt*nruns,3))
        
        self.u_infty_w = np.zeros((nt*nruns,3))
        self.u_infty_g = np.zeros((nt*nruns,3))
        
        self.F_QSM_w = np.zeros((nt*nruns,3))
        self.F_QSM_g = np.zeros((nt*nruns,3))     
        self.M_QSM_w = np.zeros((nt*nruns,3))
        self.M_QSM_g = np.zeros((nt*nruns,3))     
           
        self.Ftc_mag = np.zeros((nt*nruns))
        self.Ftd_mag = np.zeros((nt*nruns))
        self.Frc_mag = np.zeros((nt*nruns))
        self.Fam_z_mag = np.zeros((nt*nruns))
        self.Fam_x_mag = np.zeros((nt*nruns))
        self.Frd_mag = np.zeros((nt*nruns))  
              
        self.F_CFD_g = np.zeros((nt*nruns,3))
        self.F_CFD_w = np.zeros((nt*nruns,3))   
        self.M_CFD_g = np.zeros((nt*nruns,3))
        self.M_CFD_w = np.zeros((nt*nruns,3))           
        self.P_CFD = np.zeros((nt*nruns))
        
        # wing-geometry-dependent constants
        # they are computed (if desired) in setup_wing_shape
        self.Iam  = 1.0
        self.Iwe  = 1.0
        self.Ild  = 1.0
        self.Irot = 1.0
        
        self.wing = 'left'
        self.x_wingContour_w = np.zeros((10,3))
        
        self.x0_forces = np.zeros((15))
        self.x0_moments = np.zeros((2))
        self.x0_power = np.zeros((2))
        
        self.psis = np.zeros((nt*nruns))
        self.betas = np.zeros((nt*nruns))
        self.gammas = np.zeros((nt*nruns))
        self.etas = np.zeros((nt*nruns))
        self.alphas = np.zeros((nt*nruns))
        self.phis = np.zeros((nt*nruns))
        self.thetas = np.zeros((nt*nruns))
        
    def parse_many_run_directorys(self, run_directories, params_file, kinematics_file, T0=1.0, wing='right', plot=False):
        """
        This function is for reading in a bunch of CFD runs in order to fit a single
        QSM model to many runs. It parses the kinematics data (either kinematics.t or an *.ini
        file), and reads in the CFD data for the forces, moments, power.
        
        TODO:
        Same INI file all runs? 
        Same cycle all runs? 
        Same wing ?

        """
        
        # check if right amount of memory is allocated
        if not self.psis.shape[0] == self.nt*len(run_directories):
            raise ValueError("We try to process %i runs, but allocated not enough storage. Use QSM = qsm_class.QSM(nt=XXX, nruns=YYY)")
        
        i0 = 0
        for run in run_directories:
            run += '/'
            self.parse_kinematics( params_file=run+params_file, kinematics_file=run+kinematics_file, wing=wing, i0=i0, plot=plot  )
            self.read_CFD_data(run, run+params_file, T0, i0)
            
            i0 += self.nt
            
            
    def kinematics_3D_frame(self, it, ax):
        """
        Plot a single frame (at time step it) into the axis ax.
        The plots the wing (with two surfaces), tip trajectories and shadows in 
        the projected areas of the coordinate system.
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        
        sign = 1.0
        if self.wing == "right" or self.wing == "right2":
            sign = -1.0
            
        # get point set by timeStep number
        x_wingContour_w = self.x_wingContour_w
        x_wingContour_w[:,2] = sign*0.001
        points_top = np.transpose( np.matmul(self.M_g2w[it,:,:].T, x_wingContour_w.T) )
        
        x_wingContour_w = self.x_wingContour_w
        x_wingContour_w[:,2] = -0.001*sign
        points_bot = np.transpose( np.matmul(self.M_g2w[it,:,:].T, x_wingContour_w.T) )
        
        X = points_top[:, 0]
        Y = points_top[:, 1]
        Z = points_top[:, 2]

        #axis limit
        a = 2

        #3D trajectory 
        ax.plot(self.ey_wing_g[:it, 0], self.ey_wing_g[:it, 1], self.ey_wing_g[:it, 2], color='k', linestyle='dashed', linewidth=0.5)

        #x-y plane trajectory 
        ax.plot(self.ey_wing_g[:it, 0], self.ey_wing_g[:it, 1], 0*self.ey_wing_g[:it, 2]-a, color='k', linestyle='dashed', linewidth=0.5)
        
        #z-y plane prajectory 
        ax.plot(0*self.ey_wing_g[:it, 0]-a, self.ey_wing_g[:it, 1], self.ey_wing_g[:it, 2], color='k', linestyle='dashed', linewidth=0.5)

        #x-z plane prajectory 
        ax.plot(self.ey_wing_g[:it, 0], 0*self.ey_wing_g[:it, 1]+a, self.ey_wing_g[:it, 2], color='k', linestyle='dashed', linewidth=0.5)
        
        # draw the wing            
        ax.add_collection3d(Poly3DCollection(verts=[points_top], color='r', edgecolor='k'))
        ax.add_collection3d(Poly3DCollection(verts=[points_bot], color='lightsalmon', edgecolor='k'))

        #shadows
        #x-y plane shadow
        XY_plane_shadow = np.vstack((X, Y, -a*np.ones_like(Z))).transpose()
        ax.add_collection3d(Poly3DCollection(verts=[XY_plane_shadow], color='#d3d3d3'))
        
        #y-z plane shadow
        YZ_plane_shadow = np.vstack((-a*np.ones_like(X), Y, Z)).transpose()
        ax.add_collection3d( Poly3DCollection(verts=[YZ_plane_shadow], color='#d3d3d3') )
        
        #x-z plane shadow
        XZ_plane_shadow = np.vstack(((X, a*np.ones_like(Y), Z))).transpose()
        ax.add_collection3d(Poly3DCollection(verts=[XZ_plane_shadow], color='#d3d3d3'))
        
        #set the axis limits
        ax.set_xlim([-a, a])
        ax.set_ylim([-a, a])
        ax.set_zlim([-a, a])

        #set the axis labels 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
             
            
    def kinematics_3D_animation(self, fnames_out="visualization3D"):
        """
        Creates one PNG file per time step, loops over the entire time vector
        """
        
        # can be called only after parse_kinematics
        plt.figure()
        ax = plt.figure().add_subplot(projection='3d')
                        
        for it in range(self.nt):
            self.kinematics_3D_frame(it, ax)
            plt.show()
            fname_out = fnames_out+'.%04i.png' % (it)
            print('Animation 3D visualization saving: ' + fname_out)
            plt.savefig( fname_out )
        
    def parse_kinematics(self, params_file, kinematics_file=None, u_infty_g=None, plot=True, wing='auto', yawpitchroll0=None, eta_stroke=None, i0=0,
                         alpha=None, phi=None, theta=None, eta=None, psi=None, beta=None, gamma=None, verbose=True):
        """
        Evaluate the kinematics given by kinematics_file and store the results
        in the class arrays. You must run this before you can fit the QSM to the CFD data.
        
        There are 3 ways to read kinematics:
            1/ Reading kinematics.t from an existing CFD simulation (i.e. a run that has been simulated)
            
            2/ Reading an INI file for wingbeat kinematics (the input for CFD runs, but not necessarily a run that has been simulated)
            
            3/ Passing externally generated kinematics: alpha, phi, theta, eta, psi, beta, gamma as vectors sampled on equidistant time vector. 
            Use QSM.timeline as time vector.
            Do not pass kinematics_file in this case (=None). The angles you pass shall be in DEGREE.
        
        Parameters
        ----------
        params_file : string
            Parameter file (INI) of the simulation. Used to determine which wing we model, flight velocity etc.
        kinematics_file : string
            The kinematics file can be either: 
                - "kinematics.t", the output log file of a CFD run
                - an *.ini file, which is the CFD-code's descriptor file for kinematics
        u_infty_g : vector, optional
            Flight velocity of the animal. Default: read from params_file, overwritten if you pass a value explicitly. 
            Note how the params_file contains the wind velocity (wind tunnel speed), and we invert its sign to get the 
            animals flight speed. If you pass the parameter directly, give flight speed not wind tunnel speed.
        plot : bool, optional
            Plot raw kinematics or not. The default is True.
        wing : TYPE, optional
            DESCRIPTION. The default is 'auto'.
        yawpitchroll0 : TYPE, optional
            If you pass kinematics.t as kinematics_file, yaw pitch roll are determined from there, but you 
            can overwrite it with the (constant) value you pass. Useful for optimization problems. 
        eta_stroke : TYPE, optional
            You can omit this parameter if you read the kinematics from a CFD run *that is already done*
            (which I guess is the usual case). The log-file kinematics.t contains the eta_stroke as well
            as body posture angles.

        """
        nt = self.nt
        # index range to store the data to
        ii = np.arange(start=i0, stop=i0+nt-1+1 )
                        
        #generate rot wing calculates the angular velocity of the wing in all reference frames, as well as the planar angular velocity {𝛀(φ,Θ)} 
        #which will later be used to calculate the forces on the wing.  planar angular velocity {𝛀(φ,Θ)} comes from the decomposition of the motion
        #into 'translational' and rotational components, with the rotational component beig defined as ⍺ (the one around the y-axis in our convention)
        #this velocity is obtained by setting ⍺ to 0, as can be seen below
        def generate_rot_wing(M_wing, M_body_inv, M_stroke_inv, phi, phi_dt, alpha, alpha_dt, theta, theta_dt, wing): 
            if wing != "left" and wing != "right" and wing != "left2" and wing != "right2":
                raise ValueError("Wing not known (I know of left/right/left2/right2)")
            
            if wing == "right" or wing == "right2":
                phi = -phi
                phi_dt = -phi_dt
                alpha = -alpha
                alpha_dt = -alpha_dt
                
            # M_wing : M_s2w
            # M_body_inv : M_b2g
            # M_stroke_inv : M_s2b
            rot_wing_s = vct([ phi_dt-np.sin(theta)*alpha_dt, \
                           np.cos(phi)*np.cos(theta)*alpha_dt-np.sin(phi)*theta_dt, \
                           np.sin(phi)*np.cos(theta)*alpha_dt+np.cos(phi)*theta_dt])
            rot_wing_w = np.matmul(M_wing, rot_wing_s)
            rot_wing_b = np.matmul(M_stroke_inv, rot_wing_s)
            rot_wing_g = np.matmul(M_body_inv, rot_wing_b)
            
            # 'new' definition
            planar_rot_wing_w = rot_wing_w.copy()
            planar_rot_wing_w[1] = 0.0 # set y-component to zero
            
            planar_rot_wing_s = np.matmul( np.transpose(M_wing), planar_rot_wing_w ) # now in stroke frame
            planar_rot_wing_b = np.matmul( M_stroke_inv, planar_rot_wing_s ) # now in body frame
            planar_rot_wing_g = np.matmul( M_body_inv, planar_rot_wing_b ) # global frame
            
            return rot_wing_g, rot_wing_b, rot_wing_w, planar_rot_wing_g, planar_rot_wing_w #these are all (3x1) vectors 
        
       
        # since the absolute linear velocity of the wing depends both on time and on the position along the wing
        # this function calculates only its position dependency
        def generate_u_wing_g_position(rot_wing_g, ey_wing_g):
            # omega x point
            u_wing_g_position = cross(rot_wing_g, ey_wing_g)
            return u_wing_g_position

        #this function calculates the linear velocity of the wing in the wing reference frame
        def generate_u_wing_w(u_wing_g, M_body, M_stroke, M_wing):
            u_wing_w = np.matmul(M_wing, np.matmul(M_stroke, np.matmul(M_body, u_wing_g)))
            return u_wing_w

        #in this code AoA is defined as the arccos of the dot product between the unit vector along x direction and the unit vector of the absolute linear velocity
        def getAoA(ex_wing_g, e_u_wing_g):
            AoA = np.arccos(np.dot(ex_wing_g.flatten(), e_u_wing_g.flatten()))
            return AoA
        
        # left or right wing?
        if wing == 'auto':
            # read from PARAMS-file; this is the default. If we use QSM on a two (or four) winged simulation,
            # we create one QSM model for each wing.
            isLeft   = wt.get_ini_parameter(params_file, 'Insects', 'LeftWing', dtype=bool)
            isRight  = wt.get_ini_parameter(params_file, 'Insects', 'RightWing', dtype=bool)
            
            if isLeft:
                wing = "left"
            if isRight:
                wing = "right"
                
            if isLeft and isRight:
                raise ValueError("This simulation included more than one wing, you need to create one QSM model per wing, pass wing=right and wing=left")
                
        elif wing != 'auto' and wing != 'left' and wing != 'right' and wing != 'left2' and wing != 'right2':
            raise ValueError("Invalid choice for wing (auto/left/right/left2/right2)")
            
                
        self.wing = wing
            
        if kinematics_file == None:            
            for angle in [alpha, phi, theta, eta, psi, beta, gamma]:
                if angle.shape[0] != self.nt:
                    raise ValueError("The kinematics angles you pass to this routine need to have the same length as nt=QSM.timeline.shape[0]")
            
            self.psis[ii]   = psi*deg2rad
            self.betas[ii]  = beta*deg2rad
            self.gammas[ii] = gamma*deg2rad
            self.etas[ii]   = eta*deg2rad
            self.alphas[ii] = alpha*deg2rad
            self.phis[ii]   = phi*deg2rad
            self.thetas[ii] = theta*deg2rad
            
        else:            
            if "kinematics.t" in kinematics_file:            
                # load kinematics data by means of load_t_file function from insect_tools library 
                kinematics_cfd = insect_tools.load_t_file(kinematics_file, interp=True, time_out=self.timeline)
                
                self.psis[ii]   = kinematics_cfd[:, 4].copy()
                self.betas[ii]  = kinematics_cfd[:, 5].copy()
                self.gammas[ii] = kinematics_cfd[:, 6].copy()
                self.etas[ii]   = kinematics_cfd[:, 7].copy()
                
                if self.wing == "right": 
                    # right wing
                    self.alphas[ii] = kinematics_cfd[:,11].copy()
                    self.phis[ii]   = kinematics_cfd[:,12].copy()
                    self.thetas[ii] = kinematics_cfd[:,13].copy()
                    
                    # # As WABBIT also computes as stores angular velocity & acceleration
                    # # we can use this to verify the QSM computation done here
                    # self.debug_rotx_wing_g = kinematics_cfd[:, 17]
                    # self.debug_roty_wing_g = kinematics_cfd[:, 18]
                    # self.debug_rotz_wing_g = kinematics_cfd[:, 19]                
                    # self.debug_time = kinematics_cfd[:,0]                
                    # self.debug_rotx_dt_wing_g = kinematics_cfd[:, 23]
                    # self.debug_roty_dt_wing_g = kinematics_cfd[:, 24]
                    # self.debug_rotz_dt_wing_g = kinematics_cfd[:, 25]      
                elif self.wing == "left":
                    # left wing
                    self.alphas[ii] = kinematics_cfd[:,  8].copy()
                    self.phis[ii]   = kinematics_cfd[:,  9].copy()
                    self.thetas[ii] = kinematics_cfd[:, 10].copy()             
                    
                    # # As WABBIT also computes as stores angular velocity & acceleration
                    # # we can use this to verify the QSM computation done here
                    # self.debug_rotx_wing_g = kinematics_cfd[:, 14] # is this really _g ??
                    # self.debug_roty_wing_g = kinematics_cfd[:, 15]
                    # self.debug_rotz_wing_g = kinematics_cfd[:, 16]                
                    # self.debug_time = kinematics_cfd[:,0]                
                    # self.debug_rotx_dt_wing_g = kinematics_cfd[:, 20]
                    # self.debug_roty_dt_wing_g = kinematics_cfd[:, 21]
                    # self.debug_rotz_dt_wing_g = kinematics_cfd[:, 22]   
                    
                elif self.wing == "right2":
                    # right (hind) wing, second wing pair
                    self.alphas[ii] = kinematics_cfd[:, 29].copy()
                    self.phis[ii]   = kinematics_cfd[:, 30].copy()
                    self.thetas[ii] = kinematics_cfd[:, 31].copy()        
                    
                elif self.wing == "left2":
                    # left (hind) wing, second wing pair
                    self.alphas[ii] = kinematics_cfd[:, 26].copy()
                    self.phis[ii]   = kinematics_cfd[:, 27].copy()
                    self.thetas[ii] = kinematics_cfd[:, 28].copy()       
                    
                else:
                    raise ValueError("Wing code unknown (left/right/left2/right2)")
      
            elif ".ini" in kinematics_file:
                # load kinematics data by means of eval_angles_kinematics_file function from insect_tools library 
                timeline, phis, alphas, thetas = insect_tools.eval_angles_kinematics_file(fname=kinematics_file, time=self.timeline)
                
                # convert to rad
                self.phis[ii]   = np.radians(phis)
                self.alphas[ii] = np.radians(alphas)
                self.thetas[ii] = np.radians(thetas)            
                
                if yawpitchroll0 is None:
                    yawpitchroll0 = wt.get_ini_parameter(params_file, 'Insects', 'yawpitchroll_0', vector=True)
                if eta_stroke is None:
                    eta_stroke    = wt.get_ini_parameter(params_file, 'Insects', 'eta0', dtype=float )
                
                # yaw pitch roll and stroke plane angle are all constant in time
                self.psis[ii]   = np.zeros_like(self.phis) + yawpitchroll0[2]*(np.pi/180.0)
                self.betas[ii]  = np.zeros_like(self.phis) + yawpitchroll0[1]*(np.pi/180.0)
                self.gammas[ii] = np.zeros_like(self.phis) + yawpitchroll0[0]*(np.pi/180.0)
                self.etas[ii]   = np.zeros_like(self.phis) + eta_stroke*(np.pi/180.0)
                
            else:
                raise ValueError("Parsing kinematics is possible with kinematics.t, kinematics.ini or manually passing the angles.")
        
        dt, nt = self.delta_t, self.nt
        
        self.alphas_dt[ii] = diff1( self.alphas[ii], dt )
        self.phis_dt[ii]   = diff1( self.phis[ii], dt )
        self.thetas_dt[ii] = diff1( self.thetas[ii], dt )
        
        self.alphas_dtdt[ii] = diff2( self.alphas[ii], dt )
        self.phis_dtdt[ii]   = diff2( self.phis[ii], dt )
        self.thetas_dtdt[ii] = diff2( self.thetas[ii], dt )
 
        
        # MAIN loop over time steps
        for it in ii:            
            # insects cruising speed        
            if u_infty_g is None:
                self.u_infty_g[it, :] = -np.asarray( wt.get_ini_parameter(params_file, 'ACM-new', 'u_mean_set', vector=True) ) # note sign change (wind vs body)
            else:
                self.u_infty_g[it, :] = u_infty_g

            if verbose and it == ii[0]:
                print("u_infty_g = [%f, %f, %f]" % (self.u_infty_g[it, 0],self.u_infty_g[it, 1],self.u_infty_g[it, 2]))

            alpha, phi, theta, eta, psi, beta, gamma = self.alphas[it], self.phis[it], self.thetas[it], self.etas[it], self.psis[it], self.betas[it], self.gammas[it]
            alpha_dt, phi_dt, theta_dt = self.alphas_dt[it], self.phis_dt[it], self.thetas_dt[it]
            
            # define the many rotation matrices used in the code.
            # The nomenclature M_wing, M_stroke and M_body is used in WABBIT/FLUSI insect_module.
            # The (newer) nomenclature M_s2w, M_b2s, M_g2b is easier to understand.
            M_wing   = insect_tools.M_wing(alpha, theta, phi, wing=self.wing.replace("2","") ) # M_s2w
            M_stroke = insect_tools.M_stroke(eta, self.wing.replace("2","")) # M_b2s
            M_body   = insect_tools.M_body(psi, beta, gamma)# M_g2b
            M_g2w    = M_wing*M_stroke*M_body
            M_w2g    = np.transpose(M_g2w)
            M_b2w    = M_wing*M_stroke
            M_w2b    = np.transpose(M_b2w)
            
            self.M_wing[it, :]        = M_wing # M_s2w
            self.M_stroke[it, :]      = M_stroke # M_b2s
            self.M_body[it, :]        = M_body # M_g2b
            self.M_wing_inv[it, :]    = np.transpose(M_wing) # M_w2s
            self.M_stroke_inv[it, :]  = np.transpose(M_stroke) # M_s2b
            self.M_body_inv[it, :]    = np.transpose(M_body) # M_b2g
        
            # the syntax is strange but working. Is a (nt, 3, 3) matrix even though it has only two indices
            self.M_g2w[it, :] = M_g2w
            self.M_w2g[it, :] = M_w2g            
            self.M_b2w[it, :] = M_b2w
            self.M_w2b[it, :] = M_w2b
            
            # flight velocity in the wing system
            self.u_infty_w[it, :] = (M_g2w * vct(self.u_infty_g[it,:])).flatten()
        
            # these are all unit vectors of the wing 
            # ey_wing_g coincides with the tip only if R is normalized (usually the case)
            ex_wing_g = M_w2g * vct([1,0,0])
            ey_wing_g = M_w2g * vct([0,1,0])
            ez_wing_g = M_w2g * vct([0,0,1])
            
            self.ex_wing_g[it, :] = ex_wing_g.flatten()
            self.ey_wing_g[it, :] = ey_wing_g.flatten()
            self.ez_wing_g[it, :] = ez_wing_g.flatten()                
            
            # angular velocity of the wing        
            rot_wing_g, rot_wing_b, rot_wing_w, planar_rot_wing_g, planar_rot_wing_w = generate_rot_wing(
                self.M_wing[it, :], self.M_body_inv[it, :], self.M_stroke_inv[it, :], 
                phi, phi_dt, alpha, alpha_dt, theta, theta_dt, self.wing)            
        
            self.rots_wing_b[it, :] = rot_wing_b.flatten()
            self.rots_wing_w[it, :] = rot_wing_w.flatten()
            self.rots_wing_g[it, :] = rot_wing_g.flatten()
        
            self.planar_rots_wing_w[it, :] = planar_rot_wing_w.flatten()
            self.planar_rots_wing_g[it, :] = planar_rot_wing_g.flatten()        
            self.planar_rots_wing_mag[it] = norm( planar_rot_wing_w )            
        
            u_wing_g = vect( generate_u_wing_g_position(rot_wing_g, ey_wing_g).flatten() + self.u_infty_g[it, :] )
            self.u_tip_g[it, :] = u_wing_g.flatten()
            self.u_tip_w[it, :] = (M_g2w * u_wing_g).flatten()
            
            self.u_tip_mag[it] = norm(u_wing_g)        
            e_u_wing_g = normalize_vector(u_wing_g)
            
            self.e_drag_g[it :] = -e_u_wing_g.flatten()
        
            # lift unit vector. sign is determined below.
            self.e_lift_g[it, :] = normalize_vector( cross(e_u_wing_g, ey_wing_g) ).flatten()

            aoa = getAoA(ex_wing_g, e_u_wing_g) #use this one for getAoA with arccos 
            self.AoA[it] = aoa
              
        # calculation of wingtip acceleration and angular acceleration in wing reference frame 
        # a second loop over time is required, because we first need to compute ang. vel. then diff it here.
        self.a_tip_g[ii] = diff1( self.u_tip_g[ii], dt )
        self.rot_acc_wing_g[ii] = diff1( self.rots_wing_g[ii], dt )
        
        for it in ii:               
            self.a_tip_w[it, :]     = np.matmul(self.M_g2w[it, :], self.a_tip_g[it, :])
            self.rot_acc_wing_w[it, :] = np.matmul(self.M_g2w[it, :], self.rot_acc_wing_g[it, :])

        
        #----------------------------------------------------------------------
        # sign of lift vector (timing of half-cycles)
        #----------------------------------------------------------------------
        # the lift vector is until here only defined up to a sign. we decide about this sign now.
        # Many papers simply use SIGN(ALPHA) for this task, but for some kinematics we found this
        # does not work.
        # Note there is a subtlety with the sign: is it positive during up- or downstroke? This does not really
        # matter if the optimizer is used, because it can simply flip the coefficients.
        
        sign_liftvector = np.ones_like(self.e_lift_g[ii,:])
        if self.wing == "left" or self.wing == "left2" :
            # for left and right wing, the sign is inverted (hence using the array "sign", otherwise we'd just
            # flip the sign directly in e_lift_g)
            sign_liftvector *= -1.0
            
        # find minima in wingtip velocity magnitude. those, hopefully two, will be the reversals,
        # this is where the sign is flipped. We repeat the (periodic) signal to ensure we capture 
        # peaks at t=0.0 and t=1.0. The distance between peaks is 3/4 * 1/2, so we think that the two half-strokes 
        # occupy at most 3/8 and 5/8 of the complete cycle (maximum imbalance between up- and downstroke). This 
        # filters out smaller peaks (in height) automatically, so we are left with the two most likely candidates.
        ipeaks, _ = scipy.signal.find_peaks( -1*np.hstack(  3*[self.u_tip_mag[ii].flatten()] ), distance=3*self.nt/4/2)
        ipeaks -= self.nt # shift (skip 1st periodic image)
        # keep only peaks in the original signal domain (remove periodic "ghosts")
        ipeaks = ipeaks[ipeaks>=0]
        ipeaks = ipeaks[ipeaks<nt]
        
        # It should be two minima of velocity, if its not, then something weird happens in the kinematics.
        # We must then look for a different way to determine reversals or set it manually.
        if len(ipeaks) != 2 :
            plt.figure()
            plt.plot( np.hstack([self.timeline, self.timeline+1, self.timeline+2]), -1*np.hstack(  (3*[self.u_tip_mag[ii].flatten()]) ) )
            plt.xlabel('timeline (repeated identical cycle 3 times)')
            plt.ylabel('u_tip_mag (wing=%s)' % (self.wing))
            plt.plot( self.timeline[ipeaks]+0, -1*self.u_tip_mag[ipeaks], 'ro')
            plt.plot( self.timeline[ipeaks]+1, -1*self.u_tip_mag[ipeaks], 'ro')
            plt.plot( self.timeline[ipeaks]+2, -1*self.u_tip_mag[ipeaks], 'ro')
            plt.title('Wing velocity minima detection: PROBLEM (more than 2 minima found)\n Note sign inversion (search max of negative-->min)')
            
            insect_tools.indicate_strokes(tstroke=2.0)
            print(ii)
            raise ValueError("We found more than two reversals in the kinematics data...")
            
        sign_liftvector[ ipeaks[0]:ipeaks[1], : ] *= -1
        self.e_lift_g[ii] *= sign_liftvector
        # for indication in figures:
        self.T_reversals = self.timeline[ipeaks]
        
        #----------------------------------------------------------------------
        # kinematics figure
        #----------------------------------------------------------------------
        if plot:
            ## FIGURE 1
            fig, axes = plt.subplots(3, 2, figsize = (15, 15))

            # angles
            ax = axes[0,0]
            ax.plot(self.timeline, np.degrees(self.phis[ii]), label='$\\phi$')
            ax.plot(self.timeline, np.degrees(self.alphas[ii]), label ='$\\alpha$')
            ax.plot(self.timeline, np.degrees(self.thetas[ii]), label='$\\theta$')
            ax.plot(self.timeline, np.degrees(self.AoA[ii]), label='AoA', color = 'purple')
            ax.set_xlabel('$t/T$')
            ax.set_ylabel('(deg)')
            ax.legend() 
            
            # time derivatives of angles
            ax = axes[0,1]
            ax.plot(self.timeline, self.phis_dt[ii], label='$\\dot\\phi$')
            ax.plot(self.timeline, self.alphas_dt[ii], label='$\\dot\\alpha$')
            ax.plot(self.timeline, self.thetas_dt[ii], label='$\\dot\\theta$')
            
            if self.wing == "right" or self.wing == "right2" :
                ax.plot(self.timeline, np.sign(-self.alphas[ii]), 'k--', label='$\\mathrm{sign}(\\alpha)$', linewidth=0.5 )
            elif self.wing == "left" or self.wing == "left2" :
                ax.plot(self.timeline, np.sign(+self.alphas[ii]), 'k--', label='$\\mathrm{sign}(\\alpha)$', linewidth=0.5 )
            ax.set_xlabel('$t/T$')
            ax.legend()

            # u_wing_w (tip velocity in wing reference frame )
            ax = axes[1,0]
            ax.plot(self.timeline, self.u_tip_w[ii, 0], label='$u_{\\mathrm{wing},x}^{(w)}$')
            ax.plot(self.timeline, self.u_tip_w[ii, 1], label='$u_{\\mathrm{wing},y}^{(w)}$')
            ax.plot(self.timeline, self.u_tip_w[ii, 2], label='$u_{\\mathrm{wing},z}^{(w)}$')            
            ax.plot(self.timeline, self.u_tip_mag[ii], 'k--', label='$\\left| \\underline{u}_{\\mathrm{wing}} \\right|$')
            
            ax.set_xlabel('$t/T$')
            ax.set_ylabel('[Rf]')
            ax.set_title('Tip velocity in wing reference frame, $\\mathrm{mean}\\left(\\left|\\underline{u}_{\\mathrm{wing}}\\right|\\right)=%2.2f$' % (np.mean(self.u_tip_mag)))
            ax.legend()

            #a_wing_w (tip acceleration in wing reference frame )
            ax = axes[1,1]
            ax.plot(self.timeline, self.a_tip_w[ii, 0], label='$\\dot{u}_{\\mathrm{wing},x}^{(w)}$')
            ax.plot(self.timeline, self.a_tip_w[ii, 1], label='$\\dot{u}_{\\mathrm{wing},y}^{(w)}$')
            ax.plot(self.timeline, self.a_tip_w[ii, 2], label='$\\dot{u}_{\\mathrm{wing},z}^{(w)}$')
            ax.set_xlabel('$t/T$')
            ax.set_ylabel('$Rf^2$')
            ax.set_title('Tip acceleration in wing reference frame')
            ax.legend()

            #rot_wing_w (tip velocity in wing reference frame )
            ax = axes[2,0]
            ax.plot(self.timeline, self.rots_wing_w[ii, 0], label='$\\Omega_{\\mathrm{wing},x}^{(w)}$')
            ax.plot(self.timeline, self.rots_wing_w[ii, 1], label='$\\Omega_{\\mathrm{wing},y}^{(w)}$')
            ax.plot(self.timeline, self.rots_wing_w[ii, 2], label='$\\Omega_{\\mathrm{wing},z}^{(w)}$')
            ax.set_xlabel('$t/T$')
            ax.set_ylabel('rad/T')
            ax.set_title('Angular velocity in wing reference frame')
            ax.legend()

            #rot_acc_wing_w (angular acceleration in wing reference frame )
            ax = axes[2,1]
            ax.plot(self.timeline, self.rot_acc_wing_w[ii, 0], label='$\\dot\\Omega_{\\mathrm{wing},x}^{(w)}$')
            ax.plot(self.timeline, self.rot_acc_wing_w[ii, 1], label='$\\dot\\Omega_{\\mathrm{wing},y}^{(w)}$')
            ax.plot(self.timeline, self.rot_acc_wing_w[ii, 2], label='$\\dot\\Omega_{\\mathrm{wing},z}^{(w)}$')
            
            ax.plot(self.timeline, np.sqrt(self.rot_acc_wing_w[ii, 0]**2+self.rot_acc_wing_w[ii, 1]**2+self.rot_acc_wing_w[ii, 2]**2), 'k--', label='mag')
            ax.set_xlabel('$t/T$')
            ax.set_ylabel('[rad/T²]')
            ax.set_title('Angular acceleration in wing reference frame')
            ax.legend()

            plt.suptitle('Kinematics data')

            plt.tight_layout()
            plt.draw()
            
            for ax in axes.flatten():
                insect_tools.indicate_strokes(ax=ax, tstart=[self.T_reversals[0]], tstroke=2*(self.T_reversals[1]-self.T_reversals[0]) )

    def print_force_coeefs(self):

            x0 = self.x0_forces
            
            # Cl and Cd definitions from Dickinson 1999
            print("CL = %f + %f*sin( deg2rad*(2.13*AoA - 7.20) ) " % (x0[0],x0[1]) )
            print("CD = %f + %f*cos( deg2rad*(2.04*AoA - 9.82) ) " % (x0[2],x0[3]) )
            print("Cam1=%f Cam2=%f Cam3=%f (linear accel, normal force) \nCam4=%f Cam5=%f Cam6=%f (angular accel, normal force)\nCam7=%f Cam8=%f (ax, az: tangential force)" % (x0[5], x0[6], x0[8], x0[9], x0[10], x0[11],x0[12],x0[13]) )
            print("C_rot=%f C_rd=%f" % (x0[4], x0[7]))


    def read_CFD_data(self, run_directory, params_file, T0, i0=0 ):
        """
        Read in CFD data (forces, moments, power) for a single run. Starting point in 
        time is T0, we read data evenly sampled in time until T0+1.0.
        If multiple runs are used for fitting, the data are concatenated, in which case the i0 parameter is used.
        The data are then stored as [i0:i0+nt] (python notation, so actually [i0:i0+nt-1])
        """
        nt = self.nt
        # index range to store the data to
        ii = np.arange(start=i0, stop=i0+nt-1+1 )
        
        # as a means of informing the user that they need to read CFD data before fitting (training)
        self.readAtLeastOneCFDrun = True
        
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('The parsed data correspond to the %s wing.' % (self.wing))

        # to create the filenames to read      
        wing_pair = ""
        if "2" in self.wing:
            wing_pair = "2"
            
        # valid: leftwing, rightwing, leftwing2, rightwing2
        suffix = self.wing.replace("2","") + 'wing' + wing_pair
        # use np.loadtxt just this one time to figure out the bounds because it is faster
        d = np.loadtxt(run_directory+'/forces_'+suffix+'.t')
        time_start, time_end = d[0,0], d[-1,0]
        print("CFD data       t=[%f, %f]" % (time_start, time_end))
        print("QSM model uses t=[%f, %f]" % (T0, T0+1.0))
        
        # read in data from desired cycle (hence the shift by T0)
        # NOTE that load_t_file can remove outliers in the data, which turned out very useful for the 
        # musca domestica data (which have large jumps exactly at the reversals)
        forces_CFD  = insect_tools.load_t_file(run_directory+'/forces_'+suffix+'.t', interp=True, time_out=self.timeline+T0, remove_outliers=True)
        moments_CFD = insect_tools.load_t_file(run_directory+'/moments_'+suffix+'.t', interp=True, time_out=self.timeline+T0, remove_outliers=True)
            
        self.F_CFD_g[ii,0:2+1] = forces_CFD[:, 1:3+1]
        self.M_CFD_g[ii,0:2+1] = moments_CFD[:, 1:3+1]
        
        # obtain CFD data in wing reference frame
        for i in ii:
            self.M_CFD_w[i, :] = np.matmul(self.M_g2w[i, :, :], self.M_CFD_g[i, :])
            self.F_CFD_w[i, :] = np.matmul(self.M_g2w[i, :, :], self.F_CFD_g[i, :])
        
        # aerodynamic power. Can be read from t-file or computed from the dot-product of moment and angular 
        # velocity. This latter is done here: we do not store the power for each wing separately in the CFD run, and hence
        # computing it here is the better choice if possibly more than one wing has been simulated.·
        self.P_CFD[ii] = -(  np.sum( self.M_CFD_w[ii,:]*self.rots_wing_w[ii,:], axis=1 ) )
        
    def evaluate_QSM_model(self, plot=True, model_terms=[True, True, True, True, True]):
        """
        This function is a wrapper for FIT_TO_CFD. It evaluates the QSM model with a given set of 
        previously determined coefficients. You need to parse kinematics data before you can call this
        function. 
        """       
        
        self.fit_to_CFD(plot=plot, optimize=False, N_trials=1, model_terms=model_terms)
            
    def fit_to_CFD(self, optimize=True, plot=True, N_trials=3, model_terms=[True, True, True, True, True], verbose=True):
        """
        Train the QSM model with one/many CFD run. This works only if you have initialized 
            * the kinematics with parse_kinematics_file
            * read the CFD data with read_CFD_data 
            * the wing shape with setup_wing_shape
        before calling this routine. 
        
        We train the QSM model to a single wing currently. 
        
        The optimized coefficients are stored in the QSM object.
        
        model_terms: [use_ellington_liftdrag, use_sane_rotforce, use_rotationalDrag, use_addedmass_normal, use_addedmass_tangential]
        
        optimize: if True, we train the QSM model with previously read CFD data.
                  if False, this function evaluates the model with the parsed kinematics, with given QSM model coefficients
        """
        
        # special time vector used for plotting if many runs are present
        times = []
        for irun in range(self.nruns):
            times.append(self.timeline+irun)
        tt = np.hstack(times)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # remove nans and infs, and hope for the best
        self.F_CFD_g[np.isinf(self.F_CFD_g)] = 0
        self.F_CFD_g[np.isnan(self.F_CFD_g)] = 0
        self.F_CFD_w[np.isinf(self.F_CFD_w)] = 0
        self.F_CFD_w[np.isnan(self.F_CFD_w)] = 0
        self.M_CFD_g[np.isinf(self.M_CFD_g)] = 0
        self.M_CFD_g[np.isnan(self.M_CFD_g)] = 0
        self.M_CFD_w[np.isinf(self.M_CFD_w)] = 0
        self.M_CFD_w[np.isnan(self.M_CFD_w)] = 0        
        self.P_CFD[np.isinf(self.P_CFD)] = 0
        self.P_CFD[np.isnan(self.P_CFD)] = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        def getAerodynamicCoefficients(self, x0, AoA): 
            deg2rad = np.pi/180.0 
            rad2deg = 180.0/np.pi                        
            
            if self.model_CL_CD == "Dickinson":
                # Cl and Cd definitions from Dickinson 1999
                AoA = rad2deg*AoA
                Cl   = x0[0] + x0[1]*np.sin( deg2rad*(2.13*AoA - 7.20) )
                Cd   = x0[2] + x0[3]*np.cos( deg2rad*(2.04*AoA - 9.82) )
            elif self.model_CL_CD == "Nakata":
                # this is what nakata proposed:
                Cl   = x0[0]*(AoA**3 - AoA**2 * np.pi/2) + x0[1]*(AoA**2 - AoA * np.pi/2)
                Cd   = x0[2]*np.cos( AoA )**2  + x0[3]*np.sin( AoA )**2
            else:
                raise ValueError("The CL/CD model must be either Dickinson or Nakata")
            
            Crot = x0[4]
            Cam1 = x0[5]
            Cam2 = x0[6]
            Crd  = x0[7]
            Cam3 = x0[8]
            Cam4 = x0[9]
            Cam5 = x0[10]
            Cam6 = x0[11]
            Cam7, Cam8, Cam9 = x0[12], x0[13], x0[14]
            
            return Cl, Cd, Crot, Cam1, Cam2, Crd, Cam3, Cam4, Cam5, Cam6, Cam7, Cam8, Cam9
        
        #%% force optimization

        # cost function which tells us how far off our QSM values are from the CFD ones for the forces
        def cost_forces( x, self, show_plots=False):            
            Cl, Cd, Crot, Cam1, Cam2, Crd, Cam3, Cam4, Cam5, Cam6, Cam7, Cam8, Cam9 = getAerodynamicCoefficients(self, x, self.AoA)

            rho = 1.0 #1.225 # for future work, can also be set to 1.0 simply
            nt = self.nt
            
            # these are either precomputed from the wing shape or simply set to one.
            # The model can do without, as the optimizer simply adjusts the coefficients 
            # accordingly.
            Iam = self.Iam
            Ild = self.Ild
            Irot = self.Irot
        
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # lift/drag forces
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Ellingtons lift/drag forces, following Cai et al 2021. Note many other papers dealt with hovering
            # flight, which features zero cruising speed. Cai et al is a notable exception. Instead of the more
            # common planar_rots_wing_mag, he used u_tip_mag, which includes the flight velocity
            # and thus delivers nonzero values even for still wings. Cai integrated over the blades, and each blade 
            # had the correct velocity. This can be done analytically, as we do here, see my lecture notes. 
            if model_terms[0] is True:
                # self.Ftc_mag = 0.5*rho*Cl*(self.planar_rots_wing_mag**2)*Ild
                # self.Ftd_mag = 0.5*rho*Cd*(self.planar_rots_wing_mag**2)*Ild
                
                # self.Ftc_mag = 0.5*rho*Cl*(self.u_tip_mag**2)*self.S2
                # self.Ftd_mag = 0.5*rho*Cd*(self.u_tip_mag**2)*self.S2
                
                A = self.planar_rots_wing_mag**2
                B = 2.0*(self.u_infty_w[:,2]*self.rots_wing_w[:,0]-self.u_infty_w[:,0]*self.rots_wing_w[:,2])
                C = self.u_infty_w[:,0]**2+self.u_infty_w[:,1]**2+self.u_infty_w[:,2]**2
    
                self.Ftc_mag = 0.5*rho*Cl*( self.S2*A + self.S1*B + self.S0*C )
                self.Ftd_mag = 0.5*rho*Cd*( self.S2*A + self.S1*B + self.S0*C )

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # rotational forces
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # These formulations differ slightly from the above classical ones, because we include the body velocity
            # and use the angular velocity instead of alpha, alpha_dt.
            # Sane's original rotational force, "discovered" in Dickinson1999 and later modeled by Sane2002
            if model_terms[1] is True:
                self.Frc_mag = rho*Crot*self.u_tip_mag*self.rots_wing_w[:,1]*Irot # Nakata et al. 2015, Eqn. 2.6c
                # self.Frc_mag[:] = rho*Crot*self.planar_rots_wing_mag*self.rots_wing_w[:,1]*Irot # Nakata et al. 2015, Eqn. 2.6c
            
            # Rotational drag: \cite{Cai2021}. The fact that the wing rotates around its rotation axis, which is the $y$ component of the angular velocity 
            # (Which Cai identifies as $\dot{\alpha}$, even though this is only an approximation) induces a net non-zero velocity component normal 
            # to the surface of the wing. In the opposite direction appears thus a drag force, and this is called rotaional drag. It would be 
            # zero if leading- and trailing edge were symmetrical. Cai in general assumes all QSM forces are perpendicular to the wing, an 
            # approximation he introduces but does not explain well. In their reference data, the force is indeed very normal to the wing. 
            # However, why neglecting the non-normal part, unless very helpful?
            # It is however correct that Cai says: as the Ellington terms do only include the velocity of the blade point on the 
            # rotation axis (the point $(0,r,0)^T$), the rotation around that very axis ($y$) is not included in the traditional term. 
            if model_terms[2] is True:
                self.Frd_mag = -1/6*rho*Crd*np.abs(self.rots_wing_w[:,1])*self.rots_wing_w[:,1]#Cai et al. 2021, Eqn 2.13

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # added mass forces
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # *Normal* added mass force
            # this most general formulation of the added mass force is inspired by
            # Whitney2010, 2.6.5. Eqn 2.40 gives the general form of the equation, and 
            # many authors assume the majority of the coefficients are zero (because 
            # the wing is thin). 
            #
            # Here, we simply use a general form, and include all components of the linear and angular 
            # acceleration. We do, however, ignore the cross terms from Whitney2010. They wrote: "Note that many
            # of the terms in (2.40) are ‘cross-term’ accelerations and not pure rotations. These will
            # not be considered, as they will duplicate existing blade-element terms with similar forms, such 
            # as the rotational force and damping terms." 
            #
            # We follow this argument.
            #
            if model_terms[3] is True:
                self.Fam_z_mag = rho*(Cam1*self.a_tip_w[:, 0] + Cam2*self.a_tip_w[:, 1] + Cam3*self.a_tip_w[:, 2] + \
                                 Cam4*self.rot_acc_wing_w[:, 0] + Cam5*self.rot_acc_wing_w[:, 1] + Cam6*self.rot_acc_wing_w[:, 2] )
                 
            
            # *Tangential* added mass forces, like discussed in VanVeen2022, 2.1.1.
            # We assume however a more general form, which includes the components of the acceleration.
            # We do not include the acceleration in y-direction, even though it significantly reduces the error.
            # The reason is that apparently, this form has overlap with the lift+drag forces from the Ellington model
            # and thus the resulting model becomes harder to interpret.
            if model_terms[4] is True:
                self.Fam_x_mag = rho*(Cam7*self.a_tip_w[:, 0] + Cam8*self.a_tip_w[:, 2])

            # this even more complete model that included all wing acceleration components proved not a big improvement
            # and seems to have some overlap with the lift/drag definition, thus rendering interpretation more difficult.
            # We therefore drop it.
            ## self.Fam_x_mag = Cam7*self.a_tip_w[:, 0] + Cam8*self.a_tip_w[:, 1] + Cam9*self.a_tip_w[:, 2]
            

            # vector calculation of Ftc, Ftd, Frc, Fam, Frd arrays of the form (nt, 3).these vectors are in the global reference frame 
            for k in [0, 1, 2]:
                self.Ftc[:, k] = self.Ftc_mag * self.e_lift_g[:,k]
                self.Ftd[:, k] = self.Ftd_mag * self.e_drag_g[:,k]

                # using e_lift_g instead of ez_wing_g did makes approx. worse.
                # (this was suggested in Cai et al. 2021, Appendix A, just below Eqn A4)
                self.Frc[:, k] = self.Frc_mag * self.ez_wing_g[:,k] # Sane2002 also state its e_z, like Cai et al
                self.Frd[:, k] = self.Frd_mag * self.ez_wing_g[:,k]
                
                # normal added mass force
                self.Fam[:, k] = self.Fam_z_mag * self.ez_wing_g[:,k]
                # tangential added mass force
                self.Fam2[:, k] = self.Fam_x_mag * self.ex_wing_g[:,k]
                

            # total force generated by QSM
            for d in range(3):
                self.F_QSM_g[:, d] = self.Ftc[:, d] + self.Ftd[:, d] + self.Frc[:, d] + self.Fam[:, d] + self.Fam2[:, d] + self.Frd[:, d]

            
            # compute quality metric (relative L2 error)
            K_forces     = norm(self.F_QSM_g[:,0]-self.F_CFD_g[:,0]) + norm(self.F_QSM_g[:,2]-self.F_CFD_g[:,2])
            K_forces_den = norm(self.F_CFD_g[:,0]) + norm(self.F_CFD_g[:,2])
            # normalization
            if K_forces_den >= 1e-16: 
                K_forces /= K_forces_den

            # Note projection using the basis vectors is easier to vectorize; further note it could be merged 
            # with the above loop, because the forces are actually *defined* in the wing system
            self.F_QSM_w[:, 0] = self.F_QSM_g[:,0]*self.ex_wing_g[:,0] + self.F_QSM_g[:,1]*self.ex_wing_g[:,1] + self.F_QSM_g[:,2]*self.ex_wing_g[:,2]
            self.F_QSM_w[:, 1] = self.F_QSM_g[:,0]*self.ey_wing_g[:,0] + self.F_QSM_g[:,1]*self.ey_wing_g[:,1] + self.F_QSM_g[:,2]*self.ey_wing_g[:,2]
            self.F_QSM_w[:, 2] = self.F_QSM_g[:,0]*self.ez_wing_g[:,0] + self.F_QSM_g[:,1]*self.ez_wing_g[:,1] + self.F_QSM_g[:,2]*self.ez_wing_g[:,2]

            if show_plots:
                ##FIGURE 2
                fig, axes = plt.subplots(2, 2, figsize = (15, 10))

                #coefficients
                graphAoA = np.linspace(-9, 90, 100)*(np.pi/180)
                gCl, gCd, gCrot, gCam1, gCam2, gCrd, _, _,_,_, _,_,_  = getAerodynamicCoefficients(self, x, graphAoA)
                axes[0, 0].plot(np.degrees(graphAoA), gCl, label='Cl', color='#0F95F1')
                axes[0, 0].plot(np.degrees(graphAoA), gCd, label='Cd', color='#F1AC0F')
                # ax.plot(np.degrees(graphAoA), gCrot*np.ones_like(gCl), label='Crot')
                axes[0, 0].set_title('Lift and drag coeffficients') 
                axes[0, 0].set_xlabel('AoA[°]')
                axes[0, 0].set_ylabel('[-]')
                axes[0, 0].legend(loc = 'upper right') 

                
                
                #vertical forces
                axes[0, 1].plot(tt, self.Ftc[:, 2], label = 'Vert. part of F_{TC} (Ellington1984 lift force)', color='gold')
                axes[0, 1].plot(tt, self.Ftd[:, 2], label = 'Vert. part of F_{TD} (Ellington1984 drag force)', color='lightgreen')
                axes[0, 1].plot(tt, self.Frc[:, 2], label = 'Vert. part of F_{RC}  (Sane2002, rotational force)', color='orange')
                axes[0, 1].plot(tt, self.Fam[:, 2], label = 'Vert. part of F_{AMz} (Whitney2010)', color='red')
                axes[0, 1].plot(tt, self.Fam2[:, 2], '--',label = 'Vert. part of F_{AMx} (vanVeen2022)', color='red')
                axes[0, 1].plot(tt, self.Frd[:, 2], label = 'Vert. part of F_{RD} (Cai2021, Nakata2015)', color='green')
                axes[0, 1].plot(tt, self.F_QSM_g[:,2], label = 'Total Vert. part of  QSM force', ls='-.', color='blue')
                axes[0, 1].plot(tt, self.F_CFD_g[:,2], label = 'Total Vert. part of  CFD force', ls='--', color='k')
                axes[0, 1].set_xlabel('$t/T$')
                axes[0, 1].set_ylabel('force')
                axes[0, 1].set_title('Vertical components of forces in global coordinate system')
                axes[0, 1].legend(loc = 'best')
            
                #qsm + cfd force components in wing reference frame
                axes[1, 0].plot(tt, self.F_QSM_w[:, 0], label='Fx_QSM_w', c='r')
                axes[1, 0].plot(tt, self.F_CFD_w[:, 0], ls='-.', label='Fx_CFD_w', c='r')
                axes[1, 0].plot(tt, self.F_QSM_w[:, 1], label='Fy_QSM_w', c='g')
                axes[1, 0].plot(tt, self.F_CFD_w[:, 1], ls='-.', label='Fy_CFD_w', c='g')
                axes[1, 0].plot(tt, self.F_QSM_w[:, 2], label='Fz_QSM_w', c='b')
                axes[1, 0].plot(tt, self.F_CFD_w[:, 2], ls='-.', label='Fz_CFD_w', c='b')
                axes[1, 0].set_xlabel('$t/T$')
                axes[1, 0].set_ylabel('force')
                axes[1, 0].set_title('QSM + CFD force components in wing reference frame')
                axes[1, 0].legend(loc='best')

                #forces
                axes[1, 1].plot(tt[:], self.F_QSM_g[:,0], label='Fx_QSM_g', color='red')
                axes[1, 1].plot(tt[:], self.F_CFD_g[:,0], label='Fx_CFD_g', linestyle = 'dashed', color='red')
                axes[1, 1].plot(tt[:], self.F_QSM_g[:,1], label='Fy_QSM_g', color='green')
                axes[1, 1].plot(tt[:], self.F_CFD_g[:,1], label='Fy_CFD_g', linestyle = 'dashed', color='green')
                axes[1, 1].plot(tt[:], self.F_QSM_g[:,2], label='Fz_QSM_g', color='blue')            
                axes[1, 1].plot(tt[:], self.F_CFD_g[:,2], label='Fz_CFD_g', linestyle = 'dashed', color='blue')
                axes[1, 1].set_xlabel('$t/T$')
                axes[1, 1].set_ylabel('force')
                if norm(self.F_CFD_g[:,0]) > 0.0 and norm(self.F_CFD_g[:,1]) > 0.0 and norm(self.F_CFD_g[:,2]) > 0.0:
                    axes[1, 1].set_title( "Fi_QSM_g/Fi_CFD_g=(%2.2f, %2.2f, %2.2f) \nK=%2.2f" % (norm(self.F_QSM_g[:,0])/norm(self.F_CFD_g[:,0]), 
                                                                                        norm(self.F_QSM_g[:,1])/norm(self.F_CFD_g[:,1]), 
                                                                                        norm(self.F_QSM_g[:,2])/norm(self.F_CFD_g[:,2]),
                                                                                        K_forces) ) 
                else:
                    axes[1, 1].set_title( "Fi_QSM_g/Fi_CFD_g=(%2.2f, %2.2f, %2.2f) \nK=%2.2f" % (norm(self.F_QSM_g[:,0]), norm(self.F_QSM_g[:,1]), norm(self.F_QSM_g[:,2]), K_forces) ) 
                axes[1, 1].legend(loc = 'lower right') 


                for ax in axes.flatten()[1:]:
                    insect_tools.indicate_strokes(ax=ax, tstart=[self.T_reversals[0]], tstroke=2*(self.T_reversals[1]-self.T_reversals[0]) )

                plt.tight_layout()
                plt.draw()
            return K_forces
        
        #----------------------------------------------------------------------
        # optimizing using scipy.optimize.minimize which is faster        
        if optimize:            
            # as a means of informing the user that they need to read CFD data before fitting (training): 
            if not self.readAtLeastOneCFDrun:
                raise ValueError("New (11/2024): You need to read CFD before you can fit the model to it. call QSM.read_CFD_data")
                
            start = time.time()            
            bounds = [(-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-6, 6), (-60, 60)]
            K_forces = 9e9
            
            # optimize N_trials times from a different initial guess, use best solution found
            # NOTE: tests indicate the system always finds the same solution, so this could
            # be omitted. Kept for safety - we do less likely get stuck in local minima this way
            for i_trial in range(N_trials):
                x0_forces = np.random.rand(15)
                optimization = opt.minimize(cost_forces, args=(self, False), bounds=bounds, x0=x0_forces)
                
                x0_forces = optimization.x
                
                if optimization.fun < K_forces:
                    K_forces = optimization.fun
                    x0_best = x0_forces        
                    
                print('Trial %i/%i K=%2.3f' % (i_trial+1, N_trials, optimization.fun))
                    
            self.x0_forces = x0_best     
            self.K_forces = K_forces
            
            print('Completed in:', round(time.time() - start, 4), 'seconds')
            print('x0_optimized:', np.round(self.x0_forces, 5), '\nK_optimized_forces:', K_forces)
            
        self.K_forces = cost_forces(self.x0_forces, self, show_plots=plot)
        if verbose:
            print("K_forces_final=%f" % (self.K_forces))
        
        #%% moments optimization
        
        #cost_moments is defined in terms of the moments. this function will be optimized to find the lever (coordinates) that best matches (match) the QSM moments to their CFD counterparts  
        def cost_moments(x, self, show_plots=False):
            #here we define the the QSM moments as: M_QSM = [ C_lever_x_w*Fz_QSM_w, -C_lever_x_w*Fz_QSM_w, C_lever_x_w*Fy_QSM_w - C_lever_y_w*F_x_QSM_w ]
            #where C_lever_x_w and C_lever_y_w correspond to the spanwise and the chordwise locations of the lever in the wing reference frame. 
            #vector form: C_lever_w = [C_lever_x_w, C_lever_y_w, 0]
            C_lever_x_w = x[0]
            C_lever_y_w = x[1]            
            
            # moment in wing reference frame
            self.M_QSM_w[:,0] =  C_lever_y_w*self.F_QSM_w[:, 2]
            self.M_QSM_w[:,1] = -C_lever_x_w*self.F_QSM_w[:, 2]
            self.M_QSM_w[:,2] =  C_lever_x_w*self.F_QSM_w[:, 1] - C_lever_y_w*self.F_QSM_w[:, 0]

            K_moments = norm(self.M_QSM_w[:,0]-self.M_CFD_w[:,0]) + norm(self.M_QSM_w[:,1]-self.M_CFD_w[:,1]) + norm(self.M_QSM_w[:,2]-self.M_CFD_w[:,2]) 
            K_moments_den = norm(self.M_CFD_w[:,0]) + norm(self.M_CFD_w[:,1]) + norm(self.M_CFD_w[:,2]) 
            # normalization
            if K_moments_den >= 1.0e-16: 
                K_moments /= K_moments_den                

            return K_moments

        # moment optimization
        if optimize:
            x0_moments = [1.0, 1.0]
            bounds = [(-6, 6), (-6, 6)]
            
            start = time.time()
            optimization = opt.minimize(cost_moments, args=(self, False), bounds=bounds, x0=x0_moments)
            print('Completed in:', round(time.time() - start, 4), 'seconds')
            
            self.x0_moments = optimization.x
            self.K_moments  = optimization.fun
            
            print('x0_moments_optimized:', np.round(self.x0_moments, 5), '\nK_moments_optimized:', self.K_moments)
        
        # save final value
        self.K_moments = cost_moments(self.x0_moments, self, plot)

        # compute QSM moment in global reference frame
        for i in range(self.M_QSM_g.shape[0]):
            self.M_QSM_g[i, :] = np.matmul(self.M_w2g[i, :, :], self.M_QSM_w[i, :])
            



        #%% power optimization
        
        #cost_power is defined in terms of the moments and power. this function will be optimized to find the lever (coordinates) that best matches (match) the QSM power to its CFD counterpart
        def cost_power(x, self, show_plots=False):
            #here we define the the QSM moments as: M_QSM = [ C_lever_x_w_power*Fz_QSM_w, -C_lever_x_w_power*Fz_QSM_w, C_lever_x_w_power*Fy_QSM_w - C_lever_y_w_power*F_x_QSM_w ]
            #where C_lever_x_w_power and C_lever_y_w_power correspond to the spanwise and the chordwise locations of the lever in the wing reference frame. 
            #vector form: C_lever_w = [C_lever_x_w_power, C_lever_y_w_power, 0]

            C_lever_x_w_power = x[0]
            C_lever_y_w_power = x[1]
            
            self.Mx_QSM_w_power = np.zeros((self.nt,1))
            self.My_QSM_w_power = np.zeros((self.nt,1))
            self.Mz_QSM_w_power = np.zeros((self.nt,1))
            
            self.P_QSM_nonoptimized = np.zeros((self.nt,1))
            self.P_QSM = np.zeros((self.nt,1))

            self.Mx_QSM_w_power =  C_lever_y_w_power*self.F_QSM_w[:, 2]
            self.My_QSM_w_power = -C_lever_x_w_power*self.F_QSM_w[:, 2]
            self.Mz_QSM_w_power =  C_lever_x_w_power*self.F_QSM_w[:, 1] - C_lever_y_w_power*self.F_QSM_w[:, 0]


            self.P_QSM_nonoptimized = -(self.M_QSM_w[:,0]*self.rots_wing_w[:, 0]
                                      + self.M_QSM_w[:,1]*self.rots_wing_w[:, 1]
                                      + self.M_QSM_w[:,2]*self.rots_wing_w[:, 2])

            self.P_QSM = -(self.Mx_QSM_w_power*self.rots_wing_w[:, 0] 
                         + self.My_QSM_w_power*self.rots_wing_w[:, 1] 
                         + self.Mz_QSM_w_power*self.rots_wing_w[:, 2])

            K_power     = norm(self.P_QSM - self.P_CFD) 
            K_power_den = norm(self.P_CFD)
            # normalization
            if K_power_den >= 1e-16: 
                K_power /= K_power_den

            if show_plots:
                ##FIGURE 4
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 15))

                #cfd vs qsm moments
                ax1.plot(tt, self.M_QSM_w[:, 0], label='Mx_QSM_w', color='red')
                ax1.plot(tt, self.M_CFD_w[:, 0], label='Mx_CFD_w', ls='-.', color='red')
                ax1.plot(tt, self.M_QSM_w[:, 1], label='My_QSM_w', color='blue')
                ax1.plot(tt, self.M_CFD_w[:, 1], label='My_CFD_w', ls='-.', color='blue')
                ax1.plot(tt, self.M_QSM_w[:, 2], label='Mz_QSM_w', color='green')
                ax1.plot(tt, self.M_CFD_w[:, 2], label='Mz_CFD_w', ls='-.', color='green')
                ax1.set_xlabel('$t/T$')
                ax1.set_ylabel('moment]')       
                
                if norm(self.M_CFD_g[:,0])>0 and norm(self.M_CFD_g[:,1])>0 and norm(self.M_CFD_g[:,2])>0:
                    ax1.set_title( "Mi_QSM_w/Mi_CFD_w=(%2.2f, %2.2f, %2.2f)" % (norm(self.M_QSM_w[:,0])/norm(self.M_CFD_g[:,0]), 
                                                                                norm(self.M_QSM_w[:,1])/norm(self.M_CFD_g[:,1]), 
                                                                                norm(self.M_QSM_w[:,2])/norm(self.M_CFD_g[:,2])) ) 
                else:
                    ax1.set_title( "Mi_QSM_w/Mi_CFD_w=(%2.2f, %2.2f, %2.2f)" % (norm(self.M_QSM_w[:,0]), norm(self.M_QSM_w[:,1]), norm(self.M_QSM_w[:,2])) ) 
                ax1.legend()

                #optimized aerodynamic power
                ax2.plot(tt, self.P_QSM_nonoptimized, label='P_QSM (non-optimized)', c='purple')
                ax2.plot(tt, self.P_QSM, label='P_QSM (optimized)', color='b')
                ax2.plot(tt, self.P_CFD, label='P_CFD', ls='-.', color='indigo')
                ax2.set_xlabel('$t/T$')
                ax2.set_ylabel('aerodynamic power')
                ax2.set_title("P_QSM/P_CFD=%2.2f K_power=%3.3f" % (norm(self.P_QSM)/norm(self.P_CFD), K_power) )
                ax2.legend()
                plt.tight_layout()
                plt.draw()
                
                insect_tools.indicate_strokes(ax=ax1, tstart=[self.T_reversals[0]], tstroke=2*(self.T_reversals[1]-self.T_reversals[0]) )
                insect_tools.indicate_strokes(ax=ax2, tstart=[self.T_reversals[0]], tstroke=2*(self.T_reversals[1]-self.T_reversals[0]) )
                
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
            
        
        self.K_power = cost_power(self.x0_power, self, show_plots=plot)
        
        if verbose:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            

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
        
        self.x_wingContour_w  = np.vstack([xc, yc, zc])
        self.x_wingContour_w  = np.transpose(self.x_wingContour_w)
        
        self.bodyPointsSequence   = np.zeros((self.nt, self.x_wingContour_w.shape[0], 3))
        
        # this function calculates the chord length by splitting into 2 segments (LE and TE segment) 
        # and then interpolating along the yw-axis 
        # It returns the chord length c as a function of r
        def getChordLength(wingPoints, r):            
            # get the division in wing segments (leading and trailing)
            righthand_section = wingPoints[ wingPoints[:,0]>=0.0, :]
            lefthand_section  = wingPoints[ wingPoints[:,0]<0.0,  :]

            #interpolate righthand section 
            righthand_section_interpolation = interp1d(righthand_section[:, 1], righthand_section[:, 0], fill_value='extrapolate')

            #interpolate lefthand section
            lefthand_section_interpolation = interp1d(lefthand_section[:, 1], lefthand_section[:, 0], fill_value='extrapolate') 
            
            #generate the chord as a function of y coordinate
            chord_length = abs(righthand_section_interpolation(r) - lefthand_section_interpolation(r))
            return chord_length
        
        
        
        min_y = np.min(self.x_wingContour_w[:, 1])
        max_y = np.max(self.x_wingContour_w[:, 1])
        
        # chord calculation 
        r = np.linspace(min_y, max_y, nb)
        c = getChordLength(self.x_wingContour_w, r)

        #computation following Nakata 2015 eqns. 2.4a-c
        c_interpolation = interp1d(r, c) #we create a function that interpolates our chord (c) w respect to our span (r)

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
        def Cr(r):
            return (c_interpolation(r)**1) * r
        def C2(r):
            return (c_interpolation(r)**2)
        def C(r):
            return (c_interpolation(r))
        def C3r3(r):
            return(np.sqrt((c_interpolation(r)**3)*(r**3)))
        
        from scipy.integrate import simpson
        
        self.Iam  = simpson(C2(r), r)
        self.Iwe  = simpson(C3r3(r), r)
        self.Ild  = simpson(Cr2(r), r) #second moment of area for lift/drag calculations
        self.Irot = simpson(C2r(r), r) #second moment of area for rotational force calculation 
        self.IA = simpson(C(r), r) 
        
        # area moments:
        self.S2 = self.Ild
        self.S1 = simpson(Cr(r), r)
        self.S0 = area
    
        self.hinge_index   = np.argmin( self.x_wingContour_w[:, 1] )
        self.wingtip_index = np.argmax( self.x_wingContour_w[:, 1] )
        
        # for kleemeier wing, integration did not work properly...
        if np.isnan(self.Iam): self.Iam = 1.0
        if np.isnan(self.Iwe): self.Iwe = 1.0
        if np.isnan(self.Ild): self.Ild = 1.0
        if np.isnan(self.Irot): self.Irot = 1.0
        if np.isnan(self.S2): self.S2 = 1.0
        if np.isnan(self.S1): self.S1 = 1.0
        if np.isnan(self.S0): self.S0 = 1.0

    
