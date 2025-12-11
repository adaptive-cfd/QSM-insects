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
import finite_differences

# use latex
# plt.rcParams["text.usetex"] = True

latex = plt.rcParams["text.usetex"]

deg2rad = np.pi/180.0

def norm(x):
    return( np.linalg.norm(x.flatten()) )

def normalize_vector( vector ):
    mag = norm( vector.flatten() )
    if mag >= 1e-16:
        vector /= mag

    return vector

def apply_rotations_to_vectors(M, x):
    """
    Mass-rotate an array of vectors in one go (without loop).

    Parameters
    ----------
    M : arrax, size(N,3,3)
        Rotation matrices, where 1st index is time
    x : array, size(N,3)
        Vector to be rotated

    Returns
    -------
    y : array, size(N,3)
        Resulting array of rotated vectors

    """
    y = np.zeros(x.shape)
    
    # Loop version
    # for it in np.arange(M.shape[0]):
    #     y[it, :] = np.matmul(M[it, :], x[it, :])
    
    # loop-free optimized version
    y = np.einsum("nij,nj->ni", M, x, optimize=True)
        
    return y






class QSM:
    """
    QSM object. The QSM describes one wing, so for a dragonfly for example you'll have four of these objects.
    
    Initialization Parameters
    ------------
    nt :
        The kinematics, forces, etc. are evaluated on an equidistant time vector with nt time steps.
    nruns :
        The number of CFD simulations that you want to base the model on. The default case is 1 (a single CFD simulation is used
        to determine the QSM coefficients), but you can also fit the model to a number >1 of runs simultaneously. The QSM coefficient
        optimizer will fit as best possible all data.
    timeline :
        It is sometimes useful to provide a timeline (time vector) yourself. The default is [0,1) with nt samples and without then endpoint t=1
    model_CL_CD :
        The Ellington terms (lift and drag) require evaluation of CL and CD as a function of AoA, for this a function is needed. Implemented models
        are "Nakata" [Nakata2015], "Dickinson" [Dickinson1999] and 'Polhamus' [J-S Han et al Bioinspr Biomim 12 2017 036004]. The results are in general rather
        similar using the three models. 
    """
    def __init__(self, nt=300, nruns=1, timeline=None, model_CL_CD='Dickinson', 
                 model_terms=5*[True], ellington_type='utip', reversal_detector="phi_dt"):
        
        
        self.AM_model = 'Full 6DOF scaled'
        self.model_terms = model_terms
        self.ellington_type = ellington_type
        self.reversal_detector = reversal_detector
        
        self.nt = nt
        self.model_CL_CD = model_CL_CD
        self.nruns = nruns
        
        # as a means of informing the user that they need to read CFD data before fitting (training)
        self.readAtLeastOneCFDrun = False

        # timeline for one run [0..1)
        if timeline is None:
            self.timeline = np.linspace(0, 1, nt, endpoint=False)
        else:
            self.timeline = timeline
        self.dt = self.timeline[1] - self.timeline[0]
        
        
        # timeline for the concatenated data
        self.timeline_all = np.zeros((nt*nruns))   
        for irun in range(nruns):
            i0 = irun*nt
            ii = np.arange(start=i0, stop=i0+nt-1+1 )
            self.timeline_all[ii] = self.timeline + float(irun)

        self.alpha_dt = np.zeros((nt*nruns))
        self.phi_dt   = np.zeros((nt*nruns))
        self.theta_dt = np.zeros((nt*nruns))

        self.alpha_dtdt = np.zeros((nt*nruns))
        self.phi_dtdt   = np.zeros((nt*nruns))
        self.theta_dtdt = np.zeros((nt*nruns))

        self.rot_wing_b = np.zeros((nt*nruns,3))
        self.rot_wing_w = np.zeros((nt*nruns,3))
        self.rot_wing_g = np.zeros((nt*nruns,3))

        self.planar_rot_wing_w = np.zeros((nt*nruns,3))
        self.planar_rot_wing_g = np.zeros((nt*nruns,3))
        self.planar_rot_wing_mag = np.zeros((nt*nruns))

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
        self.e_lift_b = np.zeros((nt*nruns,3))        

        self.ey_wing_g = np.zeros((nt*nruns,3))
        self.ez_wing_g = np.zeros((nt*nruns,3))
        self.ex_wing_g = np.zeros((nt*nruns,3))
        
        self.M_g2b = np.zeros((nt*nruns, 3, 3))
        self.M_b2g = np.zeros((nt*nruns, 3, 3))        
        self.M_g2w = np.zeros((nt*nruns, 3, 3))
        self.M_w2g = np.zeros((nt*nruns, 3, 3))        
        self.M_b2w = np.zeros((nt*nruns, 3, 3))
        self.M_w2b = np.zeros((nt*nruns, 3, 3))        
        self.M_b2s = np.zeros((nt*nruns, 3, 3))
        self.M_s2b = np.zeros((nt*nruns, 3, 3))
        self.M_s2w = np.zeros((nt*nruns, 3, 3))
        self.M_w2s = np.zeros((nt*nruns, 3, 3))

        self.Ftc = np.zeros((nt*nruns,3))
        self.Ftd = np.zeros((nt*nruns,3))
        self.Frc = np.zeros((nt*nruns,3))
        self.Fam = np.zeros((nt*nruns,3))
        self.Fam2 = np.zeros((nt*nruns,3))
        self.Frd = np.zeros((nt*nruns,3))

        self.u_infty_w = np.zeros((nt*nruns,3))
        self.u_infty_g = np.zeros((nt*nruns,3))
        
        self.P_QSM_nonoptimized = np.zeros((nt*nruns))
        self.P_QSM = np.zeros((nt*nruns))

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
        # Default is ONES (the code can then work without calling setup_wing_shape
        # as long as the training and evaluation runs all have the same shape)
        self.S_AM1 = np.ones((nt*nruns))
        self.S_AM2 = np.ones((nt*nruns))
        self.S_RC = np.ones((nt*nruns))
        self.S_RD = np.ones((nt*nruns))
        self.S2 = np.ones((nt*nruns))
        self.S1 = np.ones((nt*nruns))
        self.S0 = np.ones((nt*nruns))

        self.wing = 'left'
        self.x_wingContour_w = np.zeros((10,3))

        self.x0_forces = np.zeros((15))
        self.x0_moments = np.zeros((2))
        self.x0_power = np.zeros((2))
        
        self.K_forces_individual = np.zeros((nruns))
        self.K_moments_individual = np.zeros((nruns))
        self.K_power_individual = np.zeros((nruns))

        self.psi = np.zeros((nt*nruns))
        self.beta = np.zeros((nt*nruns))
        self.gamma = np.zeros((nt*nruns))
        self.eta = np.zeros((nt*nruns))
        self.alpha = np.zeros((nt*nruns))
        self.phi = np.zeros((nt*nruns))
        self.theta = np.zeros((nt*nruns))
        
        # to save computing time, fill differentiation matrices only once
        self.D1 = finite_differences.D12(self.nt, self.dt)
        self.D2 = finite_differences.D22(self.nt, self.dt)


    def parse_many_run_directorys(self, run_directories, kinematics_file='kinematics.t', 
                                  T0=1.0, wing='right', plot=False, setup_wing_shape=True, verbose=True):
        """
        This function is for reading in a bunch of CFD runs in order to fit a single
        QSM model to many runs. It parses the kinematics data (either kinematics.t or an *.ini
        file), and reads in the CFD data for the forces, moments, power.
        
        Parameters
        -----------
        
        run_directories :
            Python list of the simulation folders (complete path)
        params_file : 
            just the filename (not the complete path, i.e., we look in the folders run_directories each time for this file)
        kinematics_file :
            the filename where to take the kinematics from. usually, this will be kinematics.t, but it can also be the wing kinematics INI file.
        setup_wing_shape : bool
            The QSM code can work without wing shapes, and sets all geometry constants to 1.0. This works only if all data 
            use the same wing shape. If the data contains different wing shapes, it is better to actually properly compute the geometry factors.

        Note: this function replaces using read_CFD_data() and parse_kinematics() in case you have several CFD runs.
        """
        import glob
        import inifile_tools
        import os
        
        # check if right amount of memory is allocated
        if not self.psi.shape[0] == self.nt*len(run_directories):
            raise ValueError("We try to process %i runs, but allocated not enough storage. Use QSM = qsm_class.QSM(nt=XXX, nruns=YYY)")

        i0 = 0
        for run in run_directories:
            run += '/'
            
            PARAMS = inifile_tools.find_WABBIT_main_inifile(run)
            
            if setup_wing_shape:
                file_WingShape = inifile_tools.get_ini_parameter( PARAMS , 'Insects', 'WingShape', str, 'none')[0]
                file_WingShape = file_WingShape.replace('from_file::','')
                
                if not os.path.isfile(run+file_WingShape):
                    raise ValueError("""We try to initialize the wing shape for run %i (%s), and identified %s as the WinShape file.
                                     We do however not find this file, and cannot initialize the wing geometry parameters for this run.
                                     Please check if the PARAMS file (%s) refers to an existing WING *.ini file. Note: it may be
                                     that the QSM code incorrectly identifies the ShapeFile, if two distinct ones are used for left/right wing.""")
                
                self.setup_wing_shape( run+file_WingShape, i0=i0, verbose=verbose)
            
            self.parse_kinematics( params_file=PARAMS, kinematics_file=run+kinematics_file, wing=wing, i0=i0, plot=plot, verbose=verbose  )
            self.read_CFD_data(run, T0, i0, verbose=verbose)

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
        x_wingContour_w[:,2] = sign*0.02
        points_top = np.transpose( np.matmul(self.M_g2w[it,:,:].T, x_wingContour_w.T) )

        x_wingContour_w = self.x_wingContour_w
        x_wingContour_w[:,2] = -0.02*sign
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
        ax.add_collection3d(Poly3DCollection(verts=[points_top], color='r', edgecolor='k', alpha=0.95))
        ax.add_collection3d(Poly3DCollection(verts=[points_bot], color='lightsalmon', edgecolor='k', alpha=0.95))

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


        # a few unit vectors:
        # the wing reference frame
        ax.quiver( X=[0.0, 0.0, 0.0], Y=[0.0, 0.0, 0.0], Z=[0.0, 0.0, 0.0], 
                  U=[self.ex_wing_g[it,0],self.ey_wing_g[it,0],self.ez_wing_g[it,0]], 
                  V=[self.ex_wing_g[it,1],self.ey_wing_g[it,1],self.ez_wing_g[it,1]], 
                  W=[self.ex_wing_g[it,2],self.ey_wing_g[it,2],self.ez_wing_g[it,2]],
                  arrow_length_ratio=0.15, label='wing reference frame', color='c')
        # lift /drag
        ax.quiver( X=[0.0, 0.0], Y=[0.0, 0.0], Z=[0.0, 0.0], 
                  U=[self.e_lift_g[it,0],self.e_drag_g[it,0]], 
                  V=[self.e_lift_g[it,1],self.e_drag_g[it,1]], 
                  W=[self.e_lift_g[it,2],self.e_drag_g[it,2]],
                  arrow_length_ratio=0.15, label='lift \& drag unit vectors', color='m')
        

        #set the axis limits
        ax.set_xlim([-a, a])
        ax.set_ylim([-a, a])
        ax.set_zlim([-a, a])

        #set the axis labels
        ax.set_xlabel('$x/R$')
        ax.set_ylabel('$y/R$')
        ax.set_zlabel('$z/R$')
        
        ax.legend()
        
        

    def kinematics_3D_animation(self, fnames_out="visualization3D", directory='./', dark_plot=False, dpi=200):
        """
        Creates one PNG file per time step, loops over the entire time vector
        """
        
        if dark_plot:
            plt.style.use('dark_background')

        # can be called only after parse_kinematics
        plt.figure()
        ax = plt.figure().add_subplot(projection='3d')

        for it in range(self.nt):
            ax.cla()
            self.kinematics_3D_frame(it, ax)
            plt.show()
            fname_out = directory + '/' + fnames_out+'.%04i.png' % (it)
            print('Animation 3D visualization saving: ' + fname_out)
            
            plt.savefig( fname_out, dpi=dpi )
            
    def read_kinematics(self, params_file, kinematics_file='kinematics.t'):
        print('hi')

    def parse_kinematics(self, params_file, kinematics_file=None, u_infty_g=None, plot=True, wing='auto', yawpitchroll0=None, eta_stroke=None, i0=0,
                         alpha=None, phi=None, theta=None, psi=None, beta=None, gamma=None, verbose=True, optimized_loading=True, T0=0.0):
        """
        Evaluate the kinematics (i.e. compute many rotation matrices etc) and store the results
        in the QSM class arrays. You must run this before you can fit the QSM to the CFD data.

        There are 3 ways to read kinematics:
        -------------------------------------
            _1_ Reading kinematics.t from an existing CFD simulation (i.e. a run that has been simulated): kinematics_file=kinematics.t

            _2_ Reading an INI file for wingbeat kinematics (the input for CFD runs, but not necessarily a run that has been simulated): kinematics_file=XXX.ini
                In this mode, the body angles are determined from the main PARAMS.ini file if yawpitchroll0=None, idem for the stroke plane angle
                if eta_stroke=None. You can also pass these values to this function, in which case we take those (and not the ones from the PARAMS.ini file)

            _3_ Passing externally generated kinematics: alpha, phi, theta, eta_stroke, psi, beta, gamma as vectors sampled on equidistant time vector.
            Use QSM.timeline as time vector. Do not pass `kinematics_file` in this case (=None). The angles you pass shall be in DEGREE.

        Parameters
        ----------
        params_file : string
            Parameter file (INI) of the simulation. Used to determine which wing we model, flight velocity etc.
            Note that if you pass all parameter as arguments to this function, you do not need to pass a valid *.ini
            file and can just pass None.
        kinematics_file : string
            The kinematics file can be either:
                - "kinematics.t", the output log file of a CFD run (mode _1_)
                - an *.ini file, which is the CFD-code's descriptor file for kinematics (mode _2_)
                - None (mode _3_)
        u_infty_g : vector, optional
            Flight velocity of the animal. Default: read from params_file, overwritten if you pass a value explicitly.
            Note how the params_file contains the wind velocity (wind tunnel speed), and we invert its sign to get the
            animals flight speed. If you pass the parameter directly, give flight speed not wind tunnel speed.
        plot : bool, optional
            Plot raw kinematics or not. The default is True.
        wing : str
            Which side is the wing kinematics we parse ('auto', 'left', 'right'). The default is 'auto' - in this case 
            it is read from the `params_file` *.ini file.
        yawpitchroll0 : TYPE, optional
            If you pass kinematics.t as kinematics_file, yaw pitch roll are determined from there, but you
            can overwrite it with the (constant) value you pass. Useful for optimization problems.
        eta_stroke : TYPE, optional
            You can omit this parameter if you read the kinematics from a CFD run *that is already done*
            (which I guess is the usual case). The log-file kinematics.t contains the eta_stroke as well
            as body posture angles.
            
        Parameters for Mode _3_
        -----------------------
        
        alpha, phi, theta, psi, beta, gamma, eta_stroke : np.ndarray of shape [nt]
            The various angles (feathering, flapping, deviation, roll, pitch, yaw, stoke plane)

        """
        dt, nt = self.dt, self.nt

        # index range to store the data to (used is several runs are parsed)
        ii = np.arange(start=i0, stop=i0+nt-1+1 )
        

        def diff1(x, dt):
            # differentiate vector x with respect to time
            # use non-periodic, second order finite differences for that
            # note current version of qsm_class does require one-sided FD stencils
            # because concatenating many CFD runs does result in non-periodic data.
            return np.matmul(self.D1, x)#.reshape(x.shape)

        def diff2(x, dt):
            # differentiate vector x with respect to time
            # use non-periodic, second order finite differences for that
            # note current version of qsm_class does require one-sided FD stencils
            # because concatenating many CFD runs does result in non-periodic data.
            return np.matmul(self.D2, x)#.reshape(x.shape)

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

        # wing can be left/right/left2/right2
        self.wing = wing
        # side can only be left or right
        side = self.wing.replace("2","")

        if kinematics_file == None:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # MODE _3_:
            # kinematics given directly as input vectors
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for angle in [alpha, phi, theta, eta_stroke, psi, beta, gamma]:
                if angle.shape[0] != self.nt:
                    raise ValueError("The kinematics angles you pass to this routine need to have the same length as nt=QSM.timeline.shape[0]")

            self.psi[ii]   = psi*deg2rad
            self.beta[ii]  = beta*deg2rad
            self.gamma[ii] = gamma*deg2rad
            self.eta[ii]   = eta_stroke*deg2rad
            self.alpha[ii] = alpha*deg2rad
            self.phi[ii]   = phi*deg2rad
            self.theta[ii] = theta*deg2rad

        else:
            if "kinematics.t" in kinematics_file:
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # MODE _1_:
                # read kinematics.t (this is a pre-computed CFD run)
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # load kinematics data by means of load_t_file function from insect_tools library
                # old call:
                # kinematics_cfd = insect_tools.load_t_file(kinematics_file, interp=True, time_out=self.timeline, verbose=verbose, optimized_loading=True)
                
                
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                import datetime
                # size of the *.t file (to check if this changed)
                size_t = os.path.getsize(kinematics_file)
                run_directory = kinematics_file.replace('kinematics.t', '')
                
                # optimized loading - convert to NPZ on first call, then read that
                reloading_required, Q = False, None
                if optimized_loading:
                    if not os.path.isfile( run_directory+'/QSM-kinematics-data.npz' ):
                        # file does not exist - reloading is required.
                        reloading_required = True
                    else:
                        # file exists        
                        if datetime.datetime.fromtimestamp(os.path.getmtime(run_directory+'/QSM-kinematics-data.npz' )) < datetime.datetime.fromtimestamp(os.path.getmtime(kinematics_file)):
                            # *.npz file is older than source *.t file - reloading is required
                            reloading_required = True
                            
                        # even if we won't use it, read the file now to determine at what time T0 it is (which cycle)
                        Q = np.load(run_directory+'/QSM-kinematics-data.npz')
                        
                        if "size_t" in Q.keys():
                            size_t_old = Q['size_t']
                        else:
                            size_t_old = 0
                            
                        if size_t != size_t_old:
                            reloading_required = True
                        
                        # check if same T0 and nt is used
                        if abs(Q['T0']-T0) > 1.0e-10 or Q['kinematics'].shape[0] != nt : 
                            # a different T0 was used - reloading required
                            reloading_required = True


                if not optimized_loading or reloading_required:
                    # load kinematics data by means of load_t_file function from insect_tools library
                    kinematics_cfd = insect_tools.load_t_file(kinematics_file, interp=True, time_out=self.timeline+T0, 
                                                              verbose=verbose, optimized_loading=True)
                    
                    if optimized_loading:
                        # when optimization is used, save converted data to binary npz file to read that next time.
                        np.savez(run_directory+'/QSM-kinematics-data.npz', kinematics=kinematics_cfd, T0=T0, size_t=size_t)                
                else:
                    # use optimized loading
                    if verbose:
                        print('read_CFD_data: Optimized reading from pre-converted *.npz file')       
                    
                    Q = np.load(run_directory+'/QSM-kinematics-data.npz')
                    kinematics_cfd  = Q['kinematics']     
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                # Body velocity 
                self.u_infty_g[ii,0] = diff1( kinematics_cfd[:, 1].copy(), dt )
                self.u_infty_g[ii,1] = diff1( kinematics_cfd[:, 2].copy(), dt )
                self.u_infty_g[ii,2] = diff1( kinematics_cfd[:, 3].copy(), dt )                

                self.psi[ii]   = kinematics_cfd[:, 4].copy()
                self.beta[ii]  = kinematics_cfd[:, 5].copy()
                self.gamma[ii] = kinematics_cfd[:, 6].copy()
                self.eta[ii]   = kinematics_cfd[:, 7].copy()

                if self.wing == "right":
                    # right wing
                    self.alpha[ii] = kinematics_cfd[:,11].copy()
                    self.phi[ii]   = kinematics_cfd[:,12].copy()
                    self.theta[ii] = kinematics_cfd[:,13].copy()

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
                    self.alpha[ii] = kinematics_cfd[:,  8].copy()
                    self.phi[ii]   = kinematics_cfd[:,  9].copy()
                    self.theta[ii] = kinematics_cfd[:, 10].copy()

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
                    self.alpha[ii] = kinematics_cfd[:, 29].copy()
                    self.phi[ii]   = kinematics_cfd[:, 30].copy()
                    self.theta[ii] = kinematics_cfd[:, 31].copy()

                elif self.wing == "left2":
                    # left (hind) wing, second wing pair
                    self.alpha[ii] = kinematics_cfd[:, 26].copy()
                    self.phi[ii]   = kinematics_cfd[:, 27].copy()
                    self.theta[ii] = kinematics_cfd[:, 28].copy()

                else:
                    raise ValueError("Wing code unknown (left/right/left2/right2)")

            elif ".ini" in kinematics_file:
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # MODE _2_
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # load kinematics data by means of eval_angles_kinematics_file function from insect_tools library
                timeline, phi, alpha, theta = insect_tools.eval_angles_kinematics_file(fname=kinematics_file, time=self.timeline)

                # convert to rad
                self.phi[ii]   = np.radians(phi)
                self.alpha[ii] = np.radians(alpha)
                self.theta[ii] = np.radians(theta)

                if yawpitchroll0 is None:
                    yawpitchroll0 = wt.get_ini_parameter(params_file, 'Insects', 'yawpitchroll_0', vector=True)
                if eta_stroke is None:
                    eta_stroke    = wt.get_ini_parameter(params_file, 'Insects', 'eta0', dtype=float )

                # yaw pitch roll and stroke plane angle are all constant in time
                self.psi[ii]   = np.zeros_like(self.phi[ii]) + yawpitchroll0[2]*(np.pi/180.0)
                self.beta[ii]  = np.zeros_like(self.phi[ii]) + yawpitchroll0[1]*(np.pi/180.0)
                self.gamma[ii] = np.zeros_like(self.phi[ii]) + yawpitchroll0[0]*(np.pi/180.0)
                self.eta[ii]   = np.zeros_like(self.phi[ii]) + eta_stroke*(np.pi/180.0)

            else:
                raise ValueError("Parsing kinematics is possible with kinematics.t, kinematics.ini or manually passing the angles.")
                


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Done with the input, now parsing.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.alpha_dt[ii] = diff1( self.alpha[ii], dt )
        self.phi_dt[ii]   = diff1( self.phi[ii], dt )
        self.theta_dt[ii] = diff1( self.theta[ii], dt )

        self.alpha_dtdt[ii] = diff2( self.alpha[ii], dt )
        self.phi_dtdt[ii]   = diff2( self.phi[ii], dt )
        self.theta_dtdt[ii] = diff2( self.theta[ii], dt )

        # define the many rotation matrices used in the code
        self.M_g2b[ii,:,:] = insect_tools.get_many_M_g2b(self.psi[ii], self.beta[ii], self.gamma[ii])
        self.M_s2w[ii,:,:] = insect_tools.get_many_M_s2w(self.alpha[ii], self.theta[ii], self.phi[ii], side)
        self.M_b2s[ii,:,:] = insect_tools.get_many_M_b2s(self.eta[ii], side)
        self.M_b2w[ii,:,:] = insect_tools.get_many_M_b2w(self.alpha[ii], self.theta[ii], self.phi[ii], self.eta[ii], side)
        self.M_g2w[ii,:,:] = self.M_b2w[ii,:,:] @ self.M_g2b[ii,:,:]

        self.M_w2s[ii,:,:] = self.M_s2w[ii,:,:].transpose(0, 2, 1)
        self.M_s2b[ii,:,:] = self.M_b2s[ii,:,:].transpose(0, 2, 1)
        self.M_w2b[ii,:,:] = self.M_b2w[ii,:,:].transpose(0, 2, 1)
        self.M_b2g[ii,:,:] = self.M_g2b[ii,:,:].transpose(0, 2, 1)
        self.M_w2g[ii,:,:] = self.M_g2w[ii,:,:].transpose(0, 2, 1)

        # WING angular velocities in various frames        
        rot_wing_s = np.zeros_like( self.rot_wing_w )
        if side == 'left':
            rot_wing_s[ii,0] = self.phi_dt[ii]-np.sin(self.theta[ii])*self.alpha_dt[ii]
            rot_wing_s[ii,1] = np.cos(self.phi[ii])*np.cos(self.theta[ii])*self.alpha_dt[ii]-np.sin(self.phi[ii])*self.theta_dt[ii]
            rot_wing_s[ii,2] = np.sin(self.phi[ii])*np.cos(self.theta[ii])*self.alpha_dt[ii]+np.cos(self.phi[ii])*self.theta_dt[ii]

        elif side == 'right':
            rot_wing_s[ii,0] = -self.phi_dt[ii]-np.sin(self.theta[ii])*(-self.alpha_dt[ii])
            rot_wing_s[ii,1] = np.cos(-self.phi[ii])*np.cos(self.theta[ii])*(-self.alpha_dt[ii])-np.sin(-self.phi[ii])*self.theta_dt[ii]
            rot_wing_s[ii,2] = np.sin(-self.phi[ii])*np.cos(self.theta[ii])*(-self.alpha_dt[ii])+np.cos(-self.phi[ii])*self.theta_dt[ii]
        
        self.rot_wing_w[ii,:] = apply_rotations_to_vectors( self.M_s2w[ii,:,:], rot_wing_s[ii,:])
        self.rot_wing_b[ii,:] = apply_rotations_to_vectors( self.M_s2b[ii,:,:], rot_wing_s[ii,:])
        self.rot_wing_g[ii,:] = apply_rotations_to_vectors( self.M_b2g[ii,:,:], self.rot_wing_b[ii,:])
            
        # The planar angular velocity {𝛀(φ,Θ)} comes from the decomposition of the motion
        # into 'translational' and rotational components, with the rotational component beig defined as
        # 'new' definition (not setting alpha_dt = 0)
        self.planar_rot_wing_w[ii,:] = self.rot_wing_w[ii,:].copy()
        self.planar_rot_wing_w[ii,1] = 0.0 # set y-component to zero        
        self.planar_rot_wing_g[ii,:] = apply_rotations_to_vectors( self.M_w2g[ii,:,:], self.planar_rot_wing_w[ii,:])
        
        # if it is not passed, we try to extract the insects cruising speed from the PARAMS file
        if u_infty_g is None:
            u_infty_g = -np.asarray( wt.get_ini_parameter(params_file, 'ACM-new', 'u_mean_set', vector=True) ) # note sign change (wind vs body)
        # insects cruising speed (including body velocity and negative wind velocity)
        self.u_infty_g[ii, :] += np.matlib.repmat( u_infty_g, nt, 1)
        # flight velocity in the wing system 
        self.u_infty_w[ii, :] = apply_rotations_to_vectors(self.M_g2w[ii,:,:], self.u_infty_g[ii,:])
       
        # these are all unit vectors of the wing
        # ey_wing_g coincides with the tip only if R is normalized (usually the case)
        self.ex_wing_g[ii, :] = apply_rotations_to_vectors(self.M_w2g[ii,:,:], np.matlib.repmat( np.asarray([1,0,0]), nt, 1))
        self.ey_wing_g[ii, :] = apply_rotations_to_vectors(self.M_w2g[ii,:,:], np.matlib.repmat( np.asarray([0,1,0]), nt, 1))
        self.ez_wing_g[ii, :] = apply_rotations_to_vectors(self.M_w2g[ii,:,:], np.matlib.repmat( np.asarray([0,0,1]), nt, 1))

        # wing tip velocity
        self.u_tip_g[ii, :] = np.cross(self.rot_wing_g[ii, :], self.ey_wing_g[ii, :]) + self.u_infty_g[ii, :]
        self.u_tip_w[ii]    = apply_rotations_to_vectors(self.M_g2w[ii,:,:], self.u_tip_g[ii,:])
        # and its magnitude
        self.u_tip_mag[ii]  = np.linalg.norm(self.u_tip_g[ii, :], axis=1)
        
        # drag unit vector
        for a in range(3):
            self.e_drag_g[ii,a] = -self.u_tip_g[ii,a] / self.u_tip_mag[ii]
                
        # lift unit vector
        self.e_lift_g[ii,:] = np.cross(-self.e_drag_g[ii, :], self.ey_wing_g[ii, :])
        n = np.linalg.norm(self.e_lift_g[ii,:], axis=1)
        for a in range(3):
            self.e_lift_g[ii,a] /= n
        
        # angular velocity norm (without feathering)
        self.planar_rot_wing_mag[ii] = np.linalg.norm( self.planar_rot_wing_w[ii, :], axis=1 )
        
        # angle of attack
        v = self.ex_wing_g[ii,0]*(-self.e_drag_g[ii,0]) + self.ex_wing_g[ii,1]*(-self.e_drag_g[ii,1]) + self.ex_wing_g[ii,2]*(-self.e_drag_g[ii,2])
        self.AoA[ii] = np.arccos(v)        
        # self.AoA[ii] = self.alpha[ii]
        
        # calculation of wingtip acceleration and angular acceleration in wing reference frame
        # a second loop over time is required, because we first need to compute ang. vel. then diff it here.
        self.a_tip_g[ii]        = diff1( self.u_tip_g[ii], dt )
        self.rot_acc_wing_g[ii] = diff1( self.rot_wing_g[ii], dt )

        # transform to wing system (required for the QSM model terms)
        self.a_tip_w[ii,:]        = apply_rotations_to_vectors(self.M_g2w[ii,:,:], self.a_tip_g[ii,:])
        self.rot_acc_wing_w[ii,:] = apply_rotations_to_vectors(self.M_g2w[ii,:,:], self.rot_acc_wing_g[ii,:])
   

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
            
            
        # def find_roots_vec(x, y):
        #     # Indices where sign change occurs
        #     idx = np.where(np.signbit(y[1:]) != np.signbit(y[:-1]))[0]        
        #     # Linear interpolation for each root
        #     roots = x[idx] - y[idx] * (x[idx+1] - x[idx]) / (y[idx+1] - y[idx])
        #     return roots
        
        # ipeaks = find_roots_vec(self.timeline, self.phi_dt[ii])
        # qty_to_use = self.phi_dt
            
        if self.reversal_detector == 'planar':
            qty_to_use = self.planar_rot_wing_mag #self.u_tip_mag
        elif self.reversal_detector == 'phi_dt':
            qty_to_use = np.abs(self.phi_dt)
        else:
            raise ValueError("Unknown reversal detector method: "+self.reversal_detector)
        
        # find minima in wingtip velocity magnitude. those, hopefully two, will be the reversals,
        # this is where the sign is flipped. We repeat the (periodic) signal to ensure we capture
        # peaks at t=0.0 and t=1.0. The distance between peaks is 3/4 * 1/2, so we think that the two half-strokes
        # occupy at most 3/8 and 5/8 of the complete cycle (maximum imbalance between up- and downstroke). This
        # filters out smaller peaks (in height) automatically, so we are left with the two most likely candidates.
        ipeaks, _ = scipy.signal.find_peaks( -1*np.hstack(  3*[qty_to_use[ii]] ), distance=3*self.nt/4/2)
        ipeaks -= self.nt # shift (skip 1st periodic image)
        
        # keep only peaks in the original signal domain (remove periodic "ghosts")
        ipeaks = ipeaks[ipeaks>=0]
        ipeaks = ipeaks[ipeaks<nt]

        # It should be two minima of velocity, if its not, then something weird happens in the kinematics.
        # We must then look for a different way to determine reversals or set it manually.
        if len(ipeaks) != 2 :
            plt.figure()
    #plt.plot( np.hstack([self.timeline, self.timeline+1, self.timeline+2]), -1*np.hstack(  (3*[self.u_tip_mag[ii]]) ) )
            plt.plot( np.hstack([self.timeline_all[ii], self.timeline_all[ii]+1, self.timeline_all[ii]+2]), 
                      np.hstack(  (3*[qty_to_use[ii]]) ) )
            plt.xlabel('timeline (repeated identical cycle 3 times)')
            plt.ylabel('u_tip_mag (wing=%s)' % (self.wing))
            plt.plot( self.timeline[ipeaks]+0, qty_to_use[ipeaks], 'ro')
            plt.plot( self.timeline[ipeaks]+1, qty_to_use[ipeaks], 'ro')
            plt.plot( self.timeline[ipeaks]+2, qty_to_use[ipeaks], 'ro')
            plt.title('Wing velocity minima detection: PROBLEM (more than 2 minima found)')

            insect_tools.indicate_strokes(tstroke=2.0)
            print(ii)
            raise ValueError("We found more than two reversals in the kinematics data...")

        sign_liftvector[ ipeaks[0]:ipeaks[1], : ] *= -1
        self.e_lift_g[ii] *= sign_liftvector
        # ipeaks = [0, int(nt/2)]
        # sign_liftvector[ ipeaks[0]:ipeaks[1], : ] *= -1
        # self.e_lift_g[ii] *= sign_liftvector

        # Convention: the mean lift vector in the body system should point upwards. The problem is that the code
        # sometimes identifies the first- and sometimes the second part of the stroke as downstroke.
        # This is no problem when training a model with a run: the sign of the lift coefficients will
        # simply be inverted. However, when using the trained model for prediction of a different kinematics
        # set, then it may identify the other half as downstroke - the resulting prediction is completely wrong,
        # because the sign of e_lift_g needs to be inverted.
        # Assuming the lift vectors ez_body component is positive is a convention - still, the training can invert
        # the sign of the coefficients should that be necessary in weird maneuvres when the insect is flying on its
        # back.
        self.e_lift_b[ii,:] = apply_rotations_to_vectors(self.M_g2b[ii,:,:], self.e_lift_g[ii,:])
            
        if np.mean(self.e_lift_b[ii,2]) < 0.0:
            self.e_lift_g[ii] *= -1

        # for indication in figures:
        self.T_reversals = self.timeline[ipeaks]
        if verbose:
            print("T_reversals=", self.T_reversals)

        #----------------------------------------------------------------------
        # kinematics figure
        #----------------------------------------------------------------------
        if plot:
            self.plot_kinematics()
            

    def print_force_coeefs(self):

            x0 = self.x0_forces

            # Cl and Cd definitions from Dickinson 1999
            print("CL = %f + %f*sin( deg2rad*(2.13*AoA - 7.20) ) " % (x0[0],x0[1]) )
            print("CD = %f + %f*cos( deg2rad*(2.04*AoA - 9.82) ) " % (x0[2],x0[3]) )
            print("Cam1=%f Cam2=%f Cam3=%f (linear accel, normal force) \nCam4=%f Cam5=%f Cam6=%f (angular accel, normal force)\nCam7=%f Cam8=%f (ax, az: tangential force)" % (x0[5], x0[6], x0[8], x0[9], x0[10], x0[11],x0[12],x0[13]) )
            print("C_rot=%f C_rd=%f" % (x0[4], x0[7]))


    def read_CFD_data(self, run_directory, T0, i0=0, verbose=True, optimized_loading=True ):
        """
        Read in CFD data (forces, moments, power) for a single run. Starting point in
        time is T0, we read data evenly sampled in time until T0+1.0 (interpolation is applied to CFD data).
        If multiple runs are used for fitting, the data are concatenated, in which case the i0 parameter is used.
        The data are then stored as [i0:i0+nt] (python notation, so actually [i0:i0+nt-1]).
        
        The CFD data are read from *.t files in the simulation directory (run_directory). 
        
        If you want to use the QSM model on several CFD runs, it is easier to use parse_many_run_directorys,
        although that assumes the same T0 for all runs.
        
        
        """
        
        import datetime
        import os        
        
        nt = self.nt
        # index range to store the data to
        ii = np.arange(start=i0, stop=i0+nt-1+1 )

        # as a means of informing the user that they need to read CFD data before fitting (training)
        self.readAtLeastOneCFDrun = True

        if verbose:
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
        
        if verbose:
            print("CFD data       t=[%f, %f]" % (time_start, time_end))
            print("QSM model uses t=[%f, %f]" % (T0, T0+1.0))
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # size of the *.t file (to check if this changed)
        size_t = os.path.getsize(run_directory+'/forces_'+suffix+'.t')
            
        # optimized loading - convert to NPZ on first call, then read that
        reloading_required, Q = False, None
        if optimized_loading:
            if not os.path.isfile( run_directory+'/QSM-CFD-data.npz' ):
                # file does not exist - reloading is required.
                reloading_required = True
            else:
                # file exists        
                if datetime.datetime.fromtimestamp(os.path.getmtime(run_directory+'/QSM-CFD-data.npz' )) < datetime.datetime.fromtimestamp(os.path.getmtime(run_directory+'/forces_'+suffix+'.t')):
                    # *.npz file is older than source *.t file - reloading is required
                    reloading_required = True
                    
                # even if we won't use it, read the file now to determine at what time T0 it is (which cycle)
                Q = np.load(run_directory+'/QSM-CFD-data.npz')
                
                if "size_t" in Q.keys():
                    size_t_old = Q['size_t']
                else:
                    size_t_old = 0
                    
                if size_t != size_t_old:
                    reloading_required = True
                
                # check if same T0 and nt is used
                if abs(Q['T0']-T0) > 1.0e-10 or Q['forces_CFD'].shape[0] != nt or Q['moments_CFD'].shape[0] != nt : 
                    # a different T0 was used - reloading required
                    reloading_required = True


        if not optimized_loading or reloading_required:
            # read in data from desired cycle (hence the shift by T0)
            # NOTE that load_t_file can remove outliers in the data, which turned out very useful for the
            # musca domestica data (which have large jumps exactly at the reversals)
            forces_CFD  = insect_tools.load_t_file(run_directory+'/forces_'+suffix+'.t', interp=True, 
                                                   time_out=self.timeline+T0, remove_outliers=True, 
                                                   verbose=verbose, optimized_loading=optimized_loading)
            
            moments_CFD = insect_tools.load_t_file(run_directory+'/moments_'+suffix+'.t', interp=True, 
                                                   time_out=self.timeline+T0, remove_outliers=True, 
                                                   verbose=verbose, optimized_loading=optimized_loading)
            
            if optimized_loading:
                # when optimization is used, save converted data to binary npz file to read that next time.
                np.savez(run_directory+'/QSM-CFD-data.npz', forces_CFD=forces_CFD, moments_CFD=moments_CFD, T0=T0, size_t=size_t)                
        else:
            # use optimized loading
            if verbose:
                print('read_CFD_data: Optimized reading from pre-converted *.npz file')       
            
            Q = np.load(run_directory+'/QSM-CFD-data.npz')
            forces_CFD  = Q['forces_CFD']
            moments_CFD = Q['moments_CFD']        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # copy to array
        self.F_CFD_g[ii,0:2+1] = forces_CFD[:, 1:3+1]
        self.M_CFD_g[ii,0:2+1] = moments_CFD[:, 1:3+1]

        # obtain CFD data in wing reference frame
        self.M_CFD_w[ii,:] = apply_rotations_to_vectors(self.M_g2w[ii,:,:], self.M_CFD_g[ii,:])
        self.F_CFD_w[ii,:] = apply_rotations_to_vectors(self.M_g2w[ii,:,:], self.F_CFD_g[ii,:])
        
        # aerodynamic power. Can be read from t-file or computed from the dot-product of moment and angular
        # velocity. This latter is done here: we do not store the power for each wing separately in the CFD run, and hence
        # computing it here is the better choice if possibly more than one wing has been simulated.·
        self.P_CFD[ii] = -(  np.sum( self.M_CFD_w[ii,:]*self.rot_wing_w[ii,:], axis=1 ) )

    def evaluate_QSM_model(self, plot=False):
        """
        This function is a wrapper for FIT_TO_CFD. It evaluates the QSM model with a given set of
        previously determined coefficients. You need to parse kinematics data before you can call this
        function.
        """
        # first the forces - otherwise we cannot compute the moments
        self.evaluate_forces(self.x0_forces, training=False)
        # compute moments (with constant lever assumption)
        self.evaluate_moments(self.x0_moments, training=False)
        # compute power (with constant lever assumption)
        self.evaluate_power(self.x0_power, training=False)
        
        if plot:
            self.plot_dynamics()
        
    def evaluate_forces(self, x0, training=False):
        """
        Evaluate QSM force model with the current set of parameters x0. 
        """
        
        # unpack coefficients from parameter vector
        Cl, Cd, Crot, Cam1, Cam2, Crd, Cam3, Cam4, Cam5, Cam6, Cam7, Cam8, Cam9 = self.unpack_parameters(x0)

        rho = 1.0 # for future work, can also be set to 1.0 simply
       
        if np.max(self.S2-1.0) < 1.0e-10:
            import warnings
            warnings.warn("""We try to evaluate the QSM model, but the S2 (shape function) seems to be
                             all ones. Probably you did not setup the wing shape before evaluating the model,
                             please do so using the function QSM.setup_wing_shape(). 
                             Alternatively, you can manually set QSM.S2 to a desired value (not recommended).""")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # lift/drag forces
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Ellingtons lift/drag forces, following Cai et al 2021. Note many other papers dealt with hovering
        # flight, which features zero cruising speed. Cai et al is a notable exception. Instead of the more
        # common planar_rot_wing_mag, he used u_tip_mag, which includes the flight velocity
        # and thus delivers nonzero values even for still wings. Cai integrated over the blades, and each blade
        # had the correct velocity. This can be done analytically, as we do here, see my lecture notes.
        if self.model_terms[0] is True:
            if self.ellington_type == 'utip':
                self.Ftc_mag = 0.5*rho*Cl*(self.u_tip_mag**2)*self.S2
                self.Ftd_mag = 0.5*rho*Cd*(self.u_tip_mag**2)*self.S2
                
            elif self.ellington_type == 'rot':
                self.Ftc_mag = 0.5*rho*Cl*(self.planar_rot_wing_mag**2)*self.S2
                self.Ftd_mag = 0.5*rho*Cd*(self.planar_rot_wing_mag**2)*self.S2
                
            elif self.ellington_type == 'ABC':
                # Using the bumblebee simulations at various u_infty values, the best choice for ellington_type is 'utip'.
                # Despite the fact that a similar result (ABC) is presented in Han et al 2017 (An aerodynamic model for insect
                # flapping wings in forward flight)
                A = self.planar_rot_wing_mag**2
                B = 2.0*(self.u_infty_w[:,2]*self.rot_wing_w[:,0]-self.u_infty_w[:,0]*self.rot_wing_w[:,2])
                C = self.u_infty_w[:,0]**2+self.u_infty_w[:,1]**2+self.u_infty_w[:,2]**2

                self.Ftc_mag = 0.5*rho*Cl*( self.S2*A + self.S1*B + self.S0*C )
                self.Ftd_mag = 0.5*rho*Cd*( self.S2*A + self.S1*B + self.S0*C )
            else:
                raise ValueError("The value ellington_type=%s is unkown" % (self.ellington_type))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # rotational forces
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # These formulations differ slightly from the above classical ones, because we include the body velocity
        # and use the angular velocity instead of alpha, alpha_dt.
        # Sane's original rotational force, "discovered" in Dickinson1999 and later modeled by Sane2002
        if self.model_terms[1] is True:
            # if ellington_type == 'rot':
            #     self.Frc_mag = rho*Crot*self.planar_rot_wing_mag*self.rot_wing_w[:,1]*self.S_RC # Nakata et al. 2015, Eqn. 2.6c
            # else:
            self.Frc_mag = rho*Crot*self.u_tip_mag*self.rot_wing_w[:,1]*self.S_RC # Nakata et al. 2015, Eqn. 2.6c


        # Rotational drag: \cite{Cai2021}. The fact that the wing rotates around its rotation axis, which is the $y$ component of the angular velocity
        # (Which Cai identifies as $\dot{\alpha}$, even though this is only an approximation) induces a net non-zero velocity component normal
        # to the surface of the wing. In the opposite direction appears thus a drag force, and this is called rotaional drag. It would be
        # zero if leading- and trailing edge were symmetrical. Cai in general assumes all QSM forces are perpendicular to the wing, an
        # approximation he introduces but does not explain well. In their reference data, the force is indeed very normal to the wing.
        # However, why neglecting the non-normal part, unless very helpful?
        # It is however correct that Cai says: as the Ellington terms do only include the velocity of the blade point on the
        # rotation axis (the point $(0,r,0)^T$), the rotation around that very axis ($y$) is not included in the traditional term.
        if self.model_terms[2] is True:
            self.Frd_mag = (-1/2)*rho*Crd*self.S_RD*np.abs(self.rot_wing_w[:,1])*self.rot_wing_w[:,1] # Cai et al. 2021, Eqn 2.13

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
        if self.model_terms[3] is True:
            if self.AM_model == 'Full 6DOF':
                self.Fam_z_mag = rho*(Cam1*self.a_tip_w[:, 0] + Cam2*self.a_tip_w[:, 1] + Cam3*self.a_tip_w[:, 2] + \
                                      Cam4*self.rot_acc_wing_w[:, 0] + Cam5*self.rot_acc_wing_w[:, 1] + Cam6*self.rot_acc_wing_w[:, 2] )   
                    
            elif self.AM_model == '6DOF - rot':
                self.Fam_z_mag = rho*(Cam1*self.rot_wing_w[:, 0] + Cam2*self.rot_wing_w[:, 1] + Cam3*self.rot_wing_w[:, 2] + \
                                      Cam4*self.rot_acc_wing_w[:, 0] + Cam5*self.rot_acc_wing_w[:, 1] + Cam6*self.rot_acc_wing_w[:, 2] )    
                    
            elif self.AM_model == '1DOF':
                self.Fam_z_mag = rho*(Cam3*self.a_tip_w[:, 2])
                
            elif self.AM_model == '1DOF scaled':
                self.Fam_z_mag = rho*(self.S_AM1*Cam3*self.a_tip_w[:, 2])   
                
            elif self.AM_model == '2DOF theoretical':
                self.Fam_z_mag = rho*(Cam3*self.a_tip_w[:, 2] + Cam5*self.rot_acc_wing_w[:, 1])
                
            elif self.AM_model == '2DOF theoretical scaled1':
                self.Fam_z_mag = rho*(Cam3*self.S_AM1*self.a_tip_w[:, 2] + Cam5*self.rot_acc_wing_w[:, 1])
                
            elif self.AM_model == '2DOF theoretical scaled2':
                self.Fam_z_mag = rho*(Cam3*self.S_AM1*self.a_tip_w[:, 2] + self.S_AM2*Cam5*self.rot_acc_wing_w[:, 1])
                
            elif self.AM_model == '2DOF best scaled2':
                self.Fam_z_mag = rho*(Cam3*self.S_AM1*self.a_tip_w[:, 2] + self.S_AM2*Cam4*self.rot_acc_wing_w[:, 0])                
            
            elif self.AM_model == '2DOF best scaled1':
                self.Fam_z_mag = rho*(Cam3*self.S_AM1*self.a_tip_w[:, 2] + Cam4*self.rot_acc_wing_w[:, 0])                
            
            elif self.AM_model == '2DOF best':
                self.Fam_z_mag = rho*(Cam3*self.a_tip_w[:, 2] + Cam4*self.rot_acc_wing_w[:, 0])   
                
            elif self.AM_model == 'Full 6DOF scaled':
                self.Fam_z_mag = rho*(self.S_AM1*Cam1*self.a_tip_w[:, 0] + self.S_AM1*Cam2*self.a_tip_w[:, 1] + self.S_AM1*Cam3*self.a_tip_w[:, 2] + \
                                      self.S_AM2*Cam4*self.rot_acc_wing_w[:, 0] + self.S_AM2*Cam5*self.rot_acc_wing_w[:, 1] + self.S_AM2*Cam6*self.rot_acc_wing_w[:, 2] )                   
            else:
                raise ValueError("unknown AM model")

        # *Tangential* added mass forces, like discussed in VanVeen2022, 2.1.1.
        # We assume however a more general form, which includes the components of the acceleration.
        # We do not include the acceleration in y-direction, even though it significantly reduces the error.
        # The reason is that apparently, this form has overlap with the lift+drag forces from the Ellington model
        # and thus the resulting model becomes harder to interpret.
        if self.model_terms[4] is True:
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
            self.F_QSM_g[:, k] = self.Ftc[:, k] + self.Ftd[:, k] + self.Frc[:, k] + self.Fam[:, k] + self.Fam2[:, k] + self.Frd[:, k]

        # QSM forces in wing system:
        if not training:
            self.F_QSM_w = apply_rotations_to_vectors(self.M_g2w, self.F_QSM_g)
        
    def evaluate_moments(self, x0, training=False):
        """
        Evaluate QSM moments model with the current set of parameters (stored in self.x0_moments).
        Requires to call evaluate_forces before (can't compute moments without forces)
        """
        # here we define the the QSM moments as: M_QSM = [ C_lever_x_w*Fz_QSM_w, -C_lever_x_w*Fz_QSM_w, C_lever_x_w*Fy_QSM_w - C_lever_y_w*F_x_QSM_w ]
        # where C_lever_x_w and C_lever_y_w correspond to the spanwise and the chordwise locations of the lever in the wing reference frame.
        # vector form: C_lever_w = [C_lever_x_w, C_lever_y_w, 0]
        C_lever_x_w = x0[0]
        C_lever_y_w = x0[1]

        # moment in wing reference frame
        self.M_QSM_w[:,0] =  C_lever_y_w*self.F_QSM_w[:, 2]
        self.M_QSM_w[:,1] = -C_lever_x_w*self.F_QSM_w[:, 2]
        self.M_QSM_w[:,2] =  C_lever_x_w*self.F_QSM_w[:, 1] - C_lever_y_w*self.F_QSM_w[:, 0]
        
        if not training:
            # compute QSM moment in global reference frame
            self.M_QSM_g = apply_rotations_to_vectors(self.M_w2g, self.M_QSM_w)
        
        
    def evaluate_power(self, x0, training=False):
        """
        Evaluate QSM power model with the current set of parameters (stored in self.x0_power)
        """
        
        # here we define the the QSM moments as: M_QSM = [ C_lever_x_w_power*Fz_QSM_w, -C_lever_x_w_power*Fz_QSM_w, C_lever_x_w_power*Fy_QSM_w - C_lever_y_w_power*F_x_QSM_w ]
        # where C_lever_x_w_power and C_lever_y_w_power correspond to the spanwise and the chordwise locations of the lever in the wing reference frame.
        # vector form: C_lever_w = [C_lever_x_w_power, C_lever_y_w_power, 0]

        C_lever_x_w_power = x0[0]
        C_lever_y_w_power = x0[1]

        Mx_QSM_w_power =  C_lever_y_w_power*self.F_QSM_w[:, 2]
        My_QSM_w_power = -C_lever_x_w_power*self.F_QSM_w[:, 2]
        Mz_QSM_w_power =  C_lever_x_w_power*self.F_QSM_w[:, 1] - C_lever_y_w_power*self.F_QSM_w[:, 0]

        # power using the moments (need to call evaluate_moments first)
        # this is the optimal lever for moment computation
        if not training:
            self.P_QSM_nonoptimized = -(self.M_QSM_w[:,0]*self.rot_wing_w[:, 0]
                                      + self.M_QSM_w[:,1]*self.rot_wing_w[:, 1]
                                      + self.M_QSM_w[:,2]*self.rot_wing_w[:, 2])
        
        # best estimate of the power using a constant lever optimized for power prediction.
        self.P_QSM = -(Mx_QSM_w_power*self.rot_wing_w[:, 0]
                     + My_QSM_w_power*self.rot_wing_w[:, 1]
                     + Mz_QSM_w_power*self.rot_wing_w[:, 2])
        
        
    def unpack_parameters(self, x0, AoA=None):
        deg2rad = np.pi/180.0
        rad2deg = 180.0/np.pi
        
        if AoA is None:
            AoA = self.AoA

        if self.model_CL_CD == "Dickinson":
            # Cl and Cd definitions from Dickinson 1999
            AoA = rad2deg*AoA
            Cl   = x0[0] + x0[1]*np.sin( deg2rad*(2.13*AoA - 7.20) )
            Cd   = x0[2] + x0[3]*np.cos( deg2rad*(2.04*AoA - 9.82) )
        elif self.model_CL_CD == "Nakata":
            # this is what nakata proposed:
            Cl   = x0[0]*(AoA**3 - AoA**2 * np.pi/2) + x0[1]*(AoA**2 - AoA * np.pi/2)
            Cd   = x0[2]*np.cos( AoA )**2  + x0[3]*np.sin( AoA )**2
        elif self.model_CL_CD == 'Polhamus':
            # see J-S Han et al Bioinspr Biomim 12 2017 036004
            Cl   = x0[0]*np.sin( AoA )*(np.cos( AoA )**2) + x0[1]*(np.sin( AoA )**2)*np.cos( AoA )
            Cd   = x0[2]*(np.sin( AoA )**2)*(np.cos( AoA )) + x0[3]*(np.sin( AoA )**3)
        
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

        return Cl, Cd, Crot, Cam1, Cam2, Crd, Cam3, Cam4, Cam5, Cam6, Cam7, Cam8, Cam9 # currently CAM9 is unused!



    def fit_to_CFD(self, optimize=True, plot=True, N_trials=1, verbose=True):
        """
        Train the QSM model with one/many CFD run(s). 
        ------------------
        
        This works only if you have initialized
            * the kinematics with parse_kinematics_file (or parse_many_run_directories)
            * read the CFD data with read_CFD_data (or parse_many_run_directories)
            * the wing shape with setup_wing_shape
        before calling this routine.

        We train the QSM model to a single wing currently.

        The optimized coefficients are stored in the QSM object.

        model_terms: [use_ellington_liftdrag, use_sane_rotforce, use_rotationalDrag, use_addedmass_normal, use_addedmass_tangential]

        optimize: if True, we train the QSM model with previously read CFD data.
                  if False, this function evaluates the model with the parsed kinematics, with given QSM model coefficients
        """
        
        if verbose:
            print('~~~~~~~~~~~~~~~Model training starting~~~~~~~~~~~~~~~~~~~~~')

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
            

        #%% force optimization        
        def compute_K_forces( F_QSM_g, F_CFD_g ):
            # compute quality metric (relative L2 error)
            K_forces     = norm(F_QSM_g[:,0]-F_CFD_g[:,0]) + norm(F_QSM_g[:,2]-F_CFD_g[:,2])
            K_forces_den = norm(F_CFD_g[:,0]) + norm(F_CFD_g[:,2])
            # normalization
            if K_forces_den >= 1e-16:
                K_forces /= K_forces_den
                
            return K_forces
        
        # cost function which tells us how far off our QSM values are from the CFD ones for the forces
        def cost_forces( x, self ):
            # evaluate forces with current set of parameters.
            # Training=True skips computing forces in wing system which is not useful for the training (speed-up)
            self.evaluate_forces(x, training=True)

            # compute quality metric (relative L2 error)
            return compute_K_forces( self.F_QSM_g, self.F_CFD_g)

        #----------------------------------------------------------------------
        # TRAINING: find optimal set of coefficients
        #----------------------------------------------------------------------
        if optimize:
            # as a means of informing the user that they need to read CFD data before fitting (training):
            if not self.readAtLeastOneCFDrun:
                raise ValueError("You need to read CFD before you can fit the model to it. call QSM.read_CFD_data")

            start = time.time()
            bounds = 15*[(-1000, 1000)]
            K_forces = 9e9

            # optimize N_trials times from a different initial guess, use best solution found
            # NOTE: tests indicate the system always finds the same solution, so this could
            # be omitted. Kept for safety - we do less likely get stuck in local minima this way
            for i_trial in range(N_trials):
                x0_forces    = np.random.rand(15)
                optimization = opt.minimize(cost_forces, args=(self), bounds=bounds, x0=x0_forces)
                x0_forces    = optimization.x
                
                # for readability, remove unused coefficients
                if not self.model_terms[0]:
                    x0_forces[0:3+1] = np.nan
                if not self.model_terms[1]:
                    x0_forces[4] = np.nan
                if not self.model_terms[2]:
                    x0_forces[7] = np.nan
                if not self.model_terms[3]:
                    x0_forces[ [5,6,8, 9,10,11] ] = np.nan
                if not self.model_terms[4]:
                    x0_forces[ [12,13] ] = np.nan   
                # unused:
                x0_forces[14] = np.nan

                if optimization.fun < K_forces:
                    K_forces = optimization.fun
                    x0_best = x0_forces
                if verbose:
                    print( 'Trial %i/%i K=%2.3f N_evals=%i \nx0=' % (i_trial+1, N_trials, optimization.fun, optimization.nfev), np.round(x0_forces, 3))

            self.x0_forces = x0_best
            self.K_forces  = K_forces

            if verbose:
                print('Completed in:', round(time.time() - start, 4), 'seconds')
        
        # final evaluation (called even without optimization)
        # global fit quality
        self.evaluate_forces(self.x0_forces, training=False)
        self.K_forces = compute_K_forces( self.F_QSM_g, self.F_CFD_g)
                
        # evaluate, for each run, the model quality (the local K_forces for each run)
        for irun in range( self.nruns ):
            i0 = irun*self.nt
            jj = np.arange(start=i0, stop=i0+self.nt-1+1 )       
            # error (K) for this CFD run:                         
            self.K_forces_individual[irun] = compute_K_forces( self.F_QSM_g[jj], self.F_CFD_g[jj] )
        
        if verbose:
            print("K_forces= %2.2f" % (self.K_forces))
            
            
        

        #%% moments optimization        
        def compute_K_moments(M_QSM_w, M_CFD_w):
            K_moments = norm(M_QSM_w[:,0]-M_CFD_w[:,0]) + norm(M_QSM_w[:,1]-M_CFD_w[:,1]) + norm(M_QSM_w[:,2]-M_CFD_w[:,2])
            K_moments_den = norm(M_CFD_w[:,0]) + norm(M_CFD_w[:,1]) + norm(M_CFD_w[:,2])
            # normalization
            if K_moments_den >= 1.0e-16:
                K_moments /= K_moments_den   
                
            return K_moments

        # cost_moments is defined in terms of the moments. this function will be optimized to find the lever (coordinates) that best matches (match) the QSM moments to their CFD counterparts
        def cost_moments(x, self):
            # compute moments (with current parameters)            
            self.evaluate_moments(x, training=True)
            # evaluate accuracy
            K_moments = compute_K_moments( self.M_QSM_w, self.M_CFD_w )
            return K_moments

        # moment optimization
        if optimize:
            x0_moments = [1.0, 1.0]
            bounds = [(-6, 6), (-6, 6)]

            start = time.time()
            optimization = opt.minimize(cost_moments, args=(self), bounds=bounds, x0=x0_moments)
            if verbose:
                print('Completed in:', round(time.time() - start, 4), 'seconds')

            self.x0_moments = optimization.x
            self.K_moments  = optimization.fun

            if verbose:
                print('x0_moments_optimized:', np.round(self.x0_moments, 5))
                

        # final evaluation
        self.evaluate_moments(self.x0_moments, training=False)
        # global approximation error (over all runs)
        self.K_moments = compute_K_moments( self.M_QSM_w, self.M_CFD_w )
        
        if verbose:
            print("K_moments= %2.2f" % (self.K_moments))
        
        # evaluate, for each run, the model quality (the local K_forces for each run)
        for irun in range( self.nruns ):
            i0 = irun*self.nt
            jj = np.arange(start=i0, stop=i0+self.nt-1+1 )       
            # error (K) for this CFD run:                         
            self.K_moments_individual[irun] = compute_K_moments( self.M_QSM_w[jj], self.M_CFD_w[jj] )
        

        #%% power optimization        
        def compute_K_power(P_QSM, P_CFD):
            K_power     = norm(P_QSM - P_CFD)
            K_power_den = norm(P_CFD)
            # normalization
            if K_power_den >= 1e-16:
                K_power /= K_power_den
                
            return K_power

        # cost_power is defined in terms of the moments and power. this function will be optimized to find the lever (coordinates) that best matches (match) the QSM power to its CFD counterpart
        def cost_power(x, self):
            # compute the power
            self.evaluate_power(x, training=True)
            # and evaluate its accuracy
            K_power = compute_K_power(self.P_QSM, self.P_CFD)
            return K_power

        # power optimization
        if optimize:
            x0_power = [1.0, 1.0]
            bounds = [(-6, 6), (-6, 6)]

            start = time.time()
            optimization = opt.minimize(cost_power, args=(self), bounds=bounds, x0=x0_power)
            self.x0_power = optimization.x
            self.K_power = optimization.fun
            
            if verbose:
                print('Completed in:', round(time.time() - start, 4), 'seconds')
                print('x0_power:', np.round(self.x0_power, 5))

        # global approximation error (over all runs)
        self.evaluate_power(self.x0_power, training=False)
        self.K_power = compute_K_power(self.P_QSM, self.P_CFD)
        
        if verbose:
            print("K_power= %2.2f" % (self.K_power))
        
        # evaluate, for each run, the model quality (the local K_forces for each run)
        for irun in range( self.nruns ):
            i0 = irun*self.nt
            jj = np.arange(start=i0, stop=i0+self.nt-1+1 )       
            # error (K) for this CFD run:                         
            self.K_power_individual[irun] = compute_K_power( self.P_QSM[jj], self.P_CFD[jj] )

        if plot:
            self.plot_dynamics()


        if verbose:
            print('~~~~~~~~~~~~~~~Model training complete~~~~~~~~~~~~~~~~~~~~~')


    def setup_wing_shape(self, wingShape_file, verbose=True, i0=0, force_reload=False):
        """
        Specifiy the wing shape (here, in the form of the wing contour).
        Note the code can run without this information, as the influence of
        the wing contour can also be taken into account by the optimized model
        coefficients (optimized using a reference CFD run).

        Shape data is read from an INI file.
        """
        if verbose:
            print('Parsing wing contour: '+wingShape_file)

        if os.path.isfile(wingShape_file):
            xc, yc, area = insect_tools.wing_contour_from_file( wingShape_file )
            zc = np.zeros_like(xc)
        else:
            raise ValueError("Wing shape file %s not found!" % (wingShape_file))
        
        
        nt = self.nt
        # index range to store the data to
        ii = np.arange(start=i0, stop=i0+nt-1+1 )

        self.x_wingContour_w  = np.vstack([xc, yc, zc])
        self.x_wingContour_w  = np.transpose(self.x_wingContour_w)


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        version = 100
        version_file = 0
        
        reloading_required = False
        if not os.path.isfile(wingShape_file+'.npz') and not force_reload:
            reloading_required = True
        
        if os.path.isfile(wingShape_file+'.npz'):
            Q = np.load(wingShape_file+'.npz')
            
            if 'version' in Q.keys():
                version_file = Q['version']
            else:
                version_file = 0
        
        if version_file != version:
            reloading_required = True
                
        if reloading_required:            
            if verbose:
                print('Evaluating wing shape file (may be slow but will be fast next time!)')
            dx, dy = 1e-3, 1e-3
            # 1st index: x, 2nd index: y
            X, Y, mask = insect_tools.get_wing_membrane_grid(wingShape_file, dx, dy, return_1D_list=False)
            
            np.savez(wingShape_file+'.npz', X=X, Y=Y, dx=dx, dy=dy, mask=mask, version=version)       
            
        else:            
            if verbose:
                print('Wing grid read from pre-computed *.npz file. (much faster!)')        
            X, Y, mask, dx, dy = Q['X'], Q['Y'], Q['mask'], Q['dx'], Q['dy']            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         
        # chord array (size ny):
        C = np.sum(mask, axis=0)*dx
        # radius array (size ny):
        r = Y[0,:].copy()
        
        # position of the middle of the lollipop. In [Whitney2010], called "yh" (eqn 2.41 & figure 3b), 
        # in [Liu, Sun JBTB2018] called 'delta x' (eqn 3b)
        # in [Cai2021], represented as (f_LE + f_TE).
        X2 = X.copy()
        X2[ mask<0.99 ] = np.nan
        xh = np.nanmean( X2, axis=0)
        xh[ np.isnan(xh) ] = 0.0
        
        # plt.figure()
        # plt.plot( X[mask>0].flatten(), Y[mask>0].flatten(), 'k.')
        # plt.plot( xh, r, 'r-')
        # plt.plot( xh+0.5*C, r, 'c-')
        # plt.plot( xh-0.5*C, r, 'c-')
        # plt.axis('equal')        
        # raise

        # area moments (for Ellington term)
        self.S2[ii] = np.sum( C*r**2 )*dy
        self.S1[ii] = np.sum( C*r**1 )*dy
        self.S0[ii] = np.sum( C*r**0 )*dy
        
        # For Sanes rotational circulation:
        self.S_RC[ii] = np.sum( C**2 * r )*dy
        
        # For whitneys rotational drag:
        self.S_RD[ii] = np.sum( (X*mask) * np.abs(X*mask) ) * dx*dy

        # for Added mass:
        self.S_AM1[ii] = np.sum( r*C**2 )*dy
        self.S_AM2[ii] = np.sum( xh *  C**2 )*dy

        if any(np.isnan(self.S2)) or any(np.isnan(self.S1)) or any(np.isnan(self.S0)) or any(np.isnan(self.S_RC)) or any(np.isnan(self.S_RD)) or any(np.isnan(self.S_AM1)) or any(np.isnan(self.S_AM2)):
            raise ValueError("Wing shape setup has failed and computed NANs...")


    def plot_kinematics(self):
        ## FIGURE 1
        fig, axes = plt.subplots(3, 2, figsize = (15, 15))

        # angles
        ax = axes[0,0]
        ax.plot(self.timeline_all, np.degrees(self.phi), label='phi')
        ax.plot(self.timeline_all, np.degrees(self.alpha), label ='alpha')
        ax.plot(self.timeline_all, np.degrees(self.theta), label='theta')
        ax.plot(self.timeline_all, np.degrees(self.AoA), label='AoA', color = 'purple')
        ax.set_xlabel('t/T')
        ax.set_ylabel('(deg)')
        ax.legend()

        # time derivatives of angles
        ax = axes[0,1]
        ax.plot(self.timeline_all, self.phi_dt, label='phi_dt')
        ax.plot(self.timeline_all, self.alpha_dt, label='alpha_dt')
        ax.plot(self.timeline_all, self.theta_dt, label='theta_dt')

        if self.wing == "right" or self.wing == "right2" :
            ax.plot(self.timeline_all, np.sign(-self.alpha), 'k--', label='sign(alpha)', linewidth=0.5 )
        elif self.wing == "left" or self.wing == "left2" :
            ax.plot(self.timeline_all, np.sign(+self.alpha), 'k--', label='sign(alpha)', linewidth=0.5 )
        ax.set_xlabel('$t/T$')
        ax.legend()

        # u_wing_w (tip velocity in wing reference frame )
        ax = axes[1,0]
        
        ax.plot(self.timeline_all, self.u_tip_w[:, 0], label='utip_w_x')
        ax.plot(self.timeline_all, self.u_tip_w[:, 1], label='utip_w_y')
        ax.plot(self.timeline_all, self.u_tip_w[:, 2], label='utip_w_z')
        ax.plot(self.timeline_all, self.u_tip_mag, 'k--', label='utip_mag')

        ax.set_xlabel('t/T')
        ax.set_ylabel('[Rf]')
        ax.set_title('Tip velocity magnitude in wing reference frame = %2.2f' % (np.mean(self.u_tip_mag)))
        ax.legend()

        #a_wing_w (tip acceleration in wing reference frame )
        ax = axes[1,1]
        ax.plot(self.timeline_all, self.a_tip_w[:, 0], label='$\\dot{u}_{\\mathrm{wing},x}^{(w)}$')
        ax.plot(self.timeline_all, self.a_tip_w[:, 1], label='$\\dot{u}_{\\mathrm{wing},y}^{(w)}$')
        ax.plot(self.timeline_all, self.a_tip_w[:, 2], label='$\\dot{u}_{\\mathrm{wing},z}^{(w)}$')
        ax.set_xlabel('$t/T$')
        ax.set_ylabel('$Rf^2$')
        ax.set_title('Tip acceleration in wing reference frame')
        ax.legend()

        #rot_wing_w (tip velocity in wing reference frame )
        ax = axes[2,0]
        ax.plot(self.timeline_all, self.rot_wing_w[:, 0], label='$\\Omega_{\\mathrm{wing},x}^{(w)}$')
        ax.plot(self.timeline_all, self.rot_wing_w[:, 1], label='$\\Omega_{\\mathrm{wing},y}^{(w)}$')
        ax.plot(self.timeline_all, self.rot_wing_w[:, 2], label='$\\Omega_{\\mathrm{wing},z}^{(w)}$')
        ax.set_xlabel('$t/T$')
        ax.set_ylabel('rad/T')
        ax.set_title('Angular velocity in wing reference frame')
        ax.legend()

        #rot_acc_wing_w (angular acceleration in wing reference frame )
        ax = axes[2,1]
        ax.plot(self.timeline_all, self.rot_acc_wing_w[:, 0], label='$\\dot\\Omega_{\\mathrm{wing},x}^{(w)}$')
        ax.plot(self.timeline_all, self.rot_acc_wing_w[:, 1], label='$\\dot\\Omega_{\\mathrm{wing},y}^{(w)}$')
        ax.plot(self.timeline_all, self.rot_acc_wing_w[:, 2], label='$\\dot\\Omega_{\\mathrm{wing},z}^{(w)}$')

        ax.plot(self.timeline_all, np.sqrt(self.rot_acc_wing_w[:, 0]**2+self.rot_acc_wing_w[:, 1]**2+self.rot_acc_wing_w[:, 2]**2), 'k--', label='mag')
        ax.set_xlabel('$t/T$')
        ax.set_ylabel('[rad/T²]')
        ax.set_title('Angular acceleration in wing reference frame')
        ax.legend()

        plt.suptitle('Kinematics data')

        plt.tight_layout()
        plt.draw()

        for ax in axes.flatten():
            insect_tools.indicate_strokes(ax=ax, tstart=[self.T_reversals[0]], tstroke=2*(self.T_reversals[1]-self.T_reversals[0]) )
            
            
    def plot_dynamics(self):

        ##FIGURE 2
        fig, axes = plt.subplots(2, 2, figsize = (15, 10))

        #coefficients
        graphAoA = np.linspace(-9, 90, 100)*(np.pi/180)
        gCl, gCd, gCrot, gCam1, gCam2, gCrd, _, _,_,_, _,_,_  = self.unpack_parameters(self.x0_forces, graphAoA)
        axes[0, 0].plot(np.degrees(graphAoA), gCl, label='Cl', color='#0F95F1')
        axes[0, 0].plot(np.degrees(graphAoA), gCd, label='Cd', color='#F1AC0F')
        axes[0, 0].set_title('Lift and drag coeffficients')
        axes[0, 0].set_xlabel('AoA[°]')
        axes[0, 0].set_ylabel('[-]')
        axes[0, 0].legend(loc = 'upper right')

        #vertical forces
        axes[0, 1].plot(self.timeline_all, self.Ftc[:, 2], label = 'Vert. part of F_{TC} (Ellington1984 lift force)', color='gold')
        axes[0, 1].plot(self.timeline_all, self.Ftd[:, 2], label = 'Vert. part of F_{TD} (Ellington1984 drag force)', color='lightgreen')
        axes[0, 1].plot(self.timeline_all, self.Frc[:, 2], label = 'Vert. part of F_{RC}  (Sane2002, rotational force)', color='orange')
        axes[0, 1].plot(self.timeline_all, self.Fam[:, 2], label = 'Vert. part of F_{AMz} (Whitney2010)', color='red')
        axes[0, 1].plot(self.timeline_all, self.Fam2[:, 2], '--',label = 'Vert. part of F_{AMx} (vanVeen2022)', color='red')
        axes[0, 1].plot(self.timeline_all, self.Frd[:, 2], label = 'Vert. part of F_{RD} (Cai2021, Nakata2015)', color='green')
        axes[0, 1].plot(self.timeline_all, self.F_QSM_g[:,2], label = 'Total Vert. part of  QSM force', ls='-.', color='blue')
        axes[0, 1].plot(self.timeline_all, self.F_CFD_g[:,2], label = 'Total Vert. part of  CFD force', ls='--', color='k')
        axes[0, 1].set_xlabel('$t/T$')
        axes[0, 1].set_ylabel('force')
        axes[0, 1].set_title('Vertical components of forces in global coordinate system')
        axes[0, 1].legend(loc = 'best')

        #qsm + cfd force components in wing reference frame
        axes[1, 0].plot(self.timeline_all, self.F_QSM_w[:, 0], label='Fx_QSM_w', c='r')
        axes[1, 0].plot(self.timeline_all, self.F_CFD_w[:, 0], ls='-.', label='Fx_CFD_w', c='r')
        axes[1, 0].plot(self.timeline_all, self.F_QSM_w[:, 1], label='Fy_QSM_w', c='g')
        axes[1, 0].plot(self.timeline_all, self.F_CFD_w[:, 1], ls='-.', label='Fy_CFD_w', c='g')
        axes[1, 0].plot(self.timeline_all, self.F_QSM_w[:, 2], label='Fz_QSM_w', c='b')
        axes[1, 0].plot(self.timeline_all, self.F_CFD_w[:, 2], ls='-.', label='Fz_CFD_w', c='b')
        axes[1, 0].set_xlabel('$t/T$')
        axes[1, 0].set_ylabel('force')
        axes[1, 0].set_title('QSM + CFD force components in wing reference frame')
        axes[1, 0].legend(loc='best')

        #forces
        axes[1, 1].plot(self.timeline_all, self.F_QSM_g[:,0], label='Fx_QSM_g', color='red')
        axes[1, 1].plot(self.timeline_all, self.F_CFD_g[:,0], label='Fx_CFD_g', linestyle = 'dashed', color='red')
        axes[1, 1].plot(self.timeline_all, self.F_QSM_g[:,1], label='Fy_QSM_g', color='green')
        axes[1, 1].plot(self.timeline_all, self.F_CFD_g[:,1], label='Fy_CFD_g', linestyle = 'dashed', color='green')
        axes[1, 1].plot(self.timeline_all, self.F_QSM_g[:,2], label='Fz_QSM_g', color='blue')
        axes[1, 1].plot(self.timeline_all, self.F_CFD_g[:,2], label='Fz_CFD_g', linestyle = 'dashed', color='blue')
        axes[1, 1].set_xlabel('$t/T$')
        axes[1, 1].set_ylabel('force')
        if norm(self.F_CFD_g[:,0]) > 0.0 and norm(self.F_CFD_g[:,1]) > 0.0 and norm(self.F_CFD_g[:,2]) > 0.0:
            axes[1, 1].set_title( "Fi_QSM_g/Fi_CFD_g=(%2.2f, %2.2f, %2.2f) \nK=%2.2f" % (norm(self.F_QSM_g[:,0])/norm(self.F_CFD_g[:,0]),
                                                                                norm(self.F_QSM_g[:,1])/norm(self.F_CFD_g[:,1]),
                                                                                norm(self.F_QSM_g[:,2])/norm(self.F_CFD_g[:,2]),
                                                                                self.K_forces) )
        else:
            axes[1, 1].set_title( "Fi_QSM_g/Fi_CFD_g=(%2.2f, %2.2f, %2.2f) \nK=%2.2f" % (norm(self.F_QSM_g[:,0]), norm(self.F_QSM_g[:,1]), norm(self.F_QSM_g[:,2]), self.K_forces) )
        axes[1, 1].legend(loc = 'lower right')


        for ax in axes.flatten()[1:]:
            insect_tools.indicate_strokes(ax=ax, tstart=[self.T_reversals[0]], tstroke=2*(self.T_reversals[1]-self.T_reversals[0]) )

        plt.tight_layout()
        plt.draw()
        
        ##FIGURE 4
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 15))

        #cfd vs qsm moments
        ax1.plot(self.timeline_all, self.M_QSM_w[:, 0], label='Mx_QSM_w', color='red')
        ax1.plot(self.timeline_all, self.M_CFD_w[:, 0], label='Mx_CFD_w', ls='-.', color='red')
        ax1.plot(self.timeline_all, self.M_QSM_w[:, 1], label='My_QSM_w', color='blue')
        ax1.plot(self.timeline_all, self.M_CFD_w[:, 1], label='My_CFD_w', ls='-.', color='blue')
        ax1.plot(self.timeline_all, self.M_QSM_w[:, 2], label='Mz_QSM_w', color='green')
        ax1.plot(self.timeline_all, self.M_CFD_w[:, 2], label='Mz_CFD_w', ls='-.', color='green')
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
        ax2.plot(self.timeline_all, self.P_QSM_nonoptimized, label='P_QSM (non-optimized)', c='purple')
        ax2.plot(self.timeline_all, self.P_QSM, label='P_QSM (optimized)', color='b')
        ax2.plot(self.timeline_all, self.P_CFD, label='P_CFD', ls='-.', color='indigo')
        ax2.set_xlabel('$t/T$')
        ax2.set_ylabel('aerodynamic power')
        ax2.set_title("P_QSM/P_CFD=%2.2f K_power=%3.3f" % (norm(self.P_QSM)/norm(self.P_CFD), self.K_power) )
        ax2.legend()
        plt.tight_layout()
        plt.draw()

        insect_tools.indicate_strokes(ax=ax1, tstart=[self.T_reversals[0]], tstroke=2*(self.T_reversals[1]-self.T_reversals[0]) )
        insect_tools.indicate_strokes(ax=ax2, tstart=[self.T_reversals[0]], tstroke=2*(self.T_reversals[1]-self.T_reversals[0]) )