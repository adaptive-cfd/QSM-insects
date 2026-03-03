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
import datetime

# use latex
# plt.rcParams["text.usetex"] = True

latex = plt.rcParams["text.usetex"]
deg2rad = np.pi/180.0




class QSM:   
    """
    QSM object. The QSM describes one wing, so for a dragonfly for example you'll have four of these objects.
    
    Initialization Parameters
    ------------
    model_CL_CD :
        The Ellington terms (lift and drag) require evaluation of CL and CD as a function of AoA, for this a function is needed. Implemented models
        are "Nakata" [Nakata2015], "Dickinson" [Dickinson1999] and 'Polhamus' [J-S Han et al Bioinspr Biomim 12 2017 036004]. The results are in general rather
        similar using the three models. 
        
    model_terms : 
        A list of 5 bools to turn on and off individual terms in the QSM model. The terms are: 
            [Ellington1984 (lift/drag: TC/TD), 
             Sane2002 (rotation TC),
             Whitney2010 (rotational drag RD),
             AddedMass Normal,
             AddedMass Tangential
             ]
    
    """
    
    def __init__(self, model_CL_CD='Dickinson', model_terms=5*[True], ellington_type='utip', reversal_detector="phi_dt"):

      
        self.AM_model          = 'Full 6DOF scaled'
        self.model_terms       = model_terms
        self.ellington_type    = ellington_type
        self.reversal_detector = reversal_detector
        self.model_CL_CD       = model_CL_CD
        
        self.x0_forces = np.zeros((14))
        self.x0_moments = np.zeros((2))
        self.x0_power = np.zeros((2))
        
        # initialize empty data arrays:
        self.deleteData_keepCoefficients()
    
    def deleteData_keepCoefficients( self ):
        """
        Delete previously read kinematics and force data, but keep the trained QSM coefficients.
        
        This function is useful when you first read data, train the model, and then want to use the same QSM object for 
        predicting the forces from other kinematics.
        
        You can also copy the trained QSM coefficients and create a new QSM object instead.
        """
        scalar, vector, matrix = (0), (0,3), (0,3,3)
        
        self.alpha_dt = np.zeros(scalar)
        self.phi_dt   = np.zeros(scalar)
        self.theta_dt = np.zeros(scalar)
        
        self.rot_wing_b = np.zeros(vector)
        self.rot_wing_w = np.zeros(vector)
        self.rot_wing_g = np.zeros(vector)
    
        self.planar_rot_wing_w = np.zeros(vector)
        self.planar_rot_wing_g = np.zeros(vector)
        self.planar_rot_wing_mag = np.zeros(scalar)
    
        self.u_tip_w = np.zeros(vector)
        self.u_tip_g = np.zeros(vector)
        self.u_tip_mag = np.zeros(scalar)
    
        self.a_tip_w = np.zeros(vector)
        self.a_tip_g = np.zeros(vector)
    
        self.rot_acc_wing_g = np.zeros(vector)
        self.rot_acc_wing_w = np.zeros(vector)
    
        self.AoA = np.zeros(scalar)
        self.e_drag_g = np.zeros(vector)
        self.e_lift_g = np.zeros(vector)
        self.e_lift_b = np.zeros(vector)        
    
        self.ey_wing_g = np.zeros(vector)
        self.ez_wing_g = np.zeros(vector)
        self.ex_wing_g = np.zeros(vector)
        
        self.ey_body_g = np.zeros(vector)
        self.ez_body_g = np.zeros(vector)
        self.ex_body_g = np.zeros(vector)
        
        self.M_g2b = np.zeros(matrix)
        self.M_b2g = np.zeros(matrix)        
        self.M_g2w = np.zeros(matrix)
        self.M_w2g = np.zeros(matrix)        
        self.M_b2w = np.zeros(matrix)
        self.M_w2b = np.zeros(matrix)        
        self.M_b2s = np.zeros(matrix)
        self.M_s2b = np.zeros(matrix)
        self.M_s2w = np.zeros(matrix)
        self.M_w2s = np.zeros(matrix)
    
        self.u_infty_w = np.zeros(vector)
        self.u_infty_g = np.zeros(vector)
        
        self.P_QSM_nonoptimized = np.zeros(scalar)
        self.P_QSM = np.zeros(scalar)    
        self.F_QSM_w = np.zeros(vector)
        self.F_QSM_g = np.zeros(vector)
        self.M_QSM_w = np.zeros(vector)
        self.M_QSM_g = np.zeros(vector)
    
        
        self.F_CFD_g = np.zeros(vector)
        self.F_CFD_w = np.zeros(vector)
        self.M_CFD_g = np.zeros(vector)
        self.M_CFD_w = np.zeros(vector)
        self.P_CFD = np.zeros(scalar)
    
        # wing-geometry-dependent constants        
        # Default is ONES (the code can then work without calling setup_wing_shape
        # as long as the training and evaluation runs all have the same shape)
        self.S_AM1 = np.ones(scalar)
        self.S_AM2 = np.ones(scalar)
        self.S_RC = np.ones(scalar)
        self.S_RD = np.ones(scalar)
        self.S2 = np.ones(scalar)
        self.S1 = np.ones(scalar)
        self.S0 = np.ones(scalar)
    
        self.wing = 'left'
        self.x_wingContour_w = np.zeros(vector)
    
        self.K_forces_individual = np.zeros(scalar)
        self.K_moments_individual = np.zeros(scalar)
        self.K_power_individual = np.zeros(scalar)
    
        self.psi = np.zeros(scalar)
        self.beta = np.zeros(scalar)
        self.gamma = np.zeros(scalar)
        self.eta = np.zeros(scalar)
        self.alpha = np.zeros(scalar)
        self.phi = np.zeros(scalar)
        self.theta = np.zeros(scalar)
        
        self.timeline = np.zeros(scalar)
        self.T0_reversals = np.zeros(scalar)
        self.T1_reversals = np.zeros(scalar)
        
        self.T0_cycle = np.zeros(scalar)
        self.T1_cycle = np.zeros(scalar)
        
        self.dataID = np.zeros(scalar)
        self.sign_liftvector = np.zeros(vector)
        
    def append_KinematicsShapeForces_fromCFDrun( self, run_directory, T_start, T_end, dt, wingShapeFile=None, 
                                                 wing='auto', verbose=True, optimized_loading=True ):
        """
        From an existing CFD simulation, read in the kinematics, wingshape and forces.
        
        The data are always interpolated to an equidistant grid (that makes differentiation easier). 
        This routine reads everything the QSM model requires from the CFD. It appends the newly read 
        data to the data already stored in the QSM model (except for the first call when there is no data yet in the QSM).
        
        If you want to use a manually defined kinematics, use the function "append_KinematicsShape" instead, as 
        it provides more flexibility. This is the case for a prediction: you train the model using existing data, and
        use it to predict the forces produced by new kinematics (and that you probably do not have CFD data on.)
        
        Reading the kinematics is done using the CFD run's `kinematics.t` file, where the code logs
        the kinematics that were used in the run. We read the three angles describing the wing, the body angles,
        body velocity, etc.
            
        We read the data between T_start and T_end with a temporal resolution of dt. This may be more than one cycle
        (which, probably, only makes sense with a non-periodic wingbeat in the simulation). It may also be an incomplete 
        cycle. Note T_end is excluded (PYTHON logic: open interval [T_start, T_end) ). The most often used case is that
        the data cover one stroke, e.g. [1.0, 2.0).
        
        We try to determine whether the CFD deals with the left or the right wing, and adjust the kinematics
        accordingly. Should the CFD include both wings, an error is thrown is you pass wing='auto', and you need to
        decide yourself which wing to use, 'right' or 'left'.
        
        The wingShapeFile can be None, in which case the routine automatically looks for the right file in the run
        directory. In case you explicitly provide a wingShapeFile, that one is used.
        
        Derived kinematics data (like angular velocities, etc.) are computed directly in this routine; you do 
        not need to take care of that. We just need the angles as a function of time.´
        """
        

        # timeline to read
        t  = np.arange(T_start, T_end, dt)
        nt = t.shape[0]
        
        
        #---------------------------------------------------------------------
        # find settings in the CFDs main parameter file
        #---------------------------------------------------------------------
        # first find the actual parameter file
        PARAMS = wt.find_WABBIT_main_inifile(run_directory)
        
        # left or right wing?
        if wing == 'auto':
            # read from PARAMS-file; this is the default. If we use QSM on a two (or four) winged simulation,
            # we create one QSM model for each wing.
            isLeft   = wt.get_ini_parameter(PARAMS, 'Insects', 'LeftWing', dtype=bool)
            isRight  = wt.get_ini_parameter(PARAMS, 'Insects', 'RightWing', dtype=bool)

            if isLeft:
                wing = "left"
            if isRight:
                wing = "right"

            if isLeft and isRight:
                raise ValueError("This simulation included more than one wing, you need to create one QSM model per wing, pass wing=right and wing=left")

        elif wing not in ['right', 'left', 'right2', 'left2']:
            raise ValueError("Invalid choice for wing (auto/left/right/left2/right2)")
            
        # valid: leftwing, rightwing, leftwing2, rightwing2
        suffix = wing.replace('t', 'twing')

        if verbose:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("reading CFD data t=[%f, %f]" % (T_start, T_end))
            print('The parsed data correspond to the %s wing.' % (wing))
            print("dt=%e nt=%i" % (dt, nt))
        
        # inflow (wind) velocity in the CFD
        u_wind_g = np.asarray( wt.get_ini_parameter(PARAMS, 'ACM-new', 'u_mean_set', vector=True) ) 
        
        if wingShapeFile is None:
            # if no wing shape file is given, figure out which one was used
            wingShapeFile = wt.get_ini_parameter( PARAMS , 'Insects', 'WingShape', str, 'none')[0]
            wingShapeFile = run_directory + '/' + wingShapeFile.replace('from_file::','')
            print(wingShapeFile)
        
        if not os.path.isfile(wingShapeFile):
            raise ValueError("""We try to initialize the wing shape for run (%s), and identified %s as the WingShape file.
                             We do however not find this file, and cannot initialize the wing geometry parameters for this run.
                             Please check if the PARAMS file (%s) refers to an existing WING *.ini file. Note: it may be
                             that the QSM code incorrectly identifies the ShapeFile, if two distinct ones are used for left/right wing.""" 
                             % (run_directory, wingShapeFile, PARAMS))
        
            
        #---------------------------------------------------------------------
        # read kinematics
        #---------------------------------------------------------------------
        kinematics_CFD = optimized_t_loader(run_directory+'/kinematics.t', time=t, verbose=verbose)
        
        # finite differences matrix
        D1 = finite_differences.D12( kinematics_CFD.shape[0], dt )
        
        # get column indices         
        if wing == "right":  # right wing
            ialpha, iphi, itheta = 11, 12, 13
        elif wing == "left":  # left wing
            ialpha, iphi, itheta =  8,  9, 10
        elif wing == "right2":  # right (hind) wing, second wing pair
            ialpha, iphi, itheta = 29, 30, 31
        elif wing == "left2":  # left (hind) wing, second wing pair 
            ialpha, iphi, itheta = 26, 27, 28
                        
        # wing angles
        alpha = kinematics_CFD[:, ialpha].copy() # rad
        phi   = kinematics_CFD[:, iphi].copy()   # rad
        theta = kinematics_CFD[:, itheta].copy() # rad
        
        # wing angles time derivatives (for angular velocities)
        alpha_dt = D1 @ kinematics_CFD[:, ialpha].copy()
        phi_dt   = D1 @ kinematics_CFD[:, iphi].copy()
        theta_dt = D1 @ kinematics_CFD[:, itheta].copy()
        
        # Body velocity (time derivative of position + wind velocity)
        u_infty_g = np.zeros( (nt, 3) )
        u_infty_g[:, 0] = D1 @ kinematics_CFD[:, 1].copy() - u_wind_g[0] # note sign change (wind vs body)
        u_infty_g[:, 1] = D1 @ kinematics_CFD[:, 2].copy() - u_wind_g[1] 
        u_infty_g[:, 2] = D1 @ kinematics_CFD[:, 3].copy() - u_wind_g[2] 
        
        # body angles (also stroke plane angle)
        psi   = kinematics_CFD[:, 4].copy() # rad
        beta  = kinematics_CFD[:, 5].copy() # rad
        gamma = kinematics_CFD[:, 6].copy() # rad
        eta   = kinematics_CFD[:, 7].copy() # rad
        
        # ready to parse (compute angular velocities, unit vectors, etc)
        self.__append_parse_kinematics( alpha=alpha, phi=phi, theta=theta, alpha_dt=alpha_dt, phi_dt=phi_dt, 
                                       theta_dt=theta_dt, psi=psi, beta=beta, gamma=gamma, eta=eta, 
                                       u_infty_g=u_infty_g, side=wing.replace('2',''), dt=dt, timeline=t)


        #---------------------------------------------------------------------
        # read forces, moments
        #---------------------------------------------------------------------        
        forces_CFD  = optimized_t_loader(run_directory+'/forces_'+suffix+'.t', time=t, verbose=verbose, optimized_loading=True)
        moments_CFD = optimized_t_loader(run_directory+'/moments_'+suffix+'.t', time=t, verbose=verbose, optimized_loading=True)
                
        self.F_CFD_g = np.vstack( (self.F_CFD_g, forces_CFD[:,1:3+1]) ) # hstack for scalars, vstack for vectors (annoying)
        self.M_CFD_g = np.vstack( (self.M_CFD_g, moments_CFD[:,1:3+1]) )
                
        # obtain CFD data in wing reference frame (only last nt time steps, those are the ones we read right now)
        self.M_CFD_w = np.vstack( (self.M_CFD_w, apply_rotations_to_vectors(self.M_g2w[-nt:,:,:], self.M_CFD_g[-nt:,:]) ) )
        self.F_CFD_w = np.vstack( (self.F_CFD_w, apply_rotations_to_vectors(self.M_g2w[-nt:,:,:], self.F_CFD_g[-nt:,:]) ) )
        
        #---------------------------------------------------------------------
        # compute aerodynamic power
        #---------------------------------------------------------------------
        # aerodynamic power. Can be read from t-file or computed from the dot-product of moment and angular
        # velocity. This latter is done here: we do not store the power for each wing separately in the CFD run, and hence
        # computing it here is the better choice if possibly more than one wing has been simulated.·
        self.P_CFD = np.hstack( (self.P_CFD, -(  np.sum( self.M_CFD_w[-nt:,:]*self.rot_wing_w[-nt:,:], axis=1 ) ) ) )
        
        #---------------------------------------------------------------------
        # wing shape
        #---------------------------------------------------------------------        
        self.__append_wing_shape(nt, wingShapeFile, verbose=verbose)

    def append_KinematicsShape( self, t, wing, u_body, psi, beta, gamma, eta,
                               wingShapeFile, kinematics_file=None, alpha=None, 
                               phi=None, theta=None, unit_in='deg', verbose=True):
        """
        Append a set of kinematics and wing shape to the data stored in the QSM object.
        
        This is useful if you trained the model (performed the coefficient optimization)
        on some CFD data, or obtained the relevant QSM coefficients in an other way.
        
        Then, using this routine, you can give a kinematics and wing shape, on which you can
        then evaluate the QSM model using the function `evalQSM_all` (or the forces, moments
        power individually).
        
        If the wingShapeFile is None, we set the geometry factors in the QSM model all to one.
        This is fine if you did the same in obtaining the coefficients for the model. Otherwise,
        this will not be correct. It is strongly recommended to always use the geometry factors.
        
        Derived kinematics data (like angular velocities, etc.) are computed directly in this routine; you do 
        not need to take care of that. We just need the angles as a function of time.
        
        Two ways to append kinematics here:
            
            1/ You provide an *.INI file for the wingbeat, from which we read alpha, phi and theta (kinematics_file != None)
            
            2/ You manually provide the angles alpha, phi, theta (kinematics_file == None) 
            
        In both cases, you have to provide information about the body: its velocity (u_body) and its angles 
        (beta_pitch, psi_roll, gamma_yaw) and the stroke plane angle (eta). Those are not included in a wingbeat
        kinematics file. The values for the body attitude can be constants: for the angles, you can pass a 
        single value, for the velocity an array of length 3. You can also pass the angles as arrays 
        of the same length as the time vector: in that case, the body attitude may vary over time.
        
        """
        
        nt = t.shape[0]
        dt = t[1]-t[0]
        
        #---------------------------------------------------------------------
        # pre-process input data
        #---------------------------------------------------------------------
        if unit_in=='deg':
            factor_conversion = np.pi/180
        else:
            factor_conversion = 1.0
      
        # ensure data in rad and as array
        # NOTE: do not use *= (modifies input data!!)
        eta   = np.asarray(eta)   * factor_conversion
        psi   = np.asarray(psi)   * factor_conversion
        beta  = np.asarray(beta)  * factor_conversion
        gamma = np.asarray(gamma) * factor_conversion
        
        # if a single value is passed, convert to nt vector 
        if eta.ndim == 0:
            eta = np.full( nt, eta ) # repeat scalar nt times
        if psi.ndim == 0:
            psi = np.full( nt, psi )
        if beta.ndim == 0:
            beta = np.full( nt, beta )
        if gamma.ndim == 0:
            gamma = np.full( nt, gamma )
        if u_body.ndim == 1:
            u_infty_g = np.zeros( (t.shape[0], 3) )
            u_infty_g[:, 0] = u_body[0]
            u_infty_g[:, 1] = u_body[1]
            u_infty_g[:, 2] = u_body[2]
        else:
            assert u_body.shape[0] == t.shape[0]
            u_infty_g = u_body
            
        #---------------------------------------------------------------------
        # read kinematics (or use the ones given)
        #---------------------------------------------------------------------
        if kinematics_file is not None:
            # wing angles are read from an INI configuration file
            _, phi, alpha, theta = insect_tools.eval_angles_kinematics_file(kinematics_file, time=t, unit_out='rad')
        else:
            # no INI file is given, angles are passed:
            if alpha is None and phi is None and theta is None:
                raise ValueError("You passed kinematics_file=None but did not provide all of phi, theta and alpha.")
                
            # ensure data in rad
            # NOTE: do not use *= (modifies input data!!)
            alpha = alpha * factor_conversion
            phi   = phi * factor_conversion
            theta = theta * factor_conversion
            
        assert alpha.shape == phi.shape == theta.shape == t.shape
        assert eta.shape == beta.shape == gamma.shape == psi.shape == t.shape
        
        # finite differences matrix
        D1 = finite_differences.D12( nt, dt )

        # wing angles time derivatives (for angular velocities)
        alpha_dt = D1 @ alpha
        phi_dt   = D1 @ phi
        theta_dt = D1 @ theta
                
        # ready to parse (compute angular velocities, unit vectors, etc)
        self.__append_parse_kinematics( alpha=alpha, phi=phi, theta=theta, alpha_dt=alpha_dt, phi_dt=phi_dt, 
                                       theta_dt=theta_dt, psi=psi, beta=beta, gamma=gamma, eta=eta, 
                                       u_infty_g=u_infty_g, side=wing, dt=dt, timeline=t)
        
        #---------------------------------------------------------------------
        # wing shape
        #---------------------------------------------------------------------        
        self.__append_wing_shape(nt, wingShapeFile, verbose=verbose)

        
    def __append_wing_shape(self, nt, wingShapeFile, verbose=True, force_reload=False):
        """
        Specifiy the wing shape (here, in the form of the wing contour; encoded in an *.ini file).
        Note the code can run without this information, as the influence of
        the wing contour can also be taken into account by the optimized model
        coefficients (optimized using a reference CFD run). However, this will yield a 
        QSM model specific to that shape, and it should not be used with other wing shapes.
    
        Shape data is read from an INI file.
        
        This function is not intended for users of the QSM code, indicated by the leading
        two underscores (__) - it is mainly for internal use of the code.
        """
        if verbose:
            print('Parsing wing contour: '+wingShapeFile)
    
        if os.path.isfile(wingShapeFile):
            xc, yc, area = insect_tools.wing_contour_from_file( wingShapeFile )
            zc = np.zeros_like(xc)
        else:
            raise ValueError("Wing shape file %s not found!" % (wingShapeFile))
        
            
        self.x_wingContour_w  = np.vstack([xc, yc, zc])
        self.x_wingContour_w  = np.transpose(self.x_wingContour_w)
    
    
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        version = 100
        version_file = 0        
        reloading_required = False
        if not os.path.isfile(wingShapeFile+'.npz') and not force_reload:
            reloading_required = True
        
        if os.path.isfile(wingShapeFile+'.npz'):
            Q = np.load(wingShapeFile+'.npz')
            
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
            X, Y, mask = insect_tools.get_wing_membrane_grid(wingShapeFile, dx, dy, return_1D_list=False)
            
            np.savez(wingShapeFile+'.npz', X=X, Y=Y, dx=dx, dy=dy, mask=mask, version=version)       
            
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
        
        one = np.ones((nt))
    
        # area moments (for Ellington term)
        self.S2 = np.hstack( (self.S2, np.sum( C*r**2 )*dy * one) )
        self.S1 = np.hstack( (self.S1, np.sum( C*r**1 )*dy * one) )
        self.S0 = np.hstack( (self.S0, np.sum( C*r**0 )*dy * one) )
        
        # For Sanes rotational circulation:
        self.S_RC = np.hstack( (self.S_RC, np.sum( C**2 * r )*dy * one) )
        
        # For whitneys rotational drag:
        self.S_RD = np.hstack( (self.S_RD, np.sum( (X*mask) * np.abs(X*mask) ) * dx*dy * one) )
    
        # for Added mass:
        self.S_AM1 = np.hstack( (self.S_AM1, np.sum( r*C**2 )*dy * one) )
        self.S_AM2 = np.hstack( (self.S_AM2, np.sum( xh *  C**2 )*dy * one) )
        
        if verbose:
            print(' Wing geometry factors:')
            print(' S2    = %+e' % (self.S2[-1]))
            print(' S1    = %+e' % (self.S1[-1]))
            print(' S0    = %+e' % (self.S0[-1]))
            print(' S_RC  = %+e' % (self.S_RC[-1]))
            print(' S_RD  = %+e' % (self.S_RD[-1]))
            print(' S_AM1 = %+e' % (self.S_AM1[-1]))
            print(' S_AM2 = %+e' % (self.S_AM2[-1]))
    
        if any(np.isnan(self.S2)) or any(np.isnan(self.S1)) or any(np.isnan(self.S0)) or any(np.isnan(self.S_RC)) or any(np.isnan(self.S_RD)) or any(np.isnan(self.S_AM1)) or any(np.isnan(self.S_AM2)):
            raise ValueError("Wing shape setup has failed and computed NANs...")

     
    def __append_parse_kinematics(self, alpha, phi, theta, alpha_dt, phi_dt, theta_dt, psi, beta, gamma, eta, u_infty_g, side, dt, timeline):
        """
        Parse kinematics: given the wing angles and their time derivatives, as well as body attitude, 
        compute derived kinematics qyts for modeling: angular velocities, etc. 
        
        This function is not intended for users of the QSM code, indicated by the leading
        two underscores (__) - it is mainly for internal use of the code.
        """
        
        assert alpha.shape == phi.shape == theta.shape == alpha_dt.shape == phi_dt.shape == theta_dt.shape == timeline.shape
        assert psi.shape == beta.shape == gamma.shape == timeline.shape == eta.shape
        
        if side not in ['right', 'left']:
            raise ValueError("Side must be either left or right, not: "+side)
        
        # shift the time, so that it starts at zero
        t = timeline.copy() - timeline[0]
        
        # to identify different data parts (mostly, used for cycles)
        if self.dataID.shape[0] == 0:
            dataID = 0
        else:
            dataID = self.dataID[-1] + 1
        self.dataID = np.hstack( (self.dataID, np.zeros_like(t)+dataID ) )
        
        nt = alpha.shape[0]
        
        self.alpha = np.hstack( (self.alpha, alpha  ) ) # hstack for scalars, vstack for vectors (annoying)
        self.phi   = np.hstack( (self.phi, phi ) )
        self.theta = np.hstack( (self.theta, theta ) )
        
        self.alpha_dt = np.hstack( (self.alpha_dt, alpha_dt ) )
        self.phi_dt   = np.hstack( (self.phi_dt, phi_dt ) )
        self.theta_dt = np.hstack( (self.theta_dt, theta_dt ) )
        
        self.psi   = np.hstack( (self.psi, psi) ) # hstack for scalars, vstack for vectors (annoying)
        self.beta  = np.hstack( (self.beta, beta ) )
        self.gamma = np.hstack( (self.gamma, gamma) )
        self.eta   = np.hstack( (self.eta, eta) )
                        
        # define the many rotation matrices used in the code
        M_g2b = insect_tools.get_many_M_g2b(psi, beta, gamma)
        M_s2w = insect_tools.get_many_M_s2w(alpha, theta, phi, side)
        M_b2s = insect_tools.get_many_M_b2s(eta, side)
        M_b2w = insect_tools.get_many_M_b2w(alpha, theta, phi, eta, side)
        M_g2w = insect_tools.get_many_M_b2w(alpha, theta, phi, eta, side) @ insect_tools.get_many_M_g2b(psi, beta, gamma)

        self.M_g2b = np.vstack( (self.M_g2b, M_g2b) ) 
        self.M_s2w = np.vstack( (self.M_s2w, M_s2w) ) 
        self.M_b2s = np.vstack( (self.M_b2s, M_b2s) ) 
        self.M_b2w = np.vstack( (self.M_b2w, M_b2w) ) 
        self.M_g2w = np.vstack( (self.M_g2w, M_g2w) ) 
        
        M_w2s = M_s2w.transpose(0, 2, 1)
        M_s2b = M_b2s.transpose(0, 2, 1)
        M_w2b = M_b2w.transpose(0, 2, 1)
        M_b2g = M_g2b.transpose(0, 2, 1)
        M_w2g = M_g2w.transpose(0, 2, 1)
        
        self.M_w2s = np.vstack( (self.M_w2s, M_w2s) )
        self.M_s2b = np.vstack( (self.M_s2b, M_s2b) )
        self.M_w2b = np.vstack( (self.M_w2b, M_w2b) )
        self.M_b2g = np.vstack( (self.M_b2g, M_b2g) )
        self.M_w2g = np.vstack( (self.M_w2g, M_w2g) )

        # WING angular velocities in various frames        
        rot_wing_s = np.zeros_like( u_infty_g )
        if side == 'left':
            rot_wing_s[:,0] = phi_dt-np.sin(theta)*alpha_dt
            rot_wing_s[:,1] = np.cos(phi)*np.cos(theta)*alpha_dt-np.sin(phi)*theta_dt
            rot_wing_s[:,2] = np.sin(phi)*np.cos(theta)*alpha_dt+np.cos(phi)*theta_dt

        elif side == 'right':
            rot_wing_s[:,0] = -phi_dt-np.sin(theta)*(-alpha_dt)
            rot_wing_s[:,1] = np.cos(-phi)*np.cos(theta)*(-alpha_dt)-np.sin(-phi)*theta_dt
            rot_wing_s[:,2] = np.sin(-phi)*np.cos(theta)*(-alpha_dt)+np.cos(-phi)*theta_dt
        
        rot_wing_w = apply_rotations_to_vectors( M_s2w, rot_wing_s)
        rot_wing_b = apply_rotations_to_vectors( M_s2b, rot_wing_s)
        rot_wing_g = apply_rotations_to_vectors( M_b2g, rot_wing_b)
        
        self.rot_wing_w = np.vstack( (self.rot_wing_w, rot_wing_w ) )
        self.rot_wing_b = np.vstack( (self.rot_wing_b, rot_wing_b ) )
        self.rot_wing_g = np.vstack( (self.rot_wing_g, rot_wing_g ) )
            
        # The planar angular velocity {𝛀(φ,Θ)} comes from the decomposition of the motion
        # into 'translational' and rotational components, with the rotational component beig defined as
        # 'new' definition (not setting alpha_dt = 0)
        planar_rot_wing_w = rot_wing_w.copy()
        planar_rot_wing_w[:,1] = 0.0 # set y-component to zero       

        planar_rot_wing_g = apply_rotations_to_vectors( M_w2g, planar_rot_wing_w)
        
        self.planar_rot_wing_w = np.vstack( (self.planar_rot_wing_w, planar_rot_wing_w ) )
        self.planar_rot_wing_g = np.vstack( (self.planar_rot_wing_g, planar_rot_wing_g ) )
        
        # angular velocity norm (without feathering)
        planar_rot_wing_mag = np.linalg.norm( planar_rot_wing_w, axis=1 )
        self.planar_rot_wing_mag = np.hstack( (self.planar_rot_wing_mag, planar_rot_wing_mag) )

        # insects cruising speed (including body velocity and negative wind velocity)
        self.u_infty_g = np.vstack( (self.u_infty_g, u_infty_g) )
        # flight velocity in the wing system 
        u_infty_w = apply_rotations_to_vectors(M_g2w, u_infty_g)
        self.u_infty_w = np.vstack( (self.u_infty_w, u_infty_w) )
       
        # these are all unit vectors of the wing
        # ey_wing_g coincides with the tip only if R is normalized (usually the case)
        ex_wing_g = apply_rotations_to_vectors( M_w2g, np.matlib.repmat( np.asarray([1,0,0]), nt, 1))
        ey_wing_g = apply_rotations_to_vectors( M_w2g, np.matlib.repmat( np.asarray([0,1,0]), nt, 1))
        ez_wing_g = apply_rotations_to_vectors( M_w2g, np.matlib.repmat( np.asarray([0,0,1]), nt, 1))

        self.ex_wing_g = np.vstack( (self.ex_wing_g, ex_wing_g))
        self.ey_wing_g = np.vstack( (self.ey_wing_g, ey_wing_g))
        self.ez_wing_g = np.vstack( (self.ez_wing_g, ez_wing_g))
        
        # body unit vectors
        ex_body_g = apply_rotations_to_vectors( M_b2g, np.matlib.repmat( np.asarray([1,0,0]), nt, 1))
        ey_body_g = apply_rotations_to_vectors( M_b2g, np.matlib.repmat( np.asarray([0,1,0]), nt, 1))
        ez_body_g = apply_rotations_to_vectors( M_b2g, np.matlib.repmat( np.asarray([0,0,1]), nt, 1))
        
        self.ex_body_g = np.vstack( (self.ex_body_g, ex_body_g))
        self.ey_body_g = np.vstack( (self.ey_body_g, ey_body_g))
        self.ez_body_g = np.vstack( (self.ez_body_g, ez_body_g))

        

        # wing tip velocity
        u_tip_g = np.cross(rot_wing_g, ey_wing_g) + u_infty_g
        u_tip_w = apply_rotations_to_vectors(M_g2w, u_tip_g)
        # and its magnitude
        u_tip_mag = np.linalg.norm(u_tip_g, axis=1)
        
        self.u_tip_g = np.vstack( (self.u_tip_g, u_tip_g) )
        self.u_tip_w = np.vstack( (self.u_tip_w, u_tip_w) )
        self.u_tip_mag = np.hstack( (self.u_tip_mag, u_tip_mag)) # hstack for scalars, vstack for vectors (annoying)
        
        # drag unit vector
        e_drag_g = np.zeros_like( u_infty_g )
        e_lift_g = np.zeros_like( u_infty_g )
        for a in range(3):
            e_drag_g[:,a] = -u_tip_g[:,a] / u_tip_mag
                
        # lift unit vector
        # can be appended only after sign correction !
        e_lift_g = np.cross(-e_drag_g, ey_wing_g)
        n = np.linalg.norm(e_lift_g, axis=1)
        for a in range(3):
            e_lift_g[:,a] /= n
               
        # angle of attack
        v = ex_wing_g[:,0]*(-e_drag_g[:,0]) + ex_wing_g[:,1]*(-e_drag_g[:,1]) + ex_wing_g[:,2]*(-e_drag_g[:,2])
        AoA = np.arccos(v)  
                
        self.AoA = np.hstack( (self.AoA, AoA) )
        
        
        # finite differences matrix
        D1 = finite_differences.D12( nt, dt )        
        
        # calculation of wingtip acceleration and angular acceleration in wing reference frame
        # a second loop over time is required, because we first need to compute ang. vel. then diff it here.
        a_tip_g        = D1 @ u_tip_g
        rot_acc_wing_g = D1 @ rot_wing_g

        # transform to wing system (required for the QSM model terms)
        a_tip_w        = apply_rotations_to_vectors(M_g2w, a_tip_g)
        rot_acc_wing_w = apply_rotations_to_vectors(M_g2w, rot_acc_wing_g)
    
        self.a_tip_g = np.vstack( (self.a_tip_g, a_tip_g))
        self.a_tip_w = np.vstack( (self.a_tip_w, a_tip_w))        
        
        self.rot_acc_wing_g = np.vstack( (self.rot_acc_wing_g, rot_acc_wing_g))
        self.rot_acc_wing_w = np.vstack( (self.rot_acc_wing_w, rot_acc_wing_w))

        #----------------------------------------------------------------------
        # sign of lift vector (timing of half-cycles)
        #----------------------------------------------------------------------
        # the lift vector is until here only defined up to a sign. we decide about this sign now.
        # Many papers simply use SIGN(ALPHA) for this task, but for some kinematics we found this
        # does not work.
        # Note there is a subtlety with the sign: is it positive during up- or downstroke? This does not really
        # matter if the optimizer is used, because it can simply flip the coefficients.

        sign_liftvector = np.ones_like( e_lift_g )
        if side == "left" or side == "left2" :
            # for left and right wing, the sign is inverted (hence using the array "sign", otherwise we'd just
            # flip the sign directly in e_lift_g)
            sign_liftvector *= -1.0
          
 
        if self.reversal_detector == 'planar':
            qty_to_use = planar_rot_wing_mag #self.u_tip_mag
        elif self.reversal_detector == 'phi_dt':
            qty_to_use = np.abs(phi_dt)
        else:
            raise ValueError("Unknown reversal detector method: "+self.reversal_detector)
        
        # find minima in wingtip velocity magnitude. those, hopefully two, will be the reversals,
        # this is where the sign is flipped. We repeat the (periodic) signal to ensure we capture
        # peaks at t=0.0 and t=1.0. The distance between peaks is 3/4 * 1/2, so we think that the two half-strokes
        # occupy at most 3/8 and 5/8 of the complete cycle (maximum imbalance between up- and downstroke). This
        # filters out smaller peaks (in height) automatically, so we are left with the two most likely candidates.
        ipeaks, _ = scipy.signal.find_peaks( -1*np.hstack(  3*[qty_to_use] ), distance=3*nt/4/2)
        ipeaks -= nt # shift (skip 1st periodic image)
                
        # keep only peaks in the original signal domain (remove periodic "ghosts")
        ipeaks = ipeaks[ipeaks>=0]
        ipeaks = ipeaks[ipeaks<nt]
                
        # It should be two minima of velocity, if its not, then something weird happens in the kinematics.
        # We must then look for a different way to determine reversals or set it manually.
        if len(ipeaks) != 2 :
            plt.figure()
            plt.plot( -1.0*np.hstack(3*[qty_to_use]) )
            plt.xlabel('timeline (repeated identical cycle 3 times)')
            plt.ylabel('u_tip_mag (wing=%s)' % (self.wing))
            plt.plot( ipeaks, -1.0*qty_to_use[ipeaks], 'ro')
            plt.plot( ipeaks+nt, -1.0*qty_to_use[ipeaks], 'ro')
            plt.plot( ipeaks+2*nt, -1.0*qty_to_use[ipeaks], 'ro')
            plt.title('Wing velocity minima detection: PROBLEM (more than 2 minima found)')
            raise ValueError("We found more than two reversals in the kinematics data...")

        sign_liftvector[ ipeaks[0]:ipeaks[1], : ] *= -1
        e_lift_g *= sign_liftvector
        
        # Convention: the mean lift vector in the body system should point upwards. The problem is that the code
        # sometimes identifies the first- and sometimes the second part of the stroke as downstroke.
        # This is no problem when training a model with a run: the sign of the lift coefficients will
        # simply be inverted. However, when using the trained model for prediction of a different kinematics
        # set, then it may identify the other half as downstroke - the resulting prediction is completely wrong,
        # because the sign of e_lift_g needs to be inverted.
        # Assuming the lift vectors ez_body component is positive is a convention - still, the training can invert
        # the sign of the coefficients should that be necessary in weird maneuvres when the insect is flying on its
        # back.
        e_lift_b = apply_rotations_to_vectors(M_g2b, e_lift_g)
        if np.mean(e_lift_b[:,2]) < 0.0:
            e_lift_g *= -1.0
            sign_liftvector *= -1.0
            
            
        self.sign_liftvector = np.vstack((self.sign_liftvector, sign_liftvector))
            
        self.e_lift_g = np.vstack((self.e_lift_g, e_lift_g))
        self.e_drag_g = np.vstack((self.e_drag_g, e_drag_g))
        
        if self.timeline.shape[0] == 0:
            tshift = 0.0
            t += tshift
            self.timeline = t
        else:
            tshift = self.timeline[-1] - t[0]
            t += tshift
            self.timeline = np.hstack( (self.timeline, t))

        # plt.figure()
        # plt.plot(sign_liftvector)

        # for indication in figures:
        iq = np.where(sign_liftvector[:,0] < 0.0)[0]        
        self.T0_reversals = np.hstack( (self.T0_reversals, t[iq[ 0]]) )
        self.T1_reversals = np.hstack( (self.T1_reversals, t[iq[-1]]) )
        
        self.T0_cycle = np.hstack( (self.T0_cycle, t[ 0]) )
        self.T1_cycle = np.hstack( (self.T1_cycle, t[-1]) )
    



    def evalQSM_all(self, plot=False):
        """
        This function evaluates the QSM model with a given set of previously determined coefficients. 
        These coefficients are stored in self.x0_forces You need to parse kinematics data before you can call this
        function.
        """
        # first the forces - otherwise we cannot compute the moments
        self.evalQSM_forces(self.x0_forces, training=False)
        # compute moments (with constant lever assumption)
        self.evalQSM_moments(self.x0_moments, training=False)
        # compute power (with constant lever assumption)
        self.evalQSM_power(self.x0_power, training=False)
        
        if plot:
            self.plot_dynamics()
        
    def evalQSM_forces(self, x0, training=False):
        """
        Evaluate QSM force model with the current set of parameters x0. 
        """
        
        # unpack coefficients from parameter vector
        Cl, Cd, Crot, Crd, Cam1, Cam2, Cam3, Cam4, Cam5, Cam6, Cam7, Cam8 = self.__unpack_parameters(x0)

        rho = 1.0 # for future work, can also be set to 1.0 simply
       
        if np.max(np.abs(self.S2-1.0)) < 1.0e-10:
            import warnings
            warnings.warn("""We try to evaluate the QSM model, but the S2 (shape function) seems to be
                             all ones. Probably you did not setup the wing shape before evaluating the model,
                             please do so using the function QSM.setup_wing_shape(). 
                             Alternatively, you can manually set QSM.S2 to a desired value (not recommended).""")


        self.Ftc = np.zeros_like(self.ez_wing_g)
        self.Ftd = np.zeros_like(self.ez_wing_g)
        self.Frc = np.zeros_like(self.ez_wing_g)
        self.Fam = np.zeros_like(self.ez_wing_g)
        self.Fam2 = np.zeros_like(self.ez_wing_g)
        self.Frd = np.zeros_like(self.ez_wing_g)
        
        self.Ftc_mag = np.zeros_like(self.phi)
        self.Ftd_mag = np.zeros_like(self.phi)
        self.Frc_mag = np.zeros_like(self.phi)
        self.Fam_z_mag = np.zeros_like(self.phi)
        self.Fam_x_mag = np.zeros_like(self.phi)
        self.Frd_mag = np.zeros_like(self.phi)
        
        self.P_QSM_nonoptimized = np.zeros_like(self.phi)
        self.P_QSM = np.zeros_like(self.phi)
        self.F_QSM_w = np.zeros_like(self.ez_wing_g)
        self.F_QSM_g = np.zeros_like(self.ez_wing_g)
        self.M_QSM_w = np.zeros_like(self.ez_wing_g)
        self.M_QSM_g = np.zeros_like(self.ez_wing_g)
        


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
        
    def evalQSM_moments(self, x0, training=False):
        """
        Evaluate QSM moments model with the current set of parameters (stored in self.x0_moments).
        Requires to call evalQSM_forces before (can't compute moments without forces)
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
        
        
    def evalQSM_power(self, x0, training=False):
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

        # power using the moments (need to call evalQSM_moments first)
        # this is the optimal lever for moment computation
        if not training:
            self.P_QSM_nonoptimized = -(self.M_QSM_w[:,0]*self.rot_wing_w[:, 0]
                                      + self.M_QSM_w[:,1]*self.rot_wing_w[:, 1]
                                      + self.M_QSM_w[:,2]*self.rot_wing_w[:, 2])
        
        # best estimate of the power using a constant lever optimized for power prediction.
        self.P_QSM = -(Mx_QSM_w_power*self.rot_wing_w[:, 0]
                     + My_QSM_w_power*self.rot_wing_w[:, 1]
                     + Mz_QSM_w_power*self.rot_wing_w[:, 2])
        
        
    def __unpack_parameters(self, x0, AoA=None):
        """
        This function is not intended for users of the QSM code, indicated by the leading
        two underscores (__) - it is mainly for internal use of the code.
        """
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
            raise ValueError("The CL/CD model must be either Dickinson/Nakata/Polhamus, not: "+self.model_CL_CD)

        Crot = x0[4]
        Crd  = x0[5]        
        Cam1 = x0[6]
        Cam2 = x0[7]        
        Cam3 = x0[8]
        Cam4 = x0[9]
        Cam5 = x0[10]
        Cam6 = x0[11]
        Cam7, Cam8 = x0[12], x0[13]

        return Cl, Cd, Crot, Crd, Cam1, Cam2, Cam3, Cam4, Cam5, Cam6, Cam7, Cam8



    def fit_to_CFD(self, N_trials=1, verbose=True):
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
            self.evalQSM_forces(x, training=True)

            # compute quality metric (relative L2 error)
            return compute_K_forces( self.F_QSM_g, self.F_CFD_g)

        #----------------------------------------------------------------------
        # TRAINING: find optimal set of coefficients
        #----------------------------------------------------------------------
        # as a means of informing the user that they need to read CFD data before fitting (training):
        if self.F_CFD_g.shape[0] == 0:
            raise ValueError("You need to read CFD before you can fit the model to it. call QSM.read_CFD_data")

        start = time.time()
        bounds = 14*[(-1000, 1000)]
        K_forces = 9e9

        # optimize N_trials times from a different initial guess, use best solution found
        # NOTE: tests indicate the system always finds the same solution, so this could
        # be omitted. Kept for safety - we do less likely get stuck in local minima this way
        for i_trial in range(N_trials):
            x0_forces    = np.random.rand(14)
            optimization = opt.minimize(cost_forces, args=(self), bounds=bounds, x0=x0_forces)
            x0_forces    = optimization.x
            
            # for readability, remove unused coefficients
            if not self.model_terms[0]:
                x0_forces[0:3+1] = np.nan
            if not self.model_terms[1]:
                x0_forces[4] = np.nan
            if not self.model_terms[2]:
                x0_forces[5] = np.nan
            if not self.model_terms[3]:
                x0_forces[ [6,7,8,9,10,11] ] = np.nan
            if not self.model_terms[4]:
                x0_forces[ [12,13] ] = np.nan   

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
        self.evalQSM_forces(self.x0_forces, training=False)
        self.K_forces = compute_K_forces( self.F_QSM_g, self.F_CFD_g)
                
        # evaluate, for each run, the model quality (the local K_forces for each run)
        nruns = int(np.max(self.dataID))+1
        self.K_forces_individual = np.zeros((nruns))
        for irun in range( nruns ): 
            # error (K) for this CFD run:                         
            jj = (np.abs(self.dataID-float(irun)) <= 1.0e-6)
            self.K_forces_individual[irun] = compute_K_forces( self.F_QSM_g[jj], self.F_CFD_g[jj] )
            
        

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
            self.evalQSM_moments(x, training=True)
            # evaluate accuracy
            K_moments = compute_K_moments( self.M_QSM_w, self.M_CFD_w )
            return K_moments

        # moment optimization
        x0_moments = [1.0, 1.0]
        bounds = [(-6, 6), (-6, 6)]

        start = time.time()
        optimization = opt.minimize(cost_moments, args=(self), bounds=bounds, x0=x0_moments)
        if verbose:
            print('Completed in:', round(time.time() - start, 4), 'seconds')

        self.x0_moments = optimization.x
        self.K_moments  = optimization.fun

        # final evaluation
        self.evalQSM_moments(self.x0_moments, training=False)
        # global approximation error (over all runs)
        self.K_moments = compute_K_moments( self.M_QSM_w, self.M_CFD_w )
        
        # evaluate, for each run, the model quality (the local K_moments for each run)     
        nruns = int(np.max(self.dataID))+1
        self.K_moments_individual = np.zeros((nruns))
        for irun in range( nruns ): 
            # error (K) for this CFD run:                         
            jj = (np.abs(self.dataID-float(irun)) <= 1.0e-6)
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
            self.evalQSM_power(x, training=True)
            # and evaluate its accuracy
            K_power = compute_K_power(self.P_QSM, self.P_CFD)
            return K_power

        # power optimization
        x0_power = [1.0, 1.0]
        bounds = [(-6, 6), (-6, 6)]

        start = time.time()
        optimization = opt.minimize(cost_power, args=(self), bounds=bounds, x0=x0_power)
        self.x0_power = optimization.x
        self.K_power = optimization.fun
        
        if verbose:
            print('Completed in:', round(time.time() - start, 4), 'seconds')

        # global approximation error (over all runs)
        self.evalQSM_power(self.x0_power, training=False)
        self.K_power = compute_K_power(self.P_QSM, self.P_CFD)
 
        
        # evaluate, for each run, the model quality (the local K_power for each run)
        nruns = int(np.max(self.dataID))+1
        self.K_power_individual = np.zeros((nruns))
        for irun in range( nruns ): 
            # error (K) for this CFD run:                         
            jj = (np.abs(self.dataID-float(irun)) <= 1.0e-6)
            self.K_power_individual[irun] = compute_K_power( self.P_QSM[jj], self.P_CFD[jj] )


        if verbose:
            print('\nOptimized coefficients:')
            print('x0_forces  :', np.round(self.x0_forces, 5))
            print('x0_moments :', np.round(self.x0_moments, 5))
            print('x0_power   :', np.round(self.x0_power, 5))
            print('\nResulting approximation errors:')
            print("K_forces  = %2.2f" % (self.K_forces))
            print("K_moments = %2.2f" % (self.K_moments))
            print("K_power   = %2.2f" % (self.K_power))
            print('~~~~~~~~~~~~~~~Model training complete~~~~~~~~~~~~~~~~~~~~~')


    


    def plot2D_kinematics(self):
        ## FIGURE 1
        fig, axes = plt.subplots(3, 3, figsize = (15, 15))

        # angles
        ax = axes[0,0]
        ax.plot(self.timeline, np.degrees(self.phi), label='phi')
        ax.plot(self.timeline, np.degrees(self.alpha), label ='alpha')
        ax.plot(self.timeline, np.degrees(self.theta), label='theta')
        ax.plot(self.timeline, np.degrees(self.AoA), label='AoA', color = 'purple')
        ax.set_xlabel('t/T')
        ax.set_ylabel('(deg)')
        ax.legend()

        # time derivatives of angles
        ax = axes[0,1]
        ax.plot(self.timeline, self.phi_dt, label='phi_dt')
        ax.plot(self.timeline, self.alpha_dt, label='alpha_dt')
        ax.plot(self.timeline, self.theta_dt, label='theta_dt')

        if self.wing == "right" or self.wing == "right2" :
            ax.plot(self.timeline, np.sign(-self.alpha), 'k--', label='sign(alpha)', linewidth=0.5 )
        elif self.wing == "left" or self.wing == "left2" :
            ax.plot(self.timeline, np.sign(+self.alpha), 'k--', label='sign(alpha)', linewidth=0.5 )
        ax.set_xlabel('$t/T$')
        ax.legend()
        
        ax = axes[2,2]
        self.plot2D_lollipop_diagram(ax=ax)
        ax.set_title('Lollipop diagram')
        
        

        # u_wing_w (tip velocity in wing reference frame )
        ax = axes[1,0]
        
        ax.plot(self.timeline, self.u_tip_w[:, 0], label='utip_w_x')
        ax.plot(self.timeline, self.u_tip_w[:, 1], label='utip_w_y')
        ax.plot(self.timeline, self.u_tip_w[:, 2], label='utip_w_z')
        ax.plot(self.timeline, self.u_tip_mag, 'k--', label='utip_mag')

        ax.set_xlabel('t/T')
        ax.set_ylabel('[Rf]')
        ax.set_title('Tip velocity magnitude in wing reference frame = %2.2f' % (np.mean(self.u_tip_mag)))
        ax.legend()

        #a_wing_w (tip acceleration in wing reference frame )
        ax = axes[1,1]
        ax.plot(self.timeline, self.a_tip_w[:, 0], label='$\\dot{u}_{\\mathrm{wing},x}^{(w)}$')
        ax.plot(self.timeline, self.a_tip_w[:, 1], label='$\\dot{u}_{\\mathrm{wing},y}^{(w)}$')
        ax.plot(self.timeline, self.a_tip_w[:, 2], label='$\\dot{u}_{\\mathrm{wing},z}^{(w)}$')
        ax.set_xlabel('$t/T$')
        ax.set_ylabel('$Rf^2$')
        ax.set_title('Tip acceleration in wing reference frame')
        ax.legend()

        #rot_wing_w (tip velocity in wing reference frame )
        ax = axes[2,0]
        ax.plot(self.timeline, self.rot_wing_w[:, 0], label='$\\Omega_{\\mathrm{wing},x}^{(w)}$')
        ax.plot(self.timeline, self.rot_wing_w[:, 1], label='$\\Omega_{\\mathrm{wing},y}^{(w)}$')
        ax.plot(self.timeline, self.rot_wing_w[:, 2], label='$\\Omega_{\\mathrm{wing},z}^{(w)}$')
        ax.set_xlabel('$t/T$')
        ax.set_ylabel('rad/T')
        ax.set_title('Angular velocity in wing reference frame')
        ax.legend()

        #rot_acc_wing_w (angular acceleration in wing reference frame )
        ax = axes[2,1]
        ax.plot(self.timeline, self.rot_acc_wing_w[:, 0], label='$\\dot\\Omega_{\\mathrm{wing},x}^{(w)}$')
        ax.plot(self.timeline, self.rot_acc_wing_w[:, 1], label='$\\dot\\Omega_{\\mathrm{wing},y}^{(w)}$')
        ax.plot(self.timeline, self.rot_acc_wing_w[:, 2], label='$\\dot\\Omega_{\\mathrm{wing},z}^{(w)}$')

        ax.plot(self.timeline, np.sqrt(self.rot_acc_wing_w[:, 0]**2+self.rot_acc_wing_w[:, 1]**2+self.rot_acc_wing_w[:, 2]**2), 'k--', label='mag')
        ax.set_xlabel('$t/T$')
        ax.set_ylabel('[rad/T²]')
        ax.set_title('Angular acceleration in wing reference frame')
        ax.legend()

        plt.suptitle('Kinematics data')

        plt.tight_layout()
        plt.draw()

        for ax in axes.flatten()[:-1]:
            # shaded background
            ax.fill_between( self.timeline, ax.get_ylim()[0], ax.get_ylim()[1],
                where=(self.sign_liftvector[:,0] < 0), color=[0.85, 0.85, 0.85, 1.0], step="post" )
            
            
    def plot2D_dynamics(self):

        ##FIGURE 2
        fig, axes = plt.subplots(2, 2, figsize = (15, 10))

        #coefficients
        graphAoA = np.linspace(-10, 90, 100)*(np.pi/180)
        gCl, gCd, gCrot, gCrd, gCam1, gCam2,_ ,_ ,_, _, _, _  = self.__unpack_parameters(self.x0_forces, graphAoA)
        axes[0, 0].plot(np.degrees(graphAoA), gCl, label='Cl')
        axes[0, 0].plot(np.degrees(graphAoA), gCd, label='Cd')
        axes[0, 0].set_title('Lift and drag coeffficients')
        axes[0, 0].set_xlabel('AoA[°]')
        axes[0, 0].set_ylabel('[-]')
        axes[0, 0].legend(loc = 'upper right')

        #vertical forces
        axes[0, 1].plot(self.timeline, self.Ftc[:, 2], label = 'Vert. part of F_{TC} (Ellington1984 lift force)')
        axes[0, 1].plot(self.timeline, self.Ftd[:, 2], label = 'Vert. part of F_{TD} (Ellington1984 drag force)')
        axes[0, 1].plot(self.timeline, self.Frc[:, 2], label = 'Vert. part of F_{RC}  (Sane2002, rotational force)')
        axes[0, 1].plot(self.timeline, self.Fam[:, 2], label = 'Vert. part of F_{AMz} (Whitney2010)')
        axes[0, 1].plot(self.timeline, self.Fam2[:, 2], '--',label = 'Vert. part of F_{AMx} (vanVeen2022)')
        axes[0, 1].plot(self.timeline, self.Frd[:, 2], label = 'Vert. part of F_{RD} (Cai2021, Nakata2015)')
        axes[0, 1].plot(self.timeline, self.F_QSM_g[:,2], label = 'Total Vert. part of  QSM force', ls='-', color='k')
        axes[0, 1].plot(self.timeline, self.F_CFD_g[:,2], label = 'Total Vert. part of  CFD force', ls='--', color='k')
        axes[0, 1].set_xlabel('$t/T$')
        axes[0, 1].set_ylabel('force')
        axes[0, 1].set_title('Vertical components of forces in global coordinate system')
        axes[0, 1].legend(loc = 'best')

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
        axes[1, 0].legend(loc='best')

        #forces
        axes[1, 1].plot(self.timeline, self.F_QSM_g[:,0], label='Fx_QSM_g', color='red')
        axes[1, 1].plot(self.timeline, self.F_CFD_g[:,0], label='Fx_CFD_g', linestyle = 'dashed', color='red')
        axes[1, 1].plot(self.timeline, self.F_QSM_g[:,1], label='Fy_QSM_g', color='green')
        axes[1, 1].plot(self.timeline, self.F_CFD_g[:,1], label='Fy_CFD_g', linestyle = 'dashed', color='green')
        axes[1, 1].plot(self.timeline, self.F_QSM_g[:,2], label='Fz_QSM_g', color='blue')
        axes[1, 1].plot(self.timeline, self.F_CFD_g[:,2], label='Fz_CFD_g', linestyle = 'dashed', color='blue')
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
            # shaded background
            ax.fill_between( self.timeline, ax.get_ylim()[0], ax.get_ylim()[1],
                where=(self.sign_liftvector[:,0] < 0), color=[0.85, 0.85, 0.85, 1.0], step="post" )

        plt.tight_layout()
        plt.draw()
        
        ##FIGURE 4
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 15))

        #cfd vs qsm moments
        ax1.plot(self.timeline, self.M_QSM_w[:, 0], label='Mx_QSM_w', color='red')
        ax1.plot(self.timeline, self.M_CFD_w[:, 0], label='Mx_CFD_w', ls='--', color='red')
        ax1.plot(self.timeline, self.M_QSM_w[:, 1], label='My_QSM_w', color='blue')
        ax1.plot(self.timeline, self.M_CFD_w[:, 1], label='My_CFD_w', ls='--', color='blue')
        ax1.plot(self.timeline, self.M_QSM_w[:, 2], label='Mz_QSM_w', color='green')
        ax1.plot(self.timeline, self.M_CFD_w[:, 2], label='Mz_CFD_w', ls='--', color='green')
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
        # ax2.plot(self.timeline, self.P_QSM_nonoptimized, label='P_QSM (non-optimized)', c='purple')
        ax2.plot(self.timeline, self.P_QSM, label='P_QSM')
        ax2.plot(self.timeline, self.P_CFD, label='P_CFD', ls='-.', color='k')
        ax2.set_xlabel('$t/T$')
        ax2.set_ylabel('aerodynamic power')
        ax2.set_title("P_QSM/P_CFD=%2.2f K_power=%3.3f" % (norm(self.P_QSM)/norm(self.P_CFD), self.K_power) )
        ax2.legend()
        plt.tight_layout()
        plt.draw()

        for ax in [ax1,ax2]:
            # shaded background
            ax.fill_between( self.timeline, ax.get_ylim()[0], ax.get_ylim()[1],
                where=(self.sign_liftvector[:,0] < 0), color=[0.85, 0.85, 0.85, 1.0], step="post" )
        
        
    def plot3D_singleFrame(self, it, ax):
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
        
        # ax.plot( [0, self.ex_body_g[it,0]], [0, self.ex_body_g[it,1]], [0, self.ex_body_g[it,2]], 'k-.')
        # ax.plot( [0, self.ex_body_g[it,0]], [a, a], [0, self.ex_body_g[it,2]], 'k-.')
        
        ax.quiver( X=[0.0, 0.0, 0.0], Y=[0.0, 0.0, 0.0], Z=[0.0, 0.0, 0.0], 
                  U=[self.ex_body_g[it,0],self.ey_body_g[it,0],self.ez_body_g[it,0]], 
                  V=[self.ex_body_g[it,1],self.ey_body_g[it,1],self.ez_body_g[it,1]], 
                  W=[self.ex_body_g[it,2],self.ey_body_g[it,2],self.ez_body_g[it,2]],
                  arrow_length_ratio=0.15, label='body reference frame', color='k', lw=0.25)

        #set the axis limits
        ax.set_xlim([-a, a])
        ax.set_ylim([-a, a])
        ax.set_zlim([-a, a])

        #set the axis labels
        ax.set_xlabel('$x/R$')
        ax.set_ylabel('$y/R$')
        ax.set_zlabel('$z/R$')
        
        ax.legend()
        
        

    def plot3D_allFrames(self, fnames_out="visualization3D", directory='./', dark_plot=False, dpi=200,
                         savePNG=True, continuous=False):
        """
        Creates one PNG file per time step, loops over the entire time vector
        """
        
        if dark_plot:
            plt.style.use('dark_background')

        # can be called only after parse_kinematics
        fig = plt.figure()
        ax = plt.figure().add_subplot(projection='3d')
        it = 0
        

        for it in range(self.timeline.shape[0]):
            ax.cla()
            self.plot3D_singleFrame(it, ax)
            
            plt.show()            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            fname_out = directory + '/' + fnames_out+'.%04i.png' % (it)
            print('Animation 3D visualization saving: ' + fname_out)
            
            if savePNG:
                plt.savefig( fname_out, dpi=dpi )

    def plot2D_lollipop_diagram( self, ax=None, DrawPath=True, PathColor='k', chord_length=0.1, N_lollipops=40, cmap=None, draw_stoke_plane=True):
        """
        Lollipop-diagram. 
        
        This type of diagram shows a "wing section", a line with dot for the leading edge. This is called a lollipop.
        The visualization takes place in the sagittal plane (index _m for midplane, because _s is already taken for stroke plane).
        It would be more natural to draw this diagram in the body coordinate system, but that's not the convention. The sagittal
        plane is the body reference frame with an additional rotation by -mean(beta), where beta is pitch angle.
        
        You should use this diagram only if you are looking at a single wingbeat, because it will otherwise be confusing
    
        """
        import matplotlib.pyplot as plt
        import matplotlib
    
        wing = self.wing
                
        if ax is None:
            ax = plt.gca()            

        for irun in np.arange(np.max(self.dataID)):
            ii = ( self.dataID == irun )
            
            # this diagram would be most convenient in the body coordinate system,
            # but that is not the convention. The convention is the sagittal plane, looking from the 
            # side at the insect, i.e. you can see the pitch angle. However as this is arbitrary, it
            # seems to be more logical to me to use the mean pitch angle for that, rather than the
            # instantaneous one. In most cases, this is the same, as beta is often constant.
            beta_sagittal = np.mean(self.beta[ii])
        
            # In the Lollipop diagram, we look at the insect from the side. The pitch angle
            # is visible, but yaw and roll are not (otherwise the fig is distorted and very 
            # difficult to interpret). This translates to an additional rotation around y 
            # by -beta, after going to the body system.
            # Index _m because _s is stroke plane.
            M_b2sagittal = insect_tools.Ry(-1.0*beta_sagittal)
                
            # read kinematics data:
            time  = np.linspace(0.0, 1.0, num=N_lollipops, endpoint=False)
            t2    = self.timeline[ii]
            t2   -= t2[0]
            phi   = np.interp(time, t2, self.phi[ii])
            alpha = np.interp(time, t2, self.alpha[ii])
            theta = np.interp(time, t2, self.theta[ii])        
            eta   = np.interp(time, t2, self.eta[ii]) # note how eta is time-independent but still used as a (constant) array
            # psi   = np.interp(time, t2, self.psi[ii])
            # beta  = np.interp(time, t2, self.beta[ii])
            # gamma = np.interp(time, t2, self.gamma[ii])
                            
            # wing tip in wing coordinate system
            x_tip_w   = np.asarray([0.0, 1.0, 0.0])
            x_le_w    = np.asarray([ 0.5*chord_length,1.0,0.0])
            x_te_w    = np.asarray([-0.5*chord_length,1.0,0.0])
        
            # array of color (note normalization to 1 for query values)
            if cmap is None:
                cmap = plt.cm.jet
            if type(cmap) == matplotlib.colors.LinearSegmentedColormap:
                colors = cmap( (np.arange(time.size) / time.size) )
            else:
                # if its a constant color, jus create a list of colors
                colors = time.size*[cmap]
    
        
            # step 1: draw the symbols for the wing section for some time steps
            for i in range(time.size):        
                # (true) body transformation matrix
                # M_g2b = insect_tools.get_M_g2b(psi[i], beta[i], gamma[i], unit_in='rad')
                    
                # rotation matrix (body -> wing)
                M_b2w = insect_tools.get_M_b2w(alpha[i], theta[i], phi[i], eta[i], wing, unit_in='rad')
        
                # convert wing points to sagittal coordinate system
                x_tip_m =  M_b2sagittal @ ( np.transpose(M_b2w) @ x_tip_w ) 
                x_le_m  =  M_b2sagittal @ ( np.transpose(M_b2w) @ x_le_w ) 
                x_te_m  =  M_b2sagittal @ ( np.transpose(M_b2w) @ x_te_w )
        
                # the wing chord changes in length, as the wing moves and is oriented differently
                # note if the wing is perpendicular, it is invisible
                # so this vector goes from leading to trailing edge:
                e_chord = x_te_m - x_le_m
                e_chord[1] = 0.0
                
                # normalize it to have the right length
                e_chord = e_chord / (np.linalg.norm(e_chord))
                
                # pseudo TE and LE. note this is not true TE and LE as the line length changes otherwise
                x_le_m = x_tip_m - e_chord * chord_length/2.0
                x_te_m = x_tip_m + e_chord * chord_length/2.0
        
                # draw actual lollipop
                # mark leading edge with a marker
                ax.plot( x_le_m[0], x_le_m[2], marker='o', color=colors[i], markersize=4 )
                # draw wing chord
                ax.plot( [x_te_m[0], x_le_m[0]], [x_te_m[2], x_le_m[2]], '-', color=colors[i])
                
                
        
            # step 2: draw the path of the wingtip
            if DrawPath:
                # refined time vector for drawing the wingtip path
                time  = np.linspace(0.0, 1.0, num=200, endpoint=False)
                t2    = self.timeline[ii]
                t2   -= t2[0]
                phi   = np.interp(time, t2, self.phi[ii])
                alpha = np.interp(time, t2, self.alpha[ii])
                theta = np.interp(time, t2, self.theta[ii])        
                eta   = np.interp(time, t2, self.eta[ii]) # note how eta is time-independent but still used as a (constant) array
               
                xpath, zpath = np.zeros_like(time), np.zeros_like(time)
        
                for i in range(time.size):
                    # rotation matrix from body to wing coordinate system 
                    M_b2w = insect_tools.get_M_b2w(alpha[i], theta[i], phi[i], eta[i], wing, unit_in='rad')
                    # convert wing points to sagittal coordinate system
                    x_tip_m = M_b2sagittal @ np.transpose(M_b2w) @ x_tip_w
        
                    xpath[i] = x_tip_m[0]
                    zpath[i] = x_tip_m[2]
                ax.plot( xpath, zpath, linestyle='--', color=PathColor, linewidth=1.0 )
        
        
            # Draw stroke plane as a dashed line
            # NOTE: if beta is not constant, there should be more lines...
            if draw_stoke_plane:
                M_b2s = insect_tools.get_M_b2s( eta[0], wing, unit_in='rad')
                
                # we draw the line between [0,0,-1] and [0,0,1] in the stroke system        
                x1_s = np.asarray([0.0, 0.0, +1.0])
                x2_s = np.asarray([0.0, 0.0, -1.0])
                
                # bring these points back to the global system
                x1_m = M_b2sagittal @ ( np.transpose(M_b2s) @ x1_s )
                x2_m = M_b2sagittal @ ( np.transpose(M_b2s) @ x2_s )       
            
                # remember we're in the x-z plane
                ax.plot( [x1_m[0],x2_m[0]], [x1_m[2],x2_m[2]], color='k', linewidth=1.0, linestyle='--')
        
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_xlabel('x_{sagittal}/R')
        ax.set_ylabel('z_{sagittal}/R')
        
        insect_tools.axis_equal_keepbox(plt.gcf(), ax)
        
        

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


def check_if_reloading_required( file, time ):    
    # size of the *.t file (to check if this changed)
    size_t = os.path.getsize(file)    
    
    if not os.path.isfile( file+'.npz' ):
        # NPZ file does not exist - reloading is required.
        return(True)        

    # file exists        
    if datetime.datetime.fromtimestamp(os.path.getmtime(file+'.npz' )) < datetime.datetime.fromtimestamp(os.path.getmtime(file)):
        # *.npz file is older than source *.t file - reloading is required
        return(True)
        
    # even if we won't use it, read the file now to determine at what time T0 it is (which cycle)
    Q = np.load( file+'.npz' )
    
    if "size_t" in Q.keys():
        size_t_old = Q['size_t']
    else:
        # reload (information missing - old NPZ file version?)
        return(True)
        
    if "time" in Q.keys():
        time_old = Q['time']
    else:
        # reload (information missing - old NPZ file version?)
        return(True)
        
    if size_t != size_t_old:
        # reload (wrong file size, something changed)
        return(True)

    if time_old.shape[0] != time.shape[0]:
        # reload (wrong time vector length)
        return(True)
    
    # check if same T0 and nt is used
    if np.max(np.abs(time-time_old)) > 1.0e-7 or Q['data'].shape[0] != time.shape[0]  : 
        # different time vector or wrong size of dat array - reload
        return(True)
    
    # if we arrive here, none of the conditions are met, and we do not need to reload the ASCII file.
    return False


def optimized_t_loader( file, time, optimized_loading=True, verbose=False ):      
    """
    Reads an ASCII *.t file, but possibly optimized. On the first reading of this file,
    the actual ASCII file is read from HDD. This may be slow, depending on the file. Therefore,
    we store the result of the reading operationg in a compressed, binary NPZ file. When next 
    executing the code, this NPZ file is read instead, and that is much faster. 
    """    
    
    # size of the *.t file (to check if this changed)
    size_t = os.path.getsize(file)    
    
    if optimized_loading:
        reloading_required = check_if_reloading_required(file, time)
    else:
        reloading_required = True
        
    if not optimized_loading or reloading_required:
        # read in data from desired cycle (hence the shift by T0)
        # NOTE that load_t_file can remove outliers in the data, which turned out very useful for the
        # musca domestica data (which have large jumps exactly at the reversals)
        data = insect_tools.load_t_file(file, interp=True, time_out=time, remove_outliers=True, 
                                        verbose=verbose, optimized_loading=False)
        
        if optimized_loading:
            # when optimization is used, save converted data to binary npz file to read that next time.
            np.savez(file+'.npz', data=data, time=time, size_t=size_t)                
    else:
        # use optimized loading
        if verbose:
            print('Optimized reading from pre-converted *.npz file: '+file)       
        
        Q = np.load( file+'.npz' )
        
        data = Q['data']
        
    return data

def copyQSMcoefficients(QSM1, QSM2):
    """ Copy QSM coefficients from QSM1 to QSM2. 
    
    This also copies other important settings:
    model_terms, model_CL_CD, reversal_detector, AM_model.
    
    """
    QSM2.x0_forces  = QSM1.x0_forces
    QSM2.x0_moments = QSM1.x0_moments
    QSM2.x0_power   = QSM1.x0_power
    QSM2.model_terms = QSM1.model_terms
    QSM2.model_CL_CD = QSM1.model_CL_CD
    QSM2.reversal_detector = QSM1.reversal_detector
    QSM2.AM_model = QSM1.AM_model


