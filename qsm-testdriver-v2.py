#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:01:21 2024

@author: engels
"""
import qsm_class
import insect_tools
import wabbit_tools
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
import inifile_tools
import qsm_class_v2

plt.close('all')
rad2deg = 180.0/np.pi
deg2rad = 1.0/rad2deg
# time step used in the model:
dt     = 5e-3

# bumblebee data
run  = 'example_CFD_data/'
wingShapeFile = 'example_CFD_data/bumblebee_wing_shape.ini'

# model_terms : 
#     A list of 5 bools to turn on and off individual terms in the QSM model. The terms are: 
#         [Ellington1984 (lift/drag: TC/TD), Sane2002 (rotation TC),
#           Whitney2010 (rotational drag RD), AddedMass Normal (AMz), AddedMass Tangential (AMx)]
model_terms = [True, True, True, False, True]




#%% A - new version
# create the QSM model object.
Q = qsm_class_v2.QSM(model_terms=model_terms)

# read in the existing CFD data: kinematics, forces, wing shape. Use the 2nd cycle starting at t=1
Q.append_KinematicsShapeForces_fromCFDrun( run, T_start=1.0, T_end=2.0, dt=dt, wing='right', verbose=True)

# train QSM mode, find best coefficients. Try 5 times to find the best solution, in case we get stuck 
# in a local minimum. 
Q.fit_to_CFD(N_trials=5, verbose=True)

# The CFD data are read, the QSM object contains the kinematics. As we'll modifiy some kinematics later,
# we now make a copy of the data we do not want to change. Attention on deg vs rad: below we expect degree.
# Data in Q are stored in radians.
beta, gamma, psi, eta, u_body = Q.beta*rad2deg, Q.gamma*rad2deg, Q.psi*rad2deg, Q.eta*rad2deg, Q.u_infty_g

# create a new QSM object - we use it with the modified kinematics
Q2 = qsm_class_v2.QSM()

# copy the QSM coefficients from Q to Q2 - we use the model we've trained in the first part, for the prediction.
qsm_class_v2.copyQSMcoefficients(Q, Q2)

# create new kinematics
t = np.arange(0.0, 1.0, dt)
# the function bumblebee_kinematics_model provides a convenient way to obtain the parametrized bumblebee 
# kinematics model used in [Engels et al. PRL 2016, Engels et al. PRF 2019].
_, alpha, phi, theta = insect_tools.bumblebee_kinematics_model(PHI=160, phi_m=24.0, alpha_down=70.0, alpha_up=-40.0, time=t)

# setup a new kinematics with different parameters
Q2.append_KinematicsShape(t, 'right', u_body, psi, beta, gamma, eta, wingShapeFile, kinematics_file=None, 
                          alpha=alpha, phi=phi, theta=theta, unit_in='deg')

# for the new kinematics, compute the QSM forces, moments and power.
Q2.evalQSM_all()


# compare old and new version
plt.figure()
plt.plot( Q.timeline      , Q.F_CFD_g[:,2], 'k:', label='CFD, input configuration')
plt.plot( Q.timeline      , Q.F_QSM_g[:,2], '-', label='QSM, input configuration')
plt.plot( Q2.timeline     , Q2.F_QSM_g[:,2], '--', label='QSM, modified kinematics (prediction)')
plt.ylabel('vertical force')
plt.xlabel('time')
plt.legend()
insect_tools.indicate_strokes()







# #%% B - old version
# # create model, setup wing shape, read CFD data and train the QSM model

# model_terms = [True, True, True, False, True] #5*[True]

# QSM = qsm_class.QSM(nt=300, model_CL_CD="Dickinson", model_terms=model_terms)

# QSM.setup_wing_shape(wingShapeFile)

# QSM.parse_kinematics( inifile_tools.find_WABBIT_main_inifile(run), run+'kinematics.t', wing='right', plot=True)

# QSM.read_CFD_data(run, T0=1.0)

# QSM.fit_to_CFD(optimize=True, plot=True, N_trials=5)

# #--------------------------------------------------------------------
# QSM_new = qsm_class.QSM(nt=300, model_CL_CD="Dickinson", model_terms=model_terms)

# # copy the solution of parameters to new QSM object:
# QSM_new.x0_forces = QSM.x0_forces
# QSM_new.x0_moments = QSM.x0_moments
# QSM_new.x0_power = QSM.x0_power

# # dont forget geometry factors:
# QSM_new.setup_wing_shape(wingShapeFile)

# # create new kinematics
# _, alpha, phi, theta = insect_tools.bumblebee_kinematics_model(PHI=140, phi_m=0.0, alpha_down=45.0, alpha_up=-45.0, time=QSM.timeline)
# # body angles:
# beta, gamma, psi, eta = QSM.beta, QSM.gamma, QSM.psi, QSM.eta

# QSM_new.parse_kinematics("example_CFD_data/PARAMS.ini", alpha=alpha, phi=phi, theta=theta, psi=psi*rad2deg, 
#                      eta_stroke=eta*rad2deg, gamma=gamma*rad2deg, beta=beta*rad2deg, u_infty_g=[-1.246,0,0], wing="right", plot=True)


# QSM_new.evaluate_QSM_model(plot=False)

