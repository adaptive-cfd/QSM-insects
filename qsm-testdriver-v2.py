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

# bumblebee data
run  = 'example_CFD_data/'
wing = 'example_CFD_data/bumblebee_wing_shape.ini'


#%% A - old code
# create model, setup wing shape, read CFD data and train the QSM model

model_terms = 5*[True]

QSM = qsm_class.QSM(nt=300, model_CL_CD="Dickinson", model_terms=model_terms)

QSM.setup_wing_shape(wing)

QSM.parse_kinematics( inifile_tools.find_WABBIT_main_inifile(run), run+'kinematics.t', wing='right', plot=True)

QSM.read_CFD_data(run, T0=1.0)

QSM.fit_to_CFD(optimize=True, plot=True, N_trials=1)

#--------------------------------------------------------------------
QSM_new = qsm_class.QSM(nt=300, model_CL_CD="Dickinson", model_terms=model_terms)

# copy the solution of parameters to new QSM object:
QSM_new.x0_forces = QSM.x0_forces
QSM_new.x0_moments = QSM.x0_moments
QSM_new.x0_power = QSM.x0_power

# dont forget geometry factors:
QSM_new.setup_wing_shape(wing)

# create new kinematics
_, alpha, phi, theta = insect_tools.bumblebee_kinematics_model(PHI=140, phi_m=0.0, alpha_down=45.0, alpha_up=-45.0, time=QSM.timeline)
# body angles:
beta, gamma, psi, eta = QSM.beta, QSM.gamma, QSM.psi, QSM.eta

QSM_new.parse_kinematics("example_CFD_data/PARAMS.ini", alpha=alpha, phi=phi, theta=theta, psi=psi*rad2deg, 
                     eta_stroke=eta*rad2deg, gamma=gamma*rad2deg, beta=beta*rad2deg, u_infty_g=[-1.246,0,0], wing="right", plot=True)


QSM_new.evaluate_QSM_model(plot=False)


plt.figure()
plt.plot( QSM.timeline, QSM.F_QSM_g[:,2], label='QSM vertical force, reference configuration')
plt.plot( QSM_new.timeline, QSM_new.F_QSM_g[:,2], label='QSM vertical force, modified kinematics (prediction)')
plt.legend()




#%% B - new code
# does the same thing, but with the revised version

Q = qsm_class_v2.QSM(model_terms=model_terms)
Q.append_KinematicsShapeForces_fromCFDrun( run, T_start=1.0, T_end=2.0, dt=QSM.dt, wing='right', verbose=True)

Q.fit_to_CFD(optimize=True, N_trials=1)

# copy body angles:
# attention on deg vs rad issue, below we expect DEG
beta, gamma, psi, eta, u_body = Q.beta*rad2deg, Q.gamma*rad2deg, Q.psi*rad2deg, Q.eta*rad2deg, Q.u_infty_g


Q2 = qsm_class_v2.QSM()

qsm_class_v2.copyQSMcoefficients(Q, Q2)

# create new kinematics
t = np.linspace(0.0, 1.0, 300)
_, alpha, phi, theta = insect_tools.bumblebee_kinematics_model(PHI=140, phi_m=0.0, alpha_down=45.0, alpha_up=-45.0, time=t)

Q2.append_KinematicsShape(t, 'right', u_body, psi, beta, gamma, eta, wing, kinematics_file=None, alpha=alpha, phi=phi, theta=theta, unit_in='deg')

Q2.evalQSM_all()

plt.figure()
plt.plot( Q.timeline, Q.F_QSM_g[:,2], label='QSM vertical force, reference configuration')
plt.plot( Q2.timeline, Q2.F_QSM_g[:,2], label='QSM vertical force, modified kinematics (prediction)')
plt.legend()
