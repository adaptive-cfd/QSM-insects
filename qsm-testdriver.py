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


plt.close('all')

# bumblebee data
run  = 'example_CFD_data/'
wing = 'example_CFD_data/bumblebee_wing_shape.ini'


#%% A
# create model, setup wing shape, read CFD data and train the QSM model

model_terms = 5*[True]

QSM = qsm_class.QSM(nt=300, model_CL_CD="Dickinson", model_terms=model_terms)

QSM.setup_wing_shape(wing)

QSM.parse_kinematics( inifile_tools.find_WABBIT_main_inifile(run), run+'kinematics.t', wing='right', plot=True)

QSM.read_CFD_data(run, T0=1.0)

QSM.fit_to_CFD(optimize=True, plot=True, N_trials=1)


#%%
# Alternative to A

QSM = qsm_class.QSM(nt=300, model_CL_CD="Dickinson", model_terms=model_terms)

QSM.parse_many_run_directorys([run], T0=1.0)

QSM.fit_to_CFD(optimize=True, plot=True, N_trials=1)


#%% B
# use model obtained above on different kinematics

QSM_new = qsm_class.QSM(nt=300, model_CL_CD="Dickinson", model_terms=model_terms)

# copy the solution of parameters to new QSM object:
QSM_new.x0_forces = QSM.x0_forces
QSM_new.x0_moments = QSM.x0_moments
QSM_new.x0_power = QSM.x0_power

# dont forget geometry factors:
QSM_new.setup_wing_shape(wing)

# create new kinematics
_, alpha, phi, theta = insect_tools.bumblebee_kinematics_model(PHI=140, phi_m=0.0, alpha_down=45.0, alpha_up=-45.0,
                                                                  time=QSM.timeline)
# body angles:
beta, gamma, psi, eta = QSM.beta, QSM.gamma, QSM.psi, QSM.eta

QSM_new.parse_kinematics("example_CFD_data/PARAMS.ini", alpha=alpha, phi=phi, theta=theta, psi=psi, 
                     eta_stroke=eta, gamma=gamma, beta=beta, u_infty_g=[1.246,0,0], wing="right", plot=True)


QSM_new.evaluate_QSM_model(plot=False)


plt.figure()
plt.plot( QSM.timeline, QSM.F_QSM_g[:,2], label='QSM vertical force, reference configuration')
plt.plot( QSM_new.timeline, QSM_new.F_QSM_g[:,2], label='QSM vertical force, modified kinematics (prediction)')
plt.legend()