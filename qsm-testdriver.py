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

plt.close('all')
run = '/home/engels/dev/WABBIT777/wavelet_influence/bumblebee/WABBIT_skew/fullTree-BB-medium_CDF62-JB-Bs18-signRef-skew-nodealias/'

QSM = qsm_class.QSM(nt=400)
QSM.setup_wing_shape(run+'none.ini')
QSM.parse_kinematics(run+'PARAMS.ini', run+'kinematics.t', wing='left')
QSM.read_CFD_data(run, run+"PARAMS.ini", T0=1.0)
QSM.fit_to_CFD(optimize=True, plot=True)

raise

run = '/home/engels/Documents/Research/Insects/3D/projects/hoverfly_QSM_wageningen/diptera_project/simulations_4th/Drosophilidae/'

QSM = qsm_class.QSM(nt=400)
QSM.setup_wing_shape('none')
QSM.setup_wing_shape(run+'WING_Chlorophilidae.ini')
QSM.parse_kinematics(run+'PARAMS_diptera.ini', run+'kinematics.t', plot=True)


raise


runs = glob.glob('/home/engels/Documents/Research/Insects/3D/projects/hoverfly_QSM_wageningen/diptera_project/simulations_4th/*/')
umax, names = [], []
for run in runs:
    if "CDF" in run or "RK" in run:
        continue
    plt.close('all')
    # run = '/home/engels/Documents/Research/Insects/3D/projects/hoverfly_QSM_wageningen/diptera_project/simulations_4th/Chlorophilidae/'
    
    QSM = qsm_class.QSM(nt=400)
    QSM.setup_wing_shape('none')
    # QSM.setup_wing_shape(run+'WING_Chlorophilidae.ini')
    QSM.parse_kinematics(run+'PARAMS_diptera.ini', run+'kinematics.t', plot=False)
    # QSM.read_CFD_data(run, run+"PARAMS_diptera.ini", T0=2.0)
    # QSM.fit_to_CFD(optimize=True, plot=True)
    
    umax.append(np.max(QSM.u_tip_mag))
    names.append(run)