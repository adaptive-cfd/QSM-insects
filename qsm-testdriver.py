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


#%% paratuposa
# run = '/tmp/PP2_membranous/'
# QSM = qsm_class.QSM()
# QSM.setup_wing_shape(run+'Model_03_no-flight_200602right_bwmask.ini')
# QSM.parse_kinematics(run+'PARAMS_bristled.ini', run+'kinematics.t')
# QSM.fit_to_CFD(run, run+'PARAMS_bristled.ini', T0=1.0, optimize=True)

run = '/home/engels/dev/WABBIT777/dragonfly2/Jmax6/v15-3/'
QSM = qsm_class.QSM()
QSM.setup_wing_shape(run+'forewing-shape-PF.ini')
QSM.parse_kinematics(run+'dragonfly_PF_v15-3.ini', run+'kinematics.t', plot=True)
QSM.fit_to_CFD(run, run+'dragonfly_PF_v15-3.ini', T0=1.0, optimize=True)

# run = '/home/engels/musca_model/simulation/compensation/compensation_singleWing_3parameters/intact_wing/phi140.00_phim5.00_dTau0.05/'
# # run = '/home/engels/Documents/Research/Insects/3D/projects/hoverfly_QSM_wageningen/diptera_project/simulations_corrected/'+species+'/'
# QSM = qsm_class.QSM()
# QSM.setup_wing_shape(run+'WING_musca_noAlula.ini')
# QSM.parse_kinematics(run+'PARAMS.ini', run+'kinematics.t')
# QSM.fit_to_CFD(run, run+'PARAMS.ini', T0=1.0, optimize=True)


raise

species = "Chaoborus_flavicans"

# # plt.close('all')
# run = '/home/engels/musca_model/simulation/compensation/compensation_singleWing_3parameters/intact_wing/phi140.00_phim5.00_dTau0.05/'
# # run = '/home/engels/Documents/Research/Insects/3D/projects/hoverfly_QSM_wageningen/diptera_project/simulations_corrected/'+species+'/'
# QSM = qsm_class.QSM()
# QSM.setup_wing_shape(run+'WING_musca_noAlula.ini')
# QSM.parse_kinematics(run+'PARAMS.ini', run+'kinematics.t')
# QSM.fit_to_CFD(run, run+'PARAMS.ini', T0=1.0, optimize=True)

# plt.close('all')
# run = '/home/engels/musca_model/simulation/compensation/compensation_singleWing_3parameters/intact_wing/phi80.00_phim5.00_dTau-0.05/'
# # run = '/home/engels/Documents/Research/Insects/3D/projects/hoverfly_QSM_wageningen/diptera_project/simulations_corrected/'+species+'/'
# QSM.parse_kinematics(run+'PARAMS.ini', run+'kinematics.t')
# QSM.fit_to_CFD(run, run+'PARAMS.ini', T0=1.0, optimize=False)


plt.close('all')

# run = '/home/engels/bristles/bristles_RSI/lift_based_kinematics/Re24/bristled_redesign_dB0.016/'
run = '/home/engels/bristles/bristles_RSI/DRAG_BASED_NFFT30/drag_based_kinematics/Re24/bristled_redesign_dB0.016/'

QSM = qsm_class.QSM()
QSM.setup_wing_shape(run+'bristled_wing_redesigned_dB0.016.ini')
QSM.parse_kinematics(run+'PARAMS_bristled.ini', run+'kinematics.t')
QSM.fit_to_CFD(run, run+'PARAMS_bristled.ini', T0=1.0, optimize=True)

plt.figure()

ax = plt.figure().add_subplot(projection='3d')


ax.plot(QSM.ey_wing_g[:,0], QSM.ey_wing_g[:,1], QSM.ey_wing_g[:,2])
plt.axis('equal')


# plt.close('all')
# run = '/home/engels/bristles/bristles_RSI/DRAG_BASED_NFFT30/drag_based_kinematics/Re24/bristled_redesign_dB0.016/'

# QSM.parse_kinematics(run+'PARAMS_bristled.ini', run+'kinematics.t')
# QSM.fit_to_CFD(run, run+'PARAMS_bristled.ini', T0=1.0, optimize=False)