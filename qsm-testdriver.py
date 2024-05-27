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

plt.close('all')

run = '/home/engels/Documents/Research/Insects/3D/projects/hoverfly_QSM_wageningen/diptera_project/simulations_corrected/Bibio_marcis/'
QSM = qsm_class.QSM()
QSM.setup_wing_shape(run+'WING_Bibio_marcis.ini')
QSM.parse_kinematics(run+'PARAMS_diptera.ini', run+'kinematics.t')
QSM.fit_to_CFD(run, run+'PARAMS_diptera.ini', T0=1.0)


# plt.figure()
# plt.plot(QSM.timeline, QSM.acc_wing_g[:,0])
# plt.plot(QSM.timeline+1.0, QSM.acc_wing_g[:,0])


# plt.figure()
# plt.plot(QSM.timeline, QSM.F_CFD_w[:,0], 'g-')
# plt.plot(QSM.timeline, QSM.F_CFD_w[:,1], 'r-')
# plt.plot(QSM.timeline, QSM.F_CFD_w[:,2], 'b-')



# plt.figure()
# plt.plot(QSM.timeline, QSM.M_CFD_w[:,0], 'g-')
# plt.plot(QSM.timeline, QSM.M_CFD_w[:,1], 'r-')
# plt.plot(QSM.timeline, QSM.M_CFD_w[:,2], 'b-')


# plt.figure()
# plt.plot(QSM.timeline, QSM.rots_wing_w[:,0], 'g-')
# plt.plot(QSM.timeline, QSM.rots_wing_w[:,1], 'r-')
# plt.plot(QSM.timeline, QSM.rots_wing_w[:,2], 'b-')

# plt.plot(QSM.debug_time, QSM.debug_rotx_wing_g, 'g--')
# plt.plot(QSM.debug_time, QSM.debug_roty_wing_g, 'r--')
# plt.plot(QSM.debug_time, QSM.debug_rotz_wing_g, 'b--')



# plt.figure()
# plt.plot(QSM.timeline, QSM.rot_acc_wing_w[:,0], 'g-')
# plt.plot(QSM.timeline, QSM.rot_acc_wing_w[:,1], 'r-')
# plt.plot(QSM.timeline, QSM.rot_acc_wing_w[:,2], 'b-')

# plt.plot(QSM.debug_time, QSM.debug_rotx_dt_wing_g, 'g--')
# plt.plot(QSM.debug_time, QSM.debug_roty_dt_wing_g, 'r--')
# plt.plot(QSM.debug_time, QSM.debug_rotz_dt_wing_g, 'b--')


# plt.figure()
# plt.plot(QSM.timeline, QSM.Fz_CFD_g)
# plt.plot(QSM.timeline+1.0, QSM.Fz_CFD_g)



# ax = plt.figure().add_subplot(projection='3d')


# ax.plot(QSM.ey_wing_g_sequence[:,0], QSM.ey_wing_g_sequence[:,1], QSM.ey_wing_g_sequence[:,2])
# plt.axis('equal')