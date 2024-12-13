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
run = '/home/engels/musca_model/simulation/compensation/compensation_singleWing_3parameters/intact_wing/phi110.00_phim15.00_dTau0.05/'

QSM = qsm_class.QSM(nt=400)
QSM.setup_wing_shape(run+'WING_musca_noAlula.ini')
QSM.parse_kinematics(run+'PARAMS.ini', run+'kinematics.t')
QSM.read_CFD_data(run, run+"PARAMS.ini", T0=1.0)
QSM.fit_to_CFD(optimize=True, plot=True)
