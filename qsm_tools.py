#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: nico
"""
##%kinematics
import os
import csv
os.environ['QTA_QP_PLATFORM'] = 'wayland'
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib import animation
import functools 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from python_tools_master import insect_tools as it 
from python_tools_master import wabbit_tools as wt
from debug import writeArraytoFile
from datetime import datetime
import time


#timestamp variable for saving figures with actual timestamp 
now = datetime.now()
rightnow = now.strftime(" %d-%m-%Y_%I-%M")+".png"



def animationPlot(ax, timeStep):
    #get point set by timeStep number
    points = globalPointsSequence[timeStep] #pointsSequence can either be global, body, stroke 
    #clear the current axis 
    ax.cla()

    # extract the x, y and z coordinates 
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    #axis limit
    a = 2

    trajectory = np.array(globalPointsSequence)[:, wingtip_index]
    #3D trajectory 
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='k', linestyle='dashed', linewidth=0.5)

    #x-y plane trajectory 
    ax.plot(trajectory[:, 0], trajectory[:, 1], 0*trajectory[:, 2]-a, color='k', linestyle='dashed', linewidth=0.5)
    
    #z-y plane prajectory 
    ax.plot(0*trajectory[:, 0]-a, trajectory[:, 1], trajectory[:, 2], color='k', linestyle='dashed', linewidth=0.5)

    #x-z plane prajectory 
    ax.plot(trajectory[:, 0], 0*trajectory[:, 1]+a, trajectory[:, 2], color='k', linestyle='dashed', linewidth=0.5)
    
    # #use axis to plot surface 
    ax.plot_trisurf(X, Y, Z, edgecolors='black') 
    ax.add_collection3d(Poly3DCollection(verts=[points]))

    #shadows
    #x-y plane shadow
    XY_plane_shadow = np.vstack((X, Y, -a*np.ones_like(Z))).transpose()
    ax.add_collection3d(Poly3DCollection(verts=[XY_plane_shadow], color='#d3d3d3'))
    #y-z plane shadow
    YZ_plane_shadow = np.vstack((-a*np.ones_like(X), Y, Z)).transpose()
    ax.add_collection3d(Poly3DCollection(verts=[YZ_plane_shadow], color='#d3d3d3'))
    #x-z plane shadow
    XZ_plane_shadow = np.vstack(((X, a*np.ones_like(Y), Z))).transpose()
    ax.add_collection3d(Poly3DCollection(verts=[XZ_plane_shadow], color='#d3d3d3'))
    
    #velocity at wingtip 
    u_wing_g_plot = us_wing_g[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], u_wing_g_plot[0], u_wing_g_plot[1], u_wing_g_plot[2], color='orange', label=r'$\overrightarrow{u}^{(g)}_w$' )
    
    #drag
    e_dragVector_wing_g = e_dragVectors_wing_g[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], e_dragVector_wing_g[0], e_dragVector_wing_g[1], e_dragVector_wing_g[2], color='green', label=r'$\overrightarrow{e_D}^{(g)}_w$' )
    
    #lift 
    liftVector_g = e_liftVectors_g[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], liftVector_g[0], liftVector_g[1], liftVector_g[2], color='blue', label=r'$\overrightarrow{e_L}^{(g)}_w$')
    
    #ez_wing_g 
    ez_wing_g_sequence_plot = ez_wing_g_sequence[timeStep]
    ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], ez_wing_g_sequence_plot[0], ez_wing_g_sequence_plot[1], ez_wing_g_sequence_plot[2], color='red', label=r'$\overrightarrow{e_{F_{am}}}^{(g)}_w$')

    # #e_Fam 
    # e_Fam_plot = e_Fam[timeStep]
    # ax.quiver(X[wingtip_index], Y[wingtip_index], Z[wingtip_index], e_Fam_plot[0], e_Fam_plot[1], e_Fam_plot[2], color='red', label=r'$\overrightarrow{e_{F_{am}}}^{(g)}_w$')

    ax.legend()
    
    #set the axis limits
    ax.set_xlim([-a, a])
    ax.set_ylim([-a, a])
    ax.set_zlim([-a, a])

    #set the axis labels 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Timestep: {timeStep}, ⍺: {np.round(np.degrees(alphas[timeStep]), 2)}, AoA: {np.round(np.degrees(AoA[timeStep]), 2)} \nFl: {np.round(Ftc[timeStep], 4)} \nFd: {np.round(Ftd[timeStep], 4)} \nFrot: {np.round(Frc[timeStep], 4)} \nFam: {np.round(Fam[timeStep], 4)}')

# run the live animation of the wing 
def generatePlotsForKinematicsSequence():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    anim = animation.FuncAnimation(fig, functools.partial(animationPlot, ax), frames=len(timeline), repeat=True)
    #anim.save('u&d_vectors.gif') 
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



############################################################################################################################################################################################
##%% dynamics

