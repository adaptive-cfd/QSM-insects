from kinematics import kinematics, getChordLength, generatePlotsForKinematicsSequence, load_kinematics_data
from dynamics import getAerodynamicCoefficients, load_forces_data
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from debug import writeArraytoFile
import numpy as np 
import scipy.optimize as opt
import time
import cProfile
import pstats
import io

def F(x, timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, u_wing_g_vectors, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index, Fx_CFD_interp, Fy_CFD_interp, Fz_CFD_interp, Fx_CFD, Fy_CFD, Fz_CFD, show_plots=False):
    AoA_final = np.zeros((AoA.shape[0], 3))
    for wingpoints_AoA in AoA:
        #aoa = np.max(wingpoints_AoA)
        AoA_final[wingPoints, :] = wingpoints_AoA
    #print('AoA:', AoA_final)
    cl, cd = getAerodynamicCoefficients(x, np.array(AoA_final))
    if show_plots: 
        # plt.plot(timeline, np.degrees(phis), label='ɸ')
        # plt.plot(timeline, np.degrees(alphas), label ='⍺')
        # plt.plot(timeline, np.degrees(thetas), label='Θ')
        # plt.legend()
        # plt.show()

        plt.plot(timeline, np.degrees(AoA_final), label='AoA')
        plt.xlabel('t/T')
        plt.legend()
        plt.show()
        

    min_y = np.min(wingPoints[:, 1])
    max_y = np.max(wingPoints[:, 1])
    y_space = np.linspace(min_y, max_y, 100)

    c = getChordLength(wingPoints, y_space)
    c_norm = c / np.max(c)
    #print(c_norm)

    #y_space = y_space/40
    c_norm_interpolation = interp1d(y_space, c_norm)


    def Cr2(r): 
        return c_norm_interpolation(r) * r**2
    #fxn evaluated at the intervals 
    F_r = Cr2(y_space)

    I = trapz(F_r, y_space)
    # print(I)
    # print('y space: ', y_space)

    rots_wing_w = np.array(rots_wing_w)
    planarOmegaSq = rots_wing_w[:, 0]**2 + rots_wing_w[:, 2]**2 

    rho = 1.225
    Fl_mag = 0.5*rho*cl*planarOmegaSq.flatten()*I
    Fd_mag = 0.5*rho*cd*planarOmegaSq.flatten()*I

    Fl = np.zeros((timeline.shape[0], 3))
    Fd = np.zeros((timeline.shape[0], 3))
    for i in range(timeline.shape[0]):
        Fl[i,:] = (Fl_mag[i] * e_liftVectors[i])
        Fd[i,:] = (Fd_mag[i] * dragVectors_wing_g[i])

    # Fl = np.array(Fl)
    Fd = np.array(Fd)

    Fx_QSM = Fl[:, 0]+Fd[:, 0]
    Fy_QSM = Fl[:, 1]+Fd[:, 1]
    Fz_QSM = Fl[:, 2]+Fd[:, 2]

    K_num = np.linalg.norm(Fx_QSM-Fx_CFD_interp(timeline)) + np.linalg.norm(Fz_QSM-Fz_CFD_interp(timeline))
    K_den = np.linalg.norm(Fx_CFD_interp(timeline) + np.linalg.norm(Fz_CFD_interp(timeline)))
    if K_den != 0: 
        K = K_num/K_den
    else:
        K = K_num
    #print('K:', K)
    if show_plots:
        graphAoA = np.linspace(-9, 90, 100)*(np.pi/180)
        gCl, gCd = getAerodynamicCoefficients(x, graphAoA)
        fig, ax = plt.subplots()
        ax.plot(np.degrees(graphAoA), gCl, label='Cl')
        ax.plot(np.degrees(graphAoA), gCd, label='Cd')
        ax.set_xlabel('AoA[°]')
        plt.legend()
        plt.show()

        plt.plot(timeline[:], Fx_QSM, label='Fx_QSM', color='red')
        plt.plot(timeline[:], Fz_QSM, label='Fz_QSM', color='blue')
        plt.plot(timeline[:], Fx_CFD_interp(timeline), label='Fx_CFD', linestyle = 'dashed', color='red')
        plt.plot(timeline[:], Fz_CFD_interp(timeline), label='Fz_CFD', linestyle = 'dashed', color='blue')
        plt.xlabel('t/T')
        plt.ylabel('Force')
        plt.title(f'Fx_QSM/Fx_CFD = {np.round(np.linalg.norm(Fx_QSM)/np.linalg.norm(Fx_CFD_interp(timeline)), 3)}; Fz_QSM/Fz_CFD = {np.round(np.linalg.norm(Fz_QSM)/np.linalg.norm(Fz_CFD_interp(timeline)), 3)}')
        plt.legend()
        plt.show()

        generatePlotsForKinematicsSequence(timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA_final, u_wing_g_vectors, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index, Fl, Fd)
    return K 

####Optimization 
def main():
    timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, u_wing_g_vectors, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index = kinematics()
    t, Fx_CFD, Fy_CFD, Fz_CFD = load_forces_data()
    Fx_CFD_interp = interp1d(t, Fx_CFD, fill_value='extrapolate')
    Fy_CFD_interp = interp1d(t, Fy_CFD, fill_value='extrapolate')
    Fz_CFD_interp = interp1d(t, Fz_CFD, fill_value='extrapolate')
    t, alpha_CFD, phi_CFD, theta_CFD, alpha_dot_CFD, phi_dot_CFD, theta_dot_CFD = load_kinematics_data() 
    x_0 = [0.03433548, -0.01193863,  0.0338657,  -0.023361]
    bounds = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]
    optimize = True
    if optimize:
        start = time.time()
        optimization = opt.differential_evolution(F, args=(timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, u_wing_g_vectors, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index, Fx_CFD_interp, Fy_CFD_interp, Fz_CFD_interp, Fx_CFD, Fy_CFD, Fz_CFD), bounds=bounds, x0=x_0, maxiter=20)
        x_final = optimization.x
        K_final = optimization.fun
        print('completed in:', round(time.time() - start, 3), ' seconds')
    else:
        x_final = [0.03433548, -0.01193863,  0.0338657,  -0.023361]
        K_final = 0.09349021020747196

    print('x0_final: ', x_final, 'K_final: ', K_final)
    F(x_final, timeline, globalPointsSequence, bodyPointsSequence, strokePointsSequence, wingPoints, phis, alphas, thetas, rots_wing_b, rots_wing_w, us_wing_w, us_wing_g, verifying_us_wing_g, us_wind_w, AoA, u_wing_g_vectors, dragVectors_wing_g, e_liftVectors, wingtip_index, pivot_index, Fx_CFD_interp, Fy_CFD_interp, Fz_CFD_interp, Fx_CFD, Fy_CFD, Fz_CFD, True)
    # print('K:', K)

# profile = cProfile()
# profile.enable()
# main()
# profile.disable()
# s = io.StringIO()
# ps = pstats.Stats(profile, stream=s).sort_stats('cumulative')
# ps.print_stats()
# with open('profile.txt', 'w') as file:
#     file.write(s.getvalue())

# writeArraytoFile(Fl, 'lift.txt')
# writeArraytoFile(Fd, 'drag.txt')
# writeArraytoFile(np.array(rots_wing_w), 'omegaW_w.txt')
# exit()

# plt.plot(timeline, Fl_mag, label='lift')
# plt.plot(timeline, Fd_mag, label='drag')

# plt.legend()
# plt.show() 

# plt.plot(timeline, Fl[:, 0]+Fd[:, 0], label='Fx')
# plt.plot(timeline, Fl[:, 2]+Fd[:, 2], label='Fz')

# plt.legend()
# plt.show() 

# plt.plot(timeline, np.array(e_liftVectors)[:, 2], label='lift norm vector z')

# plt.legend()
# plt.show() 

# print('Fl: ', np.array(Fl))
# print('Fd: ', np.array(Fd))

# Fl = []
# Fd = []
# for i, wing_point in enumerate(wingPoints):
#     u = np.array(us_wind_w)[:, i]
#     u_squared = (u[:, 0]**2 + u[:, 1]**2 + u[:, 2]**2 )
#     Fl_for_all_timesteps = 0.5 * rho * c_norm_interpolation(wing_point[1]) * u_squared * cl[:, i]
#     Fd_for_all_timesteps = 0.5 * rho * c_norm_interpolation(wing_point[1]) * u_squared * cd[:, i]
#     Fl.append(Fl_for_all_timesteps)
#     Fd.append(Fd_for_all_timesteps)

# Fl = np.array(Fl)
# Fd = np.array(Fd)

#Note: both Fl and Fd are arrays that store the Fl and Fd respectively for [wing_point][timestep]
#so if you want to extract the Fl and Fd for wing_point 2 at time step 10 (of 360 timesteps)
#you print Fl[2][10]

# print(Fl)
# print(Fd)
# print('Cl =', cl)
# print('\nCd =', cd)


# x0_final:  [ 0.0041443   0.02452099  0.03143651 -0.01875035] K_final:  0.536316657116528
