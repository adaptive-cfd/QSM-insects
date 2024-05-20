from qsm import main
from qsm_diptera_wrong import main as main_diptera
import os 
from datetime import datetime as dt
import csv
import numpy as np
import datetime


simulations_folder_path = '/home/nico/Documents/school/thesis/wing_motion_python/single-wing-simulations/'

insects_simulations = os.listdir(simulations_folder_path)

now = dt.now()
rightnow = now.strftime("%d-%m-%Y_%I-%M")

def make_dir(dir_name, index=0):
    split = dir_name.split('/')  
    if len(split) >= index: 
        # print(len(split))
        # print(index)
        # print(split)
        new = '/'.join(split[:-1])
        make_dir(new, index=index+1) 
    new_dir = '/home/nico/Documents/GitHub/QSM-insects/'+dir_name
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir

def writeArraytoCSV(header, array, file_name):
    with open(file_name, 'w') as file:
        for i, n in enumerate(header): 
            file.write(str(n))
            if i != len(header)-1:
                file.write(', ')
        file.write('\n')
        for i, n in enumerate(array): 
            file.write(str(n))
            if i != len(array)-1:
                file.write(', ')
        file.write('\n')

def readArrayfromCSV(file_name):
    header = []
    body = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader): 
            if i == 0:
                header = line
            else:
                body.append(line)
    return header, np.array(body)
        

# for insect_simulation in insects_simulations:
#     if insect_simulation == 'musca-simulations': #this one can be commented out. it's currently the test folder but if commented out, all folders will be parsed. 
#         insect_simulation_path = os.path.join(simulations_folder_path, insect_simulation)
#         case_studies = os.listdir(insect_simulation_path)
#         for case_study in case_studies:
#             # print(insect_simulation_path)
#             # exit()
#             case_study_path = os.path.join(simulations_folder_path, insect_simulation, case_study)
#             if os.path.isdir(case_study_path):
#                 cfd_runs = os.listdir(case_study_path)
#                 # print(case_study_path)
#                 # exit()
#                 for cfd_run in cfd_runs:
#                     cfd_run_path = os.path.join(simulations_folder_path, insect_simulation, case_study, cfd_run)
#                     # print(cfd_run_path)
#                     # exit()
#                     if os.path.isdir(cfd_run_path):
#                         folder_name = f'post-processing/{insect_simulation}/{case_study}/{cfd_run}/{rightnow}'
#                         make_dir(folder_name)
#                         forces_coefficients, moment_coefficient, power_coefficient = main(cfd_run_path, folder_name)
#                         writeArraytoCSV(['Cl1', 'Cl2', 'Cd1', 'Cd2', 'Crot', 'Cam1', 'Cam2', 'Crd', 'K0_forces'], forces_coefficients, folder_name+'/force_coefficients.csv')
#                         writeArraytoCSV(['C_lever_x_w (lever x-component optimized for moments)', 'Lever (lever y-component optimized for moments)', 'K0_moments'], moment_coefficient, folder_name+'/moment_coefficients.csv')
#                         writeArraytoCSV(['C_lever_x_w (lever x-component optimized for power)', 'Lever (lever y-component optimized for power)', 'K0_power'],  power_coefficient, folder_name+'/power_coefficients.csv')

for insect_simulation in insects_simulations:
    if insect_simulation == 'diptera': #this one can be commented out. it's currently the test folder but if commented out, all folders will be parsed. 
        insect_simulation_path = os.path.join(simulations_folder_path, insect_simulation)
        case_studies = os.listdir(insect_simulation_path)
        for case_study in case_studies:
            # print(insect_simulation_path)
            # exit()
            if case_study != 'diptera-simulation-wrong': 
                continue
            case_study_path = os.path.join(simulations_folder_path, insect_simulation, case_study)
            if os.path.isdir(case_study_path):
                cfd_runs = os.listdir(case_study_path)
                # print(case_study_path)
                # exit()
                for cfd_run in cfd_runs:
                    post_processing_coefficients = os.path.join('/home/nico/Documents/GitHub/QSM-insects/post-processing/diptera-simulation/case_study/', cfd_run)
                    cfd_run_path = os.path.join(simulations_folder_path, insect_simulation, case_study, cfd_run)
                    # print(cfd_run_path)
                    # exit()
                    if os.path.isdir(cfd_run_path):
                        folder_name = f'post-processing/{case_study}/{cfd_run}/{rightnow}'
                        post_processing_coefficients = f'post-processing/diptera-simulation/case_study/{cfd_run}/'
                        dates = os.listdir(post_processing_coefficients)
                        timestamps = []
                        for date in dates: 
                            _date = datetime.datetime.strptime(date, "%d-%m-%Y_%I-%M").timestamp()
                            timestamps.append(_date)
                        latest = max(timestamps)
                        date_index = timestamps.index(latest)
                        latest_run = dates[0]
                        post_processing_coefficients = f'/home/nico/Documents/GitHub/QSM-insects/post-processing/diptera-simulation/case_study/{cfd_run}/{latest_run}'
                        make_dir(folder_name)
                        x0_forces = readArrayfromCSV(post_processing_coefficients+'/forces_coefficients.csv')[1]
                        x0_moments = readArrayfromCSV(post_processing_coefficients+'/moment_coefficient.csv')[1]
                        x0_power = readArrayfromCSV(post_processing_coefficients+'/power_coefficient.csv')[1]
                        main_diptera(cfd_run_path, folder_name, False, x0_forces, x0_moments, x0_power)