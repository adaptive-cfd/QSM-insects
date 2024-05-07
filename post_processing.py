from qsm_power import main
import os 
from datetime import datetime as dt


simulations_folder_path = '/Users/nico/Documents/school/thesis/wing_motion_python/single-wing-simulations/diptera'

insects_simulations = os.listdir(simulations_folder_path)

now = dt.now()
rightnow = now.strftime("%d-%m-%Y_%I:%M:%S")

def make_dir(dir_name, index=0):
    split = dir_name.split('/')  
    if len(split) >= index: 
        # print(len(split))
        # print(index)
        # print(split)
        new = '/'.join(split[:-1])
        make_dir(new, index=index+1) 
    new_dir = '/Users/nico/Documents/GitHub/QSM-insects/'+dir_name
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

for insect_simulation in insects_simulations:
    if insect_simulation == 'hoverfly-simulation': #this one can be commented out. it's currently the test folder but if commented out, all folders will be parsed. 
        insect_simulation_path = os.path.join(simulations_folder_path, insect_simulation)
        case_studies = os.listdir(insect_simulation_path)
        for case_study in case_studies:
            # print(insect_simulation_path)
            # exit()
            case_study_path = os.path.join(simulations_folder_path, insect_simulation, case_study)
            if os.path.isdir(case_study_path):
                cfd_runs = os.listdir(case_study_path)
                # print(case_study_path)
                # exit()
                for cfd_run in cfd_runs:
                    cfd_run_path = os.path.join(simulations_folder_path, insect_simulation, case_study, cfd_run)
                    if os.path.isdir(cfd_run_path):
                        folder_name = f'post-processing/{insect_simulation}/{case_study}/{cfd_run}/{rightnow}'
                        make_dir(folder_name)
                        forces_coefficients, moment_coefficient, power_coefficient = main(cfd_run_path, folder_name)
                        writeArraytoCSV(['Cl1', 'Cl2', 'Cd1', 'Cd2', 'Crot', 'Cam1', 'Cam2', 'Crd', 'K0_forces'], forces_coefficients, folder_name+'/forces_coefficients.csv')
                        writeArraytoCSV(['C_lever (optimized lever)', 'Lever (non-optimized lever)', 'K0_moment'], moment_coefficient, folder_name+'/moment_coefficient.csv')
                        writeArraytoCSV(['C_power', 'K0_power'],  power_coefficient, folder_name+'/power_coefficient.csv')