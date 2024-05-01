from qsm import main
import os 

simulations_folder_path = '/home/nico/Documents/knowledge/thesis/wing_motion_python/single-wing-simulations'

insects_simulations = os.listdir(simulations_folder_path)
print(insects_simulations)

for insect_simulation in insects_simulations:
    if insect_simulation == 'musca-simulations':
        insect_simulation_path = os.path.join(simulations_folder_path, insect_simulation)
        case_studies = os.listdir(insect_simulation_path)
        for case_study in case_studies:
            case_study_path = os.path.join(simulations_folder_path, insect_simulation, case_study)
            if os.path.isdir(case_study_path):
                cfd_runs = os.listdir(case_study_path)
                for cfd_run in cfd_runs:
                    cfd_run_path = os.path.join(simulations_folder_path, insect_simulation, case_study, cfd_run)
                    if os.path.isdir(cfd_run_path):
                        results = main(cfd_run_path)