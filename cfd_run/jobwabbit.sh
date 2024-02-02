#!/bin/bash
#SBATCH --job-name=fly-kine-compensation      # nom du job
#SBATCH -A sch@cpu
#SBATCH --ntasks=80                # Nombre total de processus MPI
#SBATCH --ntasks-per-node=40       # Nombre de processus MPI par noeud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread       # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=05:00:00            # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=JOBWABBIT%j.out  # Nom du fichier de sortie
#SBATCH --error=JOBWABBIT%j.out   # Nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd ${SLURM_SUBMIT_DIR}

module purge
module load hdf5/1.10.5/intel-19.0.4-mpi intel-mpi/19.0.4 intel-mkl/19.0.4
module load intel-compilers/19.0.4 intel-all/19.0.4
export MAKEFLAGS="-j8"
export HDF_ROOT=/gpfslocalsup/spack_soft/hdf5/1.10.5/intel-19.0.4-lnysdcbocfokaq4yxc72wiplpfknw7e6

#---------------------------------------------------
# memory: 192 GB/node
# cores: 2x20 = 40 / node
# mempercore: 4.8 GB/core
#---------------------------------------------------

# command to run the code
RUN="srun"
# parameter file
INIFILE="PARAMS.ini"
# how much memory at most?
MEMORY="384.0GB"
# automatically resubmit the job or not
AUTO_RESUB=1
# maximum number of resubmissions
MAX_RESUB=20
# the name of this job script:
JOBFILE="jobwabbit.sh"

# watchdog checks if the job is still alive
# note only *.h5 *.t *.dat *.out *.err are checked
simulation_watchdog.py --ID=${BRIDGE_MSUB_JOBID} --dir=${BRIDGE_MSUB_PWD} --jobfile=${JOBFILE} &

${RUN} ./wabbit ${INIFILE} --memory=${MEMORY}

if [ "$AUTO_RESUB" == "1" ]; then
	automatic_resubmission.sh "$JOBFILE" "$INIFILE" "$MAX_RESUB"
fi
