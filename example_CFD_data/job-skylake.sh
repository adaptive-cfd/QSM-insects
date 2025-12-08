#!/bin/bash
#MSUB -r WABBIT 	# Request name
#MSUB -n 240		# Number of tasks to use. On skylake, this would be 2x24 = 48 cores per node. 
#MSUB -T 86400 		# Elapsed time limit in seconds
#MSUB -o JOB.SKY.WABBIT.%I.out 	# Standard output. %I is the job id
#MSUB -e JOB.SKY.WABBIT.%I.err	# Error output. %I is the job id
#MSUB -q skylake	# partition: skylake
#MSUB -A gen14152	# Project ID
#MSUB -m scratch		# which dataspaces this job can use
#MSUB -@ thomas.engels@univ-amu.fr:begin,end
# ------

set -x
cd ${BRIDGE_MSUB_PWD}

#-----------------------------------------------------------------------
# skylake nodes:
# CPUs: 2x24-cores Intel Skylake@2.7GHz (AVX512)
# Cores/Node: 48
# Nodes: 1 656 Ttal cores: 79 488
# RAM node: 192GB core: 4.00 GB, LIMIT: 3.75 GB
#-----------------------------------------------------------------------
# Irene Skylake, 48 CPU/node 3.50 GB/core (hard limit: 3.75 GB/core)
# Nodes CPU    memory       Nodes  CPU      memory
# 1     48     168          21     1008     3528
# 2     96     336          22     1056     3696
# 3     144    504          23     1104     3864
# 4     192    672          24     1152     4032
# 5     240    840          25     1200     4200
# 6     288    1008         26     1248     4368
# 7     336    1176         27     1296     4536
# 8     384    1344         28     1344     4704
# 9     432    1512         29     1392     4872
# 10    480    1680         30     1440     5040
# 11    528    1848         31     1488     5208
# 12    576    2016         32     1536     5376
# 13    624    2184         33     1584     5544
# 14    672    2352         34     1632     5712
# 15    720    2520         35     1680     5880
# 16    768    2688         36     1728     6048
# 17    816    2856         37     1776     6216
# 18    864    3024         38     1824     6384
# 19    912    3192         39     1872     6552
# 20    960    3360         40     1920     6720
#-----------------------------------------------------------------------


# command to run the code
RUN="ccc_mprun"
# parameter file
INIFILE="PARAMS.ini"
# automatically resubmit the job or not
AUTO_RESUB=1
# maximum number of resubmissions
MAX_RESUB=20
# name of this file
JOBFILE="job-skylake.sh"


${RUN} ./wabbit ${INIFILE} --mem-per-core=3.50GB




if [ "$AUTO_RESUB" == "1" ]; then
	automatic_resubmission.sh "$JOBFILE" "$INIFILE" "$MAX_RESUB"
else
	mail -s "IRENE: non-resubmission job is done! $PWD" thomas.engels@univ-amu.fr < /dev/null
fi
