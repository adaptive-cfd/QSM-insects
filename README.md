A PYTHON implementation of a quasi-steady model for flapping flight.

The qsm_class provides a QSM object that represents a model for one wing. If a complete insect is to be modeled, two or four of these objects can be combined.

The code heavily relies on adaptive-cfd/python-tools, and specifically the insect_tools.py file in which many basic scripts, like rotation matrices etc are collected.

The QSM code is designed to work with adaptive-cfd/WABBIT, a high performance code for CFD of flapping flight. However, it can (probably with minor modifications) also be used together with any other code or experimental data.
Using the QSM code requires parsing kinematics (i.e., calculate, based on the three angles that describe a rigid wing motion, quantities like angular velocity, angular acceleration, etc), and learning the model coefficients from some reference data, 
in our case this is CFD data. Then, the model can be evaluated for other kinematics as well (for example when optimizing kinematics, etc).

While operational, the QSM model here is still under development. 

Authors:

Nicolas Munoz Giraldo, TU Berlin
Thomas Engels, CNRS & Aix-Marseille University (for correspondence: thomas.engels@univ-amu.fr)



