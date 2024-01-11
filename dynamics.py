import numpy as np
import csv

def getAerodynamicCoefficients(x0, AoA): 
    cl = x0[0] + x0[1]*np.sin(2.13*AoA - np.radians(7.20))
    cd = x0[2] + x0[3]*np.cos(2.04*AoA - np.radians(9.82))
    return cl, cd

def load_forces_data(file = 'forces_data_for_QSM.csv'):
    t = [] 
    Fx = [] 
    Fy = [] 
    Fz = [] 

    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=';') 
        c = 0 
        for line in reader:
            if c>= 2:     
                timeStep, fx, fy, fz = line
                t.append(float(timeStep))
                Fx.append(float(fx)) 
                Fy.append(float(fy))
                Fz.append(float(fz))
            c += 1
    return np.array(t), np.array(Fx), np.array(Fy), np.array(Fz)

