import numpy as np 

def writeArraytoFile(arr, file): 
    with open(file, 'w') as fileWriter: 
        for val in arr:
            if isinstance(val, np.ndarray):
                fileWriter.writelines(str(val.flatten())+'\n')
            else: 
                fileWriter.writelines(str(val)+'\n')

def convergenceTest():
    from qsm_copy import main2
    nBlades =  (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 500, 1000, 2000, 3000, 4000, 5000)
    K_final = open('convergence/K_final_b.csv', 'w')
    x0_0 = open('convergence/x0_0_b.csv', 'w')
    x0_1 = open('convergence/x0_1_b.csv', 'w')
    x0_2 = open('convergence/x0_2_b.csv', 'w')
    x0_3 = open('convergence/x0_3_b.csv', 'w')
    for blade in nBlades:
        print('computing for:', blade, 'blades') 
        result = main2(blade)
        K_final.write(str(blade) + ', ' + str(result[0][0]) + ', ' + str(result[1][0]) + '\n') #1st column is # of blades, 2nd column is the value of the variable for the analytical run and 3rd column is that of the numerical run
        x0_0.write(str(blade) + ', ' + str(result[0][1][0]) + ', ' + str(result[1][1][0]) + '\n')
        x0_1.write(str(blade) + ', ' + str(result[0][1][1]) + ', ' + str(result[1][1][1]) + '\n')
        x0_2.write(str(blade) + ', ' + str(result[0][1][2]) + ', ' + str(result[1][1][2]) + '\n')
        x0_3.write(str(blade) + ', ' + str(result[0][1][3]) + ', ' + str(result[1][1][3]) + '\n')
    K_final.close()
    x0_0.close()
    x0_1.close()
    x0_2.close()
    x0_3.close()
convergenceTest()

def convergenceTestForces(): 
    from qsm_copy import main2
    nBlades =  (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 500, 1000, 2000, 3000, 4000, 5000)
    Fl_mean = open('convergence/Fl_mean.csv', 'w')
    Fd_mean = open('convergence/Fd_mean.csv', 'w')
    Frot_mean = open('convergence/Frot_mean.csv', 'w')
    for blade in nBlades: 
        print('computing for:', blade, 'blades') 
        result = main2(blade)
        Fl_mean.write(str(blade) + ', ' + str(result[0][0]) + ', ' + str(result[1][0]) + '\n')
        Fd_mean.write(str(blade) + ', ' + str(result[0][1]) + ', ' + str(result[1][1]) + '\n')
        Frot_mean.write(str(blade) + ', ' + str(result[0][2]) + ', ' + str(result[1][2]) + '\n')
    Fl_mean.close()
    Fd_mean.close()
    Frot_mean.close()
# convergenceTestForces()


def standardDeviation(file_a, file_n): 
    with open(file_a, 'r') as file1: 
        file_analytical = file1.readlines()
    with open(file_n, 'r') as file2: 
        file_numerical = file2.readlines()
    N = 0 
    error_sqrd = 0 
    for analytical, numerical in zip(file_analytical, file_numerical):
        error = float(analytical) - float(numerical)
        error_sqrd += error**2 
        N += 1
    return np.sqrt(error_sqrd / N)

# nb = 500000
# print('computing standard devition for', nb, 'blades')
# Fl_deviation = standardDeviation('debug/' + str(nb) + '_Fl_magnitude_a.txt', 'debug/' + str(nb) + '_Fl_magnitude_n.txt')
# print(np.round(Fl_deviation, 4))
# Fd_deviation = standardDeviation('debug/' + str(nb) + '_Fd_magnitude_a.txt', 'debug/' + str(nb) + '_Fd_magnitude_n.txt')
# print(np.round(Fd_deviation, 4))
# Frot_deviation = standardDeviation('debug/' + str(nb) + '_Frot_magnitude_a.txt', 'debug/' + str(nb) + '_Frot_magnitude_n.txt')
# print(np.round(Frot_deviation, 4))
