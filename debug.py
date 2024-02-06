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
    nBlades =  (2, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000)
    K_final = open('convergence?/K_final.csv', 'w')
    x0_0 = open('convergence?/x0_0.csv', 'w')
    x0_1 = open('convergence?/x0_1.csv', 'w')
    x0_2 = open('convergence?/x0_2.csv', 'w')
    x0_3 = open('convergence?/x0_3.csv', 'w')
    for blade in nBlades:
        print('computing for:', blade, 'blades') 
        result = main2(blade)
        K_final.write(str(blade) + ', ' + str(result[0][0]) + ', ' + str(result[1][0]) + '\n')
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

# Fl_deviation = standardDeviation('debug/100000Fl_magnitude_a.txt', 'debug/100000Fl_magnitude_n.txt')
# print(Fl_deviation)
# Fd_deviation = standardDeviation('debug/100000Fd_magnitude_a.txt', 'debug/100000Fd_magnitude_n.txt')
# print(Fd_deviation)
# Frot_deviation = standardDeviation('debug/100000Frot_magnitude_a.txt', 'debug/100000Frot_magnitude_n.txt')
# print(Frot_deviation)
