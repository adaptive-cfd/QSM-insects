import numpy as np 
def writeArraytoFile(arr, file): 
    with open(file, 'w') as fileWriter: 
        for val in arr:
            if isinstance(val, np.ndarray):
                fileWriter.writelines(str(val.flatten())+'\n')
            else: 
                fileWriter.writelines(str(val)+'\n')
