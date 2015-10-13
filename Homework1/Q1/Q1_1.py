__author__ = 'Ahmet'
__author__ = 'Ahmet'
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

days = 0,8,28,41,63,79,97,117,135,154
beetles = 2,47,192,256,768,896,1120,896,1184,1024

def calcError(args):
    error = 0;
    for i in range(len(days)):
        calcBeetles = f([args[0],beetles[0],args[1],days[i]])
        error += (calcBeetles-beetles[i])**2
    return error

def f(args):
    #K N0 r t
    calcBeetles = (args[0]*args[1])/(args[1]+(args[0]-args[1])*math.exp(-args[2]*args[3]))
    return calcBeetles


result = minimize(calcError, [1700,0.4],method = 'Nelder-Mead')
K = result.x[0]
r = result.x[1]
calcBeetles = np.zeros([len(days),1])
for i in range(len(days)):
    calcBeetles[i] = f([K,beetles[0],r,days[i]])

print(result)
plt.plot(days,calcBeetles)
plt.plot(days,beetles,'ro')
plt.show()
print(calcBeetles)