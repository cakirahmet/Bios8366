__author__ = 'Ahmet'
__author__ = 'Ahmet'
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

days = 0,8,28,41,63,79,97,117,135,154
beetles = 2,47,192,256,768,896,1120,896,1184,1024

def calcLikelihood(args):
    likelyhood = 0
    for i in range(len(beetles)):
        logft = f([args[0],beetles[0],args[1],days[i]])
        likelyhood += (-1*stats.norm.logpdf(np.log(beetles[i]),loc=logft,scale=args[2]))
    return likelyhood

def f(args):
    #K N0 r t
    try:
        calcBeetles = np.log((args[0]*args[1])/(args[1]+(args[0]-args[1])*np.exp(-args[2]*args[3])))
    except OverflowError:
        print("Error")
    return calcBeetles

logBeetles = np.zeros([len(beetles),1])
for i in range(len(beetles)):
    logBeetles[i] = math.log(beetles[i])
var0 = np.var(logBeetles)
result = minimize(calcLikelihood, [1700,0.4,var0],method = 'Nelder-Mead')
print(result)
K = result.x[0]
r = result.x[1]
calcBeetles = np.zeros([len(days),1])
for i in range(len(days)):
    calcBeetles[i] = np.exp(f([K,beetles[0],r,days[i]]))

print(result)
plt.plot(days,calcBeetles)
plt.plot(days,beetles,'ro')
plt.show()
print(calcBeetles)

