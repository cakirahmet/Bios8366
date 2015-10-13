__author__ = 'Ahmet'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def parse_latlon(x):
    d, m, s = map(float, x.split(':'))
    ms = m/60. + s/3600.
    if d<0:
        return d - ms
    return d + ms

def cooling(start_temp,alpha):
    T=start_temp
    while True:
        yield T
        T=alpha*T

def calculateDistance(cities):
    #diff = math.sqrt(np.diff(lat)**2 + np.diff(long)**2)
    #distance = np.sum(np.abs(cityLocs)**2,axis=-1)**(1./2)
    #distance = np.sum(np.abs(cityLocs)**2,axis=-1)**(1./2)
    dist = 0
    for i in range(len(cities)-1):
        dist += math.sqrt((cities[i][1]-cities[i+1][1])**2 + (cities[i][2]-cities[i+1][2])**2)
        print(i,dist)
    return dist

cities =  pd.read_csv('/Users/Ahmet/Box Sync/Classes/Vanderbilt/AdvancedStatisticalComputing/Bios8366/data/brasil_capitals.txt',
                      names=['city','lat','lon'])[['lat','lon']].applymap(parse_latlon)

beginTemp = 10
endTemp = 0.001
coolingFactor = 0.9
period = 15
iterator = 0

bestCityOrder = np.transpose(np.array((range(len(cities)),cities['lat'],cities['lon']),np.float64))
bestDistance = calculateDistance(bestCityOrder)
bestDistances = []
bestDistances.append(bestDistance)



for j in range(period):

    temperature = beginTemp

    newCities = bestCityOrder[:]
    newDistance = bestDistance

    currentDistance = bestDistance
    currentCities = bestCityOrder[:]

    while temperature > endTemp:
        citiesToSwap = np.random.random_integers(1,len(currentCities),size=(1,2))
        newCities[citiesToSwap[0]], newCities[citiesToSwap[1]] = newCities[citiesToSwap[1]], newCities[citiesToSwap[0]]
        newDistance = calculateDistance(newCities)

        if newDistance < currentDistance or math.exp((currentDistance-newDistance)/temperature) > np.random.random():
            currentDistance = newDistance
            currentCities = newCities[:]
        else:
            newDistance = currentDistance
            newCities = currentCities[:]
        if newDistance < bestDistance:
            bestDistance = newDistance
            bestCityOrder = newCities[:]
            bestDistances.append(bestDistance)
            temperature = temperature * coolingFactor


print(bestDistances)