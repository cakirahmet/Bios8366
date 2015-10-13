__author__ = 'Ahmet'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from copy import deepcopy


def parse_latlon(x):
    d, m, s = map(float, x.split(':'))
    ms = m / 60. + s / 3600.
    if d < 0:
        return d - ms
    return d + ms


def swapCities(cities, nSwap):
    if nSwap % 2 != 0:
        nSwap -= 1
    nSwap = int(max(nSwap, 2))
    nSwap = int(nSwap / 2)
    citiesToSwap = np.random.choice(len(cities) - 1,nSwap*2,replace=False)
    for i in range(nSwap):
        tempCity = np.copy(cities[citiesToSwap[i*2]])
        cities[citiesToSwap[i*2]] = cities[citiesToSwap[i*2+1]]
        cities[citiesToSwap[i*2+1]] = tempCity
        #print(i,' ',nSwap,'\n')
    return cities


def cooling(start_temp, alpha):
    T = start_temp
    while True:
        yield T
        T = alpha * T

def distance(lat1,lon1,lat2,lon2):
    p = 0.017453292519943295 #math.pi / 180
    a = 0.5 - math.cos((lat2 - lat1) * p)/2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))

def calculateTotalDistance(cities):
    # diff = math.sqrt(np.diff(lat)**2 + np.diff(long)**2)
    # distance = np.sum(np.abs(cityLocs)**2,axis=-1)**(1./2)
    # distance = np.sum(np.abs(cityLocs)**2,axis=-1)**(1./2)
    dist = 0
    for i in range(len(cities) - 1):
        dist += math.sqrt((cities[i][1] - cities[i + 1][1]) ** 2 + (cities[i][2] - cities[i + 1][2]) ** 2)
        #distance(cities[i][1],cities[i][2],cities[i + 1][1],cities[i + 1][2])
        #math.sqrt((cities[i][1] - cities[i + 1][1]) ** 2 + (cities[i][2] - cities[i + 1][2]) ** 2)
        # print(i,dist)
    return dist


cities = pd.read_csv(
    'C:/Users/cakira/Box Sync/Classes/Vanderbilt/AdvancedStatisticalComputing/Bios8366/data/brasil_capitals.txt',
    names=['city', 'lat', 'lon'])[['lat', 'lon']].applymap(parse_latlon)

beginTemp = 10
endTemp = 0.001
coolingFactor = 0.9
period = 240

cityArray = np.transpose(np.array((range(len(cities)), cities['lat'], cities['lon']), np.float64))
start = [22,23,24,25,17,16,13,3,7,9,11,14,19,18,21,20,15,12,10,4,0,1,2,6,8,5];
bestCityOrder = np.random.permutation(cityArray)
for i in range(len(start)):
    bestCityOrder[i] = cityArray[start[i]]

numSwap = 2
bestDistances = []
bestCityOrder = np.random.permutation(cityArray)
bestDistance = calculateTotalDistance(bestCityOrder)
bestDistances.append(bestDistance)

#currentCities = np.random.permutation(bestCityOrder)
#currentDistance = calculateTotalDistance(currentCities)

    #print(bestCityOrder)

for j in range(period):

    temperature = beginTemp

    currentDistance = bestDistance
    currentCities = np.copy(bestCityOrder)

    newCities = np.copy(currentCities)
    newDistance = currentDistance



    while True:
        newCities = swapCities(newCities, numSwap)
        newDistance = calculateTotalDistance(newCities)

        if newDistance < currentDistance or math.exp(
                        (currentDistance - newDistance) / temperature) > np.random.random():
            currentDistance = newDistance
            currentCities = np.copy(newCities)
        else:
            newDistance = currentDistance
            newCities = currentCities[:]
        if newDistance < bestDistance:
            bestCityOrder = deepcopy(newCities)
            bestDistance = newDistance
            # a = bestCityOrder[:]
            #print(bestCityOrder)
            # print(bestDistance)
            bestDistances.append(bestDistance)
        temperature *= coolingFactor
        #numSwap = int(numSwap * coolingFactor)
        if temperature < endTemp:
            break
        # print(bestDistance)
        # print(currentDistance)
        # print('\n')
        # print(newDistance)

print(bestDistances)
print(min(bestDistances))
print(bestCityOrder)
# print(a)
