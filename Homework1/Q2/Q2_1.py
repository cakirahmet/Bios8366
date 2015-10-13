__author__ = 'Ahmet'
__author__ = 'Ahmet'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


aic = lambda g: g.nobs * np.log((g.resid**2).sum()/g.nobs) + 2*len(g.beta)

def plotResults(aic_values,aic_best,solution_best):

    plt.plot(aic_values)
    plt.xlim(0, len(aic_values))
    plt.xlabel('Iteration')
    plt.ylabel('AIC')
    print('Best AIC: {0}\nBest solution: {1}\nDiscovered at iteration {2}'.format(aic_best,
                np.where(solution_best==True),
                np.where(aic_values==aic_best)[0][0]))
    plt.plot(np.where(aic_values==aic_best)[0][0], aic_best, 'ro')
    plt.show()

def simAnnealing(periods,cooling,tau_start,numChangeNeigh):
    baseball = pd.read_table("/Users/Ahmet/Box Sync/Classes/Vanderbilt/AdvancedStatisticalComputing/Bios8366/data/textbook/baseball.dat", sep='\s+')
    predictors = baseball.copy()
    logsalary = predictors.pop('salary').apply(np.log)

    nrows, ncols = predictors.shape

    aic_values = []
    solution_current = solution_best = np.random.binomial(1, 0.5, ncols).astype(bool)
    solution_vars = predictors[predictors.columns[solution_current]]
    g = pd.ols(y=logsalary, x=solution_vars)
    aic_best = aic(g)
    aic_values.append(aic_best)

    # Cooling schedule
    tau = [tau_start * 0.9**i for i in range(periods)]
    for j in range(periods):

        for i in range(cooling[j]):

            # Random change n-neighborhood
            flip = np.random.choice(ncols,numChangeNeigh,replace = False)
            for i in range(numChangeNeigh):
                solution_current[flip[i]] = not solution_current[flip[i]]
            solution_vars = predictors[predictors.columns[solution_current]]
            g = pd.ols(y=logsalary, x=solution_vars)
            aic_step = aic(g)
            alpha = min(1, np.exp((aic_values[-1] - aic_step)/tau[j]))

            if ((aic_step < aic_values[-1]) or (np.random.uniform() < alpha)):
                # Accept proposed solution
                aic_values.append(aic_step)
                if aic_step < aic_best:
                    # Replace previous best with this one
                    aic_best = aic_step
                    solution_best = solution_current.copy()
            else:
                # Revert solution
                for i in range(numChangeNeigh):
                    solution_current[flip[i]] = not solution_current[flip[i]]
                aic_values.append(aic_values[-1])
    return aic_values,aic_best,solution_best


periods = 15
numChangeNeigh = 1
cooling_1 = [15* i for i in range(1,periods+1)]
#[60]*int(periods/3) + [120]*int(periods/3) + [220]*int(periods/3) # spend more time at low temperatures
tau_start = 10
aic_values_1,aic_best_1,solution_best_1 = simAnnealing(periods,cooling_1,tau_start,numChangeNeigh)
plotResults(aic_values_1,aic_best_1,solution_best_1)


periods = 15
numChangeNeigh = 2
cooling_1 = [15* i for i in range(1,periods+1)]
#[60]*int(periods/3) + [120]*int(periods/3) + [220]*int(periods/3) # spend more time at low temperatures
tau_start = 10
aic_values_1,aic_best_1,solution_best_1 = simAnnealing(periods,cooling_1,tau_start,numChangeNeigh)
plotResults(aic_values_1,aic_best_1,solution_best_1)

periods = 15
numChangeNeigh = 3
cooling_1 = [15* i for i in range(1,periods+1)]
#[60]*int(periods/3) + [120]*int(periods/3) + [220]*int(periods/3) # spend more time at low temperatures
tau_start = 10
aic_values_1,aic_best_1,solution_best_1 = simAnnealing(periods,cooling_1,tau_start,numChangeNeigh)
plotResults(aic_values_1,aic_best_1,solution_best_1)


#tau_start = 20
#aic_values_1,aic_best_1,solution_best_1 = simAnnealing(periods,cooling_1,tau_start)
#plotResults(aic_values_1,aic_best_1,solution_best_1)


#cooling_2 = np.fliplr([cooling_1])[0]
# [220]*int(periods/3) + [120]*int(periods/3) + [60]*int(periods/3) # spend more time at high temperatures
#aic_values_2,aic_best_2,solution_best_2 = simAnnealing(periods,cooling_2,tau_start)
#plotResults(aic_values_2,aic_best_2,solution_best_2)

#cooling_3 = [120]*int(periods/3) + [120]*int(periods/3) + [120]*int(periods/3) # spend more time at high temperatures
#aic_values_3,aic_best_3,solution_best_3 = simAnnealing(periods,cooling_3,tau_start)
#plotResults(aic_values_3,aic_best_3,solution_best_3)