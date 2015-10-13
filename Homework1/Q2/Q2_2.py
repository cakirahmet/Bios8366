__author__ = 'Ahmet'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

aic = lambda g: g.nobs * np.log((g.resid**2).sum()/g.nobs) + 2*len(g.beta)

def calculate_fitness(aic_values):
    P = len(aic_values)
    aic_rank = (-aic_values).argsort().argsort()+1.
    return 2.*aic_rank/(P*(P+1.))

def geneticAlgo(mutation_rate,pop_size):

    baseball = pd.read_table("/Users/Ahmet/Box Sync/Classes/Vanderbilt/AdvancedStatisticalComputing/Bios8366/data/textbook/baseball.dat", sep='\s+')
    predictors = baseball.copy()
    logsalary = predictors.pop('salary').apply(np.log)
    nrows, ncols = predictors.shape
    iterations = 100

    aic_best = []
    best_solution = []
    aic_history = []

    # Initialize genotype
    current_gen = np.random.binomial(1, 0.5, pop_size*ncols).reshape((pop_size, ncols))


    for i in range(iterations):

        # Get phenotype
        current_phe = [predictors[predictors.columns[g.astype(bool)]] for g in current_gen]
        # Calculate AIC
        current_aic = np.array([aic(pd.ols(y=logsalary, x=x)) for x in current_phe])
        # Get lowest AIC
        aic_best.append(current_aic[np.argmin(current_aic)])
        best_solution.append(current_gen[np.argmin(current_aic)])

        # Calculate fitness according to AIC rank
        fitness = calculate_fitness(current_aic)

        # Choose first parents according to fitness
        moms = np.random.choice(range(pop_size), size=int(pop_size/2), p=fitness)
        # Choose second parents randomly
        dads = np.random.choice(range(pop_size), size=int(pop_size/2))

        next_gen = []
        for x,y in zip(current_gen[moms], current_gen[dads]):
            # Crossover
            cross = np.random.randint(0, ncols)
            child1 = np.r_[x[:cross], y[cross:]]
            child2 = np.r_[y[:cross], x[cross:]]
            # Mutate
            m1 = np.random.binomial(1, mutation_rate, size=ncols).astype(bool)
            child1[m1] = abs(child1[m1]-1)
            m2 = np.random.binomial(1, mutation_rate, size=ncols)
            child2[m2] = abs(child1[m2]-1)
            next_gen += [child1, child2]

        # Increment generation
        current_gen = np.array(next_gen)
        # Store AIC values
        aic_history.append(current_aic)
    return aic_best

aic_best = geneticAlgo(0.01,20)
plt.plot(aic_best,color='r')

aic_best = geneticAlgo(0.02,20)
plt.plot(aic_best,color='b')

aic_best = geneticAlgo(0.03,20)
plt.plot(aic_best,color='k')
plt.show()