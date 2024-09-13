'''
Created on 06 Feb 2020

@author: Andreagiovanni Reina.
University of Sheffield, UK.
'''
from MultiCat.MyTypes import SelfAllocationType, SpreadAccuracies
from MultiCat.Population import condorcet
import numpy as np
from copy import deepcopy
import numpy.random as rand
import sys


def giveBest(featuresAccuracies, popDistribution, allocationType):
    ## INCREMENTAL_PRE incrementally add one agent to each category, placing it in the lowest condorced group
    if allocationType == SelfAllocationType.INCREMENTAL_PRE:
        condorcets = []
        for f,group in enumerate(popDistribution):
            condorcets.append( condorcet(group, featuresAccuracies[f]) )
        #print("Agents' feature allocation is " + str(self.popDistribution) + " condorcets: " + str(condorcets))
        mins = [ i for i,c in enumerate(condorcets) if c == min(condorcets)]
        if len(mins) > 1: rand.shuffle(mins)
        return mins[0]
     
    ## INCREMENTAL_POST incrementally add one agent to each category, placing it in the group generating the highest condorceds product    
    elif allocationType == SelfAllocationType.INCREMENTAL_POST:        
        condorcets_now = []
        condorcets_plus = []
        condorcets_projections = []
        for f,group in enumerate(popDistribution):
            condorcets_now.append( condorcet(group, featuresAccuracies[f]) )
            condorcets_plus.append( condorcet(group+1, featuresAccuracies[f]) )
        for f,c_pl in enumerate(condorcets_plus):
            condorcets_projections.append( c_pl * np.prod( [ c for i,c in enumerate(condorcets_now) if not i == f] ) )
        #print("Agents' feature allocation is " + str(self.popDistribution) + " condorcets-now: " + str(condorcets_now) + " condorcets-plus: " + str(condorcets_plus) + " projections: " + str(condorcets_projections))
        maxs = [ i for i,c in enumerate(condorcets_projections) if c == max(condorcets_projections)]
        if len(maxs) > 1: rand.shuffle(maxs)
        # check if it would be the same to get the largest difference on the task (without multiplication)
        checkDiff = False
        if checkDiff:
            diffs = [ condorcets_plus[i] - condorcets_now[i] for i in range(len(condorcets_plus)) ]
            maxDiff = [ i for i,c in enumerate(diffs) if c == max(diffs)]
            if not maxs[0] == maxDiff[0]:
                print("Found difference!")
                print("Explict method found: " + str(maxs) + "(" + str(condorcets_projections[maxs[0]]) + ") simpler method found:" + str(maxDiff) + "(" + str(condorcets_projections[maxDiff[0]]) + ")" )
        return maxs[0], (max(condorcets_projections) - min(condorcets_projections)), condorcets_projections[0]-condorcets_projections[1]
            
allDistributions = []
    
def createAllDistributions(n_tasks, pop):
    dist = [0]*n_tasks
    iterateOnPopulation(0,n_tasks,dist,pop)
    
def iterateOnPopulation(i, n_tasks, dist, pop):
    #if i == 0: dist = [0]*n_tasks
    minSize=2;
    for mypop in np.arange(minSize,pop+1-minSize,2): #plus one is necessary for how arange works
        if mypop > pop-minSize: break
        dist[i] = mypop
        rest = pop-mypop
        if i < (n_tasks-2):
            iterateOnPopulation(i+1, n_tasks, dist, rest)
        else:
            dist[i+1] = rest
            allDistributions.append(deepcopy(dist))


if __name__ == '__main__':
    n_tasks = 3
    n_agents = 100
     
    allocationType = SelfAllocationType.INCREMENTAL_POST
    
    createAllDistributions(n_tasks,n_agents)

    # Spread the accuracies regularly in the range ]0.5, 1[  --including spaces on the range limits
    allAccuracies = []
    spreadAccuraciesMethod = SpreadAccuracies.MEAN_AND_CONSTANT_DIFF
    if spreadAccuraciesMethod == SpreadAccuracies.EQUALLY_SPACED: 
        accuracies = [0]*n_tasks
        for i in np.arange(1,n_tasks+1):
            accuracies[i-1] = 0.5 + (i * 0.5 / (n_tasks+1) )
        allAccuracies.append(accuracies)
        print("Accuracies are " + str(allAccuracies))
    elif spreadAccuraciesMethod == SpreadAccuracies.MEAN_AND_CONSTANT_DIFF:
        acc_mean = 0.75
        diff_step = 0.01
        for diff in np.arange(diff_step,0.5, diff_step ):
            acc_width = diff*(n_tasks-1)
            accuracies = []
            for f in range(n_tasks):
                acc = np.round( acc_mean-(acc_width/2.0) + (f*diff), 4 )
                accuracies.append( acc )
            allAccuracies.append(accuracies)
            #print("Accuracies for diff " + str(diff) + " are " + str(accuracies))
        print("Accuracies for diff are " + str(allAccuracies))
    
    for accuracies in allAccuracies:
        f = open("/Users/joefresna/BestOfN/data/nextBest/2n-tests-" + str(np.round((acc_mean-accuracies[0])*2,4)) + ".txt", "w")
        bp = None
        for distribution in allDistributions:
            best = giveBest(accuracies, distribution, allocationType)
            if best[0] == 1 and bp is None:
                bp = distribution
                #print("Break point for diff " + str(np.round((acc_mean-accuracies[0])*2,4)) + " is " + str(bp))
                #print("{" + str(np.round((acc_mean-accuracies[0])*2,4)) + ", " + str(bp[0]) + "},")
            #print(str(distribution) + " -> " + str(best))
            line = str(np.round((acc_mean-accuracies[0])*2,4)) + "\t"
            for d in distribution:
                line += str(d) + "\t" 
            line += str(best[1]) + "\t" + str(best[2]) + "\n"
            f.write(line)
        f.close()
     
    print("Process ended.")