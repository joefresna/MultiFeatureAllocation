'''
Created on 23 Aug 2018

@author: Andreagiovanni Reina.
University of Sheffield, UK.
'''

import numpy as np
import numpy.random as rand
from MultiCat.CatAgent import CatAgent
from MultiCat.MyTypes import AllocationType
import math
import copy
import sys
from scipy.special import binom

class CatPopulation:
    
    DEBUG = False
    
    def __init__(self, numNodes, numFeatures, featuresAccuracies, allocationType, debug=False):
        self.converged = False
        self.numNodes = numNodes
        self.popDistribution = None
        self.agents = []
        self.numFeatures = numFeatures
        self.featuresAccuracies = featuresAccuracies
        self.allocationType = allocationType
        self.DEBUG = debug
    
    def initPopulation(self, minSize):
        ## check if allocation with given minSize is possible
        if minSize > math.floor(self.numNodes/self.numFeatures):
            print("ERROR! Impossible to guarantee at least " + str(minSize) + " agents in each group with a swarm of size " + str(self.numNodes) + " and " + str(self.numFeatures) + " features!" )
            sys.exit()
        ## allocate population with different methods
        ## RANDOM allocation to each agent type
        if self.allocationType == AllocationType.RANDOM:
            #TODO: this could be done more efficiently
            self.popDistribution = [0]*self.numFeatures
            for _ in np.arange(self.numNodes):
                feature = 1
                rnd = rand.random()
                while True:
                    #print ("Thresh is " + str(feature * (1.0/self.numFeatures)))
                    if rnd < feature * (1.0/self.numFeatures):
                        self.popDistribution[feature-1] += 1
                        #print("Agent " + str(a) + " has type " + str(feature))
                        break
                    feature += 1
            while min(self.popDistribution) < minSize:
                self.popDistribution[ self.popDistribution.index(max(self.popDistribution)) ] -= 1
                self.popDistribution[ self.popDistribution.index(min(self.popDistribution)) ] += 1
                
        ## SPLIT of the population in equal size groups 
        elif self.allocationType == AllocationType.SPLIT:
            subPopSize = self.numNodes/self.numFeatures
            self.popDistribution = [math.floor(subPopSize)]*self.numFeatures
            #print("TMP allocation is " + str(self.popDistribution) + " check-sum("+str(sum(self.popDistribution))+")")
            # in case the population is not divisible by the number of features, the remaining agents are randomly allocated
            extraAgents = np.arange(self.numFeatures)
            rand.shuffle(extraAgents)
            for feature,_ in zip(extraAgents, np.arange(self.numNodes - sum(self.popDistribution))):
                self.popDistribution[feature] += 1
        ## ON_NOISE proportional to the feature's noise level   
        elif self.allocationType == AllocationType.ON_NOISE or self.allocationType == AllocationType.ON_NOISE_ODDS or self.allocationType == AllocationType.ON_NOISE_LLR or self.allocationType == AllocationType.ON_NOISE_ERR or self.allocationType == AllocationType.ON_NOISE_VAR:
            #trimPop = self.numNodes - (minSize*self.numFeatures)
            #self.popDistribution = [minSize]*self.numFeatures
            self.popDistribution = [0]*self.numFeatures
            if self.allocationType == AllocationType.ON_NOISE:
                featuresErrors = [ np.sqrt(acc*(1 - acc)) for acc in self.featuresAccuracies]
            if self.allocationType == AllocationType.ON_NOISE_VAR:
                featuresErrors = [ acc*(1 - acc) for acc in self.featuresAccuracies]
            if self.allocationType == AllocationType.ON_NOISE_ERR:
                featuresErrors = [ 1 - acc for acc in self.featuresAccuracies]
            elif self.allocationType == AllocationType.ON_NOISE_ODDS:
                #featuresErrors = [ (1 - acc)*(1 - acc) for acc in self.featuresAccuracies]
                featuresErrors = [ (1 - acc)/(acc*acc) for acc in self.featuresAccuracies]
            elif self.allocationType == AllocationType.ON_NOISE_LLR:
                featuresErrors = [ np.log(acc)/np.log(1-acc) for acc in self.featuresAccuracies]
            #featuresErrors = [ np.log(acc/(1 - acc)) for acc in self.featuresAccuracies]
            #featuresErrors = [ 1-(err/sum(featuresErrors)) for err in featuresErrors]
            leftOvers = {}
            if self.DEBUG: print("Noise allocation proportions are " + str( [f/sum(featuresErrors) for f in featuresErrors]))
            for f,err in enumerate(featuresErrors):
                pop = self.numNodes * err / sum(featuresErrors)
                if not _almostEqual(pop, math.floor(pop)):
                    leftOvers[f] = pop - math.floor(pop)
                self.popDistribution[f] = math.floor(pop)
                
            # if approximations resulted in less agents, they are added randomly (with probability proportional to the rounding quantities)
            #print("TMP allocation is " + str(self.popDistribution) + " check-sum("+str(sum(self.popDistribution))+")")
            while sum(self.popDistribution) < self.numNodes:
                rnd = np.random.rand() * sum(leftOvers.values())
                bottom = 0.0
                for subpop, prob in leftOvers.items():
                    if rnd >= bottom and rnd < (bottom + prob):
                        self.popDistribution[subpop] += 1
                        leftOvers[subpop] = 0
                        break
                    
            while min(self.popDistribution) < minSize:
                #print("TMP allocation is " + str(self.popDistribution) + " check-sum("+str(sum(self.popDistribution))+")")
                sortedPops = sorted( [i for i,p in enumerate(self.popDistribution) if p > minSize], key=lambda k: self.popDistribution[k], reverse=True)
                #print(sortedPops)
                for p in sortedPops:
                    self.popDistribution[p] -= 1
                    self.popDistribution[ self.popDistribution.index(min(self.popDistribution)) ] += 1
                    if min(self.popDistribution) >= minSize: break
        
        ## INCREMENTAL_PRE incrementally add one agent to each category, placing it in the lowest condorced group
        elif self.allocationType == AllocationType.INCREMENTAL_PRE:
            self.popDistribution = [minSize]*self.numFeatures
            condorcets = []
            for f,group in enumerate(self.popDistribution):
                condorcets.append( condorcet(group, self.featuresAccuracies[f]) )
            #print("Agents' feature allocation is " + str(self.popDistribution) + " condorcets: " + str(condorcets))
            while sum(self.popDistribution) < self.numNodes:
                mins = [ i for i,c in enumerate(condorcets) if c == min(condorcets)]
                if len(mins) > 1: rand.shuffle(mins)
                self.popDistribution[mins[0]] += 1
                condorcets[mins[0]] = condorcet(self.popDistribution[mins[0]], self.featuresAccuracies[mins[0]])
                #print("Agents' feature allocation is " + str(self.popDistribution) + " condorcets: " + str(condorcets))
        
        ## INCREMENTAL_POST incrementally add one agent to each category, placing it in the group generating the highest condorceds product    
        elif self.allocationType == AllocationType.INCREMENTAL_POST:        
            self.popDistribution = [max(1,minSize)]*self.numFeatures
            condorcets_now = []
            condorcets_plus = []
            condorcets_projections = []
            for f,group in enumerate(self.popDistribution):
                condorcets_now.append( condorcet(group, self.featuresAccuracies[f]) )
                condorcets_plus.append( condorcet(group+2, self.featuresAccuracies[f]) )
            for f,c_pl in enumerate(condorcets_plus):
                condorcets_projections.append( c_pl * np.prod( [ c for i,c in enumerate(condorcets_now) if not i == f] ) )
            #print("Agents' feature allocation is " + str(self.popDistribution) + " condorcets-now: " + str(condorcets_now) + " condorcets-plus: " + str(condorcets_plus) + " projections: " + str(condorcets_projections))
            while sum(self.popDistribution) < self.numNodes:
                maxs = [ i for i,c in enumerate(condorcets_projections) if c == max(condorcets_projections)]
                if len(maxs) > 1: rand.shuffle(maxs)
                self.popDistribution[maxs[0]] += 2 if self.numNodes - sum(self.popDistribution) > 1 else 1  
                condorcets_now[maxs[0]] = condorcets_plus[maxs[0]]
                condorcets_plus[maxs[0]] = condorcet(self.popDistribution[maxs[0]]+2, self.featuresAccuracies[maxs[0]])
                #condorcets_projections[maxs[0]] = condorcets_plus[maxs[0]] * np.prod( [ c for i,c in enumerate(condorcets_now) if not i == maxs[0]] )
                for f,c_pl in enumerate(condorcets_plus):
                    condorcets_projections[f] = c_pl * np.prod( [ c for i,c in enumerate(condorcets_now) if not i == f] )
                #print("Agents' feature allocation is " + str(self.popDistribution) + " condorcets-now: " + str(condorcets_now) + " condorcets-plus: " + str(condorcets_plus) + " projections: " + str(condorcets_projections))
        
        if self.DEBUG: print("Agents' feature allocation is " + str(self.popDistribution) + " check-sum("+str(sum(self.popDistribution))+")")
        for a in np.arange(self.numNodes):
            feature = 0
            while (a+1) > sum(self.popDistribution[0:feature+1]):
                feature += 1
            self.agents.append( CatAgent(feature, self.DEBUG) )
            #print("Agent " + str(a) + " is of type " + str(feature))
    
    def sampleFeatures(self):
        for agent in self.agents:
            agent.sampleFeature(self.featuresAccuracies)
            #if self.DEBUG: print("Agent " + str(agent) + " sampled " + str(agent.myEstimate))
         
    def fullAgreementCheck(self):
        decisions = [None]*self.numFeatures
        convergence = [0.0]*self.numFeatures
        ag0 = 0
        for f in np.arange(self.numFeatures):
            ## Count the opinions
            cats = [0,0]
            for ag in np.arange(ag0, ag0+self.popDistribution[f]):
                cats[ self.agents[ag].categories[f] ] += 1
            ## compute the level of convergence
            convergence[f] = max(cats) / self.popDistribution[f]
            ## determine the leading option
            leadingCats = [i for i,c in enumerate(cats) if c == max(cats) ]
            if len(leadingCats) == 1: ## otherwise it remains None (i.e. indecision)
                decisions[f] = leadingCats[0]
            ag0 += self.popDistribution[f]
        return convergence, decisions
            
    def collectiveDecision(self, neighbourSize, maxLoops):
        count = 0
        ## Print out the report
        if (self.DEBUG):
            #for i,a in enumerate(self.agents):
            #    print('Agent ' + str(i) + ' (f:' + str(a.myFeature) + ') ' + str(a.categories))
            convergence, decisions = self.fullAgreementCheck()
            print("[Step:" + str(count) + "] Converge status is " + str(convergence) + " with decisions " + str(decisions) )
            
        if neighbourSize == 0:
            decisions = [None]*self.numFeatures
            convergence = [1.0]*self.numFeatures
            ag0 = 0
            for f in np.arange(self.numFeatures):
                ## Count the opinions
                cats = [0,0]
                for ag in np.arange(ag0, ag0+self.popDistribution[f]):
                    cats[ self.agents[ag].categories[f] ] += 1
                ## determine the leading option
                leadingCats = [i for i,c in enumerate(cats) if c == max(cats) ]
                if len(leadingCats) > 1: rand.shuffle(leadingCats) ## if there is indecision, the choice is random
                decisions[f] = leadingCats[0]
                ag0 += self.popDistribution[f]
            return 1, convergence, decisions
                
        while ( not _almostEqual(1.0, min(self.fullAgreementCheck()[0])) ) and count < maxLoops:
        #while (self.numNodes > self.countOpinion(1) and self.numNodes > self.countOpinion(-1) and count < maxLoops):
        #while (count < maxLoops):
            count += 1
            # create a deep-copy of the current population (for synchronous communication)
            tmp_agents = copy.deepcopy(self.agents)
            # each node updates its opinion through a simple majority algorithm among the neighbourSize neighbours
            for i,a in enumerate(self.agents):
                #if (self.DEBUG): print('agent ' + str(i))
                a.integrateInfoFromNeighbours(i, tmp_agents, neighbourSize)
        
            #if (self.DEBUG): print("Positive nodes: " + str(self.countOpinion(1)) + " negative nodes: " + str(self.countOpinion(-1)) + " iteration: " + str(count) )
        
            ## Print out the report
            if (self.DEBUG):
                #for i,a in enumerate(self.agents):
                #    print('Agent ' + str(i) + ' (f:' + str(a.myFeature) + ') ' + str(a.categories))
                convergence, decisions = self.fullAgreementCheck()
                print("[Step:" + str(count) + "] Converge status is " + str(convergence) + " with decisions " + str(decisions) )
        
        convergence, decisions = self.fullAgreementCheck()
        return count, convergence, decisions 

def condorcet(n, p):
    if n==0: return 0
    res = 0
    for k in np.arange(np.ceil((n+1)/2), n+1):
        res += binom(n, k) * (p**k) * ((1 - p)**(n - k))
        #print(k)
    # if the number n is even, I add 50% of the probability to be correct. 
    if n%2 == 0:
        #print(res)
        res += 0.5*binom(n, n/2) * (p**(n/2)) * ((1 - p)**(n/2))
        #print(res)
    return res
                                                 

def _almostEqual(a,b):
    epsilon = 1e-10
    return abs(a-b) < epsilon
