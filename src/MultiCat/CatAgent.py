'''
Created on 23 Aug 2018

@author: Andreagiovanni Reina.
University of Sheffield, UK.
'''

import numpy.random as rand
import numpy as np
import sys

class CatAgent:

    DEBUG = False
    
    def __init__(self, agentFeature, debug=False):
        self.myFeature = agentFeature
        #self.myEstimate = None
        self.categories = None
        self.DEBUG = debug

    def sampleFeature(self, featuresAccuracies):
        myAccuracy = featuresAccuracies[self.myFeature]
        self.categories = [None]*len(featuresAccuracies)
        self.categories[self.myFeature] = 0 if (rand.random() < myAccuracy) else 1 
        #self.myEstimate = rand.random() < myAccuracy # True is correct, False is wrong

    def integrateInfoFromNeighbours(self, myID, all_agents, neighbourSize):
        agentsIds = np.arange(len(all_agents))
        rand.shuffle(agentsIds)
        ## Get the opinion of neighbourSize neighbours
        neighsOpinion = []
        for nid in agentsIds:
            if nid == myID: continue # I can't pick as my neighbour myslef
            if all_agents[nid].myFeature == self.myFeature:
                neighsOpinion.append(all_agents[nid].categories[self.myFeature])
                #print("Agent " + str(myID) + " (feature " + str(self.myFeature) + ") speaks with " + str(nid) + " with opinion " + str(all_agents[nid].categories[self.myFeature]))
                if len(neighsOpinion) == neighbourSize:
                    break
        
        if not len(neighsOpinion) == neighbourSize: ## sanity check
            print("ERROR! Agent " + str(myID) + " (feature " + str(self.myFeature) + ") couldn't find " + str(neighbourSize) + " neighbours of same feature type.")
            sys.exit()
        
        ## Count the neighs opinions
        cats = [0,0]
        for nop in neighsOpinion:
            cats[nop] += 1
        ## Add my opinion to the count
        cats[self.categories[self.myFeature]] +=1
        
        majorityCats = [i for i,c in enumerate(cats) if c == max(cats) ]
        if len(majorityCats) > 1:
            # select one of the max values at random
            rand.shuffle(majorityCats)
        self.categories[self.myFeature] = majorityCats[0]
        
        #print("Agent " + str(myID) + " cats are " + str(cats) + " and final op is " + str(self.categories))
        