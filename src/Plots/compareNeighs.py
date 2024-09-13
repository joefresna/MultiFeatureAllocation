'''
Created on 29 Aug 2018

@author: Andreagiovanni Reina.
University of Sheffield, UK.
'''

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_HOME="/Users/joefresna/MultiCategorisation"
RESULTS_DIR=PROJECT_HOME+"/data/neighs/"
RESULTS_DIR_FULL=PROJECT_HOME+"/data/fullnet/"
outplot=PROJECT_HOME+"/data/plots/neighs"

NUM_EXP=1000

#featureList=np.arange(2,11)
features=8
methods=['random', 'split', 'on-noise', 'on-noise-quad']
# neighs=0
neighsList=[3,5,7,11]
numAgentsList=[100]

colours_set = ['blue', 'crimson','forestgreen','magenta']
point_set = ['o','x','^','*']
line_set = ['-','--',':','-.']
linewidth=2
pointsize=12
labelSize=18
ticksSize=14
legendSize=14

if __name__ == '__main__':
    fullTable=[]
    for numAgents in numAgentsList:
        for neighs in [0]+neighsList:
            for method in methods:
                resFilename="out_agents-" + str(numAgents) + "_neighs-" + str(neighs) + "_features-" + str(features) + "_method-" + str(method) + ".txt"
    #             res = np.loadtxt(RESULTS_DIR+resFilename, skiprows=1, delimiter='\t')
    #             print(res)
                resFile = open((RESULTS_DIR if neighs>0 else RESULTS_DIR_FULL)+resFilename, 'r')
                resFile.readline()
                count_correct  = 0
                count_wrong    = 0
                count_undecided= 0
                for line in resFile: 
                    if len(line) == 0: continue
                    fields = line.split('\t')
                    convergence = json.loads(fields[3])
                    decisions  = json.loads(fields[4])
                    
                    if min(convergence) < 0.9999999999:
                        count_undecided += 1
                    elif max(decisions) == 0:
                        count_correct += 1
                    else:
                        count_wrong += 1
                
                count_total=count_undecided+count_correct+count_wrong
                fullTable.append([features, method, numAgents, neighs, count_total, count_undecided, count_correct, count_wrong])
                if count_undecided > 0: print(str(count_undecided) + " undecided runs with " + str(numAgents) + " agents and " + str(neighs) +" neighbours")
                #print("correct,wrong,undecided: " + str([count_correct,count_wrong,count_undecided]))
    
                resFile.close()
    
    plt.clf()
    plt.xlabel('Num. neighs', fontsize=labelSize)       
    plt.ylabel('Group accuracy', fontsize=labelSize)       
    #plt.grid(True, ls='--')
    plt.xticks(fontsize=ticksSize)
    plt.yticks(fontsize=ticksSize)
    #plt.axes().use_sticky_edges=False
    plt.axes().margins(0.02)
    
    idx_features=0
    idx_method=1
    idx_agents=2
    idx_neighs=3
    idx_total=4
    idx_correct=6
    
    #print(fullTable)
    lines = []
    zeros = []
    leg_methods = []
    leg_agents = []
    for j,agents in enumerate(numAgentsList):
        leg_agents.append( mlines.Line2D([], [], color='black', label=str(agents)+' agents', ls=line_set[j], lw=linewidth, marker=point_set[j], markersize=pointsize, antialiased=True ) )
        for i,method in enumerate(methods):
            lines.append([ (y[idx_neighs], y[idx_correct]/y[idx_total]) for y in fullTable if y[idx_method] == method and y[idx_agents] == agents and not y[idx_neighs] == 0])
            zeros.append([ y[idx_correct]/y[idx_total] for y in fullTable if y[idx_method] == method and y[idx_agents] == agents and y[idx_neighs] == 0])
            if j==0: leg_methods.append( mlines.Line2D([], [], color=colours_set[i], label=method, lw=linewidth))#, marker=point_set[i], markersize=pointsize, antialiased=True ) )

    for i,line in enumerate(lines):
        plt.plot([x[0] for x in line], [x[1] for x in line], ls=line_set[int(i/len(methods))], color=colours_set[i%len(methods)], lw=linewidth, marker=point_set[int(i/len(methods))], markersize=pointsize)
        plt.axhline(y=zeros[i], ls='--', color=colours_set[i%len(methods)], lw=linewidth )
    
    leg_methods=plt.legend(handles=leg_methods, loc='lower right', bbox_to_anchor=(1.0, 0.0), fontsize=legendSize )
    plt.legend(handles=leg_agents, loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=legendSize, numpoints=1 )
    plt.gca().add_artist(leg_methods)

    pp = PdfPages(outplot+"-f"+str(features)+".pdf")
    pp.savefig()
    pp.close()
    
    plt.show()
    