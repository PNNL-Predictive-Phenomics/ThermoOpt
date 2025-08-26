"""

Connah G. M. Johnson     
connah.johnson@pnnl.gov
March 2024

Functions for analysing the ouput from flux constrained optimization simulations of metabolic models. 


Code contents:

def ConvertToComparisonDict
def ParseData
def PlotTwoAgainstGlucosePercentageErrorbar

def PlotTwoAgainstGlucosePercentageErrorbarIndividual
def PlotTwoAgainstGlucosePercentageFillbetween
def PlotTwoAgainstGlucosePercentage
def PlotOneAgainstGlucosePercentage
def PlotOneAgainstGlucosePercentageIndividualScaled
def PlotOneAgainstGlucosePercentageScaled
def PlotOneAgainstInitialConditionScaled
def PlotOneAgainstGlucosePercentageMonodScaled
def PlotMetaboliteSolutions
def PlotFluxSolutions
def PlotBestSimulations


"""

import numpy as np
import pandas as pd
import random
import multiprocessing as mp
import subprocess
import numpy.random as nprd
from scipy.linalg import norm
from scipy.optimize import least_squares
import scipy.linalg as spL
#import neccessary packages
import numpy as np
import pandas as pd
import subprocess
import random
import sys
import re
import os
import warnings
import pickle
from scipy.optimize import least_squares
from IPython.core.display import display
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None,'display.max_rows', None)
import json
warnings.filterwarnings('ignore')
from importlib import reload
import ast
import base_functions as baf



def ConvertToComparisonDict(bestSimulationsdf,simulationList,model_file_dir,save_dir,targetReactionList,targetMetaboliteList=[]):

    parentSimNames=[]
    comparisonDict={}
    for ind in bestSimulationsdf.index:

        val = bestSimulationsdf.loc[ind]['swarm_instance_name']
        convergedSwarmNames = bestSimulationsdf.loc[ind]['converged_simulation_list']

        if val != 'none':
            obj_val = bestSimulationsdf.loc[ind]['swarm_instance_value']
            second_val= bestSimulationsdf.loc[ind]['swarm_secondary_value']
            name,swarmNum = val.rsplit('swarm_',1)
            convergedSwarmNumbers = []
            for convergedSwarm in convergedSwarmNames:
                convergedName,convergedNum = convergedSwarm.rsplit('_swarm_',1)
                convergedSwarmNumbers.append(convergedNum)
   
            for simDict in simulationList:

                if simDict['name']==name:

                    percentage=0
                    if '_' == name[-1]:
                        name = name[:-1]
                    
                    print("simName= "+str(name))
                    if 1:
                        # hyperparameter run
                        name,percentage=name.rsplit('_',1)
                    else:
                        name,percentage=name.rsplit('_',1)
                    print("Percentage= "+str(percentage))
                    print("name= "+str(name))
                    if name not in parentSimNames:
                        parentSimNames.append(name)
                        comparisonDict[name]={}
                        comparisonDict[name]['percentage']=[]
                        comparisonDict[name]['swarm']=[]
                        comparisonDict[name]['model']=[]
                        comparisonDict[name]['model_output']=[]
                        comparisonDict[name]['obj_val']=[]
                        comparisonDict[name]['second_val']=[]
                        comparisonDict[name]['converged_simulation_list']=[]
                        comparisonDict[name]['converged_simulation_objectives']=[]
                        comparisonDict[name]['converged_simulation_secondary_objectives']=[]

                    comparisonDict[name]['percentage'].append(percentage)
                    comparisonDict[name]['swarm'].append(swarmNum)
                    comparisonDict[name]['model'].append(simDict['model'])
                    comparisonDict[name]['model_output'].append(simDict['model_output'])
                    comparisonDict[name]['obj_val'].append(obj_val)
                    comparisonDict[name]['second_val'].append(second_val)
                    comparisonDict[name]['converged_simulation_list'].append(convergedSwarmNumbers)
                    comparisonDict[name]['converged_simulation_objectives'].append(bestSimulationsdf.loc[ind]['converged_simulation_objectives'])
                    comparisonDict[name]['converged_simulation_secondary_objectives'].append(bestSimulationsdf.loc[ind]['converged_simulation_secondary_objectives'])
    
    # for each experiment run
    for simName in comparisonDict.keys():
        ySols=[]
        alphaSols=[]
        nSols=[]
        percentages=[]
        obj_val=[]
        second_val=[]

        ySolsFull=[]
        alphaSolsFull=[]
        nSolsFull=[]
        obj_valFull=[]
        second_valFull=[]

        comparisonDict[simName]['ComparisonData']=[]
        comparisonDict[simName]['ComparisonDataSuboptimal']=[]
        # for each simulation model 
        for i in range(len(comparisonDict[simName]['model_output'])):
            
            solFile = comparisonDict[simName]['model_output'][i]
            modelFile = comparisonDict[simName]['model_output'][i]#comparisonDict[simName]['model'][i]
            obj_val = comparisonDict[simName]['obj_val'][i]
            second_val = comparisonDict[simName]['second_val'][i]

            # get the data
            model_file= model_file_dir+str(modelFile)

            if '_secondary' in solFile:    
                saved_solution_file = save_dir+str(solFile)+"swarm_"+comparisonDict[simName]['swarm'][i]+".pkl"
            else:
              
                if "output_extended" in solFile:
                    correctionName = str(solFile).split("output_extended")[1]
                    saved_solution_file = save_dir+str(solFile)+correctionName+"_swarm_"+comparisonDict[simName]['swarm'][i]+".pkl"


                else:

                    saved_solution_file = save_dir+str(solFile)+"swarm_"+comparisonDict[simName]['swarm'][i]+".pkl"
                    model_file = model_file+"swarm_"+comparisonDict[simName]['swarm'][i]+".pkl"
            

            S_active,active_reactions, metabolites, Keq= baf.load_model(model_file)

            y_sol,alpha_sol,n_sol,S,Keq,f_log_counts,vcount_upper_bound,obj_coefs = baf.load_model_solution(saved_solution_file)
            ySols.append(y_sol)
            alphaSols.append(alpha_sol)
            nSols.append(n_sol)
        


            Datadf = pd.DataFrame([])
            Datadf['Reaction'] = S_active.index
            Datadf['Flux'] = y_sol
            Datadf['Regulation'] = alpha_sol


            Concdf = pd.DataFrame([])
            Concdf['Metabolite'] = metabolites.index

            reactionDict={}
            percentageDict={}
            for targetReaction in targetReactionList:
                reactionDict={}
                relativeDifferenceMatrix=[]

                try:
                    if sum(Datadf['Reaction'] == targetReaction)==0:
                        # target reaction is missing 
                        val=0.0
                    else:
                        targetIdx = Datadf.loc[Datadf['Reaction'] == targetReaction].index[0]
                        reactionDict['alpha']=Datadf.iloc[targetIdx]['Regulation']
                        reactionDict['flux']=Datadf.iloc[targetIdx]['Flux']

                except Exception as e:
                    val = 0.0



                percentageDict[targetReaction]=reactionDict


            comparisonDict[simName]['ComparisonData'].append(percentageDict)


            # also get the data for the suboptimal runs 

            ySolsSuboptimal=[]
            alphaSolsSuboptimal=[]
            nSolsSuboptimal=[]
            obj_valSuboptimal=comparisonDict[simName]['converged_simulation_objectives'][i]
            second_valSuboptimal=comparisonDict[simName]['converged_simulation_secondary_objectives'][i]

            suboptimalSwarms=comparisonDict[simName]['converged_simulation_list'][i]

            ComparisonDataSuboptimal=[]
            for soSwarmNum in suboptimalSwarms:

                model_file= model_file_dir+str(modelFile)
                saved_solution_file=""

                if '_secondary' in solFile:    

                    saved_solution_file = save_dir+str(solFile)+"swarm_"+str(soSwarmNum)+".pkl"
                else:      
                    if "output_extended" in solFile:
                        correctionName = str(solFile).split("output_extended")[1]
                        saved_solution_file = save_dir+str(solFile)+correctionName+"_swarm_"+str(soSwarmNum)+".pkl"
                    else:
                        saved_solution_file = save_dir+str(solFile)+"swarm_"+str(soSwarmNum)+".pkl"
                        model_file = model_file+"swarm_"+str(soSwarmNum)+".pkl"
                        print(model_file)


                S_active,active_reactions, metabolites, Keq= baf.load_model(model_file)
                y_sol,alpha_sol,n_sol,S,Keq,f_log_counts,vcount_upper_bound,obj_coefs = baf.load_model_solution(saved_solution_file)

                ySolsSuboptimal.append(y_sol)
                alphaSolsSuboptimal.append(alpha_sol)
                nSolsSuboptimal.append(n_sol)


                Datadf = pd.DataFrame([])
                Datadf['Reaction'] = S_active.index
                Datadf['Flux'] = y_sol
                Datadf['Regulation'] = alpha_sol

                reactionDict={}
                percentageDict={}
                for targetReaction in targetReactionList:
                    reactionDict={}

                    try:
                        if sum(Datadf['Reaction'] == targetReaction)==0:
                            # target reaction is missing 
                            val=0.0
                        else:
                            targetIdx = Datadf.loc[Datadf['Reaction'] == targetReaction].index[0]

                            reactionDict['alpha']=Datadf.iloc[targetIdx]['Regulation']
                            reactionDict['flux']=Datadf.iloc[targetIdx]['Flux']
                    except Exception as e:
                        val = 0.0

                    percentageDict[targetReaction]=reactionDict

                ComparisonDataSuboptimal.append(percentageDict)
            comparisonDict[simName]['ComparisonDataSuboptimal'].append(ComparisonDataSuboptimal)
    
    return comparisonDict

def ParseData(comparisonDict,rxnNameDict,simName):
    
    
    percentageLabels=[]
    reactionLabels=[]
    for i in range(len(comparisonDict[simName]['ComparisonData'])):

        if comparisonDict[simName]['percentage'][i] not in percentageLabels:
            percentageLabels.append(comparisonDict[simName]['percentage'][i])
        for targetReaction in comparisonDict[simName]['ComparisonData'][i].keys():

            if targetReaction not in reactionLabels:
                reactionLabels.append(targetReaction)
            
    
    
    dataMatrixFlux=np.zeros([len(percentageLabels),len(reactionLabels)])
    dataMatrixRegulation=np.zeros([len(percentageLabels),len(reactionLabels)])
    dataMatrixMaxFlux=np.zeros([len(percentageLabels),len(reactionLabels)])
    dataMatrixMaxRegulation=np.zeros([len(percentageLabels),len(reactionLabels)])
    dataMatrixMinFlux=np.zeros([len(percentageLabels),len(reactionLabels)])
    dataMatrixMinRegulation=np.zeros([len(percentageLabels),len(reactionLabels)])
    dataMatrixHalfRangeFlux=np.zeros([len(percentageLabels),len(reactionLabels)])
    dataMatrixHalfRangeRegulation=np.zeros([len(percentageLabels),len(reactionLabels)])
    
    simObjList=np.zeros(len(percentageLabels))
    simSecObjList=np.zeros(len(percentageLabels))
    simObjListMax=np.zeros(len(percentageLabels))
    simSecObjListMax=np.zeros(len(percentageLabels))
    simObjListMin=np.zeros(len(percentageLabels))
    simSecObjListMin=np.zeros(len(percentageLabels))
    simObjListHalfRange=np.zeros(len(percentageLabels))
    simSecObjListHalfRange=np.zeros(len(percentageLabels))

    for i in range(len(comparisonDict[simName]['ComparisonData'])):
        percentage = comparisonDict[simName]['percentage'][i]
        
        for targetReaction in comparisonDict[simName]['ComparisonData'][i].keys():

            dataMatrixFlux[percentageLabels.index(percentage),reactionLabels.index(targetReaction)] = comparisonDict[simName]['ComparisonData'][i][targetReaction]['flux']

            maxVal=comparisonDict[simName]['ComparisonData'][i][targetReaction]['flux']
            minVal=comparisonDict[simName]['ComparisonData'][i][targetReaction]['flux']
            for instance in range(len(comparisonDict[simName]['ComparisonDataSuboptimal'][i])):

                val = comparisonDict[simName]['ComparisonDataSuboptimal'][i][instance][targetReaction]['flux']

                if val > maxVal:
                    maxVal=val
                if val < minVal:
                    minVal=val

            dataMatrixMaxFlux[percentageLabels.index(percentage),reactionLabels.index(targetReaction)] = maxVal
            dataMatrixMinFlux[percentageLabels.index(percentage),reactionLabels.index(targetReaction)] = minVal
            dataMatrixHalfRangeFlux[percentageLabels.index(percentage),reactionLabels.index(targetReaction)] = 0.5*(maxVal-minVal)

            
            
            dataMatrixRegulation[percentageLabels.index(percentage),reactionLabels.index(targetReaction)] = comparisonDict[simName]['ComparisonData'][i][targetReaction]['alpha']
            
            maxVal=comparisonDict[simName]['ComparisonData'][i][targetReaction]['alpha']
            minVal=comparisonDict[simName]['ComparisonData'][i][targetReaction]['alpha']
            for instance in range(len(comparisonDict[simName]['ComparisonDataSuboptimal'][i])):
                val = comparisonDict[simName]['ComparisonDataSuboptimal'][i][instance][targetReaction]['alpha']

                if val > maxVal:
                    maxVal=val
                if val < minVal:
                    minVal=val

            dataMatrixMaxRegulation[percentageLabels.index(percentage),reactionLabels.index(targetReaction)] = maxVal
            dataMatrixMinRegulation[percentageLabels.index(percentage),reactionLabels.index(targetReaction)] = minVal
            dataMatrixHalfRangeRegulation[percentageLabels.index(percentage),reactionLabels.index(targetReaction)] = 0.5*(maxVal-minVal)


        simObjList[percentageLabels.index(percentage)] = comparisonDict[simName]['obj_val'][i]
        simSecObjList[percentageLabels.index(percentage)] = comparisonDict[simName]['second_val'][i]

        maxVal=comparisonDict[simName]['obj_val'][i]
        minVal=comparisonDict[simName]['obj_val'][i]

        for instance in range(len(comparisonDict[simName]['converged_simulation_objectives'][i])):

            val = comparisonDict[simName]['converged_simulation_objectives'][i][instance]

            if val > maxVal:
                maxVal=val
            if val < minVal:
                minVal=val

        
        simObjListMax[percentageLabels.index(percentage)]= maxVal       
        simObjListMin[percentageLabels.index(percentage)]= minVal
        simObjListHalfRange[percentageLabels.index(percentage)]= 0.5*(maxVal-minVal)

        maxVal=comparisonDict[simName]['second_val'][i]
        minVal=comparisonDict[simName]['second_val'][i]

        for instance in range(len(comparisonDict[simName]['converged_simulation_secondary_objectives'][i])):

            val = comparisonDict[simName]['converged_simulation_secondary_objectives'][i][instance]

            if val > maxVal:
                maxVal=val
            if val < minVal:
                minVal=val
        
        simSecObjListMax[percentageLabels.index(percentage)]= maxVal
        simSecObjListMin[percentageLabels.index(percentage)]= minVal
        simSecObjListHalfRange[percentageLabels.index(percentage)]= 0.5*(maxVal-minVal)
        
        
        
            
    plotreactionLabels=[]
    for i in reactionLabels:
        if i in rxnNameDict.keys():
            reactionLabels[reactionLabels.index(i)] = rxnNameDict[i]
            plotreactionLabels.append(rxnNameDict[i])
    
    subRowSet=[]
    for i in plotreactionLabels:
        subRowSet.append(reactionLabels.index(i))
    
    plotDataFlux=dataMatrixFlux[:, subRowSet]
    plotDataReg=dataMatrixRegulation[:, subRowSet]

    plotDataFluxMax=dataMatrixMaxFlux[:, subRowSet]
    plotDataRegMax=dataMatrixMaxRegulation[:, subRowSet]
    plotDataFluxMin=dataMatrixMinFlux[:, subRowSet]
    plotDataRegMin=dataMatrixMinRegulation[:, subRowSet]
    plotDataFluxHalfRange=dataMatrixHalfRangeFlux[:, subRowSet]
    plotDataRegHalfRange=dataMatrixHalfRangeRegulation[:, subRowSet]
    
    
    
    return percentageLabels,reactionLabels,plotDataFlux,plotDataReg,plotDataFluxMax,plotDataRegMax,plotDataFluxMin,plotDataRegMin,plotDataFluxHalfRange,plotDataRegHalfRange,simObjList,simSecObjList,simObjListMax,simSecObjListMax,simObjListMin,simSecObjListMin,simObjListHalfRange,simSecObjListHalfRange
    
def PlotTwoAgainstGlucosePercentageErrorbar(plotDatadf,percentageLabels,QueryModel,QueryObjType,TargetQuery,TargetType,SecondaryType,experimentIDentifiers,colors=["b","g" ,"r"],linesSty = ['-','--','-.']):
    
    markerStyles = ["","D","o","s","*"]

    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                plt.errorbar(x=percentageLabels,y=plotDatadf.iloc[row]["fluxdata"][:,col],yerr=plotDatadf.iloc[row]["halfrangefluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])

            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
    plt.ylabel("Flux")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()
    
    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            for col in range(len(plotDatadf.iloc[row]["regulationdata"][0])):
                plt.errorbar(x=percentageLabels,y=plotDatadf.iloc[row]["regulationdata"][:,col],yerr=plotDatadf.iloc[row]["halfrangeregulationdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
    plt.ylabel("Regulation")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()

    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            plt.errorbar(x=percentageLabels,y=plotDatadf.iloc[row]["primary_obj"],yerr=plotDatadf.iloc[row]["halfrangeprimary_obj"],label='Primary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1

    plt.ylabel("Objective Value")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()

    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:

            plt.errorbar(x=percentageLabels,y=plotDatadf.iloc[row]["secondary_obj"],yerr=plotDatadf.iloc[row]["halfrangesecondary_obj"],label='Secondary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
    plt.ylabel("Objective Value")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()

def PlotTwoAgainstGlucosePercentageErrorbarIndividual(plotDatadf,percentageLabels,QueryModel,QueryObjType,TargetQuery,TargetType,SecondaryType,experimentIDentifiers,colors=["b","g" ,"r"],linesSty = ['-','--','-.']):
    
    markerStyles = ["","D","o","s","*"]

    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            print(plotDatadf.iloc[row]["fluxdata"][0])
            for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                print(plotDatadf.iloc[row]["fluxdata"][:,col])
                plt.errorbar(x=percentageLabels,y=plotDatadf.iloc[row]["fluxdata"][:,col],yerr=plotDatadf.iloc[row]["halfrangefluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])

            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
            plt.ylabel("Flux")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            #  ,'c','m' ,'y','k'
            plt.show()
            
            ax = plt.axes()            
            plt.xlabel("Glucose percentage")
            ax.set_xticks(range(len(percentageLabels)))
            ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            for col in range(len(plotDatadf.iloc[row]["regulationdata"][0])):
                plt.errorbar(x=percentageLabels,y=plotDatadf.iloc[row]["regulationdata"][:,col],yerr=plotDatadf.iloc[row]["halfrangeregulationdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
            plt.ylabel("Regulation")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            #  ,'c','m' ,'y','k'
            plt.show()

            ax = plt.axes()            
            plt.xlabel("Glucose percentage")
            ax.set_xticks(range(len(percentageLabels)))
            ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            plt.errorbar(x=percentageLabels,y=plotDatadf.iloc[row]["primary_obj"],yerr=plotDatadf.iloc[row]["halfrangeprimary_obj"],label='Primary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1

            plt.ylabel("Objective Value")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            #  ,'c','m' ,'y','k'
            plt.show()

            ax = plt.axes()            
            plt.xlabel("Glucose percentage")
            ax.set_xticks(range(len(percentageLabels)))
            ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:

            plt.errorbar(x=percentageLabels,y=plotDatadf.iloc[row]["secondary_obj"],yerr=plotDatadf.iloc[row]["halfrangesecondary_obj"],label='Secondary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
            plt.ylabel("Objective Value")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            #  ,'c','m' ,'y','k'
            plt.show()

            ax = plt.axes()            
            plt.xlabel("Glucose percentage")
            ax.set_xticks(range(len(percentageLabels)))
            ax.set_xticklabels([int(i) for i in percentageLabels])

def PlotTwoAgainstGlucosePercentageFillbetween(plotDatadf,percentageLabels,QueryModel,QueryObjType,TargetQuery,TargetType,SecondaryType,experimentIDentifiers,colors=["b","g" ,"r"],linesSty = ['-','--','-.']):
    

    markerStyles = ["","D","o","s","*"]

    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                plt.fill_between(percentageLabels,plotDatadf.iloc[row]["minfluxdata"][:,col], plotDatadf.iloc[row]["maxfluxdata"][:,col],alpha=0.1,color=colors[col])
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
    plt.ylabel("Flux")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()
    
    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            for col in range(len(plotDatadf.iloc[row]["regulationdata"][0])):
                plt.plot(plotDatadf.iloc[row]["regulationdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                plt.fill_between(percentageLabels,plotDatadf.iloc[row]["minregulationdata"][:,col], plotDatadf.iloc[row]["maxregulationdata"][:,col],alpha=0.1,color=colors[col])
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
    plt.ylabel("Regulation")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()

    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
    #         plt.plot(plotDatadf.iloc[row]["secondary_obj"],label='Secondary Obj', linestyle=linesSty[count], color=colors[1])
            plt.fill_between(percentageLabels,plotDatadf.iloc[row]["minprimary_obj"], plotDatadf.iloc[row]["maxprimary_obj"],alpha=0.1,color=colors[0])
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
    plt.ylabel("Objective Value")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()

    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:

            plt.plot(plotDatadf.iloc[row]["secondary_obj"],label='Secondary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            plt.fill_between(percentageLabels,plotDatadf.iloc[row]["minsecondary_obj"], plotDatadf.iloc[row]["maxsecondary_obj"],alpha=0.1,color=colors[1])

            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
    plt.ylabel("Objective Value")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()

def PlotTwoAgainstGlucosePercentage(plotDatadf,percentageLabels,QueryModel,QueryObjType,TargetQuery,TargetType,SecondaryType,colors=["b","g" ,"r"],linesSty = ['-','--','-.']):
    
    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    count=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[count], color=colors[col])
    
            try:
                print(SecondaryType+": "+str(plotDatadf.iloc[row][SecondaryType])+" linestyle "+str(linesSty[count]))
            except:
                print(SecondaryType+" not found")
            count+=1
    plt.ylabel("Flux")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()
    
    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])
   
    count=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            for col in range(len(plotDatadf.iloc[row]["regulationdata"][0])):
                plt.plot(plotDatadf.iloc[row]["regulationdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[count], color=colors[col])

            try:
                print(SecondaryType+": "+str(plotDatadf.iloc[row][SecondaryType])+" linestyle "+str(linesSty[count]))
            except:
                print(SecondaryType+" not found")
            count+=1
    plt.ylabel("Regulation")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()

    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    count=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[count], color=colors[0])


            try:
                print(SecondaryType+": "+str(plotDatadf.iloc[row][SecondaryType])+" linestyle "+str(linesSty[count]))
            except:
                print(SecondaryType+" not found")
            count+=1
    plt.ylabel("Objective Value")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()

    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    count=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:

            plt.plot(plotDatadf.iloc[row]["secondary_obj"],label='Secondary Obj', linestyle=linesSty[count], color=colors[1])

            try:
                print(SecondaryType+": "+str(plotDatadf.iloc[row][SecondaryType])+" linestyle "+str(linesSty[count]))
            except:
                print(SecondaryType+" not found")
            count+=1
    plt.ylabel("Objective Value")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()

def PlotOneAgainstGlucosePercentage(plotDatadf,percentageLabels,QueryModel,QueryObjType,TargetQuery,TargetType,experimentIDentifiers,colors=["b","g" ,"r"],linesSty = ['-','--','-.']):
    
    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    markerStyles = ["","D","o","s","*"]

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])

            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
            print()
            
            
    plt.ylabel("Flux")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()
    
    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            for col in range(len(plotDatadf.iloc[row]["regulationdata"][0])):
                plt.plot(plotDatadf.iloc[row]["regulationdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
    plt.ylabel("Regulation")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()

    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
    plt.ylabel("Objective Value")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()

    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:

            plt.plot(plotDatadf.iloc[row]["secondary_obj"],label='Secondary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
    plt.ylabel("Objective Value")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()

def PlotOneAgainstGlucosePercentageIndividualScaled(plotDatadf,scaledByUptake,percentageLabels,QueryModel,QueryObjType,TargetQuery,TargetType,experimentIDentifiers,colors=["b","g" ,"r"],linesSty = ['-','--','-.']):
    
    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    markerStyles = ["","D","o","s","*"]
    scalingMatrix=[]
    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            
            scalingVector = np.zeros(len(plotDatadf.iloc[row]["fluxdata"][:,0]))
            for scaleRxn in scaledByUptake:
                for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                    if scaleRxn==plotDatadf.iloc[row]["reactionLabels"][col]:
                        scalingVector = scalingVector + plotDatadf.iloc[row]["fluxdata"][:,col]
            scalingMatrix.append(scalingVector)
            for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col]/scalingVector,label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])

            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
            print()
            
            
            plt.ylabel("Flux")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            #  ,'c','m' ,'y','k'
            plt.show()
            
            ax = plt.axes()            
            plt.xlabel("Glucose percentage")
            ax.set_xticks(range(len(percentageLabels)))
            ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    scalePlot=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            
            plt.plot(scalingMatrix[scalePlot],label='Scale factor', linestyle=linesSty[lineStylecount], color='b', marker=markerStyles[markerStyleCount])
            scalePlot+=1
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
            print()
            
            
    plt.ylabel("Flux")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  ,'c','m' ,'y','k'
    plt.show()
    
    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            for col in range(len(plotDatadf.iloc[row]["regulationdata"][0])):
                plt.plot(plotDatadf.iloc[row]["regulationdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
            plt.ylabel("Regulation")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            #  ,'c','m' ,'y','k'
            plt.show()

            ax = plt.axes()            
            plt.xlabel("Glucose percentage")
            ax.set_xticks(range(len(percentageLabels)))
            ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
            plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
            plt.ylabel("Objective Value")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            #  ,'c','m' ,'y','k'
            plt.show()

            ax = plt.axes()            
            plt.xlabel("Glucose percentage")
            ax.set_xticks(range(len(percentageLabels)))
            ax.set_xticklabels([int(i) for i in percentageLabels])

    lineStylecount=0
    markerStyleCount=0
    for row in plotDatadf.index:
        if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:

            plt.plot(plotDatadf.iloc[row]["secondary_obj"],label='Secondary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            
            for iden in experimentIDentifiers:
                print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
            
            lineStylecount+=1
            if lineStylecount == len(linesSty):
                lineStylecount=0
                markerStyleCount+=1
            plt.ylabel("Objective Value")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            #  ,'c','m' ,'y','k'
            plt.show()
            ax = plt.axes()            
            plt.xlabel("Glucose percentage")
            ax.set_xticks(range(len(percentageLabels)))
            ax.set_xticklabels([int(i) for i in percentageLabels])

def PlotOneAgainstGlucosePercentageScaled(plotDatadf,scaledByUptake,percentageLabels,QueryModel,QueryObjType,TargetQuery,TargetType,experimentIDentifiers,colors=["b","g" ,"r"],linesSty = ['-','--','-.']):
    
    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    # ax.set_xticklabels([int(i) for i in percentageLabels])
    ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)

    markerStyles = ["","D","o","s","*"]
    scalingMatrix=[]
    lineStylecount=0
    markerStyleCount=0
    
    if len(TargetType)==1:
        for row in plotDatadf.index:
            
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
                
                scalingVector = np.zeros(len(plotDatadf.iloc[row]["fluxdata"][:,0]))
                for scaleRxn in scaledByUptake:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        
                        if scaleRxn==plotDatadf.iloc[row]["reactionLabels"][col]:
                            scalingVector = scalingVector + plotDatadf.iloc[row]["fluxdata"][:,col]
                
                scalingMatrix.append(scalingVector)
                if len(scaledByUptake) >0:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col]/scalingVector,label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                else:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                
                
        plt.ylabel("Flux")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)


        lineStylecount=0
        markerStyleCount=0
        scalePlot=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
                
                plt.plot(scalingMatrix[scalePlot],label='Scale factor', linestyle=linesSty[lineStylecount], color='b', marker=markerStyles[markerStyleCount])
                scalePlot+=1
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                
                
        plt.ylabel("Flux")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)






        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
                for col in range(len(plotDatadf.iloc[row]["regulationdata"][0])):
                    plt.plot(plotDatadf.iloc[row]["regulationdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Regulation")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
                plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Objective Value")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
        #         plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[count], color=colors[0])
                plt.plot(plotDatadf.iloc[row]["secondary_obj"],label='Secondary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Objective Value")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)

    elif len(TargetType)>1:
        for row in plotDatadf.index:
            
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                scalingVector = np.zeros(len(plotDatadf.iloc[row]["fluxdata"][:,0]))
                for scaleRxn in scaledByUptake:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):

                        if scaleRxn==plotDatadf.iloc[row]["reactionLabels"][col]:
                            scalingVector = scalingVector + plotDatadf.iloc[row]["fluxdata"][:,col]
                
                scalingMatrix.append(scalingVector)
                if len(scaledByUptake) >0:

                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col]/scalingVector,label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                else:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                
                
        plt.ylabel("Flux")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)


        lineStylecount=0
        markerStyleCount=0
        scalePlot=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                plt.plot(scalingMatrix[scalePlot],label='Scale factor', linestyle=linesSty[lineStylecount], color='b', marker=markerStyles[markerStyleCount])
                scalePlot+=1
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                
                
        plt.ylabel("Flux")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)






        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                for col in range(len(plotDatadf.iloc[row]["regulationdata"][0])):
                    plt.plot(plotDatadf.iloc[row]["regulationdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Regulation")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Objective Value")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
        #         plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[count], color=colors[0])
                plt.plot(plotDatadf.iloc[row]["secondary_obj"],label='Secondary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Objective Value")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)



    else:
        print("Error")

def PlotOneAgainstInitialConditionScaled(plotDatadf,scaledByUptake,conditionLabels,QueryModel,QueryObjType,TargetQuery,TargetType,experimentIDentifiers,saveDirectory="/output/",colors=["b","g" ,"r"],linesSty = ['-','--','-.']):
    
    

    markerStyles = ["","D","o","s","*"]
    scalingMatrix=[]
    lineStylecount=0
    markerStyleCount=0
    
    if len(TargetType)==1:


        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        ax.set_xticks(range(len(conditionLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])


        ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)
        lineStylecount=0
        markerStyleCount=0
        scalePlot=0
        for row in plotDatadf.index:
            
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:

                for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                    plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                     
        plt.ylabel("Flux")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.savefig(saveDirectory+'Flux'+'.png', bbox_inches='tight')
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        ax.set_xticks(range(len(conditionLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        scalePlot=0
        for row in plotDatadf.index:
            
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
                
                scalingVector = np.zeros(len(plotDatadf.iloc[row]["fluxdata"][:,0]))
                for scaleRxn in scaledByUptake:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        
                        if scaleRxn==plotDatadf.iloc[row]["reactionLabels"][col]:
                            scalingVector = scalingVector + plotDatadf.iloc[row]["fluxdata"][:,col]
                
                scalingMatrix.append(scalingVector)
                if len(scaledByUptake) >0:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col]/scalingVector,label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                else:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                     
        plt.ylabel("Scaled Flux")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.savefig(saveDirectory+'Flux'+'.png', bbox_inches='tight')
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        ax.set_xticks(range(len(conditionLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)


        lineStylecount=0
        markerStyleCount=0
        scalePlot=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
                
                plt.plot(scalingMatrix[scalePlot],label='Scale factor', linestyle=linesSty[lineStylecount], color='b', marker=markerStyles[markerStyleCount])
                scalePlot+=1
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                
                
        plt.ylabel("Flux")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.savefig(saveDirectory+'GrowthFlux'+'.png', bbox_inches='tight')
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        ax.set_xticks(range(len(conditionLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)






        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
                for col in range(len(plotDatadf.iloc[row]["regulationdata"][0])):
                    plt.plot(plotDatadf.iloc[row]["regulationdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Regulation")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.savefig(saveDirectory+'Regulation'+'.png', bbox_inches='tight')
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        ax.set_xticks(range(len(conditionLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
                plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Objective Value")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.savefig(saveDirectory+'Obj'+'.png', bbox_inches='tight')
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        ax.set_xticks(range(len(conditionLabels)))
        ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
        #         plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[count], color=colors[0])
                plt.plot(plotDatadf.iloc[row]["secondary_obj"],label='Secondary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Objective Value")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        ax.set_xticks(range(len(conditionLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)

    elif len(TargetType)>1:
        lineStylecount=0
        markerStyleCount=0
        scalePlot=0

        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        # ax.set_xticks(range(len(conditionLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])

        print([float(i) for i in conditionLabels])
        # ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)
        for row in plotDatadf.index:
            
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                    plt.plot([float(i) for i in conditionLabels],plotDatadf.iloc[row]["fluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
            
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                
        # plt.xscale('log')
        plt.ylabel("Flux")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.savefig(saveDirectory+'Flux'+'.png', bbox_inches='tight')
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        # ax.set_xticks(range(len(conditionLabels)))
        # # ax.set_xticklabels([int(i) for i in percentageLabels])
        # ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)
    
        lineStylecount=0
        markerStyleCount=0
        scalePlot=0
        for row in plotDatadf.index:
            
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                scalingVector = np.zeros(len(plotDatadf.iloc[row]["fluxdata"][:,0]))
                for scaleRxn in scaledByUptake:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):

                        if scaleRxn==plotDatadf.iloc[row]["reactionLabels"][col]:
                            scalingVector = scalingVector + plotDatadf.iloc[row]["fluxdata"][:,col]
                
                scalingMatrix.append(scalingVector)
                if len(scaledByUptake) >0:

                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        plt.plot([float(i) for i in conditionLabels],plotDatadf.iloc[row]["fluxdata"][:,col]/scalingVector,label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                else:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        plt.plot([float(i) for i in conditionLabels],plotDatadf.iloc[row]["fluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                
                
        plt.ylabel("Scaled Flux")
        # plt.xscale('log')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.savefig(saveDirectory+'Flux'+'.png', bbox_inches='tight')
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        # ax.set_xticks(range(len(conditionLabels)))
        # # ax.set_xticklabels([int(i) for i in percentageLabels])
        # ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)


        lineStylecount=0
        markerStyleCount=0
        scalePlot=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                plt.plot([float(i) for i in conditionLabels],scalingMatrix[scalePlot],label='GrowthFlux', linestyle=linesSty[lineStylecount], color='b', marker=markerStyles[markerStyleCount])
                scalePlot+=1
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                
                
        plt.ylabel("Flux")
        # plt.xscale('log')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.savefig(saveDirectory+'GrowthFlux'+'.png', bbox_inches='tight')
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        # ax.set_xticks(range(len(conditionLabels)))
        # # ax.set_xticklabels([int(i) for i in percentageLabels])
        # ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)






        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                for col in range(len(plotDatadf.iloc[row]["regulationdata"][0])):
                    plt.plot([float(i) for i in conditionLabels],plotDatadf.iloc[row]["regulationdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Regulation")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.xscale('log')
        #  ,'c','m' ,'y','k'
        plt.savefig(saveDirectory+'Regulation'+'.png', bbox_inches='tight')
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        # ax.set_xticks(range(len(conditionLabels)))
        # # ax.set_xticklabels([int(i) for i in percentageLabels])
        # ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                plt.plot([float(i) for i in conditionLabels],plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Objective Value")
        # plt.xscale('log')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.savefig(saveDirectory+'Obj'+'.png', bbox_inches='tight')
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        # ax.set_xticks(range(len(conditionLabels)))
        # # ax.set_xticklabels([int(i) for i in percentageLabels])
        # ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
        #         plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[count], color=colors[0])
                plt.plot([float(i) for i in conditionLabels],plotDatadf.iloc[row]["secondary_obj"],label='Secondary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Objective Value")
        # plt.xscale('log')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        ax = plt.axes()            
        plt.xlabel("Initial concentration")
        ax.set_xticks(range(len(conditionLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in conditionLabels],rotation=90)


        plt.show()
    else:
        print("Error")

def PlotOneAgainstGlucosePercentageMonodScaled(plotDatadf,scaledByUptake,uptakeConcentrations,percentageLabels,QueryModel,QueryObjType,TargetQuery,TargetType,experimentIDentifiers,colors=["b","g" ,"r"],linesSty = ['-','--','-.']):
    

    # Adjustable parameters:
    Vmax = 3.033e-12 # mM glucose consumed per hour per 1 micron^3 cell 
    l_hyphae = 0.003 # length of hyphal compartment in cm
    A_hyphae = 12.566e-07 # area of hyphal compartment in cm^2
    V_hyphae = l_hyphae*A_hyphae # volume of hyphal compartment
    V_hyphae = V_hyphae * 1.0e+12 # Convert to cubic microns
    Vmax = Vmax * V_hyphae # Scale Vmax to size of a hyphal compartment

    Km1 = 0.04 # moles
    Km2 = 25 # moles

    glucose_conc = metabolites.loc['D-GLUCOSE:EXTERNAL','Conc'] * 1000 # convert from M to mM.
    V1 = Vmax * glucose_conc/(Km1 + glucose_conc)
    V2 = Vmax * glucose_conc/(Km2 + glucose_conc)
    V = max(V1,V2)

    flux_scale = V/active_reactions.loc['Glucose import','Pyomo Flux']


    ax = plt.axes()            
    plt.xlabel("Glucose percentage")
    ax.set_xticks(range(len(percentageLabels)))
    # ax.set_xticklabels([int(i) for i in percentageLabels])
    ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)

    markerStyles = ["","D","o","s","*"]
    scalingMatrix=[]
    lineStylecount=0
    markerStyleCount=0
    
    if len(TargetType)==1:

        for row in plotDatadf.index:
            
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
                
                scalingVector = np.zeros(len(plotDatadf.iloc[row]["fluxdata"][:,0]))
                for scaleRxn in scaledByUptake:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        
                        if scaleRxn==plotDatadf.iloc[row]["reactionLabels"][col]:
                            scalingVector = scalingVector + plotDatadf.iloc[row]["fluxdata"][:,col]
                
                scalingMatrix.append(scalingVector)
                if len(scaledByUptake) >0:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col]/scalingVector,label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                else:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                
                
        plt.ylabel("Flux")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)


        lineStylecount=0
        markerStyleCount=0
        scalePlot=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
                
                plt.plot(scalingMatrix[scalePlot],label='Scale factor', linestyle=linesSty[lineStylecount], color='b', marker=markerStyles[markerStyleCount])
                scalePlot+=1
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                
                
        plt.ylabel("Flux")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)






        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
                for col in range(len(plotDatadf.iloc[row]["regulationdata"][0])):
                    plt.plot(plotDatadf.iloc[row]["regulationdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Regulation")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
                plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Objective Value")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] == TargetType:
        #         plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[count], color=colors[0])
                plt.plot(plotDatadf.iloc[row]["secondary_obj"],label='Secondary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Objective Value")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)

    elif len(TargetType)>1:
    
        for row in plotDatadf.index:
            
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                scalingVector = np.zeros(len(plotDatadf.iloc[row]["fluxdata"][:,0]))
                for scaleRxn in scaledByUptake:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):

                        if scaleRxn==plotDatadf.iloc[row]["reactionLabels"][col]:
                            scalingVector = scalingVector + plotDatadf.iloc[row]["fluxdata"][:,col]
                
                scalingMatrix.append(scalingVector)
                if len(scaledByUptake) >0:

                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col]/scalingVector,label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                else:
                    for col in range(len(plotDatadf.iloc[row]["fluxdata"][0])):
                        plt.plot(plotDatadf.iloc[row]["fluxdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                
                
        plt.ylabel("Flux")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)


        lineStylecount=0
        markerStyleCount=0
        scalePlot=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                plt.plot(scalingMatrix[scalePlot],label='Scale factor', linestyle=linesSty[lineStylecount], color='b', marker=markerStyles[markerStyleCount])
                scalePlot+=1
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
                print()
                
                
        plt.ylabel("Flux")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        
        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)






        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                for col in range(len(plotDatadf.iloc[row]["regulationdata"][0])):
                    plt.plot(plotDatadf.iloc[row]["regulationdata"][:,col],label=plotDatadf.iloc[row]["reactionLabels"][col], linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Regulation")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
                plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Objective Value")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()

        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)

        lineStylecount=0
        markerStyleCount=0
        for row in plotDatadf.index:
            if plotDatadf.iloc[row]["model"] == QueryModel and plotDatadf.iloc[row]["obj_type"] == QueryObjType and plotDatadf.iloc[row][TargetQuery] in TargetType:
                
        #         plt.plot(plotDatadf.iloc[row]["primary_obj"],label='Primary Obj', linestyle=linesSty[count], color=colors[0])
                plt.plot(plotDatadf.iloc[row]["secondary_obj"],label='Secondary Obj', linestyle=linesSty[lineStylecount], color=colors[col], marker=markerStyles[markerStyleCount])
                
                for iden in experimentIDentifiers:
                    print(str(iden)+": "+str(plotDatadf.iloc[row][iden])+" linestyle "+str(linesSty[lineStylecount]+" "+markerStyles[markerStyleCount]))
                
                lineStylecount+=1
                if lineStylecount == len(linesSty):
                    lineStylecount=0
                    markerStyleCount+=1
        plt.ylabel("Objective Value")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #  ,'c','m' ,'y','k'
        plt.show()
        ax = plt.axes()            
        plt.xlabel("Glucose percentage")
        ax.set_xticks(range(len(percentageLabels)))
        # ax.set_xticklabels([int(i) for i in percentageLabels])
        ax.set_xticklabels([float(i) for i in percentageLabels],rotation=90)



    else:
        print("Error")

def PlotMetaboliteSolutions(metabSolsdf):
    orderedMetabSolsdf=metabSolsdf.sort_values('metabolite_conc')
    
    plt.scatter(orderedMetabSolsdf.index, orderedMetabSolsdf['metabolite_conc'],label='metabolite concentration')
    plt.scatter(orderedMetabSolsdf.index, orderedMetabSolsdf['vconc_upper_bound'],label='concentration upper bound',marker="_")
    plt.scatter(orderedMetabSolsdf.index, orderedMetabSolsdf['vconc_lower_bound'],label='concentration lower bound',marker="_")
    
    plt.xticks(rotation=90)
    plt.title('Steady state concentrations for the metabolites in the model')
    plt.xlabel('Metabolite names')
    plt.ylabel('Concentrations')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    return

def PlotFluxSolutions(fluxSolsdf):
    orderedFluxSolsdf=fluxSolsdf.sort_values('flux')
    
    plt.scatter(orderedFluxSolsdf.index, orderedFluxSolsdf['flux'],label='Reaction flux',color='r')
    plt.scatter(orderedFluxSolsdf.index, np.zeros(len(orderedFluxSolsdf['flux'])),label='zero flux',color='orange',marker="_")

    plt.xticks(rotation=90)
    plt.title('Steady state reaction flux solutions')
    plt.xlabel('Reaction names')
    plt.ylabel('Reaction flux')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    plt.scatter(orderedFluxSolsdf.index, orderedFluxSolsdf['regulation'],label='Reaction regulation')

    plt.xticks(rotation=90)
    plt.title('Steady state reaction regulation')
    plt.xlabel('Reaction names')
    plt.ylabel('Regulation')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    return

def PlotBestSimulations(bestSimulations,control_dictionary,metaboliteSubset=[],reactionSubset=[],conditionVector=[]):

    T = 298.15
    R = 8.314e-03
    RT = R*T
    N_avogadro = 6.022140857e+23
    VolCell = 1.0e-15
    Concentration2Count = N_avogadro * VolCell

    S_active_best_vector=[]
    active_reactions_best_vector=[]
    metabolites_best_vector=[]
    metabSolsdf_vector=[]
    ysoldf_vector=[]
    name_vector=[]

    for simName in bestSimulations[0]['bestSimulations']:
        
        try:
            name_vector.append(simName)
            file=str(bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name']+'.pkl')
            file = file.replace(":",'_')
            
            file = re.sub(r"\s+",'_',file)
            
            y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound = baf.load_model_solution(control_dictionary['solution_file_dir']+"/output_"+file)

            S_active_best, active_reactions_best, metabolites_best, Keq_best = baf.load_model(control_dictionary['model_file_dir']+"/output_"+file)
            metabSols=n_sol.tolist()+f_log_counts.tolist()
            metabSolsConc=[]
            vconc_lower_bound=[]
            vconc_upper_bound=[]
            for i in range(len(metabSols)):
                metabSolsConc.append(np.exp(metabSols[i])/Concentration2Count)
                vconc_lower_bound.append(np.exp(vcount_lower_bound[i])/Concentration2Count)
                vconc_upper_bound.append(np.exp(vcount_upper_bound[i])/Concentration2Count)


                
            metabSolsdf=pd.DataFrame(metabSolsConc,index=metabolites_best.index,columns=['metabolite_conc'])
            metabSolsdf['vconc_lower_bound']=vconc_lower_bound
            metabSolsdf['vconc_upper_bound']=vconc_upper_bound    

            if metaboliteSubset !=[]:
                
                metabSolsdf = metabSolsdf.loc[metaboliteSubset]

            metabSolsdf_vector.append(metabSolsdf)

            ysoldf=pd.DataFrame(y_sol,index=active_reactions_best.index,columns=['flux'])
            ysoldf['regulation']=alpha_sol
            
            if reactionSubset !=[]:
                
                ysoldf = ysoldf.loc[reactionSubset]

            ysoldf_vector.append(ysoldf)
            
            S_active_best_vector.append(S_active_best) 
            active_reactions_best_vector.append(active_reactions_best)
            metabolites_best_vector.append(metabolites_best)
            
            print(simName)
            PlotMetaboliteSolutions(metabSolsdf)
            PlotFluxSolutions(ysoldf)

        except:
            print('simulation: '+str(simName)+' not found')


    if conditionVector !=[]:
        # plot the data together for the conditions

        metaboliteNames=metabSolsdf_vector[0].index
        reactionNames = ysoldf_vector[0].index

        concentrationdf = pd.DataFrame()
        concentrationdf['condition']=conditionVector
        
        for metab in metaboliteNames:
            concVector = np.zeros(len(conditionVector))

            for i in range(len(conditionVector)):

                concVector[i]=metabSolsdf_vector[i]['metabolite_conc'][metab]
            
            concentrationdf[metab]=concVector

        print("Concentrations:")
        display(concentrationdf)

        for metab in metaboliteNames:
            plt.scatter(concentrationdf['condition'], concentrationdf[metab],label=metab,marker="x")

        plt.xticks(rotation=90)
        plt.title('Steady state concentrations for the metabolites in the model')
        plt.xlabel('Condition')
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel('Concentrations')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        concentrationdf = concentrationdf.drop('condition', axis=1)
        plt.boxplot(concentrationdf,labels=concentrationdf.columns)
        plt.yscale("log")
        plt.xlabel('Metabolite')
        plt.ylabel('Concentrations')
        plt.title('Variation in metabolite concentration over all conditions')
        plt.xticks(rotation=90)
        plt.show()


        reactionNames = active_reactions_best_vector[0].index
        fluxdf = pd.DataFrame()
        fluxdf['condition']=conditionVector
        regulationdf = pd.DataFrame()
        regulationdf['condition']=conditionVector

        for rxn in reactionNames:
            fluxVector = np.zeros(len(conditionVector))

            for i in range(len(conditionVector)):

                fluxVector[i]=ysoldf_vector[i]['flux'][rxn]
            
            fluxdf[rxn]=fluxVector

        print("Flux:")
        display(fluxdf)

        for rxn in reactionNames:
            plt.scatter(fluxdf['condition'], fluxdf[rxn],label=rxn,marker="x")

        plt.xticks(rotation=90)
        plt.title('Steady state flux for the reactions in the model')
        plt.xlabel('Condition')
        plt.xscale("log")
        # plt.yscale("log")
        plt.ylabel('Flux')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        fluxdf = fluxdf.drop('condition', axis=1)
        plt.boxplot(fluxdf,labels=fluxdf.columns)
        plt.xticks(rotation=90)
        plt.xlabel('Reaction')
        plt.ylabel('Flux')
        plt.title('Variation in reaction flux over all conditions')
        plt.show()

        for rxn in reactionNames:
            regVector = np.zeros(len(conditionVector))

            for i in range(len(conditionVector)):

                regVector[i]=ysoldf_vector[i]['regulation'][rxn]
            
            regulationdf[rxn]=regVector

        print("Regulation:")
        display(regulationdf)

        for rxn in reactionNames:
            plt.scatter(regulationdf['condition'], regulationdf[rxn],label=rxn,marker="x")

        plt.xticks(rotation=90)
        plt.title('Steady state regulation for the reactions in the model')
        plt.xlabel('Condition')
        plt.xscale("log")
        # plt.yscale("log")
        plt.ylabel('Regulation')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        regulationdf = regulationdf.drop('condition', axis=1)
        plt.boxplot(regulationdf,labels=regulationdf.columns)
        plt.xticks(rotation=90)
        plt.xlabel('Reaction')
        plt.ylabel('Regulation')
        plt.title('Variation in reaction regulation over all conditions')
        plt.show()


    return

def ParseContinuousBestSimulations(bestSimulations,control_dictionary,metaboliteSubset=[],reactionSubset=[],verbose=False):

    T = 298.15
    R = 8.314e-03
    RT = R*T
    N_avogadro = 6.022140857e+23
    VolCell = 1.0e-15
    Concentration2Count = N_avogadro * VolCell

    S_active_best_vector=[]
    active_reactions_best_vector=[]
    metabolites_best_vector=[]
    metabSolsdf_vector=[]
    ysoldf_vector=[]
    name_vector=[]

    if len(bestSimulations[0]['bestSimulations']) >0:
        for simName in bestSimulations[0]['bestSimulations']:
            
            try:
                name_vector.append(simName)
                file=str(bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name']+'.pkl')
                file = file.replace(":",'_')
                
                y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound = baf.load_model_solution(control_dictionary['solution_file_dir']+"/output_"+file)

                S_active_best, active_reactions_best, metabolites_best, Keq_best = baf.load_model(control_dictionary['model_file_dir']+"/output_"+file)

                metabSols=n_sol.tolist()+f_log_counts.tolist()
                metabSolsConc=[]
                vconc_lower_bound=[]
                vconc_upper_bound=[]
                for i in range(len(metabSols)):
                    metabSolsConc.append(np.exp(metabSols[i])/Concentration2Count)
                    vconc_lower_bound.append(np.exp(vcount_lower_bound[i])/Concentration2Count)
                    vconc_upper_bound.append(np.exp(vcount_upper_bound[i])/Concentration2Count)


                    
                metabSolsdf=pd.DataFrame(metabSolsConc,index=metabolites_best.index,columns=['metabolite_conc'])
                metabSolsdf['vconc_lower_bound']=vconc_lower_bound
                metabSolsdf['vconc_upper_bound']=vconc_upper_bound    

                if metaboliteSubset !=[]:
                    
                    metabSolsdf = metabSolsdf.loc[metaboliteSubset]

                metabSolsdf_vector.append(metabSolsdf)

                ysoldf=pd.DataFrame(y_sol,index=active_reactions_best.index,columns=['flux'])
                ysoldf['regulation']=alpha_sol
                
                if reactionSubset !=[]:
                    
                    ysoldf = ysoldf.loc[reactionSubset]

                ysoldf_vector.append(ysoldf)
                
                S_active_best_vector.append(S_active_best) 
                active_reactions_best_vector.append(active_reactions_best)
                metabolites_best_vector.append(metabolites_best)
                
                if verbose:
                    print(simName)
                    PlotMetaboliteSolutions(metabSolsdf)
                    PlotFluxSolutions(ysoldf)
            
            except:

                S_active_best_vector.append(pd.DataFrame()) 
                active_reactions_best_vector.append(pd.DataFrame())
                metabolites_best_vector.append(pd.DataFrame())
    else:
        S_active_best_vector.append(pd.DataFrame()) 
        active_reactions_best_vector.append(pd.DataFrame())
        metabolites_best_vector.append(pd.DataFrame())

    return metabSolsdf_vector,ysoldf_vector

def ParseContinuousBestSimulations_opts(bestSimulations,control_dictionary,metaboliteSubset=[],reactionSubset=[],verbose=False):

    T = 298.15
    R = 8.314e-03
    RT = R*T
    N_avogadro = 6.022140857e+23
    VolCell = 1.0e-15
    Concentration2Count = N_avogadro * VolCell

    S_active_best_vector=[]
    active_reactions_best_vector=[]
    metabolites_best_vector=[]
    metabSolsdf_vector=[]
    ysoldf_vector=[]
    name_vector=[]
    opt_parms_vector=[]

    if len(bestSimulations[0]['bestSimulations']) >0:
        for simName in bestSimulations[0]['bestSimulations']:
            
            try:
                name_vector.append(simName)
                file=str(bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name']+'.pkl')
                file = file.replace(":",'_')
                
                y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound,opt_parms = baf.load_model_solution_opts(control_dictionary['solution_file_dir']+"/output_"+file)

                S_active_best, active_reactions_best, metabolites_best, Keq_best = baf.load_model(control_dictionary['model_file_dir']+"/output_"+file)

                opt_parms_vector.append(opt_parms)
                metabSols=n_sol.tolist()+f_log_counts.tolist()
                metabSolsConc=[]
                vconc_lower_bound=[]
                vconc_upper_bound=[]
                for i in range(len(metabSols)):
                    metabSolsConc.append(np.exp(metabSols[i])/Concentration2Count)
                    vconc_lower_bound.append(np.exp(vcount_lower_bound[i])/Concentration2Count)
                    vconc_upper_bound.append(np.exp(vcount_upper_bound[i])/Concentration2Count)


                    
                metabSolsdf=pd.DataFrame(metabSolsConc,index=metabolites_best.index,columns=['metabolite_conc'])
                metabSolsdf['vconc_lower_bound']=vconc_lower_bound
                metabSolsdf['vconc_upper_bound']=vconc_upper_bound    

                if metaboliteSubset !=[]:
                    
                    metabSolsdf = metabSolsdf.loc[metaboliteSubset]

                metabSolsdf_vector.append(metabSolsdf)

                ysoldf=pd.DataFrame(y_sol,index=active_reactions_best.index,columns=['flux'])
                ysoldf['regulation']=alpha_sol
                
                if reactionSubset !=[]:
                    
                    ysoldf = ysoldf.loc[reactionSubset]

                ysoldf_vector.append(ysoldf)
                
                S_active_best_vector.append(S_active_best) 
                active_reactions_best_vector.append(active_reactions_best)
                metabolites_best_vector.append(metabolites_best)
                
                if verbose:
                    print(simName)
                    PlotMetaboliteSolutions(metabSolsdf)
                    PlotFluxSolutions(ysoldf)
            
            except:

                S_active_best_vector.append(pd.DataFrame()) 
                active_reactions_best_vector.append(pd.DataFrame())
                metabolites_best_vector.append(pd.DataFrame())
    else:
        S_active_best_vector.append(pd.DataFrame()) 
        active_reactions_best_vector.append(pd.DataFrame())
        metabolites_best_vector.append(pd.DataFrame())

    return metabSolsdf_vector,ysoldf_vector,opt_parms_vector


def PlotContinuousSimulations(metabolitesdf_vector,fluxdf_vector,timesteps,verbose=False,metaboliteSubset=[],reactionSubset=[]):

    metaboliteNames=metabolitesdf_vector[0].index
    reactionNames = fluxdf_vector[0].index

    concentrationdf = pd.DataFrame()
    
    dt=timesteps[1]-timesteps[0]

    for metab in metaboliteNames:
        concVector = np.zeros(len(timesteps))

        for i in range(len(timesteps)):

            # if metabolitesdf_vector[i]['metabolite_conc'][metab] > 1e-4:
            #     concVector[i]=1e-4

            # elif metabolitesdf_vector[i]['metabolite_conc'][metab] < 1e-120:
            #     concVector[i]=1e-120
            # else:
            concVector[i]=metabolitesdf_vector[i]['metabolite_conc'][metab]

        concentrationdf[metab]=concVector

    if metaboliteSubset !=[]:
        concentrationdf = concentrationdf.loc[:,metaboliteSubset]
        metaboliteNames=metaboliteSubset
    
    concentrationdf['time']=timesteps
    print("Concentrations:")
    if verbose:
        display(concentrationdf)

    for metab in metaboliteNames:
        plt.scatter(concentrationdf['time'], concentrationdf[metab],label=metab,marker=".",s=10)

    plt.title('Steady state concentrations for the metabolites in the model')
    plt.xlabel('Time')
    # plt.yscale("log")
    # plt.ylim([0,2e-4])
    plt.ylabel('Concentrations')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    concentrationdf = concentrationdf.drop('time', axis=1)
    plt.boxplot(concentrationdf,labels=concentrationdf.columns)
    plt.xticks(rotation=90)
    plt.yscale("log")
    plt.xlabel('Metabolite')
    plt.ylabel('Concentrations')
    
    plt.title('Variation in metabolite concentration over all time')
    plt.show()

    fluxdf = pd.DataFrame()
    
    regulationdf = pd.DataFrame()
    

    for rxn in reactionNames:
        fluxVector = np.zeros(len(timesteps))

        for i in range(len(timesteps)):

            fluxVector[i]=fluxdf_vector[i]['flux'][rxn]

        fluxdf[rxn]=fluxVector

    if reactionSubset !=[]:
        fluxdf = fluxdf.loc[:,reactionSubset]
        reactionNames=reactionSubset

    fluxdf['time']=timesteps
    print("Flux:")
    if verbose:
        display(fluxdf)

    for rxn in reactionNames:
        plt.scatter(fluxdf['time'], fluxdf[rxn],label=rxn,marker=".",s=10)

    plt.title('Steady state flux for the reactions in the model')
    plt.xlabel('Time')
    plt.ylim([-100,150])


    # plt.yscale("log")
    plt.ylabel('Flux')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    

    for rxn in reactionNames:
        plt.scatter(fluxdf['time'], ConvertCountFluxToConcFlux(fluxdf[rxn].values.flatten().tolist(),dt),label=rxn,marker=".",s=10)

    plt.title('Steady state flux for the reactions in the model')
    plt.xlabel('Time')
    # plt.yscale("log")
    plt.ylabel('Instantaneous change in concentration')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    
    fluxdf = fluxdf.drop('time', axis=1)
    plt.boxplot(fluxdf,labels=fluxdf.columns)
    plt.xticks(rotation=90)
    plt.xlabel('Reaction')
    plt.ylabel('Flux')
    plt.title('Variation in reaction flux over all times')
    plt.show()


        


    for rxn in reactionNames:
        regVector = np.zeros(len(timesteps))

        for i in range(len(timesteps)):

            regVector[i]=fluxdf_vector[i]['regulation'][rxn]

        regulationdf[rxn]=regVector

    if reactionSubset !=[]:
        regulationdf = regulationdf.loc[:,reactionSubset]
        reactionNames=reactionSubset
        
    regulationdf['time']=timesteps
    print("Regulation:")
    if verbose:
        display(regulationdf)

    for rxn in reactionNames:
        plt.scatter(regulationdf['time'], regulationdf[rxn],label=rxn,marker=".",s=10)

    plt.title('Steady state regulation for the reactions in the model')
    plt.xlabel('Time')
    # plt.yscale("log")
    plt.ylim(-0.1,1.1)
    plt.ylabel('Regulation')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    regulationdf = regulationdf.drop('time', axis=1)
    plt.boxplot(regulationdf,labels=regulationdf.columns)
    plt.xticks(rotation=90)
    plt.xlabel('Reaction')
    plt.ylabel('Regulation')
    plt.title('Variation in reaction regulation over all conditions')
    plt.show()

    return

def PlotThermoOpt(metabolitesdf_vector,metabolitesODEConcentrationTraces,fluxdf_vector,reactionODEFluxTraces,timesteps,ode_dt,ode_maxTime,verbose=False,metaboliteSubset=[],reactionSubset=[]):
    
    
#     print('##################################')
#     print()
#     print('metabolitesODEConcentrationTraces')
# #    print(metabolitesODEConcentrationTraces.shape)
#     print(metabolitesODEConcentrationTraces)
#     print()
#     print('##################################')
#     print()
#     print('reactionODEFluxTraces')
# #     print(reactionODEFluxTraces.shape)
#     print(reactionODEFluxTraces)
    
    
    metaboliteNames=metabolitesdf_vector[0].index
    reactionNames = fluxdf_vector[0].index

    concentrationdf = pd.DataFrame()
    
    concentrationODEdf = pd.DataFrame()
    
    dt=timesteps[1]-timesteps[0]
    
    timesteps_ode = np.arange(0,ode_maxTime,ode_dt)
    fullOdeTimes = []
    
    for t in timesteps:
        for t_ode in timesteps_ode:
            fullOdeTimes.append(t+t_ode)
    
    # print(fullOdeTimes)
    
    for metab in metaboliteNames:
        concVector = np.zeros(len(timesteps))
        concODEVector= np.zeros(len(fullOdeTimes))

        for i in range(len(timesteps)):
            concVector[i]=metabolitesdf_vector[i]['metabolite_conc'][metab]
            
        for i in range(len(timesteps)):

            for j in range(len(timesteps_ode)):

                concODEVector[i*len(timesteps_ode)+j]=metabolitesODEConcentrationTraces[i].iloc[j][metab]

        concentrationdf[metab]=concVector
        concentrationODEdf[metab]=concODEVector

    if metaboliteSubset !=[]:
        concentrationdf = concentrationdf.loc[:,metaboliteSubset]
        concentrationODEdf = concentrationODEdf.loc[:,metaboliteSubset]
        metaboliteNames=metaboliteSubset
    
    concentrationdf['time']=timesteps
    concentrationODEdf['time']=fullOdeTimes

    # print("Concentrations:")
    # display(concentrationODEdf)
    
    if verbose:
        print("Concentrations:")
        display(concentrationdf)

    for metab in metaboliteNames:
        plt.scatter(concentrationdf['time'], concentrationdf[metab],label=metab,marker=".",s=20)
    for metab in metaboliteNames:
        plt.plot(concentrationODEdf['time'], concentrationODEdf[metab],label=metab)#,marker=".",s=5)

    plt.title('Steady state concentrations for the metabolites in the model')
    plt.xlabel('Time')
    # plt.yscale("log")
    # plt.ylim([0,2e-4])
    plt.ylabel('Concentrations')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    concentrationdf = concentrationdf.drop('time', axis=1)
    plt.boxplot(concentrationdf,labels=concentrationdf.columns)
    plt.xticks(rotation=90)
    plt.yscale("log")
    plt.xlabel('Metabolite')
    plt.ylabel('Concentrations')
    
    plt.title('Variation in metabolite concentration over all time')
    plt.show()
    
    
    
    
    
    
    fluxdf = pd.DataFrame()
    fluxODEdf = pd.DataFrame()
    
    regulationdf = pd.DataFrame()
    

    for rxn in reactionNames:
        fluxVector = np.zeros(len(timesteps))
        fluxODEVector= np.zeros(len(fullOdeTimes))

        for i in range(len(timesteps)):

            fluxVector[i]=fluxdf_vector[i]['flux'][rxn]
            
        for i in range(len(timesteps)):

            for j in range(len(timesteps_ode)):

                fluxODEVector[i*len(timesteps_ode)+j]=reactionODEFluxTraces[i].iloc[j][rxn]


        fluxdf[rxn]=fluxVector
        fluxODEdf[rxn]=fluxODEVector

    if reactionSubset !=[]:
        fluxdf = fluxdf.loc[:,reactionSubset]
        fluxODEdf = fluxODEdf.loc[:,reactionSubset]
        reactionNames=reactionSubset

    fluxdf['time']=timesteps
    fluxODEdf['time']=fullOdeTimes
     
    
    print("Flux:")
    if verbose:
        display(fluxdf)

    for rxn in reactionNames:
        plt.scatter(fluxdf['time'], fluxdf[rxn],label=rxn,marker=".",s=20)

    for rxn in reactionNames:
        plt.plot(fluxODEdf['time'], fluxODEdf[rxn],label=rxn)#,marker=".",s=5)


    plt.title('Flux')
    plt.xlabel('Time')
#     plt.ylim([-100,150])


    # plt.yscale("log")
    plt.ylabel('Flux')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    
    # plt.title('flux ode')
    # plt.xlabel('Time')
#     plt.ylim([-100,150])


    # plt.yscale("log")
    # plt.ylabel('Flux')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()

#     plt.title('Steady state flux for the reactions in the model')
#     plt.xlabel('Time')
# #     plt.ylim([-100,150])


#     # plt.yscale("log")
#     plt.ylabel('Flux')
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.show()

    for rxn in reactionNames:
        regVector = np.zeros(len(timesteps))

        for i in range(len(timesteps)):

            regVector[i]=fluxdf_vector[i]['regulation'][rxn]

        regulationdf[rxn]=regVector

    if reactionSubset !=[]:
        regulationdf = regulationdf.loc[:,reactionSubset]
        reactionNames=reactionSubset
        
    regulationdf['time']=timesteps
    print("Regulation:")
    if verbose:
        display(regulationdf)

    for rxn in reactionNames:
        plt.scatter(regulationdf['time'], regulationdf[rxn],label=rxn,marker=".",s=10)

    plt.title('Steady state regulation for the reactions in the model')
    plt.xlabel('Time')
    # plt.yscale("log")
    plt.ylim(-0.1,1.1)
    plt.ylabel('Regulation')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    regulationdf = regulationdf.drop('time', axis=1)
    plt.boxplot(regulationdf,labels=regulationdf.columns)
    plt.xticks(rotation=90)
    plt.xlabel('Reaction')
    plt.ylabel('Regulation')
    plt.title('Variation in reaction regulation over all conditions')
    plt.show()
    
    
    
    
    
    return

def ConvertCountFluxToConcFlux(countFlux,cell_volume = 1.0e-15):
    
    N_avogadro = 6.022140857e+23
    Concentration2Count = N_avogadro * cell_volume
    
    if type(countFlux) is list:

        concFlux=[]
        
        for rxnVal in countFlux:
            
            fluxSign = 1
            if rxnVal < 0.0:
                fluxSign = -1

            concFlux.append(fluxSign*np.exp(np.abs(rxnVal))/Concentration2Count)   #np.exp(np.abs(rxnVal))/Concentration2Count)   
        
        return concFlux
    
    else:
        fluxSign = 1
        if countFlux <0.0:
            fluxSign = -1

        concFlux=fluxSign*np.exp(np.abs(countFlux))/Concentration2Count #np.exp(np.abs(countFlux))/Concentration2Count   
        
        return concFlux
    

def DetermineEnvironmentChangeDueToCell(bestSimulations,control_dictionary,dt,flux_scale,environmentComp = 'ENVIRONMENT'):
    
    changedf_vector=[]
    scaledFluxdf_vector=[]

    N_avogadro = 6.022140857e+23
    VolCell = 1.0e-15
    Concentration2Count = N_avogadro * VolCell

    SurfaceAreaCell = pow(np.pi,1/3)*pow(6*VolCell,2/3)

    for simName in bestSimulations[0]['bestSimulations']:
        
        try:
        
            file=str(bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name']+'.pkl')
            file = file.replace(":",'_')


            y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound = baf.load_model_solution(control_dictionary['solution_file_dir']+"/output_"+file)
            S_active, active_reactions, metabolites, Keq = baf.load_model(control_dictionary['model_file_dir']+"/output_"+file)

            fluxOut=pd.DataFrame(y_sol,index=active_reactions.index,columns=['model flux'])
            fluxOut['unreg_rxn_flux']=unreg_rxn_flux

            # metabSols=n_sol.tolist()+f_log_counts.tolist()
            # metabSolsConc=[]
            # for i in range(len(metabSols)):
            #     metabSolsConc.append(np.exp(metabSols[i])/Concentration2Count)

            # metabSolsdf=pd.DataFrame(metabSolsConc,index=metabolites.index,columns=['metabolite_conc'])
            uptakeReactions,transportReactions,environmentReactions=baf.ParseReactionTypes(active_reactions,environmentComp)
            environmentMetabolites = baf.ParseMetaboliteTypes(metabolites,environmentComp)

            # concFlux=ConvertCountFluxToConcFlux(fluxOut.loc[uptakeReactions,"flux"].values.flatten().tolist())
            concchange=np.zeros(len(environmentMetabolites))

            changedf=pd.DataFrame({'conc_change':concchange},index=environmentMetabolites)
            scaledFluxdf=pd.DataFrame({'scaled_flux':flux_scale*fluxOut["model flux"].values.flatten()},index=active_reactions.index)
            scaledFluxdf_vector.append(scaledFluxdf)

            for i in range(len(uptakeReactions)):
                rxnName=uptakeReactions[i]
                rxnFlux=flux_scale*fluxOut.loc[uptakeReactions,"model flux"].values.flatten()[i]#concFlux[i]
                for metab in environmentMetabolites:
                    # print(metab)
                    # print('conc')
                    # print(metabSolsdf.at[metab,'metabolite_conc'])
                    # print('flux')
                    # print(rxnFlux)
                    changedf.at[metab,"conc_change"] += S_active.loc[rxnName,metab]*rxnFlux*SurfaceAreaCell#/VolCell#*metabSolsdf.at[metab,'metabolite_conc']
                    # print('change')
                    # print(changedf.at[metab,"conc_change"])

            changedf_vector.append(changedf)
        
        except:
            changedf_vector.append(pd.DataFrame())
            scaledFluxdf_vector.append(pd.DataFrame())
    
    return changedf_vector, scaledFluxdf_vector
