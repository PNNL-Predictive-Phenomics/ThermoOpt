"""

Connah G. M. Johnson     
connah.johnson@pnnl.gov
March 2024

Functions for analysing the ouput from flux constrained optimization simulations of metabolic models. 


Code contents:

def RunThermoOpt
def RunThermoOptNoRegulation
def RunThermoOptPopulationModel
def SaveSimulationOutput
def ReadFromPickle
def PlotFromPickle
def PlotGrowthRateFromPickle
def PlotScaling
def PlotGammaChange
def CleanThermoOpt
def SetUpEnvironmentCellDataFrames
def SetupEnvironmentChemostat
def SetUpScaling
def save_ode_model
def SetModelBoundary
def SetInitialModelBoundary
def SetJumpModelBoundary
def ComposeBaseSimulation
def ParseBestSimulations
def SetupODESystem
def ReadOPTSolution
def InitializeKineticConversion
def DetermineOptimalKinetics
def DetermineCorrectEnergyImport
def CalculateScaling
def CalculateRateConstants
def SimulateReactionStep
def SimulateReactionStepPopulationModel
def gradk_app
def DiffuseEnvironment
def gradDiff
def CalculateCellDensity
def UpdateEnvironmentDueToCellDensity
def UpdateODEBoundaryConditions
def PlotConcentrationTraces
def PlotFluxTraces
def PlotRegulationTraces
def PlotIntegratedBiomass


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
import time
import sys
import json
import pickle
import re
import os
import copy
import matplotlib.pyplot as plt
import Bolt_opt_functions as fopt
import base_functions as baf
import analysis_functions as anf
import scipy.integrate as it
from scipy.stats import pearsonr

# function to run the full thermoOpt method on the input dictionaries, involving multiple optimization calls
def RunThermoOpt(control_dictionary,ode_system_dictionary,environment_dictionary,cell_dictionary,ifVerbose=True):

    # in terms of concentration
    chemostatDf,environmentDf,cellContributionDf = SetUpEnvironmentCellDataFrames(environment_dictionary,ode_system_dictionary,False)

    # set up concentration/flux containers

    metabolitesOPT=[]
    fluxOPT=[]
    regulationOPT=[]
    timesOPT=[]
    loopTimes=[]

    n_loops = ode_system_dictionary['nmax']/ode_system_dictionary['Opt_frequency']
    nopf = ode_system_dictionary['Opt_frequency']
    n_steps = ode_system_dictionary['nmax']

    cell_dictionary = SetUpScaling(cell_dictionary,environment_dictionary,ode_system_dictionary,True)
    ode_system_dictionary['initialiseODE']=True

    numberOfRepeats=1
    if 'stall_on_opt_jump' in ode_system_dictionary.keys():
        numberOfRepeats = ode_system_dictionary['stall_on_opt_jump']

    jumpTol=1
    if 'jump_tolerance' in ode_system_dictionary.keys():
        jumpTol = ode_system_dictionary['jump_tolerance']

    cell_density = 0.0
    if 'initial_cell_density' in cell_dictionary.keys():
        cell_density = cell_dictionary['initial_cell_density']
    cell_density_vec=[]

    loop_time_one = time.time()
    loop_time_two = 0
    jt_loop = 0
    loop_time_i_one = time.time()
    loop_time_i_two = 0

    OPTUpdateSteps = np.arange(0,ode_system_dictionary['nmax'],ode_system_dictionary['Opt_frequency']).tolist()
    ifRecompute = False
        
    for i in range(ode_system_dictionary['nmax']):
        t_opt=ode_system_dictionary['timesteps'][i]

        try:
            if i in OPTUpdateSteps or ifRecompute:
                
                if i>0:
                    # save ODE solution for recalculation of kinetics
        
                    save_ode_model(control_dictionary['solution_file_dir']+"/"+previousBestModelSolutionFile+'.pkl',cell_dictionary,metaboliteTraceDf.loc[ode_system_dictionary['timesteps'][i-1],:].tolist(),fluxTraceDf.loc[ode_system_dictionary['timesteps'][i-1],:].tolist())
                    
                    # BCs determined using the environmentDf which is defined on ODE timesteps
                    # in concentrations
                    modelBoundaryConditions = SetModelBoundary(i,environment_dictionary,cell_dictionary,ode_system_dictionary,environmentDf,chemostatDf,cellContributionDf,metaboliteTraceDf)
            
                else:
                    modelBoundaryConditions = SetInitialModelBoundary(i,environment_dictionary,cell_dictionary,ode_system_dictionary,environmentDf,chemostatDf,cellContributionDf)
            
                # run the OPT routine
                
                base_simulation_list = ComposeBaseSimulation(modelBoundaryConditions,t_opt,cell_dictionary)
            
                bestSimulations = baf.RunSwarmSimulation(base_simulation_list,control_dictionary)

                metabolite_concs_opt,flux_opt,S_opt,active_reactions_opt = ParseBestSimulations(bestSimulations,control_dictionary,cell_dictionary)#,[],[],True)

                for simName in bestSimulations[0]['bestSimulations']:
                    if bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name'] != 'none':     
                        # these are used in the next optimiser iteration
                        previousBestModelSolutionFile = 'output_'+str(bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name'])

                if ode_system_dictionary['initialiseODE']:
                    # only triggers if ODE system has not already been initialised.
                    metaboliteTraceDf,fluxTraceDf,simKineticsDf,k_app_f,k_app_r,sim_concs = SetupODESystem(ode_system_dictionary,environment_dictionary,metabolite_concs_opt,flux_opt)
                    v_hat,a_hat,n_hat,S,Keq,metabolite_labels,reaction_labels = ReadOPTSolution(bestSimulations,control_dictionary,cell_dictionary)
                    cell_dictionary = InitializeKineticConversion(t_opt,v_hat,S,Keq,n_hat,cell_dictionary,environment_dictionary,ode_system_dictionary,metabolite_labels,reaction_labels)

                v_hat,a_hat,n_hat,S,Keq,metabolite_labels,reaction_labels = ReadOPTSolution(bestSimulations,control_dictionary,cell_dictionary)

                kineticsDf = DetermineOptimalKinetics(t_opt,v_hat,S,Keq,n_hat,cell_dictionary,environment_dictionary,ode_system_dictionary,metabolite_labels,reaction_labels)
                
                ifJump = False
                # test if there has been a jump in opt solution
                if i>0:
                    for j in range(len(fluxOPT[-1])):    
                        previousOptFluxVal = fluxOPT[-1]['Optimal Flux'][j]
                        newOptFluxVal = kineticsDf['Optimal Flux'].tolist()[j]

                        if np.abs((newOptFluxVal - previousOptFluxVal) / previousOptFluxVal) > jumpTol:
                            ifJump = True
                            break
                
                    if ifJump:

                        oldPreviousBestModelSolutionFile = previousBestModelSolutionFile
                        metabolite_concs_opt_old = metabolite_concs_opt
                        kineticsDf_old = kineticsDf
                        a_hat_old = a_hat

                        
                        repeatNumber=0
                        
                        while(repeatNumber<numberOfRepeats and ifJump):
                        
                        
                            # perturb BCs
                            modelBoundaryConditions = SetJumpModelBoundary(i,environment_dictionary,cell_dictionary,ode_system_dictionary,environmentDf,chemostatDf,cellContributionDf,metaboliteTraceDf)

                            # rerun the OPT routine
                            base_simulation_list = ComposeBaseSimulation(modelBoundaryConditions,t_opt,cell_dictionary)
                            bestSimulations = baf.RunSwarmSimulation(base_simulation_list,control_dictionary)
                            metabolite_concs_opt,flux_opt,S_opt,active_reactions_opt = ParseBestSimulations(bestSimulations,control_dictionary,cell_dictionary)#,[],[],True)
                            for simName in bestSimulations[0]['bestSimulations']:
                                if bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name'] != 'none':     
                                    # these are used in the next optimiser iteration
                                    previousBestModelSolutionFile = 'output_'+str(bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name'])
                            v_hat,a_hat,n_hat,S,Keq,metabolite_labels,reaction_labels = ReadOPTSolution(bestSimulations,control_dictionary,cell_dictionary)
                            kineticsDf = DetermineOptimalKinetics(t_opt,v_hat,S,Keq,n_hat,cell_dictionary,environment_dictionary,ode_system_dictionary,metabolite_labels,reaction_labels)
                            
                            repeatNumber +=1
                            
                            ifJump = False
                            for j in range(len(fluxOPT[-1])):    
                                previousOptFluxVal = fluxOPT[-1]['Optimal Flux'][j]
                                newOptFluxVal = kineticsDf['Optimal Flux'].tolist()[j]

                                if np.abs((newOptFluxVal - previousOptFluxVal) / previousOptFluxVal) > jumpTol:
                                    ifJump = True
                                    break
                            
                                
                        if ifJump:
                            # still a jump in a value, use previous opt solution
                            previousBestModelSolutionFile = oldPreviousBestModelSolutionFile
                            
                            metabolite_concs_opt = metabolite_concs_opt_old
                            kineticsDf = kineticsDf_old
                            a_hat = a_hat_old
                        # otherwise use the latest non-jump values     
                            
                
                
                metabolitesOPT.append(metabolite_concs_opt)
                fluxOPT.append(kineticsDf)
                regulationOPT.append(a_hat)
                timesOPT.append(t_opt)

                loop_time_two = time.time()
                opt_loop_time = loop_time_two - loop_time_one
                loop_time_one = loop_time_two
                loopTimes.append(opt_loop_time)
                jt_loop+=1

                if ifVerbose:
                    print('Loop Time in seconds: ',opt_loop_time)
                    print('ETA in minutes: ', (n_loops - i/nopf)*opt_loop_time/60)
        
        except:
            print('Error has occurred, output what we have saved so far')
        
        if ode_system_dictionary['initialiseODE']:
        
            ode_system_dictionary['initialiseODE']=False
            
            for k in range(len(sim_concs)):
                metaboliteTraceDf.at[t_opt,metaboliteTraceDf.columns.tolist()[k]] = sim_concs[k]
            for l in range(len(kineticsDf['Optimal Flux'].tolist())):
                fluxTraceDf.at[t_opt,fluxTraceDf.columns.tolist()[l]] = kineticsDf['Optimal Flux'].tolist()[l]
            
            k_app_f = kineticsDf.loc[:,'kf'].values
            k_app_r = kineticsDf.loc[:,'kr'].values
            
            for m in range(len(k_app_f)):
                simKineticsDf.at[t_opt,simKineticsDf.columns.tolist()[2*m]]   = k_app_f[m]
                simKineticsDf.at[t_opt,simKineticsDf.columns.tolist()[2*m+1]] = k_app_r[m]

        
        else:
            metaboliteTraceDf,fluxTraceDf,simKineticsDf,sim_concs, k_app_f, k_app_r = SimulateReactionStep(t_opt,kineticsDf,k_app_f,k_app_r, sim_concs, metaboliteTraceDf,fluxTraceDf,simKineticsDf, S, metabolite_labels,ode_system_dictionary,environment_dictionary,cell_dictionary,chemostatDf)

        cell_density = CalculateCellDensity(cell_dictionary,ode_system_dictionary,cell_density,fluxTraceDf.loc[t_opt,'Biomass'])
        cell_density_vec.append(cell_density)    
        cell_dictionary['cell_density'] = cell_density

        if i%10000 ==0:
            print(i)
            loop_time_i_two = time.time()
            opt_loop_time_i = loop_time_i_two - loop_time_i_one
            loop_time_i_one = loop_time_i_two
            print('ETA (minutes): ', ((n_steps - i )/10000)*opt_loop_time_i/60 )

        

    return metaboliteTraceDf,metabolitesOPT,fluxTraceDf,fluxOPT,regulationOPT,timesOPT,cell_density_vec,loopTimes

# function to run the thermoOpt method on the input dictionaries without involving reaction regulation outside of the first time step
# involves a single optimization call to determine initial kinetic parameters but the solely runs the ODE formulation
def RunThermoOptNoRegulation(control_dictionary,ode_system_dictionary,environment_dictionary,cell_dictionary,ifVerbose=True):

    # in terms of concentration
    chemostatDf,environmentDf,cellContributionDf = SetUpEnvironmentCellDataFrames(environment_dictionary,ode_system_dictionary,False)

    # set up concentration/flux containers

    metabolitesOPT=[]
    fluxOPT=[]
    regulationOPT=[]
    timesOPT=[]
    loopTimes=[]

    n_loops = ode_system_dictionary['nmax']/ode_system_dictionary['Opt_frequency']
    nopf = ode_system_dictionary['Opt_frequency']

    cell_dictionary = SetUpScaling(cell_dictionary,environment_dictionary,ode_system_dictionary,True)
    ode_system_dictionary['initialiseODE']=True

    numberOfRepeats=1
    if 'stall_on_opt_jump' in ode_system_dictionary.keys():
        numberOfRepeats = ode_system_dictionary['stall_on_opt_jump']

    jumpTol=1
    if 'jump_tolerance' in ode_system_dictionary.keys():
        jumpTol = ode_system_dictionary['jump_tolerance']

    cell_density = 0.0
    if 'initial_cell_density' in cell_dictionary.keys():
        cell_density = cell_dictionary['initial_cell_density']
    cell_density_vec=[]

    loop_time_one = time.time()
    loop_time_two = 0

    OPTUpdateSteps = [0]
    ifRecompute = False
        
    for i in range(ode_system_dictionary['nmax']):
        t_opt=ode_system_dictionary['timesteps'][i]

        if i in OPTUpdateSteps or ifRecompute:

            modelBoundaryConditions = SetInitialModelBoundary(i,environment_dictionary,cell_dictionary,ode_system_dictionary,environmentDf,chemostatDf,cellContributionDf)
        
            # run the OPT routine
            
            base_simulation_list = ComposeBaseSimulation(modelBoundaryConditions,t_opt,cell_dictionary)
        
            bestSimulations = baf.RunSwarmSimulation(base_simulation_list,control_dictionary)

            metabolite_concs_opt,flux_opt,S_opt,active_reactions_opt = ParseBestSimulations(bestSimulations,control_dictionary,cell_dictionary)#,[],[],True)
            
            
            for simName in bestSimulations[0]['bestSimulations']:
                if bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name'] != 'none':     
                    # these are used in the next optimiser iteration
                    previousBestModelSolutionFile = 'output_'+str(bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name'])
            
            if ode_system_dictionary['initialiseODE']:
                # only triggers if ODE system has not already been initialised.
                metaboliteTraceDf,fluxTraceDf,simKineticsDf,k_app_f,k_app_r,sim_concs = SetupODESystem(ode_system_dictionary,environment_dictionary,metabolite_concs_opt,flux_opt)
                v_hat,a_hat,n_hat,S,Keq,metabolite_labels,reaction_labels = ReadOPTSolution(bestSimulations,control_dictionary,cell_dictionary)
                cell_dictionary = InitializeKineticConversion(t_opt,v_hat,S,Keq,n_hat,cell_dictionary,environment_dictionary,ode_system_dictionary,metabolite_labels,reaction_labels)

            v_hat,a_hat,n_hat,S,Keq,metabolite_labels,reaction_labels = ReadOPTSolution(bestSimulations,control_dictionary,cell_dictionary)

            kineticsDf = DetermineOptimalKinetics(t_opt,v_hat,S,Keq,n_hat,cell_dictionary,environment_dictionary,ode_system_dictionary,metabolite_labels,reaction_labels)
            

            metabolitesOPT.append(metabolite_concs_opt)
            fluxOPT.append(kineticsDf)
            regulationOPT.append(a_hat)
            timesOPT.append(t_opt)

            loop_time_two = time.time()
            opt_loop_time = loop_time_two - loop_time_one
            loop_time_one = loop_time_two
            loopTimes.append(opt_loop_time)
            if ifVerbose:
                print('Loop Time in seconds: ',opt_loop_time)
                print('ETA in minutes: ', (n_loops - i/nopf)*opt_loop_time/60)
        
        
        
        if ode_system_dictionary['initialiseODE']:
        
            ode_system_dictionary['initialiseODE']=False
            
            for k in range(len(sim_concs)):
                metaboliteTraceDf.at[t_opt,metaboliteTraceDf.columns.tolist()[k]] = sim_concs[k]
            for l in range(len(kineticsDf['Optimal Flux'].tolist())):
                fluxTraceDf.at[t_opt,fluxTraceDf.columns.tolist()[l]] = kineticsDf['Optimal Flux'].tolist()[l]
            
            k_app_f = kineticsDf.loc[:,'kf'].values
            k_app_r = kineticsDf.loc[:,'kr'].values
            
            for m in range(len(k_app_f)):
                simKineticsDf.at[t_opt,simKineticsDf.columns.tolist()[2*m]]   = k_app_f[m]
                simKineticsDf.at[t_opt,simKineticsDf.columns.tolist()[2*m+1]] = k_app_r[m]

        
        else:
            metaboliteTraceDf,fluxTraceDf,simKineticsDf,sim_concs,k_app_f,k_app_r = SimulateReactionStep(t_opt,kineticsDf,k_app_f,k_app_r, sim_concs, metaboliteTraceDf,fluxTraceDf,simKineticsDf, S, metabolite_labels,ode_system_dictionary,environment_dictionary,cell_dictionary,chemostatDf)

        cell_density = CalculateCellDensity(cell_dictionary,ode_system_dictionary,cell_density,fluxTraceDf.loc[t_opt,'Biomass'])
        cell_density_vec.append(cell_density)    
        cell_dictionary['cell_density'] = cell_density

    return metaboliteTraceDf,metabolitesOPT,fluxTraceDf,fluxOPT,regulationOPT,timesOPT,cell_density_vec,loopTimes

# function to run ThermoOpt as a population model by focussing only on changes in the environment
def RunThermoOptPopulationModel(control_dictionary,ode_system_dictionary,environment_dictionary,cell_dictionary,ifVerbose=True):

    # in terms of concentration
    chemostatDf,environmentDf,cellContributionDf = SetUpEnvironmentCellDataFrames(environment_dictionary,ode_system_dictionary,False)

    # set up concentration/flux containers

    metabolitesOPT=[]
    fluxOPT=[]
    regulationOPT=[]
    timesOPT=[]
    loopTimes=[]

    n_loops = ode_system_dictionary['nmax']/ode_system_dictionary['Opt_frequency']
    nopf = ode_system_dictionary['Opt_frequency']

    cell_dictionary = SetUpScaling(cell_dictionary,environment_dictionary,ode_system_dictionary,True)
    ode_system_dictionary['initialiseODE']=True

    numberOfRepeats=1
    if 'stall_on_opt_jump' in ode_system_dictionary.keys():
        numberOfRepeats = ode_system_dictionary['stall_on_opt_jump']

    jumpTol=1
    if 'jump_tolerance' in ode_system_dictionary.keys():
        jumpTol = ode_system_dictionary['jump_tolerance']

    cell_density = 0.0
    if 'initial_cell_density' in cell_dictionary.keys():
        cell_density = cell_dictionary['initial_cell_density']
    cell_density_vec=[]

    loop_time_one = time.time()
    loop_time_two = 0
    jt_loop = 0

    OPTUpdateSteps = np.arange(0,ode_system_dictionary['nmax'],ode_system_dictionary['Opt_frequency']).tolist()
    ifRecompute = False
        
    for i in range(ode_system_dictionary['nmax']):
        t_opt=ode_system_dictionary['timesteps'][i]

        try:
            if i in OPTUpdateSteps or ifRecompute:
                
                if i>0:
                    # save ODE solution for recalculation of kinetics
        
                    save_ode_model(control_dictionary['solution_file_dir']+"/"+previousBestModelSolutionFile+'.pkl',cell_dictionary,metaboliteTraceDf.loc[ode_system_dictionary['timesteps'][i-1],:].tolist(),fluxTraceDf.loc[ode_system_dictionary['timesteps'][i-1],:].tolist())
                    
                    # BCs determined using the environmentDf which is defined on ODE timesteps
                    # in concentrations
                    modelBoundaryConditions = SetModelBoundary(i,environment_dictionary,cell_dictionary,ode_system_dictionary,environmentDf,chemostatDf,cellContributionDf,metaboliteTraceDf)
            
                else:
                    modelBoundaryConditions = SetInitialModelBoundary(i,environment_dictionary,cell_dictionary,ode_system_dictionary,environmentDf,chemostatDf,cellContributionDf)
            
                # run the OPT routine
                
                base_simulation_list = ComposeBaseSimulation(modelBoundaryConditions,t_opt,cell_dictionary)
            
                bestSimulations = baf.RunSwarmSimulation(base_simulation_list,control_dictionary)

                metabolite_concs_opt,flux_opt,S_opt,active_reactions_opt = ParseBestSimulations(bestSimulations,control_dictionary,cell_dictionary)#,[],[],True)

                for simName in bestSimulations[0]['bestSimulations']:
                    if bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name'] != 'none':     
                        # these are used in the next optimiser iteration
                        previousBestModelSolutionFile = 'output_'+str(bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name'])

                if ode_system_dictionary['initialiseODE']:
                    # only triggers if ODE system has not already been initialised.
                    metaboliteTraceDf,fluxTraceDf,simKineticsDf,k_app_f,k_app_r,sim_concs = SetupODESystem(ode_system_dictionary,environment_dictionary,metabolite_concs_opt,flux_opt)
                    v_hat,a_hat,n_hat,S,Keq,metabolite_labels,reaction_labels = ReadOPTSolution(bestSimulations,control_dictionary,cell_dictionary)
                    cell_dictionary = InitializeKineticConversion(t_opt,v_hat,S,Keq,n_hat,cell_dictionary,environment_dictionary,ode_system_dictionary,metabolite_labels,reaction_labels)

                v_hat,a_hat,n_hat,S,Keq,metabolite_labels,reaction_labels = ReadOPTSolution(bestSimulations,control_dictionary,cell_dictionary)

                kineticsDf = DetermineOptimalKinetics(t_opt,v_hat,S,Keq,n_hat,cell_dictionary,environment_dictionary,ode_system_dictionary,metabolite_labels,reaction_labels)
                
                ifJump = False
                # test if there has been a jump in opt solution
                if i>0:
                    for j in range(len(fluxOPT[-1])):    
                        previousOptFluxVal = fluxOPT[-1]['Optimal Flux'][j]
                        newOptFluxVal = kineticsDf['Optimal Flux'].tolist()[j]

                        if np.abs((newOptFluxVal - previousOptFluxVal) / previousOptFluxVal) > jumpTol:
                            ifJump = True
                            break
                
                    if ifJump:

                        oldPreviousBestModelSolutionFile = previousBestModelSolutionFile
                        metabolite_concs_opt_old = metabolite_concs_opt
                        kineticsDf_old = kineticsDf
                        a_hat_old = a_hat

                        
                        repeatNumber=0
                        
                        while(repeatNumber<numberOfRepeats and ifJump):
                        
                        
                            # perturb BCs
                            modelBoundaryConditions = SetJumpModelBoundary(i,environment_dictionary,cell_dictionary,ode_system_dictionary,environmentDf,chemostatDf,cellContributionDf,metaboliteTraceDf)

                            # rerun the OPT routine
                            base_simulation_list = ComposeBaseSimulation(modelBoundaryConditions,t_opt,cell_dictionary)
                            bestSimulations = baf.RunSwarmSimulation(base_simulation_list,control_dictionary)
                            metabolite_concs_opt,flux_opt,S_opt,active_reactions_opt = ParseBestSimulations(bestSimulations,control_dictionary,cell_dictionary)#,[],[],True)
                            for simName in bestSimulations[0]['bestSimulations']:
                                if bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name'] != 'none':     
                                    # these are used in the next optimiser iteration
                                    previousBestModelSolutionFile = 'output_'+str(bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name'])
                            v_hat,a_hat,n_hat,S,Keq,metabolite_labels,reaction_labels = ReadOPTSolution(bestSimulations,control_dictionary,cell_dictionary)
                            kineticsDf = DetermineOptimalKinetics(t_opt,v_hat,S,Keq,n_hat,cell_dictionary,environment_dictionary,ode_system_dictionary,metabolite_labels,reaction_labels)
                            
                            repeatNumber +=1
                            
                            ifJump = False
                            for j in range(len(fluxOPT[-1])):    
                                previousOptFluxVal = fluxOPT[-1]['Optimal Flux'][j]
                                newOptFluxVal = kineticsDf['Optimal Flux'].tolist()[j]

                                if np.abs((newOptFluxVal - previousOptFluxVal) / previousOptFluxVal) > jumpTol:
                                    ifJump = True
                                    break
                            
                                
                        if ifJump:
                            # still a jump in a value, use previous opt solution
                            previousBestModelSolutionFile = oldPreviousBestModelSolutionFile
                            
                            metabolite_concs_opt = metabolite_concs_opt_old
                            kineticsDf = kineticsDf_old
                            a_hat = a_hat_old
                        # otherwise use the latest non-jump values     

                    # for the population model, set the sim_concs for internal metabolites to steady state (n_hat)
                    externalMetaboliteNames = environment_dictionary['metabolites']
                    for metab in metabolite_labels:
                        if metab not in externalMetaboliteNames:
                            i=metabolite_labels.tolist().index(metab)
                            sim_concs[i] = n_hat[i]

                
                
                metabolitesOPT.append(metabolite_concs_opt)
                fluxOPT.append(kineticsDf)
                regulationOPT.append(a_hat)
                timesOPT.append(t_opt)

                loop_time_two = time.time()
                opt_loop_time = loop_time_two - loop_time_one
                loop_time_one = loop_time_two
                loopTimes.append(opt_loop_time)

                if ifVerbose:
                    print('Loop Time in seconds: ',opt_loop_time)
                    print('ETA in minutes: ', (n_loops - i/nopf)*opt_loop_time/60)

                jt_loop+=1
            
        except:
            print('Error has occurred, output what we have saved so far')    
        
        if ode_system_dictionary['initialiseODE']:
        
            ode_system_dictionary['initialiseODE']=False
            
            for k in range(len(sim_concs)):
                metaboliteTraceDf.at[t_opt,metaboliteTraceDf.columns.tolist()[k]] = sim_concs[k]
            for l in range(len(kineticsDf['Optimal Flux'].tolist())):
                fluxTraceDf.at[t_opt,fluxTraceDf.columns.tolist()[l]] = kineticsDf['Optimal Flux'].tolist()[l]
            
            k_app_f = kineticsDf.loc[:,'kf'].values
            k_app_r = kineticsDf.loc[:,'kr'].values
            
            for m in range(len(k_app_f)):
                simKineticsDf.at[t_opt,simKineticsDf.columns.tolist()[2*m]]   = k_app_f[m]
                simKineticsDf.at[t_opt,simKineticsDf.columns.tolist()[2*m+1]] = k_app_r[m]

        
        else:
            metaboliteTraceDf,fluxTraceDf,simKineticsDf,sim_concs = SimulateReactionStepPopulationModel(t_opt,kineticsDf,k_app_f,k_app_r, sim_concs, metaboliteTraceDf,fluxTraceDf,simKineticsDf, S, metabolite_labels,ode_system_dictionary,environment_dictionary,cell_dictionary,chemostatDf)

        cell_density = CalculateCellDensity(cell_dictionary,ode_system_dictionary,cell_density,fluxTraceDf.loc[t_opt,'Biomass'])
        cell_density_vec.append(cell_density)    
        cell_dictionary['cell_density'] = cell_density

        

    return metaboliteTraceDf,metabolitesOPT,fluxTraceDf,fluxOPT,regulationOPT,timesOPT,cell_density_vec,loopTimes




# function to save the simulation output of RunThermoOpt as a pickle file
# file_details can be the control dictionary with a 'pickle_output' entry or a string filename
def SaveSimulationOutput(file_details,metaboliteTraceDf,metabolitesOPT,fluxTraceDf,fluxOPT,regulationOPT,timesOPT,cell_density_vec,loopTimes):
    
    if isinstance(file_details,dict):
        if 'pickle_output' in file_details.keys():
            if file_details['pickle_output'] != "":
                data={}
                data['metaboliteTraceDf']=metaboliteTraceDf
                data['fluxTraceDf'] = fluxTraceDf
                data['metabolitesOPT']=metabolitesOPT
                data['fluxOPT']=fluxOPT
                data['regulationOPT']=regulationOPT
                data['timesOPT']=timesOPT
                data['cell_density_vec']=cell_density_vec
                data['loopTimes']=loopTimes
                with open(file_details['pickle_output'], 'wb') as fp:
                    pickle.dump(data, fp)
    else:
        data={}
        data['metaboliteTraceDf']=metaboliteTraceDf
        data['fluxTraceDf'] = fluxTraceDf
        data['metabolitesOPT']=metabolitesOPT
        data['fluxOPT']=fluxOPT
        data['regulationOPT']=regulationOPT
        data['timesOPT']=timesOPT
        data['cell_density_vec']=cell_density_vec
        data['loopTimes']=loopTimes
        with open(file_details, 'wb') as fp:
            pickle.dump(data, fp)
    return

def ReadFromPickle(file_details):

    metaboliteTraceDf = []
    fluxTraceDf = []
    metabolitesOPT = []
    fluxOPT = []
    regulationOPT = []
    timesOPT = []
    cell_density_vec = []
    loopTimes = []

    if isinstance(file_details,dict):

        if 'pickle_output' in file_details.keys():
            if file_details['pickle_output'] != "":
                data={}
                with open(file_details['pickle_output'], 'rb') as fp:
                    data = pickle.load(fp)
                metaboliteTraceDf = data['metaboliteTraceDf']
                fluxTraceDf = data['fluxTraceDf']
                metabolitesOPT = data['metabolitesOPT']
                fluxOPT = data['fluxOPT']
                regulationOPT = data['regulationOPT']
                timesOPT = data['timesOPT']
                cell_density_vec = data['cell_density_vec']
                loopTimes=data['loopTimes']
                
    else:
        # assume input is file name
        data={}
        with open(file_details, 'rb') as fp:
            data = pickle.load(fp)
        metaboliteTraceDf = data['metaboliteTraceDf']
        fluxTraceDf = data['fluxTraceDf']
        metabolitesOPT = data['metabolitesOPT']
        fluxOPT = data['fluxOPT']
        regulationOPT = data['regulationOPT']
        timesOPT = data['timesOPT']
        cell_density_vec = data['cell_density_vec']
        loopTimes=data['loopTimes']
    
    return metaboliteTraceDf,metabolitesOPT,fluxTraceDf,fluxOPT,regulationOPT,timesOPT,cell_density_vec,loopTimes

# function to plot the simulation data from the saved pickle files
def PlotFromPickle(file_details,plot_dictionary):
    
    metaboliteTraceDf,metabolitesOPT,fluxTraceDf,fluxOPT,regulationOPT,timesOPT,cell_density_vec,loopTimes = ReadFromPickle(file_details)

    # cell density plot
    plt.plot(cell_density_vec)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Cell density')
    plt.show()
    
    xlim=0
    if 'time_limit' in plot_dictionary.keys():
        xlim=plot_dictionary['time_limit']

    scatterSize=0
    if 'scatter_size' in plot_dictionary.keys():
        scatterSize=plot_dictionary['scatter_size']

    
    metaboliteSubset=[]
    if 'metabolite_subset' in plot_dictionary.keys():
        metaboliteSubset=plot_dictionary['metabolite_subset']
        
    reactionSubset=[]
    if 'reaction_subset' in plot_dictionary.keys():
        reactionSubset=plot_dictionary['reaction_subset']
        
    regulationSubset=[]
    if 'regulation_subset' in plot_dictionary.keys():
        regulationSubset=plot_dictionary['regulation_subset']
        
    metaboliteExchangeSubset=[]
    if 'exchange_metabolite' in plot_dictionary.keys():
        metaboliteExchangeSubset=plot_dictionary['exchange_metabolite']
        
    reactionExchangeSubset=[]
    if 'exchange_reaction' in plot_dictionary.keys():
        reactionExchangeSubset=plot_dictionary['exchange_reaction']
        
    biomassMetabolite=[]
    if 'biomass_metabolite' in plot_dictionary.keys():
        biomassMetabolite=plot_dictionary['biomass_metabolite']
        
    biomassFlux=[]
    if 'biomass_reaction' in plot_dictionary.keys():
        biomassFlux=plot_dictionary['biomass_reaction']
    
    # conc plots with ODE and OPT solution
    PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,True,metaboliteSubset,xlim,scatterSize)
    
    # conc plots with ODE solution only
    PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,False,metaboliteSubset,xlim,scatterSize)
    
    # flux plots with ODE and OPT solution
    PlotFluxTraces(fluxTraceDf,fluxOPT,timesOPT,True,reactionSubset,xlim,scatterSize)
    
    # flux plots with ODE solution only
    PlotFluxTraces(fluxTraceDf,fluxOPT,timesOPT,False,reactionSubset,xlim,scatterSize)
    
    # plot the reaction regulation values from the OPT solution
    PlotRegulationTraces(regulationOPT,timesOPT,fluxTraceDf.columns.tolist(),regulationSubset)

    # plot biomass concentraion trace only
    PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,False,[biomassMetabolite],xlim,scatterSize)
    # # plot biomass flux trace only
    # PlotFluxTraces(fluxTraceDf,fluxOPT,timesOPT,True,[biomassFlux],xlim,scatterSize)
    
    # if 'biomass_reaction' in plot_dictionary.keys():
    #     PlotIntegratedBiomass(fluxTraceDf,plot_dictionary['biomass_reaction'])
    
    
    # plot the environment exchange reactions
    
    PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,False,metaboliteExchangeSubset)

    PlotFluxTraces(fluxTraceDf,fluxOPT,timesOPT,False,reactionExchangeSubset,xlim,scatterSize)
    
    # if more than 1 opt solution
    if len(regulationOPT) > 1:
        PlotRegulationTraces(regulationOPT,timesOPT,fluxTraceDf.columns.tolist(),reactionExchangeSubset)
    
    
    # plot each trace individually
    fullMetabNames=metaboliteTraceDf.columns.tolist()
    for metab in fullMetabNames:
        PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,True,[metab])

    # fullRxnNames=fluxTraceDf.columns.tolist()
    # for rxn in fullRxnNames:
    #     PlotFluxTraces(fluxTraceDf,fluxOPT,timesOPT,True,[rxn],xlim,scatterSize)
    
    # if len(regulationOPT) > 1:
    #     for rxn in fullRxnNames:
    #         PlotRegulationTraces(regulationOPT,timesOPT,fluxTraceDf.columns.tolist(),[rxn],xlim,scatterSize)
    


    # loop times plot
    plt.scatter(timesOPT,loopTimes)
    plt.grid()
    plt.xlabel('Simulation Time')
    plt.ylabel('Loop Time')
    plt.show()

    cumulativeLoopTime=np.zeros(len(loopTimes))
    cumulativeLoopTime[0]=loopTimes[0]
    for i in range(1,len(loopTimes)):
        cumulativeLoopTime[i] = cumulativeLoopTime[i-1] + loopTimes[i]

    # loop times plot
    plt.plot(timesOPT,cumulativeLoopTime)
    plt.grid()
    plt.xlabel('Simulation Time')
    plt.ylabel('Total Loop Time')
    plt.show()

    return

# plotting function to consider th change in growth rate over environment concentration
def PlotGrowthRateFromPickle(file_details,plot_dictionary):

    metaboliteTraceDf,metabolitesOPT,fluxTraceDf,fluxOPT,regulationOPT,timesOPT,cell_density_vec,loopTimes = ReadFromPickle(file_details)

    cell_volume = 1e-15
    gluc_gram_mole = 180.156
    gramBiomass_to_cellcount = 5e-13

    Vmax = 3.033e-12
    Vmax = 1e-3*Vmax
    Km = 0.04*1e-3

    gluc_uptake = Vmax*metaboliteTraceDf['x1_e:ENVIRONMENT']/(Km + metaboliteTraceDf['x1_e:ENVIRONMENT'])

    plt.plot(metaboliteTraceDf['x1_e:ENVIRONMENT'],fluxTraceDf['Biomass']*cell_volume*(0.3)*gluc_gram_mole/gramBiomass_to_cellcount)
    plt.plot(metaboliteTraceDf['x1_e:ENVIRONMENT'],gluc_uptake*(0.3)*gluc_gram_mole/gramBiomass_to_cellcount)
    plt.plot(metaboliteTraceDf['x1_e:ENVIRONMENT'],.008*np.ones(len(metaboliteTraceDf['x1_e:ENVIRONMENT'])))

    plt.grid()
    plt.legend(['Prediction regulation','No regulation','Death rate'])
    plt.xlabel('X1 Environment Concentration')
    plt.ylabel('Growth rate')
    plt.ticklabel_format(style='sci')
    plt.show()


    return

# function to plot the population scale results
def PlotFromPicklePopulationModel(file_details,plot_dictionary):
    
    metaboliteTraceDf,metabolitesOPT,fluxTraceDf,fluxOPT,regulationOPT,timesOPT,cell_density_vec,loopTimes = ReadFromPickle(file_details)

    # cell density plot
    plt.plot(cell_density_vec)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Cell density')
    plt.show()
    
    xlim=0
    if 'time_limit' in plot_dictionary.keys():
        xlim=plot_dictionary['time_limit']


    scatterSize=0
    if 'scatter_size' in plot_dictionary.keys():
        scatterSize=plot_dictionary['scatter_size']

    
    metaboliteSubset=[]
    if 'metabolite_subset' in plot_dictionary.keys():
        metaboliteSubset=plot_dictionary['metabolite_subset']
        
    reactionSubset=[]
    if 'reaction_subset' in plot_dictionary.keys():
        reactionSubset=plot_dictionary['reaction_subset']
        
    regulationSubset=[]
    if 'regulation_subset' in plot_dictionary.keys():
        regulationSubset=plot_dictionary['regulation_subset']
        
    metaboliteExchangeSubset=[]
    if 'exchange_metabolite' in plot_dictionary.keys():
        metaboliteExchangeSubset=plot_dictionary['exchange_metabolite']
        
    reactionExchangeSubset=[]
    if 'exchange_reaction' in plot_dictionary.keys():
        reactionExchangeSubset=plot_dictionary['exchange_reaction']
        
    biomassMetabolite=[]
    if 'biomass_metabolite' in plot_dictionary.keys():
        biomassMetabolite=plot_dictionary['biomass_metabolite']
        
    biomassFlux=[]
    if 'biomass_reaction' in plot_dictionary.keys():
        biomassFlux=plot_dictionary['biomass_reaction']
    
    # conc plots with ODE and OPT solution
    PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,True,metaboliteSubset,xlim,scatterSize)
    
    # flux plots with ODE and OPT solution
    PlotFluxTraces(fluxTraceDf,fluxOPT,timesOPT,True,reactionSubset,xlim,scatterSize)

    if 'biomass_reaction' in plot_dictionary.keys():
        PlotIntegratedBiomass(fluxTraceDf,plot_dictionary['biomass_reaction'])
    
    
    # plot the environment exchange reactions
    
    PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,False,metaboliteExchangeSubset)

    PlotFluxTraces(fluxTraceDf,fluxOPT,timesOPT,False,reactionExchangeSubset,xlim)
    
    # if more than 1 opt solution
    if len(regulationOPT) > 1:
        PlotRegulationTraces(regulationOPT,timesOPT,fluxTraceDf.columns.tolist(),reactionExchangeSubset,xlim,scatterSize)
    
    
    # plot each trace individually
    fullMetabNames=metaboliteTraceDf.columns.tolist()
    for metab in fullMetabNames:
        PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,True,[metab])

    # fullRxnNames=fluxTraceDf.columns.tolist()
    # for rxn in fullRxnNames:
    #     PlotFluxTraces(fluxTraceDf,fluxOPT,timesOPT,True,[rxn],xlim,scatterSize)
    
    # if len(regulationOPT) > 1:
    #     for rxn in fullRxnNames:
    #         PlotRegulationTraces(regulationOPT,timesOPT,fluxTraceDf.columns.tolist(),[rxn],xlim,scatterSize)
    


    # loop times plot
    plt.scatter(timesOPT,loopTimes)
    plt.grid()
    plt.xlabel('Simulation Time')
    plt.ylabel('Loop Time')
    plt.show()

    cumulativeLoopTime=np.zeros(len(loopTimes))
    cumulativeLoopTime[0]=loopTimes[0]
    for i in range(1,len(loopTimes)):
        cumulativeLoopTime[i] = cumulativeLoopTime[i-1] + loopTimes[i]

    # loop times plot
    plt.plot(timesOPT,cumulativeLoopTime)
    plt.grid()
    plt.xlabel('Simulation Time')
    plt.ylabel('Total Loop Time')
    plt.show()

    return

# function to plot the single cell scale results
def PlotFromPickleRegulationModel(regulation_file,no_regulation_file,plot_dictionary):
    
    metaboliteTraceDfR,metabolitesOPTR,fluxTraceDfR,fluxOPTR,regulationOPTR,timesOPTR,cell_density_vecR,loopTimesR = ReadFromPickle(regulation_file)
    metaboliteTraceDfN,metabolitesOPTN,fluxTraceDfN,fluxOPTN,regulationOPTN,timesOPTN,cell_density_vecN,loopTimesN = ReadFromPickle(no_regulation_file)

    # cell density plot
    plt.plot(cell_density_vecR)
    plt.plot(cell_density_vecN)
    plt.legend(['gamma = 1','gamma = 0'])
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Cell density')
    plt.show()
    
    scatterSize=0
    if 'scatter_size' in plot_dictionary.keys():
        scatterSize=plot_dictionary['scatter_size']

    
    metaboliteSubset=[]
    if 'metabolite_subset' in plot_dictionary.keys():
        metaboliteSubset=plot_dictionary['metabolite_subset']
        
    reactionSubset=[]
    if 'reaction_subset' in plot_dictionary.keys():
        reactionSubset=plot_dictionary['reaction_subset']
        
    regulationSubset=[]
    if 'regulation_subset' in plot_dictionary.keys():
        regulationSubset=plot_dictionary['regulation_subset']
        
    metaboliteExchangeSubset=[]
    if 'exchange_metabolite' in plot_dictionary.keys():
        metaboliteExchangeSubset=plot_dictionary['exchange_metabolite']
        
    reactionExchangeSubset=[]
    if 'exchange_reaction' in plot_dictionary.keys():
        reactionExchangeSubset=plot_dictionary['exchange_reaction']
        
    biomassMetabolite=[]
    if 'biomass_metabolite' in plot_dictionary.keys():
        biomassMetabolite=plot_dictionary['biomass_metabolite']
        
    biomassFlux=[]
    if 'biomass_reaction' in plot_dictionary.keys():
        biomassFlux=plot_dictionary['biomass_reaction']


    for metab in metaboliteSubset:
        plt.plot(metaboliteTraceDfR[metab],label=metab+str(' gamma = 1'),linestyle='-')

    for metab in metaboliteSubset:
        plt.plot(metaboliteTraceDfN[metab],label=metab+str(' gamma = 0'),linestyle='--')

    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Concentration (molar)')
    plt.show()

    for rxn in fluxTraceDfR.columns.tolist():
        plt.plot(fluxTraceDfR[rxn],label=rxn+str(' gamma = 1'),linestyle='-')

    for rxn in fluxTraceDfN.columns.tolist():
        plt.plot(fluxTraceDfN[rxn],label=rxn+str(' gamma = 0'),linestyle='--')

    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Reaction flux')
    plt.show()

    
    # # conc plots with ODE and OPT solution
    # PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,True,metaboliteSubset,xlim,scatterSize)
    
    # # flux plots with ODE and OPT solution
    # PlotFluxTraces(fluxTraceDf,fluxOPT,timesOPT,True,reactionSubset,xlim,scatterSize)

    # if 'biomass_reaction' in plot_dictionary.keys():
    #     PlotIntegratedBiomass(fluxTraceDf,plot_dictionary['biomass_reaction'])
    
    
    # # plot the environment exchange reactions
    
    # PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,False,metaboliteExchangeSubset)

    # PlotFluxTraces(fluxTraceDf,fluxOPT,timesOPT,False,reactionExchangeSubset,xlim)
    
    # # if more than 1 opt solution
    # if len(regulationOPT) > 1:
    #     PlotRegulationTraces(regulationOPT,timesOPT,fluxTraceDf.columns.tolist(),reactionExchangeSubset,xlim,scatterSize)
    
    
    # # plot each trace individually
    # fullMetabNames=metaboliteTraceDf.columns.tolist()
    # for metab in fullMetabNames:
    #     PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,True,[metab])

    # # fullRxnNames=fluxTraceDf.columns.tolist()
    # # for rxn in fullRxnNames:
    # #     PlotFluxTraces(fluxTraceDf,fluxOPT,timesOPT,True,[rxn],xlim,scatterSize)
    
    # # if len(regulationOPT) > 1:
    # #     for rxn in fullRxnNames:
    # #         PlotRegulationTraces(regulationOPT,timesOPT,fluxTraceDf.columns.tolist(),[rxn],xlim,scatterSize)
    


    # # loop times plot
    # plt.scatter(timesOPT,loopTimes)
    # plt.grid()
    # plt.xlabel('Simulation Time')
    # plt.ylabel('Loop Time')
    # plt.show()

    # cumulativeLoopTime=np.zeros(len(loopTimes))
    # cumulativeLoopTime[0]=loopTimes[0]
    # for i in range(1,len(loopTimes)):
    #     cumulativeLoopTime[i] = cumulativeLoopTime[i-1] + loopTimes[i]

    # # loop times plot
    # plt.plot(timesOPT,cumulativeLoopTime)
    # plt.grid()
    # plt.xlabel('Simulation Time')
    # plt.ylabel('Total Loop Time')
    # plt.show()

    return

# function to plot the scaling experiments 
def PlotScaling(plot_dictionary,pop1,pop2,pop3,pop4,pop5,pop8,pop10):

    metaboliteTraceDf1,metabolitesOPT1,fluxTraceDf1,fluxOPT1,regulationOPT1,timesOPT1,cell_density_vec1,loopTimes1 = ReadFromPickle(pop1)
    metaboliteTraceDf2,metabolitesOPT2,fluxTraceDf2,fluxOPT2,regulationOPT2,timesOPT2,cell_density_vec2,loopTimes2 = ReadFromPickle(pop2)
    metaboliteTraceDf3,metabolitesOPT3,fluxTraceDf3,fluxOPT3,regulationOPT3,timesOPT3,cell_density_vec3,loopTimes3 = ReadFromPickle(pop3)
    metaboliteTraceDf4,metabolitesOPT4,fluxTraceDf4,fluxOPT4,regulationOPT4,timesOPT4,cell_density_vec4,loopTimes4 = ReadFromPickle(pop4)
    metaboliteTraceDf5,metabolitesOPT5,fluxTraceDf5,fluxOPT5,regulationOPT5,timesOPT5,cell_density_vec5,loopTimes5 = ReadFromPickle(pop5)
    metaboliteTraceDf8,metabolitesOPT8,fluxTraceDf8,fluxOPT8,regulationOPT8,timesOPT8,cell_density_vec8,loopTimes8 = ReadFromPickle(pop8)
    metaboliteTraceDf10,metabolitesOPT10,fluxTraceDf10,fluxOPT10,regulationOPT10,timesOPT10,cell_density_vec10,loopTimes10 = ReadFromPickle(pop10)

    optimal_biomass_fluxes =[]

    biomassFlux=[]
    if 'biomass_reaction' in plot_dictionary.keys():
        biomassFlux=plot_dictionary['biomass_reaction']
        optimal_biomass_fluxes.append(fluxOPT1[0].loc[biomassFlux]['Optimal Flux'])
        optimal_biomass_fluxes.append(fluxOPT2[0].loc[biomassFlux]['Optimal Flux'])
        optimal_biomass_fluxes.append(fluxOPT3[0].loc[biomassFlux]['Optimal Flux'])
        optimal_biomass_fluxes.append(fluxOPT4[0].loc[biomassFlux]['Optimal Flux'])
        optimal_biomass_fluxes.append(fluxOPT5[0].loc[biomassFlux]['Optimal Flux'])
        optimal_biomass_fluxes.append(fluxOPT8[0].loc[biomassFlux]['Optimal Flux'])
        optimal_biomass_fluxes.append(fluxOPT10[0].loc[biomassFlux]['Optimal Flux'])

    

    # loop times plot
    plt.scatter(timesOPT1,loopTimes1, label='1')
    plt.scatter(timesOPT2,loopTimes2, label='2')
    plt.scatter(timesOPT3,loopTimes3, label='3')
    plt.scatter(timesOPT4,loopTimes4, label='4')
    plt.scatter(timesOPT5,loopTimes5, label='5')
    plt.scatter(timesOPT8,loopTimes8, label='8')
    plt.scatter(timesOPT10,loopTimes10, label='10')
    plt.grid()
    plt.legend()
    plt.xlabel('Simulation Time')
    plt.ylabel('Loop Time')
    plt.show()

    plt.violinplot([loopTimes1,loopTimes2,loopTimes4,loopTimes5,loopTimes8,loopTimes10],showmeans=True,showmedians=True)

    # plt.boxplot(loopTimes2)
    # plt.boxplot(loopTimes3)
    # plt.boxplot(loopTimes4)
    # plt.boxplot(loopTimes5)
    plt.grid()
    plt.legend()
    plt.xlabel('Number of Workers')
    plt.ylabel('Loop Time')
    # plt.ylim([0,500])
    plt.yscale('log')
    plt.show()


    cumulativeLoopTime1=np.zeros(len(loopTimes1))
    cumulativeLoopTime2=np.zeros(len(loopTimes2))
    cumulativeLoopTime3=np.zeros(len(loopTimes3))
    cumulativeLoopTime4=np.zeros(len(loopTimes4))
    cumulativeLoopTime5=np.zeros(len(loopTimes5))
    cumulativeLoopTime8=np.zeros(len(loopTimes8))
    cumulativeLoopTime10=np.zeros(len(loopTimes10))
    cumulativeLoopTime1[0]=loopTimes1[0]
    cumulativeLoopTime2[0]=loopTimes2[0]
    cumulativeLoopTime3[0]=loopTimes3[0]
    cumulativeLoopTime4[0]=loopTimes4[0]
    cumulativeLoopTime5[0]=loopTimes5[0]
    cumulativeLoopTime8[0]=loopTimes8[0]
    cumulativeLoopTime10[0]=loopTimes10[0]
    for i in range(1,len(loopTimes1)):
        cumulativeLoopTime1[i] = cumulativeLoopTime1[i-1] + loopTimes1[i]
    for i in range(1,len(loopTimes2)):
        cumulativeLoopTime2[i] = cumulativeLoopTime2[i-1] + loopTimes2[i]
    for i in range(1,len(loopTimes3)):
        cumulativeLoopTime3[i] = cumulativeLoopTime3[i-1] + loopTimes3[i]
    for i in range(1,len(loopTimes4)):
        cumulativeLoopTime4[i] = cumulativeLoopTime4[i-1] + loopTimes4[i]
    for i in range(1,len(loopTimes5)):
        cumulativeLoopTime5[i] = cumulativeLoopTime5[i-1] + loopTimes5[i]
    for i in range(1,len(loopTimes8)):
        cumulativeLoopTime8[i] = cumulativeLoopTime8[i-1] + loopTimes8[i]
    for i in range(1,len(loopTimes10)):
        cumulativeLoopTime10[i] = cumulativeLoopTime10[i-1] + loopTimes10[i]

    # loop times plot
    plt.plot(timesOPT1,cumulativeLoopTime1, label='1')
    plt.plot(timesOPT2,cumulativeLoopTime2, label='2')
    plt.plot(timesOPT3,cumulativeLoopTime3, label='3')
    plt.plot(timesOPT4,cumulativeLoopTime4, label='4')
    plt.plot(timesOPT5,cumulativeLoopTime5, label='5')
    plt.plot(timesOPT8,cumulativeLoopTime8, label='8')
    plt.plot(timesOPT10,cumulativeLoopTime10, label='10')
    plt.grid()
    plt.legend()
    plt.xlabel('Simulation Time')
    plt.ylabel('Total Loop Time')

    plt.show()

    plt.scatter([1,2,3,4,5,8,10],[cumulativeLoopTime1[len(timesOPT1)-1],cumulativeLoopTime2[len(timesOPT2)-1],cumulativeLoopTime3[len(timesOPT3)-1],cumulativeLoopTime4[len(timesOPT4)-1],cumulativeLoopTime5[len(timesOPT5)-1],cumulativeLoopTime8[len(timesOPT8)-1],cumulativeLoopTime10[len(timesOPT10)-1]])
    plt.grid()
    plt.xlabel('Worker count')
    plt.ylabel('Total Loop Time')

    plt.show()

    worker_counts = [1,2,3,4,5,8,10]
    mean_times = [np.mean(loopTimes1),np.mean(loopTimes2),np.mean(loopTimes3),np.mean(loopTimes4),np.mean(loopTimes5),np.mean(loopTimes8),np.mean(loopTimes10)]
    var_times = [np.var(loopTimes1),np.var(loopTimes2),np.var(loopTimes3),np.var(loopTimes4),np.var(loopTimes5),np.var(loopTimes8),np.var(loopTimes10)]

    std_devs = np.sqrt(var_times)

    plt.errorbar(worker_counts,mean_times,yerr=std_devs,fmt='o',capsize=10,capthick=2)

    # plt.boxplot(loopTimes2)
    # plt.boxplot(loopTimes3)
    # plt.boxplot(loopTimes4)
    # plt.boxplot(loopTimes5)
    plt.grid()
    plt.legend()
    plt.xlabel('Number of Workers')
    plt.ylabel('Mean Loop Time')
    # plt.ylim([0,500])
    plt.yscale('log')
    plt.show()

    

    plt.scatter(worker_counts,[cumulativeLoopTime1[len(timesOPT1)-1],cumulativeLoopTime2[len(timesOPT2)-1],cumulativeLoopTime3[len(timesOPT3)-1],cumulativeLoopTime4[len(timesOPT4)-1],cumulativeLoopTime5[len(timesOPT5)-1],cumulativeLoopTime8[len(timesOPT8)-1],cumulativeLoopTime10[len(timesOPT10)-1]],label='Simulation time')
    # plt.scatter(worker_counts,[x*len(loopTimes10) for x in mean_times],label='Expected time')
    
    plt.grid()
    plt.xlabel('Worker count')
    plt.ylabel('Total Loop Time')
    # plt.legend()

    plt.show()

    plt.scatter(worker_counts,[x for x in mean_times],label='Expected time')
    
    plt.grid()
    plt.xlabel('Worker count')
    plt.ylabel('Total Loop Time')
    # plt.legend()

    plt.show()


    
    plt.scatter(worker_counts,optimal_biomass_fluxes,label='Expected time')
    
    plt.grid()
    plt.xlabel('Worker count')
    plt.ylabel('Objective Biomass Flux')
    # plt.legend()

    plt.show()


    # Calculate means and standard deviations
    # worker_counts = [1,2,3,4,5,8,10]
    # means = [np.mean(loopTimes1),np.mean(loopTimes2),np.mean(loopTimes3),np.mean(loopTimes4),np.mean(loopTimes5),np.mean(loopTimes8),np.mean(loopTimes10)]
    # var_times = [np.var(loopTimes1),np.var(loopTimes2),np.var(loopTimes3),np.var(loopTimes4),np.var(loopTimes5),np.var(loopTimes8),np.var(loopTimes10)]
    worker_counts = [1,2,3,5,8,10]
    means = [np.mean(loopTimes1),np.mean(loopTimes2),np.mean(loopTimes3),np.mean(loopTimes5),np.mean(loopTimes8),np.mean(loopTimes10)]
    var_times = [np.var(loopTimes1),np.var(loopTimes2),np.var(loopTimes3),np.var(loopTimes5),np.var(loopTimes8),np.var(loopTimes10)]

    std_devs = np.sqrt(var_times)

    plt.figure()

    # Error Bars
    plt.errorbar(worker_counts, means, yerr=std_devs, fmt='o', capsize=5, capthick=2, ecolor='orange', label='Mean with Error Bars')
    plt.xlabel('Worker count')
    plt.ylabel('Loop Time')

    plt.grid(True)


    # Display the plots
    plt.tight_layout()
    plt.show()

    plt.figure()
   
    # Error Bars
    plt.errorbar(worker_counts, means, yerr=std_devs, fmt='o', capsize=5, capthick=2, ecolor='orange', label='Mean with Error Bars')
    plt.xlabel('Worker count')
    plt.ylabel('Loop Time')

    plt.grid(True)


    # Display the plots
    plt.tight_layout()
    plt.yscale('log')
    plt.show()


    optimal_biomass_fluxes =[]

    biomassFlux=[]
    if 'biomass_reaction' in plot_dictionary.keys():
        biomassFlux=plot_dictionary['biomass_reaction']
        optimal_biomass_fluxes.append(fluxOPT1[0].loc[biomassFlux]['Optimal Flux'])
        optimal_biomass_fluxes.append(fluxOPT2[0].loc[biomassFlux]['Optimal Flux'])
        optimal_biomass_fluxes.append(fluxOPT3[0].loc[biomassFlux]['Optimal Flux'])
        optimal_biomass_fluxes.append(fluxOPT5[0].loc[biomassFlux]['Optimal Flux'])
        optimal_biomass_fluxes.append(fluxOPT8[0].loc[biomassFlux]['Optimal Flux'])
        optimal_biomass_fluxes.append(fluxOPT10[0].loc[biomassFlux]['Optimal Flux'])


    # Visualize the time series data
    plt.figure()
    plt.plot(worker_counts, means, label='Loop Time',color='orange')
    plt.plot(worker_counts, optimal_biomass_fluxes, label='Objective Biomass Flux', color='blue')
    plt.xlabel('Worker Count')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

    # Calculate the correlation
    correlation, p_value = pearsonr(means, optimal_biomass_fluxes)
    print(f'Pearson correlation coefficient: {correlation:.3f}')
    print(f'p-value: {p_value:.3f}')

    # Visualize the correlation with a scatter plot
    plt.figure()
    plt.scatter(means, optimal_biomass_fluxes, alpha=0.7)
    plt.xlabel('Loop Time')
    plt.ylabel('Objective Biomass Flux')
    plt.show()


    return 

# function to show the effect of changing gamma for the regulation of the cell
def PlotGammaChange(cellRegulation,cellNoRegulation,cellRegulation05,cellRegulation100,cellRegulation1000):

    metaboliteTraceDf1,metabolitesOPT1,fluxTraceDf1,fluxOPT1,regulationOPT1,timesOPT1,cell_density_vec1,loopTimes1 = ReadFromPickle(cellRegulation)
    metaboliteTraceDf2,metabolitesOPT2,fluxTraceDf2,fluxOPT2,regulationOPT2,timesOPT2,cell_density_vec2,loopTimes2 = ReadFromPickle(cellNoRegulation)
    metaboliteTraceDf3,metabolitesOPT3,fluxTraceDf3,fluxOPT3,regulationOPT3,timesOPT3,cell_density_vec3,loopTimes3 = ReadFromPickle(cellRegulation05)
    metaboliteTraceDf4,metabolitesOPT4,fluxTraceDf4,fluxOPT4,regulationOPT4,timesOPT4,cell_density_vec4,loopTimes4 = ReadFromPickle(cellRegulation100)
    metaboliteTraceDf5,metabolitesOPT5,fluxTraceDf5,fluxOPT5,regulationOPT5,timesOPT5,cell_density_vec5,loopTimes5 = ReadFromPickle(cellRegulation1000)

    metab ='x2:CYTOSOL'
    plt.plot(metaboliteTraceDf1[metab],label='gamma = inf')
    plt.plot(metaboliteTraceDf2[metab],label='gamma = 0')
    plt.plot(metaboliteTraceDf3[metab],label='gamma = 0.5')
    plt.plot(metaboliteTraceDf4[metab],label='gamma = 100')
    plt.plot(metaboliteTraceDf5[metab],label='gamma = 1000')
    plt.grid()
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Conc')
    plt.show()    

    rxn ='Uptake'
    plt.plot(fluxTraceDf1[rxn],label='gamma = inf')
    plt.plot(fluxTraceDf2[rxn],label='gamma = 0')
    plt.plot(fluxTraceDf3[rxn],label='gamma = 0.5')
    plt.plot(fluxTraceDf4[rxn],label='gamma = 100')
    plt.plot(fluxTraceDf5[rxn],label='gamma = 1000')
    plt.grid()
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.show()    

    return

# function to remove the ThermoOpt output files from the model_file_dir and solutions solution_file_dir
def CleanThermoOpt(control_dictionary):
    model_file_dir = control_dictionary['model_file_dir']
    solution_file_dir = control_dictionary['solution_file_dir']
    
    for file in os.listdir(model_file_dir):
        if re.search('output_ChemostatModel_', file):
            os.remove(os.path.join(model_file_dir, file))
    for file in os.listdir(solution_file_dir):
        if re.search('output_ChemostatModel_', file):
            os.remove(os.path.join(solution_file_dir, file))
    
    return

# set up the initial dataframes for the environment, chemostat, and the cell contribution (to the environment)
# all metabolite values are in terms of mili-molar concentrations
def SetUpEnvironmentCellDataFrames(environment_dictionary, ode_system_dictionary, ifVerbose=False):


    ode_system_dictionary['max_time'] = ode_system_dictionary['nmax']*ode_system_dictionary['dt']
    
    timesteps = np.arange(0,ode_system_dictionary['max_time'],ode_system_dictionary['dt'])
    
    ode_system_dictionary['timesteps'] = timesteps

    chemostatDf = SetupEnvironmentChemostat(environment_dictionary,ode_system_dictionary,ifVerbose)

    environmentMetabolites = environment_dictionary['metabolites']

    environmentDf = pd.DataFrame()
    environmentDf['time']=timesteps
    for em in environmentMetabolites:
        environmentDf[em]=np.zeros(len(timesteps))
    environmentDf=environmentDf.set_index('time')

    cellContributionDf = pd.DataFrame()
    cellContributionDf['time']=timesteps
    for em in environmentMetabolites:
        cellContributionDf[em]=np.zeros(len(timesteps))
    cellContributionDf=cellContributionDf.set_index('time')

    for i in range(len(environmentMetabolites)):
        environmentDf.at[0,environmentMetabolites[i]] = environment_dictionary['initial_environment_concentrations'][i]


    if ifVerbose:
        print('chemostatDf')
        display(chemostatDf.head(5))
        print()
        print('environmentDf')
        display(environmentDf.head(5))
        print()
        print('cellContributionDf')
        display(cellContributionDf.head(5))

    
    return chemostatDf,environmentDf,cellContributionDf


def SetupEnvironmentChemostat(environment_dictionary,ode_system_dictionary,ifPlot=False):

    # specify whether we can 'chemostat' the environment to a fixed value, provide a method to implement
    # dirichlet BC, currently only have neumann BCs (of which we use zero)
    
    timesteps= ode_system_dictionary['timesteps']
    chemostatDf = pd.DataFrame()
    chemostatDf['time']=ode_system_dictionary['timesteps']
    
    numberOfMetabolites = len(environment_dictionary['metabolites'])

    if numberOfMetabolites != len(environment_dictionary['chemostat_type']):
        
        print('numberOfMetabolites != len(environment_dictionary[\'chemostat_type\'])')
        print('Error: Need to know the trace chemostat_type for each metabolites')
        return chemostatDf
    
    for i in range(numberOfMetabolites):

        if environment_dictionary['chemostat_type'][i][0] == 'Zero':
            # U(t) = 0
            chemostatDf[environment_dictionary['metabolites'][i]]=np.zeros_like(timesteps)
        elif environment_dictionary['chemostat_type'][i][0] == 'Dirichlet' or environment_dictionary['chemostat_type'][i][0] == 'Fixed':
            # U(t) = a
            chemostatDf[environment_dictionary['metabolites'][i]]=float(environment_dictionary['chemostat_type'][i][1]['Magnitude'])*np.ones_like(timesteps)

        elif environment_dictionary['chemostat_type'][i][0] == 'Neumann' or environment_dictionary['chemostat_type'][i][0] == 'Constant_Flux':
            # dU/dt(t) = a
            chemostatDf[environment_dictionary['metabolites'][i]]=np.zeros_like(timesteps)

            for j in range(len(timesteps)):
                chemostatDf.loc[j,environment_dictionary['metabolites'][i]] = float(environment_dictionary['chemostat_type'][i][1]['Magnitude'])*j*ode_system_dictionary['dt']
        
        elif environment_dictionary['chemostat_type'][i][0] == 'Sine':
            chemostatDf[environment_dictionary['metabolites'][i]]=np.zeros_like(timesteps)

        elif environment_dictionary['chemostat_type'][i][0] == 'Cosine':
            chemostatDf[environment_dictionary['metabolites'][i]]=np.zeros_like(timesteps)


    chemostatDf=chemostatDf.set_index('time')
    
    if ifPlot:
        
        for i in range(numberOfMetabolites):
            plt.plot(timesteps, chemostatDf[environment_dictionary['metabolites'][i]], marker = "o", label=environment_dictionary['metabolites'][i].split(':')[0],markersize=1)

        plt.title("Environment Boundary Conditions")
        plt.xlabel("Timestep")
        plt.ylabel("Metabolite Concentration")
        plt.legend()
        plt.show()
    
    return chemostatDf

# determine scaling coefficients, such as cell volume and the volume of the near cell element of the media
def SetUpScaling(cell_dictionary,environment_dictionary,ode_system_dictionary,ifVerbose=False):
    
    # Adjustable parameters:
    Vmax = 3.033e-12 # mM glucose consumed per hour per 1 micron^3 cell 
    Vmax = 1e-3*Vmax #convert to Molar

    VolCell = cell_dictionary['cell_volume']
    cell_dictionary['vmax'] = Vmax/VolCell #convert to M per L per Hour change in internal cell concentration by uptake

    r = np.sqrt(6*ode_system_dictionary['dt']*environment_dictionary['glucose_diffusion_constant']) # outer radius of sphere diffused into in 3-dimensions
    volumeElement = (4/3)*np.pi*pow(r,3)
    volumeChemostatToCell=volumeElement/VolCell
    SurfaceAreaCell = pow(np.pi,1/3)*pow(6*VolCell,2/3)
    Concentration2Count = cell_dictionary['N_avogadro'] * VolCell

    cell_dictionary['concentration_2_count'] = Concentration2Count
    cell_dictionary['cell_surface_area'] = SurfaceAreaCell
    cell_dictionary['cell_chemostat_element'] = volumeElement
    cell_dictionary['volume_chemostat_to_cell'] = volumeChemostatToCell

    if ifVerbose:
        print('Volume of cell: '+str(VolCell))
        print('Surface area of cell: '+str(SurfaceAreaCell))
        print('Volume of near field element: '+str(volumeElement))
        print('Volume ratio of Chemostat To Cell: '+str(volumeChemostatToCell))
    
    return cell_dictionary


def save_ode_model(solutionFile,cell_dictionary,odeConcentrations,odeFluxes):
    
    ifOptParms = False

    v_ref =cell_dictionary['v_reference']
    
    try:
        # if opt_parms is saved in the model solution
        y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound,opt_parms = baf.load_model_solution_opts(solutionFile)
        ifOptParms = True

    except:
        # if opt_parms is not saved in the model solution 
        y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound = baf.load_model_solution(solutionFile)
    
    # modify the solution file by the ODE solution
    metab_log_concs=np.zeros(len(odeConcentrations))
    
    T = 298.15
    R = 8.314e-03
    RT = R*T
    N_avogadro = 6.022140857e+23
    VolCell = 1.0e-15
    Concentration2Count = N_avogadro * VolCell
    
    for i in range(len(metab_log_concs)):
        # metab_log_concs[i] = np.log((odeConcentrations[i]/1000)*Concentration2Count)
        metab_log_concs[i] = np.log((odeConcentrations[i])*Concentration2Count)
        
    new_n_sol = metab_log_concs[0:len(n_sol)]
    new_f_log_counts = metab_log_concs[len(n_sol):]
    # convert the concentrations back into log counts for saving
    
    new_y_sol = np.zeros(len(y_sol))
    
    for i in range(len(y_sol)):
        new_y_sol[i] = odeFluxes[i]/v_ref
    
 
    if ifOptParms:
        baf.save_model_solution(solutionFile,new_y_sol,alpha_sol,new_n_sol,unreg_rxn_flux,S,Keq,new_f_log_counts,vcount_lower_bound,vcount_upper_bound,opt_parms)
    else:
        baf.save_model_solution(solutionFile,new_y_sol,alpha_sol,new_n_sol,unreg_rxn_flux,S,Keq,new_f_log_counts,vcount_lower_bound,vcount_upper_bound)
    
    return
 

# update the opt boundary conditions to reflect the change in the environment due to the chemostat 
# or the contribution from the cell. All concentrations in mili-mole
def SetModelBoundary(i,environment_dictionary,cell_dictionary,ode_system_dictionary,environmentDf,chemostatDf,cellContributionDf,metaboliteTraceDf):
    

    isChemostated = False

    timesteps=ode_system_dictionary['timesteps']
    dt=ode_system_dictionary['dt']
    environmentMetabolites=environment_dictionary['metabolites']
    t=timesteps[i]
    modelBoundaryConditions={}
    if i >0:
        for em in environmentMetabolites:

            if environment_dictionary['chemostat_type'][environmentMetabolites.index(em)][0] == 'Dirichlet' or environment_dictionary['chemostat_type'][environmentMetabolites.index(em)][0] == 'Fixed':
                environmentDf.at[t,em] = chemostatDf.loc[t,em]

            else:
                # add chemostat
                environmentDf.at[t,em] = metaboliteTraceDf.at[timesteps[i-1],em]   

                if environment_dictionary['chemostat_type'][environmentMetabolites.index(em)][0] == 'Neumann' or environment_dictionary['chemostat_type'][environmentMetabolites.index(em)][0] == 'Constant_Flux':
                    environmentDf.at[t,em] += chemostatDf.loc[timesteps[i-1],em] *dt
                
                # cell contribution is an absolute amount in dt
                # environmentDf.at[t,em] += cellContributionDf.loc[timesteps[i-1],em] #/ cell_dictionary['volume_chemostat_to_cell']

            modelBoundaryConditions[em]=environmentDf.loc[t,em]

    else:
        for em in environmentMetabolites:
            modelBoundaryConditions[em]=environmentDf.loc[t,em]

    return modelBoundaryConditions

def SetInitialModelBoundary(i,environment_dictionary,cell_dictionary,ode_system_dictionary,environmentDf,chemostatDf,cellContributionDf):
    
    isChemostated = False

    timesteps=ode_system_dictionary['timesteps']
    dt=ode_system_dictionary['dt']
    environmentMetabolites=environment_dictionary['metabolites']
    t=timesteps[i]
    modelBoundaryConditions={}

    for em in environmentMetabolites:
        modelBoundaryConditions[em]=environmentDf.loc[t,em]

    return modelBoundaryConditions

def SetJumpModelBoundary(i,environment_dictionary,cell_dictionary,ode_system_dictionary,environmentDf,chemostatDf,cellContributionDf,metaboliteTraceDf):
    

    isChemostated = False

    timesteps=ode_system_dictionary['timesteps']
    dt=ode_system_dictionary['dt']
    environmentMetabolites=environment_dictionary['metabolites']
    t=timesteps[i]
    modelBoundaryConditions={}
    if i >0:
        for em in environmentMetabolites:

            if environment_dictionary['chemostat_type'][environmentMetabolites.index(em)][0] == 'Dirichlet' or environment_dictionary['chemostat_type'][environmentMetabolites.index(em)][0] == 'Fixed':
                environmentDf.at[t,em] = chemostatDf.loc[t,em]

            else:
                # add chemostat
                environmentDf.at[t,em] = metaboliteTraceDf.at[timesteps[i-1],em]   

            randChange = nprd.uniform(0.0, 1e-20 )
            direction = nprd.choice([-1,1])

            proposedPerturbation = environmentDf.loc[t,em] + direction*randChange


            if proposedPerturbation <0:
                modelBoundaryConditions[em]=environmentDf.loc[t,em]
            else:
                modelBoundaryConditions[em]= proposedPerturbation

    else:
        for em in environmentMetabolites:
            modelBoundaryConditions[em]=environmentDf.loc[t,em]

    return modelBoundaryConditions



def ComposeBaseSimulation(modelBoundaryConditions,t,cell_dictionary):
    
    # need to convert boundary conditions from concentration to log counts

    base_simulation_list=[]
    simulation_test={
        'name':'ChemostatModel',
        'model':cell_dictionary['model'],
        'model_output':'output_ChemostatModel',
        'initial_conditions':{}, # dictionary'metabolite name': val where val \in positive reals
        'boundary_conditions':modelBoundaryConditions, # dictionary'metabolite name': val where val \in positive reals
        'obj_coefs':{}, # dictionary'reaction name': val where val \in reals
        'implicit_metabolites':[], # list 'metabolite name', default for Fixed species
        'explicit_metabolites':[], # list 'metabolite name', default for Variable species
        'net_flux':{}, # dictionary 'metabolite name': val where val \in {1, 0, -1}, default 0 
        'flux_direction':{}, # dictionary 'reaction_name': val where val \in {1, 0, -1}, default 0 
        'allow_regulation':{} # dictionary 'reaction_name': val where val \in {1, 0}, default 1 
    }
    base_simulation_list.append(simulation_test)

    for parentSim in base_simulation_list:
        if parentSim['name'][-1] == "_":
            parentSim['name'] = parentSim['name']+"t_"+str(t)+"_"
        else:
            parentSim['name'] = parentSim['name']+"_"+"t_"+str(t)+"_"
        if parentSim['model_output'][-1] == "_":
            parentSim['model_output'] = parentSim['model_output']+"t_"+str(t)+"_"
        else:
            parentSim['model_output'] = parentSim['model_output']+"_"+"t_"+str(t)+"_"

        parentSim['model_output'] = parentSim['model_output'].replace(":","_")  
        parentSim['model_output'] = parentSim['model_output'].replace(" ","_") 

    for sim in base_simulation_list:
        sim['use_previous_simulation'] = cell_dictionary['previous_best_model_solution']

    return base_simulation_list

# post OPT simulation, parse the output and convert the metabolite log counts into concentrations
def ParseBestSimulations(bestSimulations,control_dictionary,cell_dictionary,metaboliteSubset=[],reactionSubset=[],verbose=False):


    Concentration2Count = cell_dictionary['concentration_2_count']

    metabSolsdf_vector=[]
    ysoldf_vector=[]
    S_active_vector=[]
    active_reactions_vector=[]
    name_vector=[]

    if len(bestSimulations[0]['bestSimulations']) >0:
        for simName in bestSimulations[0]['bestSimulations']:
            
            name_vector.append(simName)
            file=str(bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name']+'.pkl')
            file = file.replace(":",'_')
            
            y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound = baf.load_model_solution(control_dictionary['solution_file_dir']+"/output_"+file)

            S_active_best, active_reactions_best, metabolites_best, Keq_best = baf.load_model(control_dictionary['model_file_dir']+"/output_"+file)


            S_active_vector.append(S_active_best)
            active_reactions_vector.append(active_reactions_best)

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

            
            if verbose:
                print(simName)
                anf.PlotMetaboliteSolutions(metabSolsdf)
                anf.PlotFluxSolutions(ysoldf)


    # assume only one model was run for a set of swarms
    return metabSolsdf_vector[0],ysoldf_vector[0],S_active_vector[0],active_reactions_vector[0]

# for the first iteration only, use the output concentrations of the OPT solution as the initial
# concentrations of the ODE system with a random perturbation, set up solution dataframes
def SetupODESystem(ode_dictionary,environment_dictionary,initial_concs,fluxDf):
    

    ode_dictionary['max_time'] = ode_dictionary['nmax']*ode_dictionary['dt']
    ode_dictionary['timesteps'] = np.arange(0,ode_dictionary['max_time'],ode_dictionary['dt']).tolist()
    names_list=initial_concs.index.tolist()
    concs_list=initial_concs['metabolite_conc'].tolist()
    
    if 'initial_concentration_perturbation'in ode_dictionary.keys():

        for i in range(len(concs_list)):
            
            if names_list[i] not in environment_dictionary['metabolites']:

                randChange = nprd.uniform(0.0, ode_dictionary['initial_concentration_perturbation'] )
                direction = nprd.choice([-1,1])
                concs_list[i] += direction*randChange

    ode_dictionary['metabolite_names']=names_list
    ode_dictionary['initial_concentrations']=concs_list
    
    ode_dictionary['metabolite_concentration_maximum']=initial_concs['vconc_upper_bound'].tolist()
    ode_dictionary['metabolite_concentration_minimum']=initial_concs['vconc_lower_bound'].tolist()


    metaboliteTraceDf = pd.DataFrame()
    metaboliteTraceDf['Time']=ode_dictionary['timesteps']
    for em in names_list:
        metaboliteTraceDf[em]=np.zeros(len(ode_dictionary['timesteps']))

    metaboliteTraceDf = metaboliteTraceDf.set_index('Time')


    fluxTraceDf = pd.DataFrame()
    fluxTraceDf['Time']=ode_dictionary['timesteps']
    for fl in fluxDf.index:
        fluxTraceDf[fl]=np.zeros(len(ode_dictionary['timesteps']))
    fluxTraceDf = fluxTraceDf.set_index('Time')
    
    simKineticsDf = pd.DataFrame()
    simKineticsDf['Time']=ode_dictionary['timesteps']
    simKineticsDf = simKineticsDf.set_index('Time')
    for i in range(len(fluxDf.index)):

        simKineticsDf['kf_apparent_'+str(fluxDf.index[i])] = np.zeros(len(ode_dictionary['timesteps']))
        simKineticsDf['kr_apparent_'+str(fluxDf.index[i])] = np.zeros(len(ode_dictionary['timesteps']))

    # set up apparent rate containers
    k_app_f = np.zeros(len(fluxDf.index))
    k_app_r = np.zeros(len(fluxDf.index))

    sim_concs = ode_dictionary['initial_concentrations']

    return metaboliteTraceDf,fluxTraceDf,simKineticsDf,k_app_f,k_app_r, sim_concs

# for the best OPT simulation, read the solution and convert the log counts into mili-molar concentrations 
def ReadOPTSolution(bestSimulations,control_dictionary,cell_dictionary):

    # convert results to concentrations (from log_counts)

    name_vector=[]
    for sim in bestSimulations[0]['bestSimulations']:
        name_vector.append(sim)
        
    simName=name_vector[0]
    file=str(bestSimulations[0]['bestSimulations'][simName]['swarm_instance_name']+'.pkl')
    file = file.replace(":",'_')

    y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound,opt_parms = baf.load_model_solution_opts(control_dictionary['solution_file_dir']+"/output_"+file)

    metabolite_labels = opt_parms['metabolite_labels']
    reaction_labels = opt_parms['reaction_labels']
    metab_log_concs=np.concatenate((n_sol,f_log_counts))


    Concentration2Count = cell_dictionary['concentration_2_count']
    metab_concs=np.zeros(len(metab_log_concs))
    y_hat = np.zeros(len(y_sol))

    for i in range(len(metab_log_concs)):
        metab_concs[i] = (np.exp(metab_log_concs[i])/Concentration2Count)

    y_hat = y_sol.copy()

    return y_hat,alpha_sol,metab_concs,S,Keq,metabolite_labels,reaction_labels

# function to determine the kinetic values for the metabolic steady state by processing OPT solution 
def InitializeKineticConversion(t_opt,v_hat,S,Keq,n_hat,cell_dictionary,environment_dictionary,ode_system_dictionary,metabolite_labels,reaction_labels):
    
    
    DetermineCorrectEnergyImport(t_opt,cell_dictionary,metabolite_labels,reaction_labels,v_hat,n_hat,S)

    for i in range(len(reaction_labels)):

        if reaction_labels[i] == cell_dictionary['energy_import_reaction_name']:
            uptakeRate = v_hat[i] # y_sol from sim output
            break
    
    for i in range(len(metabolite_labels)):

        if metabolite_labels[i] == cell_dictionary['environment_energy_name']:
            energy_conc = n_hat[i]
            break

    vref_scalar,volumeElement = CalculateScaling(uptakeRate,energy_conc,cell_dictionary,environment_dictionary,ode_system_dictionary)

    if 'vref_override' in ode_system_dictionary.keys():

        vref_scalar = ode_system_dictionary['vref_override']
    else:
        print('vref_scalar')
        print(vref_scalar)

    cell_dictionary['v_reference']=vref_scalar

    return cell_dictionary

# function to determine the kinetic values for the metabolic steady state by processing OPT solution 
def DetermineOptimalKinetics(t_opt,v_hat,S,Keq,n_hat,cell_dictionary,environment_dictionary,ode_system_dictionary,metabolite_labels,reaction_labels):

    vref_scalar = cell_dictionary['v_reference']

    kf,kr,Keq = CalculateRateConstants(vref_scalar,v_hat,S,Keq,n_hat)

    kineticsDf = pd.DataFrame()
    kineticsDf['Reaction']=reaction_labels
    kineticsDf['Optimal Flux']=v_hat*vref_scalar
    kineticsDf['kf']=kf
    kineticsDf['kr']=kr
    kineticsDf['Keq']=Keq
    
    kineticsDf = kineticsDf.set_index('Reaction')

    return kineticsDf

# from the list of import metabolites we need to identify which metabolite is the glucose
# candidate such that we can use the uptake rate to determine the scaling factor
def DetermineCorrectEnergyImport(t_opt,cell_dictionary,metabolite_labels,reaction_labels,v_hat,n_hat,S):

    Vmax = cell_dictionary['vmax']

    previous_import_metab = cell_dictionary['environment_energy_name']

    # for each import reaction and metabolite, determine which corresponds to the largest energy uptake
    V=0
    largest_import_metabolite=""
    largest_import_reaction=""
    for i in range(len(cell_dictionary['import_energy_metabolites'])):
        for j in range(len(metabolite_labels)):
            
            # dictionary elements may not be in the same order as the simulation output
            if cell_dictionary['import_energy_metabolites'][i] == metabolite_labels[j]:
                isImported=False
                metab_concentration = n_hat[j]
                import_reaction_flux=0
                import_stoich=0
                for rid in range(len(v_hat)):
                    if reaction_labels[rid] == cell_dictionary['import_energy_reactions'][i]:
                        import_reaction_flux = v_hat[rid]
                        import_stoich = S[rid,j]
                        break

                # check whether the import metabolite is actually being imported
                if import_stoich < 0 and import_reaction_flux > 0:
                    # import metabolite is a substrate and net flux is in forward direction
                    isImported=True

                elif import_stoich > 0 and import_reaction_flux < 0:
                    # import metabolite is a product but the net flux is in the reverse direction
                    isImported=True


                if isImported:
                    energy_conc = n_hat[j]
                    Km = cell_dictionary['import_energy_michaelis_menten'][i]
                    Vnew = Vmax * energy_conc/(Km + energy_conc)

                    if V<=Vnew:
                        V=Vnew
                        largest_import_metabolite = cell_dictionary['import_energy_metabolites'][i]
                        largest_import_reaction   = cell_dictionary['import_energy_reactions'][i]

                break
    
    if largest_import_metabolite=="":
        print('Error: no import metabolite found in DetermineCorrectEnergyImport')

    cell_dictionary['environment_energy_name']= largest_import_metabolite
    cell_dictionary['energy_import_reaction_name']=largest_import_reaction


    return


def CalculateScaling(uptakeRate,energy_conc,cell_dictionary,environment_dictionary,ode_system_dictionary,ifVerbose=False):
    

    Vmax = cell_dictionary['vmax']

    r = np.sqrt(6*ode_system_dictionary['dt']*environment_dictionary['glucose_diffusion_constant']) # outer radius of sphere diffused into in 3-dimensions
    volumeElement = (4/3)*np.pi*pow(r,3)

    ifEnergyFound=False
    Km = 0.0
    for i in range(len(cell_dictionary['import_energy_metabolites'])):

        if cell_dictionary['import_energy_metabolites'][i] == cell_dictionary['environment_energy_name']:
            ifEnergyFound = True
            Km = cell_dictionary['import_energy_michaelis_menten'][i]
            break

    if not ifEnergyFound:
        print('Error: energy metabolite not found during CalculateScaling')

    
    V = Vmax * energy_conc/(Km + energy_conc)
    vref_scalar = V/np.abs(uptakeRate) 

    if ifVerbose:
        print('uptakeRate')
        print(uptakeRate)
        print('vref_scalar')
        print(vref_scalar)
        
        print('volumeElement')
        print(volumeElement)
    
    return vref_scalar,volumeElement

def CalculateRateConstants(vref_scalar,v_hat,S,Keq,metab_concs):

    
    kf=[]
    kr=[]
    keq=[]
    for i in range(S.shape[0]):
        productFactor=1
        substrateFactor=1
        for j in range(S.shape[1]):
            if S[i,j] != 0:
                if S[i,j] > 0:
                    productFactor *= pow(metab_concs[j],abs(S[i,j]))
                else:
                    substrateFactor *= pow(metab_concs[j],abs(S[i,j]))

        denomFactor = substrateFactor - (1/Keq[i])*productFactor
        if  denomFactor != 0.0:
            k1=(vref_scalar*v_hat[i])*(1/(denomFactor))
            k2=k1/Keq[i]

            k1=np.abs(k1)
            k2=np.abs(k2)

            kf.append(k1)
            kr.append(k2)
            keq.append(k1/k2)

        else:
            k1=0.0
            k2=0.0

            k1=np.abs(k1)
            k2=np.abs(k2)

            kf.append(k1)
            kr.append(k2)
            keq.append(0.0)

    
    return kf,kr,keq

# for a single timestep, update the concentration and flux dataframes
def SimulateReactionStep(t,kineticsDf,k_app_f,k_app_r, sim_concs, metaboliteTraceDf,fluxTraceDf,simKineticsDf, S, metabolite_names, ode_system_dictionary,environment_dictionary,cell_dictionary,chemostatDf):
    
    ifUseAdaptiveRateParameters = False
    adaptationConstant=1
    if 'adaptation_constant' in cell_dictionary.keys():
        ifUseAdaptiveRateParameters = True
        adaptationConstant = cell_dictionary['adaptation_constant']


    if ifUseAdaptiveRateParameters:

        k_app_f = k_app_f + gradk_app(k_app_f, kineticsDf.loc[:,'kf'].values,adaptationConstant)*ode_system_dictionary['dt'] 

        k_app_r = k_app_r + gradk_app(k_app_r, kineticsDf.loc[:,'kr'].values,adaptationConstant)*ode_system_dictionary['dt']  

    else:
        k_app_f = kineticsDf.loc[:,'kf'].values

        k_app_r = kineticsDf.loc[:,'kr'].values
    
    
    rxnFlux_vec=[]
    for rid in range(len(kineticsDf.index)):

        forwardFactor=k_app_f[rid]
        reverseFactor=k_app_r[rid]
        for metabid in range(len(sim_concs)):
            
            if S[rid,metabid] != 0:
                if S[rid,metabid] > 0:
                    reverseFactor *= pow(sim_concs[metabid],abs(S[rid,metabid]))
                else:
                    forwardFactor *= pow(sim_concs[metabid],abs(S[rid,metabid]))

        rxnNetFlux = forwardFactor - reverseFactor
        rxnFlux_vec.append(rxnNetFlux)

    
    # test each reaction to see which reactions occur within the constraints applied to concentrations
    isReactionPossible = np.ones(len(kineticsDf.index))
    for rid in range(len(kineticsDf.index)):

        for metabid in range(len(sim_concs)):
            if metabolite_names[metabid] not in ode_system_dictionary['metabolites_to_fix']:
            
                if S[rid,metabid] != 0:
                    proposedConcentration = sim_concs[metabid] + S[rid,metabid]*rxnFlux_vec[rid]*ode_system_dictionary['dt']

                    j=ode_system_dictionary['metabolite_names'].index(metabolite_names[metabid])
                    upperConcentrationBound = ode_system_dictionary['metabolite_concentration_maximum'][j]
                    lowerConcentrationBound = ode_system_dictionary['metabolite_concentration_minimum'][j]

                    if proposedConcentration > upperConcentrationBound:
        
                        isReactionPossible[rid]=0
                        break
                    elif proposedConcentration <= lowerConcentrationBound:
                        isReactionPossible[rid]=0
                        break
            else:
                # if a metabolite is fixed assume it is being removed from the system so that the reverse reaction 
                # is not possible
                if S[rid,metabid] != 0:
                    # metabolite is fixed and involved in reaction
                    if S[rid,metabid] < 0:
                        # metabolite is a substrate so reaction cannot occur if in forward sense 
                        if rxnFlux_vec[rid] > 0:
                            isReactionPossible[rid]=0
                            break
                    else:
                        # metabolite is a product so reaction cannot occur if in reverse sense 
                        if rxnFlux_vec[rid] < 0:
                            isReactionPossible[rid]=0
                            break
                
    for rid in range(len(kineticsDf.index)):

        if isReactionPossible[rid]:
            # implement the reactions
            for metabid in range(len(sim_concs)):
                if metabolite_names[metabid] not in ode_system_dictionary['metabolites_to_fix'] and metabolite_names[metabid] not in  environment_dictionary['metabolites']:

                    if S[rid,metabid] != 0:
                        sim_concs[metabid] += S[rid,metabid]*rxnFlux_vec[rid]*ode_system_dictionary['dt'] 
    
    
    sim_concs = UpdateEnvironmentDueToCellDensity(ode_system_dictionary['dt'],S,metabolite_names,sim_concs,kineticsDf.index.tolist(),rxnFlux_vec,environment_dictionary,cell_dictionary)
    sim_concs = UpdateODEBoundaryConditions(t,ode_system_dictionary['dt'],metabolite_names,sim_concs,environment_dictionary,cell_dictionary,chemostatDf)


    for i in range(len(sim_concs)):
        metaboliteTraceDf.at[t,metaboliteTraceDf.columns.tolist()[i]] = sim_concs[i]

    for i in range(len(rxnFlux_vec)):
        fluxTraceDf.at[t,fluxTraceDf.columns.tolist()[i]] = rxnFlux_vec[i]
    
    for i in range(len(k_app_f)):
        simKineticsDf.at[t,simKineticsDf.columns.tolist()[2*i]]   = k_app_f[i]
        simKineticsDf.at[t,simKineticsDf.columns.tolist()[2*i+1]] = k_app_r[i]

    
    return metaboliteTraceDf,fluxTraceDf,simKineticsDf,sim_concs, k_app_f, k_app_r


# for a single timestep, update the external concentrations only, keeping internal concentrations at OPT steady state
def SimulateReactionStepPopulationModel(t,kineticsDf,k_app_f,k_app_r, sim_concs, metaboliteTraceDf,fluxTraceDf,simKineticsDf, S, metabolite_names, ode_system_dictionary,environment_dictionary,cell_dictionary,chemostatDf):
    
    ifUseAdaptiveRateParameters = False
    adaptationConstant=1
    if 'adaptation_constant' in cell_dictionary.keys():
        ifUseAdaptiveRateParameters = True
        adaptationConstant = cell_dictionary['adaptation_constant']


    if ifUseAdaptiveRateParameters:

        k_app_f = k_app_f + gradk_app(k_app_f, kineticsDf.loc[:,'kf'].values,adaptationConstant)*ode_system_dictionary['dt'] 

        k_app_r = k_app_r + gradk_app(k_app_r, kineticsDf.loc[:,'kr'].values,adaptationConstant)*ode_system_dictionary['dt']  

    else:
        k_app_f = kineticsDf.loc[:,'kf'].values

        k_app_r = kineticsDf.loc[:,'kr'].values

    externalReactionNames = environment_dictionary['transport_reactions']
    externalMetaboliteNames = environment_dictionary['metabolites']
    
    
    rxnFlux_vec=np.zeros(len(kineticsDf.index))
    for rid in range(len(kineticsDf.index)):
        if kineticsDf.index[rid] in externalReactionNames or cell_dictionary['biomass_reaction']:
            forwardFactor=k_app_f[rid]
            reverseFactor=k_app_r[rid]
            for metabid in range(len(sim_concs)):
                
                if S[rid,metabid] != 0:
                    if S[rid,metabid] > 0:
                        reverseFactor *= pow(sim_concs[metabid],abs(S[rid,metabid]))
                    else:
                        forwardFactor *= pow(sim_concs[metabid],abs(S[rid,metabid]))

            rxnNetFlux = forwardFactor - reverseFactor
            rxnFlux_vec[rid]=rxnNetFlux

    
    # test each reaction to see which reactions occur within the constraints applied to concentrations
    isReactionPossible = np.ones(len(kineticsDf.index))
    for rid in range(len(kineticsDf.index)):
        if kineticsDf.index[rid] in externalReactionNames or cell_dictionary['biomass_reaction']:
            for metabid in range(len(sim_concs)):
                if metabolite_names[metabid] not in ode_system_dictionary['metabolites_to_fix']:
                
                    if S[rid,metabid] != 0:
                        proposedConcentration = sim_concs[metabid] + S[rid,metabid]*rxnFlux_vec[rid]*ode_system_dictionary['dt']

                        j=ode_system_dictionary['metabolite_names'].index(metabolite_names[metabid])
                        upperConcentrationBound = ode_system_dictionary['metabolite_concentration_maximum'][j]
                        lowerConcentrationBound = ode_system_dictionary['metabolite_concentration_minimum'][j]

                        if proposedConcentration > upperConcentrationBound:
            
                            isReactionPossible[rid]=0
                            break
                        elif proposedConcentration <= lowerConcentrationBound:
                            isReactionPossible[rid]=0
                            break
                else:
                    # if a metabolite is fixed assume it is being removed from the system so that the reverse reaction 
                    # is not possible
                    if S[rid,metabid] != 0:
                        # metabolite is fixed and involved in reaction
                        if S[rid,metabid] < 0:
                            # metabolite is a substrate so reaction cannot occur if in forward sense 
                            if rxnFlux_vec[rid] > 0:
                                isReactionPossible[rid]=0
                                break
                        else:
                            # metabolite is a product so reaction cannot occur if in reverse sense 
                            if rxnFlux_vec[rid] < 0:
                                isReactionPossible[rid]=0
                                break
                    
    # for rid in range(len(kineticsDf.index)):
    #     if kineticsDf.index[rid] in externalReactionNames:
    #         if isReactionPossible[rid]:
    #             # implement the reactions
    #             for metabid in range(len(sim_concs)):
    #                 if metabolite_names[metabid] in externalMetaboliteNames:
    #                     if metabolite_names[metabid] not in ode_system_dictionary['metabolites_to_fix'] :

    #                         if S[rid,metabid] != 0:
    #                             print(S[rid,metabid]*rxnFlux_vec[rid]*ode_system_dictionary['dt'] )
    #                             sim_concs[metabid] += S[rid,metabid]*rxnFlux_vec[rid]*ode_system_dictionary['dt'] 
    
    
    sim_concs = UpdateEnvironmentDueToCellDensity(ode_system_dictionary['dt'],S,metabolite_names,sim_concs,kineticsDf.index.tolist(),rxnFlux_vec,environment_dictionary,cell_dictionary)
    sim_concs = UpdateODEBoundaryConditions(t,ode_system_dictionary['dt'],metabolite_names,sim_concs,environment_dictionary,cell_dictionary,chemostatDf)


    for i in range(len(sim_concs)):
        metaboliteTraceDf.at[t,metaboliteTraceDf.columns.tolist()[i]] = sim_concs[i]

    for i in range(len(rxnFlux_vec)):
        fluxTraceDf.at[t,fluxTraceDf.columns.tolist()[i]] = rxnFlux_vec[i]
    
    for i in range(len(k_app_f)):
        simKineticsDf.at[t,simKineticsDf.columns.tolist()[2*i]]   = k_app_f[i]
        simKineticsDf.at[t,simKineticsDf.columns.tolist()[2*i+1]] = k_app_r[i]

    
    return metaboliteTraceDf,fluxTraceDf,simKineticsDf,sim_concs

# for future models implement a function to model the transition between kinetic states
def gradk_app(k_app, k_calc,paraA):
    
    return -paraA*(k_app-k_calc)


# function to account for loss and replenishment of local environment metebolite concentrations due to diffusion
def DiffuseEnvironment(t, sim_concs, metaboliteTraceDf,metabolite_labels,ode_system_dictionary,environment_dictionary):
    
    if 'diffusion_rates' in environment_dictionary.keys():

        diffusionRates = environment_dictionary['diffusion_rates']
        environmentMetabolites= environment_dictionary['metabolites']
        boundary_concs = environment_dictionary['boundary_concentrations']
    
        for i in range(len(environmentMetabolites)):

            j=metabolite_labels.tolist().index(environmentMetabolites[i])

            sim_concs[j] = sim_concs[j] + gradDiff(sim_concs[j],boundary_concs[i],diffusionRates[i])*ode_system_dictionary['dt'] 

        for i in range(len(sim_concs)):
            metaboliteTraceDf.at[t,metaboliteTraceDf.columns.tolist()[i]] = sim_concs[i]
    
    return metaboliteTraceDf,sim_concs

# for the diffusion update term 
def gradDiff(concEnvironment,concBoundary,diffusionRate):
    
    return diffusionRate*(concBoundary-concEnvironment)

def CalculateCellDensity(cell_dictionary,ode_system_dictionary,cell_density,biomass_flux):
    
    birth_rate = 0.0
    death_rate = 0.0
    
    if 'birth_rate' in cell_dictionary.keys():
        birth_rate = cell_dictionary['birth_rate']
    if 'death_rate' in cell_dictionary.keys():
        death_rate = cell_dictionary['death_rate']
    
    dt = ode_system_dictionary['dt']
    
    cell_volume = cell_dictionary['cell_volume']
    gluc_gram_mole = 180.156
    gramBiomass_to_cellcount =  5e-13
    grams_of_biomass = biomass_flux*cell_volume*(0.3)*gluc_gram_mole
    growth_rate  = grams_of_biomass/gramBiomass_to_cellcount

    cell_density = cell_density + (growth_rate*cell_density - death_rate*cell_density)*dt

    return cell_density

def UpdateEnvironmentDueToCellDensity(dt,S,metabolite_names,metabolite_concentrations,reaction_names,rxnFlux_vec,environment_dictionary,cell_dictionary):

    cell_density = cell_dictionary['cell_density']
    cell_volume = cell_dictionary['cell_volume']
    environmentMetabolites=environment_dictionary['metabolites']
    bath_volume = environment_dictionary['bath_volume']
    
    transportReactions=environment_dictionary['transport_reactions']

    for i in range(len(environmentMetabolites)):

        metab_id = metabolite_names.tolist().index(environmentMetabolites[i])

        rxn_id = reaction_names.index(transportReactions[i])

        # assume that environment is large enough for each to cell to act optimally

        # (correction) reverse the previous reaction value 
        # metabolite_concentrations[metab_id] -= S[rxn_id,metab_id]*rxnFlux_vec[rxn_id]*dt


        # add_conc=cell_density*S[rxn_id,metab_id]*rxnFlux_vec[rxn_id]*dt*bath_volume#*(cell_volume/bath_volume)


        # apply flux from cell density
        avgadro_num = 6.02214076e23
        conversion_factor = cell_volume*cell_density*(avgadro_num / 1e9)
        metabolite_concentrations[metab_id] += conversion_factor*S[rxn_id,metab_id]*rxnFlux_vec[rxn_id]*dt

    return metabolite_concentrations



def UpdateODEBoundaryConditions(t,dt,metabolite_names,metabolite_concentrations,environment_dictionary,cell_dictionary,chemostatDf):

    environmentMetabolites=environment_dictionary['metabolites']

    for em in environmentMetabolites:

        metab_id = metabolite_names.tolist().index(em)

        if environment_dictionary['chemostat_type'][environmentMetabolites.index(em)][0] == 'Dirichlet' or environment_dictionary['chemostat_type'][environmentMetabolites.index(em)][0] == 'Fixed':
            # for fixed chemostated environment, replace the ode concentration by the BCs magnitude
            metabolite_concentrations[metab_id] = chemostatDf.loc[t,em]

        else:
            # add chemostat
            if environment_dictionary['chemostat_type'][environmentMetabolites.index(em)][0] == 'Neumann' or environment_dictionary['chemostat_type'][environmentMetabolites.index(em)][0] == 'Constant_Flux':
                metabolite_concentrations[metab_id] -= chemostatDf.loc[t,em] *dt

    return metabolite_concentrations


def PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,ifPlotOpt=True,metaboliteSubset=[],xlim=0,scatterSize=0):
    optConcMatrix=[]

    if metaboliteSubset ==[]:

        for metab in metabolitesOPT[0].index.tolist():
            metabVec=[]
            for i in range(len(metabolitesOPT)):
                metabVec.append(metabolitesOPT[i].at[metab,'metabolite_conc'])

            optConcMatrix.append(metabVec)

        metabNames=metaboliteTraceDf.columns.tolist()
        if ifPlotOpt:
            for metab in metabNames:
                plt.plot(metaboliteTraceDf[metab],label=metab)
                if scatterSize !=0:
                    plt.scatter(timesOPT,optConcMatrix[metabNames.index(metab)],s=scatterSize)
                else:
                    plt.scatter(timesOPT,optConcMatrix[metabNames.index(metab)],s=scatterSize)
        else:
            for metab in metabNames:
                plt.plot(metaboliteTraceDf[metab],label=metab)

    else:

        for metab in metabolitesOPT[0].index.tolist():
            metabVec=[]
            if metab in metaboliteSubset:
                for i in range(len(metabolitesOPT)):
                    metabVec.append(metabolitesOPT[i].at[metab,'metabolite_conc'])

                optConcMatrix.append(metabVec)

        metabNames=metaboliteSubset
        if ifPlotOpt:
            for metab in metabNames:
                plt.plot(metaboliteTraceDf[metab],label=metab)
                if scatterSize !=0:
                    plt.scatter(timesOPT,optConcMatrix[metabNames.index(metab)],s=scatterSize)
                else:
                    plt.scatter(timesOPT,optConcMatrix[metabNames.index(metab)],s=scatterSize)
        else:
            for metab in metabNames:
                plt.plot(metaboliteTraceDf[metab],label=metab)

    plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    if xlim != 0:
        plt.xlim([0,xlim])
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Concentration (mili-molar)')
    plt.show()
    
    return

def PlotFluxTraces(fluxTraceDf,reactionOPT,timesOPT,ifPlotOpt=True,reactionSubset=[],xlim=0,scatterSize=0):
    optFluxMatrix=[]

    if reactionSubset ==[]:

        rxnNames=fluxTraceDf.columns.tolist()
        for rxn in rxnNames:
            fluxVec=[]
            for i in range(len(reactionOPT)):
                fluxVec.append(reactionOPT[i]['Optimal Flux'][rxnNames.index(rxn)])

            optFluxMatrix.append(fluxVec)

        if ifPlotOpt:
            for rxn in rxnNames:
                plt.plot(fluxTraceDf[rxn],label=rxn)
                if scatterSize !=0:
                    plt.scatter(timesOPT,optFluxMatrix[rxnNames.index(rxn)],s=scatterSize)
                else:
                    plt.scatter(timesOPT,optFluxMatrix[rxnNames.index(rxn)])
        else:
            for rxn in rxnNames:
                plt.plot(fluxTraceDf[rxn],label=rxn)

    else:
        rxnNames=fluxTraceDf.columns.tolist()
        for rxn in rxnNames:
            fluxVec=[]
            if rxn in reactionSubset:
                for i in range(len(reactionOPT)):
                    fluxVec.append(reactionOPT[i]['Optimal Flux'][rxnNames.index(rxn)])

                optFluxMatrix.append(fluxVec)

        if ifPlotOpt:
            for rxn in reactionSubset:
                plt.plot(fluxTraceDf[rxn],label=rxn)
                if scatterSize !=0:
                    plt.scatter(timesOPT,optFluxMatrix[rxnNames.index(rxn)],s=scatterSize)
                else:
                    plt.scatter(timesOPT,optFluxMatrix[rxnNames.index(rxn)])
        else:
            for rxn in reactionSubset:
                plt.plot(fluxTraceDf[rxn],label=rxn)


    plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Reaction flux')
    if xlim != 0:
        plt.xlim([0,xlim]) 

    plt.show()


def PlotRegulationTraces(regulationOPT,timesOPT,rxnNames,reactionSubset=[],xlim=0,scatterSize=0):
    optRegulationMatrix=[]

    delta = (timesOPT[-1]/(len(timesOPT)-1))/10 #1e5

    if reactionSubset ==[]:

        for rxn in rxnNames:
            RegulationVec=[]
            for i in range(len(regulationOPT)):
                RegulationVec.append(regulationOPT[i][rxnNames.index(rxn)])

            optRegulationMatrix.append(RegulationVec)

        for rxn in rxnNames:
            timesRxn = [t + rxnNames.index(rxn)*delta for t in timesOPT]
            
            if scatterSize !=0:
                    plt.scatter(timesRxn,optRegulationMatrix[rxnNames.index(rxn)],label=rxn,s=scatterSize)
            else:
                plt.scatter(timesRxn,optRegulationMatrix[rxnNames.index(rxn)],label=rxn)
    else:

        for rxn in rxnNames:
            RegulationVec=[]
            if rxn in reactionSubset:
                for i in range(len(regulationOPT)):
                    RegulationVec.append(regulationOPT[i][rxnNames.index(rxn)])
                optRegulationMatrix.append(RegulationVec)


        for rxn in reactionSubset:
            timesRxn = [t + reactionSubset.index(rxn)*delta for t in timesOPT]

            if scatterSize !=0:
                plt.scatter(timesRxn,optRegulationMatrix[reactionSubset.index(rxn)],label=rxn,s=scatterSize)
            else:
                plt.scatter(timesRxn,optRegulationMatrix[reactionSubset.index(rxn)],label=rxn)



    plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Reaction regulation level')
    if xlim != 0:
        plt.xlim([0,xlim]) 
    plt.ylim([0,1.05])
    plt.show()

def PlotIntegratedBiomass(fluxTraceDf,biomassRxn):

    plt.plot(fluxTraceDf.index.tolist()[:-1],it.cumtrapz(fluxTraceDf[biomassRxn],x=fluxTraceDf.index.tolist()),label=biomassRxn)
    
    plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Cumulative biomass')
    plt.show()
    
    return