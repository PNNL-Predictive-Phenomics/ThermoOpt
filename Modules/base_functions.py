"""

Connah G. M. Johnson     
connah.johnson@pnnl.gov
July 2023

Functions to implement the parallel swarm optimized of a set of metabolic simulations. 


Code contents:

def RunSimulation
def RunParallelSimulations
def RunSwarmSimulation
def ReturnBestSimulation
def derivatives
def odds
def oddsDiff
def calc_delta_S
def calc_E_step
def reduced_Stoich
def load_model
def save_model
def save_model_solution
def load_model_solution
def save_model_initial
def load_model_initial
def flux_gradient
def f_log_counts_names
def FixChemicalSpecies
def RelaxChemicalSpecies
def ChangeFixedBoundaryConditionValue
def ChangeBoundaryCondition
def RemoveMetabolite
def RemoveReaction
def CheckMetabolitesInModel
def PrintMetabolitesInModel
def CheckReactionsInModel
def ConvertBestSimulationsToDataFrameHyperparameter(
def ReportModelEnvironment
def ParseMetaboliteTypes
def ParseReactionTypes
def ScanOverRedoxRatios
def ScanOverInitialCondition
def ScanOverBoundaryCondition

def EvaluateStoich


"""

import numpy as np
import pandas as pd
import random
import re
import multiprocessing as mp
import numpy.random as nprd
import subprocess
from numpy.random import randn, seed
from scipy.linalg import norm
from scipy.optimize import least_squares
import scipy.linalg as spL
import sys
import pickle
import copy

import Bolt_opt_functions as fopt


"""
Simulation routine for each swarm agent. Called from RunParallelSimulationsHyperparameter(...). Implements the agent 
experiment and calls the optimization solver flux_ent_opt_hyperparameter(...) implementing both primary and secondary 
optimization when necessary. 

Inputs:
  - simulationDict : experiment simulation dictionary for the swarm agent;
                    'name', 'model', 'model_output', 'secondary_objective', 'primary_obj_tolerance', 'initial_conditions', 'boundary_conditions',
                    'obj_coefs', 'remove_metabolites', 'remove_reactions', 'implicit_metabolites', 'explicit_metabolites', 'redox_pairs', 
                    'redox_ratios', 'raise_bounds', 'specify_upper_bound', 'specify_lower_bound', 'net_flux', 'flux_direction'
  - model_file_dir : file directory storing each swarm agent model
  - save_dir : default ('Saved_solutions/')
  - hyperparameterDict : default ({})
  - disableInitialConditions : default (False)

Returns:
    return solved,pri_status_obj,pri_obj_val,sec_status_obj,sec_obj_val,simulationDict['name']

"""

def RunSimulation(simulationDict,model_file_dir,save_dir='Saved_solutions/',hyperparameterDict={}):
   

  # for a given simulation swarm set up the ipopt optimize routine 

  # load the model data from the simulation files
    if model_file_dir in simulationDict['model']:
      # print("\n Loading model from: "+str(simulationDict['model'])+"\n")
      S_active,active_reactions, metabolites, Keq= load_model(simulationDict['model'])
    else:
      # print("\n Loading model from: "+str(model_file_dir + '/' + simulationDict['model'])+"\n")
      S_active,active_reactions, metabolites, Keq= load_model(model_file_dir + '/' + simulationDict['model'])

    # display(metabolites)


    ifUsePrevious=False
    if 'use_previous_simulation' in simulationDict.keys():
       if str(simulationDict['use_previous_simulation']) != "":
        # print('use previous')
        # print(str(simulationDict['use_previous_simulation'])+'.pkl')
        ifUsePrevious=True


    # ensure all input dataframes are standardised, provide default values if not

    if 'Steady_State' not in metabolites.columns:
      var_to_ss_dict = {True: "Explicit", False:"Implicit"}

      metabolites['Steady_State'] = metabolites['Variable']

      metabolites.replace({"Steady_State":var_to_ss_dict},inplace = True)

    if 'Net_Flux' not in metabolites.columns:
      metabolites['Net_Flux'] = np.zeros(len(metabolites))

    if 'Upper_Bound' not in metabolites.columns:
      metabolites['Upper_Bound'] = np.ones(len(metabolites))*1e-3

    if 'Lower_Bound' not in metabolites.columns:
      metabolites['Lower_Bound'] = np.ones(len(metabolites))*1e-120

    if 'remove_metabolites' in simulationDict.keys():
      for metab in simulationDict['remove_metabolites']:
        try:
          metabolites,S_active = RemoveMetabolite(metab,metabolites,S_active)
        except:
           print("Error in remove_metabolites")  
          #  print("Metabolite "+str(metab)+" not in model")

    if 'remove_reactions' in simulationDict.keys():
      for reactionName in simulationDict['remove_reactions']:
        try:
          active_reactions,S_active = RemoveReaction(reactionName,active_reactions,S_active)
        except:
          print("Error in remove_reactions")  
          # print("Reaction "+str(reactionName)+" not in model")

    # as metabolites and reactions have been removed, reorder (shouldn't change the ordering as only dropped).
    # metabolites.sort_values('Variable',ascending = False,inplace = True)

    original_metabolite_index = list(metabolites[metabolites['Variable']==False].index)

    metabolites.sort_values(by = ['Variable','Steady_State'],ascending = [False,True],inplace = True)
    # Sort S_active so the columns are in the same order
    S_active = S_active[list(metabolites.index)]
   
    # set the initial conditions for the metabolites by changing the conc value of the metabolites dataframe, independent of the variable/fixed status
    if 'initial_conditions' in simulationDict.keys():

      for metab in simulationDict['initial_conditions']:
          try:
            if metab in metabolites.index:

              metabolites.at[metab,'Conc'] = simulationDict['initial_conditions'][metab]
            else:
              print("Metabolite "+str(metab)+" not in model\n")
          except:
            print("Error setting initial_conditions")  

    if 'boundary_conditions' in simulationDict.keys():
      # if boundary conditions are provided, change them 
      for metab in simulationDict['boundary_conditions']:
        
        try:
          # a change in the metabolite will be added to the final element of metabiltes
          metabolites = ChangeBoundaryCondition(metab,simulationDict['boundary_conditions'][metab],metabolites)
        except:
          print("except: Metabolite "+str(metab)+" not in model\n")


    # set implicit and explicit steady states

    if 'implicit_metabolites' in simulationDict.keys():
      # convert the metabolites for implicit

      for metab in simulationDict['implicit_metabolites']:
          try:
            metabolites.at[metab,"Variable"] = True
            metabolites.at[metab,"Steady_State"] = "Implicit"
          except:
            print("Error setting implicit_metabolites")   

    if 'explicit_metabolites' in simulationDict.keys():
      # convert the metabolites for explicit
      for metab in simulationDict['explicit_metabolites']:
          try:
            # metabolites.at[metab,"Variable"] = False
            metabolites.at[metab,"Steady_State"] = "Explicit"
          except:
            print("Error setting explicit_metabolites")   


    # fix the redox elements to initial conditions
    if 'redox_pairs' in simulationDict.keys():
      for redox_pair in simulationDict['redox_pairs']:
        try:
          if len(redox_pair)==2:
            # check whether either pair element is already fixed
            name_1 = redox_pair[0]
            name_2 = redox_pair[1]

            isVar_1 = metabolites.loc[name_1,'Variable']
            isVar_2 = metabolites.loc[name_2,'Variable']

            if isVar_1 and isVar_2:
              # both species variable, fix name_1
              metabolites = FixChemicalSpecies(metabolites,name_1)


            elif isVar_1==False and isVar_2==False:
              # both species are fixed, relax name_2 set to variable
              print("Both redox species in the pair are already fixed: "+str(name_1)+" and "+str(name_2) )
              #relax the second chemical
              metabolites = RelaxChemicalSpecies(metabolites,name_2)

            elif isVar_1==False and isVar_2==True:
               # the second species is fixed which may cause issues with convergence if multiple pairs are defined
               print("Second redox species in the pair is already fixed: "+str(name_1)+" and "+str(name_2) )

               #relax the second chemical and fix the first chemical instead
               metabolites = FixChemicalSpecies(metabolites,name_1)
               metabolites = RelaxChemicalSpecies(metabolites,name_2)
          else:
            print("Error redox_pairs should have two elements each")
        except:
            print("Error setting redox pairs")   


    if 'redox_ratios' in simulationDict.keys():
      # fix the second to the intial concentration and the first to the ratio*second initial concentration
      for redox_ratio in simulationDict['redox_ratios']:
        try:
          if len(redox_ratio['names'])==2:
            name_1 = redox_ratio['names'][0]
            name_2 = redox_ratio['names'][1]

            ratio_value = redox_ratio['ratio']

            metabolites.at[name_1,'Conc'] = ratio_value*metabolites.loc[name_2]['Conc']

            # fix both species
            metabolites = FixChemicalSpecies(metabolites,name_1)
            metabolites = FixChemicalSpecies(metabolites,name_2)
          else: 
            print("Ratio pairs need (only) two species names")
        except:
            print("Error setting redox ratios")   

    # the inclusion of boundary conditions have changed the ordering and number of fixed and variable metabolites    
    # reorder and recalculate the model

    # reorder such that variable metabolites are above fixed
    metabolites.sort_values(by = ['Variable','Steady_State'],ascending = [False,True],inplace = True)
    # sort stoich based on reordered metabolites
    S_active = S_active[list(metabolites.index)]
    S = S_active.values

    ## Equilibrium Constants
    T = 298.15
    R = 8.3144598e-03
    RT = R*T
    
    Keq = np.exp(-active_reactions['DGZERO'].astype('float')/RT)
    Keq = Keq.values
    nvar = len(metabolites[ metabolites['Variable']==True].values)
    # Add the equilibrium constants as a column in active_reactions
    active_reactions['Keq'] = Keq
 
    if 'Flux_Direction' not in active_reactions.columns:
      active_reactions['Flux_Direction'] = np.zeros(len(active_reactions))

    if 'flux_direction'  in simulationDict.keys():
        
        for rxnName in simulationDict['flux_direction']:
          try:
            active_reactions.at[rxnName,"Flux_Direction"] = simulationDict['flux_direction'][rxnName]
          except:
            print("Error setting flux_direction")  

    if 'Allow_Regulation' not in active_reactions.columns:
      active_reactions['Allow_Regulation'] = np.ones(len(active_reactions))

    if 'allow_regulation'  in simulationDict.keys():
        
        for rxnName in simulationDict['allow_regulation']:
          try:
            active_reactions.at[rxnName,"Allow_Regulation"] = simulationDict['allow_regulation'][rxnName]
          except:
            print("Error setting flux_direction")  


    if 'Obj_Coefs' not in active_reactions.columns:
      active_reactions['Obj_Coefs'] = np.zeros(len(active_reactions))


    if 'obj_coefs'  in simulationDict.keys() and simulationDict['obj_coefs']:

      active_reactions['Obj_Coefs'] = np.zeros(len(active_reactions))
      for rxnName in simulationDict['obj_coefs']:
        try:
          active_reactions.at[rxnName,"Obj_Coefs"] = simulationDict['obj_coefs'][rxnName]
        except:
          print("Error setting objective coefficients")

    metabolites.sort_values(by = ['Variable','Steady_State'],ascending = [False,True],inplace = True)


    pri_status_obj='1'
    try:
      isSolveSucessful=True
      solved=0


      if ifUsePrevious:
        # use the previous solution to initalise the new optimizer initial conditions
        # print('Use previous solution for opt_setup')
        previous_model_file = save_dir + str(simulationDict['use_previous_simulation'])+'.pkl'
        n_ini, y_ini, opt_parms, setupError = fopt.opt_setup(metabolites, active_reactions, S_active,hyperparameterDict,simulationDict,previous_model_file)

      else:
        # no previous solution to use as an initial optimiser point, therefore use the least-squares method

        n_ini, y_ini, opt_parms, setupError = fopt.opt_setup(metabolites, active_reactions, S_active,hyperparameterDict,simulationDict)
        pri_status_obj=setupError

      # print('n_ini: ',n_ini)
      # print('y_ini: ',y_ini)
    
      if setupError:
        solved=0
        isSolveSucessful=False
        pri_status_obj='Fail'
        pri_obj_val=0
        pri_status_obj=setupError
        raise Exception('Opt error')

          
      if model_file_dir in simulationDict['model']:

        save_model_initial(simulationDict['model']+'_inital.pkl',y_ini,n_ini,opt_parms)
      else:

        save_model_initial(model_file_dir + '/initial_' + simulationDict['model'],y_ini,n_ini,opt_parms)

      y_sol = y_ini
      n_sol=n_ini

      try:
    
        y_sol, alpha_sol, n_sol,unreg_rxn_flux,f_log_counts,vcount_lower_bound,vcount_upper_bound,solved,pri_status_obj,pri_obj_val = fopt.flux_ent_opt(n_ini, y_ini, opt_parms)

        if solved:
            pri_status_obj='ok'
      except:

        solved=0
        print("exception in flux_ent_opt_hyperparameter")
        isSolveSucessful=False
        pri_status_obj='Fail'
        pri_obj_val=0
        raise Exception("exception in flux_ent_opt_hyperparameter")

    
      if 'feasibility_as_input' in hyperparameterDict:
        
        feasibility_as_input = hyperparameterDict['feasibility_as_input']
        if feasibility_as_input:
          # print('Use feasible point as input')
          hyperparameterDict['feasibility_as_input'] = False
          hyperparameterDict['feasibility_check']  = False
          
          # print('n_ini: ',n_sol)
          # print('y_ini: ',y_sol)  

          n_ini, y_ini, opt_parms = fopt.opt_setup(metabolites, active_reactions, S_active,hyperparameterDict,simulationDict,"",n_sol,y_sol)
          # n_ini = n_sol
          # y_ini = y_sol
          # print('n_ini: ',n_ini)
          # print('y_ini: ',y_ini)  

          try:
            y_sol, alpha_sol, n_sol,unreg_rxn_flux,f_log_counts,vcount_lower_bound,vcount_upper_bound,solved,pri_status_obj,pri_obj_val = fopt.flux_ent_opt(n_ini, y_ini, opt_parms)
            if solved:
                pri_status_obj='ok'
          except:
            solved=0
            print("exception in flux_ent_opt_hyperparameter - feasible")
            isSolveSucessful=False
            pri_status_obj='Fail'
            pri_obj_val=0





      if 'secondary_objective' in simulationDict.keys() and isSolveSucessful:
      # set up secondary objective
        if simulationDict['secondary_objective'] and solved:
            print("Run secondary objective\n")
            try:
              y_sol, alpha_sol, n_sol,sec_status_obj,sec_obj_val = fopt.Max_alt_flux(n_sol,y_sol,opt_parms)
            except:
              isSolveSucessful=False
              sec_status_obj='Fail'
              sec_obj_val=0

        else:
          print("Primary solve did not converge, abort secondary solve")
          isSolveSucessful=False
          sec_status_obj='Fail'
          sec_obj_val=0

      else:
          sec_status_obj='None'
          sec_obj_val=0

      #### Save the solution
      solution_save_file = simulationDict['model_output']

      ### PUT THIS IN TO SAVE EARLY TERMINATION SOLUTIONS
      try:
        save_model_solution(save_dir + solution_save_file,y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound,opt_parms)
      except:
        print('No solution found')
  #save_file,y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound

      
      if isSolveSucessful:
        if 'secondary_objective' in simulationDict.keys():
          solution_save_file=simulationDict['model_output']
          if save_dir in solution_save_file:
            save_model_solution(solution_save_file,y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound)
          else:
            save_model_solution(save_dir + solution_save_file,y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound)

          # if the model has been edited then need to save the model as well
          file_name = str(simulationDict['model'])#+'_secondary.pkl'
          if model_file_dir in file_name:
            save_model(file_name, S_active, active_reactions, metabolites, Keq) 
          else:
            save_model(model_file_dir + '/' + file_name, S_active, active_reactions, metabolites, Keq)
        else:

          solution_save_file=simulationDict['model_output']
          model_save_file=simulationDict['model_output']

          if save_dir in solution_save_file:
            save_model_solution(solution_save_file,y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound,opt_parms)
            save_model(model_file_dir + '/' +  solution_save_file.split('/',1)[1],S_active,active_reactions,metabolites,Keq)
          else:
            save_model_solution(save_dir + solution_save_file,y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound,opt_parms)
            save_model(model_file_dir + '/' +  solution_save_file,S_active,active_reactions,metabolites,Keq)
      return (solved,pri_status_obj,pri_obj_val,sec_status_obj,sec_obj_val,simulationDict['name'])
    except:
    # print("In pool Solved: "+str(solved)+"\n"+" objective: "+str(pri_obj_val)+" status: \n"+str(pri_status_obj)+"\n")
  #   return False
        status_obj=False
        obj_val=0
        isSolveSucessful=False
        pri_obj_val=0
        sec_status_obj='Fail'
        sec_obj_val=0
        return (0,pri_status_obj,pri_obj_val,sec_status_obj,sec_obj_val,simulationDict['name'])


'''
Parallel routine for running a set of swarm simulations. Called from RunSwarmHyperparameterSimulation(...)


Inputs:
  - simulationList: List of simulation dictionaries
  - model_file_dir: string directory of where the original model is stored as a .pkl file
  - save_dir: string directory to save the solutions into (default 'Saved_solutions/')
  - hyperparameterDict: 'Mb', 'zeta', 'delta_concentration', 'delta_flux'
  - configDictionary: 'parallel_solve', 'worker_count'

Returns:
    return outputDict

'''
def init_pool_processes():
    seed()

def RunParallelSimulations(simulationList,model_file_dir,save_dir='Saved_solutions/',hyperparameterDict={},configDictionary={}):
    
    workercount = 10#mp.cpu_count()
    
    outputDict={
    }

    if 'linear_solver' in configDictionary.keys():
       # transfer over into hyperparameterDict
       hyperparameterDict['linear_solver'] = configDictionary['linear_solver']

    if 'hsllib' in configDictionary.keys():
       # transfer over into hyperparameterDict
       hyperparameterDict['hsllib'] = configDictionary['hsllib']

    if 'feasibility_check' in configDictionary.keys():
       # transfer over into hyperparameterDict
       hyperparameterDict['feasibility_check'] = configDictionary['feasibility_check']

    if 'feasibility_as_input' in configDictionary.keys():
       # transfer over into hyperparameterDict
       hyperparameterDict['feasibility_as_input'] = configDictionary['feasibility_as_input']

    if 'annealing_check' in configDictionary.keys():
       # transfer over into hyperparameterDict
       hyperparameterDict['annealing_check'] = configDictionary['annealing_check']

    if 'max_cpu_time' in configDictionary.keys():
       # transfer over into hyperparameterDict
       hyperparameterDict['max_cpu_time'] = configDictionary['max_cpu_time']

    if 'max_iter' in configDictionary.keys():
       # transfer over into hyperparameterDict
       hyperparameterDict['max_iter'] = configDictionary['max_iter']

    if 'acceptable_tol' in configDictionary.keys():
       # transfer over into hyperparameterDict
       hyperparameterDict['acceptable_tol'] = configDictionary['acceptable_tol']

    if 'solver_tol' in configDictionary.keys():
       # transfer over into hyperparameterDict
       hyperparameterDict['solver_tol'] = configDictionary['solver_tol']

    para = False
    if 'parallel_solve' in configDictionary.keys():
       para = configDictionary['parallel_solve']


    if para:
        workercount = mp.cpu_count()
        if 'worker_count' in configDictionary.keys():
           if configDictionary['worker_count']:
            workercount=configDictionary['worker_count']

        # print("Begin pool with "+str(workercount)+" workers")
        total=0
        newIterValues=[]
        arrayOfSubDicts = np.array_split(simulationList, np.ceil(len(simulationList)/(workercount)))
        for simSet in arrayOfSubDicts:
            iterValues=[]
            poolValues=[]

            for sim in simSet:
              iterValues.append((sim,model_file_dir,save_dir,hyperparameterDict))
            # print('Iter values')
            # print(iterValues)

            # try:

            output=[]
            result = list()
            with mp.Pool(processes=workercount, initializer=init_pool_processes) as pool:
                poolValues = pool.starmap(RunSimulation,iterValues)

                # for i in range(0, len(simSet)):
                #   print(poolValues[i])
                # while True:
                #     try:
                #         result.append(next(iterator))
                #     except StopIteration:
                #         break
                #     except Exception as e:
                #         # do something
                #         result.append(e)
              # poolValues = pool.starmap(RunSimulation,iterValues)
              # # poolValues.wait()       
              # print('poolValues')
              # # print(poolValues.get())  

              # for i in range(0, len(simSet)):
              #     # print(poolValues.get()[i])
              #     # output.append(poolValues.get()[i])
              #     try:
              #         output.append(next(poolValues))
              #         print(output[i])
              #     except StopIteration:
              #         break
              #     except Exception as e:
              #         # do something
              #         output.append(e)



            # except:
            #     print("Error in RunParallelSimulations::pool.starmap(RunSimulation,iterValues)")
            # print(poolValues)
            total+=len(simSet)
            # print()
            # print('----------------------------------------------------')
            # print("Processed "+str(total)+" out of "+str(len(simulationList)))    
            # print('----------------------------------------------------')
            # print()
            for valueSet in poolValues:
                # map back to reactions dataframe
                solved = valueSet[0]
                pri_status_obj = valueSet[1]
                pri_obj_val = valueSet[2]
                sec_status_obj = valueSet[3]
                sec_obj_val = valueSet[4]
                sim_name = valueSet[5]
                print("Sim Name: "+str(sim_name)+" objective: "+str(pri_obj_val)+" status: "+str(pri_status_obj))
                outputDict[sim_name] = {'solver_status':pri_status_obj,'objective_value':pri_obj_val,'secondary_status':sec_status_obj,'secondary_objective_value':sec_obj_val}               
            # print("Pool End\n")
           
    else:
      # run in serial, needed to see print outputs from the RunSimulationHyperparameter call
      # print("run serial")
      for simulation in simulationList:
          
          solved,pri_status_obj,pri_obj_val,sec_status_obj,sec_obj_val,sim_name = RunSimulation(simulation,model_file_dir,save_dir,hyperparameterDict)
          outputDict[simulation['name']] = {'solver_status':pri_status_obj,'objective_value':pri_obj_val,'secondary_status':sec_status_obj,'secondary_objective_value':sec_obj_val}
          print("Sim Name: "+str(sim_name)+" objective: "+str(pri_obj_val)+" status: "+str(pri_status_obj))
    return outputDict



"""
Main method for setting and running a swarm simulation.

This function sets up a set of simulation/optimization runs scanning over hyperparameters and initial condition perturbations
while forming the individual model files for the swarm agents. These are then passed to the parallel routines for the 
simulations. The simulation outputs are then stored in the bestSimulations list of dictionaries with the swarm agent with the
best objective values for the simulation conditions identified.  

Input:
  - simulationList : List containing the simulated experiment condition dictionary. 
  - configDictionary: dictionary of simulation configuration parameters that may be scanned over; 
                      'model_file_dir', 'delta_concentration', 'delta_flux', 'mb', 'zeta', 'swarm_size'

  

Returns:
  - bestSimulations : list of the best simulation dictionries for each set of hyper parameters and initial condition perturbations

"""
def RunSwarmSimulation(simulationList,configDictionary):
    
    # set up hyperparameters
    bestHyperParameterSimulations=[]
    
    MbList = [100]#np.arange(10,311,100)
    zeta_list = [0.1]
    deltaConc_list=[0]
    deltaFlux_list=[0]
    swarmNumber=2

    if 'model_file_dir' in configDictionary.keys():
       model_file_dir = configDictionary['model_file_dir']
       if model_file_dir[-1] != "/":
          model_file_dir = model_file_dir +"/"
    else:
       model_file_dir='Saved_models/'

    if 'solution_file_dir' in configDictionary.keys():
       solution_file_dir = configDictionary['solution_file_dir']
       if solution_file_dir[-1] != "/":
          solution_file_dir = solution_file_dir +"/"
    else:
       solution_file_dir = 'Saved_solutions/'


    if 'delta_concentration' in configDictionary.keys():
       
      if type(configDictionary['delta_concentration']) is list:
        deltaConc_list = configDictionary['delta_concentration']
      else:
        deltaConc_list = [configDictionary['delta_concentration']]

    if 'delta_flux' in configDictionary.keys():
        if type(configDictionary['delta_flux']) is list:
          deltaFlux_list = configDictionary['delta_flux']
        else:
          deltaFlux_list = [configDictionary['delta_flux']]

    if 'mb' in configDictionary.keys():
        if type(configDictionary['mb']) is list:
          MbList = configDictionary['mb']
        else:
          MbList = [configDictionary['mb']]

    if 'zeta' in configDictionary.keys():
        if type(configDictionary['zeta']) is list:
          zeta_list = configDictionary['zeta']
        else:
          zeta_list = [configDictionary['zeta']]

    if 'swarm_size' in configDictionary.keys():
       
      swarmNumber = configDictionary['swarm_size']


    # produce a list of hyperparameter lists
    hyperParameterList=[]
    for m in MbList:
      for z in zeta_list:
        for c in deltaConc_list:
          for f in deltaFlux_list:
            hyperParameterList.append([m,z,c,f]) 

    # for each set of hyperparameters, run each simulation for each swarm member; report best swarm objective per sim
    for hyperParams in hyperParameterList:
       
      hyperparameterDict={}
      hyperparameterDict['Mb'] = hyperParams[0]
      hyperparameterDict['zeta'] = hyperParams[1]
      hyperparameterDict['delta_concentration'] = hyperParams[2]
      hyperparameterDict['delta_flux'] = hyperParams[3]

      swarmSimulationList=[]

      # for each simulation experiment, set up the swarm agent model files
      for sim in simulationList:
          
          S_active,active_reactions, metabolites, Keq= load_model(model_file_dir + sim['model'])

          Keq = np.array(Keq)
          Keq[np.isinf(Keq)] = 1e+300
          Keq[Keq>1e+300] = 1e+300
          Keq[Keq<1e-16] = 1e-16

          active_reactions['Keq'] = Keq
          
          for swarm_index in range(swarmNumber):
              
              # name the swarm files by the swarm number and hyperparameter values
              swarmMetabolites = copy.deepcopy(metabolites)
              swarm_model_file=""
              swarm_output_file=""
              if '.pkl' in sim['model_output']:
                sim['model_output'] = sim['model_output'].replace('.pkl', "")
                if sim['model_output'][-1] == "_":
                  swarm_model_file = model_file_dir + sim['model_output']+"swarm_"+str(swarm_index)+"_"+str(hyperParams[0])+"_"+str(hyperParams[1])+"_"+str(hyperParams[2])+"_"+str(hyperParams[3])
                  swarm_output_file=sim['model_output']+"swarm_"+str(swarm_index)+"_"+str(hyperParams[0])+"_"+str(hyperParams[1])+"_"+str(hyperParams[2])+"_"+str(hyperParams[3])
                else:
                  swarm_model_file = model_file_dir + sim['model_output']+"_swarm_"+str(swarm_index)+"_"+str(hyperParams[0])+"_"+str(hyperParams[1])+"_"+str(hyperParams[2])+"_"+str(hyperParams[3])
                  swarm_output_file=sim['model_output']+"_swarm_"+str(swarm_index)+"_"+str(hyperParams[0])+"_"+str(hyperParams[1])+"_"+str(hyperParams[2])+"_"+str(hyperParams[3])

              else:
                if sim['model_output'][-1] == "_":
                  swarm_model_file = model_file_dir + sim['model_output']+"swarm_"+str(swarm_index)+"_"+str(hyperParams[0])+"_"+str(hyperParams[1])+"_"+str(hyperParams[2])+"_"+str(hyperParams[3])
                  swarm_output_file=sim['model_output']+"swarm_"+str(swarm_index)+"_"+str(hyperParams[0])+"_"+str(hyperParams[1])+"_"+str(hyperParams[2])+"_"+str(hyperParams[3])
                else:
                  swarm_model_file = model_file_dir + sim['model_output']+"_swarm_"+str(swarm_index)+"_"+str(hyperParams[0])+"_"+str(hyperParams[1])+"_"+str(hyperParams[2])+"_"+str(hyperParams[3])
                  swarm_output_file=sim['model_output']+"_swarm_"+str(swarm_index)+"_"+str(hyperParams[0])+"_"+str(hyperParams[1])+"_"+str(hyperParams[2])+"_"+str(hyperParams[3])
            
            
              # if the simulation utilizes the secondary objective method, make a separete output file. 
              if 'secondary_objective' in sim.keys():
                swarm_model_file = str(swarm_model_file)+'_secondary.pkl'
                swarm_output_file= str(swarm_output_file)+'_secondary.pkl'
              else:
                swarm_model_file = str(swarm_model_file)+'.pkl'
                swarm_output_file= str(swarm_output_file)+'.pkl'

              # save the swarm agent file
              save_model(swarm_model_file,S_active,active_reactions,swarmMetabolites,Keq)

              swarmDict = copy.deepcopy(sim)

              swarmName=sim['name']
              if swarmName[-1] == "_":
                swarmName=swarmName+"swarm_"+str(swarm_index)+"_"+str(hyperParams[0])+"_"+str(hyperParams[1])+"_"+str(hyperParams[2])+"_"+str(hyperParams[3])
              else:
                swarmName=swarmName+"_swarm_"+str(swarm_index)+"_"+str(hyperParams[0])+"_"+str(hyperParams[1])+"_"+str(hyperParams[2])+"_"+str(hyperParams[3])

              swarmDict['name'] = swarmName
              swarmDict['model'] = swarm_model_file
              swarmDict['model_output'] = swarm_output_file
              swarmSimulationList.append(swarmDict)
              

      # run all the simulation with a set of hyperparameters 
      output = RunParallelSimulations(swarmSimulationList,model_file_dir,solution_file_dir,hyperparameterDict,configDictionary)


      # determine the best objective value and return a set of best simulations 
      bestSimulations={}
      
      for parentSim in simulationList:
          
          simList=[]
          for swarm_index in range(swarmNumber):
              swarmName=parentSim['name']
              if swarmName[-1] == "_":
                swarmName=swarmName+"swarm_"+str(swarm_index)+"_"+str(hyperParams[0])+"_"+str(hyperParams[1])+"_"+str(hyperParams[2])+"_"+str(hyperParams[3])
              else:
                swarmName=swarmName+"_swarm_"+str(swarm_index)+"_"+str(hyperParams[0])+"_"+str(hyperParams[1])+"_"+str(hyperParams[2])+"_"+str(hyperParams[3])
              status=True
              isSucessful=True
              try:
                # print('Solver status:')
                # print(output[swarmName]['solver_status'])
                if 'ok' == output[swarmName]['solver_status']:
                    status=True
                    if 'secondary_objective' in parentSim.keys():
                      if 'ok' == output[swarmName]['secondary_status']['Solver'][0]['Status']:
                        status=True
                      else:
                        status=False
                else:
                    status=False   
                # print('status:')
                # print(status)         
              except:
                isSucessful=False

              if isSucessful and status:
                
                simList.append({'name':swarmName,'solver_status':status,'objective_value':output[swarmName]['objective_value'],'secondary_status':output[swarmName]['secondary_status'],'secondary_objective_value':output[swarmName]['secondary_objective_value']})
                      
          bestName,bestObj,isFound,averageObjective,countConverged,secondayObj,averagesecondayObjective, convergedSimList,convergedObjs,convergedSecondaryObjs = ReturnBestSimulation(simList)

          if not isFound:
              bestName="none"
              bestObj=0.0
              averageObjective=0.0
              countConverged=0
              secondayObj=0.0
              averagesecondayObjective=0.0
              
          bestSimulations[parentSim['name']]={"swarm_instance_name":bestName,"swarm_instance_value":bestObj,"swarm_secondary_value":secondayObj,"average_objective":averageObjective,"average_secondary_objective":averagesecondayObjective,"number_of_converged_simulations":countConverged, "converged_simulation_list":convergedSimList, "converged_simulation_objectives":convergedObjs, "converged_simulation_secondary_objectives":convergedSecondaryObjs }


      hyperSim={}
      hyperSim['Mb'] = hyperParams[0]
      hyperSim['zeta'] = hyperParams[1]
      hyperSim['delta_concentration'] = hyperParams[2]
      hyperSim['delta_flux'] = hyperParams[3]
      hyperSim['bestSimulations'] = bestSimulations

      bestHyperParameterSimulations.append(hyperSim)


    return bestHyperParameterSimulations
  
'''
Function to post-process simulation results to highlight the best simulation within a swarm set in terms of objective value and provide metrics for 
average objective values across the swarm. Called in RunSwarmSimulation(...)


Inputs:
  - simList: A list of the simulation output dictionaries
             
Returns:
  - bestName: The string name of the simulation that converged with the highest objective value
  - bestObj: The value of the highest objective value swarm agent that converged
  - isFound: Boolean value for whether at least one searm agent converged
  - averageObjective: The average objective value for the converged swarm agents
  - countConverged: The number of swarm agent that converged to a solution
  - secbestObj: The value of the highest converged secondary objective value within a tolerance of the converged primary objective
  - averageSecondaryObjective: The average secondary objective value for converged primary objectives
  - convergedList: A list of simulation swarm agent names that converged to a solution
  - convergedObjs: A list of the associated objective values for converged simulations
  - convergedSecondaryObjs: A list of the associated secondary objective values for converged simulations

'''

def ReturnBestSimulation(simList):
    
    bestName=""
    bestObj=0
    secbestObj=0
    averageObjective=0
    averageSecondaryObjective=0
    isFound=False
    countConverged=0
    convergedList=[]
    convergedObjs=[]
    convergedSecondaryObjs=[]

    for simDict in simList:
        # if the simulation solver run converged
        if simDict['solver_status']:
            countConverged = countConverged+1
            convergedList.append(simDict['name'])
            convergedObjs.append(simDict['objective_value'])
            convergedSecondaryObjs.append(simDict['secondary_objective_value'])
            if not isFound:
                # first sucessful run therefore replace default values
                bestName=simDict['name']
                bestObj=simDict['objective_value']
                secbestObj= simDict['secondary_objective_value']
                isFound=True
            else:
                # test if the new simulation provides a better objective score
                if simDict['objective_value'] > bestObj:
                    bestObj = simDict['objective_value']
                    secbestObj= simDict['secondary_objective_value']
                    bestName=simDict['name']
                    
            averageObjective = averageObjective+simDict['objective_value']
            averageSecondaryObjective = averageSecondaryObjective + simDict['secondary_objective_value']


    if countConverged != 0:
        averageObjective=averageObjective/countConverged
        averageSecondaryObjective=averageSecondaryObjective/countConverged

    return (bestName,bestObj,isFound,averageObjective,countConverged,secbestObj,averageSecondaryObjective,convergedList,convergedObjs,convergedSecondaryObjs)

'''
Function to calculate the rate of change of the variable metabolite concentration due to the optimised reaction flux in the system.
This function accounts for the thermodynamic odds and enzyme regulation of the reactions. 


Inputs:
  - vcounts: the variable metabolite concentration counts
  - fcounts: the fixed metabolite cocnentraiton counts 
  - mu0: the chemical potential for the reactions in the system
  - S: the stoichiometric matrix for the metaboltie and reaction in the system
  - R: the reverse reaction rate for reactions in the system
  - P: the forwards reaction rate for reactions in the system
  - delta: the change in metabolite concentration counts due to reactions in the system
  - Keq: the list of equilibrium concentrations for the reactions in the system
  - E_Regulation: the list of enzyme regulation coefficients for the reactions in the system

Returns:
- deriv: the derivative (rate of concentration change) values

'''

def derivatives(vcounts,fcounts,mu0,S, R, P, delta, Keq, E_Regulation):
  nvar = vcounts.size
  metabolites = np.append(vcounts,fcounts)
  KQ_f = odds(metabolites,mu0,S, R, P, delta, Keq, 1);
  Keq_inverse = np.power(Keq,-1);
  KQ_r = odds(metabolites,mu0,-S, P, R, delta, Keq_inverse, -1);
  deriv = S.T.dot((E_Regulation *(KQ_f - KQ_r)).T);
  #deriv = S.T.dot(((KQ_f - KQ_r)).T);
  deriv = deriv[0:nvar]
  return(deriv.reshape(deriv.size,))

'''
Function to calculate the thermodynamic odds of for a set of reactions.


Inputs:
  - log_counts: the logrithm of the molecule counts for the metabolites in the system
  - mu0: the chemical potentials for the reactions
  - S: the stoichiometric matrix for the reaction and metabolites
  - R: the reverse reaction rate vector
  - P: the forward reaction rate vector
  - delta: the change in chemical concentration counts 
  - K: the equilibrium constants for the reactions
  - direction: the direction of the reactions

Returns:

'''

def odds(log_counts,mu0,S, R, P, delta, K, direction = 1):
  counts = np.exp(log_counts)
  delta_counts = counts+delta;
  log_delta = np.log(delta_counts);
  Q_inv = np.exp(-direction*(R.dot(log_counts) + P.dot(log_delta)))
  KQ = np.multiply(K,Q_inv);
  return(KQ)

'''
Function to calcualte the difference in thermodynamic odds between the forwards and reverse reactions in a system


Inputs:
  - vcounts: the variable metabolite concentration counts
  - fcounts: the fixed metabolite cocnentraiton counts 
  - mu0: the chemical potential for the reactions in the system
  - S: the stoichiometric matrix for the metaboltie and reaction in the system
  - R: the reverse reaction rate for reactions in the system
  - P: the forwards reaction rate for reactions in the system
  - delta: the change in metabolite concentration counts due to reactions in the system
  - Keq: the list of equilibrium concentrations for the reactions in the system
  - E_Regulation: the list of enzyme regulation coefficients for the reactions in the system

Returns:
- KQdiff: a list of thermodynamic odds difference for the reactions

'''

def oddsDiff(vcounts,fcounts,mu0,S, R, P, delta,Keq,E_Regulation):
  metabolites = np.append(vcounts,fcounts)
  KQ_f = odds(metabolites,mu0,S, R, P, delta,Keq);
  Keq_inverse = np.power(Keq,-1);
  KQ_r = odds(metabolites,mu0,-S, P, R, delta,Keq_inverse,-1);
  
  #WARNING: Multiply regulation here, not on individual Keq values.
  KQdiff =  E_Regulation * (KQ_f - KQ_r);
  return(KQdiff)

'''
Function to caluclate a change in the stoichmetric ratios across a reaction to determine the amoutn a metabolite is produced or cosumed


Inputs:
  - vcounts: the variable metabolite concentration counts
  - fcounts: the fixed metabolite cocnentraiton counts 
  - P: the forwards reaction rate for reactions in the system

Returns:
  - delta_S: the list of changes in stoichiometric ratios for metabolites in the system 
'''

def calc_delta_S(vcounts, fcounts, P):
    #WARNING To avoid negative numbers do not use log concs
    metab = np.append(vcounts, fcounts)
    N_avogadro = 6.022140857e+23
    VolCell = 1.0e-15
    Concentration2Count = N_avogadro * VolCell
    target_metab = np.ones(vcounts.size) * np.log(0.001*Concentration2Count);
    target_metab = np.append(target_metab, fcounts)
    delta_S =  P.dot(metab) - P.dot(target_metab)
    return(delta_S)

'''
Function to calculate the change in enxyme concentration during a simulation step. 


Inputs:
  - E: the list of enzyme concentrations
  - vcounts: the variable metabolite concentration counts
  - fcounts: the fixed metabolite cocnentraiton counts 

Returns:
  - newE: the list of enzyme concentrations post simulation step
'''

def calc_E_step(E, vcounts, fcounts):
    newE = E -E/2
    return(newE)

'''
Function to reduce the stoichiometric matrix into the subsets for variable and fixed metabolites only and calculate a null space basis
for the variable metabolite stoichiometric matrix 


Inputs:

Returns:
  - S_v: the submatrix for the variable metabolites
  - S_f: the submatrix for the fixed metabolites
  - Sv_N: the null space basis for the variable metabolite matrix

'''

def reduced_Stoich(S,f_log_counts,target_log_vcounts):

  # Flip Stoichiometric Matrix
  S_T = S    # S_T is the Stoich matrix with rows as reactions, columns as metabolites
  S = np.transpose(S_T) # this now is the Stoich matrix with rows metabolites, and columns reactions

  #Set the System parameters 
  VarM = target_log_vcounts
  FxdM = np.reshape(f_log_counts,(len(f_log_counts),1) )
  n_M = len(VarM) + len(FxdM)

  #Metabolite parms
  n_M_f = len(f_log_counts)   #total number of fixed metabolites
  n_M_v = len(target_log_vcounts)   #total number of variable metabolites
  n_M = n_M_f + n_M_v   # total number of metabolites

  VarM_idx = np.arange(0,n_M_v) #variable metabolite indices
  FxdM_idx = np.arange(n_M_v,n_M) # fixed metabolite indices

  # Split S into the component corresponding to the variable metabolites S_v 
  # and the component corresponding to the fixed metabolites S_f
  S_v_T = np.delete(S_T,FxdM_idx,1)
  S_v = np.transpose(S_v_T)

  S_f_T = np.delete(S_T,VarM_idx,1)
  S_f = np.transpose(S_f_T)

  #find a basis for the nullspace of S_v
  Sv_N = spL.null_space(S_v)

  return S_v, S_f, Sv_N



'''
Function to load a model from a pickle file located at the model file name. 


Inputs:
  - model_file: the string name for the model 

Returns:
  - S_active: the stoichiometric dataframe describing the stoichiometrix ratios for the change in metabolites due to each reaction
  - active_reactions: the reaction dataframe for the model
  - metabolites: the metabolite dataframe for the model 
  - Keq:  the thermodynamic equilibrium constants for the reaction in the model 
          (this data may also be a column of the active_reactions dataframe)

'''

def load_model(model_file):

  with open(model_file, 'rb') as handle:
      model = pickle.load(handle)

  S_active = pd.DataFrame(model['S'])
  active_reactions = pd.DataFrame(model['active_reactions'])
  metabolites = pd.DataFrame(model['metabolites'])

  if 'KQ' in active_reactions.columns:
    Keq = active_reactions['KQ'].tolist()
  elif 'Keq' in active_reactions.columns:
    Keq = active_reactions['Keq'].tolist()

  else:
    Keq = model['Keq']

  return S_active, active_reactions, metabolites, Keq

'''
Function to save a model after modification. This will also save each swarm agent as a different model set.


Inputs:
  - model_file: the file name to save the model collective to as a pickle file
  - S_active: the stoichiometric dataframe describing the stoichiometrix ratios for the change in metabolites due to each reaction
  - active_reactions: the reaction dataframe for the model
  - metabolites: the metabolite dataframe for the model 
  - Keq:  the thermodynamic equilibrium constants for the reaction in the model 
          (this data may also be a column of the active_reactions dataframe)

Returns:
  - null

'''

def save_model(model_file,S_active,active_reactions,metabolites,Keq=[]):

  model={}
  model['S']=S_active.to_dict()
  model['active_reactions']=active_reactions.to_dict()
  model['metabolites']=metabolites.to_dict()

  if 'KQ' not in active_reactions.columns or 'Keq' not in active_reactions.columns:
    model['Keq']=Keq


  with open(model_file, 'wb') as handle:
      pickle.dump(model,handle)

  return


'''
Function to save the optimization solution output as a pickle file. Input the solution data from the optimization procedure.


Inputs:
  - save_file: the string file name for where to save the model/the name of the model output
  - y_sol: the solution vector for the flux through the reactions
  - alpha_sol: the solution vector for the enzymatic regulation coefficients [0,1] for each reaction in the model
  - n_sol: the solution vector for the metabolite concentration counts at steady state
  - S: the stoichiometrix matrix for the model
  - Keq: the vector of reaction equilibrium constants
  - f_log_counts: the vector of the logarithms of the metabolite concentrations
  - vcount_lower_bound: the vector of upper bounds of counts for the metabolite concentrations
  - vcount_upper_bound: the vector of lower bounds of counts for the metabolite concentrations

Returns:
- null

'''

def save_model_solution(save_file,y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound,opt_parms={}):
  ### Save a dictionary of the solution for later analysis (note this also saves the associated model file for future reference)

  solution_data = {}

  #Solution values
  solution_data['y'] = y_sol
  solution_data['alpha'] = alpha_sol
  solution_data['n'] = n_sol
  solution_data['unreg_rxn_flux']=unreg_rxn_flux


  #Optimization parameters
  solution_data['f_log_counts'] = f_log_counts
  solution_data['vcount_lower_bound'] = vcount_lower_bound
  solution_data['vcount_upper_bound'] = vcount_upper_bound

  #Model files
  solution_data['S'] = S
  solution_data['Keq'] = Keq

  solution_data['opt_parms'] = opt_parms

  with open(save_file, 'wb') as handle:
      pickle.dump(solution_data, handle, protocol=4)

  return 

'''
Function to load a model solution from a pickle file.


Inputs:
  - load_file: the string name for the pickle file that conatins the solution output

Returns:
  - y_sol: the solution vector for the flux through the reactions
  - alpha_sol: the solution vector for the enzymatic regulation coefficients [0,1] for each reaction in the model
  - n_sol: the solution vector for the metabolite concentration counts at steady state
  - S: the stoichiometrix matrix for the model
  - Keq: the vector of reaction equilibrium constants
  - f_log_counts: the vector of the logarithms of the metabolite concentrations
  - vcount_lower_bound: the vector of upper bounds of counts for the metabolite concentrations
  - vcount_upper_bound: the vector of lower bounds of counts for the metabolite concentrations

'''

def load_model_solution(load_file):

  with open(load_file, 'rb') as handle:
      solution_data = pickle.load(handle)

  #Solution values
  y_sol = solution_data['y']
  alpha_sol = solution_data['alpha']
  n_sol = solution_data['n']
  unreg_rxn_flux = solution_data['unreg_rxn_flux']

  #Optimization parameters
  f_log_counts = solution_data['f_log_counts']
  vcount_lower_bound = solution_data['vcount_lower_bound']
  vcount_upper_bound = solution_data['vcount_upper_bound']

  #Model files
  S = solution_data['S']
  Keq = solution_data['Keq']

  return y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound

def load_model_solution_opts(load_file):

  with open(load_file, 'rb') as handle:
      solution_data = pickle.load(handle)

  #Solution values
  y_sol = solution_data['y']
  alpha_sol = solution_data['alpha']
  n_sol = solution_data['n']
  unreg_rxn_flux = solution_data['unreg_rxn_flux']

  #Optimization parameters
  f_log_counts = solution_data['f_log_counts']
  vcount_lower_bound = solution_data['vcount_lower_bound']
  vcount_upper_bound = solution_data['vcount_upper_bound']
  opt_parms = solution_data['opt_parms']

  #Model files
  S = solution_data['S']
  Keq = solution_data['Keq']

  return y_sol,alpha_sol,n_sol,unreg_rxn_flux,S,Keq,f_log_counts,vcount_lower_bound,vcount_upper_bound,opt_parms

'''
Function to save the initial inputs and parameters for optimization as a pickle file. 


Inputs:
  - save_file: the string file name for where to save the model/the name of the model output
  - y_ini: the initial solution vector for the flux through the reactions
  - n_ini: the initial solution vector for the metabolite concentration counts at steady state
  - opt_parms: the input optimization parameters for the model

Returns:
- null

'''

def save_model_initial(save_file,y_ini,n_ini,opt_parms):
  ### Save a dictionary of the solution for later analysis (note this also saves the associated model file for future reference)

  solution_data = {}

  #Solution values
  solution_data['y'] = y_ini
  solution_data['n'] = n_ini


  #Optimization parameters
  solution_data['opt_parms'] = opt_parms

  with open(save_file, 'wb') as handle:
      pickle.dump(solution_data, handle, protocol=4)

  return 

'''
Function to load a model initial solution from a pickle file.


Inputs:
  - load_file: the string name for the pickle file that conatins the solution output

Returns:
  - y_ini: the initial solution vector for the flux through the reactions
  - n_ini: the initial solution vector for the metabolite concentration counts at steady state
  - opt_parms: the input optimization parameters for the model


'''

def load_model_initial(load_file):

  with open(load_file, 'rb') as handle:
      solution_data = pickle.load(handle)

  #Solution values
  y_ini = solution_data['y']
  n_ini = solution_data['n']


  #Optimization parameters
  opt_parms = solution_data['opt_parms']

  return y_ini,n_ini,opt_parms



'''
Function to calculate the gradient in the flux through the objective reactions.


Inputs:
  - Sv_N: the null space basis for the variable metabolite matrix
  - y_obj_coefs: the flux objective coefficients for the reactions 

Returns:
  - y_grad: the vector of flux through the objective reactions 

'''

def flux_gradient(Sv_N,y_obj_coefs):

  n_react = len(np.ravel(y_obj_coefs))
  y_obj_coefs = np.reshape(y_obj_coefs,(1,n_react))
  beta_grad = np.matmul(y_obj_coefs,Sv_N)

  beta_grad = np.transpose(beta_grad)


  y_grad = np.matmul(Sv_N,beta_grad)

  return y_grad

'''
Function to report the subset of names for the fixed metabolites with their fixed log values


Inputs:
  - f_log_counts: the fixed metabolite log counts
  - metabolites: the metabolite dataframe 

Returns:
  - logDict: a dictionary with fixed metabolite names as keys and the fixed log concentration counts as values

'''

def f_log_counts_names(f_log_counts,metabolites):

  # convert the fixed log counts into a named dictionary for processing  

  logDict={}

  nvar = len(metabolites[ metabolites['Variable']==True].values)

  fixedNum=len(f_log_counts)

  for i in range(fixedNum):
    logDict[metabolites.iloc[nvar+i].name] = f_log_counts[i]


  return logDict

'''
Function to modify the metabolites dataframe to change a single chemical species from being variable to fixed. 


Inputs:
  - metabolites: the dataframe of metabolites in the model
  - metabName: the string name for the variable metabolite in the model

Returns:
  - metabolites: the modified dataframe of metabolites in the model 

'''

def FixChemicalSpecies(metabolites,metabName):


  # check that the species is not already fixed
  metabolites_list = list(metabolites[metabolites['Variable']==False].index)
  if metabName not in metabolites_list:

    # need to make the metabolite a variable
    dndt = metabolites.loc[metabName,'dn/dt']
    Conc = metabolites.loc[metabName,'Conc']
    targetConc = metabolites.loc[metabName,'Target Conc']
    netFlux = metabolites.loc[metabName,'Net_Flux']
    upperBound = metabolites.loc[metabName,'Upper_Bound']
    lowerBound = metabolites.loc[metabName,'Lower_Bound']
    
    # remove the entry for the variable metabolite and add a fixed metabolite version with updated conc
    metabolites.drop(metabName,inplace=True)
    newMetabolite = {'Conc':Conc, 'Variable':False, 'Target Conc':targetConc,'Steady_State': 'Implicit','dn/dt':dndt,'Net_Flux':netFlux,'Upper_Bound':upperBound,'Lower_Bound':lowerBound}
    Name = [metabName]
    df = pd.DataFrame(newMetabolite,index = Name)
    metabolites = pd.concat([metabolites,df]) # add the new converted metabolite to the metabolites dataframe
  else:
    # if the metabolite is already fixed in the model, ensure the metabolite is set to 'Implicit"
    metabolites.at[metabName,'Steady_State'] = 'Implicit'


  return metabolites

'''
Function to modify the metabolites dataframe to change a single chemical species from being fixed to variable. 


Inputs:
  - metabolites: the dataframe of metabolites in the model
  - metabName: the string name for the fixed metabolite in the model

Returns:
  - metabolites: the modified dataframe of metabolites in the model 

'''

def RelaxChemicalSpecies(metabolites,metabName):

  # check tht the species is not already variable
  metabolites_list = list(metabolites[metabolites['Variable']==True].index)
  if metabName not in metabolites_list:

    # need to make the metabolite a variable
    dndt = metabolites.loc[metabName,'dn/dt']
    Conc = metabolites.loc[metabName,'Conc']
    targetConc = metabolites.loc[metabName,'Target Conc']
    netFlux = metabolites.loc[metabName,'Net_Flux']
    upperBound = metabolites.loc[metabName,'Upper_Bound']
    lowerBound = metabolites.loc[metabName,'Lower_Bound']
    
    # remove the entry for the variable metabolite and add a fixed metabolite version with updated conc
    metabolites.drop(metabName,inplace=True)
    newMetabolite = {'Conc':Conc, 'Variable':True, 'Target Conc':targetConc,'Steady_State': 'Explicit','dn/dt':dndt,'Net_Flux':netFlux,'Upper_Bound':upperBound,'Lower_Bound':lowerBound}
    Name = [metabName]
    df = pd.DataFrame(newMetabolite,index = Name)
    metabolites = pd.concat([metabolites,df]) # add the new converted metabolite to the metabolites dataframe

    # reorder dataframe to put the new variable metabolite in the correct order
    metabolites.sort_values(by = ['Variable','Steady_State'],ascending = [False,True],inplace = True)

  else:
    # if the metabolite is already fixed in the model, ensure the metabolite is set to 'Explicit"
    metabolites.at[metabName,'Steady_State'] = 'Explicit'


  return metabolites


'''
Function to change the value of a fixed metabolite such as a metabolite used as a boundary condition. 


Inputs:
- metab: the string name for the fixed metabolite to change the fixed concentration value
- new_metab_value: the concentration value to set the metabolite as
- metabolites: the dataframe containing the metabolites data
- f_log_counts: the vector of log counts for the fixed metabolites  

Returns:
- f_log_counts: the vector of log counts for the fixed metabolites 

'''

def ChangeFixedBoundaryConditionValue(metab,new_metab_value,metabolites,f_log_counts):
   
  # changed the concentration of a selected (already fixed) metabolite and output the new f_log_counts

  # list of the fixed metabolite names
  metabolites_list = list(metabolites[metabolites['Variable']==False].index)

  # new_f_log_counts=f_log_counts

  # determine index of selected metabolite
  metab_idx = np.where(np.array(metabolites_list) == metab)[0][0]

  # calcualte new log count 
  N_avogadro = 6.022140857e+23
  VolCell = 1.0e-15
  Concentration2Count = N_avogadro * VolCell
  
  f_log_counts[metab_idx] = np.log(new_metab_value*Concentration2Count)

  return f_log_counts


'''
Function to change the state and the value of a variable metabolite to a fixed metabolite at a new value
such as a metabolite used as a boundary condition. 


Inputs:
- metab: the string name for the variable metabolite to fix
- new_metab_value: the concentration value to set the metabolite as
- metabolites: the dataframe containing the metabolites data

Returns:
- metabolites: the modified dataframe containing the metabolites data

'''

def ChangeBoundaryCondition(metab,new_metab_value,metabolites):
    
  # boundary conditions need to be fixed, so test whether the selected metabolite is already fixed
  # if not, fix and then update to new value

    # test if metabolite is in the metabolite list
    if metab in metabolites.index:
        # test if metabolite is variable or fixed
        if metabolites.loc[metab,'Variable']:
            # metabolite is variable
            # print("Metabolite "+str(metab)+" is variable, fix to new concentration "+str(new_metab_value)+" \n")
            dndt = metabolites.loc[metab,'dn/dt']
            targetConc = metabolites.loc[metab,'Target Conc']
            netFlux = metabolites.loc[metab,'Net_Flux']
            upperBound = metabolites.loc[metab,'Upper_Bound']
            lowerBound = metabolites.loc[metab,'Lower_Bound']
            # remove the entry for the variable metabolite and add a fixed metabolite version with updated conc
            metabolites.drop(metab,inplace=True)
            newMetabolite = {'Conc':new_metab_value, 'Variable':False, 'Target Conc':targetConc,'Steady_State': 'Implicit','dn/dt':dndt,'Net_Flux':netFlux,'Upper_Bound':upperBound,'Lower_Bound':lowerBound}
    
            Name = [metab]
            df = pd.DataFrame(newMetabolite,index = Name)
            metabolites = pd.concat([metabolites,df]) # add the new converted metabolite to the metabolites dataframe
        
        else:
            # metabolite is fixed already, chnage the value and set to implicit steady state
            metabolites.at[metab,'Conc'] = new_metab_value
            metabolites.at[metab,'Steady_State'] = 'Implicit'
    
    else:
        print("Metabolite "+str(metab)+ " : is not in the model.")
        
    # output the updated metabolites dataframe
    return metabolites

'''
Function to remove a metabolite from the metabolitec concentration dataframe and the stoichiometrix dataframe

Inputs: 
  - metab: metabolite name that exists in the model
  - metabolties: the dataframe of metabolite in the model
  - S-active: the dataframe of the stoichiometric matrix for metabolites and reactions in the model

Returns:
  - metabolites: the modified dataframe containing the metabolites data
  - S_active: the modifies stoichiometric matrix

'''

def RemoveMetabolite(metab,metabolites,S_active):
   
  # remove a given metabolite from the model; drop from metabolite and stoich dataframe
  metabolites.drop(metab,inplace=True)
  S_active.drop(columns = metab,inplace = True)

  return metabolites,S_active

'''
Function to remove a reaction from the model reaction dataframe and the stoichiometrix dataframe

Inputs: 
  - reactionName: reaction name that exists in the model
  - active_reactions: the dataframe of reactions in the model
  - S-active: the dataframe of the stoichiometric matrix for metabolites and reactions in the model

Returns:
  - active_reactions: the modified dataframe containing reaction information
  - S_active: the modifies stoichiometric matrix

'''

def RemoveReaction(reactionName,active_reactions,S_active):

  # remove a given metabolite from the model; drop from active_reactions and stoich dataframe
  # print("Removing reaction: "+str(reactionName))
  S_active.drop(reactionName,inplace = True)

  active_reactions.drop(reactionName,inplace = True)

  return active_reactions,S_active 


'''
Function to check whether a list of metabolites are present in the model and produce whether the metabolite 
is variable or fixed.


Inputs:
  - model_file_dir: the string directory path containg the model file
  - file_name: the string name of the pickel file containing the model 
  - metaboliteList: the list of string metabolite names to check for in the model 

Returns:
  - null

'''

def CheckMetabolitesInModel(model_file_dir,file_name,metaboliteList):
    
    if model_file_dir in file_name:
       
      S_active,active_reactions, metabolites, Keq= load_model(file_name)

    else:
       
      S_active,active_reactions, metabolites, Keq= load_model(model_file_dir + '/' + file_name)
    
    
    # print("Checking whether the target metabolites list is in the model:\n")
    
    for metab in metaboliteList:
        try:
            print(str(metab)+" : Variable = "+str(metabolites.loc[[metab]]['Variable'][0]))
            print()
        except:
            print("Metabolite "+str(metab)+" not in model\n")
    print()
    return

'''
Function to print the metabolites in a model alongside whether they are variable or fixed


Inputs:
  - model_file_dir: the string directory path containg the model file
  - file_name: the string file name for the model pickle file

Returns:
  - null

'''

def PrintMetabolitesInModel(model_file_dir,file_name):
    
    if model_file_dir in file_name:
       
      S_active,active_reactions, metabolites, Keq= load_model(file_name)

    else:
       
      S_active,active_reactions, metabolites, Keq= load_model(model_file_dir + '/' + file_name)
    
    for metab in sorted(metabolites.index.tolist()):
      try:
            print(str(metab)+" : Variable = "+str(metabolites.loc[metab,'Variable'][0]))
            print()
      except:
            print("Metabolite "+str(metab)+" not in model\n")

    return

'''
Function to check whether a list of reactions are present in the model.


Inputs:
  - model_file_dir: the string directory path containg the model file
  - file_name: the string name of the pickel file containing the model 
  - reactionList: the list of string reaction names to check for in the model 

Returns:
  - null

'''

def CheckReactionsInModel(model_file_dir,file_name,reactionList):
    
    if model_file_dir in file_name:
       
      S_active,active_reactions, metabolites, Keq= load_model(file_name)

    else:
       
      S_active,active_reactions, metabolites, Keq= load_model(model_file_dir + '/' + file_name)
    
    print("Checking whether the target reaction list is in the model:\n")
    
    for rxn in reactionList:
        try:
            print(str(rxn)+" : "+str(active_reactions.loc[rxn,'Full Rxn']))
            print()
        except:
            print("Reaction "+str(rxn)+" not in model\n")
    print()
    return

'''
Function to convert the best simulations calculated from the simulation output produce in the RunSimulation(...) function


Inputs:
  - bestSimulationSet: a list of the best simulation outputs for the agent swarm for each parameter set in the simualtion experiments

Returns:
  - bestSimulationsdfArr: a dataframe containing the best values for a simulation experiment.

'''

def ConvertBestSimulationsToDataFrameHyperparameter(bestSimulationSet):
    

    bestSimulationsdfArr=[]

    for parameterRun in bestSimulationSet:
      bestSimulations = parameterRun['bestSimulations']

      simNameArr=[]
      swarm_instance_nameArr=[]
      swarm_instance_valueArr=[]
      swarm_instance_secondArr=[]
      average_objectiveArr=[]
      average_secondaryArr=[]
      number_of_converged_simulationsArr=[]
      converged_simulation_list=[]
      convergedObjs=[]
      convergedSecondaryObjs=[]


      for sim in bestSimulations:

          simNameArr.append(sim)
          swarm_instance_nameArr.append(bestSimulations[sim]['swarm_instance_name'])
          swarm_instance_valueArr.append(bestSimulations[sim]['swarm_instance_value'])
          swarm_instance_secondArr.append(bestSimulations[sim]['swarm_secondary_value'])
          average_objectiveArr.append(bestSimulations[sim]['average_objective'])
          average_secondaryArr.append(bestSimulations[sim]['average_secondary_objective'])
          number_of_converged_simulationsArr.append(bestSimulations[sim]['number_of_converged_simulations'])
          converged_simulation_list.append(bestSimulations[sim]['converged_simulation_list'])
          convergedObjs.append(bestSimulations[sim]['converged_simulation_objectives'])
          convergedSecondaryObjs.append(bestSimulations[sim]['converged_simulation_secondary_objectives'])

      bestSimulationsdf = pd.DataFrame({'swarm_instance_name':swarm_instance_nameArr,'swarm_instance_value':swarm_instance_valueArr,'swarm_secondary_value':swarm_instance_secondArr,'average_objective':average_objectiveArr,'average_secondary_objective':average_secondaryArr,'number_of_converged_simulations':number_of_converged_simulationsArr,'converged_simulation_list':converged_simulation_list,'converged_simulation_objectives':convergedObjs,'converged_simulation_secondary_objectives':convergedSecondaryObjs},index=simNameArr)
      bestSimulationsdfArr.append(bestSimulationsdf)
    return bestSimulationsdfArr

'''
Function to parse the reactions in the model and subset of reaction type. The three reaction subsets are; 'environment reactions'
where the reaction occurs in a designated environment compartment, 'uptake reactions' where the reaction occurs in the environment
on one side and a different compartment on the other, and 'transport reaction' where the reaction occurs in two different
compartments where neither is the environment. 


Inputs:
  - model_file_dir: the string directory path containg the model file
  - file_name: the string name of the pickel file containing the model 
  - environmentComp: the string name of the designated environment compartment (default 'ENVIRONMENT')
  - verbose: switch case {0,1} for the boolean logic of whether to explicitly print out the reactions subset (default 1)

Returns:
  - environmentMetabolites: a list of the name strings for metabolites in the environment compartment 
  - uptakeReactions: a list of the string names for the 'uptake' reactions
  - transportReactions: a list of the string names for the 'transport' reactions
  - environmentReactions: a list of the string names for the 'environment' reactions

'''

def ReportModelEnvironment(model_file_dir,file_name,environmentComp = 'ENVIRONMENT',verbose=1):
    
    if model_file_dir in file_name:
       
      S_active,active_reactions, metabolites, Keq= load_model(file_name)

    else:
       
      S_active,active_reactions, metabolites, Keq= load_model(model_file_dir + '/' + file_name)

    environmentMetabolites = ParseMetaboliteTypes(metabolites,environmentComp)

    if verbose:
      print("Environment Metabolites:\n")
      if len(environmentMetabolites) == 0:
          print("No environment metabolites found: check target compartment, currently looking for "+str(environmentComp)+"\n")
      else:
          for i in environmentMetabolites:
              print(i)
          print()
        
    uptakeReactions,transportReactions,environmentReactions = ParseReactionTypes(active_reactions,environmentComp)
    
    if verbose:
      print("Uptake Reactions:\n")
      
      if len(uptakeReactions) == 0:
          print("No uptake reactions found\n")
      else:

          for i in uptakeReactions:
              print(str(i) + " : "+active_reactions.loc[i,'Full Rxn'])
          print()
    
      print("Transport Reactions:\n")
      if len(transportReactions) == 0:
          print("No transport reactions found\n")
      else:
          for i in transportReactions:
              print(str(i) + " : "+active_reactions.loc[i,'Full Rxn'])
          print()
    
    
      print("Environment Reactions:\n")
      if len(environmentReactions) == 0:
          print("No environment reactions found\n")
      else:
          for i in environmentReactions:
              print(str(i) + " : "+active_reactions.loc[i,'Full Rxn'])
          print()
    
    
    return environmentMetabolites,uptakeReactions,transportReactions,environmentReactions

'''
Function to parse the metabolites in a model to extract the metabolites in a designated environment compartment. 


Inputs:
  - metabolties: the dataframe of metabolite in the model
  - environmentComp: the string name of the designated environment compartment (default 'ENVIRONMENT')

Returns:
  - environmentMetabolites: a list of the environment metabolite names

'''

def ParseMetaboliteTypes(metabolites,environmentComp = 'ENVIRONMENT'):
    
    environmentMetabolites=[]
    for metab in metabolites.index:
        try:
            comp = metab.rsplit(':',1)[1]

            if environmentComp in comp:
                environmentMetabolites.append(metab)
        except:
            a=1
    return environmentMetabolites

'''
Parse a reaction dataframe to determine the types of reactions in the model.
If the compartment name for the metabolites is different on either side then the reaction
is classed as a "transport" reaction. If only one side of the reaction occured in the environment
compartment then class the reaction as an "uptake" reaction. If both sides of the reaction occur
within the environment compartment then class the reaction as an "environment" reaction.

Inputs: 
  - active_reactions: the dataframe for the reaction information
  - environmentComp: a string label for the designated environment compartment (default 'ENVIRONMENT')

Returns:
  - uptakeReactions: a list of the string names for the 'uptake' reactions
  - transportReactions: a list of the string names for the 'transport' reactions
  - environmentReactions: a list of the string names for the 'environment' reactions 

'''  

def ParseReactionTypes(active_reactions,environmentComp = 'ENVIRONMENT'):
    uptakeReactions=[]
    environmentReactions=[]
    transportReactions=[]
    for i in active_reactions.index:
        rxnString=active_reactions.loc[[i]]['Full Rxn'][0]

        [leftString,rightString] = rxnString.split(" = ")
        substrates = leftString.split(" + ")
        products = rightString.split(" + ")

        leftCompartments = []
        rightCompartments = []
        for metab in substrates:
            try:
                comp = metab.rsplit(':',1)[1]
                if comp not in leftCompartments:
                    leftCompartments.append(comp) 
            except:
                a=1
        for metab in products:
            try:
                comp = metab.rsplit(':',1)[1]
                if comp not in rightCompartments:
                    rightCompartments.append(comp) 
            except:
                a=1

        leftCompartments = sorted(leftCompartments)
        rightCompartments = sorted(rightCompartments)

        if leftCompartments != rightCompartments:
            if environmentComp in leftCompartments and environmentComp in rightCompartments:
                environmentReactions.append(i)
            elif environmentComp in leftCompartments or environmentComp in rightCompartments:
                uptakeReactions.append(i)
            else:
                transportReactions.append(i)
    return uptakeReactions,transportReactions,environmentReactions


'''
Scan over a range of redox ratio for a given redox pair.
Modify the simulation dictionary by adding a new redox pair and assiciated value from the ratio list.

Inputs: 
  - simulation_list: the list of simulation dictionaries to scan over to change the initial conditions 
  - redox_pair_list: a list for the redox pair i.e [redox_A,redox_B]
  - ratio_list: the list of pair ratio values to scan over. The resulting simulations will be 
                duplicated for each value

Returns: 
  - simulation_list_new: The output list of simulation dictionaries 

'''

def ScanOverRedoxRatios(simulation_list,redox_pair_list, ratio_list):
    
    simulation_list_new=[]
    if len(redox_pair_list)==2:
    
        for parentSim in simulation_list:
            if 'redox_ratios' in parentSim.keys():
                if type(parentSim['redox_ratios']) is not list:
                    if type(parentSim['redox_ratios']) is dict:
                        parentSim['redox_ratios'] = [parentSim['redox_ratios']]
                    else:
                        parentSim['redox_ratios'] =[]
            else:
                parentSim['redox_ratios'] =[]
                
            for ratio in ratio_list:
                redoxPair = {'names':redox_pair_list,'ratio':ratio}
                
                newSim=copy.deepcopy(parentSim)

                newSim['redox_ratios'].append(redoxPair) 
     
                if newSim['name'][-1] == "_":
                    newSim['name'] = newSim['name']+str(redox_pair_list[0])+"_"+str(redox_pair_list[1])+"_"+str(ratio)+"_"
                else:
                    newSim['name'] = newSim['name']+"_"+str(redox_pair_list[0])+"_"+str(redox_pair_list[1])+"_"+str(ratio)+"_"
                if newSim['model_output'][-1] == "_":
                    newSim['model_output'] = newSim['model_output']+str(redox_pair_list[0])+"_"+str(redox_pair_list[1])+"_"+str(ratio)+"_"
                else:
                    newSim['model_output'] = newSim['model_output']+"_"+str(redox_pair_list[0])+"_"+str(redox_pair_list[1])+"_"+str(ratio)+"_"

                newSim['model_output'] = newSim['model_output'].replace(":","_")  
                newSim['model_output'] = newSim['model_output'].replace(" ","_")   
                simulation_list_new.append(newSim)

        return simulation_list_new
    
    else:
        print("Input redox pair needs to be a list of two chemical names, skip redox scan.")
        return simulation_list


'''
Scan over a range of initial concentrations for a given chemical.

Inputs: 
  - simulation_list: the list of simulation dictionaries to scan over to change the initial conditions 
  - chemical_name: a string name for the metabolite to scan over initial conditions
  - value_list: the list of initial condition values to scan over. The resulting simulations will be 
                duplicated for each value

Returns: 
  - simulation_list_new: The output list of simulation dictionaries 

'''

def ScanOverInitialCondition(simulation_list,chemical_name,value_list):
    
    simulation_list_new=[]
    
    for parentSim in simulation_list:
        for val in value_list:
            condition = {chemical_name:val}
        
            newSim=copy.deepcopy(parentSim)
            newSim['initial_conditions'] = newSim['initial_conditions'] | condition
            
            if newSim['name'][-1] == "_":
                newSim['name'] = newSim['name']+str(chemical_name)+"_"+str(val)+"_"
            else:
                newSim['name'] = newSim['name']+"_"+str(chemical_name)+"_"+str(val)+"_"
            if newSim['model_output'][-1] == "_":
                newSim['model_output'] = newSim['model_output']+str(chemical_name)+"_"+str(val)+"_"
            else:
                newSim['model_output'] = newSim['model_output']+"_"+str(chemical_name)+"_"+str(val)+"_"

            newSim['model_output'] = newSim['model_output'].replace(":","_")  
            newSim['model_output'] = newSim['model_output'].replace(" ","_") 
            simulation_list_new.append(newSim)
    
    return simulation_list_new

'''
Scan over a range of boundary concentrations for a given chemical.

Inputs: 
  - simulation_list: the list of simulation dictionaries to scan over to change the boundary conditions 
  - chemical_name: a string name for the metabolite to scan over boundary conditions
  - value_list: the list of boundary condition values to scan over. The resulting simulations will be 
                duplicated for each value

Returns: 
  - simulation_list_new: The output list of simulation dictionaries 

'''

def ScanOverBoundaryCondition(simulation_list,chemical_name,value_list):
    
    simulation_list_new=[]
    
    for parentSim in simulation_list:
        for val in value_list:
            condition = {chemical_name:val}
        
            newSim=copy.deepcopy(parentSim)
            newSim['boundary_conditions'] = newSim['boundary_conditions'] | condition
            
            if newSim['name'][-1] == "_":
                newSim['name'] = newSim['name']+str(chemical_name)+"_"+str(val)+"_"
            else:
                newSim['name'] = newSim['name']+"_"+str(chemical_name)+"_"+str(val)+"_"
            if newSim['model_output'][-1] == "_":
                newSim['model_output'] = newSim['model_output']+str(chemical_name)+"_"+str(val)+"_"
            else:
                newSim['model_output'] = newSim['model_output']+"_"+str(chemical_name)+"_"+str(val)+"_"

            newSim['model_output'] = newSim['model_output'].replace(":","_")  
            newSim['model_output'] = newSim['model_output'].replace(" ","_") 
            simulation_list_new.append(newSim)
    
    return simulation_list_new  


def Evaluate_Stoich(S_active,active_reactions,metabolites):
  

   #SET COPIES OF ALL INPUT DATAFRAMES THAT WILL BE INTERNALLY MANIPULATED

    metabolites_df = metabolites.copy()
    active_reactions_df = active_reactions.copy()
    S_active_df = S_active.copy()


    ''' 
    #################################
    RE-ORDER OF METABOLITES
    #################################

    This is done to making splitting the stoichiometric matrix cleaner for computing the initial solution estimate for the optimization.


    '''
    
    metabolites_df.sort_values(by = ['Variable','Steady_State'],ascending = [False,True],inplace = True)

    #ensure that S is sorted the same way
    metabolite_name_order = list(metabolites_df.index)
    S_active_df = S_active_df[metabolite_name_order]


    '''
    Ensure that S_active and active_reactions are in the same order
    '''
    active_reactions_df.reindex(list(S_active_df.index))




    ''' 
    #################################
    CONSTRUCT THE INDEXES FOR EACH metabolite and flux type
    #################################
    '''

    ### METABOLITE INDEXES

    metabolites_df.reset_index(inplace = True) #set a reference index for all of the metabolites, order must be preserved from this point forward, including Stoich matrix

   

    m_tot_idx = np.array(metabolites_df.index)
    m_var_idx = np.array(metabolites_df[(metabolites_df['Variable']==True)].index)
    m_implicit_idx = np.array(metabolites_df[(metabolites_df['Steady_State']=='Implicit')].index)
    m_var_explicit_idx = np.array(metabolites_df[(metabolites_df['Variable']==True) & (metabolites_df['Steady_State'] == 'Explicit' )].index)
    m_var_implicit_idx = np.array(metabolites_df[(metabolites_df['Variable']==True) & (metabolites_df['Steady_State'] == 'Implicit' )].index)
    m_fixed_idx = np.array(metabolites_df[(metabolites_df['Variable']==False)].index)

    ## check for consistency
    if len(m_var_explicit_idx) + len(m_var_implicit_idx) + len(m_fixed_idx) != len(m_tot_idx): print('Metabolite Assignments Are Inconsistent!!!')





    ''' 
    Get the equilibrium constants
    '''

    Keq = active_reactions_df['Keq'].values
    Keq = np.reshape(Keq,(len(Keq),1))





    '''
    Stoichiometric Matrix
    '''
    S = S_active_df.values



    S_T = S    # S_T is the Stoich matrix with rows as reactions, columns as metabolites
    S = np.transpose(S_T) # this now is the Stoich matrix with rows metabolites, and columns reactions

    n_react = S_T.shape[0]

    #Find submatrix corresponding to the explicit steady state metabolites, these define our steady state conditions
    S_e_T = np.delete(S_T,m_implicit_idx,1)
    S_e = np.transpose(S_e_T)


    #find a basis for the nullspace of S_e, this defines our set of steady state flux solutions that are feasible.
    Se_N = spL.null_space(S_e)
    dSe_N = np.shape(Se_N)[1] # the dimension of the nullspace

    S_f_T = np.delete(S_T,m_var_idx,1)
    S_f = np.transpose(S_f_T)

    S_m_T = np.delete(S_T,m_var_explicit_idx,1)
    S_m = np.transpose(S_m_T)

    S_v_T = np.delete(S_T,m_fixed_idx,1)
    S_v = np.transpose(S_v_T)


    return S_v, S_f,S_e, S_m, Se_N, S_active_df,active_reactions_df,metabolites_df

def UpdateModelForEnvironmentChange(new_model_filename,envMetabs,S,R,M,K):
    
    for i in envMetabs:
        M.at[i,'Conc'] = envMetabs[i]

    save_model(new_model_filename,S,R,M,K)
        
    return

def ApplyEnvironmentChangeToModel(new_model_filename,old_model_filename,envMetabs):
        
    S, R, M, K = load_model(old_model_filename)

    for i in envMetabs:
        M.at[i,'Conc'] = envMetabs[i]
    
    save_model(new_model_filename,S,R,M,K)

    return


