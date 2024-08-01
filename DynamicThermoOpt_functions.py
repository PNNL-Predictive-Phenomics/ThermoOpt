import numpy as np
import pandas as pd
import random
import multiprocessing as mp
import subprocess
import numpy.random as nprd
from scipy.linalg import norm
from scipy.optimize import least_squares
import scipy.linalg as spL
import sys
import pickle
import copy
import matplotlib.pyplot as plt
import Bolt_opt_functions_new as fopt
import base_functions as baf
import analysis_functions as anf
import scipy.integrate as it

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


    if 0:#ifVerbose:
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
    # Vmax = cell_dictionary['vmax']

    # l_hyphae = 0.003 # length of hyphal compartment in cm
    # A_hyphae = 12.566e-07 # area of hyphal compartment in cm^2
    # V_hyphae = l_hyphae*A_hyphae # volume of hyphal compartment
    # V_hyphae = V_hyphae * 1.0e+12 # Convert to cubic microns
    # VolCell = V_hyphae
    # N_avogadro = 6.022140857e+23
    # Vmax = Vmax * V_hyphae # Scale Vmax to size of a hyphal compartment

    VolCell = cell_dictionary['cell_volume']
    #cell_dictionary['vmax'] = Vmax * cell_dictionary['hyphae_volume']
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

    # if ifVerbose:
    #     print('Volume of cell: '+str(VolCell))
    #     print('Surface area of cell: '+str(SurfaceAreaCell))
    #     print('Volume of near field element: '+str(volumeElement))
    #     print('Volume ratio of Chemostat To Cell: '+str(volumeChemostatToCell))
    
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
        metab_log_concs[i] = np.log((odeConcentrations[i]/1000)*Concentration2Count)
        
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
    
    # print('Saved ODE model as '+str(solutionFile))
    
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

                # if environment_dictionary['chemostat_type'][environmentMetabolites.index(em)][0] == 'Neumann' or environment_dictionary['chemostat_type'][environmentMetabolites.index(em)][0] == 'Constant_Flux':
                #     environmentDf.at[t,em] += chemostatDf.loc[timesteps[i-1],em] *dt
                
                # cell contribution is an absolute amount in dt
                # environmentDf.at[t,em] += cellContributionDf.loc[timesteps[i-1],em] #/ cell_dictionary['volume_chemostat_to_cell']

            modelBoundaryConditions[em]=environmentDf.loc[t,em]#/1000 # convert to molar

    else:
        for em in environmentMetabolites:
            modelBoundaryConditions[em]=environmentDf.loc[t,em]#/1000

    return modelBoundaryConditions

def SetInitialModelBoundary(i,environment_dictionary,cell_dictionary,ode_system_dictionary,environmentDf,chemostatDf,cellContributionDf):
    
    isChemostated = False

    timesteps=ode_system_dictionary['timesteps']
    dt=ode_system_dictionary['dt']
    environmentMetabolites=environment_dictionary['metabolites']
    t=timesteps[i]
    modelBoundaryConditions={}

    for em in environmentMetabolites:
        modelBoundaryConditions[em]=environmentDf.loc[t,em]#/1000

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
            # proposedPerturbation = environmentDf.loc[t,em]/1000 + direction*randChange

            if proposedPerturbation <0:
                modelBoundaryConditions[em]=environmentDf.loc[t,em]#/1000
            else:
                modelBoundaryConditions[em]= proposedPerturbation# convert to molar

    else:
        for em in environmentMetabolites:
            modelBoundaryConditions[em]=environmentDf.loc[t,em]#/1000

    return modelBoundaryConditions



def ComposeBaseSimulation(modelBoundaryConditions,t,cell_dictionary):
    
    # need to convert boundary conditions from concentration to log counts
# np.log((cell_dictionary['concentration_2_count']/1000)*cell_dictionary['cell_surface_area'])#rxnFlux*cell_dictionary['cell_surface_area']
        
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

# post OPT simulation, parse the output and convert the metabolite log counts into mili-molar concentrations
def ParseBestSimulations(bestSimulations,control_dictionary,cell_dictionary,metaboliteSubset=[],reactionSubset=[],verbose=False):


    Concentration2Count = cell_dictionary['concentration_2_count']

    metabSolsdf_vector=[]
    ysoldf_vector=[]
    S_active_vector=[]
    active_reactions_vector=[]
    name_vector=[]

    if len(bestSimulations[0]['bestSimulations']) >0:
        for simName in bestSimulations[0]['bestSimulations']:
            
            # try:
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
                # metabSolsConc.append(np.exp(metabSols[i])*1000/Concentration2Count)
                # vconc_lower_bound.append(np.exp(vcount_lower_bound[i])*1000/Concentration2Count)
                # vconc_upper_bound.append(np.exp(vcount_upper_bound[i])*1000/Concentration2Count)
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
        
            # except:

            #     a=0

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
    # metaboliteTraceDf = metaboliteTraceDf.set_axis(metabolite_names,axis=1)

    # display(metaboliteTraceDf)

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
        # convert counts to mM
        # metab_concs[i] = (np.exp(metab_log_concs[i])*1000/Concentration2Count)
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

    cell_dictionary['v_reference']=vref_scalar

    return cell_dictionary

# function to determine the kinetic values for the metabolic steady state by processing OPT solution 
def DetermineOptimalKinetics(t_opt,v_hat,S,Keq,n_hat,cell_dictionary,environment_dictionary,ode_system_dictionary,metabolite_labels,reaction_labels):
    
    
    # DetermineCorrectEnergyImport(t_opt,cell_dictionary,metabolite_labels,reaction_labels,v_hat,n_hat,S)

    # print('cell_dictionary[energy_import_reaction_name]')
    # print(cell_dictionary['energy_import_reaction_name'])

    # print('cell_dictionary[environment_energy_name]')
    # print(cell_dictionary['environment_energy_name'])

    # for i in range(len(reaction_labels)):

    #     if reaction_labels[i] == cell_dictionary['energy_import_reaction_name']:
    #         uptakeRate = v_hat[i] # y_sol from sim output
    #         break
    
    # for i in range(len(metabolite_labels)):

    #     if metabolite_labels[i] == cell_dictionary['environment_energy_name']:
    #         energy_conc = n_hat[i] # in mili-molar concentrations
    #         break

    # print(uptakeRate)
    # print(energy_conc)

    # vref_scalar,volumeElement = CalculateScaling(uptakeRate,energy_conc,cell_dictionary,environment_dictionary,ode_system_dictionary)
    # cell_dictionary['v_reference']=vref_scalar

    # print(vref_scalar)

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
                # isImported=True
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
                    # print('Consider '+str(cell_dictionary['import_energy_metabolites'][i]))

                    energy_conc = n_hat[j]
                    Km = cell_dictionary['import_energy_michaelis_menten'][i]
                    Vnew = Vmax * energy_conc/(Km + energy_conc)
                    # print(energy_conc)
                    # print(Vnew)

                    if V<=Vnew:
                        V=Vnew
                        largest_import_metabolite = cell_dictionary['import_energy_metabolites'][i]
                        largest_import_reaction   = cell_dictionary['import_energy_reactions'][i]

                        # print('####################')       

                        # print(Vnew)
                        # print(largest_import_metabolite)
                        # print(largest_import_reaction)

                break
    
    if largest_import_metabolite=="":
        print('Error: no import metabolite found in DetermineCorrectEnergyImport')

    cell_dictionary['environment_energy_name']= largest_import_metabolite
    cell_dictionary['energy_import_reaction_name']=largest_import_reaction


    # if previous_import_metab != cell_dictionary['environment_energy_name']:
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        # print('Energy import metabolite switch from '+str(previous_import_metab)+" to "+str(cell_dictionary['environment_energy_name']))
        # print('Time: '+str(t_opt))
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        # print()


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
    #V = Vmax
    vref_scalar = V/np.abs(uptakeRate) 

    # if ifVerbose:
    #     print('uptakeRate')
    #     print(uptakeRate)
    #     print('vref_scalar')
    #     print(vref_scalar)
        
    #     print('volumeElement')
    #     print(volumeElement)
    
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
    
    # timesteps = ode_system_dictionary['timesteps']
    
    # if ifUseAdaptiveRateParameters:
    #     print('Use adaptive rate parameters')

    # simulated concentrations, use previous ODE output as initial conditions
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
        # print('forward: '+str(forwardFactor))
        # print('reverse: '+str(reverseFactor))
        # print('net: '+str(rxnNetFlux))

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
                

    testSet=['x1_e:ENVIRONMENT','x4_e:ENVIRONMENT']

    for rid in range(len(kineticsDf.index)):

        if isReactionPossible[rid]:
            # implement the reactions
            for metabid in range(len(sim_concs)):
                # print(metabolite_names[metabid])
                if metabolite_names[metabid] not in ode_system_dictionary['metabolites_to_fix'] :

                    if S[rid,metabid] != 0:
                        # if metabolite_names[metabid] in testSet :
                        #     print(S[rid,metabid]*rxnFlux_vec[rid]*ode_dictionary['dt'] )

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

            conc = sim_concs[j]
            # print('Diffuse '+str(environmentMetabolites[i]))
            # print('Old conc: '+str(sim_concs[j]))
            sim_concs[j] = sim_concs[j] + gradDiff(sim_concs[j],boundary_concs[i],diffusionRates[i])*ode_system_dictionary['dt'] 
            # print('New conc: '+str(sim_concs[j]))
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
    
    #cell_density = cell_density + (birth_rate*cell_density*biomass_flux-death_rate*cell_density)*dt

    cell_volume = cell_dictionary['cell_volume']
    gluc_gram_mole = 180.156
    gramBiomass_to_cellcount =  5e-13
    grams_of_biomass = biomass_flux*cell_volume*(0.3)*gluc_gram_mole
    growth_rate  = grams_of_biomass/gramBiomass_to_cellcount

    cell_density = cell_density + (growth_rate*cell_density - death_rate*cell_density)*dt

    print(biomass_flux)

    print(growth_rate)
    print(death_rate)

    return cell_density

def UpdateEnvironmentDueToCellDensity(dt,S,metabolite_names,metabolite_concentrations,reaction_names,rxnFlux_vec,environment_dictionary,cell_dictionary):


    cell_density = cell_dictionary['cell_density']
    cell_volume = cell_dictionary['cell_volume']
    environmentMetabolites=environment_dictionary['metabolites']
    bath_volume = environment_dictionary['bath_volume']
    
    transportReactions=environment_dictionary['transport_reactions']

    # print('Cell_volume: ')
    # print(cell_volume)
    # print('Bath_volume: ')
    # print(bath_volume)
    # print('Cell_volume/Bath_volume: ')
    # print(cell_volume/bath_volume)

    for i in range(len(environmentMetabolites)):

        metab_id = metabolite_names.tolist().index(environmentMetabolites[i])

        rxn_id = reaction_names.index(transportReactions[i])

        # assume that environment is large enough for each to cell to act optimally

        # (correction) reverse the previous reaction value 
        metabolite_concentrations[metab_id] -= S[rxn_id,metab_id]*rxnFlux_vec[rxn_id]*dt

        
        # apply flux from cell density
        avgadro_num = 6.02214076e23
        conversion_factor = cell_volume*cell_density*(avgadro_num / 1e9)
        metabolite_concentrations[metab_id] += conversion_factor*S[rxn_id,metab_id]*rxnFlux_vec[rxn_id]*dt


        #metabolite_concentrations[metab_id] += cell_density*S[rxn_id,metab_id]*rxnFlux_vec[rxn_id]*dt*(bath_volume)

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


def PlotConcentrationTraces(metaboliteTraceDf,metabolitesOPT,timesOPT,ifPlotOpt=True,metaboliteSubset=[],xlim=0):
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
                plt.scatter(timesOPT,optConcMatrix[metabNames.index(metab)])
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
                plt.scatter(timesOPT,optConcMatrix[metabNames.index(metab)])
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

def PlotFluxTraces(fluxTraceDf,reactionOPT,timesOPT,ifPlotOpt=True,reactionSubset=[],xlim=0):
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
                plt.scatter(timesOPT,optFluxMatrix[reactionSubset.index(rxn)])
        else:
            for rxn in reactionSubset:
                plt.plot(fluxTraceDf[rxn],label=rxn)


    plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Reaction flux')
    if xlim != 0:
        plt.xlim([0,xlim])
    # plt.ylim([0,1e-10])

    plt.show()


def PlotRegulationTraces(regulationOPT,timesOPT,rxnNames,reactionSubset=[]):
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
            plt.scatter(timesRxn,optRegulationMatrix[reactionSubset.index(rxn)],label=rxn)
         


    plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Reaction regulation level')
    # plt.xlim([0,0.7])
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
