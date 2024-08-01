"""

Connah G. M. Johnson     
connah.johnson@pnnl.gov
March 2024

Functions to implement a flux constrained optimization for metabolic models. 

Based on optimization functions provided by Ethan King (PNNL).

Code contents:

def signFunc
def getMisalignedSigns
def newSignFunc
def opt_setup
def SetMetaboliteToimplicit
def opt_setup_new
def flux_ent_opt
def Max_alt_flux
def rxn_flux

"""

import numpy as np
import pandas as pd
import subprocess
import sys
import re
import os
import warnings


import pickle

from scipy.linalg import norm
from scipy.optimize import least_squares
import scipy.linalg as spL
from scipy.optimize import lsq_linear
import numpy.random as nprd

from IPython.core.display import display
from IPython.core.debugger import set_trace
pd.set_option('display.max_columns', None,'display.max_rows', None)

warnings.filterwarnings('ignore')

from importlib import reload
import time
import pyomo
import pyomo.environ as pe
import itertools 
import numpy as np
from numpy.random import randn, seed
import base_functions as baf
from scipy.optimize import dual_annealing
from scipy.optimize import basinhopping

import pyutilib.services
from pyomo.opt import TerminationCondition

reload(pyomo)


'''
Function for calcualting the value of the sign difference for determining the initial variable log count values. This function is called during the 
dual simmulated annealing optimization. 

Inputs:
- x: the optimizers trial for log counts for varibale metabolites
- v: the fixed metabolite constant values
- S_v_T: the stoichioetric ratios for the variable metabolites
- y_grad: the initial flux projection into the null space

'''

def signFunc(x,v,S_v_T,Se_N,numberEta):
    
    eta = x[0:numberEta]
    beta = x[numberEta:]

    mat = np.matmul(S_v_T,eta)
    mat = np.reshape(mat,(len(mat),1))

    matR = np.matmul(Se_N,beta)
    matR = np.reshape(matR,(len(matR),1))
    lhs = np.sign(v - mat)
    rhs = np.sign(matR)

    val = np.sum(np.abs(lhs - rhs)/2)
    

    return val

def getMisalignedSigns(x,v,S_v_T,Se_N,numberEta):
    
    eta = x[0:numberEta]
    beta = x[numberEta:]

    mat = np.matmul(S_v_T,eta)
    mat = np.reshape(mat,(len(mat),1))

    matR = np.matmul(Se_N,beta)
    matR = np.reshape(matR,(len(matR),1))
    lhs = np.sign(v - mat)
    rhs = np.sign(matR)

    misalignment=[]
    for i in range(len(lhs)):
        if np.abs(lhs[i] - rhs[i])/2 == 1:
            misalignment.append(i)
    
    return misalignment

def newSignFunc(x,v,S_v_T,Se_N,numberEta):
    eta = x[0:numberEta]
    beta = x[numberEta:]

    mat = np.matmul(S_v_T,eta)
    mat = np.reshape(mat,(len(mat),1))

    matR = np.matmul(Se_N,beta)
    matR = np.reshape(matR,(len(matR),1))
    lhs = v - mat
    rhs = matR

    val = np.sum(lhs*rhs)

    return (-1)*val

''' 
INPUTS

Metabolites: A dataframe with four columns
    'Conc', If fixed this will be the fixed value, if variable this may be used as a starting estimate depending on the steady state criteria
    'upper_bound', Maximum value for the metabolite
    'lower_bound', Minimum value for the metabolite
    'Variable', True or False if the metabolite value should be allowed to change in the optimization
    'Steady_State' Explicit or Implicit, if Explicit we must compute flux such that dn/dt = 0, if implicit we remove steady state constraint.
    'Net_Flux' If the steady state is Implicit, this is a restriction on the sign of the net flux for the metabolite that is allowed, (e.g. produced or consumed)

    

S_active: This is a dataframe that contains the stoichiometric matrix with columns metabolites and rows reactions


active_reactions : This is a datframe that constains information about each reaction, 
                 we expect a column "Keq" in the dataframe that we will use to get equilibrium constants for each reaction
                    (In the future we could consider sourcing information from here about if a reaction should be included or not)

                 'Objective Coefficient' the coefficient that should be used for computing the flux, a zero if the flux is not part of the objective.

                 'Flux_Direction' if known the direction of the flux can be -1, 0, 1

                 'Allow_Regulation' If the activity coefficient can be adjusted; can be 0 or 1, 1 being True.

hyperparameterDict: dictionary of hyperparameters from the swarm agent method. [Mb,VarM_lbnd,zeeta,deltaConc,deltaFlux]

simulationDict: dictionary for any simulation parameters that act on solver properties such as variable bounds

'''


''' 
We want to add additional functionality here to allow for specifying more optionality for the optimization.

This includes..

 - the option to restrict the direction of fluxes, with designation (-1, 0 , 1) where 0 means that there is no restriction

 - the option to restict the direction of the net change of any metabolite with an implicit steady state. with designations (-1, 0, 1)

 - force flux to be a fixed value. 


'''

'''



Inputs:
    - metabolites: the metabolite dataframe for the model 
    - active_reactions: the reaction dataframe for the model
    - S_active: the stoichiometric dataframe describing the stoichiometrix ratios for the change in metabolites due to each reaction
    - hyperparameterDict: 'Mb', 'zeta', 'delta_concentration', 'delta_flux'
    - simulationDict : experiment simulation dictionary for the swarm agent;
                    'name', 'model', 'model_output', 'secondary_objective', 'primary_obj_tolerance', 'initial_conditions', 'boundary_conditions',
                    'obj_coefs', 'remove_metabolites', 'remove_reactions', 'implicit_metabolites', 'explicit_metabolites', 'redox_pairs', 
                    'redox_ratios', 'raise_bounds', 'specify_upper_bound', 'specify_lower_bound', 'net_flux', 'flux_direction'

Returns:
    - n_ini: the vector for the initial metabolite log concentration counts in the model
    - y_ini: the vector for the initial flux through the reactions
    - opt_parms: dictionary for the optimization parameters

'''

def opt_setup(metabolites, active_reactions, S_active,hyperparameterDict={},simulationDict={},previous_sol_file="",n_override=[],y_override=[]):

    #SET COPIES OF ALL INPUT DATAFRAMES THAT WILL BE INTERNALLY MANIPULATED
    metabolites_df = metabolites.copy()
    active_reactions_df = active_reactions.copy()
    S_active_df = S_active.copy()

    Keq = active_reactions_df['Keq']
    Keq = np.array(Keq)
    Keq[np.isinf(Keq)] = 1e+300
    Keq[Keq>1e+300] = 1e+300
    Keq[Keq<1e-16] = 1e-16

    active_reactions_df['Keq'] = Keq
    

    setupError=0
    ## Set some optimization hyperparameters parameters
    Mb = 100  #Big M relaxation parameter for flux consistency constraints
    zeta = 0.01
    linear_solver = 'default'
    hsllib=""
    max_cpu_time = 800000
    max_iter = 10000
    acceptable_tol = 1e-6
    solver_tol = 1e-7

    # override hyperparameters through input list
    if 'Mb' in hyperparameterDict:
        Mb = hyperparameterDict['Mb']

    if 'zeta' in hyperparameterDict:
        zeta = hyperparameterDict['zeta']

    if 'linear_solver' in hyperparameterDict:
        linear_solver = hyperparameterDict['linear_solver']

    if 'hsllib' in hyperparameterDict:
        hsllib = hyperparameterDict['hsllib']

    if 'max_cpu_time' in hyperparameterDict:
        max_cpu_time = hyperparameterDict['max_cpu_time']

    if 'max_iter' in hyperparameterDict:
        max_iter = hyperparameterDict['max_iter']

    if 'acceptable_tol' in hyperparameterDict:
        acceptable_tol = hyperparameterDict['acceptable_tol']

    if 'solver_tol' in hyperparameterDict:
        solver_tol = hyperparameterDict['solver_tol']

    delta_concentration = 0.0
    delta_flux = 0.0
    isPerturbConcentrations = False
    if 'delta_concentration' in hyperparameterDict:
        delta_concentration = hyperparameterDict['delta_concentration']
        isPerturbConcentrations=True
    
    isPerturbFlux = False
    if 'delta_flux' in hyperparameterDict:
        delta_flux = hyperparameterDict['delta_flux']
        isPerturbFlux=True


    feasibilityCheck = False
    if 'feasibility_check' in hyperparameterDict:
        feasibilityCheck = hyperparameterDict['feasibility_check']

    feasibility_as_input = False
    if 'feasibility_as_input' in hyperparameterDict:
        feasibility_as_input = hyperparameterDict['feasibility_as_input']

    useAnnealing = False
    if 'annealing_check' in hyperparameterDict:
        useAnnealing = hyperparameterDict['annealing_check']

    if feasibility_as_input:
        feasibilityCheck = True
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

    m_var_net_flux_restricted_idx = np.array(metabolites_df[ (metabolites_df['Net_Flux']== 1) | (metabolites_df['Net_Flux']==-1) ].index )


    ### FLUX INDEXES

    active_reactions_df.reset_index(inplace = True) # set a reference index for all of the fluxes, order must be preserved from this point forward including the Stoich matrix

    flux_tot_idx = np.array(active_reactions_df.index)

    metabolite_labels = metabolites_df['index'].values
    reaction_labels = active_reactions_df['index'].values

    flux_free_reg_idx = np.array(active_reactions_df[ (active_reactions_df['Flux_Direction'] == 0) & (active_reactions_df['Allow_Regulation'] ==1)  ].index)
    flux_free_unreg_idx =  np.array(active_reactions_df[ (active_reactions_df['Flux_Direction'] == 0) & (active_reactions_df['Allow_Regulation'] ==0)  ].index)
    flux_fxsgn_reg_idx =  np.array(active_reactions_df[ ( (active_reactions_df['Flux_Direction' ]== 1) | (active_reactions_df['Flux_Direction']==-1) ) & (active_reactions_df['Allow_Regulation'] ==1) ].index)
    flux_fxsgn_unreg_idx =  np.array(active_reactions_df[ ( (active_reactions_df['Flux_Direction' ]== 1) | (active_reactions_df['Flux_Direction']==-1) ) & (active_reactions_df['Allow_Regulation'] ==0) ].index)

    if len(flux_free_reg_idx) + len(flux_free_unreg_idx) + len(flux_fxsgn_reg_idx) + len(flux_fxsgn_unreg_idx)  != len(flux_tot_idx) : print('Flux Assignments Are Inconsistent!')


    '''
    ##################################
    ###### Get the parameters for the optimization
    ####################################
    '''

    ''' 
    Get the equilibrium constants
    '''

    '''
    The fixed metabolite concentrations
    '''
     
    T = 298.15
    R = 8.314e-03
    RT = R*T
    N_avogadro = 6.022140857e+23
    VolCell = 1.0e-15
    Concentration2Count = N_avogadro * VolCell


    #number of variable metabolites
    m_conc_log_counts = np.log( metabolites_df['Conc'].values * Concentration2Count, dtype=np.float64)

    m_fixed_log_counts = m_conc_log_counts[m_fixed_idx]


    
    '''
    The Variable Metabolite Bounds
    '''

    m_log_count_upper_bound = np.log( metabolites_df['Upper_Bound'].values * Concentration2Count,dtype=np.float64 )
    m_log_count_lower_bound = np.log( metabolites_df['Lower_Bound'].values * Concentration2Count,dtype=np.float64 )


    # raise the upper bound of a metabolite set to the system maximum
    if 'raise_upper_bounds' in simulationDict.keys():
      for species in simulationDict['raise_upper_bounds']:
        # in order to raise the bound on a metabolite the bound must exist i.e metabolite be variable
        if metabolites.loc[species,'Variable']:
           metabolites.at[species,'Target Conc']= (sys.float_info.max)
           metab_idx = np.where(np.array(metabolite_name_order) == species)[0][0]
           m_log_count_upper_bound[metab_idx]=(sys.float_info.max)
        else:
            print("'raise_bound': metabolite "+str(species)+" must be variable")
    

    if 'redox_pairs' in simulationDict.keys():
      for redox_pair in simulationDict['redox_pairs']:
        if len(redox_pair)==2:
            # check whether either pair element is already fixed
            name_1 = redox_pair[0]
            name_2 = redox_pair[1]

            isVar_2 = metabolites.loc[name_2,'Variable']

            if isVar_2:
                # change upper bound for redox species 2
                metabolites.at[name_2,'Target Conc']= (sys.float_info.max)
                metab_idx = np.where(np.array(metabolite_name_order) == name_2)[0][0]
                m_log_count_upper_bound[metab_idx]=(sys.float_info.max)
            else:
                print("'redox_pairs': metabolite "+str(name_2)+" must be variable")
        else:
           print("Error redox_pairs should have two elements each")

    if 'specify_upper_bound' in simulationDict.keys():
      boundsDict=simulationDict['specify_upper_bound']
      for species in boundsDict.keys():
        if metabolites.loc[species,'Variable']:
            metab_idx = np.where(np.array(metabolite_name_order) == species)[0][0]
            m_log_count_upper_bound[metab_idx]=np.log((boundsDict[species])*Concentration2Count)
        else:
            print("'specify_upper_bound': metabolite "+str(species)+" must be variable")

    if 'specify_lower_bound' in simulationDict.keys():
      boundsDict=simulationDict['specify_lower_bound']
      for species in boundsDict.keys():
        if metabolites.loc[species,'Variable']:
            metab_idx = np.where(np.array(metabolite_name_order) == species)[0][0]
            m_log_count_lower_bound[metab_idx]=np.log((boundsDict[species])*Concentration2Count)
        else:
            print("'specify_lower_bound': metabolite "+str(species)+" must be variable")


    m_var_log_count_upper_bound = m_log_count_upper_bound[m_var_idx]
    m_var_log_count_lower_bound = m_log_count_lower_bound[m_var_idx]

    '''
    Stoichiometric Matrix
    '''
    S = S_active_df.values


    ''' 
    The Objective Flux Indexes
    '''
    obj_coefs = np.array(active_reactions_df['Obj_Coefs'].values, dtype=np.float64)
    numObjs = len(obj_coefs)

    obj_coefs = np.reshape(obj_coefs,(1,len(obj_coefs)))

    if feasibilityCheck:
        print('Set all objectives to zero')
        obj_coefs = np.zeros(numObjs)
        obj_coefs = np.reshape(obj_coefs,(1,numObjs))


    secondary_obj_coefs=[]
    primary_obj_tolerance=100

    if 'secondary_objective' in simulationDict.keys():
        secondary_obj_coefs=np.zeros(len(active_reactions_df))

        for i in range(len(active_reactions_df.index)):

            if active_reactions_df.at[i,'index'] in simulationDict['secondary_objective']:

                secondary_obj_coefs[i] = simulationDict['secondary_objective'][active_reactions_df.at[i,'index']]
                if feasibilityCheck:
                    secondary_obj_coefs[i] = 0

    if 'primary_obj_tolerance' in simulationDict.keys():

        primary_obj_tolerance = simulationDict['primary_obj_tolerance']


    ''' 
    ### COMPUTE THE INITIAL CONDITION
    '''

    '''     
    First we compute the gradient direction of the fluxes with respect to the objective projected onto the steady state condition.

    Then we search for a variable metabolite solution that can produce fluxes close in direction to this gradient.

    '''

    S_T = S    # S_T is the Stoich matrix with rows as reactions, columns as metabolites
    S = np.transpose(S_T) # this now is the Stoich matrix with rows metabolites, and columns reactions

    n_react = S_T.shape[0]

    #Find submatrix corresponding to the explicit steady state metabolites, these define our steady state conditions
    S_e_T = np.delete(S_T,m_implicit_idx,1)
    S_e = np.transpose(S_e_T)


    #find a basis for the nullspace of S_e, this defines our set of steady state flux solutions that are feasible.
    Se_N = spL.null_space(S_e)
    dSe_N = np.shape(Se_N)[1] # the dimension of the nullspace


    #Find the gradient direction for the objective projected into the steady state solutions
    beta_grad = np.matmul(obj_coefs,Se_N)
    beta_grad = np.transpose(beta_grad)

    y_grad = np.matmul(Se_N,beta_grad)

    y_grad = np.reshape(y_grad,(len(y_grad),1))

    # We split S into the submatrix corresponding to variable metabolites, and the submatrix corresponding to fixed metabolites

    S_f_T = np.delete(S_T,m_var_idx,1)
    S_f = np.transpose(S_f_T)

    S_v_T = np.delete(S_T,m_fixed_idx,1)
    S_v = np.transpose(S_v_T)


    #Compute initial condtion
    K_v = np.reshape(Keq,(len(Keq),1))
    m_fixed_log_counts = np.reshape(m_fixed_log_counts,(len(m_fixed_log_counts),1))


    #### first we want to find the matrix that is S_v^T with the signs appropriately adjusted for - y_grad
    S_v_T_sgn = S_v_T*np.sign(-y_grad)
    S_v_sgn = np.transpose(S_v_T_sgn)
    S_f_T_sgn = S_f_T*np.sign(-y_grad)

    try:

        if previous_sol_file == "" and n_override==[]:
            # use least squares method to determine starting optimiser point
            
            if useAnnealing and not feasibilityCheck:

                print('Initialize using simulated annealing')

                S_id = np.eye(n_react)
                A = np.concatenate([S_v_T_sgn,-S_id],axis = 1)

                ### then we compute the right hand side
                v = np.matmul(S_f_T_sgn,m_fixed_log_counts) - np.log(K_v)*np.sign(-y_grad)

                #construct the bounds
                x_upper = np.concatenate([m_var_log_count_upper_bound,1000*np.ones(n_react)])
                x_lower = np.concatenate([m_var_log_count_lower_bound, np.zeros(n_react)])
                
                startTime = time.time()
                try:
                    opt_out = lsq_linear(A,-np.ravel(v),bounds = (x_lower,x_upper))
                except:
                    setupError=2
                    raise Exception('lsq_linear error')
                endTime = time.time()
                n_out = opt_out['x']
                n_out = n_out[0:len(m_var_log_count_upper_bound)]
                n_out = np.reshape(n_out,(len(n_out),1))
                n_ini_ls = np.ravel(n_out) 



                v_new = np.log(K_v) - np.matmul(S_f_T,m_fixed_log_counts)

                beta_ini_projection = np.ravel(np.matmul(np.transpose(Se_N),v_new))

                numberEta =  len(m_var_log_count_upper_bound)

                maxBeta=1#200
                minBeta=-1#-200

                for i in range(len(beta_ini_projection)):
                    if beta_ini_projection[i] > maxBeta:
                        beta_ini_projection[i] = maxBeta
                    elif beta_ini_projection[i] <minBeta:
                        beta_ini_projection[i] = minBeta
                        

                for i in range(len(n_ini_ls)):
                    if n_ini_ls[i] > m_var_log_count_upper_bound[i]:
                        # print(n_ini_ls[i])
                        n_ini_ls[i] = m_var_log_count_upper_bound[i]
                    elif n_ini_ls[i] <m_var_log_count_lower_bound[i]:
                        # print(n_ini_ls[i])
                        n_ini_ls[i] = m_var_log_count_lower_bound[i]

                upper = m_var_log_count_upper_bound.tolist() +  (maxBeta*np.ones(len(beta_grad))).tolist()
                lower = m_var_log_count_lower_bound.tolist() + (minBeta*np.ones(len(beta_grad))).tolist()

                boundZip =list(zip(lower,upper))
                print('Use least squares solution and projection of beta to seed annealing')
                x_init = np.concatenate((n_ini_ls, beta_ini_projection),axis=0)


                isAnneal=False
                isBasin=False
                if isAnneal:
                    print('Use dual_annealing')
                    startTime=time.time()
                    opt_out= dual_annealing(signFunc, maxiter = 1000, bounds=boundZip, x0 = x_init, args=(v_new,S_v_T,Se_N,numberEta))
                    endTime=time.time()
                    print("Time in dual_annealing: "+str(endTime-startTime))

                    percentAlignment = 100*(numberEta-signFunc(opt_out['x'],v_new,S_v_T,Se_N,numberEta))/numberEta
                    print('Dual annealing status : %s' % opt_out['message'])
                    print('Number of Function evaluations : %d' % opt_out['nfev'])
                    print('Annealing return function value : %d' % opt_out.fun)
                    print('Annealing return function value (a.k.a number misaligned signs) : %d' % signFunc(opt_out['x'],v_new,S_v_T,Se_N,numberEta))
                    print('Percentage sign alignment : %d ' % percentAlignment)

                elif isBasin:
                    print('Use basinhopping')
                    minimizer_kwargs = { "method": "L-BFGS-B","bounds":boundZip,"args": (v_new,S_v_T,Se_N,numberEta) }
                    startTime=time.time()
                    opt_out = basinhopping(signFunc, x0 = x_init, minimizer_kwargs=minimizer_kwargs,niter=1000)
                    endTime=time.time()
                    print("Time in basinhopping: "+str(endTime-startTime))
                
                    percentAlignment = 100*(numberEta-signFunc(opt_out['x'],v_new,S_v_T,Se_N,numberEta))/numberEta
                    print('Basinhopping status : %s' % opt_out['message'])
                    print('Number of Function evaluations : %d' % opt_out['nfev'])
                    print('Annealing return function value : %d' % opt_out.fun)
                    print('Basinhopping return function value (a.k.a number misaligned signs) : %d' % signFunc(opt_out['x'],v_new,S_v_T,Se_N,numberEta))
                    print('Percentage sign alignment : %d ' % percentAlignment)
        
                
                misAlignedMetabolites=[]
                isReportMisalignment = True
                if isReportMisalignment:
                    if signFunc(opt_out['x'],v_new,S_v_T,Se_N,numberEta) > 0.5:
                        # return the index of the misaligned flux
                        misalignedFlux = getMisalignedSigns(opt_out['x'],v_new,S_v_T,Se_N,numberEta)
                        # print(misalignedFlux)
                        if len(misalignedFlux) >0:
                            # print('Return misaligned fluxes:')
                            for i in misalignedFlux:
                                # print(active_reactions.index[i])
                                display(active_reactions.at[active_reactions.index[i],'Full Rxn'])
                                S_active_df
                                for j in range(len(S_active.loc[active_reactions.index[i]])):
                                    sval=S_active.loc[active_reactions.index[2]][j]
                                    if sval !=0:
                                        misAlignedMetabolites.append(metabolites.index[j])


                annealOut = np.ravel(opt_out['x'])
                n_ini = annealOut[0:numberEta]
                beta_ini = annealOut[numberEta:]

                y_ini = np.ravel(np.matmul(Se_N,beta_ini))

            else:
                # use the least squares method
                ### We append the identity for our slack variables

                S_id = np.eye(n_react)
                A = np.concatenate([S_v_T_sgn,-S_id],axis = 1)

                ### then we compute the right hand side
                rhs=np.log(K_v)*np.sign(-y_grad)
                
                v = np.matmul(S_f_T_sgn,m_fixed_log_counts) - np.log(K_v)*np.sign(-y_grad)

                #construct the bounds
                x_upper = np.concatenate([m_var_log_count_upper_bound,1000*np.ones(n_react)])
                x_lower = np.concatenate([m_var_log_count_lower_bound, np.zeros(n_react)])
            
                try:
                    opt_out = lsq_linear(A,-np.ravel(v),bounds = (x_lower,x_upper))
                except:
                    setupError=3
                    raise Exception('lsq_linear error')
                n_out = opt_out['x']
                n_out = n_out[0:len(m_var_log_count_upper_bound)]
                n_out = np.reshape(n_out,(len(n_out),1))

                v_true = np.matmul(S_f_T,m_fixed_log_counts) - np.log(K_v)
                flux_out = np.matmul(S_v_T,n_out) + v_true

                n_ini = np.ravel(n_out)    #estimate of the variable metabolite values
                y_ini = -1e0*np.ravel(flux_out) # estimate of the reaction fluxes

                if(isPerturbConcentrations):
                    # perturb concentrations that are not fixed
                    for i in range(len(n_ini)):

                        val = np.exp(n_ini[i])
                        randChange = nprd.uniform(0.0, delta_concentration)
                        direction = nprd.choice([-1,1])

                        perturbed_val = n_ini[i]
                        if val + direction*randChange > 0.0:
                            # feasible
                            perturbed_val = np.log(val + direction*randChange)

                        n_ini[i] = perturbed_val

                if (isPerturbFlux):
                    flux__directions = np.array(active_reactions_df['Flux_Direction'].values, dtype=np.float64)
                    for i in range(0,len(y_ini)):

                        oldVal = y_ini[i]
                        randChange = nprd.uniform(-delta_flux, delta_flux)
                        absFlux=np.abs(y_ini[i])

                        flux__direction = flux__directions[i]

                        if flux__direction ==1:
                            if absFlux <= delta_flux:
                                # flux is small therefore test a change in sign
                                direction = nprd.choice([-1,1])
                                y_ini[i] = direction*y_ini[i] + randChange
                            else:
                                y_ini[i] = y_ini[i] + randChange

                            if y_ini[i] <0:
                                y_ini[i] = -1*y_ini[i]

                        elif flux__direction ==1:
                            if absFlux <= delta_flux:
                                # flux is small therefore test a change in sign
                                direction = nprd.choice([-1,1])
                                y_ini[i] = direction*y_ini[i] + randChange
                            else:
                                y_ini[i] = y_ini[i] + randChange

                            if y_ini[i] >0:
                                y_ini[i] = -1*y_ini[i]

                        else:
                            if absFlux <= delta_flux:
                                # flux is small therefore test a change in sign
                                direction = nprd.choice([-1,1])
                                y_ini[i] = direction*y_ini[i] + randChange
                            else:
                                y_ini[i] = y_ini[i] + randChange

                y_ini = zeta*y_ini
        elif n_override !=[]:

            n_out = n_override
            n_out = np.reshape(n_out,(len(n_out),1))

            v_true = np.matmul(S_f_T,m_fixed_log_counts) - np.log(K_v)
            flux_out = np.matmul(S_v_T,n_out) + v_true

            n_ini = np.ravel(n_out)    #estimate of the variable metabolite values
            y_ini = -1e0*np.ravel(flux_out) # estimate of the reaction fluxes

            if(isPerturbConcentrations):
                # perturb concentrations that are not fixed
                for i in range(len(n_ini)):

                    val = np.exp(n_ini[i])
                    randChange = nprd.uniform(0.0, delta_concentration)
                    direction = nprd.choice([-1,1])

                    perturbed_val = n_ini[i]
                    if val + direction*randChange > 0.0:
                        # feasible
                        perturbed_val = np.log(val + direction*randChange)

                    n_ini[i] = perturbed_val

            if (isPerturbFlux):

                flux__directions = np.array(active_reactions_df['Flux_Direction'].values, dtype=np.float64)
                for i in range(0,len(y_ini)):

                    oldVal = y_ini[i]
                    randChange = nprd.uniform(-delta_flux, delta_flux)
                    absFlux=np.abs(y_ini[i])

                    flux__direction = flux__directions[i]

                    if flux__direction ==1:
                        if absFlux <= delta_flux:
                            # flux is small therefore test a change in sign
                            direction = nprd.choice([-1,1])
                            y_ini[i] = direction*y_ini[i] + randChange
                        else:
                            y_ini[i] = y_ini[i] + randChange

                        if y_ini[i] <0:
                            y_ini[i] = -1*y_ini[i]

                    elif flux__direction ==1:
                        if absFlux <= delta_flux:
                            # flux is small therefore test a change in sign
                            direction = nprd.choice([-1,1])
                            y_ini[i] = direction*y_ini[i] + randChange
                        else:
                            y_ini[i] = y_ini[i] + randChange

                        if y_ini[i] >0:
                            y_ini[i] = -1*y_ini[i]

                    else:
                        if absFlux <= delta_flux:
                            # flux is small therefore test a change in sign
                            direction = nprd.choice([-1,1])
                            y_ini[i] = direction*y_ini[i] + randChange
                        else:
                            y_ini[i] = y_ini[i] + randChange

            y_ini = zeta*y_ini


        else:
            # use previous solution

            y_sol_prev,alpha_sol_prev,n_sol_prev,unreg_rxn_flux_prev,S_prev,Keq_prev,f_log_counts_prev,vcount_lower_bound_prev,vcount_upper_bound_prev = baf.load_model_solution(previous_sol_file)

            n_out = n_sol_prev
            n_out = np.reshape(n_out,(len(n_out),1))

            v_true = np.matmul(S_f_T,m_fixed_log_counts) - np.log(K_v)
            flux_out = np.matmul(S_v_T,n_out) + v_true

            n_ini = np.ravel(n_out)    #estimate of the variable metabolite values
            y_ini = y_sol_prev # estimate of the reaction fluxes

            if(isPerturbConcentrations):
                # perturb concentrations that are not fixed

                for i in range(len(n_ini)):

                    val = np.exp(n_ini[i])
                    randChange = nprd.uniform(0.0, delta_concentration)
                    direction = nprd.choice([-1,1])

                    perturbed_val = n_ini[i]
                    if val + direction*randChange > 0.0:
                        # feasible
                        perturbed_val = np.log(val + direction*randChange)

                    n_ini[i] = perturbed_val

            if (isPerturbFlux):

                flux__directions = np.array(active_reactions_df['Flux_Direction'].values, dtype=np.float64)
                for i in range(0,len(y_ini)):

                    oldVal = y_ini[i]
                    randChange = nprd.uniform(-delta_flux, delta_flux)
                    absFlux=np.abs(y_ini[i])

                    flux__direction = flux__directions[i]

                    if flux__direction ==1:
                        if absFlux <= delta_flux:
                            # flux is small therefore test a change in sign
                            direction = nprd.choice([-1,1])
                            y_ini[i] = direction*y_ini[i] + randChange
                        else:
                            y_ini[i] = y_ini[i] + randChange

                        if y_ini[i] <0:
                            y_ini[i] = -1*y_ini[i]

                    elif flux__direction ==1:
                        if absFlux <= delta_flux:
                            # flux is small therefore test a change in sign
                            direction = nprd.choice([-1,1])
                            y_ini[i] = direction*y_ini[i] + randChange
                        else:
                            y_ini[i] = y_ini[i] + randChange

                        if y_ini[i] >0:
                            y_ini[i] = -1*y_ini[i]

                    else:
                        if absFlux <= delta_flux:
                            # flux is small therefore test a change in sign
                            direction = nprd.choice([-1,1])
                            y_ini[i] = direction*y_ini[i] + randChange
                        else:
                            y_ini[i] = y_ini[i] + randChange

            y_ini = zeta*y_ini

        ''' 
        #################
        ##### COLLECT INFORMATION REQUIRED FOR CONSTRUCTING THE OPTIMIZATION PROBLEM
        ####################

        will store all relevant inputs and hyperparameters in a dictionary
        '''

        opt_parms = {}

        
        #PROBLEM INDEXES , to identify the constraint types neccessary for metabolites and fluxes
        ### metabolites
        opt_parms['m_tot_idx'] = m_tot_idx
        opt_parms['m_var_idx'] = m_var_idx
        opt_parms['m_implicit_idx'] = m_implicit_idx
        opt_parms['m_var_explicit_idx'] = m_var_explicit_idx
        opt_parms['m_var_implicit_idx'] = m_var_implicit_idx
        opt_parms['m_fixed_idx'] = m_fixed_idx
        opt_parms['m_var_net_flux_restricted_idx'] = m_var_net_flux_restricted_idx

        opt_parms['metabolite_labels'] = metabolite_labels
        opt_parms['reaction_labels'] = reaction_labels

        ### fluxes
        opt_parms['flux_tot_idx'] = flux_tot_idx
        opt_parms['flux_free_reg_idx'] = flux_free_reg_idx
        opt_parms['flux_free_unreg_idx'] = flux_free_unreg_idx
        opt_parms['flux_fxsgn_reg_idx'] = flux_fxsgn_reg_idx
        opt_parms['flux_fxsgn_unreg_idx'] = flux_fxsgn_unreg_idx



        #PROBLEM DATA

        #metabolites
        opt_parms['m_log_count_upper_bound'] = m_log_count_upper_bound
        opt_parms['m_log_count_lower_bound'] = m_log_count_lower_bound
        opt_parms['m_conc_log_counts'] = m_conc_log_counts
        opt_parms['m_net_flux_restrictions'] = np.array(metabolites_df['Net_Flux'].values,dtype=np.float64)


        #fluxes
        opt_parms['Keq'] = Keq
        opt_parms['obj_coefs'] = obj_coefs
        opt_parms['secondary_obj_coefs'] = secondary_obj_coefs
        opt_parms['primary_obj_tolerance'] = primary_obj_tolerance
        opt_parms['flux_direction_restrictions'] = np.array(active_reactions_df['Flux_Direction'].values, dtype=np.float64)


        #Stoich
        opt_parms['S'] = S
        opt_parms['Se_N'] = Se_N

        #Optimization Hyperperameters
        opt_parms['Mb'] = Mb
        opt_parms['solverTol'] = solver_tol
        opt_parms['solverAcceptableTol'] = acceptable_tol
        opt_parms['max_iter'] = max_iter
        opt_parms['max_cpu_time'] = max_cpu_time
        opt_parms['linear_solver'] = linear_solver
        opt_parms['hsllib'] = hsllib
        opt_parms['feasibility_check'] = feasibilityCheck
        opt_parms['feasibility_as_input'] = feasibility_as_input
        opt_parms['annealing_check'] = useAnnealing

        return n_ini, y_ini, opt_parms, setupError

    except:
        

        return [], [], {}, setupError


def SetMetaboliteToimplicit(metabolites, active_reactions,S_active_df,misAlignedMetabolites):
    # need to determine and set metabolite(s) to be implicit 
    if misAlignedMetabolites:
        setToImplicit=[]
        # misAlignedMetabolites may contain duplicates
        duplicate_count_dict={metab:misAlignedMetabolites.count(metab) for metab in misAlignedMetabolites}
        
        keyIndexDict={}
        for metabIdx in metabolites.index:
            for metab in duplicate_count_dict.keys():
                if metabolites.at[metabIdx,'index'] == metab:
                    if metabolites.at[metabIdx,'Steady_State'] == 'Explicit':
                        keyIndexDict[metab] = metabIdx
        maxCount=0
        maxMetab=''

        for metab in keyIndexDict.keys():
            # test if the metabolite is already 
            # setToImplicit.append(metab)
            numberOfReactions=0
            for i in S_active_df[metab]:
                numberOfReactions+=np.abs(i)
            print('Metab: '+str(metab)+" number of reactions "+str(numberOfReactions))
            if numberOfReactions > maxCount:
                maxCount=numberOfReactions
                maxMetab=metab

        setToImplicit = [maxMetab]

        # more general method may set multiple metabs as implicit per pass
        for metab in setToImplicit:
            metabolites.at[keyIndexDict[metab],'Steady_State'] = 'Implicit'
            print('Set: '+str(metab))
    else:
        print('misAlignedMetabolites is empty')

    return metabolites


def opt_setup_new(metabolites, active_reactions, S_active,hyperparameterDict={},simulationDict={},previous_sol_file=""):

    #SET COPIES OF ALL INPUT DATAFRAMES THAT WILL BE INTERNALLY MANIPULATED

    metabolites_df = metabolites.copy()
    active_reactions_mod_df = active_reactions.copy()
    S_active_df = S_active.copy()

    Keq = active_reactions_mod_df['Keq']
    Keq = np.array(Keq)
    Keq[np.isinf(Keq)] = 1e+300
    Keq[Keq>1e+300] = 1e+300
    Keq[Keq<1e-16] = 1e-16

    active_reactions_mod_df['Keq'] = Keq
    

    
    ## Set some optimization hyperparameters parameters

    Mb = 100  #Big M relaxation parameter for flux consistency constraints
    zeta = 0.01
    linear_solver = 'default'
    hsllib=""
    max_cpu_time = 800000
    max_iter = 10000
    acceptable_tol = 1e-6
    solver_tol = 1e-7

    # override hyperparameters through input list
    if 'Mb' in hyperparameterDict:
        Mb = hyperparameterDict['Mb']

    if 'zeta' in hyperparameterDict:
        zeta = hyperparameterDict['zeta']

    if 'linear_solver' in hyperparameterDict:
        linear_solver = hyperparameterDict['linear_solver']

    if 'hsllib' in hyperparameterDict:
        hsllib = hyperparameterDict['hsllib']

    if 'max_cpu_time' in hyperparameterDict:
        max_cpu_time = hyperparameterDict['max_cpu_time']

    if 'max_iter' in hyperparameterDict:
        max_iter = hyperparameterDict['max_iter']

    if 'acceptable_tol' in hyperparameterDict:
        acceptable_tol = hyperparameterDict['acceptable_tol']

    if 'solver_tol' in hyperparameterDict:
        solver_tol = hyperparameterDict['solver_tol']

    delta_concentration = 0.0
    delta_flux = 0.0
    isPerturbConcentrations = False
    if 'delta_concentration' in hyperparameterDict:
        delta_concentration = hyperparameterDict['delta_concentration']
        isPerturbConcentrations=True
    
    isPerturbFlux = False
    if 'delta_flux' in hyperparameterDict:
        delta_flux = hyperparameterDict['delta_flux']
        isPerturbFlux=True

    ''' 
    #################################
    RE-ORDER OF METABOLITES
    #################################

    This is done to making splitting the stoichiometric matrix cleaner for computing the initial solution estimate for the optimization.


    '''

    maxPassThrough=20
    passNumber=0
    isNotAligned = True
    alignmentTol = 95
    unalignedMetabolites=[]
    metabolites_df.reset_index(inplace = True) #set a reference index for all of the metabolites, order must be preserved from this point forward, including Stoich matrix


    while isNotAligned and passNumber < maxPassThrough:

        # print("======================================")
        # print("           Pass number : "+str(passNumber)+ "           ")
        # print("======================================")

        # set metabolite to implicit
        # need to determine which metabolite to set to implicit
        if unalignedMetabolites:
            metabolites_df = SetMetaboliteToimplicit(metabolites_df, active_reactions,S_active_df,unalignedMetabolites)
            unalignedMetabolites=[]



        metabolites_df.sort_values(by = ['Variable','Steady_State'],ascending = [False,True],inplace = True)

        #ensure that S is sorted the same way
        metabolite_name_order = list(metabolites_df['index'])
        active_reactions_df = active_reactions_mod_df.copy()
        S_active_df = S_active.copy()
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

        m_tot_idx = np.array(metabolites_df.index)
        m_var_idx = np.array(metabolites_df[(metabolites_df['Variable']==True)].index)
        m_implicit_idx = np.array(metabolites_df[(metabolites_df['Steady_State']=='Implicit')].index)
        m_var_explicit_idx = np.array(metabolites_df[(metabolites_df['Variable']==True) & (metabolites_df['Steady_State'] == 'Explicit' )].index)
        m_var_implicit_idx = np.array(metabolites_df[(metabolites_df['Variable']==True) & (metabolites_df['Steady_State'] == 'Implicit' )].index)
        m_fixed_idx = np.array(metabolites_df[(metabolites_df['Variable']==False)].index)

        ## check for consistency
        if len(m_var_explicit_idx) + len(m_var_implicit_idx) + len(m_fixed_idx) != len(m_tot_idx): print('Metabolite Assignments Are Inconsistent!!!')

        m_var_net_flux_restricted_idx = np.array(metabolites_df[ (metabolites_df['Net_Flux']== 1) | (metabolites_df['Net_Flux']==-1) ].index )


        ### FLUX INDEXES
        active_reactions_df.reset_index(inplace = True) # set a reference index for all of the fluxes, order must be preserved from this point forward including the Stoich matrix
        flux_tot_idx = np.array(active_reactions_df.index)
        flux_free_reg_idx = np.array(active_reactions_df[ (active_reactions_df['Flux_Direction'] == 0) & (active_reactions_df['Allow_Regulation'] ==1)  ].index)
        flux_free_unreg_idx =  np.array(active_reactions_df[ (active_reactions_df['Flux_Direction'] == 0) & (active_reactions_df['Allow_Regulation'] ==0)  ].index)
        flux_fxsgn_reg_idx =  np.array(active_reactions_df[ ( (active_reactions_df['Flux_Direction' ]== 1) | (active_reactions_df['Flux_Direction']==-1) ) & (active_reactions_df['Allow_Regulation'] ==1) ].index)
        flux_fxsgn_unreg_idx =  np.array(active_reactions_df[ ( (active_reactions_df['Flux_Direction' ]== 1) | (active_reactions_df['Flux_Direction']==-1) ) & (active_reactions_df['Allow_Regulation'] ==0) ].index)

        if len(flux_free_reg_idx) + len(flux_free_unreg_idx) + len(flux_fxsgn_reg_idx) + len(flux_fxsgn_unreg_idx)  != len(flux_tot_idx) : print('Flux Assignments Are Inconsistent!')


        '''
        ##################################
        ###### Get the parameters for the optimization
        ####################################
        '''

        ''' 
        Get the equilibrium constants
        '''

        '''
        The fixed metabolite concentrations
        '''
        
        T = 298.15
        R = 8.314e-03
        RT = R*T
        N_avogadro = 6.022140857e+23
        VolCell = 1.0e-15
        Concentration2Count = N_avogadro * VolCell


        #number of variable metabolites

        m_conc_log_counts = np.log( metabolites_df['Conc'].values * Concentration2Count, dtype=np.float64)

        m_fixed_log_counts = m_conc_log_counts[m_fixed_idx]


        
        '''
        The Variable Metabolite Bounds
        '''

        m_log_count_upper_bound = np.log( metabolites_df['Upper_Bound'].values * Concentration2Count,dtype=np.float64 )
        m_log_count_lower_bound = np.log( metabolites_df['Lower_Bound'].values * Concentration2Count,dtype=np.float64 )


        # raise the upper bound of a metabolite set to the system maximum
        if 'raise_upper_bounds' in simulationDict.keys():
            for species in simulationDict['raise_upper_bounds']:
                # in order to raise the bound on a metabolite the bound must exist i.e metabolite be variable
                if metabolites.loc[species,'Variable']:
                    metabolites.at[species,'Target Conc']= (sys.float_info.max)
                    metab_idx = np.where(np.array(metabolite_name_order) == species)[0][0]
                    m_log_count_upper_bound[metab_idx]=(sys.float_info.max)
                else:
                    print("'raise_bound': metabolite "+str(species)+" must be variable")
            

        if 'redox_pairs' in simulationDict.keys():
            for redox_pair in simulationDict['redox_pairs']:
                if len(redox_pair)==2:
                    # check whether either pair element is already fixed
                    name_1 = redox_pair[0]
                    name_2 = redox_pair[1]

                    isVar_2 = metabolites.loc[name_2,'Variable']

                    if isVar_2:
                        # change upper bound for redox species 2
                        metabolites.at[name_2,'Target Conc']= (sys.float_info.max)
                        metab_idx = np.where(np.array(metabolite_name_order) == name_2)[0][0]
                        m_log_count_upper_bound[metab_idx]=(sys.float_info.max)
                    else:
                        print("'redox_pairs': metabolite "+str(name_2)+" must be variable")
                else:
                    print("Error redox_pairs should have two elements each")

        if 'specify_upper_bound' in simulationDict.keys():
            boundsDict=simulationDict['specify_upper_bound']
            for species in boundsDict.keys():
                if metabolites.loc[species,'Variable']:
                    metab_idx = np.where(np.array(metabolite_name_order) == species)[0][0]
                    m_log_count_upper_bound[metab_idx]=np.log((boundsDict[species])*Concentration2Count)
                else:
                    print("'specify_upper_bound': metabolite "+str(species)+" must be variable")

        if 'specify_lower_bound' in simulationDict.keys():
            boundsDict=simulationDict['specify_lower_bound']
            for species in boundsDict.keys():
                if metabolites.loc[species,'Variable']:
                    metab_idx = np.where(np.array(metabolite_name_order) == species)[0][0]
                    m_log_count_lower_bound[metab_idx]=np.log((boundsDict[species])*Concentration2Count)
                else:
                    print("'specify_lower_bound': metabolite "+str(species)+" must be variable")


        m_var_log_count_upper_bound = m_log_count_upper_bound[m_var_idx]
        m_var_log_count_lower_bound = m_log_count_lower_bound[m_var_idx]

        '''
        Stoichiometric Matrix
        '''
        S = S_active_df.values


        ''' 
        The Objective Flux Indexes
        '''
        obj_coefs = np.array(active_reactions_df['Obj_Coefs'].values, dtype=np.float64)
        obj_coefs = np.reshape(obj_coefs,(1,len(obj_coefs)))

        secondary_obj_coefs=[]
        primary_obj_tolerance=100

        if 'secondary_objective' in simulationDict.keys():
            secondary_obj_coefs=np.zeros(len(active_reactions_df))

            for i in range(len(active_reactions_df.index)):

                if active_reactions_df.at[i,'index'] in simulationDict['secondary_objective']:

                    secondary_obj_coefs[i] = simulationDict['secondary_objective'][active_reactions_df.at[i,'index']]


        if 'primary_obj_tolerance' in simulationDict.keys():

            primary_obj_tolerance = simulationDict['primary_obj_tolerance']


        ''' 
        ### COMPUTE THE INITIAL CONDITION
        '''

        '''     
        First we compute the gradient direction of the fluxes with respect to the objective projected onto the steady state condition.

        Then we search for a variable metabolite solution that can produce fluxes close in direction to this gradient.

        '''

        S_T = S    # S_T is the Stoich matrix with rows as reactions, columns as metabolites
        S = np.transpose(S_T) # this now is the Stoich matrix with rows metabolites, and columns reactions

        n_react = S_T.shape[0]

        #Find submatrix corresponding to the explicit steady state metabolites, these define our steady state conditions
        S_e_T = np.delete(S_T,m_implicit_idx,1)
        S_e = np.transpose(S_e_T)


        #find a basis for the nullspace of S_e, this defines our set of steady state flux solutions that are feasible.
        Se_N = spL.null_space(S_e)
        dSe_N = np.shape(Se_N)[1] # the dimension of the nullspace


        #Find the gradient direction for the objective projected into the steady state solutions
        beta_grad = np.matmul(obj_coefs,Se_N)
        beta_grad = np.transpose(beta_grad)

        y_grad = np.matmul(Se_N,beta_grad)

        y_grad = np.reshape(y_grad,(len(y_grad),1))

        # We split S into the submatrix corresponding to variable metabolites, and the submatrix corresponding to fixed metabolites

        S_f_T = np.delete(S_T,m_var_idx,1)
        S_f = np.transpose(S_f_T)

        S_v_T = np.delete(S_T,m_fixed_idx,1)
        S_v = np.transpose(S_v_T)

        #Compute initial condtion
        K_v = np.reshape(Keq,(len(Keq),1))
        m_fixed_log_counts = np.reshape(m_fixed_log_counts,(len(m_fixed_log_counts),1))

        #### first we want to find the matrix that is S_v^T with the signs appropriately adjusted for - y_grad
        S_v_T_sgn = S_v_T*np.sign(-y_grad)
        S_v_sgn = np.transpose(S_v_T_sgn)
        S_f_T_sgn = S_f_T*np.sign(-y_grad)

        # use least squares method to determine starting optimiser point
        useAnnealing = False

        try:

            if useAnnealing:

                print('Initialize using simulated annealing')

                print('Use least squares to initialise log concentrations')
                S_id = np.eye(n_react)
                A = np.concatenate([S_v_T_sgn,-S_id],axis = 1)

                ### then we compute the right hand side
                v = np.matmul(S_f_T_sgn,m_fixed_log_counts) - np.log(K_v)*np.sign(-y_grad)

                #construct the bounds
                x_upper = np.concatenate([m_var_log_count_upper_bound,1000*np.ones(n_react)])
                x_lower = np.concatenate([m_var_log_count_lower_bound, np.zeros(n_react)])
                
                startTime = time.time()
                try:
                    opt_out = lsq_linear(A,-np.ravel(v),bounds = (x_lower,x_upper))
                except:
                    raise Exception('lsq_linear error')
                endTime = time.time()

                print("Time in lsq_linear: "+str(endTime-startTime))

                n_out = opt_out['x']
                n_out = n_out[0:len(m_var_log_count_upper_bound)]
                n_out = np.reshape(n_out,(len(n_out),1))
                n_ini_ls = np.ravel(n_out) 



                v_new = np.log(K_v) - np.matmul(S_f_T,m_fixed_log_counts)

                beta_ini_projection = np.ravel(np.matmul(np.transpose(Se_N),v_new))

                numberEta =  len(m_var_log_count_upper_bound)

                maxBeta=1#200
                minBeta=-1#-200

                for i in range(len(beta_ini_projection)):
                    if beta_ini_projection[i] > maxBeta:
                        beta_ini_projection[i] = maxBeta
                    elif beta_ini_projection[i] <minBeta:
                        beta_ini_projection[i] = minBeta
                        

                for i in range(len(n_ini_ls)):
                    if n_ini_ls[i] > m_var_log_count_upper_bound[i]:
                        n_ini_ls[i] = m_var_log_count_upper_bound[i]
                    elif n_ini_ls[i] <m_var_log_count_lower_bound[i]:
                        n_ini_ls[i] = m_var_log_count_lower_bound[i]

                upper = m_var_log_count_upper_bound.tolist() +  (maxBeta*np.ones(len(beta_grad))).tolist()
                lower = m_var_log_count_lower_bound.tolist() + (minBeta*np.ones(len(beta_grad))).tolist()

                boundZip =list(zip(lower,upper))
                print('Use least squares solution and projection of beta to seed annealing')
                x_init = np.concatenate((n_ini_ls, beta_ini_projection),axis=0)


                isAnneal=False
                isBasin=True
                if isAnneal:
                    print('Use dual_annealing')
                    startTime=time.time()
                    opt_out= dual_annealing(signFunc, maxiter = 1000, bounds=boundZip, x0 = x_init, args=(v_new,S_v_T,Se_N,numberEta))
                    endTime=time.time()
                    print("Time in dual_annealing: "+str(endTime-startTime))

                    percentAlignment = 100*(numberEta-signFunc(opt_out['x'],v_new,S_v_T,Se_N,numberEta))/numberEta
                    print('Dual annealing status : %s' % opt_out['message'])
                    print('Number of Function evaluations : %d' % opt_out['nfev'])
                    print('Annealing return function value : %d' % opt_out.fun)
                    print('Annealing return function value (a.k.a number misaligned signs) : %d' % signFunc(opt_out['x'],v_new,S_v_T,Se_N,numberEta))
                    print('Percentage sign alignment : %d ' % percentAlignment)

                elif isBasin:
                    print('Use basinhopping')
                    minimizer_kwargs = { "method": "L-BFGS-B","bounds":boundZip,"args": (v_new,S_v_T,Se_N,numberEta) }
                    startTime=time.time()
                    opt_out = basinhopping(signFunc, x0 = x_init, minimizer_kwargs=minimizer_kwargs,niter=5000)
                    endTime=time.time()
                    print("Time in basinhopping: "+str(endTime-startTime))
                
                    percentAlignment = 100*(numberEta-signFunc(opt_out['x'],v_new,S_v_T,Se_N,numberEta))/numberEta
                    print('Basinhopping status : %s' % opt_out['message'])
                    print('Number of Function evaluations : %d' % opt_out['nfev'])
                    print('Annealing return function value : %d' % opt_out.fun)
                    print('Basinhopping return function value (a.k.a number misaligned signs) : %d' % signFunc(opt_out['x'],v_new,S_v_T,Se_N,numberEta))
                    print('Percentage sign alignment : %d ' % percentAlignment)
        
                
                misAlignedMetabolites=[]
                isReportMisalignment = True
                alignmentScore=0

                if isReportMisalignment:
                    alignmentScore= 100*(numberEta-signFunc(opt_out['x'],v_new,S_v_T,Se_N,numberEta))/numberEta
                    print('alignment score '+str(alignmentScore))
                    if alignmentScore > 0.1:
                        # return the index of the misaligned flux
                        misalignedFlux = getMisalignedSigns(opt_out['x'],v_new,S_v_T,Se_N,numberEta)
                        if len(misalignedFlux) >0:
                            for i in misalignedFlux:
                                for j in range(len(S_active.loc[active_reactions.index[i]])):
                                    sval=S_active.loc[active_reactions.index[i]][j]
                                    if sval !=0:
                                        misAlignedMetabolites.append(metabolites.index[j])
                    else:
                        print("completly misaligned, check biomass function and reactions")

                    
                    if alignmentTol<alignmentScore:
                        # aligned enough
                        print('Aligned enough')
                        isNotAligned= False
                    else:
                        # pass the misaligned metabolites to the next pass through
                        print('Not aligned enough')
                        unalignedMetabolites = misAlignedMetabolites

                else:
                    # not determining misalignment so set to set to aligned
                    isNotAligned= False

                annealOut = np.ravel(opt_out['x'])
                n_ini = annealOut[0:numberEta]
                beta_ini = annealOut[numberEta:]

                y_ini = np.ravel(np.matmul(Se_N,beta_ini))
                

            else:
                # use the least squares method
                ### We append the identity for our slack variables
                S_id = np.eye(n_react)
                A = np.concatenate([S_v_T_sgn,-S_id],axis = 1)

                ### then we compute the right hand side
                rhs=np.log(K_v)*np.sign(-y_grad)
                
                v = np.matmul(S_f_T_sgn,m_fixed_log_counts) - np.log(K_v)*np.sign(-y_grad)

                #construct the bounds
                x_upper = np.concatenate([m_var_log_count_upper_bound,1000*np.ones(n_react)])
                x_lower = np.concatenate([m_var_log_count_lower_bound, np.zeros(n_react)])
                try:
                    opt_out = lsq_linear(A,-np.ravel(v),bounds = (x_lower,x_upper))
                except:
                    raise Exception('lsq_linear error')
                n_out = opt_out['x']
                n_out = n_out[0:len(m_var_log_count_upper_bound)]
                n_out = np.reshape(n_out,(len(n_out),1))

                v_true = np.matmul(S_f_T,m_fixed_log_counts) - np.log(K_v)
                flux_out = np.matmul(S_v_T,n_out) + v_true

                n_ini = np.ravel(n_out)    #estimate of the variable metabolite values
                y_ini = -1e0*np.ravel(flux_out) # estimate of the reaction fluxes

            passNumber +=1

            if(isPerturbConcentrations):
                # perturb concentrations that are not fixed
                for i in range(len(n_ini)):

                    val = np.exp(n_ini[i])
                    randChange = nprd.uniform(0.0, delta_concentration)
                    direction = nprd.choice([-1,1])

                    perturbed_val = n_ini[i]
                    if val + direction*randChange > 0.0:
                        # feasible
                        perturbed_val = np.log(val + direction*randChange)

                    n_ini[i] = perturbed_val

            # apply perturbation

            if (isPerturbFlux):

                flux__directions = np.array(active_reactions_df['Flux_Direction'].values, dtype=np.float64)
                for i in range(0,len(y_ini)):

                    oldVal = y_ini[i]
                    randChange = nprd.uniform(-delta_flux, delta_flux)
                    absFlux=np.abs(y_ini[i])

                    flux__direction = flux__directions[i]

                    if flux__direction ==1:
                        if absFlux <= delta_flux:
                            # flux is small therefore test a change in sign
                            direction = nprd.choice([-1,1])
                            y_ini[i] = direction*y_ini[i] + randChange
                        else:
                            y_ini[i] = y_ini[i] + randChange

                        if y_ini[i] <0:
                            y_ini[i] = -1*y_ini[i]

                    elif flux__direction ==1:
                        if absFlux <= delta_flux:
                            # flux is small therefore test a change in sign
                            direction = nprd.choice([-1,1])
                            y_ini[i] = direction*y_ini[i] + randChange
                        else:
                            y_ini[i] = y_ini[i] + randChange

                        if y_ini[i] >0:
                            y_ini[i] = -1*y_ini[i]

                    else:
                        if absFlux <= delta_flux:
                            # flux is small therefore test a change in sign
                            direction = nprd.choice([-1,1])
                            y_ini[i] = direction*y_ini[i] + randChange
                        else:
                            y_ini[i] = y_ini[i] + randChange

            y_ini = zeta*y_ini


            ''' 
            #################
            ##### COLLECT INFORMATION REQUIRED FOR CONSTRUCTING THE OPTIMIZATION PROBLEM
            ####################

            will store all relevant inputs and hyperparameters in a dictionary
            '''
            opt_parms = {}

            #PROBLEM INDEXES , to identify the constraint types neccessary for metabolites and fluxes
            ### metabolites
            opt_parms['m_tot_idx'] = m_tot_idx
            opt_parms['m_var_idx'] = m_var_idx
            opt_parms['m_implicit_idx'] = m_implicit_idx
            opt_parms['m_var_explicit_idx'] = m_var_explicit_idx
            opt_parms['m_var_implicit_idx'] = m_var_implicit_idx
            opt_parms['m_fixed_idx'] = m_fixed_idx
            opt_parms['m_var_net_flux_restricted_idx'] = m_var_net_flux_restricted_idx


            ### fluxes
            opt_parms['flux_tot_idx'] = flux_tot_idx
            opt_parms['flux_free_reg_idx'] = flux_free_reg_idx
            opt_parms['flux_free_unreg_idx'] = flux_free_unreg_idx
            opt_parms['flux_fxsgn_reg_idx'] = flux_fxsgn_reg_idx
            opt_parms['flux_fxsgn_unreg_idx'] = flux_fxsgn_unreg_idx



            #PROBLEM DATA

            #metabolites
            opt_parms['m_log_count_upper_bound'] = m_log_count_upper_bound
            opt_parms['m_log_count_lower_bound'] = m_log_count_lower_bound
            opt_parms['m_conc_log_counts'] = m_conc_log_counts
            opt_parms['m_net_flux_restrictions'] = np.array(metabolites_df['Net_Flux'].values,dtype=np.float64)


            #fluxes
            opt_parms['Keq'] = Keq
            opt_parms['obj_coefs'] = obj_coefs
            opt_parms['secondary_obj_coefs'] = secondary_obj_coefs
            opt_parms['primary_obj_tolerance'] = primary_obj_tolerance
            opt_parms['flux_direction_restrictions'] = np.array(active_reactions_df['Flux_Direction'].values, dtype=np.float64)


            #Stoich
            opt_parms['S'] = S
            opt_parms['Se_N'] = Se_N

            #Optimization Hyperperameters
            opt_parms['Mb'] = Mb
            opt_parms['solverTol'] = solver_tol
            opt_parms['solverAcceptableTol'] = acceptable_tol
            opt_parms['max_iter'] = max_iter
            opt_parms['max_cpu_time'] = max_cpu_time
            opt_parms['linear_solver'] = linear_solver
            opt_parms['hsllib'] = hsllib



            return n_ini, y_ini, opt_parms, 0

        except:
        
            return [], [], {}, 1





''' 
This function computes optimal solutions, expects to see metabolites
that have the possibility of being free variables, but for which we do not
compute explicit steady state conditions for the fluxes, providing greater
flexibility to the solutions that we can find

Inputs:
    - n_ini: the vector for the initial metabolite log concentration counts in the model
    - y_ini: the vector for the initial flux through the reactions
    - opt_parms: dictionary for the optimization parameters

Returns:
    - y_sol: the vector of flux values of the optimized solution
    - alpha_sol: the vector of enzymatic regulation values of the optimized solution
    - n_sol: the vector of variable metabolite log concentration counts values of the optimized solution
    - np.ravel(m_fixed_log_counts): the vector of fixed metabolite log concentration counts values
    - m_log_count_lower_bound: the vector of the log concentration counts for the lower bound of the variable metabolites
    - m_log_count_upper_bound: the vector of the log concentration counts for the upper bound of the variable metabolites
    - solved: boolean switch for whether the optimizer sucessfully converged to a solution
    - status_obj: the status of the optimizer solution
    - obj_val: the objective value for the secondary objective 


'''

def flux_ent_opt(n_ini, y_ini, opt_parms):

    '''
    ###################
    #### UNPACK THE supplied optimization parameters
    ###################
    '''


    #Optimization Hyperperameters
    Mb = opt_parms['Mb']
    solverTol = opt_parms['solverTol']
    solverAcceptableTol = opt_parms['solverAcceptableTol']
    max_iter = opt_parms['max_iter']
    max_cpu_time = opt_parms['max_cpu_time']
    linear_solver = opt_parms['linear_solver']
    hsllib = opt_parms['hsllib']
    feasibilityCheck = opt_parms['feasibility_check']
    feasibility_as_input = opt_parms['feasibility_as_input']
    useAnnealing = opt_parms['annealing_check']

    #PROBLEM INDEXES , to identify the constraint types neccessary for metabolites and fluxes
    ### metabolites
    m_tot_idx = opt_parms['m_tot_idx']
    m_var_idx = opt_parms['m_var_idx']
    m_implicit_idx = opt_parms['m_implicit_idx']
    m_var_explicit_idx = opt_parms['m_var_explicit_idx']
    m_var_implicit_idx = opt_parms['m_var_implicit_idx']
    m_fixed_idx = opt_parms['m_fixed_idx']
    m_var_net_flux_restricted_idx = opt_parms['m_var_net_flux_restricted_idx']

    ### fluxes
    flux_tot_idx = opt_parms['flux_tot_idx']
    # flux_restricted_direction_idx = opt_parms['flux_restricted_direction_idx']
    # flux_free_idx = opt_parms['flux_free_idx']
    
    flux_free_reg_idx = opt_parms['flux_free_reg_idx'] 
    flux_free_unreg_idx = opt_parms['flux_free_unreg_idx']
    flux_fxsgn_reg_idx = opt_parms['flux_fxsgn_reg_idx']
    flux_fxsgn_unreg_idx = opt_parms['flux_fxsgn_unreg_idx']


    #PROBLEM DATA
    #metabolites
    m_log_count_upper_bound = opt_parms['m_log_count_upper_bound']
    m_log_count_lower_bound = opt_parms['m_log_count_lower_bound']
    m_conc_log_counts = opt_parms['m_conc_log_counts']
    m_net_flux_restrictions = opt_parms['m_net_flux_restrictions']


    #fluxes
    Keq = opt_parms['Keq']
    obj_coefs = np.ravel(opt_parms['obj_coefs'])
    if feasibilityCheck or feasibility_as_input:
        obj_coefs = np.zeros(len(obj_coefs))

    flux_direction_restrictions = opt_parms['flux_direction_restrictions']


    #Stoich
    S = opt_parms['S']
    Se_N = opt_parms['Se_N']



    #Compute partitions of data needed for the optimization

    #Some useful count parameters
    n_react = S.shape[1]
    n_M = S.shape[0]
    n_M_f = len(m_fixed_idx)
    n_M_v = len(m_var_idx)
    dSe_N = Se_N.shape[1]


    #Stoich
    S_T = np.transpose(S)
    S_f_T = np.delete(S_T,m_var_idx,1) #fixed metabolite S submatrix
    S_f = np.transpose(S_f_T)
    S_v_T = np.delete(S_T,m_fixed_idx,1) #variable metabolite S submatrix
    S_v = np.transpose(S_v_T)


    #metabolites
    m_fixed_log_counts = m_conc_log_counts[m_fixed_idx]
    m_fixed_log_counts = np.reshape(m_fixed_log_counts,(len(m_fixed_log_counts),1))

    ''' 
    SET ALL THE VARIABLE INITIALIZATIONS

    this derives the initial values for all other variables from n_ini and y_ini
    '''
    beta_ini = np.ravel(np.matmul(np.transpose(Se_N),y_ini) )
    y_ini = 1e0*np.matmul(Se_N,beta_ini)
    y_ini = np.ravel(y_ini)
    b_ini = np.matmul(S_v_T,np.reshape(n_ini,(len(n_ini),1))) + np.matmul(S_f_T, m_fixed_log_counts ) 
    b_ini = np.reshape(b_ini,len(b_ini))
    h_ini = np.sign(y_ini)*( np.log(2) - np.log( np.abs(y_ini) + np.sqrt( np.power(y_ini,2) + 4 ) )  )


    ''' 
    ### SET ADDITIONAL INDEXES FOR OPTIMIZATION
    '''

    beta_idx = np.arange(0,Se_N.shape[1])


    '''
    CONSTRUCT THE OPTIMIZATION PROBLEM IN PYOMO
    '''

    #Pyomo Model Definition
    ######################
    m =  pe.ConcreteModel()

    #Input the model parameters

    #set the indices
    #####################
    m.react_idx = pe.Set(initialize = flux_tot_idx) 
    m.TotM_idx = pe.Set(initialize = m_tot_idx)
    m.VarM_idx = pe.Set(initialize = m_var_idx)
    m.FxdM_idx = pe.Set(initialize = m_fixed_idx)
    m.beta_idx = pe.Set(initialize = beta_idx)

    m.net_flux_idx = pe.Set(initialize = m_var_net_flux_restricted_idx)

    m.flux_free_reg_idx =  pe.Set(initialize =flux_free_reg_idx ) 
    m.flux_free_unreg_idx = pe.Set(initialize = flux_free_unreg_idx ) 
    m.flux_fxsgn_reg_idx =pe.Set(initialize =flux_fxsgn_reg_idx ) 
    m.flux_fxsgn_unreg_idx = pe.Set(initialize = flux_fxsgn_unreg_idx ) 


    # Stochiometric matrix
    ###########################
    S_idx = list(itertools.product(np.arange(0,n_M),np.arange(0,n_react)))
    S_vals = list(np.reshape(S,[1,n_M*n_react])[0])
    S_dict = dict(list(zip(S_idx,S_vals)))

    m.S = pe.Param(S_idx ,initialize = S_dict,mutable = True)


    ## Nullspace basis Matrix
    ####################
    SvN_idx = list(itertools.product(np.arange(0,n_react),np.arange(0,dSe_N)))
    SvN_vals = list(np.reshape(Se_N,[1,n_react*dSe_N])[0])
    SvN_dict = dict(list(zip(SvN_idx,SvN_vals)))

    m.SvN=pe.Param(SvN_idx, initialize = SvN_dict)


    # Reaction Equilibrium constants
    ##################
    K_dict = dict(list(zip(flux_tot_idx,Keq) ))
    m.K=pe.Param(m.react_idx, initialize = K_dict)


    # Fixed metabolite log counts
    FxdM_dict = dict( list( zip(m_fixed_idx,np.ravel(m_fixed_log_counts)) ) )
    m.FxdM = pe.Param(m.FxdM_idx, initialize = FxdM_dict)


    # Bounds on the log of the metabolites
    #########
    M_ubnd_dict = dict(list(zip(m_var_idx,m_log_count_upper_bound[m_var_idx])))
    m.VarM_ubnd = pe.Param(m.VarM_idx, initialize = M_ubnd_dict)

    M_lbnd_dict = dict(list(zip(m_var_idx,m_log_count_lower_bound[m_var_idx])))
    m.VarM_lbnd = pe.Param(m.VarM_idx, initialize = M_lbnd_dict)


    # Objective Coefficients
    ##################
    obj_c_dict = dict(list(zip(flux_tot_idx,obj_coefs) ))
    m.obj_coefs=pe.Param(m.react_idx, initialize = obj_c_dict)


    #SET the Variables
    #############################


    ## Variable metabolites (log)
    ######################

    Mini_dict = dict(list(zip(m_var_idx,n_ini)))
    m.VarM = pe.Var(m.VarM_idx,initialize = Mini_dict)

    # steady state fluxes
    yini_dict = dict(list(zip(flux_tot_idx,y_ini)))
    m.y = pe.Var(m.react_idx, initialize = yini_dict)

    # flux null space representation
    betaini_dict = dict(list(zip(beta_idx,beta_ini)))
    m.beta = pe.Var(m.beta_idx, initialize = betaini_dict)

    # Steady state condition RHS
    bini_dict = dict(list(zip(flux_tot_idx,b_ini)))
    m.b = pe.Var(m.react_idx, initialize = bini_dict)

    hini_dict = dict(list(zip(flux_tot_idx,h_ini)))
    m.h = pe.Var(m.react_idx, initialize = hini_dict)

    # Set the Constraints
    #############################

    #Steady State Constraint with null space representation
    def flux_null_rep(m,i):
        return m.y[i]   ==  sum( m.SvN[(i,j)]*m.beta[j]  for j in m.beta_idx )
    m.fxnrep_cns = pe.Constraint(m.react_idx, rule = flux_null_rep)



    def steady_state_metab(m,j):
        return m.b[j]  == sum( m.S[(k,j)]*m.VarM[k] for k in m.VarM_idx ) + sum( m.S[(k,j)]*m.FxdM[k] for k in m.FxdM_idx ) 
    m.ssM_cns = pe.Constraint(m.react_idx, rule = steady_state_metab)




    def num_smooth_cns(m,i):
        return m.h[i] ==(m.y[i]*1e50/(abs(m.y[i])*1e50 + 1e-50))*(pe.log(2) -  pe.log(abs(m.y[i]) + pe.sqrt(m.y[i]**2 + 4 ) )  ) 
    m.nms_cns = pe.Constraint(m.react_idx, rule = num_smooth_cns)

    def sign_constraint(m,i):
        return (pe.log(m.K[i]) - m.b[i])*m.y[i] >= 0
    m.sign_y_cns = pe.Constraint(m.react_idx, rule = sign_constraint)


    ''' 
    ##############################################################
    ##############################################################

    
    SPECIFY ALL OF THE FLUX CONSTRAINTS


    This contains four constraint possiblities, 
    all combinations of: 

    (unregulated, regluated) #whether or not the activity coefficient can be less than one
    
    (fixed direction, free direction) #if the sign of the flux will be restricted.


    ##############################################################
    ##############################################################

    '''


    ''' 
    FREE FLUXES ALLOWED TO REGULATE
    '''

    if len(flux_free_reg_idx)>0:



        y_free_reg_sign_ini_dict = dict(list(zip(flux_free_reg_idx,np.sign(y_ini[flux_free_reg_idx]) )))
        m.u_free_reg = pe.Var(m.flux_free_reg_idx,bounds=(-1,1),initialize = y_free_reg_sign_ini_dict)



        def y_sign_relax_free_reg(m,i):
            return m.u_free_reg[i] == (m.y[i]/(abs(m.y[i]) + 1e-50) )
        m.y_sign_relax_cns_free_reg = pe.Constraint(m.flux_free_reg_idx,rule = y_sign_relax_free_reg)



        def relaxed_reg_cns_upper_free_reg(m,i):
            return (  pe.log(m.K[i]) - m.b[i] ) <=  - m.h[i] + Mb*(  1 + m.u_free_reg[i] )  
        m.rxr_cns_up_free_reg = pe.Constraint(m.flux_free_reg_idx,rule = relaxed_reg_cns_upper_free_reg)


        def relaxed_reg_cns_lower_free_reg(m,i):
            return ( pe.log(m.K[i]) - m.b[i] ) >=  - m.h[i] - Mb*(1 - m.u_free_reg[i])
        m.rxr_cns_low_free_reg = pe.Constraint(m.flux_free_reg_idx,rule = relaxed_reg_cns_lower_free_reg)

     



        y_free_reg_sign_ini_dict = dict(list(zip(flux_free_reg_idx,np.sign(y_ini[flux_free_reg_idx]) )))
        m.w_free_reg = pe.Var(m.flux_free_reg_idx,bounds=(-1,1),initialize = y_free_reg_sign_ini_dict)
        def w_sign_relax(m,i):
            return m.w_free_reg[i] == (  (pe.log(m.K[i]) - m.b[i])  / ( abs(  (pe.log(m.K[i]) - m.b[i]) ) + 1e-50) )
        m.w_sign_relax_cns = pe.Constraint(m.flux_free_reg_idx,rule = w_sign_relax)
        def w_sign_constraint_upper(m,i):
             return m.y[i] <= 1e2*Mb*(1 + m.w_free_reg[i] )
        m.sign_w_y_upper_cns = pe.Constraint(m.flux_free_reg_idx, rule = w_sign_constraint_upper)
        def w_sign_constraint_lower(m,i):
             return m.y[i] >= -1e2*Mb*(1 - m.w_free_reg[i] )
        m.sign_w_y_lower_cns = pe.Constraint(m.flux_free_reg_idx, rule = w_sign_constraint_lower)


        def u_sign_constraint_upper(m,i):
             return  (pe.log(m.K[i]) - m.b[i]) <= Mb*(1 + m.u_free_reg[i] )
        m.sign_u_y_upper_cns = pe.Constraint(m.flux_free_reg_idx, rule = u_sign_constraint_upper)


        def u_sign_constraint_lower(m,i):
             return  (pe.log(m.K[i]) - m.b[i]) >= -Mb*(1 - m.u_free_reg[i] )
        m.sign_u_y_lower_cns = pe.Constraint(m.flux_free_reg_idx, rule = u_sign_constraint_lower)







    ''' 
    FREE FLUXES NOT ALLOWED TO REGULATE
    '''


    if len(flux_free_unreg_idx)>0:



        def reg_cns_free_equality(m,i):
            return ( m.b[i] - pe.log(m.K[i]) ) == m.h[i] 
        m.reg_cns_free_equality = pe.Constraint(m.flux_free_unreg_idx,rule = reg_cns_free_equality)





    ''' 
    FIXED SIGN FLUX ALLOWED TO REGULATE
    '''


    if len(flux_fxsgn_reg_idx)>0:


        y_fixed_sign_ini_dict =  dict(list(zip(flux_fxsgn_reg_idx,.5+.5*flux_direction_restrictions[flux_fxsgn_reg_idx] )))
        m.u_fxsgn = pe.Param(m.flux_fxsgn_reg_idx,initialize = y_fixed_sign_ini_dict)



        def y_sign_relax_fxsgn(m,i):
            return 2*m.u_fxsgn[i] - 1 == (m.y[i]/(abs(m.y[i]) + 1e-50) ) 
        m.y_sign_relax_cns_fixed = pe.Constraint(m.flux_fxsgn_reg_idx,rule = y_sign_relax_fxsgn)



        def relaxed_reg_cns_upper_fxsgn(m,i):
            return ( m.b[i] - pe.log(m.K[i]) ) >= m.h[i] - Mb*(m.u_fxsgn[i])  
        m.rxr_cns_up_fixed = pe.Constraint(m.flux_fxsgn_reg_idx,rule = relaxed_reg_cns_upper_fxsgn)


        def relaxed_reg_cns_lower_fxsgn(m,i):
            return ( m.b[i] - pe.log(m.K[i]) ) <= m.h[i] + Mb*(1 - m.u_fxsgn[i])
        m.rxr_cns_low_fixed = pe.Constraint(m.flux_fxsgn_reg_idx,rule = relaxed_reg_cns_lower_fxsgn)





    ''' 
    FIXED SIGN FLUXES NOT ALLOWED TO REGULATE
    '''


    if len(flux_fxsgn_unreg_idx)>0:


        
        y_fxsgn_unreg_ini_dict =  dict(list(zip(flux_fxsgn_unreg_idx,.5+.5*flux_direction_restrictions[flux_fxsgn_unreg_idx] )))
        m.u_fxsgn_unreg = pe.Param(m.flux_fxsgn_unreg_idx,initialize = y_fxsgn_unreg_ini_dict)



        def y_sign_relax_fxsgn_unreg(m,i):
            return 2*m.u_fxsgn_unreg[i] - 1 == (m.y[i]/(abs(m.y[i]) + 1e-50) ) 
        m.y_sign_fxsgn_unreg_cns = pe.Constraint(m.flux_fxsgn_unreg_idx,rule = y_sign_relax_fxsgn_unreg)




        def reg_cns_fxsgn_equality(m,i):
            return ( m.b[i] - pe.log(m.K[i]) ) == m.h[i] 
        m.reg_cns_fxsgn_equality = pe.Constraint(m.flux_fxsgn_unreg_idx,rule = reg_cns_fxsgn_equality)



    # Metabolite Net Flux Constraints

    if len(m_var_net_flux_restricted_idx)> 0 :

        net_flux_dict = dict(list(zip(m_var_net_flux_restricted_idx,m_net_flux_restrictions[m_var_net_flux_restricted_idx])))
        m.net_flux_sign = pe.Param(m.net_flux_idx, initialize = net_flux_dict)


        def net_flux_constraint(m,i):
            return m.net_flux_sign[i]*sum( m.S[(i,j)]*m.y[j] for j in m.react_idx ) >= 0
        m.net_flux_cns = pe.Constraint(m.net_flux_idx,rule = net_flux_constraint)


    # Variable metabolite upper and lower bounds
    def M_upper_cnstrnts(m,i):
        return  m.VarM[i] <= m.VarM_ubnd[i]
    m.VarM_ub_cns = pe.Constraint(m.VarM_idx,rule = M_upper_cnstrnts)


    def M_lower_cnstrnts(m,i):
        return  m.VarM[i] >= m.VarM_lbnd[i]
    m.VarM_lb_cns = pe.Constraint(m.VarM_idx,rule = M_lower_cnstrnts)



    # Set the Objective function

    def _Obj(m):
        return sum( m.y[j]*m.obj_coefs[j]  for j in m.react_idx ) 
    m.Obj_fn = pe.Objective(rule = _Obj, sense = pe.maximize) 


    #Find a Solution
    ####################


    #Set the solver to use
    opt=pe.SolverFactory('ipopt', solver_io='nl')


    #Set solver options
    if feasibilityCheck:
        max_iter = 2000

    if linear_solver == 'default' and hsllib == "":
        # neither option is set so default for both
        opts = {'max_iter': max_iter,
            'max_cpu_time': max_cpu_time,
            'tol':solverTol,
            'acceptable_tol':solverAcceptableTol
            }
        
    elif linear_solver == 'default':
        # then hsllib option is set
        opts = {'max_iter': max_iter,
            'max_cpu_time': max_cpu_time,
            'tol':solverTol,
            'acceptable_tol':solverAcceptableTol,
            'hsllib':hsllib}
        
    elif hsllib == "":
        # then linear_solver option is set
        opts = {'max_iter': max_iter,
            'max_cpu_time': max_cpu_time,
            'tol':solverTol,
            'acceptable_tol':solverAcceptableTol,
            'linear_solver': linear_solver} 
        
    else:
        # then both options are set
        opts = {'max_iter': max_iter,
            'max_cpu_time': max_cpu_time,
            'tol':solverTol,
            'acceptable_tol':solverAcceptableTol,
            'linear_solver': linear_solver,
            'hsllib':hsllib} 

    ## Solve the Model
    status_obj = opt.solve(m, options=opts, tee=False)

    obj_val = pe.value(m.Obj_fn)

    # find out if it terminated on optimal solution
    solved = 1
    if status_obj.solver.termination_condition == TerminationCondition.optimal:
        solved = 1
    else:
        solved = 0

    n_sol=np.zeros(n_M_v)
    b_sol = np.zeros(n_react)
    y_sol = np.zeros(n_react)
    beta_sol = np.zeros(dSe_N)
    h_sol = np.zeros(n_react)

    for i in flux_tot_idx:
        b_sol[i] = pe.value(m.b[i])
        y_sol[i] = pe.value(m.y[i])
        h_sol[i] = pe.value(m.h[i])

    for i in beta_idx:
        beta_sol[i] = pe.value(m.beta[i])

    for i in m_var_idx:
        n_sol[i] = pe.value(m.VarM[i])
        

    E_regulation = np.ones(len(y_sol))
    unreg_rxn_flux = np.ravel( rxn_flux(n_sol, np.ravel(m_fixed_log_counts),S_T, Keq, E_regulation) )
    alpha_sol = y_sol/unreg_rxn_flux
    alpha_sol = np.ravel(alpha_sol)


    return(y_sol, alpha_sol, n_sol,unreg_rxn_flux,np.ravel(m_fixed_log_counts),m_log_count_lower_bound,m_log_count_upper_bound, solved, status_obj, obj_val)


##### assume that initialized values are at an optimal solution for growth production

### add an additional constraint that restricts the objective value to be within epsilon of the the starting growth objective value

### search to maximize a secondary objective.

### possible to reduce the problem complexity by removing the gradient direction for growth from the steady state basis
### such that all changes would be neutral with respect to the growth objective.


'''
Function used to calculate the maximum flux through a secondary objective pathway while remaining within a tolerence 
of the primary objetive.


Inputs:
    - n_ini: the vector for the initial metabolite log concentration counts in the model, use the flux output from the primiary 
             objective optimization 
    - y_ini: the vector for the initial flux through the reactions, use the flux output from the primiary objective optimization 
    - opt_parms: dictionary for the optimization parameters

Returns:
    - y_sol: the vector of flux values of the optimized solution
    - alpha_sol: the vector of enzymatic regulation values of the optimized solution
    - n_sol: the vector of variable metabolite log concentration values of the optimized solution
    - status_obj: the status of the optimizer solution
    - obj_val: the objective value for the secondary objective 

'''

def Max_alt_flux(n_ini,y_ini, opt_parms):


    #Optimization Hyperperameters
    Mb = opt_parms['Mb']
    solverTol = opt_parms['solverTol']
    solverAcceptableTol = opt_parms['solverAcceptableTol']
    max_iter = opt_parms['max_iter']
    max_cpu_time = opt_parms['max_cpu_time']
    linear_solver = opt_parms['linear_solver']
    hsllib = opt_parms['hsllib']

    #PROBLEM INDEXES , to identify the constraint types neccessary for metabolites and fluxes
    ### metabolites
    m_tot_idx = opt_parms['m_tot_idx']
    m_var_idx = opt_parms['m_var_idx']
    m_implicit_idx = opt_parms['m_implicit_idx']
    m_var_explicit_idx = opt_parms['m_var_explicit_idx']
    m_var_implicit_idx = opt_parms['m_var_implicit_idx']
    m_fixed_idx = opt_parms['m_fixed_idx']
    m_var_net_flux_restricted_idx = opt_parms['m_var_net_flux_restricted_idx']

    ### fluxes
    flux_tot_idx = opt_parms['flux_tot_idx']
    flux_restricted_direction_idx = opt_parms['flux_restricted_direction_idx']
    flux_free_idx = opt_parms['flux_free_idx']
    



    #PROBLEM DATA
    #metabolites
    m_log_count_upper_bound = opt_parms['m_log_count_upper_bound']
    m_log_count_lower_bound = opt_parms['m_log_count_lower_bound']
    m_conc_log_counts = opt_parms['m_conc_log_counts']
    m_net_flux_restrictions = opt_parms['m_net_flux_restrictions']


    #fluxes
    Keq = opt_parms['Keq']
    obj_coefs = np.ravel(opt_parms['obj_coefs'])
    flux_direction_restrictions = opt_parms['flux_direction_restrictions']


    #Stoich
    S = opt_parms['S']
    Se_N = opt_parms['Se_N']




    primary_obj_coefs = obj_coefs

    secondary_obj_coefs = np.ravel(opt_parms['secondary_obj_coefs'])

    primary_obj_tolerance = opt_parms['primary_obj_tolerance']



    #Compute partitions of data needed for the optimization

    #Some useful count parameters
    n_react = S.shape[1]
    n_M = S.shape[0]
    n_M_f = len(m_fixed_idx)
    n_M_v = len(m_var_idx)
    dSe_N = Se_N.shape[1]


    #Stoich
    S_T = np.transpose(S)
    S_f_T = np.delete(S_T,m_var_idx,1) #fixed metabolite S submatrix
    S_f = np.transpose(S_f_T)
    S_v_T = np.delete(S_T,m_fixed_idx,1) #variable metabolite S submatrix
    S_v = np.transpose(S_v_T)



    #metabolites
    m_fixed_log_counts = m_conc_log_counts[m_fixed_idx]
    m_fixed_log_counts = np.reshape(m_fixed_log_counts,(len(m_fixed_log_counts),1))

    
    #Set the initial condition

    beta_ini = np.ravel(np.matmul(np.transpose(Se_N),y_ini) )



    b_ini = np.matmul(S_v_T,np.reshape(n_ini,(len(n_ini),1))) + np.matmul(S_f_T, m_fixed_log_counts ) 
    b_ini = np.reshape(b_ini,len(b_ini))
    h_ini = np.sign(y_ini)*( np.log(2) - np.log( np.abs(y_ini) + np.sqrt( np.power(y_ini,2) + 4 ) )  )


   
    ''' 
    ### SET ADDITIONAL INDEXES FOR OPTIMIZATION
    '''

    beta_idx = np.arange(0,Se_N.shape[1])


    '''
    CONSTRUCT THE OPTIMIZATION PROBLEM IN PYOMO
    '''


    #Pyomo Model Definition
    ######################
    m =  pe.ConcreteModel()

    #Input the model parameters

    #set the indices
    #####################
    m.react_idx = pe.Set(initialize = flux_tot_idx) 
    m.TotM_idx = pe.Set(initialize = m_tot_idx)
    m.VarM_idx = pe.Set(initialize = m_var_idx)
    m.FxdM_idx = pe.Set(initialize = m_fixed_idx)
    m.beta_idx = pe.Set(initialize = beta_idx)

    m.net_flux_idx = pe.Set(initialize = m_var_net_flux_restricted_idx)

    m.flux_fixed_idx = pe.Set(initialize = flux_restricted_direction_idx ) 
    m.flux_free_idx = pe.Set(initialize = flux_free_idx)

    

    # Stochiometric matrix
    ###########################
    S_idx = list(itertools.product(np.arange(0,n_M),np.arange(0,n_react)))
    S_vals = list(np.reshape(S,[1,n_M*n_react])[0])
    S_dict = dict(list(zip(S_idx,S_vals)))

    m.S = pe.Param(S_idx ,initialize = S_dict,mutable = True)


    ## Nullspace basis Matrix
    ####################
    SvN_idx = list(itertools.product(np.arange(0,n_react),np.arange(0,dSe_N)))
    SvN_vals = list(np.reshape(Se_N,[1,n_react*dSe_N])[0])
    SvN_dict = dict(list(zip(SvN_idx,SvN_vals)))

    m.SvN=pe.Param(SvN_idx, initialize = SvN_dict)


    # Reaction Equilibrium constants
    ##################
    K_dict = dict(list(zip(flux_tot_idx,Keq) ))
    m.K=pe.Param(m.react_idx, initialize = K_dict)


    # Fixed metabolite log counts
    FxdM_dict = dict( list( zip(m_fixed_idx,np.ravel(m_fixed_log_counts)) ) )
    m.FxdM = pe.Param(m.FxdM_idx, initialize = FxdM_dict)


    # Bounds on the log of the metabolites
    #########
    M_ubnd_dict = dict(list(zip(m_var_idx,m_log_count_upper_bound[m_var_idx])))
    m.VarM_ubnd = pe.Param(m.VarM_idx, initialize = M_ubnd_dict)

    M_lbnd_dict = dict(list(zip(m_var_idx,m_log_count_lower_bound[m_var_idx])))
    m.VarM_lbnd = pe.Param(m.VarM_idx, initialize = M_lbnd_dict)


    # # Objective Coefficients
    # ##################

    dm_idx = np.arange(0,1)
    m.dm_idx = pe.Set(initialize = dm_idx)


    # use a supplied list of secondary objective coefficients, not the ones in the dataframe
    # Objective Coefficients
    ##################
    obj_c_dict = dict(list(zip(flux_tot_idx,secondary_obj_coefs) ))
    m.obj_coefs=pe.Param(m.react_idx, initialize = obj_c_dict)




    #SET the Variables
    #############################


    ## Variable metabolites (log)
    ######################

    Mini_dict = dict(list(zip(m_var_idx,n_ini)))
    m.VarM = pe.Var(m.VarM_idx,initialize = Mini_dict)

    # steady state fluxes
    yini_dict = dict(list(zip(flux_tot_idx,y_ini)))
    m.y = pe.Var(m.react_idx, initialize = yini_dict)

    # flux null space representation
    betaini_dict = dict(list(zip(beta_idx,beta_ini)))
    m.beta = pe.Var(m.beta_idx, initialize = betaini_dict)

    # Steady state condition RHS
    bini_dict = dict(list(zip(flux_tot_idx,b_ini)))
    m.b = pe.Var(m.react_idx, initialize = bini_dict)

    hini_dict = dict(list(zip(flux_tot_idx,h_ini)))
    m.h = pe.Var(m.react_idx, initialize = hini_dict)

    # y sign variable or fixed depending.
    y_free_sign_ini_dict = dict(list(zip(flux_free_idx,.5+.5*np.sign(y_ini[flux_free_idx]) )))
    m.u_free = pe.Var(m.flux_free_idx,bounds=(0,1),initialize = y_free_sign_ini_dict)


    y_fixed_sign_ini_dict =  dict(list(zip(flux_restricted_direction_idx,.5+.5*flux_direction_restrictions[flux_restricted_direction_idx] )))
    m.u_fixed = pe.Param(m.flux_fixed_idx,initialize = y_fixed_sign_ini_dict)



    
    
    # Set the Constraints
    #############################

    #flux null space representation constraint
    def flux_null_rep(m,i):
        return m.y[i]   ==  sum( m.SvN[(i,j)]*m.beta[j]  for j in m.beta_idx )
    m.fxnrep_cns = pe.Constraint(m.react_idx, rule = flux_null_rep)

    

    def steady_state_metab(m,j):
        return m.b[j]  == sum( m.S[(k,j)]*m.VarM[k] for k in m.VarM_idx ) + sum( m.S[(k,j)]*m.FxdM[k] for k in m.FxdM_idx ) 
    m.ssM_cns = pe.Constraint(m.react_idx, rule = steady_state_metab)

    
    
    
    def num_smooth_cns(m,i):
        return m.h[i] ==(m.y[i]*1e50/(abs(m.y[i])*1e50 + 1e-50))*(pe.log(2) -  pe.log(abs(m.y[i]) + pe.sqrt(m.y[i]**2 + 4 ) )  ) 
    m.nms_cns = pe.Constraint(m.react_idx, rule = num_smooth_cns)
    

    def sign_constraint(m,i):
        return (pe.log(m.K[i]) - m.b[i])*m.y[i] >= 0
    m.sign_y_cns = pe.Constraint(m.react_idx, rule = sign_constraint)
    
    ''' 
    Free Flux Version of Sign Constraints
    this is when they are allowed to vary
    '''

    def y_sign_relax_free(m,i):
        return 2*m.u_free[i] - 1 == (m.y[i]/(abs(m.y[i]) + 1e-50) ) 
    m.y_sign_relax_cns_free = pe.Constraint(m.flux_free_idx,rule = y_sign_relax_free)



    def relaxed_reg_cns_upper_free(m,i):
        return ( m.b[i] - pe.log(m.K[i]) ) >= m.h[i] - Mb*(m.u_free[i])  
    m.rxr_cns_up_free = pe.Constraint(m.flux_free_idx,rule = relaxed_reg_cns_upper_free)


    def relaxed_reg_cns_lower_free(m,i):
        return ( m.b[i] - pe.log(m.K[i]) ) <= m.h[i] + Mb*(1 - m.u_free[i])
    m.rxr_cns_low_free = pe.Constraint(m.flux_free_idx,rule = relaxed_reg_cns_lower_free)


    ''' 
    Fixed Flux Version of Sign Constraints
    this is when direction is fixed
    '''


    def y_sign_relax_fixed(m,i):
        return 2*m.u_fixed[i] - 1 == (m.y[i]/(abs(m.y[i]) + 1e-50) ) 
    m.y_sign_relax_cns_fixed = pe.Constraint(m.flux_fixed_idx,rule = y_sign_relax_fixed)

    
    def relaxed_reg_cns_upper_fixed(m,i):
        return ( m.b[i] - pe.log(m.K[i]) ) >= m.h[i] - Mb*(m.u_fixed[i])  
    m.rxr_cns_up_fixed = pe.Constraint(m.flux_fixed_idx,rule = relaxed_reg_cns_upper_fixed)


    def relaxed_reg_cns_lower_fixed(m,i):
        return ( m.b[i] - pe.log(m.K[i]) ) <= m.h[i] + Mb*(1 - m.u_fixed[i])
    m.rxr_cns_low_fixed = pe.Constraint(m.flux_fixed_idx,rule = relaxed_reg_cns_lower_fixed)

    
    # Metabolite Net Flux Constraints

    if len(m_var_net_flux_restricted_idx)> 0 :

        net_flux_dict = dict(list(zip(m_var_net_flux_restricted_idx,m_net_flux_restrictions[m_var_net_flux_restricted_idx])))
        m.net_flux_sign = pe.Param(m.net_flux_idx, initialize = net_flux_dict)


        def net_flux_constraint(m,i):
            return m.net_flux_sign[i]*sum( m.S[(i,j)]*m.y[j] for j in m.react_idx ) >= 0
        m.net_flux_cns = pe.Constraint(m.net_flux_idx,rule = net_flux_constraint)


    # Variable metabolite upper and lower bounds
    def M_upper_cnstrnts(m,i):
        return  m.VarM[i] <= m.VarM_ubnd[i]
    m.VarM_ub_cns = pe.Constraint(m.VarM_idx,rule = M_upper_cnstrnts)


    def M_lower_cnstrnts(m,i):
        return  m.VarM[i] >= m.VarM_lbnd[i]
    m.VarM_lb_cns = pe.Constraint(m.VarM_idx,rule = M_lower_cnstrnts)

    
    

    # Original objective constraint
    og_obj = np.sum(y_ini*primary_obj_coefs)
    og_ep = primary_obj_tolerance
    obj_bound = np.array([og_obj - og_ep])

    
    #Original Objective Coefficients
    ##################
    obj_og_idx = list(itertools.product(np.arange(0,n_react),np.arange(0,1)))
    obj_og_dict = dict(list(zip(obj_og_idx,list(primary_obj_coefs) ) ))
    m.og_obj_coefs=pe.Param(obj_og_idx, initialize = obj_og_dict)

    rhs_og_idx = np.arange(0,1)
    rhs_og_dict =  dict(list(zip(rhs_og_idx,list(obj_bound) ) ))
    m.rhs_og_coefs = pe.Param(rhs_og_idx, initialize = rhs_og_dict)


    def og_obj_cnstrnt(m,i):
        return m.rhs_og_coefs[i] <= sum( m.og_obj_coefs[(j,i)]*m.y[j]  for j in m.react_idx )
    m.og_obj_cns = pe.Constraint(m.dm_idx, rule = og_obj_cnstrnt)


    # Set the Objective function

    def _Obj(m):
        return sum( m.y[j]*m.obj_coefs[j]  for j in m.react_idx )
    m.Obj_fn = pe.Objective(rule = _Obj, sense = pe.maximize) 


    
    

    #Find a Solution
    ####################

    #Set the solver to use
    opt=pe.SolverFactory('ipopt', solver_io='nl')


    #Set solver otpions
    if linear_solver == 'default' and hsllib == "":
        # neither option is set so default for both
        opts = {'max_iter': max_iter,
            'max_cpu_time': max_cpu_time,
            'tol':solverTol,
            'acceptable_tol':solverAcceptableTol}
        
    elif linear_solver == 'default':
        # then hsllib option is set
        opts = {'max_iter': max_iter,
            'max_cpu_time': max_cpu_time,
            'tol':solverTol,
            'acceptable_tol':solverAcceptableTol,
            'hsllib':hsllib}
        
    elif hsllib == "":
        # then linear_solver option is set
        opts = {'max_iter': max_iter,
            'max_cpu_time': max_cpu_time,
            'tol':solverTol,
            'acceptable_tol':solverAcceptableTol,
            'linear_solver': linear_solver} 
        
    else:
        # then both options are set
        opts = {'max_iter': max_iter,
            'max_cpu_time': max_cpu_time,
            'tol':solverTol,
            'acceptable_tol':solverAcceptableTol,
            'linear_solver': linear_solver,
            'hsllib':hsllib} 

    ## Solve the Model
    status_obj = opt.solve(m, options=opts, tee=False)
    obj_val = pe.value(m.Obj_fn)

    # find out if it terminated on optimal solution
    solved = 1
    if status_obj.solver.termination_condition == TerminationCondition.optimal:
        solved = 1
    else:
        solved = 0

    n_sol=np.zeros(n_M_v)
    b_sol = np.zeros(n_react)
    y_sol = np.zeros(n_react)
    beta_sol = np.zeros(dSe_N)
    h_sol = np.zeros(n_react)

    for i in flux_tot_idx:
        b_sol[i] = pe.value(m.b[i])
        y_sol[i] = pe.value(m.y[i])
        h_sol[i] = pe.value(m.h[i])

    for i in beta_idx:
        beta_sol[i] = pe.value(m.beta[i])

    for i in m_var_idx:
        n_sol[i] = pe.value(m.VarM[i])
        
    
    
    E_regulation = np.ones(len(y_sol))
    unreg_rxn_flux = np.ravel( rxn_flux(n_sol, np.ravel(m_fixed_log_counts),S_T, Keq, E_regulation) )
    alpha_sol = y_sol/unreg_rxn_flux
    alpha_sol = np.ravel(alpha_sol)

    

    return(y_sol, alpha_sol, n_sol, status_obj, obj_val)


'''
Function use the calculated chemical concentrations to determine the unregulated reaction flux values.

Inputs:
    - v_log_counts: a vector for the log concentration counts fo rhte variable metabolites in the system
    - f_log_counts: a fixed for the log concentration counts fo rhte variable metabolites in the system
    - S: the stoichiometric matrix for the reactions and metabolites in the model 
    - K: the vector of reaction equilibrium constants
    - E_regulation: a vector for the amounts of enzymatic regulation for each reaction

Returns:
    - the flux through a regulated reaction
'''

def rxn_flux(v_log_counts, f_log_counts, S, K, E_regulation):
    # Flip Stoichiometric Matrix
    S_T = S    # S_T is the Stoich matrix with rows as reactions, columns as metabolites
    S = np.transpose(S) # this now is the Stoich matrix with rows metabolites, and columns reactions
    n_react = np.shape(S)[1]
    v_log_counts = np.reshape(v_log_counts,(len(v_log_counts),1))
    f_log_counts = np.reshape(f_log_counts,(len(f_log_counts),1))
    tot_log_counts = np.concatenate((v_log_counts,f_log_counts))
    K = np.reshape(K,(len(K),1))
    E_regulation = np.reshape(E_regulation,(len(E_regulation),1))
    
    forward_odds = K*np.exp(-.25*np.matmul(S_T,tot_log_counts) )**4
    reverse_odds = np.power(K,-1)*np.exp(.25*np.matmul(S_T,tot_log_counts) )**4

    return E_regulation*(forward_odds - reverse_odds )
    
