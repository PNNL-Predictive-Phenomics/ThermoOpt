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

warnings.filterwarnings('ignore')
from importlib import reload

Module_path = '../Modules'  #specify path to modules folder, change as needed for environment
sys.path.insert(0,Module_path)

import base_functions as baf
# import flux_opt_fung as fopt
import Two_stage_flux_opt as fopt
#import flux_opt_square_constraints as fopt
import fixed_sign_flux_opt as fsf

#import flux_opt_square_constraints as fopt
reload(fopt)
reload(baf)
reload(fsf)

T = 298.15
R = 8.314e-03
RT = R*T
N_avogadro = 6.022140857e+23
VolCell = 1.0e-15
Concentration2Count = N_avogadro * VolCell
concentration_increment = 1/(N_avogadro*VolCell)


simulationList=[]
simulation_1={
    'name':'Fungal_lowNADP_midNH3_highCarbon_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH.pkl',
    'model_output':'output_Fungal_lowNADP_midNH3_highCarbon_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'ALPHA-D-GLUCOSE:ENVIRONMENT': 0.0002, 'D-FRUCTOSE:ENVIRONMENT': 0.0002,'NH3:CYTOSOL':1e-5},#{'NH3:CYTOSOL':1e-5}, # change fixed metabolite concentration
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_1)

simulation_2={
    'name':'Fungal_highNADP_midNH3_highCarbon_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-highNADP2NADPH.pkl',
    'model_output':'output_Fungal_highNADP_midNH3_highCarbon_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'NH3:CYTOSOL':1e-5},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_2)

simulation_3={
    'name':'Add_Arabitol_midNH3_highCarbon_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH-Arabitol.pkl',
    'model_output':'output_Add_Arabitol_midNH3_highCarbon_gluc_fruc',
    'initial_conditions':{'D-ARABITOL:ENVIRONMENT':1e-5},
    'boundary_conditions':{'NH3:CYTOSOL':1e-5},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_3)


# add secondary objective

simulation_4={
    'name':'Fungal_lowNADP_midNH3_highCarbon_secondary_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH.pkl',
    'model_output':'output_Fungal_lowNADP_midNH3_highCarbon_secondary_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'NH3:CYTOSOL':1e-5},#{'NH3:CYTOSOL':1e-5}, # change fixed metabolite concentration
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'secondary_objective':{'PALMITOYL-COA-HYDROLASE-RXN':2e2}, # target rxn and tolerance
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_4)

simulation_5={
    'name':'Fungal_highNADP_midNH3_highCarbon_secondary_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-highNADP2NADPH.pkl',
    'model_output':'output_Fungal_highNADP_midNH3_highCarbon_secondary_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'NH3:CYTOSOL':1e-5},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'secondary_objective':{'PALMITOYL-COA-HYDROLASE-RXN':2e2}, # target rxn and tolerance
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_5)

simulation_6={
    'name':'Add_Arabitol_midNH3_highCarbon_secondary_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH-Arabitol.pkl',
    'model_output':'output_Add_Arabitol_midNH3_highCarbon_secondary_gluc_fruc',
    'initial_conditions':{'D-ARABITOL:ENVIRONMENT':1e-5},
    'boundary_conditions':{'NH3:CYTOSOL':1e-5},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'secondary_objective':{'PALMITOYL-COA-HYDROLASE-RXN':2e2}, # target rxn and tolerance
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_6)

simulation_7={
    'name':'Fungal_lowNADP_lowNH3_highCarbon_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH.pkl',
    'model_output':'output_Fungal_lowNADP_lowNH3_highCarbon_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'ALPHA-D-GLUCOSE:ENVIRONMENT': 0.0002, 'D-FRUCTOSE:ENVIRONMENT': 0.0002,'NH3:CYTOSOL':1e-6},#{'NH3:CYTOSOL':1e-5}, # change fixed metabolite concentration
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_7)

simulation_8={
    'name':'Fungal_highNADP_lowNH3_highCarbon_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-highNADP2NADPH.pkl',
    'model_output':'output_Fungal_highNADP_lowNH3_highCarbon_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'NH3:CYTOSOL':1e-6},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_8)

simulation_9={
    'name':'Add_Arabitol_lowNH3_highCarbon_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH-Arabitol.pkl',
    'model_output':'output_Add_Arabitol_lowNH3_highCarbon_gluc_fruc',
    'initial_conditions':{'D-ARABITOL:ENVIRONMENT':1e-5},
    'boundary_conditions':{'NH3:CYTOSOL':1e-6},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_9)


# add secondary objective

simulation_10={
    'name':'Fungal_lowNADP_lowNH3_highCarbon_secondary_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH.pkl',
    'model_output':'output_Fungal_lowNADP_lowNH3_highCarbon_secondary_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'NH3:CYTOSOL':1e-6},#{'NH3:CYTOSOL':1e-5}, # change fixed metabolite concentration
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'secondary_objective':{'PALMITOYL-COA-HYDROLASE-RXN':2e2}, # target rxn and tolerance
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_10)

simulation_11={
    'name':'Fungal_highNADP_lowNH3_highCarbon_secondary_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-highNADP2NADPH.pkl',
    'model_output':'output_Fungal_highNADP_lowNH3_highCarbon_secondary_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'NH3:CYTOSOL':1e-6},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'secondary_objective':{'PALMITOYL-COA-HYDROLASE-RXN':2e2}, # target rxn and tolerance
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_11)

simulation_12={
    'name':'Add_Arabitol_lowNH3_highCarbon_secondary_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH-Arabitol.pkl',
    'model_output':'output_Add_Arabitol_lowNH3_highCarbon_secondary_gluc_fruc',
    'initial_conditions':{'D-ARABITOL:ENVIRONMENT':1e-5},
    'boundary_conditions':{'NH3:CYTOSOL':1e-6},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'secondary_objective':{'PALMITOYL-COA-HYDROLASE-RXN':2e2}, # target rxn and tolerance
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_12)

simulation_13={
    'name':'Fungal_lowNADP_highNH3_highCarbon_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH.pkl',
    'model_output':'output_Fungal_lowNADP_highNH3_highCarbon_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'ALPHA-D-GLUCOSE:ENVIRONMENT': 0.0002, 'D-FRUCTOSE:ENVIRONMENT': 0.0002,'NH3:CYTOSOL':1e-4},#{'NH3:CYTOSOL':1e-5}, # change fixed metabolite concentration
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_13)

simulation_14={
    'name':'Fungal_highNADP_highNH3_highCarbon_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-highNADP2NADPH.pkl',
    'model_output':'output_Fungal_highNADP_highNH3_highCarbon_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'NH3:CYTOSOL':1e-4},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_14)

simulation_15={
    'name':'Add_Arabitol_highNH3_highCarbon_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH-Arabitol.pkl',
    'model_output':'output_Add_Arabitol_highNH3_highCarbon_gluc_fruc',
    'initial_conditions':{'D-ARABITOL:ENVIRONMENT':1e-5},
    'boundary_conditions':{'NH3:CYTOSOL':1e-4},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_15)


# add secondary objective

simulation_16={
    'name':'Fungal_lowNADP_highNH3_highCarbon_secondary_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH.pkl',
    'model_output':'output_Fungal_lowNADP_highNH3_highCarbon_secondary_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'NH3:CYTOSOL':1e-4},#{'NH3:CYTOSOL':1e-5}, # change fixed metabolite concentration
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'secondary_objective':{'PALMITOYL-COA-HYDROLASE-RXN':2e2}, # target rxn and tolerance
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_16)

simulation_17={
    'name':'Fungal_highNADP_highNH3_highCarbon_secondary_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-highNADP2NADPH.pkl',
    'model_output':'output_Fungal_highNADP_highNH3_highCarbon_secondary_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'NH3:CYTOSOL':1e-4},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'secondary_objective':{'PALMITOYL-COA-HYDROLASE-RXN':2e2}, # target rxn and tolerance
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_17)

simulation_18={
    'name':'Add_Arabitol_highNH3_highCarbon_secondary_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH-Arabitol.pkl',
    'model_output':'output_Add_Arabitol_highNH3_highCarbon_secondary_gluc_fruc',
    'initial_conditions':{'D-ARABITOL:ENVIRONMENT':1e-5},
    'boundary_conditions':{'NH3:CYTOSOL':1e-4},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'secondary_objective':{'PALMITOYL-COA-HYDROLASE-RXN':2e2}, # target rxn and tolerance
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_18)






# varying arbitol

simulation_19={
    'name':'Add_Arabitol_lowArabitol_highCarbon_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH-Arabitol.pkl',
    'model_output':'output_Add_Arabitol_lowArabitol_highCarbon_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'D-ARABITOL:ENVIRONMENT':1e-6},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_19)

simulation_20={
    'name':'Add_Arabitol_lowArabitol_highCarbon_secondary_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH-Arabitol.pkl',
    'model_output':'output_Add_Arabitol_lowArabitol_highCarbon_secondary_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'D-ARABITOL:ENVIRONMENT':1e-6},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'secondary_objective':{'PALMITOYL-COA-HYDROLASE-RXN':2e2}, # target rxn and tolerance
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_20)

simulation_21={
    'name':'Add_Arabitol_midArabitol_highCarbon_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH-Arabitol.pkl',
    'model_output':'output_Add_Arabitol_midArabitol_highCarbon_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'D-ARABITOL:ENVIRONMENT':1e-5},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_21)

simulation_22={
    'name':'Add_Arabitol_midArabitol_highCarbon_secondary_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH-Arabitol.pkl',
    'model_output':'output_Add_Arabitol_midArabitol_highCarbon_secondary_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'D-ARABITOL:ENVIRONMENT':1e-5},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'secondary_objective':{'PALMITOYL-COA-HYDROLASE-RXN':2e2}, # target rxn and tolerance
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_22)

simulation_23={
    'name':'Add_Arabitol_highArabitol_highCarbon_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH-Arabitol.pkl',
    'model_output':'output_Add_Arabitol_highArabitol_highCarbon_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'D-ARABITOL:ENVIRONMENT':1e-4},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_23)

simulation_24={
    'name':'Add_Arabitol_highArabitol_highCarbon_secondary_gluc_fruc',
    'model':'Fungal-metabolic-model-250rxns-sucrose-lowNADP2NADPH-Arabitol.pkl',
    'model_output':'output_Add_Arabitol_highArabitol_highCarbon_secondary_gluc_fruc',
    'initial_conditions':{},
    'boundary_conditions':{'D-ARABITOL:ENVIRONMENT':1e-4},
    'objective_reactions':['protein_syn', 'rna_syn','dna_syn'],
    'secondary_objective':{'PALMITOYL-COA-HYDROLASE-RXN':2e2}, # target rxn and tolerance
    'remove_metabolites':['SUCROSE:ENVIRONMENT'],
    'remove_reactions':['RXN-1461']
}
simulationList.append(simulation_24)

print()
print("Solving "+str(len(simulationList))+" parent simulations.")

# change gluc:fruc ratios
gluc_name = 'ALPHA-D-GLUCOSE:ENVIRONMENT'
fruc_name = 'D-FRUCTOSE:ENVIRONMENT'

sugar_total = 0.01

# percentage_gluc = [1,10,20,30,40,50,60,70,80,90,99]
percentage_gluc = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]

# set as boundary conditions
#'boundary_conditions':{}

gluc_fruc_simulationList=[]
for sim in simulationList:
    for percent in percentage_gluc:
        gluc_amount = sugar_total*(percent/100)
        fruc_amount = sugar_total - gluc_amount
        condition_bcs = {gluc_name:gluc_amount,fruc_name:fruc_amount}
        
        newSim=sim.copy()
        newSim['boundary_conditions'] = newSim['boundary_conditions'] | condition_bcs
        newSim['name'] = newSim['name']+"_"+str(percent)+"_"
        newSim['model_output'] = newSim['model_output']+"_"+str(percent)+"_"
        gluc_fruc_simulationList.append(newSim)

print("Solving "+str(len(gluc_fruc_simulationList))+" gluc_fruc_simulationList simulations.")

model_file_dir = 'Saved_models'
deltaConcentration=1e-17
numberOfSwarmInstances=5
bestSimulations = baf.RunSwarmSimulation(gluc_fruc_simulationList,deltaConcentration,model_file_dir,numberOfSwarmInstances)
sys.stdout.flush()
print()
print("Best simulations:\n")
print(bestSimulations)


f = open("simulationOutputFile_NH3_highCarbon.txt", "w")
f.write(str(bestSimulations))
f.close()