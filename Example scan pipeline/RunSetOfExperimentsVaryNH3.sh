#!/bin/bash
# echo " "
# echo " NH3 Continuum ICs "
# echo " "

# echo " NH3 Continuum ICs " >> RunSetOfExperimentsOutput.txt

> "simulationOutputFile_NH3_Orthophosphate_highCarbon_chemostat.txt"
chmod +x "simulationOutputFile_NH3_Orthophosphate_highCarbon_chemostat.txt"
python RunSimulation_NH3_orthophosphate_highCarbon_chemostat.py
echo "RunSimulation_NH3_orthophosphate_highCarbon_chemostat.py" >> RunSetOfExperimentsOutput.txt

> "simulationOutputFile_NH3_Orthophosphate_highCarbon.txt"
chmod +x "simulationOutputFile_NH3_Orthophosphate_highCarbon.txt"
python RunSimulation_NH3_orthophosphate_highCarbon.py
echo "RunSimulation_NH3_orthophosphate_highCarbon.py" >> RunSetOfExperimentsOutput.txt

# echo " "
# echo " NH3 Continuum BCs "
# echo " "

# echo " NH3 Continuum BCs " >> RunSetOfExperimentsOutput.txt

# > "simulationOutputFile_NH3ContinuumBCs.txt"
# chmod +x "simulationOutputFile_NH3ContinuumBCs.txt"
# python RunSimulation_NH3Continuum_BCs.py
# echo "RunSimulation_NH3Continuum_BCs.py" >> RunSetOfExperimentsOutput.txt

# echo " "
# echo " NH3 Continuum BCs - Xylose "
# echo " "

# echo " NH3 Continuum BCs - Xylose  " >> RunSetOfExperimentsOutput.txt

# > "simulationOutputFile_Xylose_NH3Continuum_BCs.txt"
# chmod +x "simulationOutputFile_Xylose_NH3Continuum_BCs.txt"
# python RunSimulation_Xylose.py
# echo "RunSimulation_Xylose.py" >> RunSetOfExperimentsOutput.txt

# for hyperparameter study

# > "simulationOutputFile_NH3ContinuumICsHyperparametersHighDeltaC_swarmSize_2.txt"
# chmod +x "simulationOutputFile_NH3ContinuumICsHyperparametersHighDeltaC_swarmSize_2.txt"
# python RunSimulation_NH3Continuum_ICsHighDeltaC.py
# echo "RunSimulation_NH3Continuum_ICsHighDeltaC.py" >> RunSetOfExperimentsOutput.txt

# > "simulationOutputFile_NH3ContinuumICsHyperparametersMidDeltaC_swarmSize_2.txt"
# chmod +x "simulationOutputFile_NH3ContinuumICsHyperparametersMidDeltaC_swarmSize_2.txt"
# python RunSimulation_NH3Continuum_ICsMidDeltaC.py
# echo "RunSimulation_NH3Continuum_ICsMidDeltaC.py" >> RunSetOfExperimentsOutput.txt

# > "simulationOutputFile_NH3ContinuumICsHyperparametersLowDeltaC_swarmSize_2.txt"
# chmod +x "simulationOutputFile_NH3ContinuumICsHyperparametersLowDeltaC_swarmSize_2.txt"
# python RunSimulation_NH3Continuum_ICsLowDeltaC.py
# echo "RunSimulation_NH3Continuum_ICsLowDeltaC.py" >> RunSetOfExperimentsOutput.txt

# > "simulationOutputFile_NH3Continuum_tailor2.txt"
# chmod +x "simulationOutputFile_NH3Continuum_tailor2.txt"
# python RunSimulation_NH3Continuum_testing.py
# echo "RunSimulation_NH3Continuum_testing.py" >> RunSetOfExperimentsOutput.txt

# > "simulationOutputFile_NH3Continuum_low_redox_test.txt"
# chmod +x "simulationOutputFile_NH3Continuum_low_redox_test.txt"
# python RunSimulation_NH3Continuum_redox_test.py
# echo "RunSimulation_NH3Continuum_redox_test.py" >> RunSetOfExperimentsOutput.txt