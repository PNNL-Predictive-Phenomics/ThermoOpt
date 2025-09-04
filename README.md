# ThermoOpt

ThermoOpt: Thermodynamically governed, multi-scale population growth in dynamic environments

Description:

**ThermoOpt** is an optimization-based software tool designed to evaluate the emergent dynamics of cell populations by modeling the regulated metabolic processes of individual cells. Using maximum entropy production rate optimization, ThermoOpt estimates kinetic parameters and models cellular regulation while minimizing reliance on experimental data. This approach overcomes challenges in traditional kinetic modeling, which often requires intensive parameterization and fails to couple reaction rates with regulatory processes. ThermoOpt provides interpretable insights into how cells regulate their metabolism in dynamic environments, predicting optimal regulation strategies for high cell growth. The software also simulates mechanistic effects offering a powerful tool for understanding population-level behavior in complex systems.

Key Features:

- **Reduced data requirements**: ThermoOpt streamlines metabolic kinetic modeling by minimizing the need for extensive experimental data, leveraging thermodynamic principles to estimate metabolite concentrations and reaction kinetics.
- **Prediction of reaction regulation**: The framework infers regulatory mechanisms and predicts cell phenotypic behavior, enabling insights into optimal cellular responses to dynamic environmental changes.
- **Multiscale modeling**: ThermoOpt bridges large-scale population dynamics and small-scale reaction regulation, supporting simulations of dynamic monoclonal cell populations across multiple length scales.
- **User-friendly modular design**: The software incorporates a modular and extendable framework, enabling users to simulate metabolic processes and extend the tool for bioengineering applications.

Use Cases:

ThermoOpt uses a dynamical systems approach coupled with ordinary differential equations (ODEs) to model environmental perturbations, metabolic behavior, and thermodynamic bottlenecks. It applies the maximum entropy production rate method to optimize reaction parameters for maximal biomass production, providing insights into processes such as byproduct inhibition and nutrient uptake. By addressing challenges like ethanol buildup during yeast fermentation, ThermoOpt mechanistically predicts cellular regulation under shifting thermodynamic gradients, offering a powerful tool for understanding metabolic dynamics without extensive experimental efforts.

Requirements:
- Coin-hsl required for interior point optimization method https://licences.stfc.ac.uk/product/coin-hsl

How To:

1) Base models are provided in the "Saved_models" directory. Models are stored in pkl format as pandas dataframes and are provided with reaction directions and reaction free energies.
2) Simulations are run using the "DynamicalThermoOpt.ipynb" notebook. Simulation parameters are provided in the simulation dictionaries.
3) A guide to parameters and options can be found in "opt_metabolism_tool.ipynb"
4) Model parameters and scales can be checked in the "opt_instance_check.ipynb" notebook.
5) Examples of how to perform large scale parameter scans are provided in "Example scan pipeline" directory.




## Citation

To be updated.

## License

ThermoOpt is made freely available under the terms of Simplified BSD license.
Copyright 2025 Battelle Memorial Institute
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

<div align="center">

PACIFIC NORTHWEST NATIONAL LABORATORY  
operated by  
BATTELLE  
for the  
UNITED STATES DEPARTMENT OF ENERGY  
under Contract DE-AC05-76RL01830  

</div>


## Acknowledgments

This work was supported by the Predictive Phenomics Initiative, under the Laboratory Directed Research and Development Program at at the Pacific Northwest National Laboratory. PNNL is a multiprogram national laboratory operated by Battelle for the U.S. Department of Energy under contract DE-AC05-76RLO 1830.
