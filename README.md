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



How To:

1) Base models are provided in the "Saved_models" directory. Models are stored in pkl format as pandas dataframes and are provided with reaction directions and reaction free energies.
2) Simulations are run using the "DynamicalThermoOpt.ipynb" notebook. Simulation parameters are provided in the simulation dictionaries.
3) A guide to parameters and options can be found in "opt_metabolism_tool.ipynb"
4) Model parameters and scales can be checked in the "opt_instance_check.ipynb" notebook.
5) Examples of how to perform large scale parameter scans are provided in "Example scan pipeline" directory.




## Citation

To be updated.

## License

ThermoOpt is made freely available under the terms of Simplified BSD license. See LICENSE for details.

## Acknowledgments

This work was supported by the Predictive Phenomics Initiative, under the Laboratory Directed Research and Development Program at at the Pacific Northwest National Laboratory. PNNL is a multiprogram national laboratory operated by Battelle for the U.S. Department of Energy under contract DE-AC05-76RLO 1830.
