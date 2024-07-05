# Source code for "Artificial Agents Facilitate Human Cooperation through Indirect Reciprocity"
Published as a main track paper for ECAI 2024 (https://www.ecai2024.eu/)
Link to be added.

This folder contains all the source code to reproduce the results of the paper.

## Requirements

The code base is divided between Julia 1.10 and Mathematica 14.0.

The Julia packages can be added via the command

```
] add Plots CairoMakie Colors Measures LaTeXStrings Memoization ArnoldiMethod SpecialFunctions
```

The Julia code also makes use of the following standard libraries: Dates, SparseArrays

## Code Structure

Most of the code files do the mathematical operations described in the paper, consisting of calculating the cooperation index for a given set of parameters. The difference is in the parameters being studied. The structure of the code is as follows:

* **ir_dynamics_with_AAs.nb** - This Mathematica notebook is used to plot the simplexes with the reputation, stationary distribution and cooperation index at each state (which are then edited to form Fig 3, 4 of the main text);
* **cooperation_study.jl** - This julia file calculates the cooperation index for all norms as a function of the fraction of AAs, storing their values in text format in a folder. It also generates a simplified plot of the cooperation index (Fig 2, 5 of the main text);
* **b_study.jl** - This is used to study the cooperation index as a function of b, the cooperation benefit, for each norm. It produces the plot seen in Figure 1 of the Supplementary Material;
* **grad_disc_alld_study.jl** - This is used to study the gradient of selection along the AllD-Disc edge, for each norm. It produces the plots seen in Figure 5 and 7 of the Supplementary Material;
* **error_study.jl** - It contains the code to study the impact of different errors across social norms. It produces the results for the plots in Figures 10-13 of the Supplementary Material;
* **reputation_dynamics.jl** - Contains all the math to calculate reputation dynamics given a strategy state;
* **strategy_dynamics.jl** - Contains all the math to calculate strategy dynamics and the resulting cooperation/reputation distributions;
* **utils.jl** - Contains utility code used throughout the code base, pertaining to data processing;

Running the Julia code produces a text file in the "Plots/<foldername>" directory, which is then read to make the plots.

Additionally, code is provided to replicate the plotting style used in the article. These are:
coop_plots.jl, coop_plots_makie.jl, error_plots.jl. 
These read the "Plots/<foldername>" directory to generate the plot
