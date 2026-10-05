# age_BIA
Numerical model for age and trajectories of ice particles in a flow tube of an ice sheet. 
This model deals with ablation areas, like found in Blue Ice Areas or scouring zones.

# What this manual is and is not?

This manual is a documentation on how to use the age_BIA software.  
It is _not_ a description of the age_BIA principles and assumptions. Please read to the scientific articles
describing age_BIA for that purpose:\
Parrenin, F., Jeaumot, Y., Legrain, É. and Capron, E.: age_BIA-1.0: a numerical age model for a pseudo-steady flow tube of an ice sheet with ablation areas, in preparation.

It is _not_ an operating system or python documentation.
Please use your operating system or python documentation instead.

# Where can I get help on age_BIA?

You can also directly email to Frédéric Parrenin: frederic.parrenin@univ-grenoble-alpes.fr

# How to download age_BIA?

Go here:  
https://github.com/parrenin/age_BIA/  
and click on the donwload button.  
In the downloaded folder, you will find the following files:
- README.md		: is the current documentation of age_flow_line.
- LICENCE		: is the age_flow_line licence file.
- age_BIA.py		: is the main program that you will run.
- Clean.py		: is a python script to clean a model experiment directory
- DC-BELDC-Jeaumot2026		: is an example experiment directory: it contains all the necessary
numerical settings and input files for the flow line between Dome C and Little Dome C.

# What do I need to run age_BIA?

age_BIA is a scientific python3- software, therefore you need a scipy distribution.  
age_BIA is developed and tested using the miniforge distribution, therefore we recommend it.  
Miniforge can be downloaded here:  
https://conda-forge.org/miniforge/

Age_BIA probably works on other scipy distributions, provided they contain the following python
modules:  
- sys
- os
- time
- math
- numpy
- matplotlib
- scipy
- resource
- yaml

# How to run age_BIA?

Assuming you use miniforge, you can install the spyder IDE and type the following commands in the
ipython interpreter:

```
cd path-to-age_BIA
run age_BIA.py exp_directory/
```

where `path-to-age_BIA` is the directory containing age_flow_line and `exp_directory` is the name of
your experiment directory. 
The `DC-LDC-Jeaumot2026` experiment directory is provided for you convenience.
It takes a few seconds to run on a recent computer.

# What is the structure of an experiment direcotry?

You can have a look at the provided `DC-LDC-Jeaumot2026` directory.

You have one parameter file `parameters.yml` which contains general parameters for the
experiment.

Then you have several input file in `.txt` format:
- `accumulation.txt`: the surface accumulation along the flow line
- `melting.txt`: the basal melting along the flow line
- `thickness.txt`: the thickness along the flow line
- `surface.txt`: the surface elevation along the flow line
- `tube_width.txt`: the flow tube width along the flow line
- `sliding.txt`: the basal sliding along the flow line
- `p_Lliboutry.txt`: the p exponent of the Lliboutry velocity profile along the flow line
- `relative_density.txt`: a depth vs relative density profile
- `temporal_factor.txt`: the accumulation/melting relative temporal variations

If you want to set up a new flow tube experiment, we suggest to copy an existing experiment directory such as DC-BELDC-Jeaumot2026.
Then you can incrementally modify the `parameters.yml` parameter file and the `.txt` data files.

# What are the outputs of a run?

If the run went correctly, it has created output files for the whole flow line:
- `flow_line_output.txt`: is the output for the whole flow line
- `age_x_z.pdf`: is the same using the (x,z) coordinate system
- `age_x_depth.pdf`: is the same using the (x,depth) coordinate system
- `temporal_factor.pdf`: is the figure with the accu/melting temporal factor
- `trajectories_x_z.pdf`: are the stream lines / trajectories in (x,z)
- `trajectories_x_depth.pdf`: same using (x,depth)
- `trajectories_color_x_z.pdf`: are the stream lines / trajectories in (x,z) with age in color scale
- `trajectories_color_x_depth.pdf`: same using (x,depth)

If you have defined virtual ice cores, it has also created some output files:
- `IC_ice_core_output.txt`: is the output for the _IC_ ice core
- `IC_ice_core.pdf`: figure with a few quantities as a function of the depth in the _IC_ ice core

# What to do if something goes wrong?

Some errors can be eliminated by restarting the kernel in spyder (under "Console">"Restart kernel").\
If the problem persist, please post an email to the author with the error message appearing on the command line.