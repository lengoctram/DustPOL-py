# DustPOL - numerical modelling - v1.5
-- This numerical modelling calculates the multi-wavelength polarization degree of absorption and thermal dust emission 
based on Radiative Torque alignment (RAT-A), Magnetically enhanced RAT (MRAT) and Radiative Torque Disruption (RAT-D).

-- The routine will save the output files (wavelength and degree of polarization) for further analysis

# Authors
Le Ngoc Tram, Hyeseung Lee, and Thiem Hoang

# Features
Current version is idealization for PDR regions and diffuse ISM
Current version is designed to predict the polarization spectrum for one- or two-dust layers

# Upnext
Up next version will be developed for starless and prostostellar cores
The manual and web-interface will be soon released

# History:
2024   : Tram implemented a two-phase model: cold and warm dust layers along the LOS

2023   : Tram optimized and improved the code to work with maximum grain size lower than the disruption size

2022   : Thiem implemented MRAT in align.py to account for iron inclusions

2020   : Tram improved Hyeseung's code

2019   : Hyeseung modified the Dustpol Code from Thiem, adding RATD (maximum grain size is higher than the disruption size)

# Dependencies

1- Python 3

2- Numpy

3- Matplotlib

4- Scipy

5- Astropy

6- Joblib for parallelization (installation: https://joblib.readthedocs.io/en/latest/installing.html)

# How to use

1- From Terminal, type python DustPOL.py -f <name_of_inputs_file>

                      e.g., python DustPOL.py -f input.dustpol
      to have a first look: python DustPOL.py -f input.dustpol --first_look yes
                      
2- From interactive ipython, type run DustPOL.py -f <name_of_inputs_file>

                      e.g., run DustPOL.py - f input.dustpol
      to have a first look: run DustPOL.py -f input.dustpol --first_look yes

3- Run for multiple radiation field: in the input.dustpol, set "U" as

                      e.g., U 1,10,100,1000

4- Parallel computing: in the input.dustpol, set "parallel" as True

# Bugs
Please reach out to us at nle@mpifr-bonn.mpg.de 

# More information, please read

1- Lee et al. (2020) https://ui.adsabs.harvard.edu/abs/2020ApJ...896...44L

2- Tram et al. (2021) https://ui.adsabs.harvard.edu/abs/2021ApJ...906..115T

3- Tram et al. (2024) https://arxiv.org/abs/2403.17088 (accepted to A&A)

# Citations
If you use this code for your scientific projects, please cite

\bibitem[{{Lee} {et~al.}(2020){Lee}, {Hoang}, {Le}, \& {Cho}}]{2020ApJ...896...44L}
{Lee}, H., {Hoang}, T., {Le}, N., \& {Cho}, J. 2020, \apj, 896, 44,
  \dodoi{10.3847/1538-4357/ab8e33}

\bibitem[{{Tram} {et~al.}(2021{\natexlab{a}}){Tram}, {Hoang}, {Lee}, {Santos}, {Soam}, {Lesaffre}, {Gusdorf}, \& {Reach}}]{2021ApJ...906..115T}
{Tram}, L.~N., {Hoang}, T., {Lee}, H., {et al.} 2021{\natexlab{a}}, \apj, 906,
  115, \dodoi{10.3847/1538-4357/abc6fe}
