# DustPOL - numerical modelling - v1.5
This numerical modelling calculates the multi-wavelength polarization degree of absorption and thermal dust emission 
based on Radiative Torque alignment (RAT-A), Magnetically enhanced RAT (MRAT) and Radiative Torque Disruption (RAT-D).

# Authors
Le Ngoc Tram, Hyeseung Lee, and Thiem Hoang

# Features
Current version is idealization for PDR regions and diffuse ISM
Current version is designed to predict the polarization spectrum for one- or two-dust layers

Up next version will be developed for starless and prostostellar cores

# History:
2024   : Tram implemented a two-phase model: cold and warm dust layers along the LOS

2022.12: Thiem implemented MRAT in align.py to account for iron inclusions

2020   : Tram improved Hyeseung's code

2019   : Hyeseung modified the Dustpol Code from Thiem, adding RATD

# Dependencies

1- Python 3

2- Numpy

3- Matplotlib

4- Scipy

5- Astropy

# How to use

1- From Terminal, type python DustPOL.py -f <name_of_inputs_file>

                      e.g., python DustPOL.py -f input.dustpol
                      
2- From interactive ipython, type run DustPOL.py -f <name_of_inputs_file>

                      e.g., run DustPOL.py - f input.dustpol

# Bugs
Please reach out to us at nle@mpifr-bonn.mpg.de 

# More information, please read

1- Lee et al. (2020) https://ui.adsabs.harvard.edu/abs/2020ApJ...896...44L

2- Tram et al. (2021) https://ui.adsabs.harvard.edu/abs/2021ApJ...906..115T

# Citations
If you use this code for your scientific projects, please cite

\bibitem[{{Lee} {et~al.}(2020){Lee}, {Hoang}, {Le}, \& {Cho}}]{2020ApJ...896...44L}
{Lee}, H., {Hoang}, T., {Le}, N., \& {Cho}, J. 2020, \apj, 896, 44,
  \dodoi{10.3847/1538-4357/ab8e33}

\bibitem[{{Tram} {et~al.}(2021{\natexlab{a}}){Tram}, {Hoang}, {Lee}, {Santos}, {Soam}, {Lesaffre}, {Gusdorf}, \& {Reach}}]{2021ApJ...906..115T}
{Tram}, L.~N., {Hoang}, T., {Lee}, H., {et al.} 2021{\natexlab{a}}, \apj, 906,
  115, \dodoi{10.3847/1538-4357/abc6fe}
