---
title: "DustPOL-py: a numerical modeling for linear dust polarization"
---

## DustPOL-py - numerical modelling - v1.6
-- This numerical modelling calculates the multi-wavelength polarization degree of absorption and thermal dust emission 
based on Radiative Torque alignment (RAT-A), Magnetically enhanced RAT (MRAT) and Radiative Torque Disruption (RAT-D).

-- The routine will save the output files (wavelength and degree of polarization) for further analysis. A built-in routine for analysis is also provided.

-- For a quick look and investigation, please use a web-interface GUI: <https://dustpol-py.streamlit.app>

-- The high-performance-computation techniques are embedded.

## Authors
```Le Ngoc Tram```, Hyeseung Lee, and Thiem Hoang

## Contributors
Pham N. Diep, Nguyen B. Ngoc, Bao Truong, Ngan Lê

## Features
-- Current version is designed to predict the polarization spectrum for starlight and thermal

-- diffuse ISM 

-- molecular clouds and star-forming regions

-- isolated dense cores (starless cores)

## Upnext
-- Globules/Pillars

-- Protostars

-- Protoplanetary disks

## History:
2024   : Tram added the modulation for starless core and embedded high-performance-computation techniques

2024   : Tram re-structured the DustPOL-py infractructure to python class object (modulation)

2024   : Tram implemented a two-phase model: cold and warm dust layers along the LOS

2023   : Tram optimized and improved the code to work with maximum grain size lower than the disruption size

2022   : Thiem implemented MRAT in align.py to account for iron inclusions

2020   : Tram improved Hyeseung's code

2019   : Hyeseung modified the Dustpol Code from Thiem, adding RATD (maximum grain size is higher than the disruption size)

## Dependencies

1- Python 3

2- Numpy

3- Matplotlib

4- Scipy

5- Astropy

6- Joblib for parallelization (installation: https://joblib.readthedocs.io/en/latest/installing.html)

7- Concurrency for parallelization

## Bugs
Please reach out to us at <nle@strw.leidenuniv.nl> or <nle@mpifr-bonn.mpg.de>

## More information, please read

1- Lee et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJ...896...44L>

2- Tram et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...906..115T>

3- Tram et al. (2024) <https://www.aanda.org/articles/aa/pdf/2024/09/aa50127-24.pdf>

## Citations
If you use this code for your scientific projects, please cite

\bibitem[{{Lee} {et~al.}(2020){Lee}, {Hoang}, {Le}, \& {Cho}}]{2020ApJ...896...44L}
{Lee}, H., {Hoang}, T., {Le}, N., \& {Cho}, J. 2020, \apj, 896, 44,
  \dodoi{10.3847/1538-4357/ab8e33}

\bibitem[{{Tram} {et~al.}(2021{\natexlab{a}}){Tram}, {Hoang}, {Lee}, {Santos}, {Soam}, {Lesaffre}, {Gusdorf}, \& {Reach}}]{2021ApJ...906..115T}
{Tram}, L.~N., {Hoang}, T., {Lee}, H., {et al.} 2021{\natexlab{a}}, \apj, 906,
  115, \dodoi{10.3847/1538-4357/abc6fe}
  
\bibitem[{{Tram} {et~al.}(2024){Tram}, {Hoang}, {Wiesemeyer}, {Ristorcelli}, {Menten}, {Ngoc}, \& {Diep}}]{2024A&A...689A.290T}
{Tram}, L.~N., {Hoang}, T., {Wiesemeyer}, H., {et al.} 2024, \aa, 689, A290, \dodoi{10.1051/0004-6361/202450127}

