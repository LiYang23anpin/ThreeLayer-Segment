to be complete...

This code is used to generate a three-layer mask for relatively high-z galaxies. 
The three layers are aiming at mask foreground stars, contaminants within 2Rpet, and close neighbours from background.
following the spirit of hot+cold mode segmentation map. 
Detalis can be found in Yulin Zhao et al. 2020

The final masks can be feed to the Statmorph in order to estimate asymmetry, shape asymmetry, and outer asymmetry. 
Before running the code, you need to install Sextractor first.

