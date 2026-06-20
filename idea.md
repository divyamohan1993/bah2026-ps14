PROBLEM STATEMENT 14
Forecasting Energetic Particle Radiation Environment for ISRO's Geostationary Satellites
Description
To develop and demonstrate an algorithm to predict energetic particle fluxes of electrons at geostationary orbit. The algorithm should be able to predict the harsh radiation fluxes at least 30 to 45 minutes in advance, and also give a reasonable forecast for 6 hours and 12 hours ahead.

Objective
Develop an algorithm, preferably using Python or any other high-level language, for reading, processing, and visualizing the electron flux and solar wind data, which is archived in CDF format
Identify an AI/ML algorithm for time-series forecasting
Develop, fine-tune, and optimize the algorithm for training, validation, and testing.
Demonstration and visualisation of the algorithm outputs and their accuracy.
Expected Outcomes
Algorithm for reading, processing, visualization, and forecasting of energetic electron fluxes at geosynchronous orbit.

Dataset Required
GOES series >2 MeV electron fluxes in cdf format.
Data for 11 years from the GOES satellite should be provided.
Data for 11 years from the Wind spacecraft, mainly solar wind parameters: speed, interplanetary magnetic field, and density, should be provided.
For comparison of the forecasted flux at Indian longitude, electron fluxes from ISRO's GRASP/GSAT payload should be provided for 1-2 years.
Suggested Tools/Technologies
Python or any other high-level language for reading, processing, and visualization of data.
AI/ML tools/sub-routines (Teams are encouraged to develop their own routines).
A basic understanding of Earth's radiation belts, solar wind, and related data products.
Expected Solution / Steps to be followed to achieve the objectives
Reading and visualization of electron flux and solar wind data from archived CDF files.
Based on science domain knowledge, identify important solar wind variables for forecasting electron fluxes.
Preprocessing data for the identified AI/ML algorithm. Removing spike, interpolation, or omission of missing data, etc.
Identifying desired AI/ML to do multi-step forecast fluxes accounting for the time history of inputs and outputs.
Predicting the electron fluxes half an hour in advance also provides a forecast for 6 hours and 12 hours ahead.
Evaluation Parameters
An understanding of the basics of solar wind and the radiation belt.
Correctness of reading and visualization of the solar wind and radiation belt electron fluxes data
Identification and optimization of an AI/ML algorithm to get the best performance.
Accuracy of the predicted fluxes at geostationary orbit.
