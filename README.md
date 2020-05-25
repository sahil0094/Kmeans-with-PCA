# Kmeans-with-PCA
Kmeans with/without PCA. Examining the effect of dimensionality reduction on model

Data Set Information:

Predicting the age of abalone from physical measurements. The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age. Further information, such as weather patterns and location (hence food availability) may be required to solve the problem.

From the original data examples with missing values were removed (the majority having the predicted value missing), and the ranges of the continuous values have been scaled for use with an ANN (by dividing by 200).

# Attribute Information:

Given is the attribute name, attribute type, the measurement unit and a brief description. The number of rings is the value to predict: either as a continuous value or as a classification problem.

Name / Data Type / Measurement Unit / Description
-----------------------------
<ul>
<li>Sex / nominal / -- / M, F, and I (infant)
<li>Length / continuous / mm / Longest shell measurement
<li>Diameter / continuous / mm / perpendicular to length
<li>Height / continuous / mm / with meat in shell
<li>Whole weight / continuous / grams / whole abalone
<li>Shucked weight / continuous / grams / weight of meat
<li>Viscera weight / continuous / grams / gut weight (after bleeding)
<li>Shell weight / continuous / grams / after being dried
<li>Rings / integer / -- / +1.5 gives the age in years
</ul>

# Approach- 
We first applied Kmeans directly on dataset. We found 3 as the optimal number of clusters and got the inertia to be 9922.820.
Then we applied PCA as there was high multicollinearity and again applied KMeans using single PC and got the inertia to be 4786.761413.
