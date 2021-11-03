# EE6222
Fundamentals of image processing & analysis. Feature Extraction Techniques. Pattern / Object Recognition and Interpretation. Three Dimensional Computer Vision. Three-Dimensional Recognition Techniques. Biometrics.
## Assignment 1 (Option 1)
Using this RVFL as the base model, you are asked to investigate various issues as listed in your lecture slides. You can refer to your lecture slides for further details on RVFL.
1.	Effect of direct links from the input layer to the output layer (i.e. with and without).
2.	Performance comparisons of 2 activation functions: one from “relu, sigmoid, radbas, sine” and one from “hardlim, tribas”.
3.	Performance of Moore-Penrose pseudoinverse and ridge regression (or regularized least square solutions) for the computation of the output weights.
You can use around 8-10 datasets. If your computer cannot handle large datasets with more than 20000 samples, you can exclude them. You should also very exclude small datasets. You can also have some 2 class and more than 2 class problems.
Optional Extension: If you’ve time, you can consider investigating the deep RVFL versions presented in: 
Random Vector Functional Link Neural Network based Ensemble Deep Learning, Pattern Recognition, 107978, 2020 (Codes available from GitHub) 
## Assignment 2
1. Find the focal length f of your hand phone (in pixels)
You may use real person or printed figure, and include one figure of the settings in your report. Make sure you turn the camera’s “zooming/auto-focusing” off.
2. Take two snaps of an outdoor scene, with 5 to 10 degrees angle difference. You need to keep the angle as ground truth.
3. Hand pick 8 points or more from one image, and find the matching points on the other image. These points should not be co-planar. You need to turn these points into N-vector, and submit them into the equation for calculation.
4. Calculate the rotation angle from the matched points using the quaternion approach (pp 14 in [4]), or the SVD(in [3]).
