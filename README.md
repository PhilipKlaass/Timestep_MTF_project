# Timestep_MTF_project

## Flow chart for calculating the MTF for a given image

### Choose Region of interest
    
Use the open_images() function to read an image and select an ROI.

    
### Apply Flat Field Correction

Read a dark and light frame with open_images(), using the same ROI.

Apply the Flat Field Correction with flatfield_correction() to the intended ROI using the light and dark frame.

### Detect edge 

1. First create a binary array of edge points from the corrected ROI using edge\_points(). 
    1. Note that edge\_points() requires a threshold argument, that           determines if a given entry is an edgepoint or not.

2. Use the hugh\_transform() function on the binary edge array to characterize edges. The hugh Transform outputs a radius an angle
$$r= x\cos(\theta)+y\sin(\theta)$$
The function also has a threshold which grades the line on how many edgepoints they intersect, this should be set on the order of your ROI size. Lastly the function prints a list of possible lines as a list of 3-tuples, where the entries or theta in radians, r, and the x-intercept.

Alternatively use the ISO method to find centroids of the each lsf to determine the slope of the edge.

### Measure Edge Spread/Responce Function(ERF/ESF)

After finding the line correponding to the edge, measure the ERF/ESF with the get\_esf() function. 

### Smooth ERF/ESF (as appropiate) 

Reduce noise in the ERF/ESF by applying an averaging filter, median filter, and lastly the Savitsky-Golay filter. Use the average\_filter() and median\_filter() functions for their respective smoothing. There is a Sav-Gol filter built into scipy, the recomended settings are a 2nd degree polynomial with a window size of 51.

### Calculate the LSF

Compute the derivative of the smoothed ERF/ESF. Any method will work, but the get\_derivative() function doesnot require any outside packages.

### Calculate the MTF from the LSF

Use the FFT() function to calculate the MTF from the LSF.



## intro()

This basic flow chart is coded in the intro() function.

## Simulated MTFs

There 
