# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 21:32:41 2025

@author: klaas
"""

import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - kernel_size//2)**2 + (y - kernel_size//2)**2) / (2 * sigma**2)), (kernel_size, kernel_size))
    normal = kernel / np.sum(kernel)
    return normal

def calc_centroid(array,r):
    '''
    
    Parameters
    ----------
    array : np.array
        ROI.
    r : int
        Specified row.

    Returns
    -------
    centroid : float
        The centroid of the derivative, or the x position of the edge.

    '''
    
    m,n = array.shape
    
    deriv_sum = 0
    weighted_sum = 0
    
    for i in range(0,n-1):
        weighted_sum = weighted_sum+ i*(array[r,i+1]-array[r,i-1])
        deriv_sum = deriv_sum + (array[r,i+1]-array[r,i-1])
        
    return weighted_sum/deriv_sum -0.5



def calc_slope(array):
    
    m,n = array.shape
    
    centroid_sum = 0
    
    for i in range(0,m-1):
        centroid_sum+= 1/(calc_centroid(array, i)- calc_centroid(array, i+1))
        
    return 2*centroid_sum/m

#def shift()