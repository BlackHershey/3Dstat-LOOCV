# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import math
import numpy as np
from scipy.stats import norm  # for the pdf of the std. normal distribution, used in function weight
from scipy.stats import t  # for function pstat

# Definitions of main variables
n_subjects = 9
fwhm = 3.0 # mm
gauss_sd = fwhm/(2*math.sqrt(2*math.log(2)))  # ~1.274 mm
# NOTE: the maximum value of the nl. distribution with this fwhm is 
#    ~0.313, when x = the mean
peak_pdf = 0.31314575956655044 # dimensionless

# Data
# TODO: read in effect and location from CSV files, rather than using fake
#   numbers entered here to test
effect = np.asarray(list(range(n_subjects)))/10
loc_mean = np.asarray([14.0,-17.0,-3.0])
loc_sd   = np.asarray([1.,1.,1.])
location = np.stack([loc_mean[0]+loc_sd[0]*np.random.randn(n_subjects),
                     loc_mean[1]+loc_sd[1]*np.random.randn(n_subjects),
                     loc_mean[2]+loc_sd[2]*np.random.randn(n_subjects)]).T
#  location[0] returns x,y,z for subject 0

# TODO: validate input to all functions

def distance(x,location):
    """returns the Euclidean distance between a point x and each point in a 
    2-D numpy array, location, as a numpy array the same shape as location
    """
    return np.sqrt(np.sum(np.square(x-location), axis=1))

def weight(x,location):
    """returns an array of weights as in Eisenstein et al 2014 based on the 
    distance between the point x and each point in an array location of 
    contact coordinates
    """
    return norm.pdf(distance(x,location),loc=0,scale=gauss_sd)/peak_pdf

def N(x,location,threshold=0.05):
    """returns the scalar value Ni from Eisenstein et al 2014 based on a 
    point x and an array location of contact coordinates
    """
    return np.sum(weight(x,location)>=threshold)

# np.delete(location,j,axis=0)  # removes the j'th point from location

def ghat(i,location,effect):
    """returns g^_i as defined in Eisenstein et al 2014, i.e. the weighted
    mean effect for a point i based on weights w_{ij} based on the distance 
    from point i to each point j in the array location of contact coordinates,
    and effects g_j corresponding to each subject j
    Input: 
        i, a 1D numpy array representing a point
        location, a 2D numpy array representing an array of points
        effect, a 1D numpy array representing the effect of stimulation
            at the corresponding point in the array location
    Output:
        a scalar, g^_i, the weighted mean, i.e. the best estimate of 
        stimulation at point i based on the effect and location data at
        (other) points
    """
    return np.sum(np.multiply(effect,weight(i,location))) / np.sum(weight(i,location))

def tstat(i,location,effect):
    if N(i,location)<6:
        return 0.0
    else:
        SSEweighted = N(i,location)* \
            (np.sum(np.multiply(weight(i,location),effect**2))/ \
             np.sum(weight(i,location)) - ghat(i,location,effect)**2)
        return ghat(i,location,effect)*np.sqrt(N(i,location))/ \
            np.sqrt(SSEweighted/(N(i,location)-1))

def pstat(i,location,effect):
    return 1.0 - t.cdf(tstat(i,location,effect), N(i,location)-1)  # N-1 d.f.

def signedlogp(i,location,effect):
    t = tstat(i,location,effect)
    p = pstat(i,location,effect)
    if p < 1e-20:
        return math.copysign(20.0,t) # = sign(t)*20
    else:
        # copysign applies the sign of temp to abs. val. of 1st argument
        return math.copysign(math.log10(p),t) # = sign(t)*(-log10(p))


