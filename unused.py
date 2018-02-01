# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:13:34 2018

@author: kevin
"""

def distance(x1,x2):
    """returns Euclidean distance between two 1-D numpy arrays, x1 and x2,
    of the same length
    """
    assert type(x1) == type(x2) == np.ndarray, \
        "{0} and {1} must be numpy arrays".format(x1,x2)
    assert len(x1.shape) == len(x2.shape) == 1, \
        "{0} and {1} must be 1D arrays".format(x1,x2)
    assert x1.size == x2.size, \
        "{0} and {1} must have the same length".format(x1,x2)
    return np.sqrt(np.sum(np.square(x1-x2), axis=0))


def weight_pts(x1,x2):
    """returns the weight as in Eisenstein et al 2014 based on the distance
    between the points x1 and x2
    """
    # TODO: check that x1 and x2 are points, as in distance()
    return norm.pdf(distance(x1,x2),loc=0,scale=gauss_sd)/peak_pdf

# the non-Pythonic way:
def N(location,x):
    """returns the scalar value Ni from Eisenstein et al 2014 based on a 
    point x and an array location of contact coordinates
    """
    sum = 0
    for contact in location:
        if weight_pts(contact,x) >= 0.05:
            sum += 1
    return sum
