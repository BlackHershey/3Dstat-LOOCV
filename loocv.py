# -*- coding: utf-8 -*-
"""
Spyder Editor

Kevin J. Black, M.D.

See https://github.com/BlackHershey/3Dstat-LOOCV for purpose and for
any newer versions.
"""

import math
import numpy as np
from scipy.stats import norm  # for the pdf of the std. normal distribution, used in function weight
from scipy.stats import t  # for function pstat
import csv
import argparse

# Definitions of variables
fwhm = 3.0 # mm
gauss_sd = fwhm/(2*math.sqrt(2*math.log(2)))  # ~1.274 mm
# NOTE: the maximum value of the nl. distribution with this fwhm is 
#    ~0.313, when x = the mean
peak_pdf = 0.31314575956655044 # dimensionless

# Input data
vdata_filename = 'Ventral_Coordinates_xyz_Atl_AG_2-10-16.txt'
ddata_filename = 'Dorsal_Coordinates_xyz_Atl_AG_2-10-16.txt'
real_data = True
if real_data:
    # https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser(description="act on a text effect file")
    parser.add_argument("effect_file", 
                help="text effect file e.g. Valence_Text_File_6-14_AG.csv2")
    args = parser.parse_args()
    # TODO: validate file input etc.
#    data = pd.read_csv(args.effect_file)
    edata = np.genfromtxt(args.effect_file, delimiter=",", names=True,
                         dtype="uint16,float64,S8") 
    subject = edata['subjects']
    effect  = edata['measures']
    dv      = edata['DV']
    location = np.zeros(subject.size)
    #  now read in coordinates data "location" for each line in effect
    vdata = np.genfromtxt(vdata_filename, delimiter=",", names=True)
    ddata = np.genfromtxt(ddata_filename, delimiter=",", names=True)
    for i in range(subject.size):
        assert dv[i]==b'dorsal' or dv[i]==b'ventral', \
            'ERROR: {0} in DV info from effect file.'.format(dv[i])
        if dv[i] == b'dorsal':
            # dv[i].decode() is the regular old string 'ventral', FYI
            location[i] = \
                np.asscalar(vdata[np.where(ddata['DVP_id']==i)])[2:]
        else: # dv[i] == b'ventral':
            location[i] = \
                np.asscalar(vdata[np.where(vdata['DVP_id']==i)])[2:]
else:  # we're testing with a toy dataset
    inputfilename  = '3Dstat_input.csv'
    outputfilename = '3Dstat_loocv.csv'
    n_points = 9
    effect = np.arange(n_points)/10
    subject = np.asarray([1,1,2,2,3,4,5,5,6])
    loc_mean = np.asarray([14.0,-17.0,-3.0])
    loc_sd   = 1.0*np.ones(3)
    location = loc_mean + loc_sd*np.random.randn(n_points,3)
    #  location[0] returns x,y,z for contact location 0

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
    """returns the scalar t from Eisenstein et al 2014 based on a 
    point i, an array location of contact coordinates, and an array effect
    of the effect observed when stimulated at that coordinate
    """
    if N(i,location)<6:
        return 0.0
    else:
        SSEweighted = N(i,location)* \
            (np.sum(np.multiply(weight(i,location),effect**2))/ \
             np.sum(weight(i,location)) - ghat(i,location,effect)**2)
        return ghat(i,location,effect)*np.sqrt(N(i,location))/ \
            np.sqrt(SSEweighted/(N(i,location)-1))

def pstat(i,location,effect):
    """returns the scalar p from Eisenstein et al 2014 based on a 
    point i, an array location of contact coordinates, and an array effect
    of the effect observed when stimulated at that coordinate
    """
    return 1.0 - t.cdf(tstat(i,location,effect), N(i,location)-1)  # N-1 d.f.

def signedlogp(i,location,effect):
    """Returns the scalar value we used to create 2D and 3D p images for
    display, in Eisenstein et al 2014 (because 3D Slicer didn't have
    logarithmic display color scales, and because we wanted to show
    p values where mean effect and t were negative in a different color
    than p values where mean effect and t were positive).
    Input: a point i, an array location of contact coordinates, and 
    an array effect of the effect observed when stimulated at that coordinate
    Output: sign(t)*(-log10(p))
    """
    t = tstat(i,location,effect)
    p = pstat(i,location,effect)
    if N(i,location)<6:
        # TODO: check that Jon's log10p images use zero where N<6
        return 0.0 
    elif p < 1e-20:
        return math.copysign(20.0,t) # = sign(t)*20
    else:
        # copysign applies the sign of temp to _abs. val._ of 1st argument
        return math.copysign(math.log10(p),t) # = sign(t)*(-log10(p))

def loocv(location,effect):
    # TODO: deal with over-writing file with 'w' below, if it exists
    with open(outputfilename,'w') as outfile:
        header='subject,x,y,z,observed,predicted,N,p,signedlog10p'
        outfile.write(header+'\n')
    with open(outputfilename,'a') as outfile:
        writer = csv.writer(outfile)
        for i in range(effect.size):
            # Find index(indices) corresponding to this subject
            s = subject[i]
            s_index = subject==s
            esses = s_index.nonzero()[0]
            # Drop ALL the subject's values from (a copy of) the location and 
            # effect arrays, to create new location and effect arrays with that 
            # subject's values missing.
            loc2 = np.delete(location,esses,axis=0)
            eff2 = np.delete(effect,  esses,axis=0)
    
            # Using those new arrays, report (for ordinate on later plot) the
            # weighted mean for this location at which this subject was stimulated,
            # i.e. the expected effect predicted by all the other subjects' data
            # for stimulation at that location. 
            # BUT, also report N at that location, and the p value at that
            # location, so we can ignore (or weight lower) any prediction made 
            # at locations where we had little data [not counting this subject's
            # data], or at which we had low confidence at that point anyway.
            
            # Sample header and one subject's data for output CSV file:
            # "subject","x","y","z","observed","predicted","N","p"
            # subject_id,x1,y1,z1,effect1,weighted_mean1,N1,p1,logp1
            # subject_id,x2,y2,z2,effect2,weighted_mean2,N2,p2logp2

            row = [subject[i], *location[i], effect[i],
                   ghat(location[i],loc2,eff2), N(location[i],loc2),
                   pstat(location[i],loc2,eff2), 
                   signedlogp(location[i],loc2,eff2)]
            writer.writerow(row)
        # end for loop (for each contact)
    # end with (open outfile)

    # After this loop we should be finished making the file we'll need to
    # test how well we predict--at points where we are relatively more
    # confident that the prediction may be meaningful. NOTE: as we warn
    # in the Eisenstein et al 2014 report, the permutation method we 
    # implemented does not tell us whether any given point in the statistical 
    # images is significant, or whether any given p threshold is low enough 
    # to render the prediction at a given point trustworthy. So any 
    # threshold for p (or for N, for that matter) is arbitrary, and that
    # caveat should be kept in mind in interpreting the results. 

# TODO:
# make this executable and add a main function body, viz.:
# read in the input file
# run loocv(location,effect)
