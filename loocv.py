# -*- coding: utf-8 -*-
"""
Spyder Editor

Kevin J. Black, M.D.

See https://github.com/BlackHershey/3Dstat-LOOCV for purpose and for
any newer versions.
"""

import argparse
import os
import math
import numpy as np
from scipy.stats import norm  # for the pdf of the std. normal distribution, 
# used in function weight
from scipy.stats import t  # for function pstat
import csv

ln2 = math.log(2)

# Read real data from files? (as opposed to generate a toy data set)
real_data = True

# Write out results?
write_results = True

# Definitions of variables
fwhm = np.linspace(1.0,4.0,num=1+30)
# NOTE: in the Eisenstein et al. 2014 paper, we used FWHM = 3.0mm.

# Default input data filenames
default_data_dir = os.path.join(os.getcwd(),'data','from_linux')
default_vdata_filename = os.path.join(default_data_dir,
                              'Ventral_Coordinates_xyz_Atl_AG_2-10-16.txt')
default_ddata_filename = os.path.join(default_data_dir,
                              'Dorsal_Coordinates_xyz_Atl_AG_2-10-16.txt')

# Default output filenames
outroot = 'test'
outputfilename = outroot + '_LOOCV.csv'
checkfilename  = outroot + '_checkp.csv'

#######################
# FUNCTION DEFINITIONS
#######################

def get_data(real_data=True):
    """Reads files named on command line (or defaults), and returns
    the following numpy arrays:
    subject, effect, dv, location, vdata, ddata
    """
    global outputfilename, checkfilename
    if real_data:
        # https://docs.python.org/3/howto/argparse.html
        parser = argparse.ArgumentParser(
                description="act on a text effect file")
        parser.add_argument("effect", type=str,
                help="text effect file e.g. Valence_Text_File_6-14_AG.csv")
        parser.add_argument('-d',"--dorsal", type=str,
                help="file with dorsal contact coordinates, e.g. "+
                    default_ddata_filename, 
                default=default_ddata_filename)
        parser.add_argument('-v',"--ventral", type=str,
                help="file with dorsal contact coordinates, e.g. "+
                    default_ddata_filename,  
                default=default_vdata_filename)
        args = parser.parse_args()
        # TODO: validate file input etc.
        effectfilename = args.effect
        ddata_filename = args.dorsal
        vdata_filename = args.ventral
        edata = np.genfromtxt(effectfilename, delimiter=",", names=True,
                             dtype="uint16,float64,S8") 
        outroot, fileext = os.path.splitext(effectfilename)
        outputfilename = outroot + '_LOOCV.csv'
        checkfilename  = outroot + '_checkp.csv'
        subject = edata['subjects']
        effect  = edata['measures']
        dv      = edata['DV']
        location = np.zeros((subject.size,3)) 
        # 1 row, location[i], for each subject, by 3 columns (x,y,z)
        #  now read in coordinates data "location" for each line in effect
        vdata = np.genfromtxt(vdata_filename, delimiter="\t", names=True,
                              dtype='uint16,S1,float64,float64,float64')
        ddata = np.genfromtxt(ddata_filename, delimiter="\t", names=True,
                              dtype='uint16,S1,float64,float64,float64')
        for i in range(subject.size):
            assert dv[i]==b'dorsal' or dv[i]==b'ventral', \
                'ERROR: {0} in DV info from effect file.'.format(dv[i])
            if dv[i] == b'dorsal':  # literal Unicode? byte string
                # dv[i].decode() is the regular old string 'ventral', FYI
                location[i] = \
                    np.asscalar(ddata[np.where(ddata['DVP_id']==\
                                               edata['subjects'][i])])[2:]
            else: # dv[i] == b'ventral':
                location[i] = \
                    np.asscalar(vdata[np.where(vdata['DVP_id']==\
                                               edata['subjects'][i])])[2:]
        return subject, effect, dv, location
    else:  # if not real_data, we're testing with a toy dataset
        # inputfilename  = '3Dstat_input.csv'
        n_points = 9
        subject = np.asarray([1,1,2,2,3,4,5,5,6])
        effect = np.arange(n_points)/10
        loc_mean = np.asarray([14.0,-17.0,-3.0])
        loc_sd   = 1.0*np.ones(3)
        location = loc_mean + loc_sd*np.random.randn(n_points,3)
        #  location[0] returns x,y,z for contact location 0
        dv = np.which(np.random.random_integers(0,1,n_points),
                      b'dorsal',b'ventral')
        return subject, effect, dv, location

# TODO: validate input to all functions

def distance(x,location):
    """returns the Euclidean distance between a point x and each point in a 
    2-D numpy array, location, as a numpy array the same shape as location
    """
    return np.sqrt(np.sum(np.square(x-location), axis=1))

def weight(x,location):
    """returns an array of weights as in Eisenstein et al 2014 based on the 
    distance between the point x and each point in an array 'location' of 
    contact coordinates, scaled by the scalar pdfpeak so that at x=0 the
    (maximal) weight is 1. NOTE: scaling by pdfpeak was inadvertently omitted 
    from the discussion in the paper.
    """
    global gauss_sd, peak_pdf
    return norm.pdf(distance(x,location),loc=0,scale=gauss_sd)/peak_pdf
    # We divide by peak_pdf because in the functions below we want to 
    # threshold at 1/20 of, or 0.05 times, the maximum possible probability 

def N(x,location,pdfpeak=3.0/(2*math.sqrt(2*ln2)),threshold=0.05):
    """returns the scalar value Ni from Eisenstein et al 2014 based on a 
    point x and an array location of contact coordinates. The default
    value for pdfpeak is for FWHM = 3.0mm.
    Note the division in weight(x,location) by the maximum of the p.d.f.
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
    return np.sum(np.multiply(effect,weight(i,location)))/np.sum(weight(i,location))

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
    Output: sign(t)*(-log10(p)), unless N<6, when it returns zero.
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
        # but for 0<p<1, |log10p| = -log10p.
        return math.copysign(math.log10(p),t) # = sign(t)*(-log10(p))

def check_vs_p_image(location,effect):
    """Creates a file 'checkfilename' that should contain more or less
       the same numbers for weighted mean ("predicted"), N and p as 
       are in the weighted mean, N and p images. "More or less" is because
       the point value at the exact contact location is not the mean over
       the volume of the voxel in which that contact is located.
       Input: 
           location: a numpy array with 3 columns x,y,z for each contact
           effect: a 1-D numpy array with the effect observed with stimulation
               at the contact whose location is at the same index in the
               "location" array
       Output:
           a file 'checkfilename' defined near the top of this file.
           signedlog10p means sign(t)*|log10(p)|, or zero where N<6. 
           See the signedlogp() docstring for rationale.
    """
    # TODO: the "DV" column comes from a global variable rather than being 
    #    passed in as a parameter. Fix that?
    
    if write_results:
        with open(checkfilename,'w') as f:
            header='subject,DV,x,y,z,observed,predicted,N,p,signedlog10p'
            f.write(header+'\n')
        with open(checkfilename,'a') as f:
            writer = csv.writer(f)
            for i in range(effect.size):
                row = [subject[i], dv[i].decode(), *location[i], effect[i],
                       ghat(location[i],location,effect), 
                       N(location[i],location),
                       pstat(location[i],location,effect), 
                       signedlogp(location[i],location,effect)]
                writer.writerow(row)
        print('Check p image at each contact location with {0}'.format(
                checkfilename))


def loocv(location,effect):
    """Creates a file 'outputfilename' that contains, for each contact tested,
       a leave-one-out cross-validation measure of utility of the statistical
       images created by the procedure in Eisenstein et al 2014. 
       Input: 
           location: a numpy array with 3 columns x,y,z for each contact
           effect: a 1-D numpy array with the effect observed with stimulation
               at the contact whose location is at the same index in the
               "location" array
       Output:
           a CSV file 'outputfilename' defined near the top of this file, 
           in which subject, DV, x,y,z and observed effect are copied from 
           the effect file provided as a command-line argument and the D & V 
           location files named near the top of this file. "Predicted" is the
           weighted mean predicted for that location based only on the data
           remaining after removing all data from this subject. N, p and
           signedlog10p are as described for the function check_vs_p_image().
    """
    # TODO: the "DV" column comes from a global variable rather than being 
    #    passed in as a parameter. Fix that?
    if write_results:
    # TODO: deal with over-writing file with 'w' below, if it exists
        with open(outputfilename,'w') as outfile:
            header='subject,DV,x,y,z,observed,predicted,N,p,signedlog10p'
            outfile.write(header+'\n')
        with open(outputfilename,'a') as outfile:
            writer = csv.writer(outfile)
            for i in range(effect.size):
                # Find index(indices) corresponding to this subject
                s = subject[i]
                s_index = subject==s
                esses = s_index.nonzero()[0]  # same as np.where(subject==s)
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
    
                row = [subject[i], dv[i].decode(), *location[i], effect[i],
                       ghat(location[i],loc2,eff2), N(location[i],loc2),
                       pstat(location[i],loc2,eff2), 
                       signedlogp(location[i],loc2,eff2)]
                writer.writerow(row)
            # end for loop (for each contact)
        # end with (open outfile)
        print('LOOCV results written to {0}'.format(outputfilename))

    # After this loop we should be finished making the file we'll need to
    # test how well we predict--at points where we are relatively more
    # confident that the prediction may be meaningful. NOTE: as we warn
    # in the Eisenstein et al 2014 report, the permutation method we 
    # implemented does not tell us whether any given point in the statistical 
    # images is significant, or whether any given p threshold is low enough 
    # to render the prediction at a given point trustworthy. So any 
    # threshold for p (or for N, for that matter) is arbitrary, and that
    # caveat should be kept in mind in interpreting the results. 

# Finally, functions defined, so do what we came to do:

#######################
# main() equivalent
#######################

subject, effect, dv, location = get_data(real_data)
output1 = outputfilename
check1  = checkfilename
for fwhm1 in fwhm: 
    fwhm_string = '_fwhm_' + str(round(fwhm,2)).replace('.','p') + 'mm'
    # define two global variables, to avoid passing them
    # through all the functions down to weight() and recalculating
    # them each time we run weight()
    outputfilename = output1 + fwhm_string
    checkfilename  = check1  + fwhm_string
    gauss_sd = fwhm1/(2*math.sqrt(2*ln2))  # ~1.274 mm, for FWHM=3.0mm
    peak_pdf = norm.pdf(0.0,scale = gauss_sd) 
    # NOTE: peak_pdf is the maximum value of the normal distribution with 
    # this fwhm. For FWHM=3, it's ~0.3131 (dimensionless).
    check_vs_p_image(location,effect)
    loocv(location,effect)
