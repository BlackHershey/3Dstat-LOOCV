# -*- coding: utf-8 -*-
"""
Spyder Editor

Kevin J. Black, M.D.

See https://github.com/BlackHershey/3Dstat-LOOCV for purpose and for
any newer versions.
"""

#import sys # for exit(), used if DEBUG
import argparse
import os
import math
import numpy as np
from scipy.stats import norm  # for the pdf of the std. normal distribution, 
# used in function weight
from scipy.stats import t  # for function pstat
import csv

ln2 = math.log(2)
DEBUG = True

# Read real data from files? (as opposed to generate a toy data set)
real_data = True

# Filter results by p value at each active contact location?
# (Don't see why we should--info from where DBS does nothing should be
# equally as valuable. Could imagine filtering by N, though.)
filter_results_p = False
filter_results_N = True

# Definitions of variables
fwhmmax = 3.0 # mm
fwhmmin = 3.0 # mm
# check it out every 0.5mm
fwhm = np.linspace(fwhmmin,fwhmmax,num=int(round(1+2*(fwhmmax-fwhmmin),0)))
# NOTE: in the Eisenstein et al. 2014 paper, we used FWHM = 3.0mm.

# Default input data filenames
#default_data_dir = os.path.join(os.getcwd(),'data','revision')
default_data_dir = os.getcwd()
default_vdata_filename = 'Ventral_Contact_Coordinate_Locations.txt'
default_ddata_filename = 'Dorsal_Contact_Coordinate_Locations.txt'

#######################
# FUNCTION DEFINITIONS
#######################

def parse_arguments():
    # https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser(
            description="act on a text effect file")
    parser.add_argument("effect", type=str,
            help="text effect file e.g. Valence_Text_File_6-14_AG.csv")
    parser.add_argument('-d',"--dorsal", type=str,
            help="file with dorsal contact coordinates, e.g. "+
                default_ddata_filename, 
            default=os.path.join(default_data_dir,default_ddata_filename))
    parser.add_argument('-v',"--ventral", type=str,
            help="file with ventral contact coordinates, e.g. "+
                default_vdata_filename,  
            default=os.path.join(default_data_dir,default_vdata_filename))
    parser.add_argument('-w','--write', dest='write_results', 
            help="write out results into .csv files",
            action='store_true')
    parser.add_argument('-w-','--no-write', dest='write_results', 
            help="do not write results into .csv files",
            action='store_false')
    parser.set_defaults(write_results=False)
    return parser.parse_args()

def get_data(real_data=True):
    """Reads files named on command line (or defaults), and returns
    the following numpy arrays:
    subject, effect, dv, location, vdata, ddata
    """
    if real_data:
        # TODO: validate file input etc.
        effectfilename = args.effect
        ddata_filename = args.dorsal
        vdata_filename = args.ventral
        edata = np.genfromtxt(effectfilename, delimiter=",", names=True,
                             dtype="uint16,float64,S8") 
        outroot, fileext = os.path.splitext(effectfilename)
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
                # dv[i].decode() is the regular old string 'dorsal', FYI
                location[i] = \
                    np.asscalar(ddata[np.where(ddata['DVP_id']==\
                                               edata['subjects'][i])])[2:]
            else: # dv[i] == b'ventral':
                location[i] = \
                    np.asscalar(vdata[np.where(vdata['DVP_id']==\
                                               edata['subjects'][i])])[2:]
        return outroot, subject, effect, dv, location
    else:  # if not real_data, we're testing with a toy dataset
        # inputfilename  = '3Dstat_input.csv'
        outroot = 'test'
        n_points = 9
        subject = np.asarray([1,1,2,2,3,4,5,5,6])
        effect = np.arange(n_points)/10
        loc_mean = np.asarray([14.0,-17.0,-3.0])
        loc_sd   = 1.0*np.ones(3)
        location = loc_mean + loc_sd*np.random.randn(n_points,3)
        #  location[0] returns x,y,z for contact location 0
        dv = np.which(np.random.random_integers(0,1,n_points),
                      b'dorsal',b'ventral')
        return outroot, subject, effect, dv, location

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

def N(x,location,threshold=0.05):
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
        (generally) other points
    """
    return np.sum(np.multiply(effect,weight(i,location)))/ \
            np.sum(weight(i,location))

def tstat(i,location,effect):
    """returns the scalar t from Eisenstein et al 2014 based on a 
    point i, an array location of contact coordinates, and an array effect
    of the effect observed when stimulated at that coordinate
    """
    if N(i,location)<6:
        return 0.0
    else:
        # TODO: IMPORTANT: is this correct? Does this exclude data from
        # contacts whose weight at this point is < .05?
        SSEweighted = N(i,location)* \
            (np.sum(np.multiply(weight(i,location),effect**2))/ \
             np.sum(weight(i,location)) - ghat(i,location,effect)**2)
        if np.sum(np.isnan(SSEweighted))>0:
            print('*** SSEweighted has a NaN value. ***')
        if SSEweighted < 1e-10:
            print('*** ghat denominator problem, SSEweighted = {0} ***'.\
                  format(SSEweighted))
        return ghat(i,location,effect)*np.sqrt(N(i,location))/ \
            np.sqrt(SSEweighted/(N(i,location)-1))

def pstat(i,location,effect):
    """returns the scalar p from Eisenstein et al 2014 based on a 
    point i, an array location of contact coordinates, and an array effect
    of the effect observed when stimulated at that coordinate
    """
    df = N(i,location)-1  # N-1 d.f.
    if df <=0:
        return 1.0 
    else:
        return 1.0 - t.cdf(tstat(i,location,effect), df)

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
        # Note that for 0<p<1, |log10p| = -log10p.
        return math.copysign(math.log10(p),t) # = sign(t)*(-log10(p))

def check_vs_p_image(location,effect,write_results=True):
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
    
    predicted = np.zeros(effect.size)
    pstats    = np.zeros(effect.size)
    ns        = np.zeros(effect.size)
    for i in range(effect.size):
        predicted[i] = ghat(location[i],location,effect)
        pstats[i]    = pstat(location[i],location,effect)
        ns[i]        = N(location[i],location,threshold=.5) # *****
    print('All data: correlation of effect vs. predicted, N={0:d}, r={1:.4f}'.\
          format(effect.size, np.corrcoef(predicted,effect)[0,1]))
    if filter_results_p:
        for p in [0.05, 0.005]:
            mask = np.where(pstats<p)
            print('All data: correlation only for contacts where '+
                  'pstat<{0:.3f}, N={1:d}, r={2:.4f}'.\
                  format(p, len(mask[0]),
                         np.corrcoef(predicted[mask],effect[mask])[0,1]))
    if filter_results_N:
        for Nmin in [6,10,20]:
            mask = np.where(ns>=Nmin)
            print('All data: correlation only for contacts where '+
                  'N>={0:d}, N={1:d}, r={2:.4f}'.\
                  format(Nmin, len(mask[0]),
                         np.corrcoef(predicted[mask],effect[mask])[0,1]))

    if write_results:
        with open(checkfilename,'w') as f:
            header='subject,DV,x,y,z,observed,predicted,N,p,signedlog10p'
            f.write(header+'\n')
        with open(checkfilename,'a') as f:
            writer = csv.writer(f)
            for i in range(effect.size):
                row = [subject[i], dv[i].decode(), *location[i], effect[i],
                       predicted[i], ns[i], pstats[i], 
                       signedlogp(location[i],location,effect)]
                writer.writerow(row)
        print('Check p image at each contact location with {0}'.format(
                checkfilename))

def loocv(location,effect,write_results=True):
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
    global DEBUG
    # TODO: the "DV" column comes from a global variable rather than being 
    #    passed in as a parameter. Fix that?
    predicted = np.zeros(effect.size)
    pstats    = np.zeros(effect.size)
    ns        = np.zeros(effect.size)
    signedlogps = np.zeros(effect.size)
    # TODO: deal with over-writing file with 'w' below, if it exists
    for i in range(effect.size):
        # Find index(indices) corresponding to this subject
        esses = np.where(subject==subject[i])
		# Create new location and effect arrays with 
		# ALL of this subject's contacts missing.
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
        predicted[i] =  ghat(location[i],loc2,eff2)
        ns[i]        = N(location[i],loc2,threshold=.5) # ****
        pstats[i]    = pstat(location[i],loc2,eff2)
        signedlogps[i] = signedlogp(location[i],loc2,eff2)
    # end for loop (for each contact)

    #if DEBUG:
    #    print('predicted,effect:\n',np.around(predicted,2),
    #          '\n',np.around(effect,2))
    #    sys.exit('Exit at line 295. DEBUG is {0}.'.format(DEBUG))
    print('LOOCV: correlation of effect vs. predicted, N={0:d}, r={1:.4f}'.\
          format(effect.size, np.corrcoef(predicted,effect)[0,1]))
    if filter_results_p:
        for p in [0.05, 0.005]:
            mask = np.where(pstats<p)
            print('LOOCV correlation only for contacts where '+
                  'pstat<{0:.3f}, N={1:d}, r={2:.4f}'.\
                  format(p, len(mask[0]),
                         np.corrcoef(predicted[mask],effect[mask])[0,1]))
    if filter_results_N:
        for Nmin in [6,10,20]:
            mask = np.where(ns>=Nmin)
            print('LOOCV: correlation only for contacts where '+
                  'N>={0:d}, N={1:d}, r={2:.4f}'.\
                  format(Nmin, len(mask[0]),
                         np.corrcoef(predicted[mask],effect[mask])[0,1]))
        
    if write_results:
        with open(outputfilename,'w') as outfile:
            header='subject,DV,x,y,z,observed,predicted,N,p,signedlog10p'
            outfile.write(header+'\n')
        with open(outputfilename,'a') as outfile:
            writer = csv.writer(outfile)
            for i in range(effect.size):
                row = [subject[i], dv[i].decode(), *location[i], 
                       effect[i], predicted[i], 
                       ns[i], pstats[i], signedlogps[i]]
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

args = parse_arguments()
outroot, subject, effect, dv, location = get_data(real_data)
for fwhm1 in fwhm: 
    fwhm_string = '_fwhm_' + str(round(fwhm1,2)).replace('.','p') + 'mm'
    outputfilename = outroot + fwhm_string + '_LOOCV.csv'
    checkfilename  = outroot + fwhm_string + '_checkp.csv'
    # The next 2 variables are global variables, used in function weight().
    gauss_sd = fwhm1/(2*math.sqrt(2*ln2))  # ~1.274 mm, for FWHM=3.0mm
    peak_pdf = norm.pdf(0.0,scale = gauss_sd) 
    # NOTE: peak_pdf is the maximum value of the normal distribution with 
    # this fwhm. For FWHM=3, it's ~0.3131 (dimensionless).
    print('\n*** USING FWHM = {0:.1f}: ***'.format(fwhm1))
    check_vs_p_image(location,effect,args.write_results)
    loocv(location,effect,args.write_results)
