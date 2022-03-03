# cvec_perm.py - little helper script to compute all permutations of a modular
# affiliation vector and choose the one most similar to a given vector 
#
# Author: Jana Schill
# Created: July 08 2020
# Last modified: <2022-02-22 09:59>
import numpy as np
import h5py
from itertools import permutations 


def cvec_perm(cvec1,cvec2):
    '''
    Compute a permutation of cvec2 so that it is maximilly similar to cvec1.
    Modular affiliations will be kept, but module numbers are permuted.

    Parameters
    ----------
    cvec1 : array
        Target modular affiliation vector
    cvec2 : array
        Modular affiliation vector to be permuted such that it is most simular to
        cvec1

    Returns
    -------
    cvec : array
        Permutation of cvec2 that is most similar to cvec1 
    '''
    maxmod=max(cvec2) # number of modules in cvec2

    # determine all possible permutations of modules in cvec2
    ps = list(permutations(range(1, maxmod+1)))

    # initialise array to hold all permutations of cvec2
    vecs=np.zeros([len(ps),len(cvec2)])

    # for every modular permutation
    for counter,p in enumerate(ps):

        # create new vector according to permutation p
        new_vec2=[p[node-1] for node in cvec2]
        vecs[counter,:]=new_vec2

    # compute for each new vector how much it differs from cvec1
    # and remember the index of the vector with minimum difference
    diff=len(cvec1)+1
    ind=None
    for c,v in enumerate(vecs):
        d=sum(((cvec1-v)!=0).astype(int))
        if d<diff:
            diff=d
            ind=c

    # choose the vector with minimum difference and return it
    cvec=vecs[ind]
    return cvec

# Results directory to use
resdir = #path to results directory

# Load data
fd = h5py.File(resdir+'filename_disgusted.h5','a')
fn =h5py.File(resdir+'filename_neutral.h5','a')
cvec_d_group1     = fd['group1']['thresh']['cvec'][()].astype(int)[0]
cvec_n_group1     = fn['group1']['thresh']['cvec'][()].astype(int)[0]
cvec_d_group2     = fd['group2']['thresh']['cvec'][()].astype(int)[0]
cvec_n_group2     = fn['group2']['thresh']['cvec'][()].astype(int)[0]

# compute cvec for group1 (disgusted condition) based on group1 (neutral condition)
cvec_d_group1=cvec_perm(cvec_n_group1,cvec_d_group1)

# compute cvec for group2 (neutral condition) based on group1 (neutral condition)
cvec_n_group2=cvec_perm(cvec_n_group1,cvec_n_group2)

# compute cvec group2 (disguted condition) based on group2 (neutral condition
cvec_d_group2=cvec_perm(cvec_n_group2,cvec_d_group2)

### save everything in the respective containers
if 'group1'+'/thresh/cvec_r' in fd:
    del fd['group1']['thresh']['cvec_r']
grp=fd['group1']['thresh']
grp.create_dataset('cvec_r',data=cvec_d_group1)

if 'group2'+'/thresh/cvec_r' in fn:
    del fn['group2']['thresh']['cvec_r']
grp=fn['group2']['thresh']
grp.create_dataset('cvec_r',data=cvec_n_group2)

if 'group2'+'/thresh/cvec_r' in fd:
    del fd['group2']['thresh']['cvec_r']
grp=fd['group2']['thresh']
grp.create_dataset('cvec_r',data=cvec_d_group2)

# for continuity, save unchanged cvec_n_group1 as floats
if 'group1'+'/thresh/cvec_r' in fn:
    del fn['group1']['thresh']['cvec_r']
grp=fn['group1']['thresh']
grp.create_dataset('cvec_r',data=cvec_n_group1.astype(float))

fn.close()
fd.close()
