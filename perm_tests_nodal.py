# perm_tests_nodal.py - run permutation tests on nodal network metrics
#
# Author: Jana Schill
# Contributor: Stefan Fuertinger
# Created: July 23 2020
# Last modified: <2022-02-23 10:01>

from __future__ import division
import numpy as np
import h5py
import os
import mytools as mt 

emo='disgusted'
#emo='neutral'
nwsname = #name of h5 file

# results directory
resdir = #path to output directory

# directory for saving images 
imgdir = #path to image output directory
if not os.path.exists(imgdir):
    os.makedirs(imgdir)
    
# load data
fd = h5py.File(resdir+'disgusted'+nwsname,'r')
fn = h5py.File(resdir+'neutral'+nwsname,'r')

# get list of all nodes in the networks and all groups in the container
lbl    = fn.attrs['labels'].tolist()
N      = len(lbl)
groups = fn.keys()
lbl = np.array(lbl)


####### compute stats for group-averaged network comparisons
    

# define pairs to compare and metrics
compare = #list of tuples to compare, each tupel contains tuples of group and condition, eg. (('young','n'),('elderly','n')),where n means neutral
metrics = ['m_deg','m_str','m_cc']

# perform statistical analysis for all pairs
for pair in compare:
    print('Computing stats for :',pair)
    # extract network metric values
    vals = {}
    for (group,cond) in pair:
        if cond=='n':
            f=fn
        elif cond=='d':
            f=fd            
        N=len(f[group]['thresh'].attrs['labels_red'])
        nodes = np.arange(N) # define which nodes to include in analysis (e.g. all nodes, hubs, ...)
        n       = nodes.size
        vals[group+cond] = np.zeros((n,len(metrics)))
        m = 0
        for met in metrics:
            vals[group+cond][:,m] = f[group]['thresh'][(met)][:].squeeze()[nodes]
            if met=='m_deg' or met=='m_str':
                vals[group+cond][:,m] = vals[group+cond][:,m]/N #scale degree and strength by number of nodes in network to make network metrics comparable
            m += 1



    fname = resdir+'comparison/'+pair[0][0]+pair[0][1]+'_'+pair[1][0]+pair[1][1]+'_stats'
    if not os.path.exists(resdir+'comparison/'):
        os.makedirs(resdir+'comparison/')

    n_perms=20000 # number of permutations
    T=np.zeros((n_perms+1,len(metrics)))
    pvals=np.zeros((len(metrics),))
    for m in range(len(metrics)):
        print('Looking at :',metrics[m])
        # determine how many values to exchange (half of smaller group)
        length1=len(vals[pair[0][0]+pair[0][1]][:,m])
        length2=len(vals[pair[1][0]+pair[1][1]][:,m])
        nrnd = int(np.round(min(length1,length2)/2))
        lenvec1 = np.arange(length1)
        lenvec2 = np.arange(length2)

        # run permutation test for metric m
        for k in range(n_perms):
            # show progress (will take a long time)
            if k%1000==0.:
                print('We are at ',k,'of ',n_perms,' permutations now.')

            # prepare exchange of values
            met_g1=vals[pair[0][0]+pair[0][1]][:,m].copy()
            met_g2=vals[pair[1][0]+pair[1][1]][:,m].copy()
            # determine which values to exchange
            ridx1 = np.random.choice(lenvec1,nrnd,replace=False)
            ridx2 = np.random.choice(lenvec2,nrnd,replace=False)
            # randomly exchange half of the samples
            met_g1[ridx1] = vals[pair[1][0]+pair[1][1]][:,m][ridx2]
            met_g2[ridx2] = vals[pair[0][0]+pair[0][1]][:,m][ridx1]
            # compute new mean values
            mean_met_g1=np.mean(met_g1)
            mean_met_g2=np.mean(met_g2)
            # compute test statistic
            T[k,m]=abs(mean_met_g1-mean_met_g2)
        # get reference values (actual empirical values)
        mean_ref1=np.mean(vals[pair[0][0]+pair[0][1]][:,m])
        mean_ref2=np.mean(vals[pair[1][0]+pair[1][1]][:,m])
        T_ref=abs(mean_ref1-mean_ref2)
        T[-1,m]=T_ref
        # determine how likely it was to obtain T_ref given T as null distribution
        pvals[m]=(T[:,m]>=T_ref).sum()*1.0/T[:,m].size

    # save evverything
    ft = "Statistical significance of group-differences between "+\
            "samples was assessed using a permutation test.\n"\
    mt.printstats(metrics,pvals,vals[pair[0][0]+pair[0][1]],vals[pair[1][0]+pair[1][1]],pair[0][0]+pair[0][1],pair[1][0]+pair[1][1],foot=ft,fname=fname+str(n_perms)+str(2))

f.close()
