# nulldistributions.py - compute null distributions for density, partition distance and global efficiency
# 
# Author: Jana Schill
# Contributor: Stefan Fuertinger
# Created: September 24 2021
# Last modified: <2022-02-23 11:42>

import numpy as np
import h5py
from oct2py import octave
import mytools as mt

# define pairs of groups to compare
comparisons = #list of tuples to compare, each tupel contains tuples of group and condition, eg. (('young','n'),('elderly','n')),where n means neutral

nwsname = #name of h5 file holding network data
pdname = #name of h5 file in which to save null distributions of partition distance and global efficiency
denname= #name of h5 file in shich to save null distribution of density 

# results directory
resdir = #path to output directory

# load data
fd = h5py.File(resdir+'disgusted'+nwsname,'r')
fn = h5py.File(resdir+'neutral'+nwsname,'r')

# prepare container to save null distributions
fpd = h5py.File(resdir+pdname,'w')
fden = h5py.File(resdir+denname,'w')

# set up number of permutations
n_perms = 20000

# set up arrays to save null distributions, p-values and refernce values
den0 = np.zeros((n_perms,))
den_pvals = np.zeros((len(comparisons),))
denrefs = np.zeros((len(comparisons),))

pd0 = np.zeros((n_perms,))
pd_pvals = np.zeros((len(comparisons),))
pdrefs = np.zeros((len(comparisons),))

eg0 = np.zeros((n_perms,))
eg_pvals=np.zeros((len(comparisons),))
egrefs=np.zeros((len(comparisons),))


for ck, comp in enumerate(comparisons):
    print('Now in '+comp[0][0]+comp[0][1]+' vs '+comp[1][0]+comp[1][1])

    # get reference values
    if comp[0][1]=='n':
        nws1 = fn[comp[0][0]][('nws')][()]
        cref1 = fn[comp[0][0]]['thresh'][('cvec')][()]
        eg1 = fn[comp[0][0]]['thresh'][('m_eff')][()]
        rden1 = fn[comp[0][0]]['thresh'][('density_red')][()]
    else:
        nws1 = fd[comp[0][0]][('nws')][()]
        cref1 = fd[comp[0][0]]['thresh'][('cvec')][()]
        eg1 = fd[comp[0][0]]['thresh'][('m_eff')][()]
        rden1 = fd[comp[0][0]]['thresh'][('density_red')][()]
    if comp[1][1]=='n':
        nws2 = fn[comp[1][0]][('nws')][()]
        cref2 = fn[comp[1][0]]['thresh'][('cvec')][()]
        eg2 = fn[comp[1][0]]['thresh'][('m_eff')][()]
        rden2 = fn[comp[1][0]]['thresh'][('density_red')][()]
    else:
        nws2 = fd[comp[1][0]][('nws')][()]
        cref2 = fd[comp[1][0]]['thresh'][('cvec')][()]
        eg2 = fd[comp[1][0]]['thresh'][('m_eff')][()]
        rden2 = fd[comp[1][0]]['thresh'][('density_red')][()]

    # determine minimun number of subjects in a group
    N=nws1.shape[1]
    nsubs = np.min([nws1.shape[-1],nws2.shape[-1]])
    # determine how many subjects will be exchanged
    nrnd = int(np.round(nsubs/2))
    subvec = np.arange(nsubs)
    
    for k in range(n_perms):
        # show progress via prints
        if k%500==0:
            print('Permutation ',k,' of ',n_perms)

        # prepare exchange of values
        rnws1 = nws1.copy()
        rnws2 = nws2.copy()
        # determine which values to exchange
        ridx = np.random.choice(subvec,nrnd,replace=False)
        # randomly exchange half of the samples
        rnws1[:,:,ridx] = nws2[:,:,ridx]
        rnws2[:,:,ridx] = nws1[:,:,ridx]

        # threshold new 'group' networks and obtain density
        rmnw1,den1 = mt.perco_thresh(mt.get_meannw(rnws1)[0])
        rmnw2, den2 = mt.perco_thresh(mt.get_meannw(rnws2)[0])
        # compute new modular affiliation
        cvec1, dum = octave.community_louvain(rmnw1,nout=2)
        cvec2, dum = octave.community_louvain(rmnw2,nout=2)

        #save entries to null distributions
        den0[k]=abs(den1-den2)
        pd0[k]=octave.partition_distance(np.transpose(cvec1),np.transpose(cvec2))
        eg0[k]=abs(octave.efficiency_wei(rmnw1)-octave.efficiency_wei(rmnw2))

    # compute empirical test statistics and their p-values
    denrefs[ck] = abs(rden1-rden2)
    den_pvals[ck] = (den0 <= denrefs[ck]).sum()/den0.size

    pdrefs[ck] = octave.partition_distance(np.transpose(cref1),np.transpose(cref2))
    pd_pvals[ck] = (pd0 >= pdrefs[ck]).sum()/pd0.size
    
    egrefs[ck] = abs(eg1-eg2)
    eg_pvals[ck] = (eg0 >= egrefs[ck]).sum()/eg0.size
        
    # save null distributions to containers
    grp=fden.create_group(comp[0][0]+comp[0][1]+'_vs_'+comp[1][0]+comp[1][1])
    grp.create_dataset('den',data=den0)

    grp=fpd.create_group(comp[0][0]+comp[0][1]+'_vs_'+comp[1][0]+comp[1][1])
    grp.create_dataset('pd',data=pd0)
    grp.create_dataset('eg',data=eg0)
    

# print results         
print("++++++++++++++++++++++++++++++\n\n")
print('densities:')
for ck, comp in enumerate(compare):
    print(comp[0][0]+comp[0][1]+'_vs_'+comp[1][0]+comp[1][1]+": den = "+str(denrefs[ck])+", p = "+str(den_pvals[ck]))
print('partition distances:')
for ck, comp in enumerate(comparisons):
    print(comp[0][0]+comp[0][1]+'_vs_'+comp[1][0]+comp[1][1]+": pd = "+str(pdrefs[ck])+", p = "+str(pd_pvals[ck]))
print('global efficiencies:')
for ck, comp in enumerate(comparisons):
    print(comp[0][0]+comp[0][1]+'_vs_'+comp[1][0]+comp[1][1]+": eg = "+str(egrefs[ck])+", p = "+str(eg_pvals[ck]))


fpd.close()
    

