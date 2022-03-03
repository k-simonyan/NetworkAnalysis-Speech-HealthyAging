# perm_tests_global.py - run permutation tests on global network metrics
# 
# Author: Jana Schill
# Contributor: Stefan Fuertinger
# Created: October 01 2020
# Last modified: <2022-02-23 10:39>
import numpy as np
import h5py
from oct2py import octave

# name of .h5 containers 
nwsname = #path to container holding network data
denname= #path to container holding null distribution for densities
pdname = #path to container holding null distributions for partition distance and global efficiency

# results directory
resdir = #path to results directory

# load data
fd = h5py.File(resdir+'stats/disgusted'+nwsname,'r')
fn = h5py.File(resdir+'stats/neutral'+nwsname,'r')
fden = h5py.File(resdir+'final/'+denname,'r')
fpd = h5py.File(resdir+'final/'+pdname,'r')
groups = fn.keys()

compare = #list of tuples to compare, each tupel contains tuples of group and condition, eg. (('young','n'),('elderly','n')),where n means neutral

# set up arrays to save p-values and refernce values
den_pvals = np.zeros((len(compare),))
denrefs = np.zeros((len(compare),))
pd_pvals = np.zeros((len(compare),))
pdrefs = np.zeros((len(compare),))
eg_pvals=np.zeros((len(compare),))
egrefs=np.zeros((len(compare),))

for ck, comp in enumerate(compare):

    # get null distributions for density, partition distance and global efficiency
    den0=fden[comp[0][0]+comp[0][1]+'_vs_'+comp[1][0]+comp[1][1]]['den'][()]
    pd0=fpd[comp[0][0]+comp[0][1]+'_vs_'+comp[1][0]+comp[1][1]]['pd'][()]
    eg0=fpd[comp[0][0]+comp[0][1]+'_vs_'+comp[1][0]+comp[1][1]]['eg'][()]
    
    # get reference densities, partition distance and global efficiency
    if comp[0][1]=='n':
        rden1 = fn[comp[0][0]]['thresh'][('density_red')][()]
        cref1 = fn[comp[0][0]]['thresh'][('cvec')][()]
        eg1 = fn[comp[0][0]]['thresh'][('m_eff')][()]
    else:
        rden1 = fd[comp[0][0]]['thresh'][('density_red')][()]
        cref1 = fd[comp[0][0]]['thresh'][('cvec')][()]
        eg1 = fd[comp[0][0]]['thresh'][('m_eff')][()]     
    if comp[1][1]=='n':
        rden2 = fn[comp[1][0]]['thresh'][('density_red')][()]
        cref2 = fn[comp[1][0]]['thresh'][('cvec')][()]
        eg2 = fn[comp[1][0]]['thresh'][('m_eff')][()]
    else:
        rden2 = fd[comp[1][0]]['thresh'][('density_red')][()]
        cref2 = fd[comp[1][0]]['thresh'][('cvec')][()]
        eg2 = fd[comp[1][0]]['thresh'][('m_eff')][()]

    # compute test statistic for density and determine its p-value
    denrefs[ck] = abs(rden1-rden2)
    den_pvals[ck] = (den0 >= denrefs[ck]).sum()/den0.size

    # compute test statistic for partition distance and determine its p-value
    pdrefs[ck] = octave.partition_distance(np.transpose(cref1),np.transpose(cref2))
    pd_pvals[ck] = (pd0 >= pdrefs[ck]).sum()/pd0.size

    # compute test statistic for global efficiency and determine its p-value
    egrefs[ck] = abs(eg1-eg2)
    eg_pvals[ck] = (eg0 >= egrefs[ck]).sum()/eg0.size

    #print everything
    print(comp[0][0]+comp[0][1]+'_vs_'+comp[1][0]+comp[1][1])
    print("den = "+str(denrefs[ck])+", p = "+str(den_pvals[ck]))
    print("pd = "+str(pdrefs[ck])+", p = "+str(pd_pvals[ck]))
    print("eg = "+str(egrefs[ck])+", p = "+str(eg_pvals[ck]))
    print("")

