# networks.py - compute association matrices from time series and
# run network analysis steps on them
#
# Author: Jana Schill
# Contributor: Stefan Fuertinger
# Created: March 05 2020
# Last modified: <2022-02-23 12:04>

import h5py
import numpy as np
import mytools as mt
from makeNetworks.ROIs import ROIs_hemi_sorted
import os
from oct2py import octave as oct
import cutNodes


######switches
emo='disgusted' #which condition to analyze
#emo='neutral'
make_matrices=0 #make association matrices
thresh=0        #threshold matrices
verbose=0       #get more print outputs
metrics=0       #compute network metrics
modules=0       #compute network modules
show_mod=0      #show network modules in a figure
test=0          #run sanity test
do_hubs=0       #compute network hubs

egl=0           #compute global efficiency

ncalc=100

######directories and files

ts_dir = #path to subject time series
savdir = #path to output directory
fn=#filename of h5 container
# create file object to write data
f = h5py.File(savdir+fn,'a')

######maps and dicts

# Subject lists for all groups
mapdict= #dictionary mapping groups to folders with time series data

if emo=='disgusted':
    subdict = #dictionary mapping each group to all subject IDs in the group
elif emo=='neutral':
    subdict = #dictionary mapping each group to all subject IDs in the group
    
# get labels from time series directory (ROIs of all groups are the same) and thus network dimension
rois   = ROIs_hemi_sorted() 
names  = np.array(list(rois.keys()),dtype='S')
labels = np.array(list(rois.values()),dtype='S')

# save ROI information
if not 'names' in f:
    f.attrs.create('names',data=names)
    f.attrs.create('labels',data=labels)


# dictionaries to temporarily save connectivity matrices
nwsdict = dict()
cordict = dict()

# cycle through groups
for name,folder in list(mapdict.items()):
    print('Group: ',name)
    
    #################### computing the correlation matrices
    if make_matrices:
        res = mt.get_corr(ts_dir+emo+os.sep+folder,sublist=subdict[name],corrtype='mi',n_bins=8,verb=verbose)
        corrs=np.nan_to_num(res['corrs']) #sets NaN to 0
        #remove self-connections (zero diagonal)
        nws = mt.rm_selfies(corrs) 
        #compute group network (mean of sunject networks)
        mnw = mt.get_meannw(nws)[0] 
        #save in h5
        if name in f:
            del f[name]
        grp   = f.create_group(name)
        grp.create_dataset('bigmat',data=res['bigmat'])
        grp.create_dataset('corrs',data=corrs)
        grp.create_dataset('nws',data=nws)
        grp.create_dataset('sublist',data=np.array(res['sublist'],dtype='S'))
        grp.create_dataset('mnw',data=mnw)

    #################### threshold matrix (percolation analysis)
    if thresh:
        #threshold subject wise
        try:
            nws=f[name]['nws'][:]
        except:
            raise AttributeError('Networks have not been computed and cannot be accessed.')
         
        nws_thr,sub_dens=mt.perco_thresh(nws)
        
        if name+'/thresh' in f:
            del f[name]['thresh']
        grp   = f.create_group(name+'/thresh')
        grp.create_dataset('nws_thr',data=nws_thr)
        grp.create_dataset('sub_densities',data=sub_dens)

        #threshold mean network
        try:
            mnw=f[name]['mnw'][:]
        except:
            raise AttributeError('Mean network has not been computed and cannot be accessed.')
        mnw_thr,dens=mt.perco_thresh(mnw)
        dens=dens[0]
        print('density for ', name,dens)
        
        grp.create_dataset('mnw',data=mnw_thr)
        grp.create_dataset('density',data=dens)
        
        #compute reduced thresholded mean network (exclude nodes with no remaining connections)
        mnw_red=mnw_thr.copy()
        drop=np.array([])
        for node in range(mnw_thr.shape[0]):
            if max(mnw_red[node,:])==0:
                drop=np.concatenate((drop,node))
                mnw_red=np.delete(mnw_red,node,0)
                mnw_red=np.delete(mnw_red,node,1)
               
        dens_red=mt.density_und(mnw_red)
        grp.create_dataset('drop',data=drop)
        grp.create_dataset('mnw_red',data=mnw_red)
        grp.create_dataset('density_red',data=dens_red)
         
        #make cvs files for nodal coordinates
        mask=cutNodesExcel.cutNodes(drop,name)
        #save reduced names and labels
        names_red=names[mask]
        labels_red=labels[mask]
        grp.attrs.create('names_red',data=names_red)
        grp.attrs.create('labels_red',data=labels_red)

    #################### compute and save metrics on thresholded networks
    if metrics:
        try:
            mnw_red=f[name]['thresh']['mnw_red'][:]
        except:
            raise AttributeError('Reduced thresholded mean network has not yet been computed and cannot be accessed.')
        
        #use octave to use bct to calculate everything
        m_deg = oct.degrees_und(mnw_red)
        m_str = oct.strengths_und(mnw_red)
        m_cc  = oct.clustering_coef_wu(mnw_red)
        m_eff = oct.efficiency_wei(mnw_red)
        
        #save metrics
        if name+'/thresh/m_deg' in f:
            del f[name]['thresh']['m_deg']
            del f[name]['thresh']['m_str']
            del f[name]['thresh']['m_cc']
            del f[name]['thresh']['m_eff']
        
        grp=f[name]['thresh']
        grp.create_dataset('m_deg',data=m_deg)
        grp.create_dataset('m_str',data=m_str)
        grp.create_dataset('m_cc',data=m_cc)
        grp.create_dataset('m_eff',data=m_eff)

    if egl:
        try:
            nws_thr=f[name]['thresh']['nws_thr'][:]
        except:
            raise AttributeError('Thresholded mean network has not yet been computed and cannot be accessed.')
        
        nsubs=nws_thr.shape[-1]
        Eglobs=np.zeros([nsubs,1])
        for s in range(nsubs):
            print(s)
            Eglobs[s] = oct.efficiency_wei(nws_thr[:,:,s])
        
        grp=f[name]['thresh']
        grp.create_dataset('Eglobs',data=Eglobs)

    #################### compute and save modules on thresholded networks  
    if modules:
        try:
            mnw_red=f[name]['thresh']['mnw_red'][:]
        except:
            raise AttributeError('Thresholded mean network has not yet been computed and cannot be accessed.')
        
        cvec=mt.get_modules(mnw_red, ncalc)
        pcoef= oct.participation_coef(mnw_red,cvec);

        
        if name+'/thresh/cvec' in f:
            del f[name]['thresh']['cvec']
            del f[name]['thresh']['pcoef']
        grp=f[name]['thresh']
        grp.create_dataset('cvec',data=cvec)
        grp.create_dataset('pcoef',data=pcoef)
        
    if show_mod:
        try:
            mnw_red=f[name]['thresh']['mnw_red'][:]
            cvec=f[name]['thresh']['cvec'][:]
        except:
            raise AttributeError('Thresholded mean network and/or cvec have not yet been computed and cannot be accessed.')
        
        oct.show_modularity(mnw_red,cvec,'modno',1,'vmin',0,'vmax',1);
        

    #################### compute and save hubs on thresholded networks        
    if do_hubs:
        th_m=0.9
        th_h=0.7
        
        modnum=np.unique(cvec).size #number of modules
        degree   = f[name]['thresh']['m_deg'].value.squeeze()
        strength = f[name]['thresh']['m_str'].value.squeeze()
        pcoef    = f[name]['thresh']['pcoef'].value.squeeze()
        
        # Compute hubs/high-influence nodes
        hubs = []#nodes that are more than 1 SD above the mean for nodal degree or nodal strength
        hin  = []
        for vec in [strength,degree]:
            hubs.append(np.where(vec > (vec.mean() + vec.std()))[0])
            hin.append(np.where(vec >= th_h*vec.max())[0])
        hubs = np.union1d(*hubs)
        hin  = np.setdiff1d(np.intersect1d(*hin),hubs)
        
        # Calculate per-network threshold and classify provincial/connector hubs
        hub_th = th_m*(1-1./modnum)
        con    = hubs[pcoef[hubs] > hub_th]
        prov    = np.setdiff1d(hubs,con)
        
                                           
        hubvec=np.zeros(degree.shape)
        for i in range(hubvec.shape[0]):
            if i in con:
                hubvec[i]=3
            if i in prov:
                hubvec[i]=2
            if i in hin:
                hubvec[i]=1
        if name+'/thresh/con' in f:
            del f[name]['thresh']['con']
            del f[name]['thresh']['prov']
            del f[name]['thresh']['hin']
            del f[name]['thresh']['hubvec']
        grp=f[name]['thresh']
        grp.create_dataset('con',data=con)
        grp.create_dataset('prov',data=prov)
        grp.create_dataset('hin',data=hin)
        grp.create_dataset('hubvec',data=hubvec)
        
f.close()
