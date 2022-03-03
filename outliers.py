# outliers.py - determine outliers based on mean correlation strength
#
# Author: Jana Schill
# Created: June 15 2020
# Last modified: <2022-02-23 15:08>
from __future__ import division

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import h5py

# set condition
emo='neutral'
#emo='disgusted'


# subject lists for all groups
mapdict= #dictionary mapping groups to folders with time series data
subdict = #dictionary mapping each group to all subject IDs in the group
savdir = #path to output directory
fn=emo+'.h5'
# create file object to write data
f = h5py.File(savdir+fn,'r')

# cycle through groups
for name,folder in list(mapdict.items()):
    print('Group: ',name)
    # determine number of subjects in the group
    nsub=len(subdict[name])
    # load subject networks
    nws=f[name]['nws']

    ###plot correlation strengths
    nrows=5
    ncols=6
    fig, axs = plt.subplots(nrows, ncols, figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    axs=axs.reshape(nrows*ncols,1)
    cor_str=np.zeros([nsub,1])
    for ax, titl,count in zip(axs, subdict[name],range(nsub)):
        cor_str[count]=np.mean(nws[:,:,count])
        ax[0].imshow(nws[:,:,count],vmin=0,vmax=1,interpolation='none')
        ax[0].set_title(titl.capitalize())
        ax[0].grid(False)
        ax[0].axis('off')
    for ax in axs[nsub:-1]:
        ax[0].axis('off')
    plt.plot(cor_str)
    fig.savefig(savdir+name+'.png')

    # determine correlation strength at different quartiles
    quartiles = np.percentile(cor_str, [25, 50, 75])
    # calculate inter-quartile range
    IQR=quartiles[2]-quartiles[0]
    print(cor_str)
    # determine if correlation strength is 1.5 IQRs below first quartile or
    # 1.5 IQR above third quartile (Tukeys fences)
    bools=(cor_str < (quartiles[0]-1.5*IQR))|(cor_str > (quartiles[2]+1.5*IQR))
    l=[pair[1] for pair in enumerate(subdict[name]) if bools[pair[0]] ]
    print(bools)
    print(l)
