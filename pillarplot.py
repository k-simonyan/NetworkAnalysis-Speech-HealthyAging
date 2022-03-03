# pillarplot.py - plot pillars as colored barchart
#
# Author: Jana Schill
# Contributor:  Stefan Fuertinger
# Created: May 26 2015
# Last modified: <2022-03-02 11:08>
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from matplotlib.colors import colorConverter
from matplotlib.lines import Line2D
import descartes
from shapely import affinity, geometry
from collections import OrderedDict
import os
from functools import reduce

# results directory
resdir = #path to results directory

# directory for saving images 
imgdir = #path to directory in which to save figures
if not os.path.exists(imgdir):
    os.makedirs(imgdir)

# load data
fd = h5py.File(resdir+'disgusted.h5','r')
fn = h5py.File(resdir+'neutral.h5','r')

# get list of all nodes in the networks and all groups in the container
lbl    = fn.attrs['labels'].tolist() 
N      = len(lbl)
groups = fn.keys()

lbl = np.array(lbl)

# define metrics to be plotted
metrics = ['degrees','strengths']


# define the colors used in the bar charts
trns=0.8
col = {'nothing':colorConverter.to_rgba('White',1.0),\
       'default_bar':colorConverter.to_rgba('pillarslateGray',0.5),\
       'default_bar_2':colorConverter.to_rgba('DarkGray',0.5),\
       'default_edge':colorConverter.to_rgba('DimGray',1.0),\
       'pillars_edge':colorConverter.to_rgba('Black',1.0),\
       'significant_hub':colorConverter.to_rgba('Lime',1.0),\
       'significant_nod':colorConverter.to_rgba('Lime',0.5),\
       'provincial_hub':colorConverter.to_rgba('Gold',0.5),\
       'provincial_edg':colorConverter.to_rgba('GoldenRod',1.0),\
       'connector_hub':colorConverter.to_rgba('Red',0.5),\
       'connector_edg':colorConverter.to_rgba('Crimson',1.0),\
       'hi_inf':colorConverter.to_rgba('MediumSlateBlue',0.5),\
       'hi_inf_edg':colorConverter.to_rgba('DarkSlateBlue',1.0),\
       'module1':(0.8594,0.0781,0.2344,trns),\
       'module2':(0.1953,0.8008,0.195,trns),\
       'module3':(0.5430,0,0.5430,trns),\
       'module4':(0.8203,0.4102,0.1172,trns),\
       'module5':(0.1172,0.5625,1.0000,trns),\
       'module6':(1.0000,0.8398,0,trns),\
       'textcol':'black'}

# set up dictionaries to hold groupwise values
hubz = {}
hinz = {}
conz  = {}
provz  = {} 
thz  = {}
pillarz={}
colz={}

compare= #list of group-condition tuples to be included in plot, e.g. ('young','n'), where n means neutral
fig_labels=#dictionary mapping group-condition-IDs to figure labels

# extract values from the group-averaged networks 
for (group,cond) in compare:
    if cond=='n':
        f=fn
    elif cond=='d':
        f=fd

    labels_red= f[group]['thresh'].attrs['labels_red']
            
    # get necessary values/arrays
    cvec     = f[group]['thresh']['cvec_r'][()].squeeze()
    mnw      = f[group]['thresh']['mnw'][()]
    con      = f[group]['thresh']['con'][()]
    prov      = f[group]['thresh']['prov'][()]
    hin      = f[group]['thresh']['hin'][()]

    degree   = f[group]['thresh'][('m_deg')][:].squeeze()
    strength = f[group]['thresh'][('m_str')][:].squeeze()

    # compute hubs/high-influence nodes
    hubz[group+cond] = labels_red[np.union1d(con,prov)]
    hinz[group+cond] = labels_red[hin]
    conz[group+cond] = labels_red[con]
    provz[group+cond] = labels_red[prov]

    # compute pillars
    pillars = []
    for vec in [strength,degree]:
        pillars.append(np.where(vec > (vec.mean() + 1.5*vec.std()))[0])           
    pillars = np.union1d(*pillars)
    pillarz[group+cond]=labels_red[pillars]
    colz[group+cond]=cvec

# get hubs across all groups and conditions
allhubs = reduce(np.union1d,hubz.values())
# get hubs that are shared between groups and conditions
sharedhubs=reduce(np.intersect1d,hubz.values())
# get pillars that are shared between groups and conditions
sharedpillars=reduce(np.intersect1d,pillarz.values())
# get number of hubs
n       = allhubs.size
# sort hubs
allhubs=np.array(sorted(allhubs.tolist()))

# set up figure
plt.ioff()
fig=plt.figure(dpi=300,figsize=(6.5,4.5),facecolor='w',tight_layout=True)
gs=fig.add_gridspec(4,hspace=0)
axs=gs.subplots(sharex=True,sharey=True)

# get values for plotting
cnt = 0
for met in metrics:
    gval = {} 
    for (group,cond) in compare:
        if cond=='n':
            f=fn
        elif cond=='d':
            f=fd
        if met == 'degrees':
            gval[group+cond] = f[group]['thresh'][('m_deg')][:].squeeze()
        elif met == 'strengths':
            gval[group+cond] = f[group]['thresh'][('m_str')][:].squeeze()

    # allocate dictionaries for coloring the bar chart
    barcolors = {}
    edgcolors = {}
    dotcolors = {}
    means     = {}
    rads      = {}
    for (group,cond) in compare:
        barcolors[group+cond] = {}
        edgcolors[group+cond] = {}
        dotcolors[group+cond] = {}
        means[group+cond]     = np.zeros((n,))
        rads[group+cond]      = np.full((n,),'    ')

    # extract values and assign colors for each region
    k = 0
    for hb in allhubs:
        for (group,cond) in compare:
            if cond=='n':
                f=fn
            elif cond=='d':
                f=fd
            labels_red    = f[group]['thresh'].attrs['labels_red']
            try:
                # get name of hub
                ind=labels_red.tolist().index(hb)
                # get group-averaged network values
                means[group+cond][k] = gval[group+cond][ind]
                # get mocule color
                dotcolors[group+cond][k] =col['module'+str(int(colz[group+cond][ind]))]
            except: #if the hub node is not a node in our network, set metric to 0 
                means[group+cond][k] = 0
                dotcolors[group+cond][k] =col['nothing']

            # determine colors and radii for plotting
            if cnt == 1:
                barcolors[group+cond][k] = col['default_bar']
            else:
                barcolors[group+cond][k] = col['default_bar_2']
            edgcolors[group+cond][k] = col['default_edge']
            if hb in conz[group+cond]:
                rads[group+cond][k] = 'con'
            elif hb in provz[group+cond]:
                rads[group+cond][k] = 'prov'
            elif hb in hinz[group+cond]:
                rads[group+cond][k] = 'hin'
            else:
                dotcolors[group+cond][k] = col['nothing']
                rads[group+cond][k] = 'nothing'
            if hb in pillarz[group+cond]:
                barcolors[group+cond][k] = dotcolors[group+cond][k]

        k += 1


    radr=7
    # plot the bar charts as subfigures
    for i,(group,cond) in enumerate(compare):
        axs[i].bar(np.arange(0,n),means[group+cond],\
            color=barcolors[group+cond].values(),edgecolor=edgcolors[group+cond].values(),\
            align='center')
        plt.ylim([0,260])
        plt.xlim([-1,n])
        axs[i].tick_params(labelsize=6)
        axs[i].set_ylabel(fig_labels[group+cond])
        dotc=[dotcolors[group+cond][k] for k in np.arange(0,n)]
        dotsz=[radr if rads[group+cond][k]=='con' else radr*0.3 if rads[group+cond][k]=='prov' else 0 if rads[group+cond][k]=='hin' else 0 for k in np.arange(0,n)]
        axs[i].scatter(np.arange(0,n),np.ones([1,n])*220,c=dotc, s=dotsz,edgecolors=dotc,marker='o')
        
    cnt+=1

# add x-ticks  
lbl = allhubs
lbl=[lab.decode("utf-8") for lab in lbl]
lbl=[lab.replace('LTH','LT') for lab in lbl]
lbl=[lab.replace('RTH','RT') for lab in lbl]
allhubs=np.array(lbl)
plt.xticks(np.arange(0,n),allhubs,fontsize=6,color=col['textcol'],rotation=70)


# plot
plt.draw()

# save figure
fig.savefig(imgdir+'pillars1_pat.png',bbox_inches="tight",ppad_inches=0,dpi=fig.get_dpi(),format='png',)
