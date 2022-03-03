# hubplot.py - plot hubs in circles, arranged by module
#
# Author: Jana Schill, Stefan Fuertinger
# Created: May 26 2015
# Last modified: <2022-02-25 14:07>
from __future__ import division
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import h5py
import os

from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

emo='disgusted'
emo='neutral'

all_groups = #list of group names

# name of container with network data
nwsname=emo+'.h5'
# results directory
resdir = #path to results directory


# figure output directory 
savdir = #path to directory in which to save figures
if not os.path.exists(savdir):
    os.makedirs(savdir)
    
# load data
f = h5py.File(resdir+nwsname,'r')

# Plot within-group stars
for group in all_groups:
    # get list of all nodes in the networks
    lbl = f[group]['thresh'].attrs['labels_red'].tolist()
    # make ROI abbreviations more readable for figures
    lbl=[lab.decode("utf-8") for lab in lbl]
    lbl=[lab.replace('-','\n') for lab in lbl]
    lbl=[lab.replace('/','') for lab in lbl]
    lbl=[lab.replace('LTH','LT') for lab in lbl]
    lbl=[lab.replace('RTH','RT') for lab in lbl]
    lbl=[lab.replace('OC4','OC\n4') for lab in lbl]
    lbl=[lab.replace('OC3','OC\n3') for lab in lbl]
    lbl=[lab.replace('PFcm','PF\ncm') for lab in lbl]
    lbl=[lab.replace('LTE','LTE\n') for lab in lbl]
    lbl=[lab.replace('Hip','Hip\n') for lab in lbl]

    # get number of regions
    N   = len(lbl)
    lbl = np.array(lbl)

    # transparency value for plots
    myal = 0.8
    # chose module colors to fit other (non-python) plots
    col_array=np.array([(0,(1,1,1,1)),\
               (0.125,(0.8594,0.0781,0.2344,1)),\
               (0.25,(0.1953,0.8008,0.195,1)),\
               (0.375,(0.5430,0,0.5430,1)),\
               (0.5,(0.8203,0.4102,0.1172,1)),\
               (0.625,(0.1172,0.5625,1.0000,1)),\
               (0.75,(1.0000,0.8398,0,1)),\
               (0.875,(0.8242,0.8242,0.8242,1)),\
               (1.0,(0, 0, 1,1))])
    col_map = LinearSegmentedColormap.from_list('my_cmap',col_array.tolist(),N=64)

    # set up dictionary that will contain all infos for the plot
    plot_dict = {'outer':('hin',0.25),\
                 'inner_1':('con',1.0),\
                 'inner_2':('prov',0.5)}
    
    # load data from container
    print(group)
    cvec     = f[group]['thresh']['cvec_r'][()].squeeze()
    mnw      = f[group]['thresh']['mnw'][()]
    con      = f[group]['thresh']['con'][()]
    prov      = f[group]['thresh']['prov'][()]
    hin      = f[group]['thresh']['hin'][()]

    # determine number of modules
    modnum=np.unique(cvec).size
    
    # detemrine which nodes to plot
    all_nodes = np.union1d(np.union1d(con,prov),hin)
    n         = all_nodes.size

    # sort nodes
    tmp = sorted(lbl[all_nodes].tolist())
    lhb = lbl[all_nodes].tolist()
    lal = [0]*n
    i   = 0
    for tp in tmp:
        lal[i] = all_nodes[lhb.index(tp)]
        i += 1
    all_nodes = np.sort(np.array(lal))

    # get connections between nodes
    hubconns=mnw.copy()[all_nodes,:]
    hubconns=hubconns[:,all_nodes]
    
    # convert global 212-node-based indices to local ones
    for position,(nodes,value) in plot_dict.items():
        # get global indices of nodes
        vec = eval(nodes)
        # set up array to hold local indices
        idx = np.zeros([vec.shape[0],1],dtype=int)
        # set up array to assign module colors to nodes
        clr=np.zeros([2,vec.shape[0]],dtype=float)
        k   = 0
        for node in vec:
            # determine local index of node
            idx[k] = np.where(all_nodes == node)[0]
            # save local index and corresponding module color in clr
            clr[0,k]=np.where(all_nodes == node)[0]
            clr[1,k] = cvec[node]
            k += 1
        # update plot_dict with new information
        plot_dict[position] = (idx,value,clr)

        
    # combine all color vectors (clr for con, prov and hin)
    all_clr=np.hstack([plot_dict['inner_1'][2],plot_dict['inner_2'][2],plot_dict['outer'][2]])

    # sort entire color vector according to module colors
    sind=np.argsort(all_clr[1,:])
    all_clr=all_clr[:,sind]
    
    # sort circle-specific color vectors (con, prov, hin) the same way
    innercon=plot_dict['inner_1'][2]
    sind=np.argsort(innercon[1,:])
    innercon=innercon[:,sind]
    (idx,val,clr)=plot_dict['inner_1']
    plot_dict['inner_1']=(innercon[0,:].astype(int),val,innercon)

    innerprov=plot_dict['inner_2'][2]
    sind=np.argsort(innerprov[1,:])
    innerprov=innerprov[:,sind]
    (idx,val,clr)=plot_dict['inner_2']
    plot_dict['inner_2']=(innerprov[0,:].astype(int),val,innerprov)

    outerhin=plot_dict['outer'][2]
    sind=np.argsort(outerhin[1,:])
    outerhin=outerhin[:,sind]
    (idx,val,clr)=plot_dict['outer']
    plot_dict['outer']=(outerhin[0,:].astype(int),val,outerhin)

    # combine all local indices in one vector 
    indvec=np.hstack([plot_dict['inner_1'][0],plot_dict['inner_2'][0],plot_dict['outer'][0]])
    # prepare new color vector for all circles (now ordered by color (= modules))
    cvec1=np.hstack([plot_dict['inner_1'][2][1,:],plot_dict['inner_2'][2][1,:],plot_dict['outer'][2][1,:]])#[all_clr[1,:]#np.hstack([all_inner_clr[1,:],plot_dict['outer'][2][1,:]])
    # assign color codes to modules (this relates to color-map values)
    cvec1[np.where(cvec1==1)]=0.125
    cvec1[np.where(cvec1==2)]=0.25
    cvec1[np.where(cvec1==3)]=0.375
    cvec1[np.where(cvec1==4)]=0.5
    cvec1[np.where(cvec1==5)]=0.625
    cvec1[np.where(cvec1==6)]=0.75     
    cvec1[np.where(cvec1==7)]=0.875
    cvec1[np.where(cvec1==8)]=1

    #resort cvec1 such that it corresponds to the local indices
    sind=np.argsort(indvec)
    cvec1=cvec1[sind]   # now cvec1 is in the order that we need for networkx

    # construct connection matrix
    mat = hubconns
    
    # build NetworkX graph
    G     = nx.from_numpy_matrix(mat)

    # manually compute nodal positions to have finer control over radii of circles
    rad0 = 2            # con
    rad1  = 3.5         # prov
    rad2  = 5           # hin
    start = np.pi/2     # starting point within circles (start left hem. regions at top = pi/2)

    # initialize position dictionary and get no. of nodes on the different circles
    pos = {}
    ni0 = plot_dict['inner_1'][0].size
    ni1 = plot_dict['inner_2'][0].size
    no  = plot_dict['outer'][0].size
    
    #get number of nodes per module per circle
    all_mods=np.zeros([3,modnum])
    modsi0,ni0_mods=np.unique(plot_dict['inner_1'][2][1,:],return_counts=True)
    for i in range(modnum):
        if i+1 in modsi0:
            all_mods[0,i]=ni0_mods[np.where(modsi0==i+1)]
        else:
            all_mods[0,i]=0
    modsi1,ni1_mods=np.unique(plot_dict['inner_2'][2][1,:],return_counts=True)
    for i in range(modnum):
        if i+1 in modsi1:
            all_mods[1,i]=ni1_mods[np.where(modsi1==i+1)]
        else:
            all_mods[1,i]=0
    modso,no_mods=np.unique(plot_dict['outer'][2][1,:],return_counts=True)
    for i in range(modnum):
        if i+1 in modso:
            all_mods[2,i]=no_mods[np.where(modso==i+1)]
        else:
            all_mods[2,i]=0
    all_mods=all_mods.astype(int)

    l=list()
    for i in range(all_mods.shape[1]):
        if max(all_mods[:,i])==0:
            modnum-=1
            l.append(i)
    all_mods=np.delete(all_mods,l,1)

    
    
    # compute coordinates of nodes on the inner and outer circles
    modangle=2*np.pi/modnum
    counter0=0
    counter1=0
    countero=0
    modtotal=np.sum(all_mods)
    modstart=start
    for mod in range(modnum):
        modnodes=np.sum(all_mods[:,mod])
        segment=modnodes/modtotal*np.pi*2
        modend=modstart+segment
        angles = np.linspace(modstart,modend,all_mods[0,mod]+2)
        for i in range(all_mods[0,mod]):
            pos[plot_dict['inner_1'][0][counter0]] = rad0*np.array([np.cos(angles[i+1]),np.sin(angles[i+1])])
            counter0+=1
        angles = np.linspace(modstart,modend,all_mods[1,mod]+2)
        for i in range(all_mods[1,mod]):
            pos[plot_dict['inner_2'][0][counter1]] = rad1*np.array([np.cos(angles[i+1]),np.sin(angles[i+1])])
            counter1+=1
        angles = np.linspace(modstart,modend,all_mods[2,mod]+2)
        for i in range(all_mods[2,mod]):
            pos[plot_dict['outer'][0][countero]] = rad2*np.array([np.cos(angles[i+1]),np.sin(angles[i+1])])
            countero+=1
        modstart=modend 
        
    # create (correctly ordered) labels for all nodes and assign correct nodal sizes
    labeldict = {}
    svec = np.zeros(all_nodes.shape)
    j = 0
    for idx in all_nodes:
        if idx in con:
            labeldict[j] = lbl[idx]
            svec[j] = 250*2
        elif idx in prov:
            labeldict[j] = lbl[idx]
            svec[j] = 200*2
        elif idx in hin:
            labeldict[j] = lbl[idx]
            svec[j] = 120*2
        else:
            labeldict[j] = ''
        j += 1
        
    # prepare figure
    plt.ioff()
    fig = plt.figure(facecolor='w',figsize=(7.0,7.0),dpi=300,tight_layout=True)
    fig.canvas.set_window_title(group)
    ax = plt.subplot(111,frameon=False)
    ax.set_facecolor('w')
    ax.set_xticks([])
    ax.set_yticks([])

    # draw the network using networkx       
    nx.draw_networkx(G,pos,\
                     ax=ax,\
                     node_size=svec,\
                     node_color=cvec1,\
                     cmap=col_map,\
                     vmin=0.0,\
                     vmax=1.0,\
                     linewidths=0,\
                     width=0.5,\
                     edge_color='w',\
                     edge_cmap=col_map,\
                     edge_vmin=0.,\
                     edge_vmax=1.0,\
                     labels=labeldict,\
                     font_color='k',\
                     font_size=6,\
                     with_labels=True)
    plt.axis('equal')
     
    # determine width of connections based on edge strength
    width=[hubconns[n1,n2]*2 for (n1,n2) in G.edges()]
    # determine color of connection based on color of first node of the connection
    color=[col_map(cvec1[n1])[0:3]+(hubconns[n1,n2],) for (n1,n2) in G.edges()]

    # draw connections
    nx.draw_networkx_edges(G, pos, edge_color=color, width=width, edge_vmin=0., edge_vmax=1.0, ax=ax)

    # determine color of connection based on color of second node of the connection
    color=[col_map(cvec1[n2])[0:3]+(hubconns[n1,n2],) for (n1,n2) in G.edges()]

    # draw connections again (overlay colors)
    nx.draw_networkx_edges(G, pos, edge_color=color, width=width, edge_cmap=col_map, edge_vmin=0., edge_vmax=1.0, ax=ax)

    ax.set_xlim((-5.2, 5.2))
    ax.set_ylim((-5.2, 5.2))

    # save the plot    
    fig.savefig(str(savdir+group+'.png'),bbox_inches="tight",ppad_inches=0,dpi=fig.get_dpi(),format='png')

f.close()
