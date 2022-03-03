# cutNodes.py - function to eliminate nodes from network
# 
# Author: Jana Schill
# Created: September 26 2019
# Last modified: <2022-02-21 11:10:00>

import csv
import numpy as np

def cutNodes(drop, group=None):
    '''
    Create a mask for a network that does not contain the nodes in drop.

    Also create csv files for further analysis.

    Parameters
    ----------
    drop : array
        Array containing the indices of nodes to be dropped from the network.
    group : string
        Name of the group for which nodes are to be dropped.

    Returns
    -------
    mask : array
        Boolean array indicating which nodes are in the network 
    '''
    fpath=#path to directory containing the csv files
    sav=fpath+'created/'

    # read nodes and their indices from csv file
    ind,names=np.loadtxt(fpath+'roi_indices.csv', delimiter=",", dtype='S',unpack=True)
    coords=np.loadtxt(fpath+'212coordinates.csv', delimiter=",", dtype='S')
    coords_f=np.loadtxt(fpath+'coords_filesorted.csv', delimiter=",", dtype='S')

    # get names of nodes to be dropped
    if drop.size==0:
        delete=np.array([])
    else:
        delete=names[drop]

    # create new csv file containing only coordinates of nodes that were not dropped
    tab3=[row[1:4] for row in coords if not np.isin(row[0],delete)]
    tab3_a=np.array(tab3,dtype=np.int)
    np.savetxt(sav+'nodes_reduced_'+group+str(tab3_a.shape[0])+'.csv', tab3_a, delimiter=",")

    # create new csv file containing only coordinates of nodes that were not dropped, but in 'filesorted' order
    tab4=[row[1:4] for row in coords_f if not np.isin(row[0],delete)]
    tab4_a=np.array(tab4,dtype=np.int)
    np.savetxt(sav+'nodes_reduced_filesorted_'+group+str(tab3_a.shape[0])+'.csv', tab4_a, delimiter=",")

    # create new csv file containing the remaining nodes in filesorted order and append the dropped nodes
    tab5=[row[1:4] for row in coords_f if np.isin(row[0],delete)]
    tab5_a=np.array(tab5,dtype=np.int) 
    if len(tab5)==0:
        tab5_b=tab4_a.copy()
    else:
        tab5_b=np.concatenate((tab4_a,tab5_a),axis=0)
    np.savetxt(sav+'nodes_reduced_filesorted_212_'+group+str(tab3_a.shape[0])+'.csv', tab5_b, delimiter=",")

    # return mask (bolean array) which is 1 for nodes in the network, 0 otherwise
    mask=np.isin(names,delete,invert=True)
    return mask
