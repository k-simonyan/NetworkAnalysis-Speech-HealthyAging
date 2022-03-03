# mytools.py - collection of functions for network analysis
#
# Authors: Stefan Fuertinger, Jana Schill
# Created: December 22 2014
# Last modified: <2022-02-23 14:52>

import os
import natsort
import numpy as np
import fnmatch
from oct2py import octave as oct
from texttable import Texttable


##########################################################################################
def get_corr(txtpath,corrtype='pearson',sublist=[],drop=[],verb=False,**kwargs):
    """
    Compute pair-wise statistical dependence of time-series

    Parameters
    ----------
    txtpath : string
        Path to directory holding ROI-averaged time-series dumped in `txt` files.
        The following file-naming convention is required

                `sNxy_bla_bla.txt`,

        where `N` is the group id (1,2,3,...), `xy` denotes the subject number
        (01,02,...,99 or 001,002,...,999) and anything else is separated
        by an underscore. The files will be read in lexicographic order,
        i.e., `s101_1.txt`, `s101_2.txt`,... or `s101_Amygdala.txt`, `s101_Beemygdala`,...
        See Notes for more details.
    corrtype : string
        Specifier indicating which type of statistical dependence to use to compute
        pairwise correlations. Currently supported options are

                `pearson`: the classical zero-lag Pearson correlation coefficient
                (see NumPy's corrcoef for details)

                `mi`: (normalized) mutual information
                (see the docstring of mutual_info in this module for details)
    sublist : list or NumPy 1darray
        List of subject codes to process, e.g., `sublist = ['s101','s102']`.
        By default all subjects found in `txtpath` will be processed.
    **kwargs : keyword arguments
        Additional keyword arguments to be passed on to the function computing
        the pairwise correlations (currently either NumPy's corrcoef or mutual_info
        in this module).

    Returns
    -------
    res : dict
        Dictionary with fields:

        corrs : NumPy 3darray
            `N`-by-`N` matrices of pair-wise regional statistical dependencies
        of `numsubs` subjects. Format is
                    `corrs.shape = (N,N,numsubs)`,
            s.t.
                    `corrs[:,:,i]` = `N x N` correlation matrix of `i`-th subject
        bigmat : NumPy 3darray
            Tensor holding unprocessed time series of all subjects. Format is
                    `bigmat.shape = (tlen,N,numsubs)`,
            where `tlen` is the maximum time-series-length across all subjects
            (if time-series of different lengths were used in the computation,
            any unfilled entries in `bigmat` will be NumPy `nan`'s, see Notes
            for details) and `N` is the number of regions (=nodes in the networks).
        sublist : list of strings
            List of processed subjects specified by `txtpath`, e.g.,
                    `sublist = ['s101','s103','s110','s111','s112',...]`

    Notes
    -----
    Per-subject time-series do not necessarily have to be of the same length across
    a subject cohort. However, all ROI-time-courses *within* the same subject must have
    the same number of entries.
    For instance, all ROI-time-courses in `s101` can have 140 entries, and time-series
    of `s102` might have 130 entries, but if `s101_2.txt` contains 140 data-points while only
    130 entries are found in `s101_3.txt`, the code will raise a `ValueError`.

    See also
    --------
    numpy.corrcoef, mutual_info
    """

    # Make sure txtpath doesn't contain nonsense and points to an existing location
    if str(txtpath) != txtpath:
        raise TypeError('Input has to be a string specifying the path to the txt-file directory!')
    txtpath = str(txtpath)
    if txtpath.find("~") == 0:
        txtpath = os.path.expanduser('~') + txtpath[1:]
    if not os.path.isdir(txtpath):
        raise ValueError('Invalid directory: '+txtpath+'!')

    # Check corrtype
    try:
        corrtype = corrtype.lower()
    except: raise TypeError('Correlation type input must be a string, not '+type(corrtype).__name__+'!')

    # Check sublist
    if str(sublist) == sublist:
        raise TypeError('Subject codes have to be provided as Python list/NumPy 1darray, not as string!')
    try:
        sublist = list(sublist)
    except:
        raise TypeError('Subject codes have to be provided as Python list/NumPy 1darray, not '+type(sublist).__name__+'!')
    try:
        if verb==1:
            verb=True
        elif verb==0:
            verb=False
    except:
        pass

    if type(verb).__name__ != 'bool':
        raise TypeError('The verb flag must be Boolean!')
    # Get length of sublist (to see if a subject list was provided)
    numsubs = len(sublist)

    # Get list of all txt-files in txtpath and order them lexicographically
    if txtpath[-1] == ' '  or txtpath[-1] == os.sep: txtpath = txtpath[:-1]
    txtfiles = natsort.natsorted(myglob(txtpath,"*.[Tt][Xx][Tt]"), key=lambda y: y.lower())
    if len(txtfiles) < 2: raise ValueError('Found fewer than 2 text files in '+txtpath+'!')

    # If no subject-list was provided, take first subject to get the number of ROIs to be processed
    if numsubs == 0:
        raise ValueError('Zero subjects provided!')

    else:

        # Just take the first entry of user-provided subject list
        subject = sublist[0]

        # Prepare output message
        msg = "Processing "

    # Talk to the user
    substr = str(sublist)
    substr = substr.replace('[','')
    substr = substr.replace(']','')
    print(msg+str(numsubs)+" subjects: "+substr)

    # Get number of regions
    numregs = ''.join(txtfiles).count(subject)

    # Get (actual) number of subjects
    numsubs = len(sublist)

    # Scan files to find time-series length
    tlens = np.zeros((numsubs,))
    for k in range(numsubs):
        if verb:print('Reading time series of subject '+sublist[k])
        roi = 0
        for fl in txtfiles:
            if fl.count(sublist[k]):
                try:
                    ts_vec = np.loadtxt(fl)
                except: raise ValueError("Cannot read file "+fl)
                if roi == 0: tlens[k] = ts_vec.size     # Subject's first TS sets our reference length
                if ts_vec.size != tlens[k]:
                    raise ValueError("Error reading file: "+fl+\
                                     " Expected a time-series of length "+str(int(tlens[k]))+", "+
                                     "but actual length is "+str(ts_vec.size))
                roi += 1

        # Safeguard: stop if subject is missing, i.e., roi = 0 still (weirder things have happened...)
        if roi == 0:
            raise ValueError("Subject "+sublist[k]+" is missing!")

        # Safeguard: stop if subject hast more/fewer ROIs than expected
        elif roi != numregs:
            raise ValueError("Found "+str(int(roi+1))+" time-series for subject "+sublist[k]+", expected "+str(int(numregs)))

    # Check the lengths of the detected time-series
    if tlens.min() <= 2:
        raise ValueError('Time-series of Subject '+sublist[tlens.argmin()]+' is empty or has fewer than 2 entries!')

    # Allocate tensor to hold all time series

    tlens=tlens.astype(int)
    ndrop=len(drop)

    bigmat = np.zeros((tlens.max(),numregs-ndrop,numsubs)) + np.nan

    # Allocate tensor holding correlation matrices of all subjects
    corrs = np.zeros((numregs-ndrop,numregs-ndrop,numsubs))

    # Ready to do this...
    print("Extracting data and calculating "+corrtype.upper()+" coefficients")

    # Cycle through subjects and save per-subject time series data column-wise
    for k in range(numsubs):
        if verb:print('Calculating correlations for subject '+sublist[k])
        col = 0
        for fl in txtfiles:                             #account for 0-indexing by +1
            if fl.count(sublist[k]) and not any(fl.count('_'+str(d+1)+'.') for d in drop):  ####added this, check if it is working!!! - J
                                                #this way, we drop 147 if drop contains 47...
                ts_vec = np.loadtxt(fl)
                ts_vec = ts_vec * (ts_vec > 0)

                if ts_vec.max() == 0:
                    ts_vec += 0.001
                bigmat[:tlens[k],col,k] = ts_vec
                col += 1

        # Compute correlations based on corrtype
        if corrtype == 'pearson':
            corrs[:,:,k] = np.corrcoef(bigmat[:tlens[k],:,k],rowvar=0,**kwargs)
        elif corrtype == 'mi':
            corrs[:,:,k] = mutual_info(bigmat[:tlens[k],:,k],**kwargs)

    corrs=np.fabs(corrs)
    # Happy breakdown
    print("Done")
    return {'corrs':corrs, 'bigmat':bigmat, 'sublist':sublist}

##########################################################################################
def myglob(flpath,spattern):
    """
    Return a glob-like list of paths matching a path-name pattern BUT support fancy shell syntax

    Parameters
    ----------
    flpath : str
        Path to search (to search current directory use `flpath=''` or `flpath='.'`
    spattern : str
        Pattern to search for in `flpath`

    Returns
    -------
    flist : list
        A Python list of all files found in `flpath` that match the input pattern `spattern`

    Examples
    --------
    List all png/PNG files in the folder `MyHolidayFun` found under `Documents`

    >>> filelist = myglob('Documents/MyHolidayFun','*.[Pp][Nn][Gg]')
    >>> print filelist
    >>> ['Documents/MyHolidayFun/img1.PNG','Documents/MyHolidayFun/img1.png']

    See also
    --------
    glob
    """

    # Make sure provided path is a string and does not contain weird unicode characters
    if str(flpath) != flpath:
        raise TypeError('Filepath has to be a string!')
    flpath = str(flpath)
    if flpath.find("~") == 0:
        flpath = os.path.expanduser('~') + flpath[1:]
    if str(spattern) != spattern:
        raise TypeError('Pattern has to be a string!')

    # If user wants to search current directory, make sure that works as expected
    if (flpath == '') or (flpath.count(' ') == len(flpath)):
        flpath = '.'

    # Append trailing slash to filepath
    else:
        if flpath[-1] != os.sep: flpath = flpath + os.sep

    # Return glob-like list
    return [os.path.join(flpath, fnm) for fnm in fnmatch.filter(os.listdir(flpath),spattern)]

##########################################################################################
def mutual_info(tsdata, n_bins=32, normalized=True, norm_ts=True):
    """
    Calculate the (normalized) mutual information matrix at zero lag

    Parameters
    ----------
    tsdata : NumPy 2d array
        Array of data values per time point. Format is: `timepoints`-by-`N`. Note that
        both `timepoints` and `N` have to be `>= 2` (i.e., the code needs at least two time-series
        of minimum length 2)
    n_bins : int
        Number of bins for estimating probability distributions
    normalized : bool
        If `True`, the normalized mutual information (NMI) is computed
        otherwise the raw mutual information (not bounded from above) is calculated
        (see Notes for details).
    norm_ts : bool
        If `True` the input time-series is normalized to zero mean and unit variance (default).

    Returns
    -------
    mi : NumPy 2d array
        `N`-by-`N` matrix of pairwise (N)MI coefficients of the input time-series

    Notes
    -----
    For two random variables :math:`X` and :math:`Y` the raw mutual information
    is given by

    .. math:: MI(X,Y) = H(X) + H(Y) - H(X,Y),

    where :math:`H(X)` and :math:`H(Y)` denote the Shannon entropies of
    :math:`X` and :math:`Y`, respectively, and :math:`H(X,Y)` is their joint
    entropy. By default, this function normalizes the raw mutual information
    :math:`MI(X,Y)` by the geometric mean of :math:`H(X)` and :math:`H(Y)`

    .. math:: NMI(X,Y) = {MI(X,Y)\over\sqrt{H(X)H(Y)}}.

    The heavy lifting in this function is mainly done by code parts taken from
    the `pyunicorn` package, developed by Jonathan F. Donges
    and Jobst Heitzig. It is currently available
    `here <http://www.pik-potsdam.de/~donges/pyunicorn/index.html>`_
    The code has been modified so that weave and pure Python codes are now
    part of the same function. Further, the original code computes the raw mutual information
    only. Both Python and C++ parts have been extended to compute a normalized
    mutual information too.

    See also
    --------
    pyunicorn.pyclimatenetwork.mutual_info_climate_network : classes in this module

    Examples
    --------
    >>> tsdata = np.random.rand(150,2) # 2 time-series of length 150
    >>> NMI = mutual_info(tsdata)
    """

    # Sanity checks
    try:
        shtsdata = tsdata.shape
    except:
        raise TypeError('Input must be a timepoint-by-index NumPy 2d array, not '+type(tsdata).__name__+'!')
    if len(shtsdata) != 2:
        raise ValueError('Input must be a timepoint-by-index NumPy 2d array')
    if (min(shtsdata)==1):
        raise ValueError('At least two time-series/two time-points are required to compute (N)MI!')
    if np.isnan(tsdata).max()==True or np.isinf(tsdata).max()==True or np.isreal(tsdata).min()==False:
        raise ValueError('Input must be a real valued NumPy 2d array without Infs or NaNs!')

    try:
        tmp = (n_bins != int(n_bins))
    except:
        raise TypeError('Bin number must be an integer!')
    if (tmp): raise ValueError('Bin number must be an integer!')

    if type(normalized).__name__ != 'bool':
        raise TypeError('The normalized flag must be Boolean!')

    if type(norm_ts).__name__ != 'bool':
        raise TypeError('The norm_ts flag must be Boolean!')

    #  Get faster reference to length of time series = number of samples
    #  per grid point.
    (n_samples,N) = tsdata.shape

    #  Normalize tsdata time series to zero mean and unit variance
    if norm_ts:
        normalize_time_series(tsdata)

    #  Initialize mutual information array
    mi = np.zeros((N,N), dtype="float32")

    #  Define references to NumPy functions for faster function calls
    histogram = np.histogram
    histogram2d = np.histogram2d
    log = np.log

    #  Get common range for all histograms
    range_min = tsdata.min()
    range_max = tsdata.max()

    #  Calculate the histograms for each time series
    p = np.zeros((N, n_bins))
    for i in range(N):
        p[i, :] = (histogram(tsdata[:, i], bins=n_bins,
                        range=(range_min, range_max))[0]).astype("float64")

    #  Normalize by total number of samples = length of each time series
    p /= n_samples

    #  Make sure that bins with zero estimated probability are not counted
    #  in the entropy measures.
    p[p == 0] = 1

    #  Compute the information entropies of each time series
    H = -(p * log(p)).sum(axis=1)

    #  Calculate only the lower half of the MI matrix, since MI is
    #  symmetric with respect to X and Y.
    for i in range(N):

        for j in range(i):

            #  Calculate the joint probability distribution
            pxy = (histogram2d(tsdata[:, i], tsdata[:, j], bins=n_bins,
                        range=((range_min, range_max),
                        (range_min, range_max)))[0]).astype("float64")

            #  Normalize joint distribution
            pxy /= n_samples

            #  Compute the joint information entropy
            pxy[pxy == 0] = 1
            HXY = -(pxy * log(pxy)).sum()

            # Normalize by entropies (or not)
            if (normalized):
                mi.itemset((i, j), (H.item(i) + H.item(j) - HXY) / (np.sqrt(H.item(i) * H.item(j))))
            else:
                mi.itemset((i, j), H.item(i) + H.item(j) - HXY)

            # Symmetrize MI
            mi.itemset((j, i), mi.item((i, j)))

    # Put ones on the diagonal
    np.fill_diagonal(mi, 1)

# Return (N)MI matrix
    return mi

##########################################################################################
def normalize_time_series(time_series_array):
    """
    Normalizes a (real/complex) time series to zero mean and unit variance.
    WARNING: Modifies the given array in place!

    Parameters
    ----------
    time_series_array : NumPy 2d array
        Array of data values per time point. Format is: `timepoints`-by-`N`

    Returns
    -------
    Nothing : None

    Notes
    -----
    This function does *not* do any error checking and assumes you know what you are doing
    This function is part of the `pyunicorn` package, developed by
    Jonathan F. Donges and Jobst Heitzig. The package is currently available
    `here <http://www.pik-potsdam.de/~donges/pyunicorn/index.html>`_

    See also
    --------
    pyunicorn : A UNIfied COmplex Network and Recurrence aNalysis toolbox

    Examples
    --------
    >>> ts = np.arange(16).reshape(4,4).astype("float")
    >>> normalize_time_series(ts)
    >>> ts.mean(axis=0)
    array([ 0.,  0.,  0.,  0.])
    >>> ts.std(axis=0)
    array([ 1.,  1.,  1.,  1.])
    >>> ts[:,0]
    array([-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079])
    """

    #  Remove mean value from time series at each node (grid point)
    time_series_array -= time_series_array.mean(axis=0)

    #  Normalize the variance of anomalies to one
    time_series_array /= np.sqrt((time_series_array *
                                time_series_array.conjugate()).mean(axis=0))

    #  Correct for grid points with zero variance in their time series
    time_series_array[np.isnan(time_series_array)] = 0

##########################################################################################
def rm_selfies(conns):
    """
    Remove self-connections from connectivity matrices

    Parameters
    ----------
    conns : NumPy 3darray
        An array of `K` connectivity matrices of dimension `N`-by-`N`. Format is
                `conns.shape = (N,N,K)`,
        s.t.
                `conns[:,:,i]` is the `i`-th `N x N` connectivity matrix

    Returns
    -------
    nws : NumPy 3darray
        Same format as input array but `np.diag(conns[:,:,k]).min() = 0.0`.

    Notes
    -----
    None

    See also
    --------
    None
    """

    # Sanity checks
    tensorcheck(conns)

    # Create output quantity and zero its diagonals
    nws = conns.copy()
    for i in range(nws.shape[-1]):
        np.fill_diagonal(nws[:,:,i],0)

    return nws

##########################################################################################
def tensorcheck(corrs):
    """
    Local helper function performing sanity checks on a N-by-N-by-k tensor
    """

    try:
        shc = corrs.shape
    except:
        raise TypeError('Input must be a N-by-N-by-k NumPy array, not '+type(corrs).__name__+'!')
    if len(shc) != 3:
        raise ValueError('Input must be a N-by-N-by-k NumPy array')
    if (min(shc[0],shc[1])==1) or (shc[0]!=shc[1]):
        raise ValueError('Input must be a N-by-N-by-k NumPy array!')
    if np.isnan(corrs).max()==True or np.isinf(corrs).max()==True or np.isreal(corrs).min()==False:
        raise ValueError('Input must be a real valued NumPy array without Infs or NaNs!')

##########################################################################################
def get_modules(nw,ncalc=100,c=None):
    """
    Computes network communities ('modules') using Newman modularity.
    ATTENTION: Needs octave to call functions provided in the Brain Network Toolbox (BNT).

    Parameters
    ----------
    nw : NumPy 2darray
        Connectivity matrix of dimension `N`-by-`N`.

    ncalc : int
        Number of calculations (Newman modularity is a heuristic approach and yields
        different results in every run. Rerunning the finetuning stabilizes the results.)

    c : NumPy 1darray
         Community affiliation vector

    Returns
    -------
    cvec_fin : NumPy 1darray
        Refined community affiliation vector


    Notes
    -----
    None

    See also
    --------
    None
    """

    #sanity check nw
    try:
        shnw = nw.shape
    except:
        raise TypeError('Input must be a N-by-N NumPy array, not '+type(nw).__name__+'!')
    if len(shnw) != 2:
        raise ValueError('Input must be a N-by-N NumPy array')
    if (min(shnw[0],shnw[1])==1) or (shnw[0]!=shnw[1]):
        raise ValueError('Input must be a N-by-N NumPy array!')

    #sanity check ncalc
    try:
        tmp = (ncalc != int(ncalc))
    except:
        raise TypeError('ncalc must be an integer!')
    if (tmp): raise ValueError('ncalc must be an integer!')

    if c is not None:
        try:
            lc=len(c)
        except:
            raise TypeError('c must by a 1d NumPy array!')
        if lc[0]!=shnw[0]:
            raise ValueError('c has to contain a module assignment for every node of nw!')
    if (c is None) or (lc==0):
        c=oct.community_louvain(nw)

    #initialize variables
    N=shnw[0]
    cvec=np.zeros([ncalc,N])
    cv=np.zeros([1,N])

    for n in range(ncalc):
        #finetune modularity
        c,q=oct.modularity_finetune_und(nw,c,nout=2)
        #helpers
        cv[:]=0 #will be changed in following loop, needs to be reset for every run n
        tempc=np.copy(c)

        k=1 #module number
        tempc_min=0

        while np.max(tempc):#while tempc has at least one non-zero element
            nind=np.where(c==tempc[0,tempc_min])
            cv[nind]=k
            tempc[0,nind]=0
            try: #ugly workaround because in the last loop cycle, np.nonzero(tempc) will be empty, therefore indexing it throws an IndexError
                tempc_min=np.nonzero(tempc)[1][0]#find fist non-zero entry
            except IndexError:
                pass
            k=k+1
        cvec[n,:]=cv
    cvec_fin=np.zeros([1,N])
    for col in range(N):
        val,cnt=np.unique(cvec[:,col],return_counts=True)
        mostfreq=val[cnt==cnt.max()]#this is the most frequent result of the ncalc module assignments for node col
        mostfreq=mostfreq[np.random.choice(mostfreq.shape[0], 1, replace=False)][0]#if mostfreq contains several elements, choose one randomly

        cvec_fin[0,col]=mostfreq

    #finally rearrange cvec_fin, so that modules always start with 1 for node 1

    #helpers
    cv[:]=0
    tempc=cvec_fin
    k=1 #module number
    tempc_min=1


    while np.max(tempc):#while tempc has at least one non-zero element
        nind=np.where(cvec_fin==tempc[0,tempc_min])
        cv[nind]=k
        tempc[0,nind]=0
        try: #ugly workaround because in the last loop cycle, np.nonzero(tempc) will be empty, therefore indexing it throws an IndexError
                tempc_min=np.nonzero(tempc)[1][0]#find fist non-zero entry
        except IndexError:
                pass
        k=k+1
    cvec_fin=cv
    return cvec_fin


##########################################################################################
def get_meannw(nws,percval=0.0):
    """
    Helper function to compute group-averaged networks

    Parameters
    ----------
    nws : NumPy 3darray
        `N`-by-`N` connectivity matrices of numsubs subjects. Format is
                `nws.shape = (N,N,numsubs)`,
        s.t.
                `nws[:,:,i] = N x N` connectivity matrix of `i`-th subject
    percval : float
        Percentage value, s.t. connections not present in at least `percval`
        percent of subjects are not considered, thus `0 <= percval <= 1`.
        Default setting is `percval = 0.0`

    Returns
    -------
    mean_wghted : NumPy 2darray
        `N`-by-`N` mean value matrix of `numsubs` matrices stored in `nws` where
        only connections present in at least `percval` percent of subjects
        are considered
    percval : float
        Percentage value used to generate `mean_wghted`

    Notes
    -----
    If the current setting of `percval` leads to a disconnected network,
    the code increases `percval` in 5% steps to assure connectedness of the group-averaged graph.
    The concept of using only a certain percentage of edges present in subjects was taken from
    M. van den Heuvel, O. Sporns: "Rich-Club Organization of the Human Connectome" (2011), J. Neurosci.
    Currently available `here <http://www.jneurosci.org/content/31/44/15775.full>`_

    See also
    --------
    None
    """

    # Sanity checks
    tensorcheck(nws)
    try: tmp = percval > 1 or percval < 0
    except: raise TypeError("Percentage value must be a floating point number >= 0 and <= 1!")
    if (tmp): raise ValueError("Percentage value must be >= 0 and <= 1!")

    # Get shape of input tensor
    N       = nws.shape[0]
    numsubs = nws.shape[-1]

    # Remove self-connections
    nws = rm_selfies(nws)

    # Allocate memory for binary/weighted group averaged networks
    mean_binary = np.zeros((N,N))
    mean_wghted = np.zeros((N,N))

    # Compute mean network and keep increasing percval until we get a connected mean network
    docalc = True
    while docalc:

        # Reset matrices
        mean_binary[:] = 0
        mean_wghted[:] = 0

        # Cycle through subjects to compute average network
        for i in range(numsubs):
            mean_binary = mean_binary + (nws[:,:,i]!=0).astype(float)
            mean_wghted = mean_wghted + nws[:,:,i]

        # Kick out connections not present in at least percval% of subjects (in binary and weighted NWs)
        mean_binary = (mean_binary/numsubs >= percval).astype(float)
        mean_wghted = mean_wghted/numsubs * mean_binary

        # Check connectedness of mean network
        if degrees_und(mean_binary).min() == 0:
            print("WARNING: Mean network disconnected for percval = "+str(np.round(1e2*percval))+"%")
            if percval < 1:
                print("Decreasing percval by 5%...")
                percval -= 0.05
                print("New value for percval is now "+str(np.round(1e2*percval))+"%")
            else:
                msg = "Mean network disconnected for percval = 0%. That means at least one node is "+\
                      "disconnected in ALL per-subject networks..."
                raise ValueError(msg)
        else:
            docalc = False

    return mean_wghted, percval

##########################################################################################
def degrees_und(CIJ):
    """
    Compute nodal degrees in an undirected graph

    Parameters
    ----------
    CIJ : NumPy 2darray
        Undirected binary/weighted connection matrix

    Returns
    -------
    deg : NumPy 1darray
        Nodal degree vector

    Notes
    -----
    This function does *not* do any error checking and assumes you know what you are doing

    See also
    --------
    degrees_und.m : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available
                    `here <https://sites.google.com/site/bctnet/>`_

    bctpy : An unofficial Python port of the BCT is currently available at the
            `Python Package Index <https://pypi.python.org/pypi/bctpy>`_
            and can be installed using pip.
    """

    return (CIJ != 0).sum(1)

##########################################################################################
def broken(nw_thr,mod_asg,mod):
    """
    Use breath-first search to determine if a thresholded network still contains enough
    edges to keep a specific module connected

    Parameters
    ----------
    nw_thr : NumPy 2darray
        Thresholded network of size NxN, where N is the number of nodes.

    mod_asg :NumPy 1darray
        Module assignment of the original (unthresholded) network. Must be of size 1xN,
        where N is the number of nodes.

    mod : int
        The module in question. Must be an element of mod_asg.

    Returns
    -------
    broken : Boolean
        Boolean indicating whether the module in question is still fully connected in the
        thresholded network. True if at least one node belonging to the module is no longer
        connected to the module (that is, any other node belonging to the module).
    """

    ind=np.where(mod_asg==mod)[0]

    root=ind[0]

    if not nw_thr.max()==0:
        conns=bfs(nw_thr,root)
        #we must not forget the root node, which belongs to the module but is not connected
        #to itself...
        conns=np.unique(np.concatenate((conns,np.array([root]))))
        broken=not min(np.in1d(ind,conns))
    else:
        broken=True

    return broken

##########################################################################################
def bfs(nw,node):
    """
    Use breadth first search to find component in network 'nw' which node 'node'
    belongs to.


    Parameters
    ----------
    nw : NumPy 2darray
        The network to be searched.

    node : int
        The starting point for the search.

    Returns
    -------
    component : NumPy array
        All nodes that are connected to 'node'.

    """
    # set up queue of all nodes to be traversed
    queue=np.array([node])
    # set up array containing all nodes that were already traversed
    seen=np.array([])
    # set up array to save all traversed nodes
    component=np.array([])
    # as long as there are still nodes to traverse
    while not queue.size==0:
        # take first node in queue
        node=queue[0]
        # add to component (if it is not already there)
        component=np.unique(np.concatenate((component,np.array([node]))))
        # delete node from queue
        queue=np.delete(queue,0)
        # determine which nodes 'node' has connections with
        conns=np.nonzero(nw[node,:])[0]
        # add all nodes connected to 'node' that have not yet been traversed to queue
        queue=np.unique(np.concatenate((queue,np.setdiff1d(conns,seen))))
        # consider all these nodes as traversed
        seen=np.unique(np.concatenate((seen,conns)))

    return component

##########################################################################################
def density_und(CIJ):
    """
    Compute the connection density of an undirected graph

    Parameters
    ----------
    CIJ : NumPy 2darray
        Undirected binary/weighted connection matrix

    Returns
    -------
    den : float
        density (fraction of present connections to possible connections)

    Notes
    -----
    This function does *not* do any error checking and assumes you know what you are doing

    See also
    --------
    density_und.m : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available
                    `here <https://sites.google.com/site/bctnet/>`_

    bctpy : An unofficial Python port of the BCT is currently available at the
            `Python Package Index <https://pypi.python.org/pypi/bctpy>`_
            and can be installed using pip.
    """

    N = CIJ.shape[0]                    # no. of nodes
    K = (np.triu(CIJ,1)!=0).sum()       # no. of edges
    return K/((N**2 - N)/2.0)

##########################################################################################

def printstats(variables,pvals,group1,group2,g1str='group1',g2str='group2',foot='',verbose=True,fname=None):
    """
    Pretty-print previously computed statistical results

    Parameters
    ----------
    variables : list or NumPy 1darray
        Python list/NumPy array of strings representing variables that have been tested
    pvals : Numpy 1darray
        Aray of `p`-values (floats) of the same size as `variables`
    group1 : NumPy 2darray
        An #samples-by-#variables array holding the data of the first group sample used in the previously
        performed statistical comparison
    group2 : NumPy 2darray
        An #samples-by-#variables array holding the data of the second group sample used in the previously
        performed statistical comparison
    g1str : string
        Name of the first group that will be used in the generated table
    g2str : string
        Name of the first group that will be used in the generated table
    fname : string
        Name of a csv-file (with or without extension '.csv') used to save the table
        (WARNING: existing files will be overwritten!). Can also be a path + file-name
        (e.g., `fname='path/to/file.csv'`). By default output is not saved.

    Returns
    -------
    Nothing : None

    Notes
    -----
    Uses the `texttable` module to print results

    See also
    --------
    texttable : a module for creating simple ASCII tables (currently available at the
                `Python Package Index <https://pypi.python.org/pypi/texttable/0.8.1>`_)
    printdata : a function that pretty-prints/-saves data given in an array (part of
                `nws_tools.py <http://research.mssm.edu/simonyanlab/analytical-tools/nws_tools.printdata.html#nws_tools.printdata>`_)
    """

    # Make sure that the groups, p-values and tested variables have appropriate dimensions
    if not isinstance(variables,(list,np.ndarray)):
        raise TypeError('Input variables must be a Python list or NumPy 1d array of strings, not '+\
                        type(variables).__name__+'!')
    m = len(variables)
    for var in variables:
        if not isinstance(var,str):
            raise TypeError('All variables must be strings!')

    if not isinstance(pvals,(list,np.ndarray)):
        raise TypeError('The p-values must be provided as NumPy 1d array, not '+type(variables).__name__+'!')
    pvals = np.array(pvals)
    if not np.isfinite(pvals.any()):       # Don't check for NaNs and Infs - some tests might return that...
        raise ValueError('Provided p-values must be real-valued!')
    M = pvals.size
    if M != m:
        raise ValueError('No. of variables (=labels) and p-values do not match up!')

    # Don't check for NaNs and Infs - just make sure we can compute mean and std
    try:
        N,M = group1.shape
    except:
        raise TypeError('Data-set 1 must be a NumPy 2d array, not '+type(group1).__name__+'!')
    if M != m:
        raise ValueError('No. of variables (=labels) and dimension of group1 do not match up!')
    if not np.isfinite(group1.any()):
        raise ValueError('Provided value-arrays must be real-valued!')
    try:
        N,M = group2.shape
    except:
        raise TypeError('Data-set 2 must be a NumPy 2d array, not '+type(group2).__name__+'!')
    if M != m:
        raise ValueError('No. of variables (=labels) and dimension of group2 do not match up!')
    if not np.isfinite(group2.any()):
        raise ValueError('Provided value-arrays must be real-valued!')

    # If column labels were provided, make sure they are printable strings
    if not isinstance(g1str,str):
        raise TypeError('The optional column label `g1str` has to be a string!')
    if not isinstance(g2str,str):
        raise TypeError('The optional column label `g2str` has to be a string!')

    # If a footer was provided, make sure it is a printable string
    if not isinstance(foot,str):
        raise TypeError('The optional footer `foot` has to be a string!')

    # See if we're supposed to print stuff to the terminal or just save everything to a csv file
    if not isinstance(verbose,bool):
        raise TypeError("The switch `verbose` has to be Boolean!")

    # If a file-name was provided make sure it's a string and check if the path exists
    if fname != None:
        if not isinstance(fname,str):
            raise TypeError('Input fname has to be a string specifying an output file-name, not '\
                            +type(fname).__name__+'!')
        fname = str(fname)
        if fname.find("~") == 0:
            fname = os.path.expanduser('~') + fname[1:]
        slash = fname.rfind(os.sep)
        if slash >= 0 and not os.path.isdir(fname[:fname.rfind(os.sep)]):
            raise ValueError('Invalid path for output file: '+fname+'!')
        if fname[-4::] != '.csv':
            fname = fname + '.csv'
        save = True
    else:
        save = False

    # Construct table head
    head = [" ","p","mean("+g1str+")"," ","std("+g1str+")","</>",\
            "mean("+g2str+")"," ","std("+g2str+")"]

    # Compute mean/std of input data
    g1mean = group1.mean(axis=0)
    g1std  = group1.std(axis=0)
    g2mean = group2.mean(axis=0)
    g2std  = group2.std(axis=0)

    # Put "<" if mean(base) < mean(test) and vice versa
    gtlt = np.array(['<']*g1mean.size)
    gtlt[np.where(g1mean > g2mean)] = '>'

    # Prettify table
    pmstr = ["+/-"]*g1mean.size

    # Assemble data array
    Data = np.column_stack((variables,\
                            pvals.astype('str'),\
                            g1mean.astype('str'),\
                            pmstr,\
                            g1std.astype('str'),\
                            gtlt,\
                            g2mean.astype('str'),\
                            pmstr,\
                            g2std.astype('str')))

    # Construct texttable object
    table = Texttable()
    table.set_cols_align(["l","l","r","c","l","c","r","c","l"])
    table.set_cols_valign(["c"]*9)
    table.set_cols_dtype(["t"]*9)
    table.set_cols_width([12,18,18,3,18,3,18,3,18])
    table.add_rows([head],header=True)
    table.add_rows(Data.tolist(),header=False)
    table.set_deco(Texttable.HEADER)

    # Pump out table if wanted
    if verbose:
        print("Summary of statistics:\n")
        print(table.draw() + "\n")
        print(foot + "\n")

    # If wanted, save stuff in a csv file
    if save:
        head = str(head)
        head = head.replace("[","")
        head = head.replace("]","")
        head = head.replace("'","")
        np.savetxt(fname,Data,delimiter=",",fmt="%s",header=head,footer=foot,comments="")

##########################################################################################

def perco_thresh(nws):
     """
    Use percolation approach to threshold network(s) nws.


    Parameters
    ----------
    nws : NumPy array
        The network(s) to be thresholded.

    Returns
    -------
    nws_thresh : NumPy array
        The thresholded network(s).
    sub_dens: Numpy array
        The density (densities) of the network(s).

    """
    # if only one network is given (2darray), reshape to 3darray
    if len(nws.shape)<3:
        n=1
        nws=np.reshape(nws, nws.shape + (1,))
    else:
        # determine number of networks
        n=nws.shape[2]
    # set up array to save densities
    sub_dens=np.zeros([n,1])
    # set up array to save thresholded networks
    nws_thr=np.zeros(nws.shape)

    # do for every network
    for sub in range(n):
        ### determine largest component via breadth first search
        through=False
        root=0      # start with first node (index 0)
        module=1    # assign module number 1 to that node
        # set up array for modular assignments
        mod_asg=np.zeros([nws.shape[0],1])
        # as long as there are still nodes to be traversed
        while not through:
            # find all nodes connected to root
            conn=bfs(nws[:,:,sub],root)
            # assign all of them to the same module
            mod_asg[conn.astype(int),0]=module
            # do not forget root belongs to the same module
            mod_asg[root,0]=module
            # find new root (i.e.; a node that has not been assigned to a module)
            root=np.where(mod_asg==0)

            # if there is no root left, stop while loop
            if root[0].size==0:
                through=True
            # else, take the first untraversed node as root, increase module number
            else:
                root=root[0][0]
                module=module+1
        # determine how many modules are in the network and how large they are
        val,cnt=np.unique(mod_asg,return_counts=True)
        # choose largest module
        largest=val[cnt==cnt.max()]

        #if largest contains several elements, choose one randomly
        #(if there are several largest modules, we only look at one for simplicity)
        if len(largest)>1:
            print('There were at least two equally large largest modules.')
            largest=largest[np.random.choice(largest.shape[0], 1, replace=False)]
            
        ### delete edges until largest module breaks apart
        maxedge=nws[:,:,sub].max()
        broke=False
        perc=1 # remember how many percolation steps we took
        while not broke:
            # calculate threshold as percentage of strongest edge
            thr=maxedge*perc/100
            # make shallow copy (don't change mnw when changing mnw_thr)
            nw_thr=nws[:,:,sub].copy()
            # delete connections below threshold
            nw_thr[nw_thr<thr]=0
            # check if largest component broke apart
            broke=broken(nw_thr,mod_asg,largest)
            perc=perc+1
        # go one step back (at perc, the module broke, so backtrack)
        perc=perc-2

        ### threshold network with the appropriate theshold
        nw_thr=nws[:,:,sub].copy()
        thr=maxedge*perc/100
        nw_thr[nw_thr<thr]=0
        dens=density_und(nw_thr)
        # save individual thresholded network
        nws_thr[:,:,sub]=nw_thr
        # save individual density
        sub_dens[sub]=dens

    return nws_thr.squeeze(),sub_dens
