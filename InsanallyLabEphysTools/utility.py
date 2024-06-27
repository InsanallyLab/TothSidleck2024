import numpy as np
import os

from .io import loadSessionCached

def getRandomSession(directory):
    directory_list = os.listdir(directory)
    idx = np.random.randint(len(directory_list))
    return loadSessionCached(directory,directory_list[idx])

def generateDateString(sessionfile):
    namemodifier = 'ERROR'
    if sessionfile.meta.task == 'CNO':
        namemodifier = 'CNO'
    elif sessionfile.meta.task in ['nonreversal','switch','reversal','second switch','second reversal']:
        namemodifier = str(sessionfile.meta.day_of_recording)
    elif sessionfile.meta.task in ['opto nonreversal','opto switch','opto reversal']:
        namemodifier = str(sessionfile.meta.day_of_recording)
    elif sessionfile.meta.task in ['opto control nonreversal','opto control switch','opto control reversal']:
        namemodifier = str(sessionfile.meta.day_of_recording)
    elif sessionfile.meta.task in ['tuning nonreversal','tuning switch','tuning reversal']:
        namemodifier = str(sessionfile.meta.day_of_recording)
    elif sessionfile.meta.task in ['passive no behavior']:
        namemodifier = str(sessionfile.meta.day_of_recording)+'_passive'
    elif sessionfile.meta.task == 'thalamus tuning':
        namemodifier = str(sessionfile.meta.day_of_recording)

    return sessionfile.meta.animal + '_' + namemodifier + '_' + sessionfile.meta.region + '_' + str(sessionfile.meta.date).replace('/','-')

def getSpikeTimes(sessionfile,clust=np.nan,starttime=np.nan,endtime=np.nan,cachedtimes=None):
    """
    set clust to control what neuron id to search for
    set starttime and endtime (in samples) to control time span
    if not set, starttime and endtime will each default to the start
    and end of the recording
    pass cached searches into cachedtimes to speed up sequential reads
    from the same neuron's spike times

    returns spike times relative to start of recording in samples
    """

    #Since we don't actually return any spike cluster identity, we cannot cache it
    #and thus cannot split any cached data according to cluster identity. This could
    #be amended but it is not currently intended behavior to allow this operation.
    if (not cachedtimes is None) and (not np.isnan(clust)):
        print('ERROR: Cannot split cached spike times according to unit. Please amend query')
        raise Exception

    #If we've not cached a search already, we need to reset our cache to the total times
    if cachedtimes is None:
        cachedtimes = sessionfile.spikes.times

    #Prune data by cluster. Remove spikes from all others
    if not np.isnan(clust):
        clustidx = np.equal(sessionfile.spikes.clusters,clust)
        cachedtimes = cachedtimes[clustidx]

    #Remove spikes before starttime if applicable
    if not np.isnan(starttime):
        startidx = np.greater(cachedtimes,starttime)
        cachedtimes = cachedtimes[startidx]

    #Remove spikes after endtime if applicable
    if not np.isnan(endtime):
        endidx = np.less(cachedtimes,endtime)
        cachedtimes = cachedtimes[endidx]

    return cachedtimes

def getSpikeAmps(sessionfile,clust=np.nan,starttime=np.nan,endtime=np.nan,cachedamps=None):
    """
    set clust to control what neuron id to search for
    set starttime and endtime (in samples) to control time span
    if not set, starttime and endtime will each default to the start
    and end of the recording
    pass cached searches into cachedtimes to speed up sequential reads
    from the same neuron's spike times

    returns spike amplitudes
    """

    #Since we don't actually return any spike cluster identity, we cannot cache it
    #and thus cannot split any cached data according to cluster identity. This could
    #be amended but it is not currently intended behavior to allow this operation.
    if (not cachedamps is None) and (not np.isnan(clust)):
        print('ERROR: Cannot split cached spike times according to unit. Please amend query')
        raise Exception

    #If we've not cached a search already, we need to reset our cache to the total times
    if cachedamps is None:
        cachedamps = sessionfile.spikes.amplitudes

    #Prune data by cluster. Remove spikes from all others
    if not np.isnan(clust):
        clustidx = np.equal(sessionfile.spikes.clusters,clust)
        cachedamps = cachedamps[clustidx]

    #Remove spikes before starttime if applicable
    if not np.isnan(starttime):
        startidx = np.greater(cachedamps,starttime)
        cachedamps = cachedamps[startidx]

    #Remove spikes after endtime if applicable
    if not np.isnan(endtime):
        endidx = np.less(cachedamps,endtime)
        cachedamps = cachedamps[endidx]

    return cachedamps

def getTrialSpikes(sessionfile,trial,clust=np.nan,cachedtimes=None,startbuffer=0,endbuffer=0,outunits='samples'):
    """
    set trial to control time span (0-indexed)
    set clust to control what neuron id to search for
    pass cached searches from getSpikeTimes into cachedtimes
    to speed up sequential reads from the same neuron's
    spike times
    set startbuffer and endbuffer to buffer data at the start or end of a trial

    returns spike times relative to trial start in units of outunits
    """

    startbuffer *= sessionfile.meta.fs
    endbuffer *= sessionfile.meta.fs

    trialstart = sessionfile.trials.starts[trial]
    trialend = sessionfile.trials.ends[trial]

    times = np.array(getSpikeTimes(sessionfile,clust=clust,starttime=(trialstart-startbuffer),endtime=(trialend+endbuffer),cachedtimes=cachedtimes),dtype='float')
    times -= trialstart

    if outunits in ['ms','milliseconds']:
        times = times / sessionfile.meta.fs * 1000
    elif outunits in ['s','seconds']:
        times = times / sessionfile.meta.fs
    elif outunits=='samples':
        pass

    return times

def rmnan(X,positiveOnly=False):
    '''
    removes nans from data. Requires 1-dimension. Returns as ndarray.
    '''
    X = np.array(X)
    if len(X.shape) != 1:
        raise Exception(f"data is of dimension {len(X.shape)} but only 1-dimensional data is supported")
    mask = np.isfinite(X)
    if positiveOnly:
        mask = np.logical_and(mask,np.greater(X,0))
    X = X[mask]
    return X

def paired_rmnan(X,Y,positiveOnly=False):
    '''
    removes nans from paired data. Requires 1-dimension. Returns as ndarrays.
    '''
    X = np.array(X)
    Y = np.array(Y)
    if len(X.shape) != 1:
        raise Exception(f"X is of dimension {len(X.shape)} but only 1-dimensional data is supported")
    if len(Y.shape) != 1:
        raise Exception(f"Y is of dimension {len(Y.shape)} but only 1-dimensional data is supported")
    mask = np.logical_and(np.isfinite(X),np.isfinite(Y))
    if positiveOnly:
        mask = np.logical_and(mask,np.logical_and(np.greater(X,0),np.greater(Y,0)))
    X = X[mask]
    Y = Y[mask]
    return X,Y

def many_paired_rmnan(Xs,positiveOnly=False):
    '''
    removes nans from paired data. Requires 1-dimension. Returns as ndarrays.
    '''
    mask = np.full_like(Xs[0],True)
    
    for X in Xs:
        X = np.array(X)
        if len(X.shape) != 1:
            raise Exception(f"X is of dimension {len(X.shape)} but only 1-dimensional data is supported")
        mask = np.logical_and(np.isfinite(X),mask)
        if positiveOnly:
            mask = np.logical_and(mask,np.greater(X,0))

    ret = [np.array(X)[mask] for X in Xs]
    return ret