import numpy as np
from types import SimpleNamespace

def disqualifyISI(sessionfile,ISIms=1,threshpercent=1,verbose=False):

    disqualifiedUnits = []
    for clust in (sessionfile.clusters.good):#tqdm here
        ISIs = np.diff(sessionfile.spikes.times[(np.equal(sessionfile.spikes.clusters,clust))])
        ISIs_ms = ISIs / sessionfile.meta.fs * 1000
        if np.mean(ISIs_ms < ISIms) >= (threshpercent/100):
            disqualifiedUnits.append(clust)

    if verbose:
        print('ISI: clusters '+str(disqualifiedUnits)+' disqualified from '+sessionfile.meta.animal + ' ' + sessionfile.meta.region + ' ' + str(sessionfile.meta.day_of_training))

    if not hasattr(sessionfile,'disqualified'):
        sessionfile.disqualified = SimpleNamespace()
    sessionfile.disqualified.ISI = disqualifiedUnits
    for clust in disqualifiedUnits:
        index = np.argwhere(sessionfile.clusters.good==clust)
        sessionfile.clusters.good = np.delete(sessionfile.clusters.good, index)

    return sessionfile

def disqualifyTrials(sessionfile,numberoftrials=100,verbose=False):

    disqualifiedUnits = []
    for clust in sessionfile.clusters.good:

        #If a unit has no trials after trimming, then get rid of it
        if len(sessionfile.trim[clust].trimmed_trials) < numberoftrials:
            disqualifiedUnits.append(clust)
            continue

    if verbose:
        print('TRIALS: clusters '+str(disqualifiedUnits)+' disqualified from '+sessionfile.meta.animal + ' ' + sessionfile.meta.region + ' ' + str(sessionfile.meta.day_of_training))

    if not hasattr(sessionfile,'disqualified'):
        sessionfile.disqualified = SimpleNamespace()
    sessionfile.disqualified.NumTrials = disqualifiedUnits
    for clust in disqualifiedUnits:
        index = np.argwhere(sessionfile.clusters.good==clust)
        sessionfile.clusters.good = np.delete(sessionfile.clusters.good, index)

    return sessionfile
        
#Disqualify units based on their firing rates
def disqualifyFR(sessionfile,FRthresh=0.5,verbose=False):
    starttime = 0
    endtime = 2.5

    disqualifiedUnits = []
    for clust in sessionfile.clusters.good:
        
        #Caching search -- Unit
        idx = np.equal(sessionfile.spikes.clusters,clust)
        totaltimes = sessionfile.spikes.times[idx]
        
        FRtrials = []
        for trial in sessionfile.trim[clust].trimmed_trials:
            trialstart = sessionfile.trials.starts[trial]
            #Caching search -- Condition
            idx = np.logical_and(      np.greater(totaltimes,trialstart+starttime*sessionfile.meta.fs) , np.less(totaltimes,trialstart+endtime*sessionfile.meta.fs)     )
            FRtrials.append( np.sum(idx) / (endtime-starttime) )
        FR = np.mean(FRtrials)

        if FR < FRthresh:
            disqualifiedUnits.append(clust)

    if verbose:
        print('FR: clusters '+str(disqualifiedUnits)+' disqualified from '+sessionfile.meta.animal + ' ' + sessionfile.meta.region + ' ' + str(sessionfile.meta.day_of_training))

    if not hasattr(sessionfile,'disqualified'):
        sessionfile.disqualified = SimpleNamespace()
    sessionfile.disqualified.FRthresh = disqualifiedUnits
    for clust in disqualifiedUnits:
        index = np.argwhere(sessionfile.clusters.good==clust)
        sessionfile.clusters.good = np.delete(sessionfile.clusters.good, index)
    
    return sessionfile