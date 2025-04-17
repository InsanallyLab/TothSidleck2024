from .io import loadSessionInitial

import sys, os, pickle
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from scipy.stats import gaussian_kde
import ruptures as rpt

def binKDE(sessionfile,clust,bw = 0.05):
    spikeidxs = np.equal(sessionfile.spikes.clusters,clust)
    spiketimes = sessionfile.spikes.times[spikeidxs]

    numentries = int(sessionfile.meta.length_in_seconds)
    xrange = np.linspace(0,sessionfile.meta.length_in_samples,numentries)
    KDE = gaussian_kde(np.concatenate(spiketimes),bw_method=bw).evaluate(xrange)
    return xrange,KDE

def calculateRunningSTDEV(binnedData,bw=5):
    stdevs = []
    for idx in range(len(binnedData)-bw):
        if(np.mean(binnedData[idx:idx+bw])) > 0:
            stdevs.append(np.std(binnedData[idx:idx+bw]))
    return np.median(stdevs)

def calculateSegments(sessionfile,clust,scaling = 1):
    #bins = binData(datatobin)
    #sigma = calculateRunningSTDEV(bins)
    
    xrange,signal = binKDE(sessionfile,clust)
    sigma = calculateRunningSTDEV(signal,bw=60)
    
    model = "l2"  # "l1", "rbf", "linear", "normal", "ar",...
    algo = rpt.Binseg(model=model).fit(signal)
    #algo = rpt.BottomUp(model=model).fit(bins)
    #algo = rpt.Window(width=5, model=model).fit(bins)
    #my_bkps = algo.predict(pen=np.log(len(bins)) * 1 * 5**2)
    my_bkps = algo.predict(pen= np.log(len(signal)) * sigma**2 / scaling)
    #breakpoints is in seconds
    return my_bkps,xrange,signal

def getTrialOfTime(sessionfile,timeInSeconds):
    time = timeInSeconds * sessionfile.meta.fs
    for trial in range(sessionfile.meta.length_in_trials):
        if sessionfile.trials.starts[trial] < time and sessionfile.trials.ends[trial] > time:
            return trial+1
    
    for trial in range(sessionfile.meta.length_in_trials - 1):
        if sessionfile.trials.ends[trial] < time and sessionfile.trials.starts[trial+1] > time:
            return trial+1.5
    
    if time < min(sessionfile.trials.starts):
        return 1
    if time > max(sessionfile.trials.ends):
        return sessionfile.meta.length_in_trials
    
    return np.nan

def trimSessionInternal(sessionfile,FRthreshold = 0.5,scaling = 1/1000):
    trim = dict()
    for clust in sessionfile.clusters.good:
        trimmedunit = trimClusterInternal(sessionfile,clust,FRthreshold=FRthreshold,scaling=scaling)
        trim[clust] = trimmedunit
    return trim

def trimClusterInternal(sessionfile,clust,FRthreshold = 0.5,scaling = 1/1000):
    breakpoints,xrange,signal = calculateSegments(sessionfile,clust,scaling=scaling)
    breakpoints = np.array(breakpoints)
    breakpoints = np.concatenate(([0],breakpoints,[int(sessionfile.meta.length_in_seconds)]))
    breakpoints = np.sort(np.unique(breakpoints))
    
    bins = [ [breakpoints[idx],breakpoints[idx+1]] for idx in range(len(breakpoints)-1)]

    #Bins start at valid because we will skip "middle bins" during validation by cutting each end short
    validbins = np.zeros(len(bins))
    validbins.fill(True)
        
    #Caching search -- Unit
    idx = np.equal(sessionfile.spikes.clusters,clust)
    totaltimes = sessionfile.spikes.times[idx]

    #Traverse bins in order -- trim from start of recording
    for binidx,bin in enumerate(bins):
        idx = np.logical_and(      np.greater(totaltimes,bin[0]*sessionfile.meta.fs) , np.less(totaltimes,bin[1]*sessionfile.meta.fs)     )
        binFR = np.sum(idx) / (bin[1]-bin[0])
        
        if binFR < FRthreshold:
            validbins[binidx] = False
        else:
            break
            
    #Traverse bins in reverse order -- trim from end of recording
    for binidx,bin in reversed(list(enumerate(bins))):
        idx = np.logical_and(      np.greater(totaltimes,bin[0]*sessionfile.meta.fs) , np.less(totaltimes,bin[1]*sessionfile.meta.fs)     )
        binFR = np.sum(idx) / (bin[1]-bin[0])
        
        if binFR < FRthreshold:
            validbins[binidx] = False
        else:
            break

    trim = SimpleNamespace()

    #Find start of trimmed region
    trimstartseconds = np.nan
    for binidx,bin in enumerate(bins):
        if validbins[binidx]:
            trimstartseconds = bin[0]
            break
    trim.trimmed_start_in_seconds = trimstartseconds
    trim.trimmed_start_in_samples = trimstartseconds * sessionfile.meta.fs # This can be up to a second off.

    #Find end of trimmed region
    trimendseconds = np.nan
    for binidx,bin in reversed(list(enumerate(bins))):
        if validbins[binidx]:
            trimendseconds = bin[1]
            break
    if trimendseconds == int(sessionfile.meta.length_in_seconds):
        trim.trimmed_end_in_seconds = sessionfile.meta.length_in_seconds
        trim.trimmed_end_in_samples = sessionfile.meta.length_in_samples # We do this because there may be some rounding
    else:
        trim.trimmed_end_in_seconds = trimendseconds
        trim.trimmed_end_in_samples = trimendseconds * sessionfile.meta.fs # This can be up to a second off.

    #Find out which trials are valid according to the trimming
    validtrials = []
    for trial in range(sessionfile.meta.length_in_trials):
        if sessionfile.trials.starts[trial] > trim.trimmed_start_in_samples and sessionfile.trials.ends[trial] < trim.trimmed_end_in_samples:
            validtrials.append(trial)
    trim.trimmed_trials = validtrials

    #Determine valid trials in tuning curve recordings
    if sessionfile.meta.task in ['tuning nonreversal', 'tuning switch', 'tuning reversal', 'passive no behavior'] and sessionfile.tuning.number_of_tones > 0:
        validtuningtrials = []
        for trial in range(sessionfile.tuning.number_of_tones):
            if sessionfile.tuning.trial_starts[trial] > trim.trimmed_start_in_samples and (sessionfile.tuning.trial_starts[trial] + sessionfile.meta.fs * 1) < trim.trimmed_end_in_samples:
                validtuningtrials.append(trial)
        trim.trimmed_tuning_trials = validtuningtrials

    trim.breakpoints = breakpoints
    return trim

def trimSessions(directory,replace=False):
    animals = os.listdir(directory)
    for animal in tqdm(animals):

        # if not animal in ['AE_344','AE_346','AE_367','AE_350','AE_351','AE_359']:
        #     continue
        if not animal in ['AE_350','AE_351','AE_359']:
            continue
        #if not animal in ['TH_217']:
        #    continue
        #if not animal in ['BS_173','BS_175','BS_187','BS_188','BS_213','BS_214']:
        #    continue
        #if not animal in ['TH_201','TH_203','LA_204','LA_205','TH_200','TH_230','AE_229','TH_233','TH_234']:
        #    continue
        #if animal != 'BS_187':
        #    continue
        #if not animal in ['BS_213','BS_214']:
        #    continue
        #if not animal in ['AE_235','AE_236','TH_237']:
        #    continue
        #if not animal in ['AE_231']:
            #continue
        #if not animal in ['AE_235','AE_236','TH_237','TH_230','AE_229','TH_233','TH_234']:
        #    continue

        sessions = os.listdir(os.path.join(directory,animal))
        for session in sessions:
            for region in ['AC','M2','Striatum']:

                if os.path.isfile(os.path.join(directory,animal,session,region,'session_metadata.json')):
                    try:

                        #Skip sessions that have already been trimmed unless you are in the replace context
                        if os.path.isfile(os.path.join(directory,animal,session,region,'trim.pickle')) and not replace:
                            continue

                        sessionfile = loadSessionInitial(os.path.join(directory,animal),session,region)
                        trim = trimSessionInternal(sessionfile)

                        with open(os.path.join(directory,animal,session,region,'trim.pickle'), 'wb') as f:
                            pickle.dump(trim, f, protocol=pickle.HIGHEST_PROTOCOL)

                        print(os.path.join(directory,animal,session,region,'session_metadata.json') + ' complete')
                    except Exception as e:
                        print(os.path.join(directory,animal,session,region) + ': ' + str(e))
                else:
                    print(os.path.join(directory,animal,session,region,'session_metadata.json') + ' missing. Skipping')

        print(animal + ' complete')
    print(directory + ' complete')
#NOTE: does not return anything because this is used to save out cached files