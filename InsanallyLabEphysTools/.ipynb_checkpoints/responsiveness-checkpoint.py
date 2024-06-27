from .io import loadSessionInitial,loadSessionCached
from .analysis import getAllConditions

import sys, os, pickle
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from scipy.stats import mannwhitneyu, sem

def get_response_times_with_default(sessionfile):
    response_times_in_trial = (sessionfile.trials.response-sessionfile.trials.starts)
    default_response_time = np.nanmedian(response_times_in_trial)

    response_times = sessionfile.trials.response
    nogo_trials = np.isnan(response_times)
    response_times[nogo_trials] = sessionfile.trials.starts[nogo_trials]+default_response_time
    return response_times

def calculateResponsivenessInternal(sessionfile,populationAvgLickResponseInSamples,verbose=False):
    responsiveness = dict()
    for clust in sessionfile.clusters.good:
        unitresponsiveness = calculateResponsivenessClusterInternal(sessionfile,populationAvgLickResponseInSamples,clust,verbose=verbose)
        if unitresponsiveness != None:
            responsiveness[clust] = unitresponsiveness
        else:
            print('responsiveness dict returned None')
    return responsiveness

def calculateResponsivenessClusterInternal(sessionfile,populationAvgLickResponseInSamples,clust,eLife_iterations=5000,verbose=False):#Window in ms
    responsiveness = dict()

    if not hasattr(sessionfile,'trim'):
        if verbose:
            print(sessionfile.meta.animal+' '+str(sessionfile.meta.date).replace('/','-')+' '+sessionfile.meta.region+' not trimmed')
        return None

    #Determine the no-go window to use
    avgLickResponseInSamples = populationAvgLickResponseInSamples
    if not np.all(np.isnan(sessionfile.trials.response)):
        avgLickResponseInSamples = np.nanmean(sessionfile.trials.response-sessionfile.trials.starts)

    response_times = get_response_times_with_default(sessionfile)
    
    #Set up conditions
    #Starting with just all trials because it seems like a good place to start
    ##########################################################################################################################

    all_conditions = getAllConditions(sessionfile,clust)
    for cond in all_conditions:
        responsiveness[cond] = all_conditions[cond]

    if verbose:
        for cond in responsiveness:
            print(cond)

    ##########################################################################################################################
    
    #Caching search -- Unit
    idx = np.equal(sessionfile.spikes.clusters,clust)
    totaltimes = sessionfile.spikes.times[idx]
    
    for cond in responsiveness:

        #Need to check for empty conditions because the mann-whitney test later will fail otherwise
        if len(responsiveness[cond].trials) == 0:
            responsiveness[cond].FRbaseline = np.nan
            responsiveness[cond].FR = np.nan
            responsiveness[cond].FRmodulation = np.nan
            responsiveness[cond].FRmodpval = np.nan
            responsiveness[cond].FRchoice = np.nan
            responsiveness[cond].FRmodulationchoice = np.nan
            responsiveness[cond].FRmodpvalchoice = np.nan
            continue
        
        baselinewindow = 150 #Baseline window length
        slidewindow = 50#ms #Sliding window length

        slideincrement = 10#ms              #Sliding window implemented ~July 8th
        slideend = 200#ms
        slidestart_choice = 200#ms
        numincrements = int((slideend - slidewindow)/slideincrement + 1)#Plus one because range(x) is zero indexed
        numincrements_choice = int((slidestart_choice - slidewindow)/slideincrement + 1)#Plus one because range(x) is zero indexed

        numtrials = len(responsiveness[cond].trials)
        baselineFR = np.array([np.nan] * numtrials)
        t150FR = np.array([np.nan] * numtrials)
        modulationFR = np.zeros((numtrials,numincrements))
        modulationFR.fill(np.nan)
        modulationFR_choice = np.zeros((numtrials,numincrements))
        modulationFR_choice.fill(np.nan)
        
        for trialidx,trial in enumerate(responsiveness[cond].trials):
            trialstart = sessionfile.trials.starts[trial]
            response_time = response_times[trial]

            #Caching search -- Trial
            trialcachestart = trialstart - 1*sessionfile.meta.fs
            trialcacheend = trialstart + 2.5*sessionfile.meta.fs
            cachespikeidxs = np.logical_and(      np.greater(totaltimes,trialcachestart) , np.less(totaltimes,trialcacheend)     )
            trialtimes = totaltimes#totaltimes[cachespikeidxs]

            #Caching search -- This trial -- Baseline bin 1 -- -150ms to 0ms
            starttime = trialstart - sessionfile.meta.fs * (baselinewindow/1000)
            endtime = trialstart - sessionfile.meta.fs * 0
            baselinespikeidxs = np.logical_and(      np.greater(trialtimes,starttime) , np.less(trialtimes,endtime)     )
            baselineFR[trialidx] = np.sum(baselinespikeidxs) / (baselinewindow/1000)

            #Caching search -- This trial -- 150ms tone bin -- 0ms to 150ms
            starttime = trialstart - sessionfile.meta.fs * 0
            endtime = trialstart + sessionfile.meta.fs * (baselinewindow/1000)
            t150spikeidxs = np.logical_and(      np.greater(trialtimes,starttime) , np.less(trialtimes,endtime)     )
            t150FR[trialidx] = np.sum(t150spikeidxs) / (baselinewindow/1000)
            
            for increment in range(numincrements):
                windowstart = trialstart + sessionfile.meta.fs * (increment*slideincrement)/1000
                windowend = windowstart + sessionfile.meta.fs * (slidewindow)/1000
                modulationspikeidxs = np.logical_and(      np.greater(trialtimes,windowstart) , np.less(trialtimes,windowend)     )
                modulationFR[trialidx,increment] = np.sum(modulationspikeidxs) / (slidewindow/1000)

            for increment_choice in range(numincrements_choice):
                windowend = response_time - sessionfile.meta.fs * (increment*slideincrement)/1000
                windowstart = windowend - sessionfile.meta.fs * (slidewindow)/1000
                modulationspikeidxs_choice = np.logical_and(      np.greater(trialtimes,windowstart) , np.less(trialtimes,windowend)     )
                modulationFR_choice[trialidx,increment_choice] = np.sum(modulationspikeidxs_choice) / (slidewindow/1000)

        #FRmodulation150 is an old calculation of stimulus-responsiveness.
        #We do not use it anymore as mice respond too fast for us to distinguish
        #between stimulus and choice related modulations
        responsiveness[cond].FRmodulation150 = np.abs(np.mean(t150FR - baselineFR))

        #Fill out baselineFR so as to make it the same shape as FRmodulation so
        #We can subtract it. It's mean does not change.
        baselineFR_big = np.transpose(np.tile(baselineFR,(numincrements,1)))

        

        #Calculate different stages of firing rate modulation that allow
        #us to determine firing rate modulation as well as firing rate at
        #max modulation and baseline
        FR = np.mean(modulationFR,axis=0)
        modulation = np.mean(modulationFR-baselineFR_big,axis=0)
        absmodulation = np.abs(modulation)

        #Calculate the index of maximum modulation. This is the point at which
        #we calculate the firing rate modulation
        maxmodidx = np.argmax(absmodulation)

        

        FRchoice = np.mean(modulationFR_choice,axis=0)
        modulation_choice = np.mean(modulationFR_choice-baselineFR_big,axis=0)
        absmodulation_choice = np.abs(modulation_choice)
        maxmodidx_choice = np.argmax(absmodulation_choice)





        ### Need to calculate significance between baselineFR and modulationFR at the maximum index
        trialsFR = modulationFR[:,maxmodidx]
        pval_stimulus = mannwhitneyu(baselineFR,trialsFR).pvalue

        trialsFR_choice = modulationFR_choice[:,maxmodidx_choice]
        pval_choice = mannwhitneyu(baselineFR,trialsFR_choice).pvalue





        ## Calculate Insanally et al. 2019 eLife CR/NCR Criterion for stimulus responsiveness
        eLife_means = []
        total_trials_to_sample_from = np.array(range(len(trialsFR)))
        for iteration in range(eLife_iterations):
            bootstrapped_sample = np.random.permutation(total_trials_to_sample_from)
            bootstrapped_sample = bootstrapped_sample[0:int(0.9*len(bootstrapped_sample))]
            bootstrapped_mean = np.mean( trialsFR[bootstrapped_sample] - baselineFR[bootstrapped_sample] )
            eLife_means.append(bootstrapped_mean)
        #Note that the values here are -1 and 1 instead of -0.1 and 0.1. This is because here we are
        #Using spike rates per second as opposed to the eLife paper, which used spike counts in a 0.1s
        #bin of time. These are equivalent spike rates.
        eLife_stimulus_pval = (np.sum(np.less(eLife_means,-1)) + np.sum(np.greater(eLife_means,1))) / eLife_iterations


        ## Calculate Insanally et al. 2019 eLife CR/NCR Criterion for choice responsiveness
        eLife_means_choice = []
        total_trials_to_sample_from = np.array(range(len(trialsFR_choice)))
        for iteration in range(eLife_iterations):
            bootstrapped_sample = np.random.permutation(total_trials_to_sample_from)
            bootstrapped_sample = bootstrapped_sample[0:int(0.9*len(bootstrapped_sample))]
            bootstrapped_mean = np.mean( trialsFR_choice[bootstrapped_sample] - baselineFR[bootstrapped_sample] )
            eLife_means_choice.append(bootstrapped_mean)
        #Note that the values here are -1 and 1 instead of -0.1 and 0.1. This is because here we are
        #Using spike rates per second as opposed to the eLife paper, which used spike counts in a 0.1s
        #bin of time. These are equivalent spike rates.
        eLife_choice_pval = (np.sum(np.less(eLife_means_choice,-1)) + np.sum(np.greater(eLife_means_choice,1))) / eLife_iterations





        #Save results to the dict to return up the execution stack
        responsiveness[cond].FRbaseline = np.mean(baselineFR)
        responsiveness[cond].FR = FR[maxmodidx]
        responsiveness[cond].FRmodulation = absmodulation[maxmodidx]
        responsiveness[cond].FRmodpval = pval_stimulus
        responsiveness[cond].FRmodpval_eLife = eLife_stimulus_pval

        responsiveness[cond].FRchoice = FRchoice[maxmodidx_choice]
        responsiveness[cond].FRmodulationchoice = absmodulation_choice[maxmodidx_choice]
        responsiveness[cond].FRmodpvalchoice = pval_choice
        responsiveness[cond].FRmodpvalchoice_eLife = eLife_choice_pval
            
    return responsiveness

def calculatePopulationAvgResponseInSamples(directory,sessions):
    populationResponses = []
    for session in tqdm(sessions):
        sessionfile = loadSessionCached(directory,session)      #It is not ideal that this method uses the cache, since it is supposed to run before the caching
        populationResponses = np.concatenate((populationResponses,sessionfile.trials.response-sessionfile.trials.starts))
    populationAverageLickResponse = np.nanmean(populationResponses)
    return populationAverageLickResponse

def calculateResponsiveness(directory,cachedDirectory='D:\\Analysis_Cache',replace=False,verbose=False):# Window in ms

    populationAvgLickResponseInSamples = calculatePopulationAvgResponseInSamples(cachedDirectory,os.listdir(cachedDirectory));
    if verbose:
        print('Population Average Lick Response is: '+str(populationAvgLickResponseInSamples))

    animals = os.listdir(directory)
    for animal in tqdm(animals):

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
        if not animal in ['AE_235','AE_236','TH_237','TH_230','AE_229','TH_233','TH_234']:
            continue

        sessions = os.listdir(os.path.join(directory,animal))
        for session in sessions:
            for region in ['AC','M2']:

                if os.path.isfile(os.path.join(directory,animal,session,region,'session_metadata.json')):
                    try:

                        #Skip sessions that have already been trimmed unless you are in the replace context
                        if os.path.isfile(os.path.join(directory,animal,session,region,'responsiveness.pickle')) and not replace:
                            continue

                        sessionfile = loadSessionInitial(os.path.join(directory,animal),session,region)
                        #trim = trimSessionInternal(sessionfile)
                        responsiveness = calculateResponsivenessInternal(sessionfile,populationAvgLickResponseInSamples,verbose=verbose)

                        with open(os.path.join(directory,animal,session,region,'responsiveness.pickle'), 'wb') as f:
                            pickle.dump(responsiveness, f, protocol=pickle.HIGHEST_PROTOCOL)

                        print(os.path.join(directory,animal,session,region,'session_metadata.json') + ' complete')
                    except Exception as e:
                        print(os.path.join(directory,animal,session,region) + ': ' + str(e))
                        raise(e)
                else:
                    print(os.path.join(directory,animal,session,region,'session_metadata.json') + ' missing. Skipping')

        print(animal + ' complete')
    print(directory + ' complete')
#NOTE: does not return anything because this is used to save out cached files

















def getAllTuningConditions(sessionfile,clust):
    allconditions = dict()

    #Condition A -- All Trials
    condition = SimpleNamespace()
    all_tuning_trials = np.full((sessionfile.tuning.number_of_tones), True)
    all_tuning_trials = np.array(np.where(all_tuning_trials)[0])
    all_tuning_trials = all_tuning_trials[np.isin(all_tuning_trials,sessionfile.trim[clust].trimmed_tuning_trials)]
    condition.trials = all_tuning_trials
    condition.label = 'all_tuning_trials'
    condition.color = 'grey'
    allconditions[condition.label] = condition

    tones_in_recording = np.sort(np.unique(sessionfile.tuning.trial_freqs))
    for tone in tones_in_recording:
        condition = SimpleNamespace()
        tone_trials = np.equal(sessionfile.tuning.trial_freqs, tone)
        tone_trials = np.array(np.where(tone_trials)[0])
        tone_trials = tone_trials[np.isin(tone_trials,sessionfile.trim[clust].trimmed_tuning_trials)]
        condition.trials = tone_trials
        condition.label = str(tone)
        condition.color = 'grey'
        allconditions[condition.label] = condition

    return allconditions

def calculateTuningResponsivenessInternal(sessionfile,verbose=False):
    responsiveness = dict()
    for clust in sessionfile.clusters.good:
        unitresponsiveness = calculateTuningResponsivenessClusterInternal(sessionfile,clust,verbose=verbose)
        if unitresponsiveness != None:
            responsiveness[clust] = unitresponsiveness
        else:
            if verbose:
                print('tuning responsiveness dict returned None')
    return responsiveness

def calculateTuningResponsivenessClusterInternal(sessionfile,clust,eLife_iterations=5000,verbose=False):#Window in ms
    responsiveness = dict()

    if sessionfile.meta.task not in ['tuning nonreversal','tuning switch','tuning reversal','passive no behavior'] or sessionfile.tuning.number_of_tones <= 0:
        if verbose:
            print(sessionfile.meta.animal+' '+str(sessionfile.meta.date).replace('/','-')+' '+sessionfile.meta.region+' not a tuning recording')
        return None

    if not hasattr(sessionfile,'trim'):
        if verbose:
            print(sessionfile.meta.animal+' '+str(sessionfile.meta.date).replace('/','-')+' '+sessionfile.meta.region+' not trimmed')
        return None
    
    #Set up conditions
    #Starting with just all trials because it seems like a good place to start
    ##########################################################################################################################

    all_conditions = getAllTuningConditions(sessionfile,clust)
    for cond in all_conditions:
        responsiveness[cond] = all_conditions[cond]

    if verbose:
        for cond in responsiveness:
            print(cond)

    ##########################################################################################################################
    
    #Caching search -- Unit
    idx = np.equal(sessionfile.spikes.clusters,clust)
    totaltimes = sessionfile.spikes.times[idx]
    
    for cond in responsiveness:

        #Need to check for empty conditions because the mann-whitney test later will fail otherwise
        if len(responsiveness[cond].trials) == 0:
            responsiveness[cond].FRbaseline = np.nan
            responsiveness[cond].FRmax = np.nan
            responsiveness[cond].FRmin = np.nan
            responsiveness[cond].FRevoked = np.nan
            responsiveness[cond].FRsuppressed = np.nan
            continue
        
        baselinewindow = 150 #Baseline window length
        slidewindow = 50#ms #Sliding window length

        slideincrement = 10#ms
        slideend = 200#ms
        slidestart_choice = 200#ms
        numincrements = int((slideend - slidewindow)/slideincrement + 1)#Plus one because range(x) is zero indexed

        numtrials = len(responsiveness[cond].trials)
        baselineFR = np.array([np.nan] * numtrials)
        modulationFR = np.zeros((numtrials,numincrements))
        modulationFR.fill(np.nan)
        
        for trialidx,trial in enumerate(responsiveness[cond].trials):
            trialstart = sessionfile.tuning.trial_starts[trial]

            #Caching search -- Trial
            trialcachestart = trialstart - 1*sessionfile.meta.fs
            trialcacheend = trialstart + 2.5*sessionfile.meta.fs
            cachespikeidxs = np.logical_and(      np.greater(totaltimes,trialcachestart) , np.less(totaltimes,trialcacheend)     )
            trialtimes = totaltimes#totaltimes[cachespikeidxs]

            #Caching search -- This trial -- Baseline bin 1 -- -150ms to 0ms
            starttime = trialstart - sessionfile.meta.fs * (baselinewindow/1000)
            endtime = trialstart - sessionfile.meta.fs * 0
            baselinespikeidxs = np.logical_and(      np.greater(trialtimes,starttime) , np.less(trialtimes,endtime)     )
            baselineFR[trialidx] = np.sum(baselinespikeidxs) / (baselinewindow/1000)
            
            for increment in range(numincrements):
                windowstart = trialstart + sessionfile.meta.fs * (increment*slideincrement)/1000
                windowend = windowstart + sessionfile.meta.fs * (slidewindow)/1000
                modulationspikeidxs = np.logical_and(      np.greater(trialtimes,windowstart) , np.less(trialtimes,windowend)     )
                modulationFR[trialidx,increment] = np.sum(modulationspikeidxs) / (slidewindow/1000)

        #Fill out baselineFR so as to make it the same shape as FRmodulation so
        #We can subtract it. It's mean does not change.
        baselineFR_big = np.transpose(np.tile(baselineFR,(numincrements,1)))

        #Calculate different stages of firing rate modulation that allow
        #us to determine firing rate modulation as well as firing rate at
        #max modulation and baseline
        FR = np.mean(modulationFR,axis=0)
        FRSEM = sem(modulationFR,axis=0)
        FRSTD = np.std(modulationFR,axis=0)
        FR_evoked = np.mean( (modulationFR - baselineFR_big) ,axis=0)
        FR_evokedSEM = sem( (modulationFR - baselineFR_big) ,axis=0)
        FR_evokedSTD = np.std( (modulationFR - baselineFR_big) ,axis=0)
        #modulation = np.mean(modulationFR-baselineFR_big,axis=0)
        #absmodulation = np.abs(modulation)

        #Calculate the index of maximum modulation. This is the point at which
        #we calculate the firing rate modulation
        maxevidx = np.argmax(FR)
        maxsupidx = np.argmin(FR)

        ### Calculate STD and SEM for all values

        #Save results to the dict to return up the execution stack
        responsiveness[cond].FRbaseline = np.mean(baselineFR)

        responsiveness[cond].FRmax = FR[maxevidx]
        responsiveness[cond].FRmaxSEM = FRSEM[maxevidx]
        responsiveness[cond].FRmaxSTD = FRSEM[maxevidx]

        responsiveness[cond].FRmin = FR[maxsupidx]
        responsiveness[cond].FRminSEM = FRSEM[maxsupidx]
        responsiveness[cond].FRminSTD = FRSEM[maxsupidx]

        responsiveness[cond].FRevoked = FR_evoked[maxevidx]
        responsiveness[cond].FRevokedSEM = FR_evokedSEM[maxevidx]
        responsiveness[cond].FRevokedSTD = FR_evokedSTD[maxevidx]

        responsiveness[cond].FRsuppressed = FR_evoked[maxsupidx]
        responsiveness[cond].FRsuppressedSEM = FR_evokedSEM[maxsupidx]
        responsiveness[cond].FRsuppressedSTD = FR_evokedSTD[maxsupidx]
            
    return responsiveness

def calculateTuningResponsiveness(directory,cachedDirectory='D:\\Analysis_Cache',replace=False,verbose=False):# Window in ms

    animals = os.listdir(directory)
    for animal in tqdm(animals):

        #if not animal in ['TH_217']:
        #    continue
        #if not animal in ['BS_173','BS_175','BS_187','BS_188','BS_213','BS_214','TH_201','TH_203','LA_204','LA_205','TH_200']:
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
        if not animal in ['AE_235','AE_236','TH_237','TH_230','AE_229','TH_233','TH_234']:
            continue

        sessions = os.listdir(os.path.join(directory,animal))
        for session in sessions:
            for region in ['AC','M2']:

                if os.path.isfile(os.path.join(directory,animal,session,region,'session_metadata.json')):
                    try:

                        #Skip sessions that have already been trimmed unless you are in the replace context
                        if os.path.isfile(os.path.join(directory,animal,session,region,'tuning_responsiveness.pickle')) and not replace:
                            continue

                        sessionfile = loadSessionInitial(os.path.join(directory,animal),session,region)
                        if sessionfile.meta.task not in ['tuning nonreversal','tuning switch','tuning reversal','passive no behavior']:
                            continue

                        tuning_responsiveness = calculateTuningResponsivenessInternal(sessionfile,verbose=verbose)

                        with open(os.path.join(directory,animal,session,region,'tuning_responsiveness.pickle'), 'wb') as f:
                            pickle.dump(tuning_responsiveness, f, protocol=pickle.HIGHEST_PROTOCOL)

                        print(os.path.join(directory,animal,session,region,'session_metadata.json') + ' complete')
                    except Exception as e:
                        print(os.path.join(directory,animal,session,region) + ': ' + str(e))
                        #raise(e)
                else:
                    print(os.path.join(directory,animal,session,region,'session_metadata.json') + ' missing. Skipping')

        print(animal + ' complete')
    print(directory + ' complete')





























































































































################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

                                                            #OLD (DEPRECATED) CODE

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################





def calculateTargetResponsiveness(session,clust):
    trialsInQuestion = np.where(session.trials.target)[0]
    return calculateStimulusResponsiveness(session,clust,trialsInQuestion)

def calculateNontargetResponsiveness(session,clust):
    trialsInQuestion = np.where(np.logical_not(session.trials.target))[0]
    return calculateStimulusResponsiveness(session,clust,trialsInQuestion)

def calculateResponsivenessOLD(session,clust):
    trialsInQuestion = range(session.meta.length_in_trials)
    return calculateStimulusResponsiveness(session,clust,trialsInQuestion)

def calculateStimulusResponsiveness(session,clust,trialsInQuestion):
    numbersamples = 5000
    
    #Caching search
    idx = np.equal(session.spikes.clusters,clust)
    totaltimes = session.spikes.times[idx]
    
    precounts = np.empty((len(trialsInQuestion),3))
    postcounts = np.empty((len(trialsInQuestion),2))
    
    for trialidx in range(len(trialsInQuestion)):
        trial = trialsInQuestion[trialidx]
        trialstart = session.trials.starts[trial]
        
        #Caching search
        idx = np.logical_and(      np.greater(totaltimes,trialstart-0.31*session.meta.fs) , np.less(totaltimes,trialstart+0.21*session.meta.fs)     )
        times = totaltimes[idx]
        
        prestart1 = trialstart - 0.1*session.meta.fs
        preend1 = prestart1 + 0.1*session.meta.fs
        prestart2 = trialstart - 0.2*session.meta.fs
        preend2 = prestart2 + 0.1*session.meta.fs
        prestart3 = trialstart - 0.3*session.meta.fs
        preend3 = prestart3 + 0.1*session.meta.fs
        precounts[trialidx,0] = np.sum(np.logical_and(      np.greater(times,prestart1) , np.less(times,preend1)     ))
        precounts[trialidx,1] = np.sum(np.logical_and(      np.greater(times,prestart2) , np.less(times,preend2)     ))
        precounts[trialidx,2] = np.sum(np.logical_and(      np.greater(times,prestart3) , np.less(times,preend3)     ))
        
        poststart1 = trialstart + 0.0*session.meta.fs
        postend1 = poststart1 + 0.1*session.meta.fs
        poststart2 = trialstart + 0.1*session.meta.fs
        postend2 = poststart2 + 0.1*session.meta.fs
        postcounts[trialidx,0] = np.sum(np.logical_and(      np.greater(times,poststart1) , np.less(times,postend1)     ))
        postcounts[trialidx,1] = np.sum(np.logical_and(      np.greater(times,poststart2) , np.less(times,postend2)     ))
    
    #print(np.sum(precounts))
    #print(precounts)
    
    means = []
    FRmods = []
    for num in (range(numbersamples)):
        thismean = []
        perm = np.random.permutation(range(len(trialsInQuestion)))
        perm = perm[range(int(np.floor(0.90*len(perm))))]
        for trialidx in perm:
            prebin = int(np.floor(np.random.random()*3))
            postbin = int(np.floor(np.random.random()*2))
            if precounts[trialidx,prebin] == 0:
                FRmods.append(np.nan)
            else:
                FRmods.append((postcounts[trialidx,postbin])/precounts[trialidx,prebin]) #Ratio of post/pre
            #Allow trials with pre-bins == 0???
            thismean.append(postcounts[trialidx,postbin]-precounts[trialidx,prebin])
        thismean = np.mean(thismean)
        means.append(thismean)
    #return means
    modulation = np.nanmean(FRmods)
    if np.sum(np.logical_and(  np.greater(means,-0.1) , np.less(means,0.1)   )) >= 0.95*len(means):
        return 'NCR',modulation
    else:
        if modulation < 1:
            return 'Supressed',modulation
        if modulation > 1:
            return 'Evoked',modulation
    return 'ERROR',np.nan

def calculateRamping(session,clust):
    numbersamples = 5000
    print(clust)
    
    #Caching search
    idx = np.equal(session.spikes.clusters,clust)
    totaltimes = session.spikes.times[idx]
    
    #Find all trials with responses
    gotrials = np.where(session.trials.go)[0]
    
    baseline = np.zeros((len(gotrials),3))
    bins = np.zeros((len(gotrials),17))
    
    for trialidx in range(len(gotrials)):
        trial = gotrials[trialidx]
        
        trialstart = session.trials.starts[trial]
        responsetime = session.trials.response[trial]
        
        #Caching search
        idx = np.logical_and(      np.greater(totaltimes,responsetime-0.86*session.meta.fs) , np.less(totaltimes,responsetime+0.01*session.meta.fs)     )
        times = totaltimes[idx]
        
        prestart1 = trialstart - 0.05*session.meta.fs
        preend1 = prestart1 + 0.05*session.meta.fs
        prestart2 = trialstart - 0.1*session.meta.fs
        preend2 = prestart2 + 0.05*session.meta.fs
        prestart3 = trialstart - 0.15*session.meta.fs
        preend3 = prestart3 + 0.05*session.meta.fs
        baseline[trialidx,0] = np.sum(np.logical_and(      np.greater(times,prestart1) , np.less(times,preend1)     ))
        baseline[trialidx,1] = np.sum(np.logical_and(      np.greater(times,prestart2) , np.less(times,preend2)     ))
        baseline[trialidx,2] = np.sum(np.logical_and(      np.greater(times,prestart3) , np.less(times,preend3)     ))
        
        binidxs = np.flipud(range(17))
        for idx in range(len(binidxs)):
            bin = binidxs[idx]
            binstart = responsetime - 0.05*session.meta.fs*(bin+1)
            binend = responsetime - 0.05*session.meta.fs*(bin)
            bins[trialidx,idx] = np.sum(np.logical_and(      np.greater(times,binstart) , np.less(times,binend)     ))
    
    changes = []
    #Now pre-baseline and Ramping activity is all calculated and cached
    #calculate ramping indices n stuff
    for i in range(numbersamples):
        perm = np.random.permutation(range(len(gotrials)))
        perm = perm[range(int(np.floor(0.90*len(perm))))]
        
        tempbins = np.empty(17)
        for idx in range(17):
            tempbins[idx] = np.mean(bins[perm,idx])
        
        maxmag = 0
        for idx in range(8):
            regXs = np.array(range(idx,idx+10))
            regYs = tempbins[regXs]
            
            lm = linear_model.LinearRegression()
            lm.fit(regXs.reshape(-1,1),regYs.reshape(-1,1))
            #plt.plot(regXs,lm.predict(regXs.reshape(-1, 1)))
            if (lm.coef_[0][0] > maxmag):
                maxmag = lm.coef_[0][0]
        baselinebin = int(np.floor(np.random.random()*3))
        baselineFR = np.mean(baseline[perm,baselinebin])
        #Must check for FR==0 to prevent divide by zero
        if baselineFR == 0:
            changes.append(np.nan)
        else:
            changes.append( (baselineFR+10*maxmag)/baselineFR )
    
    #Check that enough data is non-nan to have a reasonable estimate. Setting to max of 10% nans
    if np.sum(np.isnan(changes)) >= 0.1*len(changes):
        return 'Inconclusive',np.nan
    
    rampingChange = np.nanmean(changes)
    if (rampingChange >= 1.5):
        return 'Ramping',rampingChange
    else:
        return 'Non-Ramping',rampingChange
    
    
    #maxmag = 0;
    ##plt.plot(bins)
    #for idx in range(8):
    #    regXs = np.array(range(idx,idx+10))
    #    regYs = bins[regXs]
    #    
    #    lm = linear_model.LinearRegression()
    #    lm.fit(regXs.reshape(-1,1),regYs.reshape(-1,1))
    #    #plt.plot(regXs,lm.predict(regXs.reshape(-1, 1)))
    #    if (lm.coef_[0][0] > maxmag):
    #        maxmag = lm.coef_[0][0]

def calculate_all_responsiveness(session,numsamples=5000):
    
    session.clusters.resp = SimpleNamespace()
    session.clusters.resp.overall = SimpleNamespace()
    session.clusters.resp.target = SimpleNamespace()
    session.clusters.resp.nontarget = SimpleNamespace()
    
    session.clusters.resp.overall.labels = dict()
    session.clusters.resp.overall.baselines = dict()
    session.clusters.resp.overall.baselinestds = dict()
    session.clusters.resp.overall.FRmods = dict()
    session.clusters.resp.overall.FRmodstds = dict()
    session.clusters.resp.overall.FRmodnorms = dict()
    session.clusters.resp.overall.rampinglabels = dict()
    session.clusters.resp.overall.rampingmods = dict()
    session.clusters.resp.overall.rampingmodstds = dict()
    session.clusters.resp.overall.rampingmodnorms = dict()
    session.clusters.resp.overall.CR = []
    session.clusters.resp.overall.NCR = []
    session.clusters.resp.overall.RMP = []
    session.clusters.resp.overall.NRMP = []
    
    session.clusters.resp.target.labels = dict()
    session.clusters.resp.target.baselines = dict()
    session.clusters.resp.target.baselinestds = dict()
    session.clusters.resp.target.FRmods = dict()
    session.clusters.resp.target.FRmodstds = dict()
    session.clusters.resp.target.FRmodnorms = dict()
    session.clusters.resp.target.rampinglabels = dict()
    session.clusters.resp.target.rampingmods = dict()
    session.clusters.resp.target.rampingmodstds = dict()
    session.clusters.resp.target.rampingmodnorms = dict()
    session.clusters.resp.target.CR = []
    session.clusters.resp.target.NCR = []
    session.clusters.resp.target.RMP = []
    session.clusters.resp.target.NRMP = []
    
    session.clusters.resp.nontarget.labels = dict()
    session.clusters.resp.nontarget.baselines = dict()
    session.clusters.resp.nontarget.baselinestds = dict()
    session.clusters.resp.nontarget.FRmods = dict()
    session.clusters.resp.nontarget.FRmodstds = dict()
    session.clusters.resp.nontarget.FRmodnorms = dict()
    session.clusters.resp.nontarget.rampinglabels = dict()
    session.clusters.resp.nontarget.rampingmods = dict()
    session.clusters.resp.nontarget.rampingmodstds = dict()
    session.clusters.resp.nontarget.rampingmodnorms = dict()
    session.clusters.resp.nontarget.CR = []
    session.clusters.resp.nontarget.NCR = []
    session.clusters.resp.nontarget.RMP = []
    session.clusters.resp.nontarget.NRMP = []
    
    neuronslist = np.concatenate((session.clusters.good,session.clusters.disqualified))
    
    for clust in (neuronslist):
        labels,baselines,baselinestds,mods,modstds,modnorms = calculate_individual_responsiveness(session,clust,numsamples=numsamples)
        
        session.clusters.resp.overall.labels[clust] = labels[0]
        session.clusters.resp.target.labels[clust] = labels[1]
        session.clusters.resp.nontarget.labels[clust] = labels[2]
        session.clusters.resp.overall.rampinglabels[clust] = labels[3]
        session.clusters.resp.target.rampinglabels[clust] = labels[4]
        session.clusters.resp.nontarget.rampinglabels[clust] = labels[5]
        
        session.clusters.resp.overall.baselines[clust] = baselines[0]
        session.clusters.resp.target.baselines[clust] = baselines[1]
        session.clusters.resp.nontarget.baselines[clust] = baselines[2]
        
        session.clusters.resp.overall.baselinestds[clust] = baselinestds[0]
        session.clusters.resp.target.baselinestds[clust] = baselinestds[1]
        session.clusters.resp.nontarget.baselinestds[clust] = baselinestds[2]
        
        session.clusters.resp.overall.FRmods[clust] = mods[0]
        session.clusters.resp.target.FRmods[clust] = mods[1]
        session.clusters.resp.nontarget.FRmods[clust] = mods[2]
        session.clusters.resp.overall.rampingmods[clust] = mods[3]
        session.clusters.resp.target.rampingmods[clust] = mods[4]
        session.clusters.resp.nontarget.rampingmods[clust] = mods[5]
        
        session.clusters.resp.overall.FRmodstds[clust] = modstds[0]
        session.clusters.resp.target.FRmodstds[clust] = modstds[1]
        session.clusters.resp.nontarget.FRmodstds[clust] = modstds[2]
        session.clusters.resp.overall.rampingmodstds[clust] = modstds[3]
        session.clusters.resp.target.rampingmodstds[clust] = modstds[4]
        session.clusters.resp.nontarget.rampingmodstds[clust] = modstds[5]
        
        session.clusters.resp.overall.FRmodnorms[clust] = modnorms[0]
        session.clusters.resp.target.FRmodnorms[clust] = modnorms[1]
        session.clusters.resp.nontarget.FRmodnorms[clust] = modnorms[2]
        session.clusters.resp.overall.rampingmodnorms[clust] = modnorms[3]
        session.clusters.resp.target.rampingmodnorms[clust] = modnorms[4]
        session.clusters.resp.nontarget.rampingmodnorms[clust] = modnorms[5]
        
        if labels[0] == 'evoked' or labels[0] == 'suppressed':
            session.clusters.resp.overall.CR.append(clust)
        elif labels[0] == 'NCR':
            session.clusters.resp.overall.NCR.append(clust)
            
        if labels[1] == 'evoked' or labels[1] == 'suppressed':
            session.clusters.resp.target.CR.append(clust)
        elif labels[1] == 'NCR':
            session.clusters.resp.target.NCR.append(clust)
            
        if labels[2] == 'evoked' or labels[2] == 'suppressed':
            session.clusters.resp.nontarget.CR.append(clust)
        elif labels[2] == 'NCR':
            session.clusters.resp.nontarget.NCR.append(clust)
            
        if labels[3] == 'ramping':
            session.clusters.resp.overall.RMP.append(clust)
        elif labels[3] == 'nonramping':
            session.clusters.resp.overall.NRMP.append(clust)
            
        if labels[4] == 'ramping':
            session.clusters.resp.target.RMP.append(clust)
        elif labels[4] == 'nonramping':
            session.clusters.resp.target.NRMP.append(clust)
            
        if labels[5] == 'ramping':
            session.clusters.resp.nontarget.RMP.append(clust)
        elif labels[5] == 'nonramping':
            session.clusters.resp.nontarget.NRMP.append(clust)
    
    return session

def calculate_individual_responsiveness(session,clust,numsamples=1,conclusivefraction=0.1,VERBOSE=False):
    #Caching search
    idx = np.equal(session.spikes.clusters,clust)
    totaltimes = session.spikes.times[idx]
    
    numtarget = np.sum(session.trials.target)
    currenttarget = 0
    numnontarget = np.sum(np.logical_not(session.trials.target))
    currentnontarget = 0
    numgo = np.sum(session.trials.go)
    currentgo = 0
    numtargetgo = np.sum(np.logical_and(session.trials.target,session.trials.go))
    currenttargetgo = 0
    numnontargetgo = np.sum(np.logical_and(np.logical_not(session.trials.target),session.trials.go))
    currentnontargetgo = 0
    
    if (VERBOSE):
        print('Total trials: '+str(session.meta.length_in_trials))
        print('Target trials: '+str(numtarget))
        print('Nontarget trials: '+str(numnontarget))
        print('Go trials: '+str(numgo))
        print('Targetgo trials: '+str(numtargetgo))
        print('Nontargetgo trials: '+str(numnontargetgo))
    
    precounts = np.empty((session.meta.length_in_trials,3))
    postcounts = np.empty((session.meta.length_in_trials,2))
    precountstarget = np.empty((numtarget,3))
    postcountstarget = np.empty((numtarget,2))
    precountsnontarget = np.empty((numnontarget,3))
    postcountsnontarget = np.empty((numnontarget,2))
    precountsgo = np.empty((numgo,3))
    postcountsgo = np.empty((numgo,17))
    precountstargetgo = np.empty((numtargetgo,3))
    postcountstargetgo = np.empty((numtargetgo,17))
    precountsnontargetgo = np.empty((numnontargetgo,3))
    postcountsnontargetgo = np.empty((numnontargetgo,17))
    
    #Populate arrays
    for trialidx in range(session.meta.length_in_trials):
        trial = trialidx #vestigial
        
        trialstart = session.trials.starts[trial]
        #Response is not guaranteed yet. Extract inside if guard later on
        
        #Caching search
        idx = np.logical_and(      np.greater(totaltimes,trialstart-0.31*session.meta.fs) , np.less(totaltimes,trialstart+0.21*session.meta.fs)     )
        times = totaltimes[idx]
        
        #100ms bins for prestimulus
        prestart1 = trialstart - 0.1*session.meta.fs
        preend1 = prestart1 + 0.1*session.meta.fs
        prestart2 = trialstart - 0.2*session.meta.fs
        preend2 = prestart2 + 0.1*session.meta.fs
        prestart3 = trialstart - 0.3*session.meta.fs
        preend3 = prestart3 + 0.1*session.meta.fs
        precounts[trialidx,0] = np.sum(np.logical_and(      np.greater(times,prestart1) , np.less(times,preend1)     ))
        precounts[trialidx,1] = np.sum(np.logical_and(      np.greater(times,prestart2) , np.less(times,preend2)     ))
        precounts[trialidx,2] = np.sum(np.logical_and(      np.greater(times,prestart3) , np.less(times,preend3)     ))
        if session.trials.target[trialidx]:
            precountstarget[currenttarget,range(3)] = precounts[trialidx,range(3)]
            #Increment after postcount
        else:
            precountsnontarget[currentnontarget,range(3)] = precounts[trialidx,range(3)]
            #Increment after postcount
        
        #Stimulus onset and offset for evoked/supressed analysis
        poststart1 = trialstart + 0.0*session.meta.fs
        postend1 = poststart1 + 0.1*session.meta.fs
        poststart2 = trialstart + 0.1*session.meta.fs
        postend2 = poststart2 + 0.1*session.meta.fs
        postcounts[trialidx,0] = np.sum(np.logical_and(      np.greater(times,poststart1) , np.less(times,postend1)     ))
        postcounts[trialidx,1] = np.sum(np.logical_and(      np.greater(times,poststart2) , np.less(times,postend2)     ))
        if session.trials.target[trialidx]:
            postcountstarget[currenttarget,range(2)] = postcounts[trialidx,range(2)]
            currenttarget+=1
        else:
            postcountsnontarget[currentnontarget,range(2)] = postcounts[trialidx,range(2)]
            currentnontarget+=1
            
        #Now handle the response trials for the ramping analysis
        if session.trials.go[trialidx]:
            responsetime = session.trials.response[trial]
            
            #Must do pre-bins first while the trial start period is still cached
            prestart1 = trialstart - 0.05*session.meta.fs
            preend1 = prestart1 + 0.05*session.meta.fs
            prestart2 = trialstart - 0.1*session.meta.fs
            preend2 = prestart2 + 0.05*session.meta.fs
            prestart3 = trialstart - 0.15*session.meta.fs
            preend3 = prestart3 + 0.05*session.meta.fs
            precountsgo[currentgo,0] = np.sum(np.logical_and(      np.greater(times,prestart1) , np.less(times,preend1)     ))
            precountsgo[currentgo,1] = np.sum(np.logical_and(      np.greater(times,prestart2) , np.less(times,preend2)     ))
            precountsgo[currentgo,2] = np.sum(np.logical_and(      np.greater(times,prestart3) , np.less(times,preend3)     ))
            if session.trials.target[trialidx]:
                precountstargetgo[currenttargetgo,range(3)] = precountsgo[currentgo,range(3)]
                #Increment after ramping calculations
            else:
                precountsnontargetgo[currentnontargetgo,range(3)] = precountsgo[currentgo,range(3)]
                #Increment after ramping calculations
            #Increment after ramping calculations
            
            #Now cache response period for calculating ramping
            idx = np.logical_and(      np.greater(totaltimes,responsetime-0.86*session.meta.fs) , np.less(totaltimes,responsetime+0.01*session.meta.fs)     )
            times = totaltimes[idx]
            
            binidxs = np.flipud(range(17))
            for idx in range(len(binidxs)):
                thisbin = binidxs[idx]
                binstart = responsetime - 0.05*session.meta.fs*(thisbin+1)
                binend = responsetime - 0.05*session.meta.fs*(thisbin)
                postcountsgo[currentgo,idx] = np.sum(np.logical_and(      np.greater(times,binstart) , np.less(times,binend)     ))
            if session.trials.target[trialidx]:
                postcountstargetgo[currenttargetgo,range(17)] = postcountsgo[currentgo,range(17)]
                currenttargetgo+=1
            else:
                postcountsnontargetgo[currentnontargetgo,range(17)] = postcountsgo[currentgo,range(17)]
                currentnontargetgo+=1
            #Incremented after target/nontarget to preserve indices for transfer
            currentgo+=1
            
    #Now all baseline and stimulus/ramping period counts are cached and ready to calculate
    if (VERBOSE):
        print('sum precounts: '+str(np.sum(precounts)))
        print('sum precountstarget: '+str(np.sum(precountstarget)))
        print('sum precountsnontarget: '+str(np.sum(precountsnontarget)))
        print('sum precountsgo: '+str(np.sum(precountsgo)))
        print('sum precountstargetgo: '+str(np.sum(precountstargetgo)))
        print('sum precountsnontargetgo: '+str(np.sum(precountsnontargetgo)))
        print('sum postcounts: '+str(np.sum(postcounts)))
        print('sum postcountstarget: '+str(np.sum(postcountstarget)))
        print('sum postcountsnontarget: '+str(np.sum(postcountsnontarget)))
        print('sum postcountsgo: '+str(np.sum(postcountsgo)))
        print('sum postcountstargetgo: '+str(np.sum(postcountstargetgo)))
        print('sum postcountsnontargetgo: '+str(np.sum(postcountsnontargetgo)))
                
    def calculate_responsiveness_internal(precounts,postcounts,numsamples):
        p_vals = []
        baselineFR = []
        baselinestds = []
        means = []
        stds = []
        meansnorm = []
        
        numbertrials = precounts.shape[0]
        for sample in range(numsamples):
            perm = np.random.permutation(range(numbertrials))
            perm = perm[range(int(np.floor(0.90*len(perm))))]
            pre = np.empty(len(perm))
            post = np.empty(len(perm))
            for idx in range(len(perm)):
                trialidx = perm[idx]
                prebin = int(np.floor(np.random.random()*3))
                postbin = int(np.floor(np.random.random()*2))
                pre[idx] = precounts[trialidx,prebin]
                post[idx] = postcounts[trialidx,postbin]

            baseline = np.mean(pre)
            if baseline == 0:
                p_vals.append(np.nan)
                baselineFR.append(np.nan)
                baselineFR.append(np.nan)
                means.append(np.nan)
                stds.append(np.nan)
                meansnorm.append(np.nan)
            else:                                                                #Currently Trial-averaged changes except stds
                p_vals.append( np.abs(np.mean(post)-np.mean(pre)) /2/baseline) #Trial-averaged
                baselineFR.append(baseline)
                baselinestds.append(np.std(pre))
                means.append((np.mean(post)-np.mean(pre)))
                stds.append(np.std(post-pre))
                meansnorm.append((np.mean(post)-np.mean(pre))  /baseline) #Trial-averaged

        p_val = np.nanmean(p_vals)
        #print('target p_val: '+str(p_val))
        #Check that enough data is non-nan to have a reasonable estimate. Setting to max of 10% nans
        if np.sum(np.isnan(p_vals)) >= conclusivefraction*len(p_vals):
            label = 'inconclusive'
            baseline = np.nan
            baselinestdev = np.nan
            mod = np.nan
            modstdev = np.nan
            modnorm = np.nan
        else:
            baseline = np.nanmean(baselineFR)
            baselinestdev = np.nanmean(baselinestds)
            mod = np.nanmean(means)
            modstdev = np.nanmean(stds)
            modnorm = np.nanmean(meansnorm)
            if(p_val <= 0.05):
                label = 'NCR'
            else:
                if mod > 0:
                    label = 'evoked'
                else:
                    label = 'suppressed'
        return label,baseline,baselinestdev,mod,modstdev,modnorm
    #END OF INTRNAL FUNCTION
    
    def calculate_ramping_internal(precounts,postcounts,numsamples):
        changes = []
        changesstd = []
        changesnorm = []
        
        numbertrials = precounts.shape[0]
        for i in range(numsamples):
            perm = np.random.permutation(range(numbertrials))
            perm = perm[range(int(np.floor(0.90*len(perm))))]

            tempbins = np.empty(17)
            for idx in range(17):
                tempbins[idx] = np.mean(postcounts[perm,idx])

            maxmag = 0
            for idx in range(8):
                regXs = np.array(range(idx,idx+10))
                regYs = tempbins[regXs]

                lm = linear_model.LinearRegression()
                lm.fit(regXs.reshape(-1,1),regYs.reshape(-1,1))
                #plt.plot(regXs,lm.predict(regXs.reshape(-1, 1)))
                if (lm.coef_[0][0] > maxmag):
                    maxmag = lm.coef_[0][0]
            baselinebin = int(np.floor(np.random.random()*3))
            baselineFR = np.mean(precounts[perm,baselinebin])
            #Must check for FR==0 to prevent divide by zero
            if baselineFR == 0:
                changes.append(np.nan)
                changesnorm.append(np.nan)
            else:
                changes.append( (baselineFR+10*maxmag))
                changesnorm.append( (baselineFR+10*maxmag)/baselineFR )

        #Check that enough data is non-nan to have a reasonable estimate. Setting to max of 10% nans
        if np.sum(np.isnan(changes)) >= conclusivefraction*len(changes):
            label = 'inconclusive'
            mod = np.nan
            modstd = np.nan
            modnorm = np.nan
        else:
            mod = np.nanmean(changes)
            modstd = np.nanstd(changes)
            modnorm = np.nanmean(changesnorm)
            if (modnorm >= 1.5):
                label = 'ramping'
            else:
                label = 'nonramping'
        return label,mod,modstd,modnorm
    #END OF INTRNAL FUNCTION
    
    
    overallresults = calculate_responsiveness_internal(precounts,postcounts,numsamples)
    targetresults = calculate_responsiveness_internal(precountstarget,postcountstarget,numsamples)
    nontargetresults = calculate_responsiveness_internal(precountsnontarget,postcountsnontarget,numsamples)
    
    overallramping = calculate_ramping_internal(precountsgo,postcountsgo,numsamples)
    targetramping = calculate_ramping_internal(precountstargetgo,postcountstargetgo,numsamples)
    nontargetramping = calculate_ramping_internal(precountsnontargetgo,postcountsnontargetgo,numsamples)
                
    if(VERBOSE):
        print(overallresults)
        print(targetresults)
        print(nontargetresults)
        print(overallramping)
        print(targetramping)
        print(nontargetramping)
    
    labels = [overallresults[0],targetresults[0],nontargetresults[0],overallramping[0],targetramping[0],nontargetramping[0]]
    baselines = [overallresults[1],targetresults[1],nontargetresults[1]]
    baselinestds = [overallresults[2],targetresults[2],nontargetresults[2]]
    mods = [overallresults[3],targetresults[3],nontargetresults[3],overallramping[1],targetramping[1],nontargetramping[1]]
    modstds = [overallresults[4],targetresults[4],nontargetresults[4],overallramping[2],targetramping[2],nontargetramping[2]]
    modnorms = [overallresults[5],targetresults[5],nontargetresults[5],overallramping[3],targetramping[3],nontargetramping[3]]
    
    #TODO: What to return???
    return labels,baselines,baselinestds,mods,modstds,modnorms