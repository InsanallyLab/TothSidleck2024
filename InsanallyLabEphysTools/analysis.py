import numpy as np
import os
import pickle
import random
from types import SimpleNamespace
from scipy.stats import gaussian_kde
from itertools import product
from .utility import getSpikeTimes,getTrialSpikes,generateDateString
from .io import loadSessionCached

#The condition structure exists to make passing subsets of recordings between analyses
#easier.
#Every condition contains the following
#cond.trials: zero indexed trials included
#cond.label: condition label
#cond.color: color for plotting

def calculate_latency(sessionfile,clust,cond='all_trials',startTime=0,endTime=300):
    all_conditions = getAllConditions(sessionfile,clust)
    trials = all_conditions[cond].trials
    
    trialSpikes = []
    for trial in trials:
        starttime = sessionfile.trials.starts[trial]+(startTime-250)/1000*sessionfile.meta.fs #0.25
        endtime = sessionfile.trials.starts[trial]+(endTime+250)/1000*sessionfile.meta.fs #1.5
        this_trial_spikes = getSpikeTimes(sessionfile,clust=clust,starttime=starttime,endtime=endtime)
        this_trial_spikes = this_trial_spikes - sessionfile.trials.starts[trial]
        this_trial_spikes = this_trial_spikes * 1000 / sessionfile.meta.fs
        trialSpikes.append(this_trial_spikes)
    trialSpikes = np.array(trialSpikes,dtype='object')
    
    #Plot PSTH
    xrange = np.linspace(startTime,endTime,num=(endTime-startTime))
    PSTHspikes = np.concatenate(trialSpikes)
    assert len(PSTHspikes)>0
    bw = len(PSTHspikes)**(-1./5) / 3#(bw_scott_factor) Same as rasters
    KDE = gaussian_kde(PSTHspikes,bw_method=bw)
    FR = KDE.evaluate(xrange)
    FR = FR * len(PSTHspikes) / len(trials) * 1000 #1000 is conversion to s/s because units are all in ms for this plot
    diff_from_baseline = np.abs(FR - sessionfile.responsiveness[clust][cond].FRbaseline)
    
    peak = np.max(diff_from_baseline) #len=1000
    peak_idx = np.argmax(diff_from_baseline) #ms
    peak_time = xrange[peak_idx]
    return peak_time

def getTrialSet(sessionfile,clust,listOfTrialSets,trialsPerDayLoaded=None):
    if clust is None:
        trimmed_trials_active = np.array(range(sessionfile.meta.length_in_trials))
    elif type(clust) == int or type(clust) == float or type(clust) == np.int64:
        trimmed_trials_active = np.array(sessionfile.trim[clust].trimmed_trials)
    elif type(clust) == list or type(clust) == np.ndarray or type(clust) == tuple:
        trimmed_trials_active = np.array(range(sessionfile.meta.length_in_trials))
        for cluster_ID in clust:
            trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,sessionfile.trim[cluster_ID].trimmed_trials)]
    if trialsPerDayLoaded is None:
        try:
            #with open('/bgfs/balbanna/jmt195/trialsToUsePerDay', 'rb') as f:
            with open('C:\\Users\\insan\\Desktop\\trialsToUsePerDay', 'rb') as f:
            # with open('Z:\\JMT Data Cache\\trialsToUsePerDay','rb') as f:
                trialsPerDayLoaded = pickle.load(f)
            active_trials = trialsPerDayLoaded[sessionfile.meta.animal][sessionfile.meta.day_of_training]
            trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,active_trials)]
        except Exception as e:
            print(f"session {generateDateString(sessionfile)} clust {clust} tried to load saved active trials but failed. Error: {e}")
    elif trialsPerDayLoaded == 'NO_TRIM':
        pass
    else:
        active_trials = trialsPerDayLoaded[sessionfile.meta.animal][sessionfile.meta.day_of_training]
        trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,active_trials)]

    trials = np.copy(trimmed_trials_active)
    label = ''
    for trialSet in listOfTrialSets:
        if trialSet == 'NONE':
            continue
        if trialSet == 'target':
            trialsTrim = np.array(np.where(sessionfile.trials.target)[0])
        elif trialSet == 'nontarget':
            trialsTrim = np.array(np.where(np.logical_not(sessionfile.trials.target))[0])
        elif trialSet == 'go':
            trialsTrim = np.array(np.where(sessionfile.trials.go)[0])
        elif trialSet == 'nogo':
            trialsTrim = np.array(np.where(np.logical_not(sessionfile.trials.go))[0])
        elif trialSet == 'hit':
            trialsTrim = np.array(np.where(np.logical_and(sessionfile.trials.target,sessionfile.trials.go))[0])
        elif trialSet == 'miss':
            trialsTrim = np.array(np.where(np.logical_and(sessionfile.trials.target,np.logical_not(sessionfile.trials.go)))[0])
        elif trialSet == 'falarm':
            trialsTrim = np.array(np.where(np.logical_and(np.logical_not(sessionfile.trials.target),sessionfile.trials.go))[0])
        elif trialSet == 'creject':
            trialsTrim = np.array(np.where(np.logical_and(np.logical_not(sessionfile.trials.target),np.logical_not(sessionfile.trials.go)))[0])
        elif trialSet == 'slow_go':
            response_times = np.array(sessionfile.trials.response) - np.array(sessionfile.trials.starts)
            response_times_go = response_times[sessionfile.trials.go]
            mean_response_time = np.mean(response_times_go)
            slow_go_threshold = max(2*mean_response_time,0.5*sessionfile.meta.fs)
            slow_go_response = np.where(np.greater(response_times,slow_go_threshold))[0]
            trialsTrim = slow_go_response
        elif trialSet == 'fast_go':
            response_times = np.array(sessionfile.trials.response) - np.array(sessionfile.trials.starts)
            response_times_go = response_times[sessionfile.trials.go]
            mean_response_time = np.mean(response_times_go)
            fast_go_threshold = max(2*mean_response_time,0.5*sessionfile.meta.fs)
            fast_go_response = np.where(np.less(response_times,fast_go_threshold))[0]
            fast_go_response = fast_go_response[np.isin(fast_go_response,trimmed_trials_active)]
            trialsTrim = fast_go_response
        elif trialSet == 'correct':
            trialsHit = np.array(np.where(np.logical_and(sessionfile.trials.target,sessionfile.trials.go))[0])
            trialsCrej = np.array(np.where(np.logical_and(np.logical_not(sessionfile.trials.target),np.logical_not(sessionfile.trials.go)))[0])
            trialsTrim = np.unique(np.concatenate((trialsHit,trialsCrej)))
        elif trialSet == 'incorrect':
            trialsMiss = np.array(np.where(np.logical_and(sessionfile.trials.target,np.logical_not(sessionfile.trials.go)))[0])
            trialsFal = np.array(np.where(np.logical_and(np.logical_not(sessionfile.trials.target),sessionfile.trials.go))[0])
            trialsTrim = np.unique(np.concatenate((trialsMiss,trialsFal)))
        elif trialSet == 'laser_on':
            trialsTrim = np.array(np.where(sessionfile.trials.laser_stimulation)[0])
        elif trialSet == 'laser_off':
            trialsTrim = np.array(np.where(np.logical_not(sessionfile.trials.laser_stimulation))[0])
        elif trialSet == 'even':
            trialsTrim = np.array(np.where(np.array(np.array(range(sessionfile.meta.length_in_trials))%2,dtype='bool'))[0])
        elif trialSet == 'odd':
            trialsTrim = np.array(np.where(np.logical_not(np.array(np.array(range(sessionfile.meta.length_in_trials))%2,dtype='bool')))[0])
        elif trialSet == 'pre_switch':
            trialsTrim = np.array(range(sessionfile.meta.first_reversal_trial))
        elif trialSet == 'post_switch':
            trialsTrim = np.array(range(sessionfile.meta.first_reversal_trial,sessionfile.meta.length_in_trials))
        else:
            raise Exception('Unrecognized Trial Set')
        trials = trials[np.isin(trials,trialsTrim)]

        if label == '':
            label = trialSet
        else:
            label = label + '_' + trialSet
    if label == '':
        label = 'all_trials'

    condition = SimpleNamespace()
    condition.trials = trials
    condition.label = label
    return condition

def getAllConditions(sessionfile,clust,trialsPerDayLoaded=None):
    if sessionfile.meta.task == 'passive no behavior':
        trialsPerDayLoaded = 'NO_TRIM'

    #Some conditions currently disabled for the purposes of decoding
    trialOutcomes = ['NONE','target','nontarget','go','nogo','hit','miss','falarm','creject','slow_go','fast_go','correct','incorrect']
    laserConditions = ['NONE']
    switchConditions = ['NONE']

    if sessionfile.meta.task == 'passive no behavior':
        trialOutcomes = ['NONE','target','nontarget']

    #if sessionfile.meta.task in ['opto nonreversal','opto switch','opto reversal']:
    if hasattr(sessionfile.trials,'laser_stimulation'):
        laserConditions = ['NONE','laser_on','laser_off','even','odd']
    else:
        laserConditions = ['NONE','even','odd']
    if sessionfile.meta.task in ['switch','opto switch','tuning switch','second switch']:
        switchConditions = ['NONE','pre_switch','post_switch']
    
    conditions = product(switchConditions,laserConditions,trialOutcomes)

    allconditions = dict()
    for cond in conditions:
        condition = getTrialSet(sessionfile,clust,cond,trialsPerDayLoaded=trialsPerDayLoaded)
        allconditions[condition.label] = condition

    return allconditions

# def getAllConditions(sessionfile,clust):
#     allconditions = dict()

#     trimmed_trials_active = np.array(sessionfile.trim[clust].trimmed_trials)
#     try:
#         with open('C:\\Users\\insan\\Desktop\\trialsToUsePerDay', 'rb') as f:
#             trialsPerDayLoaded = pickle.load(f)
#         active_trials = trialsPerDayLoaded[sessionfile.meta.animal][sessionfile.meta.day_of_training]
#         print(f"active trials are {active_trials}")
#         trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,active_trials)]
#     except Exception as e:
#         #raise e
#         pass

#     #Condition A -- All Trials
#     condition = SimpleNamespace()
#     all_trials = np.full((sessionfile.meta.length_in_trials), True)
#     all_trials = np.array(np.where(all_trials)[0])
#     all_trials = all_trials[np.isin(all_trials,trimmed_trials_active)]
#     condition.trials = all_trials
#     condition.label = 'all_trials'
#     condition.color = 'grey'
#     allconditions[condition.label] = condition

#     #Target
#     condition = SimpleNamespace()
#     target_tone = sessionfile.trials.target
#     target_tone = np.array(np.where(target_tone)[0])
#     target_tone = target_tone[np.isin(target_tone,trimmed_trials_active)]
#     condition.trials = target_tone
#     condition.label = 'target_tone'
#     condition.color = 'green'
#     allconditions[condition.label] = condition

#     #Nontarget
#     condition = SimpleNamespace()
#     nontarget_tone = np.logical_not(sessionfile.trials.target)
#     nontarget_tone = np.array(np.where(nontarget_tone)[0])
#     nontarget_tone = nontarget_tone[np.isin(nontarget_tone,trimmed_trials_active)]
#     condition.trials = nontarget_tone
#     condition.label = 'nontarget_tone'
#     condition.color = 'purple'
#     allconditions[condition.label] = condition

#     #Go
#     condition = SimpleNamespace()
#     go_response = sessionfile.trials.go
#     go_response = np.array(np.where(go_response)[0])
#     go_response = go_response[np.isin(go_response,trimmed_trials_active)]
#     condition.trials = go_response
#     condition.label = 'go_response'
#     condition.color = 'green'
#     allconditions[condition.label] = condition

#     #No-go
#     condition = SimpleNamespace()
#     nogo_response = np.logical_not(sessionfile.trials.go)
#     nogo_response = np.array(np.where(nogo_response)[0])
#     nogo_response = nogo_response[np.isin(nogo_response,trimmed_trials_active)]
#     condition.trials = nogo_response
#     condition.label = 'nogo_response'
#     condition.color = 'purple'
#     allconditions[condition.label] = condition

#     #Hit
#     condition = SimpleNamespace()
#     hit_trials = np.logical_and(sessionfile.trials.target,sessionfile.trials.go)
#     hit_trials = np.array(np.where(hit_trials)[0])
#     hit_trials = hit_trials[np.isin(hit_trials,trimmed_trials_active)]
#     condition.trials = hit_trials
#     condition.label = 'hit_trials'
#     condition.color = 'green'
#     allconditions[condition.label] = condition

#     #Miss
#     condition = SimpleNamespace()
#     miss_trials = np.logical_and(sessionfile.trials.target,np.logical_not(sessionfile.trials.go))
#     miss_trials = np.array(np.where(miss_trials)[0])
#     miss_trials = miss_trials[np.isin(miss_trials,trimmed_trials_active)]
#     condition.trials = miss_trials
#     condition.label = 'miss_trials'
#     condition.color = 'red'
#     allconditions[condition.label] = condition

#     #F. Alarm
#     condition = SimpleNamespace()
#     falarm_trials = np.logical_and(np.logical_not(sessionfile.trials.target),sessionfile.trials.go)
#     falarm_trials = np.array(np.where(falarm_trials)[0])
#     falarm_trials = falarm_trials[np.isin(falarm_trials,trimmed_trials_active)]
#     condition.trials = falarm_trials
#     condition.label = 'falarm_trials'
#     condition.color = 'red'
#     allconditions[condition.label] = condition

#     #C. Reject
#     condition = SimpleNamespace()
#     creject_trials = np.logical_and(np.logical_not(sessionfile.trials.target),np.logical_not(sessionfile.trials.go))
#     creject_trials = np.array(np.where(creject_trials)[0])
#     creject_trials = creject_trials[np.isin(creject_trials,trimmed_trials_active)]
#     condition.trials = creject_trials
#     condition.label = 'creject_trials'
#     condition.color = 'green'
#     allconditions[condition.label] = condition

#     #Slow-go
#     condition = SimpleNamespace()
#     response_times = np.array(sessionfile.trials.response) - np.array(sessionfile.trials.starts)
#     response_times_go = response_times[sessionfile.trials.go]
#     mean_response_time = np.mean(response_times_go)
#     slow_go_threshold = max(2*mean_response_time,0.5*sessionfile.meta.fs)
#     slow_go_response = np.where(np.greater(response_times,slow_go_threshold))[0]
#     slow_go_response = slow_go_response[np.isin(slow_go_response,trimmed_trials_active)]
#     condition.trials = slow_go_response
#     condition.label = 'slow_go_response'
#     condition.color = 'yellow'
#     allconditions[condition.label] = condition

#     #Fast-go
#     condition = SimpleNamespace()
#     response_times = np.array(sessionfile.trials.response) - np.array(sessionfile.trials.starts)
#     response_times_go = response_times[sessionfile.trials.go]
#     mean_response_time = np.mean(response_times_go)
#     fast_go_threshold = max(2*mean_response_time,0.5*sessionfile.meta.fs)
#     fast_go_response = np.where(np.less(response_times,fast_go_threshold))[0]
#     fast_go_response = fast_go_response[np.isin(fast_go_response,trimmed_trials_active)]
#     condition.trials = fast_go_response
#     condition.label = 'fast_go_response'
#     condition.color = 'green'
#     allconditions[condition.label] = condition

#     #Correct
#     condition = SimpleNamespace()
#     correct_trials = np.concatenate([hit_trials,creject_trials])
#     condition.trials = correct_trials
#     condition.label = 'correct_trials'
#     condition.color = 'green'
#     allconditions[condition.label] = condition

#     #Incorrect
#     condition = SimpleNamespace()
#     incorrect_trials = np.concatenate([miss_trials,falarm_trials])
#     condition.trials = incorrect_trials
#     condition.label = 'incorrect_trials'
#     condition.color = 'red'
#     allconditions[condition.label] = condition

#     #Handle optogenetic stimulation conditions
#     if sessionfile.meta.task in ['opto nonreversal','opto reversal','opto switch']:
#         #laser_on
#         condition = SimpleNamespace()
#         laser_on = sessionfile.trials.laser_stimulation
#         laser_on = np.array(np.where(laser_on)[0])
#         laser_on = laser_on[np.isin(laser_on,trimmed_trials_active)]
#         condition.trials = laser_on
#         condition.label = 'laser_on'
#         condition.color = 'blue'
#         allconditions[condition.label] = condition

#         #laser_off
#         condition = SimpleNamespace()
#         laser_off = np.logical_not(sessionfile.trials.laser_stimulation)
#         laser_off = np.array(np.where(laser_off)[0])
#         laser_off = laser_off[np.isin(laser_off,trimmed_trials_active)]
#         condition.trials = laser_off
#         condition.label = 'laser_off'
#         condition.color = 'grey'
#         allconditions[condition.label] = condition

#         #laser_on_target
#         condition = SimpleNamespace()
#         laser_on_target = np.copy(laser_on[np.isin(laser_on,target_tone)])
#         condition.trials = laser_on_target
#         condition.label = 'laser_on_target'
#         condition.color = 'blue'
#         allconditions[condition.label] = condition

#         #laser_off_target
#         condition = SimpleNamespace()
#         laser_off_target = np.copy(laser_off[np.isin(laser_off,target_tone)])
#         condition.trials = laser_off_target
#         condition.label = 'laser_off_target'
#         condition.color = 'grey'
#         allconditions[condition.label] = condition

#         #laser_on_nontarget
#         condition = SimpleNamespace()
#         laser_on_nontarget = np.copy(laser_on[np.isin(laser_on,nontarget_tone)])
#         condition.trials = laser_on_nontarget
#         condition.label = 'laser_on_nontarget'
#         condition.color = 'blue'
#         allconditions[condition.label] = condition

#         #laser_off_nontarget
#         condition = SimpleNamespace()
#         laser_off_nontarget = np.copy(laser_off[np.isin(laser_off,nontarget_tone)])
#         condition.trials = laser_off_nontarget
#         condition.label = 'laser_off_nontarget'
#         condition.color = 'grey'
#         allconditions[condition.label] = condition

#         #laser_on_hit
#         condition = SimpleNamespace()
#         laser_on_hit = np.copy(laser_on[np.isin(laser_on,hit_trials)])
#         condition.trials = laser_on_hit
#         condition.label = 'laser_on_hit'
#         condition.color = 'green'
#         allconditions[condition.label] = condition

#         #laser_off_hit
#         condition = SimpleNamespace()
#         laser_off_hit = np.copy(laser_off[np.isin(laser_off,hit_trials)])
#         condition.trials = laser_off_hit
#         condition.label = 'laser_off_hit'
#         condition.color = 'green'
#         allconditions[condition.label] = condition

#         #laser_on_miss
#         condition = SimpleNamespace()
#         laser_on_miss = np.copy(laser_on[np.isin(laser_on,miss_trials)])
#         condition.trials = laser_on_miss
#         condition.label = 'laser_on_miss'
#         condition.color = 'red'
#         allconditions[condition.label] = condition

#         #laser_off_miss
#         condition = SimpleNamespace()
#         laser_off_miss = np.copy(laser_off[np.isin(laser_off,miss_trials)])
#         condition.trials = laser_off_miss
#         condition.label = 'laser_off_miss'
#         condition.color = 'red'
#         allconditions[condition.label] = condition

#         #laser_on_falarm
#         condition = SimpleNamespace()
#         laser_on_falarm = np.copy(laser_on[np.isin(laser_on,falarm_trials)])
#         condition.trials = laser_on_falarm
#         condition.label = 'laser_on_falarm'
#         condition.color = 'red'
#         allconditions[condition.label] = condition

#         #laser_off_falarm
#         condition = SimpleNamespace()
#         laser_off_falarm = np.copy(laser_off[np.isin(laser_off,falarm_trials)])
#         condition.trials = laser_off_falarm
#         condition.label = 'laser_off_falarm'
#         condition.color = 'red'
#         allconditions[condition.label] = condition

#         #laser_on_creject
#         condition = SimpleNamespace()
#         laser_on_creject = np.copy(laser_on[np.isin(laser_on,creject_trials)])
#         condition.trials = laser_on_creject
#         condition.label = 'laser_on_creject'
#         condition.color = 'green'
#         allconditions[condition.label] = condition

#         #laser_off_creject
#         condition = SimpleNamespace()
#         laser_off_creject = np.copy(laser_off[np.isin(laser_off,creject_trials)])
#         condition.trials = laser_off_creject
#         condition.label = 'laser_off_creject'
#         condition.color = 'green'
#         allconditions[condition.label] = condition

#         #laser_on_correct
#         condition = SimpleNamespace()
#         laser_on_correct = np.copy(laser_on[np.isin(laser_on,correct_trials)])
#         condition.trials = laser_on_correct
#         condition.label = 'laser_on_correct'
#         condition.color = 'green'
#         allconditions[condition.label] = condition

#         #laser_off_correct
#         condition = SimpleNamespace()
#         laser_off_correct = np.copy(laser_off[np.isin(laser_off,correct_trials)])
#         condition.trials = laser_off_correct
#         condition.label = 'laser_off_correct'
#         condition.color = 'green'
#         allconditions[condition.label] = condition

#         #laser_on_incorrect
#         condition = SimpleNamespace()
#         laser_on_incorrect = np.copy(laser_on[np.isin(laser_on,incorrect_trials)])
#         condition.trials = laser_on_incorrect
#         condition.label = 'laser_on_incorrect'
#         condition.color = 'red'
#         allconditions[condition.label] = condition

#         #laser_off_correct
#         condition = SimpleNamespace()
#         laser_off_incorrect = np.copy(laser_off[np.isin(laser_off,incorrect_trials)])
#         condition.trials = laser_off_incorrect
#         condition.label = 'laser_off_incorrect'
#         condition.color = 'red'
#         allconditions[condition.label] = condition

#     #Handle day of reversal conditions
#     if sessionfile.meta.task in ['switch','opto switch']:
#         condition = SimpleNamespace()
#         preswitch = np.array(range(sessionfile.meta.first_reversal_trial))
#         preswitch = preswitch[np.isin(preswitch,trimmed_trials_active)]
#         condition.trials = preswitch
#         condition.label = 'pre-switch'
#         condition.color = 'grey'
#         allconditions[condition.label] = condition

#         condition = SimpleNamespace()
#         postswitch = np.array(range(sessionfile.meta.first_reversal_trial,sessionfile.meta.length_in_trials))
#         postswitch = postswitch[np.isin(postswitch,trimmed_trials_active)]
#         condition.trials = postswitch
#         condition.label = 'post-switch'
#         condition.color = 'blue'
#         allconditions[condition.label] = condition


#     return allconditions

def getPSTH(sessionfile,clust,condition,PSTHstart = np.nan,PSTHend = np.nan,PSTHbuffer = np.nan,units='seconds',xpoints=1000,bw=0.05):
    #NOTE: All default values are set as np.nan because we need to know whether the values
    #have been overridden or not so we know whether they are in default units or in units
    #passed in in the units variable. If default units we will set them later on after unit
    #conversions

    if not units in ['s','seconds','ms','milliseconds']:
        error('Unrecognized units in PSTH')

    #Handle unit conversions between seconds and milliseconds
    if units in ['ms','milliseconds']:
        PSTHstart /= 1000
        PSTHend /= 1000
        PSTHbuffer /= 1000

    #Handle default values
    if np.isnan(PSTHstart):
        PSTHstart = 0
    if np.isnan(PSTHend):
        PSTHend = 2.5
    if np.isnan(PSTHbuffer):
        PSTHbuffer = 0.1

    #Caching search -- Unit
    idx = np.equal(sessionfile.spikes.clusters,clust)
    totaltimes = sessionfile.spikes.times[idx]

    peristimulustimes = []
    totalspiketimes = 0
    for trialidx,trial in enumerate(condition.trials):
        trialstart = sessionfile.trials.starts[trial]

        #Get total spike counts
        starttime = trialstart + PSTHstart*sessionfile.meta.fs
        endtime = trialstart + PSTHend*sessionfile.meta.fs
        #Caching search -- Condition
        idx = np.logical_and(      np.greater(totaltimes,starttime) , np.less(totaltimes,endtime)     )
        totalspiketimes += np.sum(idx)

        #Get spike times for PSTH
        starttime = trialstart + (PSTHstart - PSTHbuffer)*sessionfile.meta.fs
        endtime = trialstart + (PSTHend + PSTHbuffer)*sessionfile.meta.fs
        #Caching search -- Condition
        idx = np.logical_and(      np.greater(totaltimes,starttime) , np.less(totaltimes,endtime)     )
        trialtimes = totaltimes[idx]

        #PSTH times
        peristimulustimes = np.concatenate((peristimulustimes,(trialtimes-trialstart)/sessionfile.meta.fs))

    xrange = np.linspace(PSTHstart,PSTHend,num=xpoints)
    KDE = gaussian_kde(peristimulustimes,bw_method=bw).evaluate(xrange)
    FR = KDE * totalspiketimes / len(condition.trials)

    #Handle unit conversions again
    if units in ['ms','milliseconds']:
        #FR *= 1000
        xrange *= 1000

    return xrange,FR

def OLDgetRaster(sessionfile,clust,condition,rasterStart = np.nan,rasterEnd = np.nan,units='seconds'):
    #NOTE: All default values are set as np.nan because we need to know whether the values
    #have been overridden or not so we know whether they are in default units or in units
    #passed in in the units variable. If default units we will set them later on after unit
    #conversions

    if not units in ['s','seconds','ms','milliseconds']:
        error('Unrecognized units in raster')

    #Handle unit conversions between seconds and milliseconds
    if units in ['ms','milliseconds']:
        rasterStart /= 1000
        rasterEnd /= 1000

    #Handle default values
    if np.isnan(rasterStart):
        rasterStart = 0
    if np.isnan(rasterEnd):
        rasterEnd = 2.5

    #Caching search -- Unit
    idx = np.equal(sessionfile.spikes.clusters,clust)
    totaltimes = sessionfile.spikes.times[idx]

    timestoplot = []
    trialstoplot = []

    for trialidx,trial in enumerate(condition.trials):
        trialstart = sessionfile.trials.starts[trial]

        starttime = trialstart + rasterStart*sessionfile.meta.fs
        endtime = trialstart + rasterEnd*sessionfile.meta.fs

        #Caching search -- Condition
        idx = np.logical_and(      np.greater(totaltimes,starttime) , np.less(totaltimes,endtime)     )
        trialtimes = totaltimes[idx]
        #Convert from samples to seconds
        trialtimes = (trialtimes - trialstart) / sessionfile.meta.fs

        trialstoplot = np.concatenate(( trialstoplot,[trialidx]*len(trialtimes) ))
        timestoplot = np.concatenate(( timestoplot,trialtimes ))

    #Handle unit conversions again
    if units in ['ms','milliseconds']:
        timestoplot *= 1000

    return timestoplot,trialstoplot

def getRaster(sessionfile,clust,condition,startbuffer=0.5,endbuffer=0,units='seconds'):
    #NOTE: All default values are set as np.nan because we need to know whether the values
    #have been overridden or not so we know whether they are in default units or in units
    #passed in in the units variable. If default units we will set them later on after unit
    #conversions

    cachedtimes = getSpikeTimes(sessionfile,clust=clust)
    trialspikes = []
    for trialidx,trial in enumerate(condition.trials):
        trialspikes.append(getTrialSpikes(sessionfile,trial,startbuffer=startbuffer,endbuffer=endbuffer,cachedtimes=cachedtimes,outunits=units))

    return trialspikes








def getPossibleCombinationsAZ(sessionfile, trialsPerDayLoaded, minClust = 1, MaxClust=1, ClustSkip=1, MaxSample=1, MinTrials = 100, MaxIter = 100, categories = 'stimulus', verbose = False):
    # session, finds all the good cluster list, returns all combinations with more than 50 trials in session
    full_clust_list = sessionfile.clusters.good.tolist()
    #full_subsets = []
    possible_subsets = []
    # take the smaller as max cluster number in a given ensemble
    MaxClust = min(MaxClust, len(full_clust_list))
    # update category
    if categories == 'stimulus':
        categories = ['target','nontarget']
    elif categories == 'response':
        categories = ['go','nogo']

    elif categories == 'stimulus_off':
        categories = ['laser_off_target','laser_off_nontarget']
    elif categories == 'stimulus_on':
        categories = ['laser_on_target','laser_on_nontarget']
    elif categories == 'response_off':
        categories = ['laser_off_go','laser_off_nogo']
    elif categories == 'response_on':
        categories = ['laser_on_go','laser_on_nogo']


    elif categories == 'stimulus_pre':
        categories = ['pre_switch_target','pre_switch_nontarget']
    elif categories == 'stimulus_post':
        categories = ['post_switch_target','post_switch_nontarget']

    elif categories == 'response_pre':
        categories = ['pre_switch_go','pre_switch_nogo']
    elif categories == 'response_post':
        categories = ['post_switch_go','post_switch_nogo']
    # for each cluster length, sample MaxSample number of combinations
    current_ensemble_list = []
    for L in range(minClust, MaxClust + 1, ClustSkip):
        curr_sample_count = 0
        curr_iter = 0
        while curr_sample_count < MaxSample and curr_iter < MaxIter:
            clust_list = random.sample(full_clust_list, L)
            # sort list here
            
            clust_list.sort()

            # test if clust_list is legal
            trimmed_trials_activeList = []
            # try:
            for clust in clust_list:
                trimmed_trials_active = np.array(sessionfile.trim[clust].trimmed_trials)
                if trialsPerDayLoaded != 'NO_TRIM':
                    active_trials = trialsPerDayLoaded[sessionfile.meta.animal][sessionfile.meta.day_of_training]
                    trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,active_trials)] #
                if clust == clust_list[0]:
                    trimmed_trials_activeList = trimmed_trials_active
                else:
                    trimmed_trials_activeList = list(set(trimmed_trials_activeList).intersection(trimmed_trials_active))
            # add in condition filtering to ensure that we have enough conditions of that cell in trimmed
            trimmed_trials_activeList = np.array(trimmed_trials_activeList)
            #Remove all trials that do not belong to one of the conditions in question
            included_in_conditions_mask = []
            all_conditions = getAllConditions(sessionfile,clust,trialsPerDayLoaded=trialsPerDayLoaded)
            for category in categories:
                included_in_conditions_mask = np.concatenate((included_in_conditions_mask,all_conditions[category].trials))
            trimmed_trials_activeList = trimmed_trials_activeList[np.isin(trimmed_trials_activeList,included_in_conditions_mask)]
            
            if len(trimmed_trials_activeList)>=MinTrials and not str(clust_list) in current_ensemble_list:
                # print('doing thresholding')
                possible_subsets.append(clust_list)
                current_ensemble_list.append(str(clust_list))
                curr_sample_count = curr_sample_count + 1
                curr_iter = 0
            curr_iter = curr_iter + 1
            # except:
            #     curr_iter = curr_iter + 1
            #     continue
        if curr_sample_count == MaxSample:
            if verbose:
                print(generateSaveString(sessionfile) + ' find enough trial for ensemble '+str(L))
                print(generateSaveString(sessionfile) + ' find enough trial in '+str(curr_iter) +' iterations')
        if curr_iter >= MaxIter:
            if verbose:
                print(generateSaveString(sessionfile) + ' cannot find enough trial for ensemble '+str(L))
            # no point in trying a higher value
            break
    
    return possible_subsets

def flatten_rates(sessionfile,clusts,cond,bin_length_msec=50,pre_stim_padding_sec = 0.2,trialsPerDayLoaded=None):
    if type(clusts) == int or type(clusts) == float:
        clusts = [clusts]
    assert type(clusts) == list or type(clusts) == np.ndarray
    
    allConds = getAllConditions(sessionfile,clusts,trialsPerDayLoaded=trialsPerDayLoaded)
    trials = allConds[cond].trials
    
    n_clusts = len(clusts)
    n_trials = len(trials)
    n_bins = int((sessionfile.meta.triallength+pre_stim_padding_sec)*1000 / bin_length_msec)
    
    rates = np.full((n_clusts,n_trials*n_bins),np.nan)
    
    for clust_idx,clust in enumerate(clusts):
        for trial_idx,trial in enumerate(trials):
            for bin_idx,bin in enumerate(range(n_bins)):

                bin_start = sessionfile.trials.starts[trial] - pre_stim_padding_sec * sessionfile.meta.fs + bin*(bin_length_msec/1000*sessionfile.meta.fs)
                bin_end = bin_start + bin_length_msec/1000*sessionfile.meta.fs

                spike_times = getSpikeTimes(sessionfile,clust=clust,starttime=bin_start,endtime=bin_end)
                rates[clust_idx,trial_idx*n_bins + bin_idx] = len(spike_times) / bin_length_msec * 1000

    return rates

def calc_dim(N,v,cv):
    num = 1 + (N-1)*v
    den = 1 + (N-1)*cv
    return num/den

def calc_dim_GT(cov):
    return(np.trace(cov)**2 / np.trace(cov@cov))

def calculateDimensionalityParallel(session,directory,output_dir,ensemble,trialsPerDayLoaded,bin_length_msec=50,trial_start_time_ms=-200):
    
    #First check if this ensemble has already been calculated
    ensemble_txt = str(ensemble).replace('[','').replace(']','').replace(', ','-')
    filename = 'Session_'+session+'_ensemble_'+ensemble_txt+'_cached_data.pickle'
    print(f"output_dir is: ->{output_dir}<-")
    print(f"ensemble_txt is: ->{ensemble_txt}<-")
    filename = os.path.join(output_dir,filename)
    if os.path.isfile(filename):
        try:
            with open(filename, 'rb') as f:
                results = pickle.load(f)
                return results
        except Exception as e:
            print(f"Analysis - line 720: {e}")

    #If not, then calculate dimensionality
    sessionfile = loadSessionCached(directory,session)

    FRmods = [sessionfile.responsiveness[clust]['all_trials'].FRmodulation for clust in ensemble]
    NCR_ratio = np.mean(np.less(FRmods,3.5))
    d_var,d_covar,d_GT,indiv = calculate_dimensionality(sessionfile,ensemble,bin_length_msec=bin_length_msec,trialsPerDayLoaded=trialsPerDayLoaded,trial_start_time_ms=trial_start_time_ms)

    results = dict()
    results['d_var'] = d_var
    results['d_covar'] = d_covar
    results['d_GT'] = d_GT
    results['FRmods'] = FRmods
    results['NCR_ratio'] = NCR_ratio
    results['indiv'] = indiv
    try:
        with open(filename, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Analysis - line 740: {e}")
    print('regular results')
    print(results)
    return results

def calculate_dimensionality(sessionfile,ensemble,bin_length_msec=50,trialsPerDayLoaded=None,trial_start_time_ms=-200):
    # clusters = sessionfile.clusters.good
    n_clust = len(ensemble)

    if n_clust < 2:
        print(f"Too few neurons to calculate dimensionality for {generateDateString(sessionfile)}: {n_clust} cells")
        return np.nan,np.nan,np.nan,np.nan

    rates = flatten_rates(sessionfile,ensemble,'all_trials',bin_length_msec=bin_length_msec,trialsPerDayLoaded=trialsPerDayLoaded,pre_stim_padding_sec=-1*trial_start_time_ms/1000)
    cov_mat = np.cov(rates)

    variances = cov_mat[np.eye(n_clust,n_clust,dtype='bool')]
    covariances = cov_mat[~np.eye(n_clust,n_clust,dtype='bool')]

    avg_sq_vars = np.mean(variances**2)
    avg_sq_covars = np.mean(covariances**2)
    avg_pred_covars = np.mean(np.outer(variances,variances)[~np.eye(n_clust,n_clust,dtype='bool')])

    dim_var = calc_dim(n_clust,avg_pred_covars/avg_sq_vars,0)
    # dim_covar = calc_dim(n_clust,avg_pred_covars/avg_sq_vars,avg_sq_covars/avg_sq_vars)       #This calculates the mathematical equivalent of the GT dimensionality
    dim_covar = calc_dim(n_clust,1,avg_sq_covars/avg_sq_vars)                                   #This calculates the dimensionality solely due to covariances
    dim_GT = calc_dim_GT(cov_mat)
    
    individual_contributions = calculate_individual_contribution(cov_mat)
    
    return dim_var,dim_covar,dim_GT,individual_contributions

def calculate_individual_contribution(cov_mat):
    assert cov_mat.shape[0] == cov_mat.shape[1]
    n_clust = cov_mat.shape[0]

    variances = cov_mat[np.eye(n_clust,n_clust,dtype='bool')]
    N = len(variances)
    indiv = np.full_like(variances,np.nan)

    for idx in range(N):
        numerator = np.sum(variances)*variances[idx] - variances[idx]*variances[idx]
        denominator = 2*np.sum(variances*variances)
        indiv[idx] = 1/N + (N-1)/N*numerator/denominator
        
    return indiv










#Bootstrapped Mw and SEMw calculation adapted from Gatz & Smith (1993)
# THE STANDARD ERROR OF A WEIGHTED MEAN CONCENTRATION-I. BOOTSTRAPPING VS OTHER METHODS

def bootstrap_Mw_SEMw(FRmods,B=1000,n_frac=1,CR_NCR_thresh=3.5,hierarchical=False):
    FRmods = np.array(FRmods)
    mask = np.greater([len(l) for l in FRmods],0)
    FRmods = FRmods[mask]
    
    Mw_list = []
    for idx_B in range(B):
        if hierarchical:
            this_sample_animals = []
            N = len(FRmods)
            this_sample_animals = np.random.choice(FRmods,N)
        else:
            this_sample_animals = FRmods
        
        this_sample = []
        for this_FRmod_list in this_sample_animals:
            n = int(n_frac*len(this_FRmod_list))
            if n < 1:
                n=1
            sample = np.random.choice(this_FRmod_list,n)
            this_sample.append(sample)
        
        this_pcs = [np.mean(np.less(FRmod,CR_NCR_thresh)) for FRmod in this_sample]
        this_sizes = [len(l) for l in this_sample]
        Mw = np.average(this_pcs,weights=this_sizes)
        Mw_list.append(Mw)
        
    Mw_estimate = np.mean(Mw_list)*100
    SEMw_estimate = np.std(Mw_list)*100# / np.sqrt(B)
        
    return Mw_estimate,SEMw_estimate

def bootstrap_pc_NCR_test(X,Y,B=250000,CR_NCR_thresh=[3.5],hierarchical=False):
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7906290/
    #Application of the hierarchical bootstrap to multi-level data in neuroscience
    #Saravanan et al.
    
    #Testing against the null hypothesis that these two groups are the same
    X = np.array(X)
    mask = np.greater([len(l) for l in X],0)
    X = X[mask]
    
    Y = np.array(Y)
    mask = np.greater([len(l) for l in Y],0)
    Y = Y[mask]

    #Handle differing CR NCR Thresholds. Scale up to 2 if only one is provided
    if len(CR_NCR_thresh) == 1:
        CR_NCR_thresh = [CR_NCR_thresh[0],CR_NCR_thresh[0]]
    
    MwX_list = []
    for idx_B in range(B):
        if hierarchical:
            this_sample_animals = []
            N = len(FRmods)
            this_sample_animals = np.random.choice(X,N)
        else:
            this_sample_animals = X
        
        this_sample = []
        for this_list in this_sample_animals:
            n = len(this_list)
            sample = np.random.choice(this_list,n)
            this_sample.append(sample)
        
        this_pcs = [np.mean(np.less(X,CR_NCR_thresh[0])) for X in this_sample]
        this_sizes = [len(l) for l in this_sample]
        Mw = np.average(this_pcs,weights=this_sizes)
        MwX_list.append(Mw)
        
    MwY_list = []
    for idx_B in range(B):
        if hierarchical:
            this_sample_animals = []
            N = len(FRmods)
            this_sample_animals = np.random.choice(Y,N)
        else:
            this_sample_animals = Y
        
        this_sample = []
        for this_list in this_sample_animals:
            n = len(this_list)
            sample = np.random.choice(this_list,n)
            this_sample.append(sample)
        
        this_pcs = [np.mean(np.less(Y,CR_NCR_thresh[1])) for Y in this_sample]
        this_sizes = [len(l) for l in this_sample]
        Mw = np.average(this_pcs,weights=this_sizes)
        MwY_list.append(Mw)
        
    #Two-sided test
    diffs = (np.array(MwY_list) - np.array(MwX_list))
    pY = 1 - np.mean(np.greater(diffs,0))
    pX = 1 - np.mean(np.less(diffs,0))
    pboot = min(pX,pY)*2
    return pboot