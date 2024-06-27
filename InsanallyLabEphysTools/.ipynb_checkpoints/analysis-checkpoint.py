import numpy as np
from types import SimpleNamespace
from scipy.stats import gaussian_kde
from .utility import getSpikeTimes,getTrialSpikes

#The condition structure exists to make passing subsets of recordings between analyses
#easier.
#Every condition contains the following
#cond.trials: zero indexed trials included
#cond.label: condition label
#cond.color: color for plotting

def getAllConditions(sessionfile,clust):
    allconditions = dict()

    #Condition A -- All Trials
    condition = SimpleNamespace()
    all_trials = np.full((sessionfile.meta.length_in_trials), True)
    all_trials = np.array(np.where(all_trials)[0])
    all_trials = all_trials[np.isin(all_trials,sessionfile.trim[clust].trimmed_trials)]
    condition.trials = all_trials
    condition.label = 'all_trials'
    condition.color = 'grey'
    allconditions[condition.label] = condition

    #Target
    condition = SimpleNamespace()
    target_tone = sessionfile.trials.target
    target_tone = np.array(np.where(target_tone)[0])
    target_tone = target_tone[np.isin(target_tone,sessionfile.trim[clust].trimmed_trials)]
    condition.trials = target_tone
    condition.label = 'target_tone'
    condition.color = 'green'
    allconditions[condition.label] = condition

    #Nontarget
    condition = SimpleNamespace()
    nontarget_tone = np.logical_not(sessionfile.trials.target)
    nontarget_tone = np.array(np.where(nontarget_tone)[0])
    nontarget_tone = nontarget_tone[np.isin(nontarget_tone,sessionfile.trim[clust].trimmed_trials)]
    condition.trials = nontarget_tone
    condition.label = 'nontarget_tone'
    condition.color = 'purple'
    allconditions[condition.label] = condition

    #Go
    condition = SimpleNamespace()
    go_response = sessionfile.trials.go
    go_response = np.array(np.where(go_response)[0])
    go_response = go_response[np.isin(go_response,sessionfile.trim[clust].trimmed_trials)]
    condition.trials = go_response
    condition.label = 'go_response'
    condition.color = 'green'
    allconditions[condition.label] = condition

    #No-go
    condition = SimpleNamespace()
    nogo_response = np.logical_not(sessionfile.trials.go)
    nogo_response = np.array(np.where(nogo_response)[0])
    nogo_response = nogo_response[np.isin(nogo_response,sessionfile.trim[clust].trimmed_trials)]
    condition.trials = nogo_response
    condition.label = 'nogo_response'
    condition.color = 'purple'
    allconditions[condition.label] = condition

    #Handle optogenetic stimulation conditions
    if sessionfile.meta.task in ['opto nonreversal','opto reversal','opto switch']:
        condition = SimpleNamespace()
        laser_on = sessionfile.trials.laser_stimulation
        laser_on = np.array(np.where(laser_on)[0])
        laser_on = laser_on[np.isin(laser_on,sessionfile.trim[clust].trimmed_trials)]
        condition.trials = laser_on
        condition.label = 'laser_on'
        condition.color = 'blue'
        allconditions[condition.label] = condition

        condition = SimpleNamespace()
        laser_off = np.logical_not(sessionfile.trials.laser_stimulation)
        laser_off = np.array(np.where(laser_off)[0])
        laser_off = laser_off[np.isin(laser_off,sessionfile.trim[clust].trimmed_trials)]
        condition.trials = laser_off
        condition.label = 'laser_off'
        condition.color = 'grey'
        allconditions[condition.label] = condition

        condition = SimpleNamespace()
        laser_on_target = np.copy(laser_on[np.isin(laser_on,target_tone)])
        condition.trials = laser_on_target
        condition.label = 'laser_on_target'
        condition.color = 'blue'
        allconditions[condition.label] = condition

        condition = SimpleNamespace()
        laser_off_target = np.copy(laser_off[np.isin(laser_off,target_tone)])
        condition.trials = laser_off_target
        condition.label = 'laser_off_target'
        condition.color = 'grey'
        allconditions[condition.label] = condition

        condition = SimpleNamespace()
        laser_on_nontarget = np.copy(laser_on[np.isin(laser_on,nontarget_tone)])
        condition.trials = laser_on_nontarget
        condition.label = 'laser_on_nontarget'
        condition.color = 'blue'
        allconditions[condition.label] = condition

        condition = SimpleNamespace()
        laser_off_nontarget = np.copy(laser_off[np.isin(laser_off,nontarget_tone)])
        condition.trials = laser_off_nontarget
        condition.label = 'laser_off_nontarget'
        condition.color = 'grey'
        allconditions[condition.label] = condition

    #Handle day of reversal conditions
    if sessionfile.meta.task in ['switch','opto switch']:
        condition = SimpleNamespace()
        preswitch = np.array(range(sessionfile.meta.first_reversal_trial))
        preswitch = preswitch[np.isin(preswitch,sessionfile.trim[clust].trimmed_trials)]
        condition.trials = preswitch
        condition.label = 'pre-switch'
        condition.color = 'grey'
        allconditions[condition.label] = condition

        condition = SimpleNamespace()
        postswitch = np.array(range(sessionfile.meta.first_reversal_trial,sessionfile.meta.length_in_trials))
        postswitch = postswitch[np.isin(postswitch,sessionfile.trim[clust].trimmed_trials)]
        condition.trials = postswitch
        condition.label = 'post-switch'
        condition.color = 'blue'
        allconditions[condition.label] = condition

    return allconditions

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