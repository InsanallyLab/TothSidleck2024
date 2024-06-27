from .analysis import getAllConditions
from .utility import getTrialSpikes, getSpikeTimes
from .io import loadSessionCached, generateSaveString
from .utility import generateDateString

import numpy as np
import pickle
import os
import random
import traceback
from types import SimpleNamespace
from scipy.stats import gaussian_kde, sem, mannwhitneyu
from scipy.interpolate import interp1d
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.neighbors import KernelDensity
from KDEpy import FFTKDE




################################### Validation ################################

def Train_Test_Split(trials,frac_test = 0.1):
    """
    Splits a set of trails into test and train datasets
    trials: set of trials available in dataset
    frac_test: fraction of trails to use for test (0 for leave-one-out)
    
    returns (train_trials,test_trials)
    """
    
    N = len(trials)
    
    #Test set size. Must be at least one
    N_test = int(frac_test * N)
    if N_test < 1:
        N_test = 1
        
    #Train set size. Must also be at least one.
    N_train = N - N_test
    if N_train < 1:
        print('ERROR: No training data. Test fraction likely too high')
        raise Exception
        
    test_idxs = np.concatenate(( [False]*N_train , [True]*N_test ))
    test_idxs = np.random.permutation(test_idxs)
    train_idxs = np.logical_not(test_idxs)
    
    test_trials = trials[test_idxs]
    train_trials = trials[train_idxs]
    
    return (train_trials,test_trials)

def K_fold(trials,K):
   trials = np.array(trials)
   N = len(trials)
   
   if K >= N:
       K = N
       
   #all_trials = range(N)
   all_trials = np.random.permutation(list(range(N)))
   
   idxs = np.array(all_trials) % K
   
   train_test_pairs = []
   
   for k in range(K):
       test_idxs = np.equal(idxs,k)
       train_idxs = np.logical_not(test_idxs)
       
       train_trials = trials[train_idxs]
       test_trials = trials[test_idxs]
       
       train_test_pairs.append((train_trials,test_trials))
   
   return train_test_pairs

def K_fold_strat(sessionfile,trials,K):
    trials = np.array(trials)
    
    X = np.ones(len(trials))
    y = np.ones(len(trials))
    for idx,trial in enumerate(trials):
        if sessionfile.trials.target[trial] and sessionfile.trials.go[trial]:
            y[idx] = 1
        elif sessionfile.trials.target[trial] and not sessionfile.trials.go[trial]:
            y[idx] = 2
        elif not sessionfile.trials.target[trial] and sessionfile.trials.go[trial]:
            y[idx] = 3
        elif not sessionfile.trials.target[trial] and not sessionfile.trials.go[trial]:
            y[idx] = 4
    
    train_test_pairs = []
    # print(f"Data length is {len(X)}/{len(y)}, K is equal to {K}")
    skf = StratifiedKFold(n_splits=K,shuffle=True)
    for splitX,splitY in skf.split(X, y):
#         plt.figure()
#         plt.hist(y[splitY])
        
        train_trials = trials[splitX]
        test_trials = trials[splitY]
        train_test_pairs.append((train_trials,test_trials))
    return train_test_pairs

def K_fold_strat_MATCHED_CHOICE(sessionfile,trials,K):
    trials = np.array(trials)
    
    X = np.ones(len(trials))
    y = np.ones(len(trials))
    for idx,trial in enumerate(trials):
        if sessionfile.trials.target[trial] and sessionfile.trials.go[trial]:
            y[idx] = 1
        elif sessionfile.trials.target[trial] and not sessionfile.trials.go[trial]:
            y[idx] = 2
        elif not sessionfile.trials.target[trial] and sessionfile.trials.go[trial]:
            y[idx] = 3
        elif not sessionfile.trials.target[trial] and not sessionfile.trials.go[trial]:
            y[idx] = 4

    #Enforce an even number of go and nogo trials
    y_go_mask   = np.logical_or(np.equal(y,1) , np.equal(y,3))
    y_nogo_mask = np.logical_or(np.equal(y,2) , np.equal(y,4))
    ymin = min(np.sum(y_go_mask),np.sum(y_nogo_mask))
    y_go_idx = np.where(y_go_mask)[0]
    y_nogo_idx = np.where(y_nogo_mask)[0]
    idx_go_new = np.random.choice(y_go_idx,ymin,replace=False)
    idx_nogo_new = np.random.choice(y_nogo_idx,ymin,replace=False)              #Needs to stay in order
    idx_new = np.concatenate((idx_go_new,idx_nogo_new))
    X = X[idx_new]
    y = y[idx_new]
    
    train_test_pairs = []
    # print(f"Data length is {len(X)}/{len(y)}, K is equal to {K}")
    skf = StratifiedKFold(n_splits=K,shuffle=True)                          #X,y = features,labels.    splitX,splitY = train/test
    for splitX,splitY in skf.split(X, y):
#         plt.figure()
#         plt.hist(y[splitY])
        train_trials = trials[idx_new][splitX]
        test_trials = trials[idx_new][splitY]
        train_test_pairs.append((train_trials,test_trials))
    return train_test_pairs













################################### Bandwidth #################################

def getBWs_elife2019():
    #RNN
    return np.linspace(.005, 0.305, 11)
    #Insanally 2017 set_bw_in_time
    #return np.linspace(.010, .500, 50)
    #Insanally 2017 set_bw
    #return np.linspace(.001, 1.00, 100)

def sklearn_grid_search_bw(sessionfile,clust,trialsPerDayLoaded,interval,folds = 50):
    conditions = getAllConditions(sessionfile,clust,trialsPerDayLoaded=trialsPerDayLoaded)
    
    trialsToUse = conditions['all_trials'].trials
    trialsToUse = np.random.permutation(trialsToUse)
    trialsToUse = trialsToUse[0:int(len(trialsToUse)/2)]

    LogISIs,_ = getLogISIs(sessionfile,clust,trialsToUse,interval)
    #Ensure that we don't try to cross validate more than there are ISIs
    folds = np.min([folds,len(LogISIs)])

    LogISIs = LogISIs.reshape(-1, 1)#Required to make GridSearchCV work
    #print(f"There are {len(LogISIs)} ISIs for bw selection")
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': getBWs_elife2019()},
                    cv=folds) # 20-fold cross-validation
    grid.fit(LogISIs)
    return grid.best_params_['bandwidth']



################################### Training ##################################

#Trials is passed in 
def splitByConditions(sessionfile,clust,trialsPerDayLoaded,trials,condition_names):
    all_conditions = getAllConditions(sessionfile,clust,trialsPerDayLoaded=trialsPerDayLoaded)
    
    decoding_conditions = dict()
    for cond in condition_names:
        decoding_conditions[cond] = SimpleNamespace()
    
    for cond in decoding_conditions:
        condition_trials = trials[np.isin(trials, all_conditions[cond].trials )]
        decoding_conditions[cond].trials = condition_trials
        
    return decoding_conditions
    
def LogISIsToLikelihoods(LogISIs, bw):
    #minLogISI = 0 #Set minimum to 1ms. This is shorter than the refractory period so we can guarantee we won't get any ISIs less than this
    #maxLogISI = np.max(LogISIs) +1 #Set this to 10x the maximum ISI. This nearly guarantees that there won't be any ISIs outside this range
    
    #KDE = gaussian_kde(LogISIs,bw_method=bw)
    #return KDE

    #print(f"There are {len(LogISIs)} ISIs for Likelihood estimation")

    x = np.linspace(-2,6,100)
    y = FFTKDE(bw=bw, kernel='gaussian').fit(LogISIs, weights=None).evaluate(x)
    
    # Use scipy to interplate and evaluate on arbitrary grid
    f = interp1d(x, y, kind='linear', assume_sorted=True)

    #Also want to generate an inverse CDF for smapling
    #This should go [0,1] -> xrange
    norm_y = np.cumsum(y) / np.sum(y)
    norm_y[0] = 0
    norm_y[len(norm_y)-1] = 1
    inv_f = interp1d(norm_y,x, kind='linear', assume_sorted=True)

    return f,inv_f



















class TrialInterval:
    _startTimeSamples = 0
    _endTimeSamples = 0
    _isStartTimeRelToResponse = True
    _isEndTimeRelToResponse = True
    _averageLickDelay = 0.2 * 30000

    def __init__(self,start,end,startresp,endresp):
        self._startTimeSamples = start
        self._endTimeSamples = end
        self._isStartTimeRelToResponse = startresp
        self._isEndTimeRelToResponse = endresp

    def _CalculateAvgLickDelay(self,sessionfile):
        go_responses = np.array(sessionfile.trials.response)[sessionfile.trials.go]
        go_starts = np.array(sessionfile.trials.starts)[sessionfile.trials.go]
        self._averageLickDelay = np.nanmean(go_responses - go_starts)

    def _ToTimestamp(self,sessionfile,trial):
        starttime = sessionfile.trials.starts[trial]
        if self._isStartTimeRelToResponse:
            starttime = sessionfile.trials.response[trial]
            if np.isnan(starttime):
                starttime = sessionfile.trials.starts[trial] + self._averageLickDelay
        starttime = starttime + self._startTimeSamples

        endtime = sessionfile.trials.starts[trial]
        if self._isEndTimeRelToResponse:
            endtime = sessionfile.trials.response[trial]
            if np.isnan(endtime):
                endtime = sessionfile.trials.starts[trial] + self._averageLickDelay
        endtime = endtime + self._endTimeSamples

        return [starttime,endtime]

    def _ToString(self):
        return f"Interval has start {self._startTimeSamples}, end {self._startTimeSamples}. startresp {self._isStartTimeRelToResponse} and endresp {self._isEndTimeRelToResponse}"

def getLogISIs(sessionfile,clust,trials,interval): #This code is probably redundant. Could be replaced entirely by cacheLogISIs
    ISIs = []
    times = []
    for trial in trials:
        starttime,endtime = interval._ToTimestamp(sessionfile,trial)
        spiketimes = getSpikeTimes(sessionfile,clust=clust,starttime = starttime, endtime = endtime)
        spiketimes = spiketimes * 1000 / sessionfile.meta.fs

        ISIs.append(np.diff(spiketimes))
        times.append(spiketimes[1:])

    ISIs = np.concatenate(ISIs)
    LogISIs = np.log10(ISIs)
    times = np.concatenate(times)
    return LogISIs, times

def cacheLogISIs(sessionfile,clust,interval):#Include conditions
    ISIs = []
    for trial in range(sessionfile.meta.length_in_trials):
        starttime,endtime = interval._ToTimestamp(sessionfile,trial)
        spiketimes = getSpikeTimes(sessionfile,clust=clust,starttime = starttime, endtime = endtime)
        spiketimes = spiketimes * 1000 / sessionfile.meta.fs

        ISIs.append(np.diff(spiketimes))

    LogISIs = np.array([np.log10(tISIs) for tISIs in ISIs],dtype='object')
    return LogISIs

# def synthetic_spiketrain(model,trial_length=2500):#only model.all is required to be filled out. Units in ms
#     LogISIs = []
#     ctime = 0
#     while True:
#         ISI = model.all.Inv_Likelihood(np.random.rand())
#         ctime += 10**ISI
#         if ctime <= trial_length:
#             LogISIs.append(ISI)
#         else:
#             break
#     return np.array(LogISIs)

def flatten(responses):
    return(np.array([i for j in responses for i in j]))

def synthetic_spiketrain(trial_ISIs,trial_length=2500):
    total_ISIs = flatten(trial_ISIs)
    LogISIs = []
    ctime = 0
    while True:
        ISI = np.random.choice(total_ISIs)
        ctime += 10**ISI
        if ctime <= trial_length:
            LogISIs.append(ISI)
        else:
            break
    return np.array(LogISIs)


def cachedtrainDecodingAlgorithm(sessionfile,clust,trialsPerDayLoaded,bw,cachedLogISIs,Train_X,condition_names,synthetic = False):
    model = SimpleNamespace()
    model.conds = dict()
    model.all = SimpleNamespace()
    for cond in condition_names:
        model.conds[cond] = SimpleNamespace()
    
    #Determine trials to use for each condition. Uses the conditions structures from
    #ilep to increase modularity and to ensure that all code is always using the same
    #conditions
    decoding_conditions = splitByConditions(sessionfile,clust,trialsPerDayLoaded,Train_X,condition_names)
    
    #Handle all_trials
    #print(f"Starting model training.\nThere are {len(Train_X)} trials")
    LogISIs = cachedLogISIs[Train_X]
    #print(f"Starting model training.\nThere are {len(LogISIs)} trial_ISIs")
    if len(LogISIs)<5:
        #print(f"Skipping fold. Not enough trials for ISI concatenation")
        return None
    LogISIs = np.concatenate(LogISIs)
    #print(f"Starting model training.\nThere are {len(LogISIs)} ISIs")
    #Check for feasability of this model
    if len(LogISIs)<5:
        #print(f"Skipping fold. Not enough ISIs")
        return None

    f,inv_f = LogISIsToLikelihoods(LogISIs,bw)
    model.all.Likelihood = f
    model.all.Inv_Likelihood = inv_f

    #Handle synthetic spiketrain generation
    if synthetic:
        synthetic_spiketrain_construction_set = np.copy(cachedLogISIs)
        cachedLogISIs = [ [] ]*sessionfile.meta.length_in_trials
        for trial in Train_X:
            cachedLogISIs[trial] = synthetic_spiketrain(synthetic_spiketrain_construction_set[Train_X])
        cachedLogISIs = np.array(cachedLogISIs,dtype='object')

    #Handle individual conditions
    LogISIs_per_trial = [len(l) for l in cachedLogISIs[Train_X]]
    total_empty_ISI_trials = np.sum(np.equal(  LogISIs_per_trial,0  ))
    for cond in decoding_conditions:        
        
        LogISIs = cachedLogISIs[decoding_conditions[cond].trials]
        LogISIs = np.concatenate(LogISIs)

        if len(LogISIs)<5:
            print(f"Skipping fold. Not enough ISIs for the {cond} condition")
            return None

        #LogISIs,_ = getLogISIs(sessionfile,clust,decoding_conditions[cond].trials)

        f,inv_f = LogISIsToLikelihoods(LogISIs,bw) #This is a gaussian KDE
        model.conds[cond].Likelihood = f
        model.conds[cond].Inv_Likelihood = inv_f
        model.conds[cond].Prior_0 = 1.0 / len(condition_names)

        #Calculate likelihood for 0-ISI trials
        LogISIs_per_trial = [len(l) for l in cachedLogISIs[decoding_conditions[cond].trials]]
        numerator = np.sum(np.equal(LogISIs_per_trial,0)) + 1
        denominator = total_empty_ISI_trials + len(condition_names)
        model.conds[cond].Prior_empty = numerator / denominator
        
    return model





def cachedpredictTrial(sessionfile,clust,model,trialISIs,conditions = ['target_tone','nontarget_tone'],synthetic = False,synthetic_spiketrain_construction_set=None):
    # if synthetic_spiketrain_construction_set is None:
    #     print(sessionfile.meta)
        
    if synthetic:
        LogISIs = synthetic_spiketrain(synthetic_spiketrain_construction_set)
        # try:
        #     LogISIs = synthetic_spiketrain(synthetic_spiketrain_construction_set)
        # except:
        #     print(sessionfile.meta)
        #     print(trialISIs)
        #     print(synthetic)
        #     print(synthetic_spiketrain_construction_set)
    else:
        LogISIs = trialISIs
    
    #Set up probabilities with initial values equal to priors
    probabilities = dict()
    for cond in conditions:
        probabilities[cond] = SimpleNamespace()
        probabilities[cond].prob = np.full(len(LogISIs)+1,np.nan)
        probabilities[cond].prob[0] =  np.log10(model.conds[cond].Prior_0)
    
    #Calculate change in W. LLR after each LogISI
    for cond in conditions:
        probabilities[cond].prob = np.cumsum(np.log10(np.concatenate((   [model.conds[cond].Prior_0] , model.conds[cond].Likelihood(LogISIs)   ))))

    ##Calculate change in W. LLR after each LogISI
    #for idx,logISI in enumerate(LogISIs):
    #    for cond in conditions:
    #        probabilities[cond].prob[idx+1] = probabilities[cond].prob[idx] + np.log10(model[cond].Likelihood.evaluate(logISI))

    #Exponentiate back from log so that normalization can take place
    for cond in conditions:
        probabilities[cond].prob = np.power(10,probabilities[cond].prob)
    
    #Calculate total probability sum to normalize against
    sum_of_probs = np.zeros(len(LogISIs)+1)  
    for cond in conditions:
        sum_of_probs += probabilities[cond].prob

    #Normalize all probabilities
    for cond in conditions:
        probabilities[cond].prob /= sum_of_probs


    #No ISIs in trial. guess instead.
    if len(LogISIs) < 1:
        keys = [cond for cond in probabilities]
        probs = [model.conds[cond].Prior_empty for cond in probabilities]
        maxidx = np.concatenate(np.argwhere(np.equal(probs,np.max(probs))))
        if len(maxidx) > 1:
            maxidx = np.random.permutation(maxidx)
        maxidx = maxidx[0]
        maxCond = keys[maxidx]
        #return None,None,None,None
        return maxCond,probs[maxidx],probabilities,True
    #ISIs were present in the trial. We can take out a proper choice by the algorithm.
    else:
        keys = [cond for cond in probabilities]
        probs = [probabilities[cond].prob for cond in probabilities]
        probs = [p[len(p)-1] for p in probs]
        maxidx = np.argmax(probs)
        maxCond = keys[maxidx]
        return maxCond,probs[maxidx], probabilities,False
        #return maxCond,probabilities[maxCond].prob[len(times)-1], probabilities,times

def cachedcalculateAccuracyOnFold(sessionfile,clust,model,cachedLogISIs,Test_X,weights,conditions=['target','nontarget'],synthetic=False):
    #Note: this call to getAllConditions uses NO_TRIM all the time because it is only used to determine correctness
    all_conditions = getAllConditions(sessionfile,clust,trialsPerDayLoaded='NO_TRIM')
    
    accumulated_correct = 0
    #accumulated_total = 0 # I have changed how weighting works to be weighted by condition prevalence so this is no longer necessary
    num_correct = 0
    num_total = 0
    num_empty = 0
    
    
    
    for trial in Test_X:
        
        synthetic_spiketrain_construction_set = np.copy(cachedLogISIs)[Test_X]
        cond,prob,_,empty_ISIs = cachedpredictTrial(sessionfile,clust,model,cachedLogISIs[trial],conditions=conditions,synthetic=synthetic,synthetic_spiketrain_construction_set=synthetic_spiketrain_construction_set)
        
        if cond is None:
            continue

        if np.isin(trial,all_conditions[cond].trials):
            accumulated_correct += weights[cond]
            num_correct += 1
        #accumulated_total += prob
        num_total += 1

        if empty_ISIs:
            num_empty += 1
        
    if num_total <= 0:
        return np.nan,np.nan,np.nan
    else:
        #print(f"Successful calculation of accuracy")
        return (num_correct/num_total),(accumulated_correct/num_total),(num_empty/num_total)

def calculateDecodingForSingleNeuron(session,clust,trialsPerDayLoaded,cache_directory,output_directory,trainInterval,testInterval,reps = 1,categories='stimulus'):
    sessionfile = loadSessionCached(cache_directory,session)
    filename = generateDateString(sessionfile) + ' cluster ' + str(clust) + ' decoding cached result.pickle'
    filename = os.path.join(output_directory,filename)
    
    print(f"\nStarting calculation for {filename}")
    if os.path.isfile(filename):
        print(f"file exists for {filename}")
        try:
            with open(filename, 'rb') as f:
                res = pickle.load(f)
                print(f"session {session} cluster {clust} has loaded cached results")
                return res
        except Exception as e:
            print(f"session {session} cluster {clust} has thrown error while loading file: {e}")
            traceback.print_exc()
            raise e
    else:
        print(f"{filename} not cached. Running from scratch")
        try:    
            #a,astd,asem,wa,wastd,wasem,ca,castd,casem,pval = ilep.cachedCalculateClusterAccuracy(sessionfile,clust,reps=reps,categories='stimulus')
            trainInterval._CalculateAvgLickDelay(sessionfile)
            testInterval._CalculateAvgLickDelay(sessionfile)
            res = cachedCalculateClusterAccuracy(sessionfile,clust,trialsPerDayLoaded,trainInterval,testInterval,reps=reps,categories=categories)
        except Exception as e:
            res = generateNullResults()
            print(f"session {session} cluster {clust} has thrown error {e}")
            traceback.print_exc()
            raise e
        try:
            with open(filename, 'wb') as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Problem saving {f} to {filename}. Error: {e}")
            traceback.print_exc()
            raise e
            
    print(f"finished with {generateDateString(sessionfile)} cluster {clust}")
    return res

def calculate_weights(sessionfile,clust,trimmed_trials_active,categories,trialsPerDayLoaded=None):
    num_total_trials = len(trimmed_trials_active)
    all_conditions = getAllConditions(sessionfile,clust,trialsPerDayLoaded=trialsPerDayLoaded)

    weights = dict()
    for cat in categories:
        weights[cat] = len(all_conditions[cat].trials)/num_total_trials
        weights[cat] = (1 / weights[cat]) / len(categories)
    return weights

def cachedCalculateClusterAccuracy(sessionfile,clust,trialsPerDayLoaded,trainInterval,testInterval,reps = 1,categories='stimulus',bw=None,K_fold_num=10):
    if bw is None:
        #bwInterval = TrialInterval(0,2.5*sessionfile.meta.fs,False,False)
        best_bw = sklearn_grid_search_bw(sessionfile,clust,trialsPerDayLoaded,trainInterval)
    else:
        best_bw = bw
        
    accuracy_per_fold = []
    waccuracy_per_fold = []
    accuracy_std_per_fold = []
    waccuracy_std_per_fold = []
    accuracy_sem_per_fold = []
    waccuracy_sem_per_fold = []
    
    # control_accuracy_per_fold = []
    # control_accuracy_std_per_fold = []
    # control_accuracy_sem_per_fold = []

    # control_waccuracy_per_fold = []
    # control_waccuracy_std_per_fold = []
    # control_waccuracy_sem_per_fold = []
    
    synthetic_accuracy_per_fold = []
    synthetic_accuracy_std_per_fold = []
    synthetic_accuracy_sem_per_fold = []

    synthetic_waccuracy_per_fold = []
    synthetic_waccuracy_std_per_fold = []
    synthetic_waccuracy_sem_per_fold = []

    fraction_empty = []

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
    
    cachedTrainLogISIs = cacheLogISIs(sessionfile,clust,trainInterval)
    cachedTestLogISIs = cacheLogISIs(sessionfile,clust,testInterval)

    ##eLife decoding criterion. 3 Spikes on 80% of trials
    #numISIs = [len(trialISIs) for trialISIs in cachedLogISIs]
    #if np.mean(np.greater_equal(numISIs,2)) < 0.8:
    #    return np.nan,np.nan,np.nan, np.nan,np.nan,np.nan, np.nan,np.nan,np.nan, np.nan

    model = None
    # model_c = None #Condition permutation
    model_s = None #Synthetic spiketrains

    #Get active trimmed trials
    trimmed_trials_active = np.array(sessionfile.trim[clust].trimmed_trials)
    if trialsPerDayLoaded != 'NO_TRIM':
        active_trials = trialsPerDayLoaded[sessionfile.meta.animal][sessionfile.meta.day_of_training]
        trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,active_trials)]

    #Remove all trials that do not belong to one of the conditions in question
    included_in_conditions_mask = []
    all_conditions = getAllConditions(sessionfile,clust,trialsPerDayLoaded=trialsPerDayLoaded)
    for category in categories:
        included_in_conditions_mask = np.concatenate((included_in_conditions_mask,all_conditions[category].trials))
    trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,included_in_conditions_mask)]

    weights = calculate_weights(sessionfile,clust,trimmed_trials_active,categories,trialsPerDayLoaded=trialsPerDayLoaded)
    #print(f"weights = {weights}")

    #print(f"trimmed trials active: {trimmed_trials_active}")
    #[print(f"{cat}: {all_conditions[cat].trials}") for cat in categories]

    for rep in (range(int(reps/K_fold_num))):

        # #Need to shuffle only trimmed trials without perturbing where they lie in the the total set of trials
        # shuffled_cachedTrainLogISIs = np.copy(cachedTrainLogISIs)
        # shuffled_cachedTestLogISIs = np.copy(cachedTestLogISIs)
        # perm = np.random.permutation(range(len(trimmed_trials_active)))
        # shuffled_cachedTrainLogISIs[trimmed_trials_active] = shuffled_cachedTrainLogISIs[trimmed_trials_active][perm]
        # shuffled_cachedTestLogISIs[trimmed_trials_active] = shuffled_cachedTestLogISIs[trimmed_trials_active][perm]

        # folds = K_fold_strat_MATCHED_CHOICE(sessionfile,trimmed_trials_active,K_fold_num)
        folds = K_fold_strat(sessionfile,trimmed_trials_active,K_fold_num)
        for K, (Train_X, Test_X) in enumerate(folds):

            #if model is None:
            model = cachedtrainDecodingAlgorithm(sessionfile,clust,trialsPerDayLoaded,best_bw,cachedTrainLogISIs,Train_X,categories)
            if model is None:
                fold_accuracy,fold_waccuracy,frac_empty = (np.nan,np.nan,np.nan)
            else:
                fold_accuracy,fold_waccuracy,frac_empty = cachedcalculateAccuracyOnFold(sessionfile,clust,model,cachedTestLogISIs,Test_X,weights,conditions = categories)
                print(f"fold accuracy is {fold_accuracy}. w = {fold_waccuracy}")
            accuracy_per_fold.append(fold_accuracy)
            waccuracy_per_fold.append(fold_waccuracy)
            fraction_empty.append(frac_empty)
            
            # #if model_c is None:
            # model_c = cachedtrainDecodingAlgorithm(sessionfile,clust,trialsPerDayLoaded,best_bw,shuffled_cachedTrainLogISIs,Train_X,categories)
            # if model_c is None:
            #     cfold_accuracy = np.nan
            # else:
            #     cfold_accuracy,cfold_waccuracy,_ = cachedcalculateAccuracyOnFold(sessionfile,clust,model_c,shuffled_cachedTestLogISIs,Test_X,weights,conditions = categories)
            # control_accuracy_per_fold.append(cfold_accuracy)
            # control_waccuracy_per_fold.append(cfold_waccuracy)

            #if model_s is None:
            model_s = cachedtrainDecodingAlgorithm(sessionfile,clust,trialsPerDayLoaded,best_bw,cachedTrainLogISIs,Train_X,categories,synthetic = True)
            if model_s is None:
                sfold_accuracy = np.nan
            else:
                sfold_accuracy,sfold_waccuracy,_ = cachedcalculateAccuracyOnFold(sessionfile,clust,model_s,cachedTestLogISIs,Test_X,weights,conditions = categories, synthetic = True)
            synthetic_accuracy_per_fold.append(sfold_accuracy)
            synthetic_waccuracy_per_fold.append(sfold_waccuracy)
            
    accuracy = np.nanmean(accuracy_per_fold)
    accuracy_std = np.nanstd(accuracy_per_fold)
    accuracy_sem = sem(accuracy_per_fold,nan_policy='omit')
    if type(accuracy_sem) == np.ma.core.MaskedConstant:
        accuracy_sem = np.nan
    
    waccuracy = np.nanmean(waccuracy_per_fold)
    waccuracy_std = np.nanstd(waccuracy_per_fold)
    waccuracy_sem = sem(waccuracy_per_fold,nan_policy='omit')
    if type(waccuracy_sem) == np.ma.core.MaskedConstant:
        waccuracy_sem = np.nan
    
    # control_accuracy = np.nanmean(control_accuracy_per_fold)
    # control_accuracy_std = np.nanstd(control_accuracy_per_fold)
    # control_accuracy_sem = sem(control_accuracy_per_fold,nan_policy='omit')
    # if type(control_accuracy_sem) == np.ma.core.MaskedConstant:
    #     control_accuracy_sem = np.nan

    # control_waccuracy = np.nanmean(control_waccuracy_per_fold)
    # control_waccuracy_std = np.nanstd(control_waccuracy_per_fold)
    # control_waccuracy_sem = sem(control_waccuracy_per_fold,nan_policy='omit')
    # if type(control_waccuracy_sem) == np.ma.core.MaskedConstant:
    #     control_waccuracy_sem = np.nan

    synthetic_accuracy = np.nanmean(synthetic_accuracy_per_fold)
    synthetic_accuracy_std = np.nanstd(synthetic_accuracy_per_fold)
    synthetic_accuracy_sem = sem(synthetic_accuracy_per_fold,nan_policy='omit')
    if type(synthetic_accuracy_sem) == np.ma.core.MaskedConstant:
        synthetic_accuracy_sem = np.nan

    synthetic_waccuracy = np.nanmean(synthetic_waccuracy_per_fold)
    synthetic_waccuracy_std = np.nanstd(synthetic_waccuracy_per_fold)
    synthetic_waccuracy_sem = sem(synthetic_waccuracy_per_fold,nan_policy='omit')
    if type(synthetic_waccuracy_sem) == np.ma.core.MaskedConstant:
        synthetic_waccuracy_sem = np.nan

    frac_empty = np.nanmean(fraction_empty)
    
    # pval_c = mannwhitneyu(accuracy_per_fold,control_accuracy_per_fold).pvalue
    pval_s = mannwhitneyu(accuracy_per_fold,synthetic_accuracy_per_fold).pvalue
    # pval_wc = mannwhitneyu(waccuracy_per_fold,control_waccuracy_per_fold).pvalue
    pval_ws = mannwhitneyu(waccuracy_per_fold,synthetic_waccuracy_per_fold).pvalue

    results = dict()
    results['accuracy'] = accuracy
    results['accuracy_std'] = accuracy_std
    results['accuracy_sem'] = accuracy_sem
    results['weighted_accuracy'] = waccuracy
    results['weighted_accuracy_std'] = waccuracy_std
    results['weighted_accuracy_sem'] = waccuracy_sem
    # results['shuffled_control_accuracy'] = control_accuracy
    # results['shuffled_control_accuracy_std'] = control_accuracy_std
    # results['shuffled_control_accuracy_sem'] = control_accuracy_sem
    # results['shuffled_control_weighted_accuracy'] = control_waccuracy
    # results['shuffled_control_weighted_accuracy_std'] = control_waccuracy_std
    # results['shuffled_control_weighted_accuracy_sem'] = control_waccuracy_sem
    results['synthetic_control_accuracy'] = synthetic_accuracy
    results['synthetic_control_accuracy_std'] = synthetic_accuracy_std
    results['synthetic_control_accuracy_sem'] = synthetic_accuracy_sem
    results['synthetic_control_weighted_accuracy'] = synthetic_waccuracy
    results['synthetic_control_weighted_accuracy_std'] = synthetic_waccuracy_std
    results['synthetic_control_weighted_accuracy_sem'] = synthetic_waccuracy_sem
    # results['pval_shuffled_control'] = pval_c
    results['pval_synthetic_control'] = pval_s
    # results['pval_weighted_shuffled_control'] = pval_wc
    results['pval_weighted_synthetic_control'] = pval_ws
    results['fraction_empty_trials'] = frac_empty
    print('regular results')
    print(results)
    return results

def generateNullResults():
    results = dict()
    results['accuracy'] = np.nan
    results['accuracy_std'] = np.nan
    results['accuracy_sem'] = np.nan
    results['weighted_accuracy'] = np.nan
    results['weighted_accuracy_std'] = np.nan
    results['weighted_accuracy_sem'] = np.nan
    # results['shuffled_control_accuracy'] = np.nan
    # results['shuffled_control_accuracy_std'] = np.nan
    # results['shuffled_control_accuracy_sem'] = np.nan
    # results['shuffled_control_weighted_accuracy'] = np.nan
    # results['shuffled_control_weighted_accuracy_std'] = np.nan
    # results['shuffled_control_weighted_accuracy_sem'] = np.nan
    results['synthetic_control_accuracy'] = np.nan
    results['synthetic_control_accuracy_std'] = np.nan
    results['synthetic_control_accuracy_sem'] = np.nan
    results['synthetic_control_weighted_accuracy'] = np.nan
    results['synthetic_control_weighted_accuracy_std'] = np.nan
    results['synthetic_control_weighted_accuracy_sem'] = np.nan
    # results['pval_shuffled_control'] = np.nan
    results['pval_synthetic_control'] = np.nan
    # results['pval_weighted_shuffled_control'] = np.nan
    results['pval_weighted_synthetic_control'] = np.nan
    results['fraction_empty_trials'] = np.nan
    print('null results')
    print(results)
    return results