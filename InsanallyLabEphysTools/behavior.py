import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy.stats import norm
from itertools import product

PHASE_COLORS = ['#DEB9E0','#B78AB9','#906D92','#ABC5E8','#869BB7','#5E6C80']
SWITCH_COLOR = '#B07A3B'













































































def pcdpFromOutcomes(outcomes):
    hit = np.sum(np.equal(outcomes,1))
    miss = np.sum(np.equal(outcomes,2))
    falarm = np.sum(np.equal(outcomes,3))
    creject = np.sum(np.equal(outcomes,4))

    pc = (hit+creject)/(hit+miss+falarm+creject)
    dp = norm.ppf((hit+1)/(hit+miss+2)) - norm.ppf((falarm+1)/(falarm+creject+2))
    
    return pc,dp

def getOutcomesFromSession(sessionfile):
    outcomes = np.full_like(sessionfile.trials.starts,np.nan)
    hits = np.logical_and(sessionfile.trials.go,sessionfile.trials.target)
    misses = np.logical_and(np.logical_not(sessionfile.trials.go),sessionfile.trials.target)
    falarms = np.logical_and(sessionfile.trials.go,np.logical_not(sessionfile.trials.target))
    crejects = np.logical_and(np.logical_not(sessionfile.trials.go),np.logical_not(sessionfile.trials.target))
    outcomes[hits] = 1*np.ones(np.sum(hits))
    outcomes[misses] = 2*np.ones(np.sum(misses))
    outcomes[falarms] = 3*np.ones(np.sum(falarms))
    outcomes[crejects] = 4*np.ones(np.sum(crejects))
    return outcomes

def getActiveTrials(outcomes,max_trim=1000,window_length=50,active_threshold=0.8,sated_threshold=0.1):
    hits = np.equal(outcomes,1)
    misses = np.equal(outcomes,2)
    falarms = np.equal(outcomes,3)
    crejects = np.equal(outcomes,4)
    gos = np.logical_or(hits,falarms)
    nogos = np.logical_or(misses,crejects)

    #Disable max trim
    #max_trim = len(outcomes)
    max_trim = min(len(outcomes),max_trim)
    
    #Determine lick rate over time
    number_windows = len(outcomes) - window_length + 1
    if number_windows <= 0:
        return list(range(len(outcomes)))
    lick_rates = np.full(number_windows,np.nan)
    for window_idx in range(number_windows):
        
        window_trials = list(range(window_idx,window_idx+window_length))
        window_gos = gos[window_trials]
        
        window_lick_rate = np.mean(window_gos)
        lick_rates[window_idx] = window_lick_rate
        
    #Find where lick rate is acceptable -- overlicking
    lick_rates_below_threshold = np.less_equal(lick_rates,active_threshold)
    first_valid_window = number_windows
    if np.sum(lick_rates_below_threshold) > 0:
        first_valid_window = np.min(np.where(lick_rates_below_threshold)[0])
    first_valid_window = int(  min(max_trim,first_valid_window) + 0.5*window_length  )
    if lick_rates[0] < active_threshold:
        first_valid_window = 0
    
    #Final result from first trim -- trials that are not overlicking
    valid_trials_overlicking = list(range(first_valid_window,len(outcomes)))
    valid_trials_overlicking = np.array(valid_trials_overlicking)
    
    #############################################################################
    
    #Find where lick rate is acceptable -- overlicking
    lick_rates_above_threshold = np.greater_equal(lick_rates,sated_threshold)
    last_valid_window = 0
    if np.sum(lick_rates_above_threshold) > 0:
        last_valid_window = np.max(np.where(lick_rates_above_threshold)[0])
    last_valid_window = int(  max(len(outcomes)-max_trim,last_valid_window) + 0.5*window_length  )
    #if lick_rates[len(lick_rates)-1] > sated_threshold:
    #    last_valid_window = len(lick_rates)-1 + window_length
    
    valid_trials_sated = list(range(0,last_valid_window))
    valid_trials_sated = np.array(valid_trials_sated)
    
    #############################################################################
    
    valid_trials = valid_trials_overlicking[np.isin(valid_trials_overlicking,valid_trials_sated)]
    
    #trimmed_outcomes = outcomes[valid_trials]    
    #return lick_rates,hits,falarms,valid_trials,trimmed_outcomes
    return valid_trials

def getExpertDaysPerStage(animalBehaviors,animal,phase_days,noTrim=False):
    expert_days = []
    days_to_check = np.sort(phase_days)[::-1]
    days_to_check = days_to_check[np.isin(days_to_check,list(animalBehaviors[animal].sessions.keys()))]
    for day in days_to_check:
        outcomes = np.array(list(animalBehaviors[animal].sessions[day].outcomes))
        
        if noTrim:
            trimmed_outcomes = outcomes
        else:
            exceptionTrials = exceptionsForSpecificBehaviorDays(animal,day)
            if exceptionTrials is None:
                valid_trials_mask = getActiveTrials(outcomes)
            else:
                valid_trials_mask = exceptionTrials
            trimmed_outcomes = outcomes[valid_trials_mask]
        
        PCt,dpt = pcdpFromOutcomes(trimmed_outcomes)
        
        if PCt >= 0.7 and dpt >= 1.5:
            for d in days_to_check[np.greater_equal(days_to_check,day)]:
                expert_days.append(d)
        elif day != days_to_check[0]:
            pass
        
    ### Exceptions for specific animals
    #if animal == 'BS_40':
    #    expert_days = np.array([17,16])
    if animal == 'BS_42':
        expert_days = np.array([6,5,15,16,17,18,19,20,21,22,23,24,25])
    if animal == 'BS_50':
        expert_days = np.array([11,10])
    if animal == 'AE_239':
        expert_days = np.array([4,5,6,20,21,43])

    expert_days = np.sort(np.unique(expert_days))
    expert_days = expert_days[np.isin(expert_days,phase_days)]
    expert_days = expert_days[np.isin(expert_days,list(animalBehaviors[animal].sessions.keys()))]
    return expert_days

def getExpertDays(animalBehaviors,animal):
    days = np.sort([k for k in animalBehaviors[animal].sessions])
    num_days = np.max(days)#len(days)
    
    reversal = np.Inf
    if hasattr(animalBehaviors[animal],'reversal'):
        reversal = animalBehaviors[animal].reversal
    
    #############################################################
    #                   Find prereversal expert                 #
    #############################################################
    
    pre_expert_days = []
    
    if np.isfinite(reversal):
        days_to_check = np.sort(list(range(1,reversal)))[::-1]
    else:
        days_to_check = np.sort(list(range(1,num_days+1)))[::-1]
        reversal = np.max(days_to_check)+1
    days_to_check = days_to_check[np.isin(days_to_check,list(animalBehaviors[animal].sessions.keys()))]
    for day in days_to_check:
        outcomes = np.array(list(animalBehaviors[animal].sessions[day].outcomes))
        
        exceptionTrials = exceptionsForSpecificBehaviorDays(animal,day)
        if exceptionTrials is None:
            valid_trials_mask = getActiveTrials(outcomes)
        else:
            valid_trials_mask = exceptionTrials
        trimmed_outcomes = outcomes[valid_trials_mask]
        
        PCt,dpt = pcdpFromOutcomes(trimmed_outcomes)
        
        if PCt >= 0.7 and dpt >= 1.5:
            if np.isfinite(reversal):
                [pre_expert_days.append(d) for d in list(range(day,reversal+1))]
            else:
                [pre_expert_days.append(d) for d in list(range(day,np.max(days_to_check)+1))]
        elif day != days_to_check[0]:
            pass#break
    #if len(pre_expert_days)>0:
    #    pre_expert_days = np.concatenate((pre_expert_days))
    pre_expert_days = np.sort(np.unique(pre_expert_days))[::-1]
    pre_expert_days = pre_expert_days[np.isin(pre_expert_days,list(animalBehaviors[animal].sessions.keys()))]
        
    ############### Exceptions for specific animals ##################
    
    if animal == 'BS_40':
        pre_expert_days = np.array([16]) #Remove day 17
    if animal == 'BS_42':
        pre_expert_days = np.array([6,5])
    if animal == 'BS_50':
        pre_expert_days = np.array([11,10])
        
    #############################################################
    #                  Find postreversal expert                 #
    #############################################################
    
    if not np.isfinite(reversal):
        return pre_expert_days,[]
    
    post_expert_days = []
    days_to_check = np.sort(list(range(reversal+1,num_days+1)))[::-1]
    days_to_check = days_to_check[np.isin(days_to_check,list(animalBehaviors[animal].sessions.keys()))]
    for day in days_to_check:
        outcomes = np.array(list(animalBehaviors[animal].sessions[day].outcomes))
        valid_trials_mask = getActiveTrials(outcomes)
        trimmed_outcomes = outcomes[valid_trials_mask]
        
        PCt,dpt = pcdpFromOutcomes(trimmed_outcomes)
        
        if PCt >= 0.7 and dpt >= 1.5:
            #post_expert_days.append(day)
            #if day == days_to_check[1]:
            [post_expert_days.append(d) for d in list(range(day,num_days+1))]
                #post_expert_days.append(day+1)
        elif day != days_to_check[0]:
            pass#break
    #if len(post_expert_days)>0:
    #    post_expert_days = np.concatenate((post_expert_days))
    post_expert_days = np.sort(np.unique(post_expert_days))[::-1]
    post_expert_days = post_expert_days[np.isin(post_expert_days,list(animalBehaviors[animal].sessions.keys()))]
        
    ############### Exceptions for specific animals ##################
    
    if animal == 'BS_49':
        post_expert_days = np.array([20,19,18])
    if animal == 'DS_28':
        post_expert_days = np.array([35])
        
    return pre_expert_days,post_expert_days

def getPCDPfromBehavior(animalBehaviors,animal,days,expert,exclude_first_switch=False,noTrim=False):
    if exclude_first_switch:
        days = np.unique(days)
        
    cond = np.full_like(days,np.nan,dtype='float')
    cond_pc = np.full_like(days,np.nan,dtype='float')
    if not animal in animalBehaviors:
        return cond,cond_pc
    
    reversal = np.Inf
    if hasattr(animalBehaviors[animal],'reversal'):
        reversal = animalBehaviors[animal].reversal
    second_reversal = np.Inf
    if hasattr(animalBehaviors[animal],'second_reversal'):
        second_reversal = animalBehaviors[animal].second_reversal
        
    #We will remove the duplicate reversal day because we will calculate both
    #pre and post reversal when we get there at once
    days = np.unique(days)
    days = days[np.isin(days,list(animalBehaviors[animal].sessions.keys()))]
    
        
    for idx,day in enumerate(days):
        if not day in animalBehaviors[animal].sessions:
            continue
            
        #We still want to put the post-reversal days in the right place, so we have to account
        #for the double reversal day
        if not exclude_first_switch:
            if hasattr(animalBehaviors[animal],'reversal') and day > reversal and reversal in days:
                idx+=1
            if hasattr(animalBehaviors[animal],'reversal') and day > second_reversal and second_reversal in days:
                idx+=1
            
        ########## Pre/Postreversal ##########
        if day != reversal:
            outcomes = np.array(list(animalBehaviors[animal].sessions[day].outcomes))

            if not noTrim:
                #We will trim for active trials only on expert days
                exceptionTrials = exceptionsForSpecificBehaviorDays(animal,day)
                if not exceptionTrials is None:
                    pass#print(f"Trimming via exception trials")
                    valid_trials_mask = exceptionTrials
                    outcomes = outcomes[valid_trials_mask]
                elif day in expert:
                    valid_trials_mask = getActiveTrials(outcomes)
                    outcomes = outcomes[valid_trials_mask]

            if animal == 'AE_239' and day == 39:
                print(f"there are {len(outcomes)} trials in {animal} day {day}")

            hit = np.sum(np.equal(outcomes,1))
            miss = np.sum(np.equal(outcomes,2))
            falarm = np.sum(np.equal(outcomes,3))
            creject = np.sum(np.equal(outcomes,4))

            pc = (hit+creject)/(hit+miss+falarm+creject)
            dp = norm.ppf((hit+1)/(hit+miss+2)) - norm.ppf((falarm+1)/(falarm+creject+2))

            if animal == 'AE_239' and day == 20:
                print(f"there are {len(outcomes)} trials in {animal} day {day}")
                print(f"PC is {pc} and DP is {dp}")

            cond[idx] = dp
            cond_pc[idx] = pc
            
        ########## Switch ##########
        if day == reversal or day == second_reversal:
            
            trials_per_session = animalBehaviors[animal].sessions[day].trials_per_session
            pre_reversal_trials = range(int(trials_per_session[0]))
            post_reversal_trials = range(int(trials_per_session[0]),int(np.sum(trials_per_session)))
            
            if exclude_first_switch == False:
                #Prereversal
                outcomes = np.array(list(animalBehaviors[animal].sessions[day].outcomes))
                outcomes = outcomes[pre_reversal_trials]
                hit = np.sum(np.equal(outcomes,1))
                miss = np.sum(np.equal(outcomes,2))
                falarm = np.sum(np.equal(outcomes,3))
                creject = np.sum(np.equal(outcomes,4))
                pc = (hit+creject)/(hit+miss+falarm+creject)
                dp = norm.ppf((hit+1)/(hit+miss+2)) - norm.ppf((falarm+1)/(falarm+creject+2))
                
                cond[idx] = dp
                cond_pc[idx] = pc
                # firstreversaldp = dp
                # firstreversalpc = pc
            
            #Postreversal
            outcomes = np.array(list(animalBehaviors[animal].sessions[day].outcomes))
            outcomes = outcomes[post_reversal_trials]
            hit = np.sum(np.equal(outcomes,1))
            miss = np.sum(np.equal(outcomes,2))
            falarm = np.sum(np.equal(outcomes,3))
            creject = np.sum(np.equal(outcomes,4))
            pc = (hit+creject)/(hit+miss+falarm+creject)
            dp = norm.ppf((hit+1)/(hit+miss+2)) - norm.ppf((falarm+1)/(falarm+creject+2))
            if exclude_first_switch == False:
                cond[idx+1] = dp
                cond_pc[idx+1] = pc
            else:
                cond[idx] = dp
                cond_pc[idx] = pc
            
    return cond,cond_pc

def calculatePhasesPerStage(animalBehaviors,animal,phase_days,forced_minimum = None,noTrim=False):
    phase_days = np.array(phase_days)
    phase_days = phase_days[np.isin(phase_days,list(animalBehaviors[animal].sessions.keys()))]
    
    expert_days = getExpertDaysPerStage(animalBehaviors,animal,phase_days,noTrim=noTrim)
    dp_list,pc_list = getPCDPfromBehavior(animalBehaviors,animal,phase_days,expert_days,noTrim=noTrim)
    
    dp_min = min(np.min(dp_list),0)
    if not forced_minimum is None:
        dp_min = forced_minimum
    dp_max = max(np.max(dp_list),1.5)
    dp_thresh = dp_min + 0.3 * (dp_max - dp_min)
    
    late_days_mask = np.greater_equal(dp_list,dp_thresh)
    for idx in range(len(late_days_mask)):
        if np.any(late_days_mask[range(idx)]):
            late_days_mask[idx] = True
    late_days = phase_days[late_days_mask] #+1 because 1-indexed
    late_days = late_days[np.logical_not(np.isin(late_days,expert_days))]
    
    early_days = np.copy(phase_days)
    early_days = early_days[np.logical_not(np.isin(early_days,late_days))]
    early_days = early_days[np.logical_not(np.isin(early_days,expert_days))]

    days_to_remove = []
    days_to_make_late = []
    if animal == 'BS_40':
        days_to_remove=[17]
    if animal == 'BS_163':
        days_to_remove=[12]
    if animal == 'AE_252':
        days_to_make_late=[15,16,17,18,19,20,21,23,24,25]
    if animal == 'AO_273':
        days_to_remove=[17,20,21]
    if animal == 'AO_274':
        days_to_remove=[10,11,12,13,16]

    early_days = early_days[np.logical_not(np.isin(early_days,days_to_remove))]
    late_days = late_days[np.logical_not(np.isin(late_days,days_to_remove))]
    expert_days = expert_days[np.logical_not(np.isin(expert_days,days_to_remove))]

    days_to_make_late = np.array(days_to_make_late)
    if len(days_to_make_late) > 0:
        early_days = early_days[np.logical_not(np.isin(early_days,days_to_make_late))]
        expert_days = expert_days[np.logical_not(np.isin(expert_days,days_to_make_late))]
        days_to_add = days_to_make_late[np.isin(days_to_make_late,phase_days)]
        if len(late_days) > 0 and len(days_to_add) > 0:
            print(late_days)
            print(days_to_add)
            late_days = np.concatenate((np.array(late_days),np.array(days_to_add)))
            late_days = np.unique(late_days)
        else:
            late_days = days_to_make_late
    
    return early_days,late_days,expert_days

def calculatePhasesPerAnimal(animalBehaviors,animal,noTrim=False,noTrimPost=False):
    max_day_number = np.max(list(animalBehaviors[animal].sessions.keys()))
    second_reversal_post_dp = None
    if hasattr(animalBehaviors[animal],'second_reversal') and np.isfinite(animalBehaviors[animal].second_reversal):
        second_reversal_post_dp,_ = getPCDPfromBehavior(animalBehaviors,animal,[animalBehaviors[animal].second_reversal]*2,[])
        second_reversal_post_dp = second_reversal_post_dp[1]
    reversal_post_dp = None
    if hasattr(animalBehaviors[animal],'reversal') and np.isfinite(animalBehaviors[animal].reversal):
        reversal_post_dp,_ = getPCDPfromBehavior(animalBehaviors,animal,[animalBehaviors[animal].reversal]*2,[])
        reversal_post_dp = reversal_post_dp[1]
    
    animalPhases = SimpleNamespace()
    if hasattr(animalBehaviors[animal],'second_reversal') and np.isfinite(animalBehaviors[animal].second_reversal) and max_day_number > animalBehaviors[animal].second_reversal:
        second_rev_days = np.arange(animalBehaviors[animal].second_reversal+1,np.max(list(animalBehaviors[animal].sessions.keys()))+1)
        animalPhases.second_early_days,animalPhases.second_late_days,animalPhases.second_expert_days = calculatePhasesPerStage(animalBehaviors,animal,second_rev_days,forced_minimum=second_reversal_post_dp,noTrim=noTrimPost)
        post_rev_days = np.arange(animalBehaviors[animal].reversal+1,animalBehaviors[animal].second_reversal)
        animalPhases.post_early_days,animalPhases.post_late_days,animalPhases.post_expert_days = calculatePhasesPerStage(animalBehaviors,animal,post_rev_days,forced_minimum=reversal_post_dp,noTrim=noTrimPost)
        pre_rev_days = np.arange(1,animalBehaviors[animal].reversal)
        animalPhases.pre_early_days,animalPhases.pre_late_days,animalPhases.pre_expert_days = calculatePhasesPerStage(animalBehaviors,animal,pre_rev_days,noTrim=noTrim)
    elif hasattr(animalBehaviors[animal],'reversal') and np.isfinite(animalBehaviors[animal].reversal) and max_day_number > animalBehaviors[animal].reversal:
        animalPhases.second_early_days,animalPhases.second_late_days,animalPhases.second_expert_days = ([],[],[])
        post_rev_days = np.arange(animalBehaviors[animal].reversal+1,np.max(list(animalBehaviors[animal].sessions.keys()))+1)
        if hasattr(animalBehaviors[animal],'second_reversal'):
            post_rev_days = np.delete(post_rev_days, np.where(post_rev_days == animalBehaviors[animal].second_reversal))
        animalPhases.post_early_days,animalPhases.post_late_days,animalPhases.post_expert_days = calculatePhasesPerStage(animalBehaviors,animal,post_rev_days,forced_minimum=reversal_post_dp,noTrim=noTrimPost)
        pre_rev_days = np.arange(1,animalBehaviors[animal].reversal)
        animalPhases.pre_early_days,animalPhases.pre_late_days,animalPhases.pre_expert_days = calculatePhasesPerStage(animalBehaviors,animal,pre_rev_days,noTrim=noTrim)
    else:
        animalPhases.second_early_days,animalPhases.second_late_days,animalPhases.second_expert_days = ([],[],[])
        animalPhases.post_early_days,animalPhases.post_late_days,animalPhases.post_expert_days = ([],[],[])
        pre_rev_days = np.arange(1,np.max(list(animalBehaviors[animal].sessions.keys()))+1)
        if hasattr(animalBehaviors[animal],'reversal'):
            pre_rev_days = np.delete(pre_rev_days, np.where(pre_rev_days == animalBehaviors[animal].reversal))
        animalPhases.pre_early_days,animalPhases.pre_late_days,animalPhases.pre_expert_days = calculatePhasesPerStage(animalBehaviors,animal,pre_rev_days,noTrim=noTrim)
        
    return animalPhases

def calculateLearningPhasesV2(animals,animalBehaviors,plot=False,noTrim=False,noTrimPost=False,trimLastDayLate=False):
    if noTrim:
        noTrimPost = True

    animalPhases = dict()
    animalPreThresh = dict()
    animalPostThresh = dict()

    ### Set up grid of panels for plotting
    grid_size = int(np.round(np.sqrt(len(animals))+0.5))
    if len(animals) == 1:
        grid_size = 1
    if plot:
        if grid_size > 1:
            fig = plt.figure(figsize=(15,15))
            gs = fig.add_gridspec(grid_size,grid_size,hspace=0.5)
            axs = [plt.subplot(gs[i,j]) for i,j in product(range(grid_size),range(grid_size))]
            for idx in range(len(animals),grid_size**2):
                plt.delaxes(axs[idx])
        else:
            fig,ax = plt.subplots(figsize=(5,4.5))
            axs = [ax]

    ### Calculate learning phases
    for animalidx,animal in enumerate(animals):
        animalPhases[animal] = calculatePhasesPerAnimal(animalBehaviors,animal,noTrim=noTrim,noTrimPost=noTrimPost)
        
        PLOTTING_PHASE_COLORS = PHASE_COLORS
        
        if plot:
            
            reversal = np.nan
            second_reversal = np.nan
            
            days = list(np.unique(list(animalBehaviors[animal].sessions.keys())))
            if hasattr(animalBehaviors[animal],'reversal') and np.isfinite(animalBehaviors[animal].reversal):
                reversal = animalBehaviors[animal].reversal
                days.append(reversal)
            if hasattr(animalBehaviors[animal],'second_reversal') and np.isfinite(animalBehaviors[animal].second_reversal):
                second_reversal = animalBehaviors[animal].second_reversal
                days.append(second_reversal)
            days = np.sort(days)
            #Needs days
            expert_days = np.concatenate(( animalPhases[animal].pre_expert_days,animalPhases[animal].post_expert_days,animalPhases[animal].second_expert_days ))

            ###
            ### TRIM THE LAST DAY OF LATE LEARNING
            ###
            if trimLastDayLate:
                pre_late_days_TEMP = days[np.isin(days,animalPhases[animal].pre_late_days)]
                post_late_days_TEMP = days[np.isin(days,animalPhases[animal].post_late_days)]
                late_last_days = []
                if len(pre_late_days_TEMP) > 0:
                    late_last_days.append(np.max(pre_late_days_TEMP))
                if len(post_late_days_TEMP) > 0:
                    late_last_days.append(np.max(post_late_days_TEMP))
                if len(late_last_days) > 0:
                    print(f"{animal}: adding day {late_last_days}")
                    expert_days = np.concatenate((expert_days,late_last_days))
                else:
                    print(f"{animal}: no last late day")
            ###
            ###
            ###

            dp_list,pc_list = getPCDPfromBehavior(animalBehaviors,animal,days,expert_days)
            #Needs dps
            
            ax = axs[animalidx]    
            #All Days no color
            ax.plot(days,dp_list,color='k',lw=1)
            
            #Pre Early
            pre_early = np.isin(days,animalPhases[animal].pre_early_days)
            ax.scatter(days[pre_early],dp_list[pre_early],color=PLOTTING_PHASE_COLORS[0],s=20,zorder=10)
            
            #Pre Late
            pre_late = np.isin(days,animalPhases[animal].pre_late_days)
            ax.scatter(days[pre_late],dp_list[pre_late],color=PLOTTING_PHASE_COLORS[1],s=20,zorder=10)
            #ax.axhline(animalPreThresh[animal],c=PHASE_COLORS[1],linestyle='--',lw=1,zorder=-20)
            
            #Pre Expert
            pre_expert = np.isin(days,animalPhases[animal].pre_expert_days)
            ax.scatter(days[pre_expert],dp_list[pre_expert],color=PLOTTING_PHASE_COLORS[2],s=20,zorder=10)
            
            #Post Early
            post_early = np.isin(days,animalPhases[animal].post_early_days)
            ax.scatter(days[post_early],dp_list[post_early],color=PLOTTING_PHASE_COLORS[3],s=20,zorder=10)
            
            #Post Late
            post_late = np.isin(days,animalPhases[animal].post_late_days)
            ax.scatter(days[post_late],dp_list[post_late],color=PLOTTING_PHASE_COLORS[4],s=20,zorder=10)
            #ax.axhline(animalPostThresh[animal],c=PHASE_COLORS[4],linestyle='--',lw=1,zorder=-20)
            
            #Post Expert
            post_expert = np.isin(days,animalPhases[animal].post_expert_days)
            ax.scatter(days[post_expert],dp_list[post_expert],color=PLOTTING_PHASE_COLORS[5],s=20,zorder=10)
            
            #Switch
            switch = np.equal(days,animalBehaviors[animal].reversal)
            #ax.scatter(days[switch],dp_list[switch],color=SWITCH_COLOR,s=20,zorder=10)
            #ax.scatter(days[switch],dp_list[switch],color='b',s=20,zorder=10)
            if np.sum(switch)>0:
                ax.scatter(days[switch][1],dp_list[switch][1],color=SWITCH_COLOR,s=20,zorder=10)
            
            #Switch
            second_switch = np.equal(days,animalBehaviors[animal].second_reversal)
            if np.sum(second_switch)>0:
                ax.scatter(days[second_switch],dp_list[second_switch],color=SWITCH_COLOR,s=20,zorder=10)
            #ax.scatter(days[switch],dp_list[switch],color='b',s=20,zorder=10)
            
            #Second Early
            second_early = np.isin(days,animalPhases[animal].second_early_days)
            ax.scatter(days[second_early],dp_list[second_early],color=PLOTTING_PHASE_COLORS[0],s=20,zorder=10)
            
            #Post Late
            second_late = np.isin(days,animalPhases[animal].second_late_days)
            ax.scatter(days[second_late],dp_list[second_late],color=PLOTTING_PHASE_COLORS[1],s=20,zorder=10)
            #ax.axhline(animalPostThresh[animal],c=PHASE_COLORS[4],linestyle='--',lw=1,zorder=-20)
            
            #Post Expert
            second_expert = np.isin(days,animalPhases[animal].second_expert_days)
            ax.scatter(days[second_expert],dp_list[second_expert],color=PLOTTING_PHASE_COLORS[2],s=20,zorder=10)

            ax.set_ylim([-2,3])
            
            ax.text(reversal,ax.get_ylim()[1],'Reversal',color=SWITCH_COLOR,horizontalalignment='center',verticalalignment='top')
            ax.axvline(reversal,c=SWITCH_COLOR,linestyle='--',lw=1,zorder=-10)

            ax.text(second_reversal,ax.get_ylim()[1],'Reversal',color=SWITCH_COLOR,horizontalalignment='center',verticalalignment='top')
            ax.axvline(second_reversal,c=SWITCH_COLOR,linestyle='--',lw=1,zorder=-10)
            
            ax.axhline(1.5,c='r',linestyle='--',lw=1,zorder=-10)
            
            #days_recorded = training_days_recorded[animal]
            #ax.scatter(days_recorded,np.ones_like(days_recorded)*2.9,s=5,marker='*',color='orange',zorder=11)
            ax.set_title(animal)
            
            #days_recorded = training_days_recorded[animal]
            #ax.scatter(days_recorded,np.ones_like(days_recorded)*2.9,s=5,marker='*',color='orange',zorder=11)

    if plot:            
        if grid_size > 1:
            fig.text(0.075, 0.5, 'Behavioral performance (d\')', ha='center', va='center', rotation='vertical',fontsize=20)
            fig.text(0.5,0.075, 'Day of training', ha='center', va='center', rotation='horizontal',fontsize=20)
        else:
            ax.set_xlabel('Day of training')
            ax.set_ylabel('Behavioral performance (d\')')
        #plt.savefig(os.path.join('D:\\\\TempFigures','Automated Learning Phases 2.pdf'),transparent=False,facecolor="white")
        #plt.savefig(os.path.join('D:\\\\TempFigures','Automated Learning Phases 2 Danimals.pdf'),transparent=False,facecolor="white")
        #plt.savefig(os.path.join('D:\\\\TempFigures','Automated Learning Phases 2 Tuning Animals.pdf'),transparent=False,facecolor="white")
        pass

    return animalPhases,animalPreThresh,animalPostThresh

def calculateLearningPhases(animals,animalBehaviors,plot=False):
    animalPhases = dict()
    animalPreThresh = dict()
    animalPostThresh = dict()

    grid_size = int(np.round(np.sqrt(len(animals))+0.5))
    if plot:
        fig = plt.figure(figsize=(15,15))
        gs = fig.add_gridspec(grid_size,grid_size,hspace=0.5)
        axs = [plt.subplot(gs[i,j]) for i,j in product(range(grid_size),range(grid_size))]
        for idx in range(len(animals),grid_size**2):
            plt.delaxes(axs[idx])

    for animalidx,animal in enumerate(animals):
        
        animalPhases[animal] = SimpleNamespace()
        
        reversal = np.Inf
        if hasattr(animalBehaviors[animal],'reversal'):
            reversal = animalBehaviors[animal].reversal
        
        days = [k for k in animalBehaviors[animal].sessions]
        if np.isfinite(reversal):
            days.append(reversal)
        days = np.sort(days)
        
        pre_exp,post_exp = getExpertDays(animalBehaviors,animal)
        expert_days = np.concatenate((pre_exp,post_exp))
        dp_list,pc_list = getPCDPfromBehavior(animalBehaviors,animal,days,expert_days)
                
        ############### Calculation of learning phases ###################
        
        pre_reversal_days_mask = np.less(days,reversal)
        post_reversal_days_mask = np.greater(days,reversal)
        
        pre_reversal_dp_list = dp_list[pre_reversal_days_mask]
        post_reversal_dp_list = dp_list[post_reversal_days_mask]
        if np.isfinite(reversal):
            pre_switch_dp = dp_list[reversal-1]
            post_switch_dp = dp_list[reversal]
        else:
            pre_switch_dp = dp_list[len(dp_list)-1]
            post_switch_dp = []
        
        pre_rev_dp_concat = pre_reversal_dp_list
        pre_rev_min = min(np.min(pre_reversal_dp_list),0)
        pre_rev_max = max(np.max(pre_reversal_dp_list),1.5)
        pre_reversal_dp_thresh = pre_rev_min + 0.3 * (pre_rev_max - pre_rev_min)
        post_rev_dp_concat = pre_reversal_dp_list
        try:
            post_rev_min = post_switch_dp
            post_rev_max = max(np.max(post_reversal_dp_list),1.5)
            post_reversal_dp_thresh = post_rev_min + 0.3 * (post_rev_max - post_rev_min)
        except:
            post_rev_min = np.nan
            post_rev_max = np.nan
            post_reversal_dp_thresh = np.Inf
            
        animalPreThresh[animal] = pre_reversal_dp_thresh
        animalPostThresh[animal] = post_reversal_dp_thresh

        pre_late_days_mask = np.greater_equal(pre_reversal_dp_list,pre_reversal_dp_thresh)
        post_late_days_mask = np.greater_equal(post_reversal_dp_list,post_reversal_dp_thresh)
        
        ############ Export final results to a list of days ############
        
        pre_reversal_days = np.where(pre_reversal_days_mask)[0] + 1
        post_reversal_days = np.where(post_reversal_days_mask)[0] # No +1 because double reversal day
        
        pre_expert_days = np.sort(pre_exp)
        post_expert_days = np.sort(post_exp)
        
        pre_late_days = np.where(pre_late_days_mask)[0] + 1
        post_late_days = np.where(post_late_days_mask)[0] + 1 + reversal # No +1 because adding reversal. +1 because double reversal
        
        if len(pre_late_days)>0:
            last_day_late = reversal
            if not np.isfinite(reversal):
                last_day_late = np.max(days)
            pre_late_days = np.arange(np.min(pre_late_days),last_day_late)
        if len(post_late_days)>0:
            post_late_days = np.arange(np.min(post_late_days),max(days))
        pre_late_days = pre_late_days[np.logical_not(np.isin(pre_late_days,pre_expert_days))]
        post_late_days = post_late_days[np.logical_not(np.isin(post_late_days,post_expert_days))]
        
        pre_early_days = pre_reversal_days[np.logical_not(np.isin(pre_reversal_days,pre_late_days))]
        post_early_days = post_reversal_days[np.logical_not(np.isin(post_reversal_days,post_late_days))]
        pre_early_days = pre_early_days[np.logical_not(np.isin(pre_early_days,pre_expert_days))]
        post_early_days = post_early_days[np.logical_not(np.isin(post_early_days,post_expert_days))]
            
        #Handle one particular exception for BS_50 due to animal not behaving
        if animal == 'BS_50':
            pre_early_days = list(pre_early_days)
            pre_late_days = list(pre_late_days)
            pre_late_days.remove(5)
            #I have verified that day is is below the pre-threshold
            pre_late_days.remove(6)
            pre_early_days.append(6)
            pre_early_days = np.sort(pre_early_days)
            pre_late_days = np.sort(pre_late_days)
        if animal == 'BS_173':
            pre_expert_days = np.array([7,9,10,11])
        if animal == 'BS_175':
            pre_expert_days = np.array([4,6,7,9,10,11])
            
        animalPhases[animal].pre_early_days = pre_early_days
        animalPhases[animal].pre_late_days = pre_late_days
        animalPhases[animal].pre_expert_days = pre_expert_days
        if np.isfinite(reversal):
            animalPhases[animal].switch_days = [reversal]
        else:
            animalPhases[animal].switch_days = []
        animalPhases[animal].post_early_days = post_early_days
        animalPhases[animal].post_late_days = post_late_days
        animalPhases[animal].post_expert_days = post_expert_days
        
        #PLOTTING_PHASE_COLORS = ['k','r','g']*2
        PLOTTING_PHASE_COLORS = PHASE_COLORS
        
        if plot:
            ax = axs[animalidx]    
            #All Days no color
            ax.plot(days,dp_list,color='k',lw=1)
            
            #Pre Early
            pre_early = np.isin(days,pre_early_days)
            ax.scatter(days[pre_early],dp_list[pre_early],color=PLOTTING_PHASE_COLORS[0],s=20,zorder=10)
            
            #Pre Late
            pre_late = np.isin(days,pre_late_days)
            ax.scatter(days[pre_late],dp_list[pre_late],color=PLOTTING_PHASE_COLORS[1],s=20,zorder=10)
            #ax.axhline(animalPreThresh[animal],c=PHASE_COLORS[1],linestyle='--',lw=1,zorder=-20)
            
            #Pre Expert
            pre_expert = np.isin(days,pre_expert_days)
            ax.scatter(days[pre_expert],dp_list[pre_expert],color=PLOTTING_PHASE_COLORS[2],s=20,zorder=10)
            
            #Post Early
            post_early = np.isin(days,post_early_days)
            ax.scatter(days[post_early],dp_list[post_early],color=PLOTTING_PHASE_COLORS[3],s=20,zorder=10)
            
            #Post Late
            post_late = np.isin(days,post_late_days)
            ax.scatter(days[post_late],dp_list[post_late],color=PLOTTING_PHASE_COLORS[4],s=20,zorder=10)
            #ax.axhline(animalPostThresh[animal],c=PHASE_COLORS[4],linestyle='--',lw=1,zorder=-20)
            
            #Post Expert
            post_expert = np.isin(days,post_expert_days)
            ax.scatter(days[post_expert],dp_list[post_expert],color=PLOTTING_PHASE_COLORS[5],s=20,zorder=10)
            
            #Switch
            switch = np.equal(days,reversal)
            ax.scatter(days[switch],dp_list[switch],color=SWITCH_COLOR,s=20,zorder=10)
            #ax.scatter(days[switch],dp_list[switch],color='b',s=20,zorder=10)

            ax.set_ylim([-2,3])
            
            ax.text(reversal,ax.get_ylim()[1],'Reversal',color=SWITCH_COLOR,horizontalalignment='center',verticalalignment='top')
            ax.axvline(reversal,c=SWITCH_COLOR,linestyle='--',lw=1,zorder=-10)
            
            ax.axhline(1.5,c='r',linestyle='--',lw=1,zorder=-10)
            
            #days_recorded = training_days_recorded[animal]
            #ax.scatter(days_recorded,np.ones_like(days_recorded)*2.9,s=5,marker='*',color='orange',zorder=11)
            ax.set_title(animal)
            
            #days_recorded = training_days_recorded[animal]
            #ax.scatter(days_recorded,np.ones_like(days_recorded)*2.9,s=5,marker='*',color='orange',zorder=11)
            
    fig.text(0.075, 0.5, 'Behavioral performance (d\')', ha='center', va='center', rotation='vertical',fontsize=20)
    fig.text(0.5,0.075, 'Day of training', ha='center', va='center', rotation='horizontal',fontsize=20)

    if plot:
        #plt.savefig(os.path.join('D:\\\\TempFigures','Automated Learning Phases 2.pdf'),transparent=False,facecolor="white")
        #plt.savefig(os.path.join('D:\\\\TempFigures','Automated Learning Phases 2 Danimals.pdf'),transparent=False,facecolor="white")
        #plt.savefig(os.path.join('D:\\\\TempFigures','Automated Learning Phases 2 Tuning Animals.pdf'),transparent=False,facecolor="white")
        pass

    return animalPhases,animalPreThresh,animalPostThresh































#All days with exceptions are expert days where the trimming algorithm misbehaves with the exception of those described
def exceptionsForSpecificBehaviorDays(animal,day):
    trials = None

    if animal == 'DS_15' and day == 11:
        trials = np.arange(300,300+100+67)
    if animal == 'DS_15' and day == 23:
        trials = np.arange(0,400) #Note that the performances listed in the drive are not actually correct
    if animal == 'DS_15' and day == 25:
        trials = np.arange(300+100+100,300+100+100+49)

    if animal == 'DS_19' and day == 5:
        trials = np.arange(300+100,300+100+100)
    if animal == 'DS_19' and day == 8:
        trials = np.arange(0,300+100)

    if animal == 'DS_28' and day == 35:
        trials = np.arange(99+68,99+68+100+100+100)

    if animal == 'BS_40' and day == 10:
        trials = np.arange(0,599) #Unsure why this one is needed
    if animal == 'BS_40' and day == 16:
        trials = np.arange(235+112,235+112+53)
    if animal == 'BS_40' and day == 17:
        trials = np.arange(0,350)
        trials = np.arange(235+112,235+112+53)
    if animal == 'BS_40' and day == 31:
        trials = np.arange(300,300+50+92)  #Unclear if this is correct

    if animal == 'BS_41' and day == 5:
        trials = np.arange(0,300)
    if animal == 'BS_41' and day == 18:
        trials = np.arange(300,300+100)

    if animal == 'BS_42' and day == 5:
        trials = np.arange(0,600)
    if animal == 'BS_42' and day == 6:
        trials = np.arange(459,459+100)
    if animal == 'BS_42' and day == 19:
        trials = np.arange(0,300)

    #Pre-reversal days. Excluded because experimentalists specifically note poor motivation for these animals during these days but did not cut trials
    if animal == 'BS_49' and day == 5:
        trials = np.arange(0,100)
    if animal == 'BS_49' and day == 6:
        trials = np.arange(64+73,64+73+75)
    if animal == 'BS_49' and day == 7:
        trials = np.arange(0,52)
    if animal == 'BS_49' and day == 8:
        trials = np.arange(0,52)

    if animal == 'BS_49' and day == 12:
        trials = np.arange(300,300+51)
    if animal == 'BS_49' and day == 13:
        trials = np.arange(300,300+100+65)
    if animal == 'BS_49' and day == 18:
        trials = np.arange(191,191+61)
    if animal == 'BS_49' and day == 19:
        trials = np.arange(0,300)

    if animal == 'BS_50' and day == 2:
        trials = np.arange(0,129)#Shouldn't be needed
    if animal == 'BS_50' and day == 5:
        #Exclude day
        #trials = np.array([])
        pass#trials = np.arange(175,171+100+71)
    if animal == 'BS_50' and day == 11:
        trials = np.arange(175,171+100+71)

    if animal == 'BS_51' and day == 16:
        trials = np.arange(300+100,300+100+100+100+26+64)

    if animal == 'BS_56' and day == 4:
        trials = np.arange(400+34,400+34+100+61)
    if animal == 'BS_56' and day == 5:
        trials = np.arange(380,380+67)

    if animal == 'BS_59' and day == 3:
        trials = np.arange(331,331+48+52)
    if animal == 'BS_59' and day == 11:
        trials = np.arange(300+100+100,300+100+100+100+100)
    if animal == 'BS_59' and day == 18:
       trials = np.arange(0,300+68)

    if animal == 'BS_70' and day == 5:
        trials = np.arange(300,300+100+73)

    if animal == 'BS_86' and day == 8:
        trials = np.arange(0,356)

    #Pre-reversal days. Excluded because they hit expert when trimmed but experimentalists do not note expert behavior on any session. Likely overtrimmed
    if animal == 'BS_87' and day == 2:
        trials = np.arange(0,300+100+64)
    if animal == 'BS_87' and day == 3:
        trials = np.arange(0,400)
    if animal == 'BS_87' and day == 11:
        trials = np.arange(0,300+58)

    if animal == 'BS_87' and day == 14:
        trials = np.arange(300+100,300+100+77)

    if animal == 'BS_119' and day == 8:
        trials = np.arange(300,300+100+100)

    if animal == 'BS_163' and day == 2:
        trials = np.arange(0,300+100+75)
    if animal == 'BS_163' and day == 9:
        trials = np.arange(0,300+51+100)
    if animal == 'BS_163' and day == 11:
        pass#trials = np.arange(300,300+51+100)
    #if animal == 'BS_163' and day == 12:       #This session is missing from the drive, so I will remove it for now
    #    trials = np.arange(0,247)

    if animal == 'BS_173' and day == 8:
        pass#session removed
    if animal == 'BS_173' and day == 12:
        pass#session removed
    if animal == 'BS_173' and day == 21:
        trials = np.arange(0,273)

    if animal == 'BS_174' and day == 7:
        trials = np.arange(0,300+100+51)
    if animal == 'BS_174' and day == 8:
        trials = np.arange(0,300)
    if animal == 'BS_174' and day == 10:
        trials = np.arange(300,300+100+57+50)
    if animal == 'BS_174' and day == 12:
        trials = np.arange(0,300)

    if animal == 'BS_175' and day == 5:
        #trials = np.arange(0,273)
        pass
    if animal == 'BS_175' and day == 8:
        #trials = np.arange(0,273)
        pass

    if animal == 'AE_236' and day == 9:
        trials = np.arange(127,127+63+149)
    if animal == 'AE_236' and day == 10:
        trials = np.arange(0,200)

    #Decoding related trimming
    if animal == 'BS_40' and day == 31:###
        trials = np.arange(300+50,300+50+92)
    if animal == 'BS_40' and day == 32:###
        trials = np.arange(300,300+165)
    if animal == 'BS_41' and day == 17:
        trials = np.arange(0,493)#NOTE THIS DOESN'T ALIGN WITH DRIVE!!!
    if animal == 'BS_56' and day == 16:
        trials = np.arange(300,300+100+82)
    if animal == 'BS_56' and day == 17:
        trials = np.arange(300,300+100+90)
    if animal == 'BS_56' and day == 18:
        trials = np.arange(0,300+100)
    if animal == 'BS_67' and day == 13:
        trials = np.arange(300,300+100+100+49)
    if animal == 'BS_87' and day == 14:
        trials = np.arange(300,300+100+77)
    if animal == 'BS_87' and day == 15:
        trials = np.arange(300+100,300+100+100+52)

    # if animal == 'AE_238' and day == 17:
    #     trails = np.arange(200,200+200)
    if animal == 'AE_238' and day == 39:
        trials = np.arange(200,200+200+200)
    if animal == 'AE_238' and day == 40:
        trials = np.arange(200,200+200+200)

    if animal == 'AE_239' and day == 5:
        trials = np.arange(200+32+160+200,200+32+160+200+144)
    if animal == 'AE_239' and day == 20:
        trials = np.arange(200,200+116)
    if animal == 'AE_239' and day == 39:
        trials = np.arange(0,200+198)
    if animal == 'AE_239' and day == 43:
        trials = np.arange(200,200+200+100) #OCL noted expert behavior in this session, but last 45 trials are empty of behavior, reducing dp. Trimmed last 45 trials from the set of trials included by OCL on drive (100 vs 145)

    # Trimming to match the available recorded data (drift event I think. Should double check)
    if animal == 'BS_128' and day == 7:
        trials = np.arange(0,300)

    if animal == 'AO_273' and day == 19:
        trials = np.arange(0,180)

    if animal == 'AE_367' and day == 2: #Experimentalist notes animal quit early
        trials = np.arange(0,92+64)

    return trials














def getAllBehavior(beh_directory):
    donotinclude = ['BS_108_1v1_opto.txt','BS_108_1v2_opto.txt','BS_108_1v3_opto.txt','BS_108_1v4_opto.txt']
    makenolaser = ['BS_86_2v1_opto.txt','BS_86_2v2_opto.txt','BS_86_3v1_opto.txt','BS_86_3v2_opto.txt','BS_86_4v1_opto.txt','BS_86_4v2_opto.txt','BS_86_5v1_opto.txt','BS_86_5v2_opto.txt','BS_86_6v1_opto.txt','BS_86_6v2_opto.txt','BS_86_6v3_opto.txt','BS_83_11v1_opto.txt','BS_83_11v2_opto.txt','BS_83_11v3_opto.txt']

    beh_dualnames = ['BS_41','BS_42','BS_49','BS_67','BS_70','BS_72','BS_87','BS_95','BS_108','BS_113']
    post_exp_animals = ['BS_40','BS_41','BS_42','BS_49','BS_51','BS_56','BS_59','BS_61','BS_67','BS_70','BS_72','BS_87','BS_108']
    danimals = ['DS_15','DS_16','DS_19','DS_24']

    animalsMask = {
        #Naive
        'BS_51':None,
        'BS_52':None,
        'BS_56':None,
        'BS_59':None,
        'BS_61':None,
        
        #Opsin
        'BS_86':None,
        'BS_92':None,
        'BS_100':None,
        'BS_103':None,
        'BS_111':None,
        'BS_119':None,
        'BS_123':None,
        'BS_128':None,
        'BS_131':None,
        'BS_139':None,
        'BS_163':None,
        'BS_165':None,
        'BS_174':None,
        'BS_179':None,
        'BS_191':None,
        'BS_192':None,
        
        #Dual Recording
        'BS_33':None,
        'BS_67':None,
        'BS_73':None,
        'BS_78':None,
        'BS_108':None,
        'BS_40':None,
        'BS_41':None,
        'BS_42':None,
        'BS_49':None,
        'BS_50':None,
        'BS_70':None,
        'BS_72':None,
        'BS_83':None,
        'BS_85':None,
        'BS_87':None,
        'BS_95':None,
        'BS_113':None,
        
        #Dan
        'DS_15':None,
        'DS_16':None,
        'DS_17':None,
        'DS_19':None,
        'DS_22':None,
        'DS_23':None,
        'DS_24':None,
        'DS_27':None,
        'DS_28':None,
        'DS_17':None,
        'DS_22':None,
        'DS_13':None,
        'DS_23':None,

        #Tuning
        'BS_173':None,
        'BS_175':None,
        'BS_187':None,
        'BS_188':None,
        'BS_213':None,
        'BS_214':None,
        'TH_217':None,
        'AE_235':None,
        'AE_236':None,
        'TH_237':None,
        'AE_252':None,
        'AE_254':None,

        #Second Reversal
        'AE_238':None,
        'AE_239':None,
        'AE_240':None,

        #Opsin Control
        'AE_267':None,
        'AO_273':None,
        'AO_274':None,#14,17
        'AE_287':None,
        'AE_301':None,
        'AE_312':None,

        #Pre opsin
        'AE_344':None,
        'AE_346':None,
        'AE_367':None,
        #Pre opsin control
        'AE_350':None,
        'AE_351':None,
        'AE_359':None,
        
    }

    cloudiness_start_day = {
        'BS_86':20, #Was 19. Appears to be 20 based on drive
        'BS_92':15,
        'BS_100':15, #Was listed as 22. Don't know where that number came from. Have listed 15 as the first day when gliosis became too intense to record from
        'BS_103':12, #Day 12 is when the window reached 3/10 visibility. Blake said it was "slightly cloudy" on day 10 though
        'BS_111':np.Inf, #Window quality last recorded on day 19
        'BS_119':np.Inf, #Window quality last recorded on day 15
        'BS_123':20,
        'BS_128':14, #Window reaches 5/10 quality on day 14. Reaches 6/10 on day 8 and 7 on day 6
        'BS_131':np.Inf, #Reaches 6/10 clarity on day 22 and 7/10 on day 16
        'BS_139':np.Inf,
        'BS_163':19, #6/10 quality on day 19, 7/10 quality on day 17
        'BS_165':13, #7/10 quality on day 13 in ephys log, 5/10 in behavior log.
        'BS_174':24, #5/10 window quality on day 24, 6/10 on day 20
        'BS_179':21, #5/10 window quality on day 21, 6/10 on day 19
        'BS_191':np.Inf, #Window quality not recorded until retirement, but 7/10 a few days prior so presumably fine
        'BS_192':16  #Window 5/10 on day 16, 6/10 on day 13
    }

    allnames = [k for k in animalsMask]
    animalnames = allnames
    #animalBehaviors = getAllSessions(animalnames,beh_directory,donotinclude=donotinclude,mask=animalsMask)
    animalBehaviors = getAllSessions(animalnames,beh_directory,donotinclude=donotinclude,mask=None)
    animalBehaviors = loadAllSessions(animalBehaviors,beh_directory,makenolaser=makenolaser)
    #animalBehaviors = calculateEILPhase(animalBehaviors)
    animalBehaviors = handleCloudiness(animalBehaviors,cloudiness_start_day)
    #animalBehaviors = handleDanAnimals(animalBehaviors)
    return animalBehaviors





def handleCloudiness(animalBehaviors,cloudiness_start_day):
    for animal in animalBehaviors:
        if animal in cloudiness_start_day:
            animalBehaviors[animal].cloudiness = cloudiness_start_day[animal]
    return animalBehaviors















def getAllSessions(animalnames,directory,donotinclude=[],mask=None,verbose=False):
    animalBehaviors = dict()
    for name in animalnames:
        print('Fetching files for '+name)

        animaldirectory = os.path.join(directory,name,'behavior')
        diritems = os.listdir(animaldirectory)

        animalBehavior = SimpleNamespace()
        animalBehavior.ID = name
        animalBehavior.reversal = np.Inf
        animalBehavior.second_reversal = np.Inf
        animalBehavior.sessions = dict()

        #Get all valid behavior files in the directory
        itemstoremove = []
        for item in diritems:
            if item[(len(item)-13):len(item)] == '.txtlicks.txt' or item[(len(item)-4):len(item)] == '.xls' or item in donotinclude:
                itemstoremove.append(item)
        behaviorfiles = [x for x in diritems if x not in itemstoremove]

        #Get all sessions
        sessionnumbers = []
        for file in behaviorfiles:
            sessionstart = len(name)+1
            sessionend = file[sessionstart:len(file)].find('v')

            sessionnumber = int(file[sessionstart:(sessionstart+sessionend)])
            sessionnumbers.append(sessionnumber)
            #print(sessionnumber)
        sessionnumbers = np.sort(np.unique(sessionnumbers))

        #Ephys
        if name == 'DS_19':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,30)]
        if name == 'DS_27':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,29)]
        if name == 'DS_28':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,35)]
        if name == 'BS_41':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,16)]###
        if name == 'BS_49':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,18)]
        if name == 'BS_50':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,12)]
        if name == 'BS_51':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,21)]
        if name == 'BS_56':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,21)]
        if name == 'BS_67':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,14)]
        if name == 'BS_87':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,14)]###

        #Tuning
        if name == 'BS_173':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,28)]
        if name == 'BS_175':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,24)]
        if name == 'BS_213':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,8)]
        if name == 'AE_235':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,9)]
        if name == 'AE_236':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,10)]
        if name == 'TH_237':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,8)]
        if name == 'AE_240':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,35)]
        if name == 'AE_252':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,26)]

        #Opsin
        if name == 'BS_86':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,20)]
        #Opsin
        if name == 'BS_100':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,15)]
        #Opsin
        if name == 'BS_123':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,20)]
        #Opsin
        if name == 'BS_128':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,14)]
        #Opsin
        if name == 'BS_163':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,19)]
        #Opsin
        if name == 'BS_174':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,24)]

        #Opto control            
        if name == 'AE_267':
            sessionnumbers = sessionnumbers[np.less_equal(sessionnumbers,17)]

        #For each session, get all block numbers
        for sessionnumber in sessionnumbers:
            animalBehavior.sessions[sessionnumber] = SimpleNamespace()
            animalBehavior.sessions[sessionnumber].animal = name
            animalBehavior.sessions[sessionnumber].session = sessionnumber
            animalBehavior.sessions[sessionnumber].files = []
            animalBehavior.sessions[sessionnumber].blocknumbers = []
            #These parameters will be set later
            animalBehavior.sessions[sessionnumber].type = 'ERROR'
            prefixtofind = name+'_'+str(sessionnumber)+'v'

            tempfiles = []
            order = []

            #################################    PRE REVERSAL   #################################
            for file in behaviorfiles:
                if file[0:len(prefixtofind)] == prefixtofind:
                    #Note that the order of these checks matters as some conditions are could be caught mistakenly by some checks
                    if file[(len(file)-23):len(file)] == '_RECOVERED_reversal.txt':
                        pass
                    elif file[(len(file)-14):len(file)] == '_RECOVERED.txt':
                        tempfiles.append(file)
                        order.append(file[len(prefixtofind):(len(file)-14)])
                    elif file[(len(file)-9):len(file)] == '_opto.txt':
                        pass
                    elif file[(len(file)-13):len(file)] == '_reversal.txt':
                        pass
                    elif file[(len(file)-4):len(file)] == '.txt':
                        tempfiles.append(file)
                        order.append(file[len(prefixtofind):(len(file)-4)])
                    else:
                        raise Exception('Unrecognized file suffix')

            #################################   POST REVERSAL   #################################
            for file in behaviorfiles:
                if file[0:len(prefixtofind)] == prefixtofind:
                    #Note that the order of these checks matters as some conditions are could be caught mistakenly by some checks
                    if file[(len(file)-23):len(file)] == '_RECOVERED_reversal.txt':
                        tempfiles.append(file)
                        order.append(file[len(prefixtofind):(len(file)-23)])
                    elif file[(len(file)-14):len(file)] == '_RECOVERED.txt':
                        pass
                    elif file[(len(file)-9):len(file)] == '_opto.txt':
                        tempfiles.append(file)
                        order.append(file[len(prefixtofind):(len(file)-9)])
                    elif file[(len(file)-13):len(file)] == '_reversal.txt':
                        tempfiles.append(file)
                        order.append(file[len(prefixtofind):(len(file)-13)])
                    elif file[(len(file)-4):len(file)] == '.txt':
                        pass
                    else:
                        raise Exception('Unrecognized file suffix')

            tempfiles = np.array(tempfiles)
            order = np.array(order,dtype='int')
            #Put all files in order by block number
            idxs = np.argsort(order)
            tempfiles = tempfiles[idxs]
            order = order[idxs]

            #If there's a mask, remove the relevant files here
            if not mask is None:

                maskedidxs = np.isin(order,mask[name][sessionnumber])
                order = order[maskedidxs]
                tempfiles = tempfiles[maskedidxs]

            animalBehavior.sessions[sessionnumber].files = tempfiles
            animalBehavior.sessions[sessionnumber].blocknumbers = order
        animalBehaviors[name] = animalBehavior
    return animalBehaviors


def loadAllSessions(animalBehaviors,directory,makenolaser=[]):
    for animalname in animalBehaviors:
        for session in animalBehaviors[animalname].sessions:
            filenames = animalBehaviors[animalname].sessions[session].files
            filenames = [os.path.join(directory,animalname,'behavior',filename) for filename in filenames]
            
            csv,num_trials = loadBehaviorSession(filenames,makenolaser=makenolaser)

            if csv is None:
                print('loadAllSessions: '+animalname+' session '+str(session)+' is None')
                continue

            animalBehaviors[animalname].sessions[session].trials_per_session = num_trials

            outcomes = csv['outcome']
            animalBehaviors[animalname].sessions[session].outcomes = outcomes

            if hasattr(csv,'response time'):
                response_time = csv['response time']
                animalBehaviors[animalname].sessions[session].response_time = response_time
            else:
                print(f"{animalname} session {session} has no response time")

            tones = csv['tone']
            animalBehaviors[animalname].sessions[session].tones = tones
            laser = np.array([False]*len(outcomes),dtype='bool')
            if 'laser' in csv.columns:
                laser = np.array(csv['laser'],dtype='bool')
            animalBehaviors[animalname].sessions[session].laser = laser

            hits = np.array(np.equal(outcomes,1))
            misses = np.array(np.equal(outcomes,2))
            falarms = np.array(np.equal(outcomes,3))
            crejects = np.array(np.equal(outcomes,4))

            laser_on = laser
            laser_off = np.logical_not(laser)

            target = np.logical_or(hits,misses)
            nontarget = np.logical_or(falarms,crejects)
            hightone = np.equal(tones,11260)
            lowtone = np.equal(tones,5648)

            prereversal = np.logical_or( np.logical_and(hightone,target) , np.logical_and(lowtone,nontarget) )
            postreversal = np.logical_or( np.logical_and(lowtone,target) , np.logical_and(hightone,nontarget) )

            on_pre = np.logical_and(laser_on,prereversal)
            on_post = np.logical_and(laser_on,postreversal)
            off_pre = np.logical_and(laser_off,prereversal)
            off_post = np.logical_and(laser_off,postreversal)

            if np.sum(on_pre)+np.sum(off_pre) > 0 and np.sum(on_post)+np.sum(off_post) == 0:
                animalBehaviors[animalname].sessions[session].type = 'prereversal'
            elif np.sum(on_pre)+np.sum(off_pre) == 0 and np.sum(on_post)+np.sum(off_post) > 0:
                animalBehaviors[animalname].sessions[session].type = 'postreversal'
            elif np.sum(on_pre)+np.sum(off_pre) > 0 and np.sum(on_post)+np.sum(off_post) > 0:
                animalBehaviors[animalname].sessions[session].type = 'switch'

                #normal reversal
                if np.array(on_pre)[0] or np.array(off_pre)[0]:
                    if not np.isfinite(animalBehaviors[animalname].reversal):
                        animalBehaviors[animalname].reversal = session
                    else:
                        print('ERROR: MULTIPLE REVERSALS IN ANIMAL '+animalname)
                #second reversal
                elif np.array(on_post)[0] or np.array(off_post)[0]:
                    if not np.isfinite(animalBehaviors[animalname].second_reversal):
                        animalBehaviors[animalname].second_reversal = session
                    else:
                        print('ERROR: MULTIPLE SECOND REVERSALS IN ANIMAL '+animalname)

            criteria = [on_pre,on_post,off_pre,off_post]
            for idx,criterion in enumerate(criteria):
                if np.sum(criterion) == 0:
                    continue

                behavior = SimpleNamespace()
                behavior.hits = np.sum(np.logical_and(hits,criterion))
                behavior.misses = np.sum(np.logical_and(misses,criterion))
                behavior.falarms = np.sum(np.logical_and(falarms,criterion))
                behavior.crejects = np.sum(np.logical_and(crejects,criterion))

                if idx==0:
                    animalBehaviors[animalname].sessions[session].onPreBehavior = behavior
                elif idx==1:
                    animalBehaviors[animalname].sessions[session].onPostBehavior = behavior
                elif idx==2:
                    animalBehaviors[animalname].sessions[session].offPreBehavior = behavior
                elif idx==3:
                    animalBehaviors[animalname].sessions[session].offPostBehavior = behavior

            #Generate performance stats
            #By trails and by sessions
            num_hits = np.sum(hits)
            num_misses = np.sum(misses)
            num_falarms = np.sum(falarms)
            num_crejects = np.sum(crejects)
            pc_trials = (num_hits + num_crejects) / (num_hits + num_misses + num_falarms + num_crejects)
            dp_trials = norm.ppf((num_hits+1)/(num_hits+num_misses+2)) - norm.ppf((num_falarms+1)/(num_falarms+num_crejects+2))

            # session_pcs = []
            # session_dps = []
            # trials = list(range(int(np.sum(num_trials))))
            # for idx,num_session in enumerate(num_trials):
            #     passed_sessions = np.sum(num_trials[range(idx)]) #This is 1-indexed.
            #     start_trials = np.greater_equal(trials,passed_sessions) #or equal because we're going from one to zero indexing
            #     end_trials = np.less(trials,passed_sessions+num_session)
            #     in_range = np.logical_and(start_trials,end_trials)

            #     if np.sum(in_range) != num_session:
            #         print('ERROR: session range calculation incorrect')
            #         raise Exception

            #     session_hits = np.sum(np.logical_and(in_range,hits))
            #     session_misses = np.sum(np.logical_and(in_range,misses))
            #     session_falarms = np.sum(np.logical_and(in_range,falarms))
            #     session_crejects = np.sum(np.logical_and(in_range,crejects))
            #     session_pc = (session_hits + session_crejects) / num_session
            #     session_dp = norm.ppf((session_hits+1)/(session_hits+session_misses+2)) - norm.ppf((session_falarms+1)/(session_falarms+session_crejects+2))
            #     session_pcs.append(session_pc)
            #     session_dps.append(session_dp)
            # pc_sessions = np.mean(session_pcs)
            # dp_sessions = np.mean(session_dps)

            #PREREVERSAL ONLY
            if animalBehaviors[animalname].sessions[session].type in ['prereversal','switch']:
                session_pcs = []
                session_dps = []
                trials = list(range(int(np.sum(num_trials))))
                for idx,num_session in enumerate(num_trials):
                    passed_sessions = np.sum(num_trials[range(idx)]) #This is 1-indexed.
                    start_trials = np.greater_equal(trials,passed_sessions) #or equal because we're going from one to zero indexing
                    end_trials = np.less(trials,passed_sessions+num_session)
                    in_range = np.logical_and(start_trials,end_trials)

                    in_range = np.logical_and(in_range,prereversal)
                    if(np.sum(in_range) <= 0):
                        continue

                    if np.sum(in_range) != num_session:
                        print('ERROR: session range calculation incorrect')
                        raise Exception

                    session_hits = np.sum(np.logical_and(in_range,hits))
                    session_misses = np.sum(np.logical_and(in_range,misses))
                    session_falarms = np.sum(np.logical_and(in_range,falarms))
                    session_crejects = np.sum(np.logical_and(in_range,crejects))
                    session_pc = (session_hits + session_crejects) / num_session
                    session_dp = norm.ppf((session_hits+1)/(session_hits+session_misses+2)) - norm.ppf((session_falarms+1)/(session_falarms+session_crejects+2))
                    session_pcs.append(session_pc)
                    session_dps.append(session_dp)
                pc_sessions = np.mean(session_pcs)
                dp_sessions = np.mean(session_dps)

                animalBehaviors[animalname].sessions[session].pre_pc_sessions = pc_sessions
                animalBehaviors[animalname].sessions[session].pre_dp_sessions = dp_sessions

            #POSTREVERSAL ONLY
            if animalBehaviors[animalname].sessions[session].type in ['postreversal','switch']:
                session_pcs = []
                session_dps = []
                trials = list(range(int(np.sum(num_trials))))
                for idx,num_session in enumerate(num_trials):
                    passed_sessions = np.sum(num_trials[range(idx)]) #This is 1-indexed.
                    start_trials = np.greater_equal(trials,passed_sessions) #or equal because we're going from one to zero indexing
                    end_trials = np.less(trials,passed_sessions+num_session)
                    in_range = np.logical_and(start_trials,end_trials)

                    if animalname == 'DS_19' and session == 27: # There is one unidentifiable tone in this recording. toneID == -1. 125 threshold crossings = 1kHz???
                        postreversal = np.ones_like(in_range) # This is known to be a post-reversal session

                    in_range = np.logical_and(in_range,postreversal)
                    if(np.sum(in_range) <= 0):
                        continue

                    if np.sum(in_range) != num_session:
                        print(f"passed_sessions = {passed_sessions}, start_trials = {start_trials}, end_trials = {end_trials}, in_range = {in_range}, sum(in_range) = {np.sum(in_range)}, num_session = {num_session}")
                        print(f"start = {np.sum(start_trials)}")
                        print(f"end = {np.sum(end_trials)}")
                        print(f"postreversal = {np.sum(postreversal)}")
                        print(f"prerevtrial = {np.where(np.logical_not(postreversal))[0]+1}")
                        raise Exception(f"Session range calculation for {animalname} session {session}.")

                    session_hits = np.sum(np.logical_and(in_range,hits))
                    session_misses = np.sum(np.logical_and(in_range,misses))
                    session_falarms = np.sum(np.logical_and(in_range,falarms))
                    session_crejects = np.sum(np.logical_and(in_range,crejects))
                    session_pc = (session_hits + session_crejects) / num_session
                    session_dp = norm.ppf((session_hits+1)/(session_hits+session_misses+2)) - norm.ppf((session_falarms+1)/(session_falarms+session_crejects+2))
                    session_pcs.append(session_pc)
                    session_dps.append(session_dp)
                pc_sessions = np.mean(session_pcs)
                dp_sessions = np.mean(session_dps)

                animalBehaviors[animalname].sessions[session].post_pc_sessions = pc_sessions
                animalBehaviors[animalname].sessions[session].post_dp_sessions = dp_sessions

    return animalBehaviors

def loadAllPerformance(animalBehaviors,filename):
    perf_df = pd.read_csv(filename)
    for animalname in animalBehaviors:
        try:
            for session in animalBehaviors[animalname].sessions:
                thissession = animalBehaviors[animalname].sessions[session]

                if thissession.type=='prereversal':
                    day = session

                    days = perf_df['Day']
                    pc = list(perf_df['Percent correct '+animalname])
                    dp = list(perf_df['d\' '+animalname])

                    idx = np.where( np.equal(days,day) )[0][0]
                    thissession.prePC = pc[idx]
                    thissession.preDP = dp[idx]
                elif thissession.type=='switch':
                    day = session

                    days = perf_df['Day']
                    pc = perf_df['Percent correct '+animalname]
                    dp = perf_df['d\' '+animalname]

                    idx = np.where( np.equal(days,day) )[0]
                    thissession.prePC = pc[idx]
                    thissession.preDP = dp[idx]
                    thissession.postPC = pc[idx+1]
                    thissession.postDP = dp[idx+1]
                elif thissession.type=='postreversal':
                    day = session+1

                    days = perf_df['Day']
                    pc = perf_df['Percent correct '+animalname]
                    dp = perf_df['d\' '+animalname]

                    idx = np.where( np.equal(days,day) )[0]
                    thissession.postPC = pc[idx]
                    thissession.postDP = dp[idx]

                animalBehaviors[animalname].sessions[session] = thissession
        except KeyError as e:
            print(animalname + ' does not have listed performance data: ' + str(e))
    return animalBehaviors

def loadBehaviorSession(filenames,makenolaser=[],verbose=False):
    csv = None
    filename = None
    num_trials = np.zeros(len(filenames))
    for filename_idx,filename in enumerate(filenames):
        num_trials_session = 0
        try:
            if csv is None:
                try:
                    csv = pd.read_csv(filename,header=None)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    raise e

                if len(csv.columns) == 15: #Laser included
                    csv.columns = ['training day','trial number','tone','outcome','trial duration','tone start time','response time','delay time','delay time (real)','lick time','trial start (master)','tone on (master)','lick time (master)','reward time (master)','laser']
                    if np.any([filename[(len(filename)-len(x)):len(filename)] == x for x in makenolaser]):
                        if verbose:
                            print(filename)
                        csv = csv.drop(['laser'],axis=1)
                elif len(csv.columns) == 14: #No laser
                    csv.columns = ['training day','trial number','tone','outcome','trial duration','tone start time','response time','delay time','delay time (real)','lick time','trial start (master)','tone on (master)','lick time (master)','reward time (master)']
                elif len(csv.columns) == 4: #Recovered file
                    csv.columns = ['training day','trial number','tone','outcome']
                else:
                    raise Exception(f"Incorrect number of columns! {len(csv.columns)} columns were found")
                
                #Count number of trials
                num_trials_session = csv.shape[0]
            else:
                tempcsv = pd.read_csv(filename,header=None)
                if len(tempcsv.columns) == 15: #Laser included
                    tempcsv.columns = ['training day','trial number','tone','outcome','trial duration','tone start time','response time','delay time','delay time (real)','lick time','trial start (master)','tone on (master)','lick time (master)','reward time (master)','laser']
                    if np.any([filename[(len(filename)-len(x)):len(filename)] == x for x in makenolaser]):
                        if verbose:
                            print(filename)
                        tempcsv = tempcsv.drop(['laser'],axis=1)
                elif len(tempcsv.columns) == 14: #No laser
                    tempcsv.columns = ['training day','trial number','tone','outcome','trial duration','tone start time','response time','delay time','delay time (real)','lick time','trial start (master)','tone on (master)','lick time (master)','reward time (master)']
                elif len(tempcsv.columns) == 4: #Recovered file
                    tempcsv.columns = ['training day','trial number','tone','outcome']
                else:
                    raise Exception(f"Incorrect number of columns! {len(tempcsv.columns)} columns were found")

                #Count number of trials
                num_trials_session = tempcsv.shape[0]

                #Concatenate trial numbers
                trialnumbers = np.array(tempcsv['trial number'])
                oldtrialnumbers = np.array(csv['trial number'])
                trialnumbers += np.max(oldtrialnumbers)
                tempcsv['trial number'] = trialnumbers

                csv = pd.concat([csv,tempcsv])
        except pd.errors.EmptyDataError as e:
            print(str(filename)+': '+str(e))

        num_trials[filename_idx] = num_trials_session

    #Replace all nans in laser with zero
    #This is so that switch recordings are
    #handled correctly. Nans evaluate to
    #True, not False
    if not csv is None and 'laser' in csv.columns:
        laser = csv['laser']
        laser = np.nan_to_num(laser)
        csv['laser'] = laser

    return csv, num_trials



def calculateEILPhase(animalBehaviors):
    animals = [a for a in animalBehaviors]
    for animalidx,animal in enumerate(animals):
        
        prereversaldays = []
        prereversaldps = []
        maxreversaldp = -np.Inf
        minreversaldp = np.Inf
        postreversaldays = []
        postreversaldps = []
        for session in animalBehaviors[animal].sessions:
            if animalBehaviors[animal].sessions[session].type in ['prereversal','opto prereversal']:
                prereversaldps.append(animalBehaviors[animal].sessions[session].pre_dp_sessions)
            if animalBehaviors[animal].sessions[session].type in ['switch','opto switch']:
                maxreversaldp = animalBehaviors[animal].sessions[session].pre_dp_sessions
                minreversaldp = animalBehaviors[animal].sessions[session].post_dp_sessions
            if animalBehaviors[animal].sessions[session].type in ['postreversal','opto postreversal']:
                postreversaldps.append(animalBehaviors[animal].sessions[session].post_dp_sessions)
                
        #Prereversal threshold
        try:
            maxperf = np.nanmax(np.concatenate((prereversaldps,[maxreversaldp])))
            minperf = np.nanmin(prereversaldps)
            prethreshold = minperf + 0.4*(maxperf-minperf)
        except:
            maxperf = np.nan
            minperf = np.nan
            prethreshold = np.nan
        #Postreversal threshold
        try:
            maxperf = np.nanmax(postreversaldps)
            minperf = np.nanmin(np.concatenate((postreversaldps,[minreversaldp])))
            postthreshold = minperf + 0.4*(maxperf-minperf)
        except:
            maxperf = np.nan
            minperf = np.nan
            postthreshold = np.nan
        
        #postthresholds[animal] = postthreshold
        
        prereversaldays = np.array(prereversaldays)
        prereversaldps = np.array(prereversaldps)
        postreversaldays = np.array(postreversaldays)
        postreversaldps = np.array(postreversaldps)
        
        
        preint = False
        prelate = False
        postint = False
        postlate = False
        for session in animalBehaviors[animal].sessions:
            if animalBehaviors[animal].sessions[session].type in ['prereversal','opto prereversal']:
                if animalBehaviors[animal].sessions[session].pre_dp_sessions >= prethreshold or preint:
                    animalBehaviors[animal].sessions[session].phase = 'intermediate'
                    preint=True
                else:
                    animalBehaviors[animal].sessions[session].phase = 'early'
                    
                if (animalBehaviors[animal].sessions[session].pre_dp_sessions >= 1.5 and animalBehaviors[animal].sessions[session].pre_pc_sessions >= 0.7) or prelate:
                    animalBehaviors[animal].sessions[session].phase = 'late'
                    prelate=True
                    
            if animalBehaviors[animal].sessions[session].type in ['switch','opto switch']:
                animalBehaviors[animal].sessions[session].phase = 'switch'
                
            if animalBehaviors[animal].sessions[session].type in ['postreversal','opto postreversal']:
                if animalBehaviors[animal].sessions[session].post_dp_sessions >= postthreshold or postint:
                    animalBehaviors[animal].sessions[session].phase = 'intermediate'
                    postint = True
                else:
                    animalBehaviors[animal].sessions[session].phase = 'early'
                    
                if (animalBehaviors[animal].sessions[session].post_dp_sessions >= 1.5 and animalBehaviors[animal].sessions[session].post_pc_sessions >= 0.7) or postlate:
                    animalBehaviors[animal].sessions[session].phase = 'late'
                    postlate = True
                    
        if animal == 'BS_49':
            #Set first day pre intermediate to 7
            for session in animalBehaviors[animal].sessions:
                if animalBehaviors[animal].sessions[session].type in ['prereversal','opto prereversal'] and session >= 7 and animalBehaviors[animal].sessions[session].phase != 'late':
                    animalBehaviors[animal].sessions[session].phase = 'intermediate'
        
        if animal == 'BS_95':
            #Set first day pre intermediate to 4
            for session in animalBehaviors[animal].sessions:
                if animalBehaviors[animal].sessions[session].type in ['prereversal','opto prereversal'] and session >= 4 and animalBehaviors[animal].sessions[session].phase != 'late':
                    animalBehaviors[animal].sessions[session].phase = 'intermediate'
        if animal in ['BS_83','BS_50']:
            #These animals never reach intermediate
            for session in animalBehaviors[animal].sessions:
                if animalBehaviors[animal].sessions[session].type in ['postreversal','opto postreversal']:
                    animalBehaviors[animal].sessions[session].phase = 'early'

    return animalBehaviors






















