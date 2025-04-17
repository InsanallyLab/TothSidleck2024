import sys, os
import json, pickle
import numpy as np
import pandas as pd
import time as pytime
from tqdm import tqdm
from types import SimpleNamespace
from scipy.stats import norm

from .disqualification import disqualifyTrials, disqualifyISI, disqualifyFR
from .tuning import identifyNumberOfTuningTrials,determineTuningCurveTones

def loadSessionInitial(DIRPATH,NAME,region):
    DIR = os.path.join(DIRPATH,NAME)
    folder = os.path.join(DIR,region)

    session = SimpleNamespace()
    session.meta = SimpleNamespace()
    session.clusters = SimpleNamespace()
    session.channels = SimpleNamespace()
    session.spikes = SimpleNamespace()
    session.trials = SimpleNamespace()
    session.behavior = SimpleNamespace()

    #Load metadata
    try:
        with open(os.path.join(folder, 'session_metadata.json'), 'r') as f:
            meta = json.load(f)
        for k in meta.keys():
            setattr(session.meta,k,meta[k])
    except Exception as e:
        print(e)
    #Load analogue metadata
    try:
        with open(os.path.join(folder, 'analog_metadata.json'), 'r') as f:
            meta = json.load(f)
        for k in meta.keys():
            setattr(session.meta,k,meta[k])
    except Exception as e:
        print(e)
    #Load condition metadata
    if session.meta.task in ['nonreversal','switch','reversal','opto nonreversal','opto switch','opto reversal','tuning nonreversal','tuning switch','tuning reversal','opto control nonreversal','opto control switch','opto control reversal']:
        try:
            with open(os.path.join(folder, 'condition_metadata.json'), 'r') as f:
                meta = json.load(f)
            for k in meta.keys():
                setattr(session.meta,k,meta[k])
        except Exception as e:
            print(e)
    #Load condition metadata
    if session.meta.task in ['opto nonreversal','opto switch','opto reversal']:
        try:
            with open(os.path.join(folder, 'cloudy_metadata.json'), 'r') as f:
                meta = json.load(f)
            for k in meta.keys():
                setattr(session.meta,k,meta[k])
        except Exception as e:
            print(e)
    #Load trimming data
    if session.meta.task in ['nonreversal','switch','reversal','opto nonreversal','opto switch','opto reversal','tuning nonreversal','tuning switch','tuning reversal','second switch','second reversal','passive no behavior','opto control nonreversal','opto control switch','opto control reversal']:
        try:
            with open(os.path.join(folder,'trim.pickle'), 'rb') as f:
                trim = pickle.load(f)
            session.trim = trim
        except Exception as e:
            print(e)
    #Load responsiveness data
    if session.meta.task in ['nonreversal','switch','reversal','opto nonreversal','opto switch','opto reversal','tuning nonreversal','tuning switch','tuning reversal','second switch','second reversal','passive no behavior','opto control nonreversal','opto control switch','opto control reversal']:
        try:
            with open(os.path.join(folder,'responsiveness.pickle'), 'rb') as f:
                responsiveness = pickle.load(f)
            session.responsiveness = responsiveness
        except Exception as e:
            print(e)
    if session.meta.task in ['tuning nonreversal','tuning switch','tuning reversal','passive no behavior']:
        session.tuning = SimpleNamespace()
        try:
            with open(os.path.join(folder,'tuning_responsiveness.pickle'), 'rb') as f:
                tuning_responsiveness = pickle.load(f)
            session.tuning.tuning_responsiveness = tuning_responsiveness
        except Exception as e:
            print(e)

    #Try to load raw audio data for data verification
    try:
        raw_audio_threshold_crossings = np.load(os.path.join(DIR,"tone_counts.npy"))
        raw_audio_durations = np.load(os.path.join(DIR,"tone_durations.npy"))
        session.trials.raw_audio_threshold_crossings = raw_audio_threshold_crossings[0] #Indexing here because for some reason the files are stored as [[data]] rather than [data]
        session.trials.raw_audio_durations = raw_audio_durations[0]
    except Exception as e:
        #print(e)
        print('Recording does not contain raw audio data. Use saveTrials.m to generate')
        pass

    #Hardcoded parameters
    session.meta.arraystart = 0
    session.meta.tonelength = 0.1
    session.meta.triallength = 2.5
    session.meta.lickdelayrelativetostart = 0#0.2
    session.meta.lickdelayrelativetotone = session.meta.lickdelayrelativetostart - session.meta.tonelength
    session.meta.fs = 30000
    session.meta.length_in_seconds = session.meta.length_in_samples/session.meta.fs
    session.meta.length_in_minutes = session.meta.length_in_seconds/60

    #Import spike data
    spike_times = np.load(os.path.join(folder,"spike_times.npy"))
    amplitudes = np.load(os.path.join(folder,"amplitudes.npy"))
    templates = np.load(os.path.join(folder,"spike_templates.npy"))
    session.spikes.times = spike_times
    session.spikes.amplitudes = amplitudes
    session.spikes.templates = templates
    try:
        spike_clusters = np.load(os.path.join(folder,"spike_clusters.npy"))
        session.spikes.clusters = spike_clusters
        curated=True
    except Exception as e:
        print(e)
        curated=False
        
    #Import Laser Events
    if os.path.isfile(os.path.join(DIR,"laser_events.npy")):
        session.trials.laser_events = np.load(os.path.join(DIR,"laser_events.npy"))
    
    #Import cluster data
    phylabel = pd.read_csv(os.path.join(folder,"cluster_group.tsv"),delimiter="\t")
    unitmapping = np.unique(np.array(phylabel["cluster_id"]))
    session.clusters.list = unitmapping
    depthcalc=False
    if curated:
        good = phylabel["group"].values=="good"
        good = unitmapping[good]
        mua = phylabel["group"].values=="mua"
        mua = unitmapping[mua]
        noise = phylabel["group"].values=="noise"
        noise = unitmapping[noise]

        cluster_info = pd.read_csv(os.path.join(folder,"cluster_info.tsv"),delimiter="\t")
        if "id" in cluster_info:
            cluster_info_df = cluster_info.set_index(["id"])
        else:
            cluster_info_df = cluster_info.set_index(["cluster_id"])
        session.clusters.channels = dict()
        session.clusters.labels = dict()
        session.clusters.depth = dict()
        depthcalc=True
        session.clusters.x = dict()
        for unit in unitmapping:
            session.clusters.channels[unit] = cluster_info_df.loc[unit]["ch"]
            session.clusters.labels[unit] = cluster_info_df.loc[unit]["group"]
            try:
                session.clusters.depth[unit] = session.meta.depth - (812.5-cluster_info_df.loc[unit]["depth"])
            except:
                print('Depth calculation failed')
                session.clusters.depth[unit] = np.nan
                depthcalc = False;
    else:
        print("Recording has not been curated. Uncurated sessions are currently unsupported")
    session.clusters.good = good
    session.clusters.mua = mua
    session.clusters.noise = noise

    #Verify Depth
    if depthcalc:
        session.channels.list = list(range(64))

        #H5 Probe
        xcoords = np.array([50,    50,    50,    25,    25,    50,    50,    25,    25,    50,    50,    25,    25,    25,    50,    50,    25,    25,    25,    25,    25,    25,    25,    25,    25,    25,    25,    25,    25,    25,    25,    50,    25,    25,    25,    50,    50,    25,    25,
        50,    50,    25,    25,    50,    50,    50,    25,    50,    50,    25,    50,    50,    50,    50,    50,    50,    50,    50,    50,    50,    50,    50,    25,    50])
        ycoords = np.array([725.0000,  525.0000,  775.0000,  662.5000,  762.5000,  675.0000,  650.0000,  787.5000,  637.5000,  800.0000,  625.0000,  812.5000,  737.5000,  712.5000,  750.0000,  700.0000,  287.5000,  187.5000,  262.5000,  212.5000,   37.5000,  412.5000,  62.5000,
        387.5000,   87.5000,  362.5000,  312.5000,  137.5000,  337.5000,  112.5000,  237.5000,  200.0000,  687.5000,  512.5000,  537.5000,  500.0000,  550.0000,  487.5000,  562.5000,  475.0000,  575.0000,  462.5000,  587.5000,  450.0000,  600.0000,  425.0000,
        612.5000,  400.0000,  375.0000,  437.5000,  350.0000,   25.0000,  325.0000,   50.0000,  300.0000,   75.0000,  275.0000,  100.0000,  250.0000,  125.0000,  225.0000,  150.0000,  162.5000,  175.0000])

        #H6 Probe
        #Not currently implemented

        channel_depth = session.meta.depth - (812.5-ycoords)
        channel_x = xcoords-25
        session.channels.depth = channel_depth
        session.channels.x = channel_x
        depth_match = True;
        for clust in session.clusters.list:
            if (channel_depth[session.clusters.channels[clust]] == session.clusters.depth[clust]):
                session.clusters.x[clust] = channel_x[session.clusters.channels[clust]]
            else:
                depth_match = False;
                print('Depth mismatch on cluster '+str(clust))
        if(depth_match):
            pass
            #print('All depths matched')

    #Import trial data
    tone_times = np.load(os.path.join(DIR,"tone_times.npy"))[0]
    tone_IDs = np.load(os.path.join(DIR,"tone_IDs.npy"))[0]
    lick_times = np.load(os.path.join(DIR,"licks.npy"))[0]
    reward_times = np.load(os.path.join(DIR,"rewards.npy"))[0]

    session.behavior.lick_times = lick_times
    session.behavior.reward_times = reward_times

    session.meta.length_in_trials = len(tone_times)
    session.trials.starts = tone_times
    session.trials.ends = session.trials.starts + (session.meta.triallength*session.meta.fs)
    session.trials.freqs = tone_IDs

    if session.meta.task in ['tuning nonreversal','tuning switch','tuning reversal','passive no behavior']:
        #number_of_tones = session.meta.length_in_trials - session.meta.first_tuning_trial

        ##This is required because of the fact that different tuning recordings have different numbers of
        ##tone presentations and the numbers that are written down in the session_metadata files are not
        ##completely reliable. Within 10 trials or so I can likely rely on their accuracy
        #if np.abs(number_of_tones - 250) < 10:
        #    number_of_tones = 250
        #elif np.abs(number_of_tones - 350) < 10:
        #    number_of_tones = 350
        #elif np.abs(number_of_tones - 450) < 10:
        #    number_of_tones = 450
        #elif np.abs(number_of_tones - 0) < 10:
        #    number_of_tones = 0

        number_of_tones = identifyNumberOfTuningTrials(session)

        #print(session.meta.length_in_trials)
        #print(number_of_tones)
        session.meta.first_tuning_trial_corrected = session.meta.length_in_trials - number_of_tones #This is no longer 1-indexed
        session.meta.length_in_trials = session.meta.first_tuning_trial_corrected
        #session.tuning = SimpleNamespace()
        session.tuning.number_of_tones = number_of_tones

    if os.path.isfile(os.path.join(DIR,"laser_events.npy")):
        session.trials.laser_stimulation = np.zeros(session.meta.length_in_trials, dtype=bool)
    session.trials.target = np.zeros(session.meta.length_in_trials, dtype=bool)
    session.trials.go = np.zeros(session.meta.length_in_trials, dtype=bool)
    session.trials.reward = np.zeros(session.meta.length_in_trials)
    session.trials.reward[:] = np.nan
    session.trials.response = np.zeros(session.meta.length_in_trials)
    session.trials.response[:] = np.nan
    session.trials.valid = np.ones(session.meta.length_in_trials, dtype=bool)
    session.trials.correct = np.zeros(session.meta.length_in_trials, dtype=bool)

    #print(session.trials.freqs)
    #print(session.meta.length_in_trials)
    #print(len(session.trials.starts))
    #print(len(session.trials.freqs))

    for trial in range(session.meta.length_in_trials):
        #Nonreversal
        if session.meta.task in ['nonreversal','opto nonreversal','opto control nonreversal','tuning nonreversal','second reversal'] or (session.meta.task in ['switch','opto switch','opto control switch','tuning switch'] and trial < session.meta.first_reversal_trial) or (session.meta.task in ['second switch'] and trial >= session.meta.first_reversal_trial) or session.meta.task == 'passive no behavior':
            if session.trials.freqs[trial] == 11260:
                session.trials.target[trial] = True;
            elif session.trials.freqs[trial] == 5648:
                session.trials.target[trial] = False;
            else:
                error("Unrecognized tone frequency");
        #Reversal
        elif session.meta.task in ['reversal','opto reversal','opto control reversal','tuning reversal'] or (session.meta.task in ['switch','opto switch','opto control switch','tuning switch'] and trial >= session.meta.first_reversal_trial) or (session.meta.task in ['second switch'] and trial < session.meta.first_reversal_trial):
            if session.trials.freqs[trial] == 5648:
                session.trials.target[trial] = True;
            elif session.trials.freqs[trial] == 11260:
                session.trials.target[trial] = False;
            else:
                error("Unrecognized tone frequency");
        elif(session.meta.task == 'thalamus tuning'):
            continue
        elif session.meta.task in ['passive no behavior']:
            pass
        elif session.meta.task in ['CNO','muscimol']:
            pass
        else:
            print("Unrecognized Reversal type: "+session.meta.task);
            #raise Exception
        #Go-nogo
        gonogostart = session.trials.starts[trial] + session.meta.fs * session.meta.lickdelayrelativetostart
        gonogoend = session.trials.ends[trial]
        temp_licks = session.behavior.lick_times[ np.logical_and(    np.greater(session.behavior.lick_times,gonogostart)  ,  np.less(session.behavior.lick_times,gonogoend)    ) ]
        if len(temp_licks) > 0:
            session.trials.go[trial] = True;
            session.trials.response[trial] = np.min(temp_licks)
        else:
            pass
            #session.trials.go[trial] = False;
            #session.trials.response[trial] = np.nan

        #Reward
        rewardstart = session.trials.starts[trial]
        rewardend = session.trials.ends[trial]
        temp_rewards = session.behavior.reward_times[ np.logical_and(    np.greater(session.behavior.reward_times,rewardstart)  ,  np.less(session.behavior.reward_times,rewardend)    ) ]
        if len(temp_rewards) > 0:
            session.trials.reward[trial] = np.min(temp_rewards)
        else:
            pass
            #session.trials.reward[trial] = np.nan

        #Laser
        if os.path.isfile(os.path.join(DIR,"laser_events.npy")):
            laser_events = np.logical_and(      np.greater(session.trials.laser_events,session.trials.starts[trial]-0.1*session.meta.fs)      ,      np.less(session.trials.laser_events,session.trials.ends[trial])            )
            if np.sum(laser_events) == 2:
                session.trials.laser_stimulation[trial] = True
            elif np.sum(laser_events) == 0:
                session.trials.laser_stimulation[trial] = False
            else:
                print('ERROR: Calculating laser stimulation')
        #Correct / Incorrect
        hit = session.trials.target[trial] and session.trials.go[trial]
        cr = not session.trials.target[trial] and not session.trials.go[trial]
        session.trials.correct[trial] = hit or cr

        #Validation of outcome-reward correspondences
        if session.trials.target[trial] and session.trials.go[trial]:
            if np.isfinite(session.trials.reward[trial]):
                session.trials.valid[trial] = True
            else:
                session.trials.valid[trial] = False
        else:
            if np.isfinite(session.trials.reward[trial]):
                session.trials.valid[trial] = False
            else:
                session.trials.valid[trial] = True
            


    if session.meta.task in ['tuning nonreversal','tuning switch','tuning reversal','passive no behavior']:

        session.tuning.all_tone_times = session.trials.starts
        session.tuning.trial_starts = session.trials.starts[  (session.meta.first_tuning_trial_corrected) : len(session.trials.starts) ]
        session.tuning.trial_freqs = session.trials.freqs[  (session.meta.first_tuning_trial_corrected) : len(session.trials.starts) ]
        #TODO Determine trial frequencies

        session.trials.starts = session.trials.starts[range(session.meta.first_tuning_trial_corrected)]
        session.trials.ends = session.trials.ends[range(session.meta.first_tuning_trial_corrected)]
        session.trials.freqs = session.trials.freqs[range(session.meta.first_tuning_trial_corrected)]
        session.trials.response = session.trials.response[range(session.meta.first_tuning_trial_corrected)]
        session.trials.target = session.trials.target[range(session.meta.first_tuning_trial_corrected)]
        session.trials.go = session.trials.go[range(session.meta.first_tuning_trial_corrected)]
        session.trials.correct = session.trials.correct[range(session.meta.first_tuning_trial_corrected)]
        session.trials.reward = session.trials.reward[range(session.meta.first_tuning_trial_corrected)]
        session.trials.valid = session.trials.valid[range(session.meta.first_tuning_trial_corrected)]

        session = determineTuningCurveTones(session)

    #print(len(session.tuning.all_tone_times))
    #print(session.meta.length_in_trials)
    #print(session.tuning.number_of_tones)

    return session

#Sometimes Kilosort duplicates a unit. This shows up as two templates with normal looking
#CCGs indicating being the same unit, but a large spike at exactly T=0. These spikes at
#T=0 need to be removed and turned into just one unit
def removeDuplicateSpikes(sessionfile,window_ms=0.25,verbose=False):
    window_samples = window_ms * sessionfile.meta.fs / 1000
    if verbose:
        print('Window: ' + str(window_ms) + ' ms / ' + str(window_samples) + ' samples')

    spikes_to_be_deleted = [False] * len(sessionfile.spikes.times)
    for clust in sessionfile.clusters.list:
        #Caching search -- Unit
        idxs = np.equal(sessionfile.spikes.clusters,clust)
        idxs = np.where(idxs)[0]

        #Check if there's even more than one template. If not, we can skip
        cluster_templates = np.sort(np.unique(sessionfile.spikes.templates[idxs]))
        if(len(cluster_templates)<=1):
            continue

        if verbose:
            print(str(clust) + ": " + str(cluster_templates))

        numremoved = 0
        for idx in range(len(idxs)):
            spikeidx = idxs[idx]

            spiketime = sessionfile.spikes.times[spikeidx]
            spikeclust = sessionfile.spikes.clusters[spikeidx]
            spiketemp = sessionfile.spikes.templates[spikeidx]

            #Now check 

            nextidx = idx+1
            while nextidx < len(idxs) and sessionfile.spikes.times[idxs[nextidx]] <= spiketime + window_samples:
                if sessionfile.spikes.templates[idxs[nextidx]] != spiketemp:
                    spikes_to_be_deleted[idxs[nextidx]] = True
                    numremoved += 1
                    if verbose:
                        print(str(spikeclust) + ' ==> ' + str(spiketemp) + '/' + str(sessionfile.spikes.templates[idxs[nextidx]]) + ': ' + str(spiketime) + ',' + str(sessionfile.spikes.times[idxs[nextidx]]))
                nextidx += 1

        if verbose:
            print( str(clust) + ': ' + str(numremoved / len(idxs) * 100) + '% of spikes removed' )

    #Now actually remove all those spikes
    sessionfile.spikes.times = np.delete(sessionfile.spikes.times,spikes_to_be_deleted)
    sessionfile.spikes.clusters = np.delete(sessionfile.spikes.clusters,spikes_to_be_deleted)
    sessionfile.spikes.amplitudes = np.delete(sessionfile.spikes.amplitudes,spikes_to_be_deleted)
    sessionfile.spikes.templates = np.delete(sessionfile.spikes.templates,spikes_to_be_deleted)

    return sessionfile

#Need to make this come from labeled data
def calculateStats(session):
    hits = 0
    misses = 0
    falarms = 0
    crejects = 0
    
    for trial in range(session.meta.length_in_trials):
        if session.trials.target[trial]:
            if session.trials.go[trial]:
                hits = hits + 1
            else:
                misses = misses + 1
        else:
            if session.trials.go[trial]:
                falarms = falarms + 1
            else:
                crejects = crejects + 1
                
    session.meta.hits = hits
    session.meta.misses = misses
    session.meta.falarms = falarms
    session.meta.crejects = crejects
    session.meta.pc = (hits+crejects)/(hits+misses+falarms+crejects)
    session.meta.dp = norm.ppf((hits+1)/(hits+misses+2)) - norm.ppf((falarms+1)/(falarms+crejects+2))
    return session

def generateSaveString(sessionfile):
    namemodifier = 'ERROR'
    if sessionfile.meta.task == 'CNO':
        namemodifier = 'CNO_'+str(sessionfile.meta.date).replace('/','-')
    elif sessionfile.meta.task == 'muscimol':
        namemodifier = str(sessionfile.meta.day_of_recording)+'_muscimol_'+str(sessionfile.meta.date).replace('/','-')

    elif sessionfile.meta.task in ['nonreversal','switch','reversal','second switch','second reversal']:
        namemodifier = str(sessionfile.meta.day_of_recording)
    elif sessionfile.meta.task in ['opto nonreversal','opto switch','opto reversal']:
        namemodifier = str(sessionfile.meta.day_of_recording)+'_opto'
    elif sessionfile.meta.task in ['opto control nonreversal','opto control switch','opto control reversal']:
        namemodifier = str(sessionfile.meta.day_of_recording)+'_opto_control'
    elif sessionfile.meta.task in ['tuning nonreversal','tuning switch','tuning reversal']:
        namemodifier = str(sessionfile.meta.day_of_recording)+'_tuning'
    elif sessionfile.meta.task in ['passive no behavior']:
        namemodifier = str(sessionfile.meta.day_of_recording)+'_passive'
    elif sessionfile.meta.task == 'thalamus tuning':
        namemodifier = str(sessionfile.meta.recording_sessionfile)+'_thalamus_tuning'

    return sessionfile.meta.animal + '_' + namemodifier + '_' + sessionfile.meta.region

def saveSession(session,destination):
    filename = generateSaveString(session)
    filename = os.path.join(destination, filename + '.pickle')
    print(filename)
    with open(filename, 'wb') as f:
        pickle.dump(session, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadSessionCached(directory,filename):
    if isinstance(directory,str):
        with open(os.path.join(directory,filename), 'rb') as f:
            session = pickle.load(f)
        return session
    elif isinstance(directory,list):
        for direc in directory:
            try:
                with open(os.path.join(direc,filename), 'rb') as f:
                    session = pickle.load(f)
                return session
            except Exception as e:
                # print(f"Error while loading {directory} {filename}: {e}")
                pass
        raise Exception('sessionfile not found')
    else:
        print(f"Error while loading {directory} {filename}: directory is of type {type(directory)}")

def loadSessionsComplete(directory,destination='D:\\Analysis_Cache',numsamples=5000,verbose=False):
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
        #if animal in ['AE_235','AE_236','TH_237','TH_230','AE_229','TH_233','TH_234']:
        #    continue
        #if not animal in ['AE_231']:
            #continue

        sessions = os.listdir(os.path.join(directory,animal))
        for session in sessions:
            for region in ['AC','M2','MGB','Striatum']:
                
                if os.path.isfile(os.path.join(directory,animal,session,region,'session_metadata.json')):
                    try:
                        
                        sessionfile = loadSessionInitial(os.path.join(directory,animal),session,region)
                        sessionfile = removeDuplicateSpikes(sessionfile)
                        sessionfile = disqualifyISI(sessionfile)

                        if not sessionfile.meta.task in ['thalamus tuning','CNO','muscimol']:
                            sessionfile = disqualifyTrials(sessionfile)
                            sessionfile = disqualifyFR(sessionfile)

                        print('A')
                        saveSession(sessionfile,destination)
                        print('B')
                        if verbose:
                            print(os.path.join(directory,animal,session,region,'session_metadata.json') + ' complete')
                    
                    except Exception as e:
                        print(os.path.join(directory,animal,session,region) + ': ' + str(e))
                        #raise e
                else:
                    print(os.path.join(directory,animal,session,region,'session_metadata.json') + ' missing. Skipping')
        print(animal + ' complete')
    print(directory + ' complete')
#NOTE: does not return anything because this is used to save out cached files