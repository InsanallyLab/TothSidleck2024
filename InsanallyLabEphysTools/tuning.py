import sys, os, pickle
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

def identifyNumberOfTuningTrials(sessionfile):
	ITIs = np.diff(sessionfile.trials.starts)
	tuning_ITIs = np.less(ITIs,45000)

	if np.sum(tuning_ITIs) == 0:
		number_tuning_ITIs = 0
	else:
		number_tuning_ITIs = np.sum(tuning_ITIs)+1
	return number_tuning_ITIs


#This function exists because the tuning curve recordings were done with little recording of what tones were played
#Thus we have to reconstruct what tones were presented

#Cohort 1 received non-randomized tones in blocks of 50 identical tones in ascending frequency. Number of tones may vary
#Cohort 2 received tones listed in pseudorandom file except 22k and 45k
#Cohort 3 received tones listed in pseudorandom file
def determineTuningCurveTones(sessionfile):
	#TuningCurvePseudorandomFile = os.path.join('C:\\Users','insan','Desktop','FreqListTuningCurve.txt')
	TuningCurvePseudorandomOrder = [16000,45255,4000,11260,64000,5648,32000,22627,8000]

	cohort1 = ['BS_173','BS_175']
	cohort2 = ['BS_187','BS_188']
	cohort3 = ['BS_213','BS_214']

	tones_250 = [4000,8000,16000,32000,64000]
	tones_350 = [4000,5648,8000,11260,16000,32000,64000]
	tones_450 = [4000,5648,8000,11260,16000,22627,32000,45255,64000]

	number_of_tones = sessionfile.tuning.number_of_tones

	tones_to_use = []
	if number_of_tones == 250:
		tones_to_use = tones_250
	elif number_of_tones == 350:
		tones_to_use = tones_350
	elif number_of_tones == 450 or number_of_tones == 449:
		tones_to_use = tones_450

	tonesArePresentedInOrder = testInOrderTonePresentation(sessionfile)

	trial_freqs = []
	if tonesArePresentedInOrder:
		trial_freqs = np.sort(tones_to_use * 50)
	else:
		#with open(TuningCurvePseudorandomFile) as file:
		#	trial_freqs = [line.rstrip() for line in file]
		#trial_freqs = trial_freqs[1:(len(trial_freqs)-1)]
		#trial_freqs = np.array([int(f) for f in trial_freqs])

		trial_freqs = np.array(TuningCurvePseudorandomOrder * 50)
		if sessionfile.tuning.number_of_tones == 449:
			trial_freqs = trial_freqs[range(len(trial_freqs)-1)]

		trial_freqs = trial_freqs[np.isin(trial_freqs,tones_to_use)]

	sessionfile.tuning.trial_freqs = trial_freqs
	return sessionfile






def testInOrderTonePresentation(sessionfile):
	number_tone_presentations = len(sessionfile.tuning.all_tone_times)
	tuning_tone_idxs = list(range(sessionfile.meta.length_in_trials,number_tone_presentations))
	toneCounts = sessionfile.trials.raw_audio_threshold_crossings[tuning_tone_idxs]
	toneFreqs = toneCounts * 10
	
	all_tones_tested = 0
	if (len(toneFreqs) == 250):    
		#Find 4k
		tones_4k = np.less(np.abs(toneFreqs-4000),500)
		tones_4k = tones_4k[0:50]
		
		#Find 8k
		tones_8k = np.less(np.abs(toneFreqs-8000),500)
		tones_8k = tones_8k[50:100]
		
		all_tones_tested = np.mean(np.concatenate((tones_4k,tones_8k)))
		
	elif (len(toneFreqs) == 350 or len(toneFreqs) == 450):
		#Find 4k
		tones_4k = np.less(np.abs(toneFreqs-4000),500)
		tones_4k = tones_4k[0:50]

		#Find 5k
		tones_5k = np.less(np.abs(toneFreqs-5648),500)
		tones_5k = tones_5k[50:100]

		#Find 8k
		tones_8k = np.less(np.abs(toneFreqs-8000),500)
		tones_8k = tones_8k[100:150]

		#Find 11k
		tones_11k = np.less(np.abs(toneFreqs-11260),500)
		tones_11k = tones_11k[150:200]
		
		all_tones_tested = np.mean(np.concatenate((tones_4k,tones_5k,tones_8k,tones_11k)))

	if all_tones_tested > 0.9:
		return True
	else:
		return False
		
	#Can't test 16k
	#Can't test 22k
	#Can't test 45k
	#Can't test 64k