import sys
import os
from tqdm import tqdm
import pandas as pd
sys.path.append("..")
sys.path.append("/ihome/minsanally/jmt195/EphysAnalysis/")
import InsanallyLabEphysTools as ilep

from collections import OrderedDict
from itertools import product
import multiprocessing as mp
import numpy as np
import pickle




results_params = []
results_async = []
def log_result(result, final_list=results_async):
    final_list.append(result)

def log_error(x, final_list=results_async):
    final_list.append({})






if __name__ == "__main__":
    full_dir='/bgfs/balbanna/jmt195'
    print("Starting experiment_run_choice_decoding.py")

    CACHE_DIRECTORY = sys.argv[1]
    OUTPUT_DIRECTORY = sys.argv[2]
    REPETITIONS = int(sys.argv[3])
    CATEGORIES = sys.argv[4]

    pool = mp.Pool(mp.cpu_count())

    ####################################################################################################################################
    n_reps = range(REPETITIONS)

    EnumSession = []
    EnumClust = []
    sessions = os.listdir(CACHE_DIRECTORY)
    for session in sessions:
        sessionfile = ilep.loadSessionCached(CACHE_DIRECTORY,session)

        if sessionfile.meta.task == 'passive no behavior':
            continue
        if sessionfile.meta.task in ['tuning nonreversal','tuning switch','tuning reversal']:
            continue
        if sessionfile.meta.region != 'AC':
            continue
        # if sessionfile.meta.animal not in ['BS_40','BS_41','BS_42','BS_49','BS_50','BS_51','BS_56','BS_59','BS_67','BS_70','BS_72','BS_87','BS_108','DS_15','DS_19','AE_238','AE_239','AE_240']:
        #     continue
        
        for clust in sessionfile.clusters.good:
            EnumSession.append(session)
            EnumClust.append(clust)

    ####################################################################################################################################

    #EnumSession = EnumSession[0:16]
    #EnumClust = EnumClust[0:16]

    try:
        with open('/bgfs/balbanna/jmt195/trialsToUsePerDay', 'rb') as f:
            trialsPerDayLoaded = pickle.load(f)
    except Exception as e:
        print(e)
        raise e

    ####################################################################################################################################

    progress_bar = tqdm(zip(EnumSession,EnumClust), desc=f"Calculating {CATEGORIES} decoding")
    for session,clust in progress_bar:


        results = {}
        results['n_rep'] = REPETITIONS
        results['categories'] = CATEGORIES
        results['session'] = session
        results['clust'] = clust
        try:
            progress_bar.write(f"{session} cluster {clust} is present.")
            
            #Need to create interval
            trainInterval = ilep.TrialInterval(-0.2*30000,0,False,True)
            testInterval = ilep.TrialInterval(0,0,False,True)
            temp = pool.apply_async(ilep.calculateDecodingForSingleNeuron,(session,clust,trialsPerDayLoaded,CACHE_DIRECTORY,OUTPUT_DIRECTORY,trainInterval,testInterval,REPETITIONS,CATEGORIES))
            progress_bar.write(f"{session} cluster {clust} result: {temp}")
            results_async.append(temp)
            results_params.append(results)

        except Exception as e:
            progress_bar.write(f"{session},{clust} error: {e}")
            continue

    # Closing the worker pool
    pool.close()
    pool.join()

    results_2 = []
    for r in results_async:
        try:
            results_2.append(r.get())
        except Exception as e:
            results_2.append({})
    
    progress_bar.write(f"results_2: {results_2}")
    progress_bar.write(f"results_async: {results_async}")

    # Combining results
    total_results = [{**d1, **d2} for d1, d2 in zip(results_params, results_2)]

    # Saving as CSV
    results_df = pd.DataFrame(total_results)

    variables=['session','clust']
    results_df = results_df.sort_values(by=variables).reset_index(drop=True)
    results_df.to_csv(os.path.join(full_dir, "choicedecoding.csv"))