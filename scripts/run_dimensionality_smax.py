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
import random




results_params = []
results_async = []
def log_result(result, final_list=results_async):
    final_list.append(result)

def log_error(x, final_list=results_async):
    final_list.append({})






if __name__ == "__main__":
    full_dir='/bgfs/balbanna/jmt195'
    print("Starting experiment_run_dimensionality.py")

    CACHE_DIRECTORY = sys.argv[1]
    OUTPUT_DIRECTORY = sys.argv[2]
    ENSEMBLE_SIZE = int(sys.argv[3])
    BIN_LENGTH = float(sys.argv[4])
    SAMPLING_COEFFICIENT = int(sys.argv[5])
    MAX_ITERS = int(sys.argv[6])
    OUTPUT_FILENAME = sys.argv[7]
    pool = mp.Pool(mp.cpu_count())

    ####################################################################################################################################

    try:
        with open('/bgfs/balbanna/jmt195/trialsToUsePerDay', 'rb') as f:
            trialsPerDayLoaded = pickle.load(f)
    except Exception as e:
        print(f"run_dimm line 51: {e}")
        raise e

    ####################################################################################################################################

    EnumSession = []
    EnumEnsembles = []
    sessions = os.listdir(CACHE_DIRECTORY)
    for session in sessions:
        sessionfile = ilep.loadSessionCached(CACHE_DIRECTORY,session)

        if sessionfile.meta.task == 'passive no behavior':
            continue
        if sessionfile.meta.task in ['tuning nonreversal','tuning switch','tuning reversal']:
            continue
        if sessionfile.meta.region != 'AC':
            continue
        
        try:
            ensemble = sessionfile.clusters.good
            EnumSession.append(session)
            EnumEnsembles.append(ensemble)
        except Exception as e:
            print(f"Encountered exception in session {session}: {e}")

    ### Save generated Enum lists
    try:
        with open(EnumSessionFilename, 'wb') as f:
            pickle.dump(EnumSession, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(EnumEnsemblesFilename, 'wb') as f:
            pickle.dump(EnumEnsembles, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"run_dimm line 106: {e}")

    ####################################################################################################################################

    progress_bar = tqdm(zip(EnumSession,EnumEnsembles), desc=f"Calculating {BIN_LENGTH}ms dimensionality")
    for session,ensemble in progress_bar:

        results = {}
        results['bin_length'] = BIN_LENGTH
        results['size'] = len(ensemble)
        results['session'] = session
        results['clust'] = ensemble
        results['sampling_coefficient'] = SAMPLING_COEFFICIENT
        results['max_iters'] = MAX_ITERS
        try:
            progress_bar.write(f"{session} ensemble {ensemble} is present.")
            
            #Need to create interval
            temp = pool.apply_async(ilep.calculateDimensionalityParallel,(session,CACHE_DIRECTORY,OUTPUT_DIRECTORY,ensemble,trialsPerDayLoaded,BIN_LENGTH))
            progress_bar.write(f"{session} ensemble {ensemble}: {temp}")
            results_async.append(temp)
            results_params.append(results)

        except Exception as e:
            progress_bar.write(f"{session},{ensemble}: {e}")
            continue

    # Closing the worker pool
    pool.close()
    pool.join()

    results_2 = []
    for r in results_async:
        try:
            results_2.append(r.get())
        except Exception as e:
            print(f"Exception occurred: {e}")
            results_2.append({})
    
    progress_bar.write(f"results_2: {results_2}")
    progress_bar.write(f"results_async: {results_async}")

    # Combining results
    total_results = [{**d1, **d2} for d1, d2 in zip(results_params, results_2)]

    # Saving as CSV
    results_df = pd.DataFrame(total_results)

    variables=['session']
    results_df = results_df.sort_values(by=variables).reset_index(drop=True)
    results_df.to_csv(os.path.join(full_dir, OUTPUT_FILENAME+'.csv'))