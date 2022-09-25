#
# Analysis Shuffle Seeds
#
# Peter Turney, September 11, 2022
#
# Given a set of evolved seeds, shuffle each evolved
# seed and then make the evolved seed compete with its
# corresponding shuffled seed. Report the fitness
# of the evolved seed, relative to the shuffled seed.
# Note that the evolved seed and the shuffled seed 
# necessarily have the same area and the same number
# of 1s.
#
import golly as g
import model_classes as mclass
import model_functions as mfunc
import model_parameters as mparam
import numpy as np
import scipy.stats as st
import pickle
import random
import os
import sys
#
# Set some paths for finding the input evolved seeds
# and for writing the output results. The directory
# is specified relative to the location of this file
# (analysis_shuffle_seeds.py).
#
run_dir = "../Experiments"
#
# The file where the results will be stored.
#
results_file = run_dir + "/analysis_shuffle_seeds.txt"
results_handle = open(results_file, "w")
results_handle.write("gen_num\tlower_95\tmean\tupper_95\n")
#
# The runs are stored in pickle_dir in subdirectories of
# the form "run1", "run2", etc. We only need to know the
# total number of runs.
#
num_runs = 40 # 0 to 39
#
# Each run contains results for each of the generations
# in a run. The first generation is generation 0. The last
# generation is specified here:
#
num_gens = 101 # 0 to 100
#
# Size of one generation.
#
pop_size = mparam.pop_size # 0 to 199
#
# Parameters for competitions between seeds.
#
width_factor = mparam.width_factor
height_factor = mparam.height_factor
time_factor = mparam.time_factor
num_trials = mparam.num_trials
#
# Sample probability for pickles.
#
sample_rate = 0.2 # 200 pickles * 0.2 = 40
#
# Read, analyze, and write the data one generation at a time.
#
for gen_num in range(num_gens):
  score_list = []
  for run_num in range(num_runs): # 0 to 39
    # find the subdirectory for run_num
    pickle_dir = run_dir + "/run" + str(run_num + 1) # 1 to 40
    # find the pickle for gen_num
    for file_name in os.listdir(pickle_dir):
      if (file_name.startswith("log-") and 
          file_name.endswith("-" + str(gen_num) + ".bin")):
        pickle_file = file_name
        break
    assert (file_name.startswith("log-") and 
            file_name.endswith("-" + str(gen_num) + ".bin"))
    # load the pickle
    pickle_path = pickle_dir + "/" + pickle_file
    pickle_handle = open(pickle_path, "rb") # rb = read binary
    pickle_sample = pickle.load(pickle_handle)
    pickle_handle.close()
    # pickle_sample is a list of pop_size seeds
    for pickle_num in range(pop_size): # 0 to 199
      # use sample_rate to choose a subset of the pickles (so that this
      # script does not run too long)
      if (random.uniform(0, 1) < sample_rate):
        evolved_seed = pickle_sample[pickle_num]
        # make the seed compete with shuffled versions of itself
        random_seed = evolved_seed.shuffle()
        # compare the evolved seed to the random seed
        [random_score, evolved_score] = mfunc.score_pair(g, random_seed, \
          evolved_seed, width_factor, height_factor, time_factor, num_trials)
        # append evolved_score to the list
        score_list.append(evolved_score)
  #
  # Convert the score_list to the form <gen_num, lower 95%, middle, upper 95%>
  #
  (lower95, upper95) = st.t.interval(confidence=0.95, df=len(score_list)-1, \
    loc=np.mean(score_list), scale=st.sem(score_list))
  middle = (lower95 + upper95) / 2.0
  results_handle.write(str(gen_num) + "\t" + str(lower95) + "\t" + \
    str(middle) + "\t" + str(upper95) + "\n")
  # force output, so we can see what's happening while it runs
  results_handle.flush()
  #
results_handle.close()
#
#