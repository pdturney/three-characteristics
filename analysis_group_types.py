#
# Analysis Group Types
#
# Peter Turney, September 14, 2022
#
# For each generation (each population, size 200), divide the
# population into five overlapping groups:
#
# - management (one manager)
# - mutualism (no outsiders)
# - interaction (no soloists)
# - management & mutualism & interaction (all together)
# - singleton (no symbiosis)
# - symbiote (all symbiotes)
#
# Plot the percentage of these five groups over the course of
# generations 0 to 100.
#
import golly as g
import model_classes as mclass
import model_functions as mfunc
import model_parameters as mparam
import numpy as np
import scipy.stats as st
import pickle
import os
import sys
#
# Set some paths for finding the input seeds
# and for writing the output results. The directory
# is specified relative to the location of this file
# (analysis_group_types.py).
#
run_dir = "../Experiments"
#
# The file where the results will be stored.
#
results_file = run_dir + "/analysis_group_types.txt"
results_handle = open(results_file, "w")
results_handle.write("gen_num\tmanagement_pct\tmutualism_pct\t" + 
  "interaction_pct\tman_mut_int_pct\tsingleton_pct\tsymbiote_pct\n")
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
# Number of steps for using Management, Mutualism, and 
# Interaction.
#
num_steps = 1000
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
#
# Colour weights proportional to red's contribution:
# [red_weight, blue_weight, orange_weight, green_weight]
#
colour_weights = [1, 0, 2/3, 1/3]
#
# Calculate the sample size for one generation.
#
sample_size = pop_size * num_runs # 200 x 40 = 8,000
#
# Read, analyze, and write the data one generation at a time.
#
for gen_num in range(num_gens):
  seed_list = [] # will grow to pop_size * num_runs (200 x 40 = 8,000)
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
      seed = pickle_sample[pickle_num]
      seed_list.append(seed)
  #
  # For each seed in the current generation, figure out the groups
  # that the seed belongs in.
  #
  management_count = 0
  mutualism_count = 0
  interaction_count = 0
  man_mut_int_count = 0
  singleton_count = 0
  symbiote_count = 0
  #
  for seed in seed_list:
    #
    # management (one manager)
    #
    if (mfunc.check_for_borders(seed) == True):
      [manager_count, worker_count, manager_growth, worker_growth] = \
        mfunc.seed_management_stats(g, seed, num_steps)
      if (manager_count == 1):
        # exactly one manager: no confusion
        management_count += 1
    #
    # mutualism (no outsiders)
    #
    if (mfunc.check_for_borders(seed) == True):
      [insider_count, outsider_count, insider_growth, outsider_growth] = \
        mfunc.seed_mutualism_stats(g, seed, num_steps, colour_weights)
      if (outsider_count == 0):
        # complete mutualism: no outsiders
        mutualism_count += 1
    #
    # interaction (no soloists)
    #
    if (mfunc.check_for_borders(seed) == True):
      [ensemble_count, soloist_count, ensemble_growth, soloist_growth] = \
        mfunc.seed_interaction_stats(g, seed, num_steps)
      if (soloist_count == 0):
        # complete ensemble: no soloists
        interaction_count += 1
    #
    # management & mutualism & interaction (all together)
    #
    if (mfunc.check_for_borders(seed) == True):
      if ((manager_count == 1) and (outsider_count == 0) and \
          (soloist_count == 0)):
        man_mut_int_count += 1
    #
    # singletons (no symbiosis)
    #
    if (mfunc.check_for_borders(seed) == False):
      # seed is a singleton
      singleton_count += 1
    #
    # symbiotes (all symbiotes)
    #
    if (mfunc.check_for_borders(seed) == True):
      # seed is a symbiote
      symbiote_count += 1
    #
  #
  # Write out the percents for each generation.
  #
  management_pct = 100 * management_count / sample_size
  mutualism_pct = 100 * mutualism_count / sample_size
  interaction_pct = 100 * interaction_count / sample_size
  man_mut_int_pct = 100 * man_mut_int_count / sample_size
  singleton_pct = 100 * singleton_count / sample_size
  symbiote_pct = 100 * symbiote_count / sample_size
  #
  results_handle.write(str(gen_num) + "\t" + str(management_pct) + "\t" + \
    str(mutualism_pct) + "\t" + str(interaction_pct) + "\t" + \
    str(man_mut_int_pct) + "\t" + str(singleton_pct) + "\t" + \
    str(symbiote_pct) + "\n")
  # force output, so we can see what's happening while it runs
  results_handle.flush()
  #
results_handle.close()
#
#