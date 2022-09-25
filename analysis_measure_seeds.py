#
# Analysis Measure Seeds
#
# Peter Turney, September 13, 2022
#
# Given a set of evolved seeds, measure the area
# of each seed and the number of ones in each seed.
# Report the average area and ones for each generation.
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
g.show("Measuring seeds ... Please wait a minute ..")
g.update()
#
# The file where the results will be stored.
#
results_file = run_dir + "/analysis_measure_seeds.txt"
results_handle = open(results_file, "w")
results_handle.write("gen_num\tavg_area\tavg_ones\tavg_density\n")
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
# Read, analyze, and write the data one generation at a time.
#
for gen_num in range(num_gens):
  area_list = []
  ones_list = []
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
      # measure the area of the seed
      seed = pickle_sample[pickle_num]
      area = seed.xspan * seed.yspan
      area_list.append(area)
      # count the number of cells in the seed that
      # have the value 1
      ones = seed.count_ones()
      ones_list.append(ones)
  #
  # Write <gen_num, avg_area, avg_ones>
  #
  avg_area = sum(area_list) / len(area_list)
  avg_ones = sum(ones_list) / len(ones_list)
  avg_density = avg_ones / avg_area
  results_handle.write(str(gen_num) + "\t" + str(avg_area) + "\t" + \
    str(avg_ones) + "\t" + str(avg_density) + "\n")
  #
results_handle.close()
g.show("... done measuring seeds.")
#
#