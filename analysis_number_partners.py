#
# Analysis Number Partners
#
# Peter Turney, September 15, 2022
#
# For each generation, report the average number
# of partners in a symbiote.
#
import golly as g
import model_classes as mclass
import model_functions as mfunc
import model_parameters as mparam
import numpy as np
import statistics
import pickle
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
results_file = run_dir + "/analysis_number_partners.txt"
results_handle = open(results_file, "w")
results_handle.write("gen_num\tavg\tdev\n")
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
  partner_counts = []
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
      # count the number of partners
      seed = pickle_sample[pickle_num]
      seed_map = mfunc.region_map(seed)
      # NOTE: below it is absolutely necessary to convert np.amax(seed_map)
      # to a floating number so that we can accurately caculate averages!!
      num_parts = float(np.amax(seed_map))
      # add the count to the list
      partner_counts.append(num_parts)
      # write a little progress report in the Golly window bar
      g.show(str(gen_num) + " " + str(num_parts) + " " + str(len(partner_counts)))
  #
  # Convert partner_counts to the form <gen_num, avg, dev>
  #
  avg = statistics.mean(partner_counts)
  dev = statistics.stdev(partner_counts)
  results_handle.write(str(gen_num) + "\t" + str(avg) + "\t" + str(dev) + "\n")
  # force output, so we can see what's happening while it runs
  results_handle.flush()
  #
results_handle.close()
#
#