"""
Model Functions

Peter Turney, September 11, 2022
"""
import golly as g
import model_classes as mclass
import model_parameters as mparam
import random as rand
import numpy as np
import copy
import time
import pickle
import os
import re
import sys
import pyautogui # tool for taking photos of the screen
"""
Various functions for working with Golly
"""
#
# show_message(g, log_handle, message) -- returns NULL
#
def show_message(g, log_handle, message):
  """
  A function for writing a message to both the Golly window
  and the log file.
  """
  log_handle.write(message)
  g.show(message)
#
# set_mag(g) -- returns mag
#
def set_mag(g):
  """
  A function for setting the Golly screen magnification.
  """
  # the maximum of the X span and the Y span
  g_maxspan = np.amax([g.getwidth(), g.getheight()])
  # the Golly magnification ratio is 2 to the power of mag
  if (g_maxspan < 80):
    mag = 5 # 2^5 = 32
  elif (g_maxspan < 160):
    mag = 4 # 2^4 = 16
  elif (g_maxspan < 320):
    mag = 3 # 2^3 = 8
  elif (g_maxspan < 640):
    mag = 2 # 2^2 = 4
  elif (g_maxspan < 1280):
    mag = 1 # 2^1 = 2
  else:
    mag = 0 # 2^0 = 1
  return mag
#
# show_parameters() -- returns a list of parameters and values
#
def show_parameters():
  """
  Make a list of the parameters in mparam and show
  the value of each parameter.
  """
  parameter_names = sorted(dir(mparam))
  display_list = []
  for name in parameter_names:
    # skip over system names
    # - system names have the form "__file__"
    if (name[0] != "_"): 
      value = str(getattr(mparam, name))
      display_list.append(name + " = " + value)
  return display_list
#
# get_minmax(g) -- returns [g_xmin, g_xmax, g_ymin, g_ymax]
#
def get_minmax(g):
  """
  Calculate the min and max of the Golly toroid coordinates.
  """
  # get height and width
  g_xspan = g.getwidth()
  g_yspan = g.getheight()
  # calculate min and max
  g_xmin = - int(g_xspan / 2)
  g_xmax = g_xspan + g_xmin
  g_ymin = - int(g_yspan / 2)
  g_ymax = g_yspan + g_ymin
  #
  return [g_xmin, g_xmax, g_ymin, g_ymax]
#
# count_pops(g) -- returns [count1, count2]
#
def count_pops(g):
  """
  Count the populations of red/orange and blue/green.
  """
  #
  # 0 = dead                       = white
  # 1 = player 1 alone             = red
  # 2 = player 2 alone             = blue
  # 3 = player 1 with interaction  = orange (red + yellow)
  # 4 = player 2 with interaction  = green (blue + yellow)
  # 5 = border marker              = purple
  #
  # find the min and max of the Golly toroid coordinates
  [g_xmin, g_xmax, g_ymin, g_ymax] = get_minmax(g)
  #
  count1 = 0 # red/orange
  count2 = 0 # blue/green
  #
  for x in range(g_xmin, g_xmax):
    for y in range(g_ymin, g_ymax):
      a = g.getcell(x, y)
      if (a == 1) or (a == 3):
        count1 += 1
      elif (a == 2) or (a == 4):
        count2 += 1
  #
  return [count1, count2]
#
# count_colours(g) -- returns [red, blue, orange, green]
#
def count_colours(g):
  """
  Count the number of cells for each of the colours. We only
  care about red, blue, orange, and green.
  """
  #
  # 0 = dead                       = white
  # 1 = player 1 alone             = red
  # 2 = player 2 alone             = blue
  # 3 = player 1 with interaction  = orange (red + yellow)
  # 4 = player 2 with interaction  = green (blue + yellow)
  # 5 = border marker              = purple
  #
  # find the min and max of the Golly toroid coordinates
  [g_xmin, g_xmax, g_ymin, g_ymax] = get_minmax(g)
  #
  red    = 0
  blue   = 0
  orange = 0
  green  = 0
  #
  for x in range(g_xmin, g_xmax):
    for y in range(g_ymin, g_ymax):
      a = g.getcell(x, y)
      if (a == 1):
        red += 1
      elif (a == 2):
        blue += 1
      elif (a == 3):
        orange += 1
      elif (a == 4):
        green += 1
  #
  return [red, blue, orange, green]
#
# initialize_population(pop_size, s_xspan, s_yspan, seed_density)
# -- returns population
#
def initialize_population(pop_size, s_xspan, s_yspan, seed_density):
  """
  Randomly initialize the population of seeds.
  """
  #
  # Initialize the population: a list of seeds.
  #
  # Here a seed is an initial Game of Life pattern (it is
  # not a random number seed).
  #
  population = []
  #
  for i in range(pop_size):
    # Make an empty seed (all zeros). 
    seed = mclass.Seed(s_xspan, s_yspan, pop_size) 
    # Randomly set some cells to state 1 (red).
    seed.randomize(seed_density)  
    # Set the count of living cells.
    seed.num_living = seed.count_ones()
    # Set the position of the new seed in the population array.
    seed.address = i 
    # Assign a unique ID number to the new seed.
    seed.unique_ID_num = i
    # Set the birth type to "random", since the initial population
    # is entirely random.
    seed.birth_type = "random"
    # Set the parent ID numbers to -1, since the initial random
    # seeds have no parents.
    seed.parent_A_ID_num = -1
    seed.parent_B_ID_num = -1
    # Add the seed to the population.
    population.append(seed) 
    #
  return population
#
# dimensions(s1, s2, width_factor, height_factor, time_factor)
# -- returns [g_width, g_height, g_time]
#
def dimensions(s1, s2, width_factor, height_factor, time_factor):
  """
  Define the dimensions of the Golly universe, based on the
  sizes of the two seeds and various multiplicative factors.
  """
  #
  # Suggested values:
  #
  #   width_factor  = 6.0
  #   height_factor = 3.0
  #   time_factor   = 6.0
  #
  assert width_factor > 2.0 # need space for two seeds, left and right
  assert height_factor > 1.0 # need space for tallest seed
  assert time_factor > 1.0 # time should increase with increased space
  #
  # Find the maximum of the dimensions of the two seeds.
  #
  max_size = np.amax([s1.xspan, s1.yspan, s2.xspan, s2.yspan])
  #
  # Apply the various factors.
  #
  g_width = int(max_size * width_factor)
  g_height = int(max_size * height_factor)
  g_time = int((g_width + g_height) * time_factor)
  #
  return [g_width, g_height, g_time]
#
# score_pair(g, seed1, seed2, width_factor, height_factor, \
#   time_factor, num_trials) -- returns [score1, score2]
#
def score_pair(g, seed1, seed2, width_factor, height_factor, \
  time_factor, num_trials):
  """
  Put seed1 and seed2 into the Immigration Game g and see which 
  one wins and which one loses. Note that this function does
  not update the histories of the seeds. For updating histories,
  use update_history().
  """
  #
  # Make copies of the original two seeds, so that the following
  # manipulations do not change the originals.
  #
  s1 = copy.deepcopy(seed1)
  s2 = copy.deepcopy(seed2)
  #
  # Verify that the seeds are limited to 0 (white), 1 (red), and
  # 5 (purple).
  #
  assert s1.check_colour() == True
  assert s2.check_colour() == True
  #
  # Check the number of living cells in the seeds. If the number
  # is zero, it is probably a mistake. The number is initially
  # set to zero and it should be updated when the seed is filled
  # with living cells. We could use s1.count_ones() here, but
  # we're trying to be efficient by counting only once and 
  # storing the count.
  #
  assert s1.num_living > 0
  assert s2.num_living > 0
  #
  # Initialize scores
  #
  score1 = 0.0
  score2 = 0.0
  #
  # Run several trials with different rotations and locations.
  #
  for trial in range(num_trials):
    #
    # Randomly rotate and flip s1 and s2
    #
    s1 = s1.random_rotate()
    s2 = s2.random_rotate()
    #
    # Switch cells in the second seed (s2) from state 1 (red) to state 2 (blue)
    #
    s2.red2blue()
    #
    # Rule file
    #
    rule_name = "Management"
    #
    # Set toroidal universe of height yspan and width xspan
    # Base the s1ze of the universe on the s1zes of the seeds
    #
    # g = the Golly universe
    #
    [g_width, g_height, g_time] = dimensions(s1, s2, \
      width_factor, height_factor, time_factor)
    #
    # set algorithm -- "HashLife" or "QuickLife"
    #
    g.setalgo("QuickLife") # use "HashLife" or "QuickLife"
    g.autoupdate(False) # do not update the view unless requested
    g.new(rule_name) # initialize cells to state 0
    g.setrule(rule_name + ":T" + str(g_width) + "," + str(g_height)) # make a toroid
    #
    # Find the min and max of the Golly toroid coordinates
    #
    [g_xmin, g_xmax, g_ymin, g_ymax] = get_minmax(g)
    #
    # Set magnification for Golly viewer
    #
    g.setmag(set_mag(g))
    #
    # Randomly place seed s1 somewhere in the left s1de of the toroid
    #
    s1.insert(g, g_xmin, -1, g_ymin, g_ymax)
    #
    # Randomly place seed s2 somewhere in the right s1de of the toroid
    #
    s2.insert(g, +1, g_xmax, g_ymin, g_ymax)
    #
    # Run for a fixed number of generations.
    # Base the number of generations on the sizes of the seeds.
    # Note that these are generations ins1de one Game of Life, not
    # generations in an evolutionary sense. Generations in the 
    # Game of Life correspond to growth and decay of a phenotype,
    # whereas generations in evolution correspond to the reproduction
    # of a genotype.
    #
    g.run(g_time) # run the Game of Life for g_time time steps
    g.update() # need to update Golly to get counts
    #
    # Count the populations of the two colours. State 1 = red = seed1.
    # State 2 = blue = seed2.
    #
    [count1, count2] = count_pops(g)
    #
    # We need to make an adjustment to these counts. We don't want to 
    # use the total count of living cells; instead we want to use
    # the increase in the number of living cells over the course of
    # the contest between the two organisms. The idea here is that
    # we want to reward seeds according to their growth during the
    # contest, not according to their initial states. This should
    # avoid an evolutionary bias towards larger seeds simply due
    # to size rather than due to functional properties. It should
    # also encourage efficient use of living cells, as opposed to
    # simply ignoring useless living cells.
    #
    # s1.num_living = initial number of living cells in s1
    # s2.num_living = initial number of living cells in s2
    #
    if (s1.num_living < count1):
      count1 = count1 - s1.num_living
    else:
      count1 = 0
    #
    if (s2.num_living < count2):
      count2 = count2 - s2.num_living
    else:
      count2 = 0
    #
    # Now we are ready to determine the winner.
    #
    if (count1 > count2):
      score1 = score1 + 1.0
    elif (count2 > count1):
      score2 = score2 + 1.0
    else:
      score1 = score1 + 0.5
      score2 = score2 + 0.5
    #
  #
  # Normalize the scores
  #
  score1 = score1 / num_trials
  score2 = score2 / num_trials
  #
  return [score1, score2]
#
# score_management(g, seed1, seed2, width_factor, height_factor, \
#   time_factor, num_trials) -- returns [score1, score2]
#
def score_management(g, seed1, seed2, width_factor, height_factor, \
  time_factor, num_trials):
  """
  Put seed1 and seed2 into the Management Game g and see which 
  one wins and which one loses, based on orange and green counts. 
  Note that this function does not update the histories of the seeds. 
  For updating histories, use update_history().
  """
  #
  # Make copies of the original two seeds, so that the following
  # manipulations do not change the originals.
  #
  s1 = copy.deepcopy(seed1)
  s2 = copy.deepcopy(seed2)
  #
  # Check the number of living cells in the seeds. If the number
  # is zero, it is probably a mistake. The number is initially
  # set to zero and it should be updated when the seed is filled
  # with living cells. We could use s1.count_ones() here, but
  # we're trying to be efficient by counting only once and 
  # storing the count.
  #
  assert s1.num_living > 0
  assert s2.num_living > 0
  #
  # Initialize scores
  #
  score1 = 0.0
  score2 = 0.0
  #
  # Run several trials with different rotations and locations.
  #
  for trial in range(num_trials):
    #
    # Randomly rotate and flip s1 and s2
    #
    s1 = s1.random_rotate()
    s2 = s2.random_rotate()
    #
    # Switch cells in the second seed (s2) from state 1 (red) to state 2 (blue)
    #
    s2.red2blue()
    #
    # Rule file
    #
    rule_name = "Management"
    #
    # Set toroidal universe of height yspan and width xspan
    # Base the s1ze of the universe on the s1zes of the seeds
    #
    # g = the Golly universe
    #
    [g_width, g_height, g_time] = dimensions(s1, s2, \
      width_factor, height_factor, time_factor)
    #
    # set algorithm -- "HashLife" or "QuickLife"
    #
    g.setalgo("QuickLife") # use "HashLife" or "QuickLife"
    g.autoupdate(False) # do not update the view unless requested
    g.new(rule_name) # initialize cells to state 0
    g.setrule(rule_name + ":T" + str(g_width) + "," + str(g_height)) # make a toroid
    #
    # Find the min and max of the Golly toroid coordinates
    #
    [g_xmin, g_xmax, g_ymin, g_ymax] = get_minmax(g)
    #
    # Set magnification for Golly viewer
    #
    g.setmag(set_mag(g))
    #
    # Randomly place seed s1 somewhere in the left s1de of the toroid
    #
    s1.insert(g, g_xmin, -1, g_ymin, g_ymax)
    #
    # Randomly place seed s2 somewhere in the right s1de of the toroid
    #
    s2.insert(g, +1, g_xmax, g_ymin, g_ymax)
    #
    # Run for a fixed number of generations.
    # Base the number of generations on the sizes of the seeds.
    # Note that these are generations ins1de one Game of Life, not
    # generations in an evolutionary sense. Generations in the 
    # Game of Life correspond to growth and decay of a phenotype,
    # whereas generations in evolution correspond to the reproduction
    # of a genotype.
    #
    g.run(g_time) # run the Game of Life for g_time time steps
    g.update() # need to update Golly to get counts
    #
    # Count orange cells for red (s1, colour 1) and count green cells
    # for blue (s2, colour 2). We don't need to subtract their initial
    # count at t = 0, because their initial count is necessarily zero.
    #
    [red, blue, orange, green] = count_colours(g)
    #
    count1 = orange # the red seed is rewarded for orange cells
    count2 = green  # the blue seed is rewarded for green cells
    #
    # Now we are ready to determine the winner.
    #
    if (count1 > count2):
      score1 = score1 + 1.0
    elif (count2 > count1):
      score2 = score2 + 1.0
    else:
      score1 = score1 + 0.5
      score2 = score2 + 0.5
    #
  #
  # Normalize the scores
  #
  score1 = score1 / num_trials
  score2 = score2 / num_trials
  #
  return [score1, score2]
#
# update_history(g, pop, i, j, width_factor, height_factor, \
#   time_factor, num_trials) -- returns NULL
#
def update_history(g, pop, i, j, width_factor, height_factor, \
  time_factor, num_trials):
  """
  Put the i-th and j-th seeds into the Immigration Game g and
  see which one wins and which one loses. The history of the 
  seeds will be updated in pop.
  """
  #
  # If i == j, let's just call it a tie.
  #
  if (i == j):
    pop[i].history[i] = 0.5
    return
  #
  # Call score_pair()
  #
  [scorei, scorej] = score_pair(g, pop[i], pop[j], width_factor, \
    height_factor, time_factor, num_trials)
  #
  # Update pop[i] and pop[j] with the new scores. 
  #
  pop[i].history[j] = scorei
  pop[j].history[i] = scorej
  # 
  # returns NULL
  # 
#
# update_similarity(pop, i, j) -- returns NULL
#
def update_similarity(pop, i, j):
  """
  Calculate the similarity between the two given seeds and 
  update their internal records with the result.
  """
  #
  # If i == j, the similarity score is the maximum.
  #
  if (i == j):
    pop[i].similarities[i] = 1.0
    return
  #
  # Calculate the similarity and update the population record.
  #
  sim = similarity(pop[i], pop[j])
  pop[i].similarities[j] = sim
  pop[j].similarities[i] = sim
  # 
  # returns NULL
  # 
#
# find_top_seeds(population, sample_size) -- returns sample_pop
#
def find_top_seeds(population, sample_size):
  """
  Find the best (fittest) sample_size seeds in the population.
  """
  pop_size = len(population)
  assert pop_size >= sample_size
  assert sample_size > 0
  # calculate fitness for each seed in the population, from their history
  scored_pop = []
  for i in range(pop_size):
    item = [population[i].fitness(), population[i]]
    scored_pop.append(item)
  # sort population in order of decreasing fitness (reverse=True)
  scored_pop.sort(key = lambda x: x[0], reverse=True) # sort by fitness
  # take the top sample_size items from scored_pop and
  # remove their attached fitness numbers
  sample_pop = []
  for i in range(sample_size):
    sample_pop.append(scored_pop[i][1]) # drop fitness number
  # return the cleaned-up list of sample_size seeds
  return sample_pop
#
# random_sample(population, sample_size) -- returns sample_pop
#
def random_sample(population, sample_size):
  """
  Get a random sample of sample_size seeds from the population.
  """
  #
  # To avoid duplicates in the sample, randomize the order of the
  # population and then take the first sample_size seeds
  # from the randomized list.
  #
  pop_size = len(population)
  assert pop_size > sample_size
  assert sample_size > 0
  # attach a random number to each seed in the population
  randomized_pop = []
  for i in range(pop_size):
    # item = [random real number between 0 and 1, the i-th seed]
    item = [rand.uniform(0, 1), population[i]]
    randomized_pop.append(item)
  # sort randomized_pop in order of the attached random numbers
  randomized_pop.sort(key = lambda x: x[0]) # sort by random number
  # take the top sample_size items from randomized_pop and
  # remove their attached random numbers
  sample_pop = []
  for i in range(sample_size):
    sample_pop.append(randomized_pop[i][1]) # drop random number
  # return the cleaned-up list of sample_size seeds
  return sample_pop
#
# find_best_seed(sample) -- returns best_seed
#
def find_best_seed(sample):
  """
  In the list of seeds in sample, find the seed (not necessarily
  unique) with maximum fitness.
  """
  sample_size = len(sample)
  assert sample_size > 0
  best_seed = sample[0]
  best_score = best_seed.fitness()
  for i in range(len(sample)):
    if (sample[i].fitness() > best_score):
      best_seed = sample[i]
      best_score = best_seed.fitness()
  return best_seed
#
# find_worst_seed(sample) -- returns worst_seed
#
def find_worst_seed(sample):
  """
  In the list of seeds in sample, find the seed (not necessarily
  unique) with minimum fitness.
  """
  sample_size = len(sample)
  assert sample_size > 0
  worst_seed = sample[0]
  worst_score = worst_seed.fitness()
  for i in range(len(sample)):
    if (sample[i].fitness() < worst_score):
      worst_seed = sample[i]
      worst_score = worst_seed.fitness()
  return worst_seed
#
# average_fitness(sample) -- returns average
#
def average_fitness(sample):
  """
  Given a list of sample seeds, return their average fitness,
  relative to the whole population.
  """
  sample_size = len(sample)
  assert sample_size > 0
  total_fitness = 0.0
  for i in range(len(sample)):
    total_fitness = total_fitness + sample[i].fitness()
  average = total_fitness / sample_size
  return average
#
# archive_elite(population, elite_size, log_directory, log_name, run_id_number) 
# -- returns NULL
#
def archive_elite(population, elite_size, log_directory, log_name, run_id_number):
  """
  Store an archive file of the elite members of the population,
  for future testing. The elite consists of the top elite_size
  most fit seeds in the current population.
  """
  history_sample = find_top_seeds(population, elite_size)
  history_name = log_name + "-pickle-" + str(run_id_number)
  history_path = log_directory + "/" + history_name + ".bin"
  history_handle = open(history_path, "wb") # wb = write binary
  pickle.dump(history_sample, history_handle)
  history_handle.close()
  # 
  # returns NULL
  # 
#
# fusion_storage(s2, s3, s4, n) -- returns NULL
#
def fusion_storage(s2, s3, s4, n):
  """
  After fusion has occurred, store the parts (s2, s3) and their
  fusion (s4) in a binary file for future analysis and inspection.
  The seed s4 is the n-th child born.
  """
  # make a file name for storage
  fusion_path = mparam.log_directory + "/fusion_storage.bin"
  # "ab+" opens a file for both appending and reading in binary mode
  fusion_handle = open(fusion_path, "ab+")
  # store the seeds and the generation number
  pickle.dump(s2, fusion_handle)    # s2 is part of s4 (after rotation)
  pickle.dump(s3, fusion_handle)    # s3 is part of s4 (after rotation)
  pickle.dump(s4, fusion_handle)    # s4 is the fusion of s2 and s3
  pickle.dump(n, fusion_handle)     # s4 is n-th child
  # close handle
  fusion_handle.close()
  # 
  # returns NULL
  #
#
# seed_storage(seed) -- returns NULL
#
def seed_storage(seed):
  """
  Append the given seed to the specified file.
  """
  #
  # mparam.log_directory is the desired folder for storing the seed.
  # For example, mparam.log_directory might be "../Experiments/run1".
  # We want to append a unique subdirectory to mparam.log_directory, 
  # so that future experiments do not get mixed in to the file.
  #
  storage_path = mparam.log_directory + "/all_seed_storage.bin"
  # "ab+" opens a file for both appending and reading in binary mode
  storage_handle = open(storage_path, "ab+")
  pickle.dump(seed, storage_handle)
  storage_handle.close()
  # 
  # returns NULL
  #
#
# similarity(seed0, seed1) -- returns similarity
#
def similarity(seed0, seed1):
  """
  Measure the bit-wise similarity of two seeds. If the seeds
  have different sizes, return zero. If they have different
  borders (buffer zones), return zero.
  """
  # Make sure the seeds are the same size: the same number
  # of rows and columns.
  if (seed0.xspan != seed1.xspan):
    return 0.0
  if (seed0.yspan != seed1.yspan):
    return 0.0
  # Make sure that the seeds have the same borders: that is,
  # they should have matching purple states (state 5).
  for x in range(seed0.xspan):
    for y in range(seed0.yspan):
      if ((seed0.cells[x][y] == 5) and (seed1.cells[x][y] != 5)):
        return 0.0
      if ((seed0.cells[x][y] != 5) and (seed1.cells[x][y] == 5)):
        return 0.0
  # Initialize count.
  num_agree = 0.0
  # Count agreements.
  for x in range(seed0.xspan):
    for y in range(seed0.yspan):
      if (seed0.cells[x][y] == seed1.cells[x][y]):
        num_agree = num_agree + 1.0
  # Calculate a similarity score ranging from zero to one.
  similarity = num_agree / (seed0.xspan * seed0.yspan)
  # Return the degree of similarity between the two seeds.
  return similarity
#
# find_similar_seeds(target_seed, pop, min_similarity, max_similarity)
# -- returns similar_seeds
# 
def find_similar_seeds(target_seed, pop, min_similarity, max_similarity):
  """
  Given a target seed, find seeds in the population with similarities
  to the target in the range from min_similarity to max_similarity.
  This function assumes that target_seed is in the population and
  the list target_seed.similarities is up-to-date. 
  """
  similar_seeds = []
  for i in range(len(pop)):
    if ((target_seed.similarities[i] >= min_similarity) and \
      (target_seed.similarities[i] <= max_similarity) and \
      (target_seed.address != i)):
      similar_seeds.append(pop[i])
  # return the seeds that satisfy the conditions
  return similar_seeds
#
# mate(seed0, seed1) -- returns child_seed
#
def mate(seed0, seed1):
  """
  Apply crossover to seed0 and seed1. We only have one crossover point,
  because multiple crossover points would be more disruptive to the
  structure of the seeds.
  """
  # This function is designed with the assumption that the seeds are 
  # the same size.
  assert seed0.xspan == seed1.xspan
  assert seed0.yspan == seed1.yspan
  # Note the spans of seed0 and seed1.
  xspan = seed0.xspan
  yspan = seed0.yspan
  # Randomly swap the seeds. Because s0 is always the top part of
  # a split that cuts across the Y axis and the left part of a split 
  # that cuts across the X axis, we need to swap the seeds in order
  # to add some variety.
  if (rand.uniform(0, 1) < 0.5):
    s0 = seed0
    s1 = seed1
  else:
    s0 = seed1
    s1 = seed0
  # Initialize the child to zero.
  child_seed = mclass.Seed(xspan, yspan, mparam.pop_size) 
  # Randomly choose whether to split on the X axis or
  # the Y axis.
  if (rand.uniform(0, 1) < 0.5):
    # Choose the Y axis split point. There will always be
    # at least one row on either side of the split point.
    assert yspan > 1
    y_split_point = rand.randrange(yspan - 1)
    for x in range(xspan):
      for y in range(yspan):
        if (y <= y_split_point):
          child_seed.cells[x][y] = s0.cells[x][y]
        else:
          child_seed.cells[x][y] = s1.cells[x][y]
  else:
    # Choose the X axis split point. There will always be
    # at least one column on either side of the split point.
    assert xspan > 1
    x_split_point = rand.randrange(xspan - 1)
    for x in range(xspan):
      for y in range(yspan):
        if (x <= x_split_point):
          child_seed.cells[x][y] = s0.cells[x][y]
        else:
          child_seed.cells[x][y] = s1.cells[x][y]
  # Return the resulting child.
  return child_seed
#
# region_map(seed) -- returns a map of the regions in seed
#
def region_map(seed):
  """
  Given a seed, make a matrix of the same size as the seed,
  such that the matrix is a map of the regions in the seed.
  Each cell in the matrix will contain a number that uniquely
  identifies the region it belongs to, as determined by the
  purple borders in the seed.
  """
  # start with a map the same size as the seed, but all zeros
  map = np.zeros((seed.xspan, seed.yspan), dtype=np.int) 
  # write the purple borders (state 5) of the seed into the map
  # -- write -1 as the new border state, since we could have six or
  #    more regions and we want to assign a unique number to each region
  for x in range(seed.xspan):
    for y in range(seed.yspan):
      if (seed.cells[x][y] == 5):
        # the new border
        map[x][y] = -1
  # initialize some variables
  current_num = 1     # the number that marks the current region
  total_num = 1       # the total number of regions seen so far
  border_crossing = 0 # 0 = not currently crossing a border, 1 = crossing
  # scan through map, left to right, top to bottom
  for x in range(seed.xspan):
    for y in range(seed.yspan):
      # check to see if we are crossing a border
      if (map[x][y] == -1):
        border_crossing = 1
      # if the current cell is not a border, write current_num in the map
      if (border_crossing == 0):
        map[x][y] = current_num
      # if the border_crossing flag was triggered and we are not right
      # on top of the border, then look in the neighbourhood to see
      # if we have been here before
      if ((border_crossing == 1) and (map[x][y] != -1)):
        # look around
        for delta_x in [-1, 0, 1]:
          for delta_y in [-1, 0, 1]:
            near_x = x + delta_x
            near_y = y + delta_y
            if (near_x < 0):
              continue
            elif (near_x >= seed.xspan):
              continue
            elif (near_y < 0):
              continue
            elif (near_y >= seed.yspan):
              continue
            elif (map[near_x][near_y] > 0):
              # reset current_num to the value of the neighbour
              current_num = map[near_x][near_y]
              # write current_num in the map
              map[x][y] = current_num
              # reset the border flag
              border_crossing = 0
              # stop looking around
              break
      # if the border_crossing flag was triggered and looking in
      # the neighbourhood didn't help, then we must be in a new
      # region
      if ((border_crossing == 1) and (map[x][y] != -1)):
        total_num += 1
        current_num = total_num
        map[x][y] = current_num
        border_crossing = 0
    # incrementing x is like crossing a border, because y goes
    # back to zero
    border_crossing = 1
  # return the map
  return map
#
# extract_parts(seed, seed_map, target_region) -- returns target part of seed
#
def extract_parts(seed, seed_map, target_region):
  """
  Given a seed, a seed_map, and a target_region, extract the region of
  the seed that corresponds to cells in seed_map that have the value
  target_region. Reduce the size of the extracted seed by dropping empty
  outside rows and columns.
  """
  #
  # seed          = a seed object as defined in model_classes.py
  # seed_map      = an array where rectangular regions are given by cell values
  # target_region = the cell value that picks out the desired rectangular region
  #
  # find the size of the seed
  num_rows = seed_map.shape[0]
  num_cols = seed_map.shape[1]
  # check that seed and seed map have the same size
  assert seed.cells.shape[0] == seed_map.shape[0]
  assert seed.cells.shape[1] == seed_map.shape[1]
  # the target_region cannot be zero
  assert target_region != 0
  # make sure target_region is in seed_map
  assert target_region in seed_map
  # find first_row containing target_region
  first_row = 0
  for row in range(num_rows):
    if (target_region in seed_map[row, :]):
      first_row = row
      break
  # find last_row containing target_region
  last_row = num_rows - 1
  for row in range(num_rows-1, -1, -1):
    if (target_region in seed_map[row, :]):
      last_row = row
      break
  # find first_col containing target_region
  first_col = 0
  for col in range(num_cols):
    if (target_region in seed_map[:, col]):
      first_col = col
      break
  # find last_col containing target_region
  last_col = num_cols - 1
  for col in range(num_cols-1, -1, -1):
    if (target_region in seed_map[:, col]):
      last_col = col
      break
  # make a new seed with the new reduced size
  new_num_rows = last_row - first_row + 1 
  new_num_cols = last_col - first_col + 1
  new_seed = mclass.Seed(new_num_rows, new_num_cols, 0)
  # for each cell in seed_map that matches the value target_region,
  # if the corresponding value in seed is not zero, then write
  # a value of one into the appropriate location in new_seed
  for row in range(first_row, last_row + 1):
    for col in range(first_col, last_col + 1):
      if (seed.cells[row][col] > 0):
        new_seed.cells[row - first_row, col - first_col] = 1
  # return new_seed
  return new_seed
#
# found_empty_cells(seed) -- returns True (empty) or False (not empty)
#
def found_empty_cells(seed):
  """
  Check all the cells in the given seed to make sure that none of
  the cells are empty.
  """
  # first make a map of the seed
  seed_map = region_map(seed)
  # count the number of regions in the map
  num_regions = np.amax(seed_map)
  # examine each region
  for i in range(num_regions):
    # target_region ranges from 1 to num_regions (inclusive)
    target_region = i + 1
    # before we try to extract the seed part in the given
    # target region, let's verify that the target region
    # exists in the seed map
    if (not (target_region in seed_map)):
      return True # bad seed
    # now extract the seed part from the target region
    seed_part = extract_parts(seed, seed_map, target_region)
    # the cells in seed_part should not all be zero
    if (np.amax(seed_part.cells) == 0):
      # if seed_part is all zeros, then it is True that there
      # are empty cells
      return True # bad seed
  # if we reach here, then it is False that there are empty cells
  return False # good seed
#
# measure_growth_management(g, seed, num_steps)
# -- return [red_count, blue_count, orange_count, green_count]
#
def measure_growth_management(g, seed, num_steps):
  """
  Given a Management Game seed pattern with four live states,
  red (1), blue (2), orange (3), green (4), and two dead states,
  white (0), purple (5), run the pattern for num_steps and calculate
  four growth numbers, one for each live state.
  """
  #
  # make a matrix that is a map of the regions in seed; the map will
  # have the same size matrix as seed and each cell in the map will 
  # contain a number that uniquely defines the region it belongs to,
  # as determined by the purple borders in seed
  #
  seed_map = region_map(seed)
  num_rows = seed_map.shape[0]
  num_cols = seed_map.shape[1]
  #
  # find out how many different regions there are in the map
  # and then generate the p 1-vs-(p-1) colourings
  #
  num_regions = np.amax(seed_map)
  #
  # make a matrix for the final output, with one row for each region
  # (each part) in the seed, such that each row gives the growth
  # of the four colours
  #
  growth_matrix = []
  #
  for target_region in range(1, num_regions + 1):
    # make a copy of seed -- we will create a new colouring
    # for the copy
    seed_colouring = copy.deepcopy(seed)
    # - if a cell in the region map seed_map has the value target_region
    #   and the corresponding cell in seed_colouring is 1 or 2 (red or blue),
    #   then the corresponding cell in seed_colouring will be set
    #   to red (state 1)
    # - if a cell in the region map seed_map does not have the value target_region
    #   and the corresponding cell in seed_colouring is 1 or 2 (red or blue),
    #   then the corresponding cell in seed_colouring will be set
    #   to blue (state 2)
    # - if a cell in the region map seed_map has the value -1,
    #   then the corresponding cell in seed_colouring should be
    #   purple (state 5 -- the colour of the borders); if it is not
    #   purple, then signal an error
    for x in range(num_rows): 
      for y in range(num_cols):
        # state 1 -- red -- target_region
        if ((seed_map[x][y] == target_region) and \
           ((seed_colouring.cells[x][y] == 1) or \
           (seed_colouring.cells[x][y] == 2))):
          seed_colouring.cells[x][y] = 1
        # state 2 -- blue -- not target_region
        elif ((seed_map[x][y] != target_region) and \
             ((seed_colouring.cells[x][y] == 1) or \
             (seed_colouring.cells[x][y] == 2))):
          seed_colouring.cells[x][y] = 2
        # state 5 -- purple -- the border between regions
        elif (seed_map[x][y] == -1):
          assert seed_colouring.cells[x][y] == 5
    # initialize Golly
    rule_name = "Management" # the Management Game
    g.setalgo("QuickLife") # use "HashLife" or "QuickLife"
    g.autoupdate(False) # do not update the view unless requested
    g.new(rule_name) # initialize cells to state 0
    g.setrule(rule_name) # make an infinite plane
    # initialize the counts for the five states:
    # [white (0), red (1), blue (2), orange (3), green (4)]
    num_colours = 5
    start_size = [0, 0, 0, 0, 0] 
    end_size = [0, 0, 0, 0, 0]
    # copy seed into Golly 
    for x in range(num_rows):
      for y in range(num_cols):
        state = seed_colouring.cells[x][y]
        # ignore purple colours (state 5)
        if (state < 5):
          g.setcell(x, y, state)
          # update start_size
          start_size[state] += 1
    # run for the requested number of steps
    g.run(num_steps)
    g.update()
    # update end_size
    boundary = g.getrect()
    if (len(boundary) == 0): # if no live cells ...
      end_size = [0, 0, 0, 0, 0]
    else:
      cell_list = g.getcells(boundary)
      # if cell_list ends in 0, then delete the 0 -- note that stateN
      # will never be zero, since dead cells (state 0) are not included
      # in cell_list
      if (cell_list[-1] == 0):
        cell_list.pop()
      # end_size = [white (0), red (1), blue (2), orange (3), green (4)]
      end_size = [0, 0, 0, 0, 0] # initialize
      for (x, y, state) in zip(*[iter(cell_list)] * 3):
        # ignore purple colours (state 5)
        if (state < 5):
          # update count
          end_size[state] += 1
    # calculate growth
    growth_vector = []
    for colour_num in range(num_colours):
      growth_vector.append(end_size[colour_num] - start_size[colour_num])
    # update growth_matrix
    growth_matrix.append(growth_vector)
    #
  #
  return growth_matrix
#
# seed_mutualism_stats(g, fusion_seed, num_steps, colour_weights)
# -- return [insider_count, outsider_count, insider_growth, outsider_growth]
#
def seed_mutualism_stats(g, fusion_seed, num_steps, colour_weights):
  """
  Given a fusion seed with N parts, calculate how much each part
  grows on its own (with the other parts removed) compared with
  how much each part grows when it is together with the other parts.
  """
  #
  # for each part in a seed, calculate the growth of the part alone
  # and the growth of the part with the rest of the seed
  #
  # -- suggested weights for [red, blue, orange, green]:
  #
  #    [1, 0, 2/3, 1/3] -- proportional to red's contribution
  #    [1, 0, 1, 0]     -- the Immigration rule from red's view
  #
  red_weight = colour_weights[0]
  blue_weight = colour_weights[1]
  orange_weight = colour_weights[2]
  green_weight = colour_weights[3]
  #
  # make a map of the seed
  #
  seed_map = region_map(fusion_seed)
  num_parts = np.amax(seed_map)
  if (num_parts < 2):
    return [0, 0, 0, 0]
  #
  # - calculate growth of each part of fusion_seed in the context of all of 
  #   the other parts in the seed
  #
  growth_matrix = measure_growth_management(g, fusion_seed, num_steps)
  growth_in_seed = []
  for part_num in range(num_parts):
    # each row in growth_matrix is information about the i-th part of
    # a seed, consisting of counts of the colours in the i-th part
    # -- we don't actually care about white_grow, but it's more clear
    #    when the colour values match the array positions
    [white_grow, red_grow, blue_grow, orange_grow, green_grow] \
      = growth_matrix[part_num]
    red_score = red_grow * red_weight
    blue_score = blue_grow * blue_weight
    orange_score = orange_grow * orange_weight
    green_score = green_grow * green_weight
    total_score = red_score + blue_score + orange_score + green_score
    growth_in_seed.append(total_score)
  #
  # - calculate growth of each part of fusion_seed by itself, outside
  #   of the context of all of the other parts in the seed
  #
  growth_alone = []
  for part_num in range(num_parts):
    # part_num gives the target region to extract from the seed
    # -- the target_part will be changed to red (state 1)
    # -- the seed_map values range from 1 to N, but the part_num values
    #    range from 0 to N-1, so we need to shift part_num
    target_part = extract_parts(fusion_seed, seed_map, (part_num + 1))
    # calculate the growth of target_part
    growth_alone.append(measure_growth_life(g, target_part, num_steps))
  #
  # - mutualism has two aspects: "insiders" that grow more when inside
  #   the symbiote and "outsiders" that grow more when outside the symbiote
  #
  insider_count = 0
  outsider_count = 0
  insider_growth = 0
  outsider_growth = 0
  for part_num in range(num_parts):
    red_grow = growth_matrix[part_num][1]
    orange_grow = growth_matrix[part_num][3]
    if (growth_in_seed[part_num] > growth_alone[part_num]):
      insider_count += 1
      insider_growth += red_grow + orange_grow
    else:
      outsider_count += 1
      outsider_growth += red_grow + orange_grow
  #
  return [insider_count, outsider_count, insider_growth, outsider_growth]
#
# seed_interaction_stats(g, fusion_seed, num_steps)
# -- return [ensemble_count, soloist_count, ensemble_growth, soloist_growth]
#
def seed_interaction_stats(g, fusion_seed, num_steps):
  """
  Given a fusion seed with N parts, calculate for each part
  orange count + green count - red count. The count will be 
  positive when a part interacts and negative when a part does 
  not interact.
  """
  #
  # make a map of the seed
  #
  seed_map = region_map(fusion_seed)
  num_parts = np.amax(seed_map)
  if (num_parts < 2):
    return [0, 0, 0, 0]
  #
  # - interaction has two aspects: "soloists" who prefer to work on
  #   their own and "ensembles" who prefer to work together
  # 
  ensemble_count = 0
  soloist_count = 0
  ensemble_growth = 0
  soloist_growth = 0
  #
  # - calculate growth of each part of fusion_seed in the context of all of 
  #   the other parts in the seed
  #
  growth_matrix = measure_growth_management(g, fusion_seed, num_steps)
  for part_num in range(num_parts):
    # each row in growth_matrix is information about the i-th part of
    # a seed, consisting of counts of the colours in the i-th part
    # -- we don't actually care about white_grow, but it's more clear
    #    when the colour values match the array positions
    [white_grow, red_grow, blue_grow, orange_grow, green_grow] \
      = growth_matrix[part_num]
    # in the seed (t = 0), the i-th part is coloured red and all other
    # parts are coloured blue -- at t = 1000, we say the i-th part is
    # interactive if there are more orange and green cells than red cells
    # -- blue is not relevant, because red has no control over what
    # blue does
    interactivity = (orange_grow + green_grow) - red_grow
    if (interactivity > 0):
      ensemble_count += 1
      ensemble_growth += red_grow + orange_grow
    else:
      soloist_count += 1
      soloist_growth += red_grow + orange_grow
  #
  return [ensemble_count, soloist_count, ensemble_growth, soloist_growth]
#
# seed_management_stats(g, fusion_seed, num_steps)
# -- return [manager_count, worker_count, manager_growth, worker_growth]
#
def seed_management_stats(g, fusion_seed, num_steps):
  """
  Given a fusion seed with N parts, calculate the number of 
  managers.
  """
  #
  # make a map of the seed
  #
  seed_map = region_map(fusion_seed)
  num_parts = np.amax(seed_map)
  if (num_parts < 2):
    return [0, 0, 0, 0]
  #
  # management has two aspects: "managers" lead and "workers" follow
  #
  manager_count = 0
  worker_count = 0
  manager_growth = 0
  worker_growth = 0
  #
  # - calculate growth of each part of fusion_seed in the context of all of 
  #   the other parts in the seed
  #
  growth_matrix = measure_growth_management(g, fusion_seed, num_steps)
  for part_num in range(num_parts):
    # each row in growth_matrix is information about the i-th part of
    # a seed, consisting of counts of the colours in the i-th part
    # -- we don't actually care about white_grow, but it's more clear
    #    when the colour values match the array positions
    [white_grow, red_grow, blue_grow, orange_grow, green_grow] \
      = growth_matrix[part_num]
    count = orange_grow - green_grow
    if (count > 0):
      manager_count += 1
      manager_growth += red_grow + orange_grow
    else:
      worker_count += 1
      worker_growth += red_grow + orange_grow
  #
  return [manager_count, worker_count, manager_growth, worker_growth]
#
# test_mutual_benefit(g, seed) -- returns True or False
#
def test_mutual_benefit(g, seed):
  """
  If a seed is a symbiote, then we check each partner in the symbiote
  to see whether the partners benefit from being in the symbiote. If
  all partners benefit from being in the symbiote (that is, they all
  grow more when inside the symbiote than when outside the symbiote),
  then return True (meaning that all partners benefit). If one or more
  partners do not benefit from being in the symbiote, then return False.
  If a seed is not a symbiote, return True. 
  """
  # Count the number of regions in the map of the seed. 
  seed_map = region_map(seed)
  num_regions = np.amax(seed_map)
  # If the number of regions is one, then the seed is not a symbiote,
  # so return True.
  if (num_regions == 1):
    return True
  # Calculate mutualism statistics.
  #
  #                [red, blue,   orange,     green]
  colour_weights = [1.0, 0.0, (2.0 / 3.0), (1.0 / 3.0)]
  num_steps = 1000
  [insider_count, outsider_count, insider_growth, outsider_growth] = \
    seed_mutualism_stats(g, seed, num_steps, colour_weights)
  # If there are outsiders, then return False.
  if (outsider_count > 0):
    return False
  # Otherwise, return True.
  return True
#
# uniform_asexual(candidate_seed, pop, n, next_unique_ID_number) 
# -- returns [pop, message]
#
def uniform_asexual(candidate_seed, pop, n, next_unique_ID_number):
  """
  Create a new seed by randomly mutating an existing seed. The
  new seed is generated by selecting a parent seed and flipping
  bits in the parent. The size of the seed does not change; it
  is uniform.
  """
  # The most fit member of the tournament.
  s0 = candidate_seed
  # Mutate the best seed to make a new child. The only mutation
  # here is flipping bits.
  mutation_rate = mparam.mutation_rate
  s1 = copy.deepcopy(s0)
  s1.flip_bits(mutation_rate)
  # If there are empty cells in s1, then try again with uniform_asexual.
  if found_empty_cells(s1):
    return uniform_asexual(candidate_seed, pop, n, next_unique_ID_number)
  # Update the count of living cells.
  s1.num_living = s1.count_ones() 
  s1.unique_ID_num = next_unique_ID_number
  s1.birth_type = "uniform_asexual"
  s1.parent_A_ID_num = s0.unique_ID_num # the one and only parent of s1
  s1.parent_B_ID_num = -1 # there is no second parent
  # Find the least fit old seed in the population. It's not a problem
  # if there are ties.
  s2 = find_worst_seed(pop)
  # Now we have:
  #
  # s0 = fit parent seed
  # s1 = the mutated new child
  # s2 = the least fit old seed, which will be replaced by the mutated child
  #
  # Replace the least fit old seed in the population (s2) with the
  # new child (s1).
  i = s2.address # find the position of the old seed (s2)
  s1.address = i # copy the old position of the old seed into s1, the child
  pop[i] = s1 # replace s2 (old seed) in population (pop) with s1 (new child)
  # Build a history for the new seed, by matching it against all seeds
  # in the population.
  width_factor = mparam.width_factor
  height_factor = mparam.height_factor
  time_factor = mparam.time_factor
  num_trials = mparam.num_trials
  pop_size = len(pop)
  for j in range(pop_size):
    update_history(g, pop, i, j, width_factor, height_factor, \
      time_factor, num_trials)
    update_similarity(pop, i, j)
  # store the new seed
  seed_storage(s1)
  # Report on the new history of the new seed
  message = "Run: {}".format(n) + \
    "  Parent fitness (s0): {:.3f}".format(s0.fitness()) + \
    "  Child fitness (s1): {:.3f}".format(s1.fitness()) + \
    "  Replaced seed fitness (s2): {:.3f}\n".format(s2.fitness())
  # It is possible that s1 is worse than s2, if there was a bad mutation in s1.
  # Let's not worry about that, since s1 will soon be replaced if it is less
  # fit than the least fit seed (that is, s2).
  return [pop, message]
#
# variable_asexual(candidate_seed, pop, n, max_seed_area, 
#                  next_unique_ID_number) 
# -- returns [pop, message]
#
def variable_asexual(candidate_seed, pop, n, max_seed_area, 
                     next_unique_ID_number):
  """
  Create a new seed by randomly mutating, growing, and shrinking
  an existing seed. The new seed is generated by selecting a parent 
  seed and randomly flipping bits, removing rows and columns, or
  adding rows and columns. The size of the seed is variable; it 
  may increase or decrease in size.
  """
  # The most fit member of the tournament.
  s0 = candidate_seed
  # Mutate the best seed to make a new child. The mutations here
  # are flipping bits, removing rows and columns (shrinking), and
  # adding rows and columns (growing).
  prob_grow = mparam.prob_grow
  prob_flip = mparam.prob_flip
  prob_shrink = mparam.prob_shrink
  seed_density = mparam.seed_density
  mutation_rate = mparam.mutation_rate
  s1 = copy.deepcopy(s0)
  s1 = s1.mutate(prob_grow, prob_flip, prob_shrink, seed_density, mutation_rate)
  # If there are empty cells in s1, then try again with uniform_asexual.
  if found_empty_cells(s1):
    return uniform_asexual(candidate_seed, pop, n, next_unique_ID_number)
  # Update the count of living cells.
  s1.num_living = s1.count_ones() # update count of living cells
  s1.unique_ID_num = next_unique_ID_number
  s1.birth_type = "variable_asexual"
  s1.parent_A_ID_num = s0.unique_ID_num # the one and only parent of s1
  s1.parent_B_ID_num = -1 # there is no second parent
  # Make sure the area of the new seed is not greater than the maximum.
  # If it is too big, then default to uniform_asexual reproduction.
  if ((s1.xspan * s1.yspan) > max_seed_area):
    return uniform_asexual(candidate_seed, pop, n, next_unique_ID_number)
  # Find the least fit old seed in the population. It's not a problem
  # if there are ties.
  s2 = find_worst_seed(pop)
  # Now we have:
  #
  # s0 = fit parent seed
  # s1 = the mutated new child
  # s2 = the least fit old seed, which will be replaced by the mutated child
  #
  # Replace the least fit old seed in the population (s2) with the
  # new child (s1).
  i = s2.address # find the position of the old seed (s2)
  s1.address = i # copy the old position of the old seed into s1, the child
  pop[i] = s1 # replace s2 (old seed) in population (pop) with s1 (new child)
  # Build a history for the new seed, by matching it against all seeds
  # in the population.
  width_factor = mparam.width_factor
  height_factor = mparam.height_factor
  time_factor = mparam.time_factor
  num_trials = mparam.num_trials
  pop_size = len(pop)
  for j in range(pop_size):
    update_history(g, pop, i, j, width_factor, height_factor, \
      time_factor, num_trials)
    update_similarity(pop, i, j)
  # store the new seed
  seed_storage(s1)
  # Report on the new history of the new seed
  message = "Run: {}".format(n) + \
    "  Parent fitness (s0): {:.3f}".format(s0.fitness()) + \
    "  Child fitness (s1): {:.3f}".format(s1.fitness()) + \
    "  Replaced seed fitness (s2): {:.3f}\n".format(s2.fitness())
  # It is possible that s1 is worse than s2, if there was a bad mutation in s1.
  # Let's not worry about that, since s1 will soon be replaced if it is less
  # fit than the least fit seed (that is, s2).
  return [pop, message]
#
# sexual(candidate_seed, pop, n, max_seed_area, next_unique_ID_number)
# -- returns [pop, message]
#
def sexual(candidate_seed, pop, n, max_seed_area, next_unique_ID_number):
  """
  Create a new seed either asexually or sexually. First a single parent
  is chosen from the population. If a second parent can be found that
  is sufficiently similar to the first parent, then the child will have
  two parents (sexual reproduction). If no similar second parent can be
  found, then the child will have one parent (asexual reproduction).
  """
  # Let s0 be the most fit member of the tournament.
  s0 = candidate_seed
  # Find similar seeds in the population (members of the same species).
  min_similarity = mparam.min_similarity
  max_similarity = mparam.max_similarity
  similar_seeds = find_similar_seeds(s0, pop, min_similarity, max_similarity)
  num_similar_seeds = len(similar_seeds)
  # If no similar seeds were found, then use variable asexual reproduction.
  if (num_similar_seeds == 0):
    return variable_asexual(candidate_seed, pop, n, max_seed_area, 
                            next_unique_ID_number)
  # Run a new tournament to select a second seed s1 as a mate for s0.
  tournament_size = mparam.tournament_size
  if (num_similar_seeds <= tournament_size):
    s1 = find_best_seed(similar_seeds)
  else:
    tournament_sample = random_sample(similar_seeds, tournament_size)
    s1 = find_best_seed(tournament_sample)
  # Mate the parents to make a new child.
  s2 = mate(s0, s1)
  # Mutate the child.
  prob_grow = mparam.prob_grow
  prob_flip = mparam.prob_flip
  prob_shrink = mparam.prob_shrink
  seed_density = mparam.seed_density
  mutation_rate = mparam.mutation_rate
  s3 = s2.mutate(prob_grow, prob_flip, prob_shrink, seed_density, mutation_rate)
  s3.num_living = s3.count_ones() # update count of living cells
  s3.unique_ID_num = next_unique_ID_number
  s3.birth_type = "sexual"
  s3.parent_A_ID_num = s0.unique_ID_num
  s3.parent_B_ID_num = s1.unique_ID_num
  # Make sure the area of the new seed is not greater than the maximum.
  # If it is too big, then default to uniform_asexual reproduction.
  if ((s3.xspan * s3.yspan) > max_seed_area):
    return uniform_asexual(candidate_seed, pop, n, next_unique_ID_number)
  # If there are empty cells in s3, then try again with uniform_asexual.
  if found_empty_cells(s3):
    return uniform_asexual(candidate_seed, pop, n, next_unique_ID_number)
  # Find the least fit old seed in the population. It's not a problem
  # if there are ties.
  s4 = find_worst_seed(pop)
  # Now we have:
  #
  # s0 = parent 0
  # s1 = parent 1
  # s2 = the new child, before mutation
  # s3 = the mutated new child
  # s4 = the least fit old seed, which will be replaced by the mutated child
  #
  # Replace the least fit old seed in the population (s4) with the
  # new child (s3).
  i = s4.address # find the position of the old seed (s4)
  s3.address = i # copy the old position of the old seed into s3, the child
  pop[i] = s3 # replace s4 (old seed) in population (pop) with s3 (new child)
  # Build a history for the new seed, by matching it against all seeds
  # in the population.
  width_factor = mparam.width_factor
  height_factor = mparam.height_factor
  time_factor = mparam.time_factor
  num_trials = mparam.num_trials
  pop_size = len(pop)
  for j in range(pop_size):
    update_history(g, pop, i, j, width_factor, height_factor, \
      time_factor, num_trials)
    update_similarity(pop, i, j)
  # store the new seed
  seed_storage(s3)
  # Report on the new history of the new seed
  message = "Run: {}".format(n) + \
    "  Parent 0 fitness (s0): {:.3f}".format(s0.fitness()) + \
    "  Parent 1 fitness (s1): {:.3f}".format(s1.fitness()) + \
    "  Child fitness (s3): {:.3f}".format(s3.fitness()) + \
    "  Replaced seed fitness (s4): {:.3f}\n".format(s4.fitness())
  # It is possible that s3 is worse than s4, if there was a bad mutation in s3.
  # Let's not worry about that, since s3 will soon be replaced if it is less
  # fit than the least fit seed (that is, s4).
  return [pop, message]
#
# fusion(candidate_seed, pop, n, max_seed_area, next_unique_ID_number) 
# -- returns [pop, message]
#
def fusion(candidate_seed, pop, n, max_seed_area, next_unique_ID_number):
  """
  Fuse two seeds together. Randomly rotate the seeds before
  joining them. Let's put one seed on the left and the other 
  seed on the right. Insert one empty column between the two 
  seeds, as a kind of buffer, so that the two seeds do not 
  immediately interact. This empty column also helps fission
  later on, to split joined seeds at the same point where they
  were initially joined.
  """
  # The most fit member of the tournament.
  s0 = candidate_seed
  # Run another tournament to select a second seed. The second
  # seed might be identical to the first seed. That's OK.
  tournament_size = mparam.tournament_size
  tournament_sample = random_sample(pop, tournament_size)
  s1 = find_best_seed(tournament_sample)
  # If the flag fusion_test_flag is set to 1, then randomize s1
  # by shuffling its cells. This operation is expected to reduce
  # the fitness of the new fusion seed. Usually fusion_test_flag
  # should be set to 0. Note that s1.shuffle() makes a copy, so the
  # original of s1 is not affected by the shuffling.
  if (mparam.fusion_test_flag == 1):
    s1 = s1.shuffle()
  # Randomly rotate the seeds. These rotations (s2 and s3) are copies. 
  # The originals (s0 and s1) are not affected by the rotations.
  s2 = s0.random_rotate()
  s3 = s1.random_rotate()
  # Get dimensions for the new fusion seed.
  pop_size = mparam.pop_size
  xspan = s2.xspan + s3.xspan + 1 # left width + right width + empty gap
  yspan = max(s2.yspan, s3.yspan) # the larger of the two heights
  # Make sure the area of the new seed is not greater than the maximum.
  # If it is too big, then default to sexual reproduction.
  if ((xspan * yspan) > max_seed_area):
    return sexual(candidate_seed, pop, n, max_seed_area, next_unique_ID_number)
  # Copy s2 into the left side of s4.
  s4 = mclass.Seed(xspan, yspan, pop_size) # cells initialized to zero
  for x in range(s2.xspan):
    for y in range(s2.yspan):
      s4.cells[x][y] = s2.cells[x][y]
  # Copy s3 into the right side of s4.
  for x in range(s3.xspan):
    for y in range(s3.yspan):
      s4.cells[x + s2.xspan + 1][y] = s3.cells[x][y]
  # Insert a border (purple state) between s2 and s3.
  x = s2.xspan # border (location of vertical line in Golly)
  for y in range(s4.yspan): # s4.yspan equals max(s2.yspan, s3.yspan)
    s4.cells[x][y] = 5 # purple border state
  # If s2 and s3 are not the same height, then the shorter of the two
  # may have a vertical purple border that does not reach all the way
  # to the bottom of s4. Therefore we may need to extend the vertical
  # borders so that they all reach to the bottom. All of the vertical
  # purple borders should already reach the top of s4, so we check 
  # the top for purple and then extend the purple all the way down.
  for x in range(s4.xspan): # range over the width of the seed
    # if there is purple in the top row of the seed ...
    if (s4.cells[x][0] == 5): 
      # ... then make it purple all the way down
      for y in range(s4.yspan):
        s4.cells[x][y] = 5
  # Mutate s4
  prob_grow = mparam.prob_grow
  prob_flip = mparam.prob_flip
  prob_shrink = mparam.prob_shrink
  seed_density = mparam.seed_density
  mutation_rate = mparam.mutation_rate
  s4 = s4.mutate(prob_grow, prob_flip, prob_shrink, seed_density, mutation_rate)
  # Update count of living cells
  s4.num_living = s4.count_ones()
  s4.unique_ID_num = next_unique_ID_number
  s4.birth_type = "fusion"
  s4.parent_A_ID_num = s0.unique_ID_num
  s4.parent_B_ID_num = s1.unique_ID_num
  # If there are empty cells in s4, then try again with uniform_asexual.
  if found_empty_cells(s4):
    return uniform_asexual(candidate_seed, pop, n, next_unique_ID_number)
  # Find the least fit old seed in the population. It's not a problem
  # if there are ties.
  s5 = find_worst_seed(pop)
  # Now we have:
  #
  # s0 = seed 0
  # s1 = seed 1
  # s2 = rotated seed 0
  # s3 = rotated seed 1
  # s4 = the fusion of s2 and s3
  # s5 = the least fit old seed, which will be replaced by s4
  #
  # Replace the least fit old seed in the population (s5) with the
  # new fusion seed (s4).
  i = s5.address # find the position of the old seed (s5)
  s4.address = i # copy the old position of the old seed into s4, the new fusion seed
  pop[i] = s4 # replace s5 (old seed) in population (pop) with s4 (new fusion seed)
  # Build a history for the new seed, by matching it against all seeds
  # in the population.
  width_factor = mparam.width_factor
  height_factor = mparam.height_factor
  time_factor = mparam.time_factor
  num_trials = mparam.num_trials
  for j in range(pop_size):
    update_history(g, pop, i, j, width_factor, height_factor, \
      time_factor, num_trials)
    update_similarity(pop, i, j)
  # If the flag immediate_symbiosis_flag is set to "1", then
  # we must test to see whether s4 is more fit than both s1 and s2.
  if (mparam.immediate_symbiosis_flag == 1):
    if ((s0.fitness() >= s4.fitness()) or (s1.fitness() >= s4.fitness)):
      # If either of the parts (s0 or s1) has a fitness greater than
      # or equal to the fitness of s4, then default to sexual reproduction.
      # Symbiosis means that the whole is more fit than the parts.
      # When the flag immediate_symbiosis_flag is set to "1", we
      # insist that symbiosis should happen immediately, rather than
      # hoping that it will happen in some future generation.
      return sexual(candidate_seed, pop, n, max_seed_area, next_unique_ID_number)
  # store the new seed
  seed_storage(s4)
  # Report on the new history of the new seed.
  message = "Run: {}".format(n) + \
    "  Seed 0 fitness (s0): {:.3f}".format(s0.fitness()) + \
    "  Seed 1 fitness (s1): {:.3f}".format(s1.fitness()) + \
    "  Fusion fitness (s4): {:.3f}".format(s4.fitness()) + \
    "  Replaced seed fitness (s5): {:.3f}\n".format(s5.fitness())
  # Store the new seed (s4) and its parts (s2, s3) for future analysis.
  fusion_storage(s2, s3, s4, n)
  # Return with the updated population and a message.
  return [pop, message]
#
# fission(candidate_seed, pop, n, max_seed_area, next_unique_ID_number) 
# -- returns [pop, message]
#
def fission(candidate_seed, pop, n, max_seed_area, next_unique_ID_number):
  """
  In fusion, we use the convention of putting one seed on 
  the left and the other seed on the right, before we fuse
  the two seeds. In fission, we assume that fission will 
  split the left part from the right part. Find the most 
  sparse column in the candidate seed and split the seed along 
  this column. If both parts are at least the minimum allowed 
  seed size, randomly choose one of them. If only one part
  is at least the minimum allowed seed size, choose that
  one part. If neither part is at least the minimum allowed 
  seed size, then default to sexual reproduction.
  """
  # The most fit member of the tournament.
  s0 = candidate_seed
  # Minimum xspan. Only xspan is relevant, since we are splitting
  # left and right parts.
  min_s_xspan = mparam.min_s_xspan
  # See whether the seed is big enough to split. If it is too
  # small, then default to sexual reproduction.
  if (s0.xspan <= min_s_xspan):
    return sexual(candidate_seed, pop, n, max_seed_area, 
                  next_unique_ID_number)
  # In the seed matrix, x = row index, y = column index.
  # In Golly, g.setcell(g_x, g_y, s_state) refers to the cell
  # in horizontal position g_x and vertical position g_y, where
  # g_x increases from left to right and g_y increases from top
  # to bottom. Unfortunately, x in the seed matrix ranges
  # vertically over matrix rows and y in the seed matrix ranges
  # horizontally over matrix columns, whereas x in Golly ranges
  # horizontally and y in Golly ranges vertically.
  #
  # Speaking in Golly terms, we want to split the seed along
  # any purple border (cells in state 5) such that the border
  # spans the entire seed in a straight line. Due to the design
  # of fusion(), the border will be a vertical purple stripe in
  # Golly.
  #
  # There may be several vertical purple strips (that is, borders,
  # buffer zones, lines of cells in state 5) in the seed.
  # We will take the first one that we find.
  border_line = -1 # no border found yet
  border_colour = 5 # purple, state 5
  for x in range(s0.xspan):
    for y in range(s0.yspan):
      if (s0.cells[x][y] != border_colour):
        break # not a border -- try the next x
    # if we make it here, then we have found a border
    border_line = x
    break # stop looking
  # If no border was found, then use sexual reproduction
  if (border_line == -1):
    return sexual(candidate_seed, pop, n, max_seed_area, 
                  next_unique_ID_number)
  # Left and right parts.
  left_cells = s0.cells[0:border_line, :]
  right_cells = s0.cells[(border_line + 1):, :]
  # Initialize a seed for the left or right part.
  s1 = copy.deepcopy(s0)
  # If both parts are big enough, randomly choose one of them.
  if ((left_cells.shape[0] >= min_s_xspan) \
    and (right_cells.shape[0] >= min_s_xspan)):
    if (rand.uniform(0, 1) < 0.5):
      s1.cells = left_cells
    else:
      s1.cells = right_cells
  # If only the left part is big enough, use the left part.
  elif (left_cells.shape[0] >= min_s_xspan):
    s1.cells = left_cells
  # If only the right part is big enough, use the right part.
  elif (right_cells.shape[0] >= min_s_xspan):
    s1.cells = right_cells
  # If neither part is big enough, use sexual reproduction
  else: 
    return sexual(candidate_seed, pop, n, max_seed_area, 
                  next_unique_ID_number)
  # Set the correct dimensions for the new seed
  s1.xspan = s1.cells.shape[0]
  s1.yspan = s1.cells.shape[1]
  # Mutate s1
  prob_grow = mparam.prob_grow
  prob_flip = mparam.prob_flip
  prob_shrink = mparam.prob_shrink
  seed_density = mparam.seed_density
  mutation_rate = mparam.mutation_rate
  s1 = s1.mutate(prob_grow, prob_flip, prob_shrink, seed_density, mutation_rate)
  # Update count of living cells
  s1.num_living = s1.count_ones()
  s1.unique_ID_num = next_unique_ID_number
  s1.birth_type = "fission"
  s1.parent_A_ID_num = s0.unique_ID_num # the one and only parent of s1
  s1.parent_B_ID_num = -1 # there is no second parent
  # If there are empty cells in s1, then try again with uniform_asexual.
  if found_empty_cells(s1):
    return uniform_asexual(candidate_seed, pop, n, next_unique_ID_number)
  # Find the least fit old seed in the population. It's not a problem
  # if there are ties.
  s2 = find_worst_seed(pop)
  # Now we have:
  #
  # s0 = seed 0
  # s1 = left or right side of seed 0
  # s2 = the least fit old seed, which will be replaced by s1
  #
  # Replace the least fit old seed in the population (s2) with the
  # chosen part (s1).
  i = s2.address # find the position of the old seed (s2)
  s1.address = i # copy the old position of the old seed into s1
  pop[i] = s1 # replace s2 (old seed) in population (pop) with s1
  # Build a history for the new seed, by matching it against all seeds
  # in the population.
  width_factor = mparam.width_factor
  height_factor = mparam.height_factor
  time_factor = mparam.time_factor
  num_trials = mparam.num_trials
  pop_size = len(pop)
  for j in range(pop_size):
    update_history(g, pop, i, j, width_factor, height_factor, \
      time_factor, num_trials)
    update_similarity(pop, i, j)
  # store the new seed
  seed_storage(s1)
  # Report on the new history of the new seed
  message = "Run: {}".format(n) + \
    "  Whole fitness (s0): {:.3f}".format(s0.fitness()) + \
    "  Fragment fitness (s1): {:.3f}".format(s1.fitness()) + \
    "  Replaced seed fitness (s2): {:.3f}\n".format(s2.fitness())
  # Return with the updated population and a message.
  return [pop, message]
#
# symbiotic(candidate_seed, pop, n, max_seed_area, next_unique_ID_number) 
# -- returns [pop, message]
#
def symbiotic(candidate_seed, pop, n, max_seed_area, next_unique_ID_number):
  """
  Create a new seed by joining two existing seeds (fusion) or
  by splitting one seed into two seeds (fission). If fission
  is chosen, only one of the two resulting seeds is used.
  If neither fission nor fusion is chosen, we default to 
  sexual reproduction.
  """
  # Decide whether to use fission, fusion, or sexual reproduction.
  # To avoid bias, it makes sense to set these two probabilities to
  # the same value. Because fusion can result in large seeds, which
  # will slow down the simulation, it makes sense to set the
  # probability of fusion relatively low.
  #
  prob_fission = mparam.prob_fission
  prob_fusion = mparam.prob_fusion
  #
  uniform_random = rand.uniform(0, 1)
  #
  if (uniform_random < prob_fission):
    # this will be invoked with a probability of prob_fission
    return fission(candidate_seed, pop, n, max_seed_area, 
                   next_unique_ID_number)
  elif (uniform_random < (prob_fission + prob_fusion)):
    # this will be invoked with a probability of prob_fusion
    return fusion(candidate_seed, pop, n, max_seed_area, 
                  next_unique_ID_number)
  else:
    # if neither fission nor fusion, then sexual reproduction
    return sexual(candidate_seed, pop, n, max_seed_area, 
                  next_unique_ID_number)
#
# hash_pickles(pickle_list) -- returns pickle_hash
#
def hash_pickles(pickle_list):
  """
  Assume we have a list of pickle files of the following general form:
     ------------------------------------------------
     log-2019-02-22-12h-45m-00s-pickle-0.bin,
     log-2019-02-22-12h-45m-00s-pickle-1.bin, 
     ...
     log-2019-02-22-12h-45m-00s-pickle-100.bin,
     log-2019-02-22-12h-45m-12s-pickle-0.bin,
     log-2019-02-22-12h-45m-12s-pickle-1.bin, 
     ...
     log-2019-02-22-12h-45m-12s-pickle-100.bin
     ------------------------------------------------
  We split each pickle name into a base part ("log-2019-02-22-12h-45m-00s")
  and a numerical part ("0", "1", ..., "100") and we return a hash table
  that maps each unique base part to the maximum numerical part for that
  given base part (e.g., in examples above, the maximum is 100).
  """
  # initialize the hash of pickles
  pickle_hash = {}
  # process the items in the pickle list
  for pickle in pickle_list:
    # extract the base part of the pickle
    pickle_base_search = re.search(r'(log-.+\d\ds)-pickle-', pickle)
    assert pickle_base_search, "No pickles were found in the directory."
    pickle_base = pickle_base_search.group(1)
    # extract the numerical part of the pickle
    pickle_num_search = re.search(r'-pickle-(\d+)\.bin', pickle)
    assert pickle_num_search, "No pickles were found in the directory."
    pickle_num = int(pickle_num_search.group(1))
    # use the base part of the pickle as the hash key
    # and set the value to the largest numerical part
    if (pickle_base in pickle_hash):
      current_largest = pickle_hash[pickle_base]
      if (pickle_num > current_largest):
        pickle_hash[pickle_base] = pickle_num
    else:
      pickle_hash[pickle_base] = pickle_num
    #
  return pickle_hash
#
# choose_pickles(g) -- returns [pickle_dir, analysis_dir, 
#                      sorted_pickle_names, smallest_pickle_size]
#
def choose_pickles(g):
  """
  Present a GUI to ask the users which folder of pickles they
  would like to analyze.
  """
  #
  # Open a dialog window and ask the user to select a folder.
  #
  g.note("Analyze Pickles\n\n" + \
         "Select a FOLDER of pickled seeds (not a FILE of seeds).\n" + \
         "The pickles will be analyzed and the results will be\n" + \
         "stored in the same directory as the pickles.\n")
  #
  pickle_dir = g.opendialog("Choose a folder of pickled seeds", \
               "dir", g.getdir("app"))
  #
  analysis_dir = pickle_dir
  #
  g.note("Verify Selection\n\n" + \
         "The folder of pickled seeds:\n\n" + \
         "   " + pickle_dir + "\n\n" + \
         "The folder for the analysis results:\n\n" + \
         "   " + analysis_dir + "\n\n" + \
         "Exit now if this is incorrect.")
  #
  # Make a list of the pickles in pickle_dir.
  #
  pickle_list = []
  for file in os.listdir(pickle_dir):
    if (file.endswith(".bin") and file.startswith("log-")):
      pickle_list.append(file)
  #
  # Verify that there are some ".bin" files in the list.
  #
  if (len(pickle_list) == 0):
    g.note("No pickles were found in the directory:\n\n" + \
           "   " + pickle_dir + "\n\n" + \
           "Exiting now.")
    sys.exit(0)
  #
  # Make a hash table that maps pickle names to the last
  # generation number of the given group of pickles.
  #
  pickle_hash = hash_pickles(pickle_list)
  #
  # Calculate the size of the smallest group of pickles.
  #
  smallest_pickle_size = min(pickle_hash.values())
  #
  # Report the base parts of the pickles and their maximum
  # values
  #
  sorted_pickle_names = sorted(pickle_hash.keys())
  pickle_note = ""
  for pickle_base in sorted_pickle_names:
    pickle_note = pickle_note + \
      pickle_base + " ranges from 0 to " + \
      str(pickle_hash[pickle_base]) + "\n"
  g.note("These pickles were found:\n\n" +
    pickle_note + "\n" + \
    "The analysis will range from 0 to " + \
    str(smallest_pickle_size) + "\n\n" + \
    "Exit now if this is not what you expected.")
  #
  return [pickle_dir, analysis_dir, \
    sorted_pickle_names, smallest_pickle_size]
#
# validate_designed_seed(g, seed_path, max_area) 
# -- returns 0 for bad, 1 for good
#
def validate_designed_seed(g, seed_path, max_area):
  """
  This function checks whether we can convert a human-made pattern file
  into a seed.
  """
  #
  # We only want *.rle or *.lif
  #
  file_base, file_extension = os.path.splitext(seed_path)
  if (file_extension != ".rle") and (file_extension != ".lif"):
    return 0
  #
  # Golly has two kinds of cell lists, one that contains an even number 
  # of members and one that contains an odd number of members. The 
  # former is intended for two states (0 and 1) and the latter is intended 
  # for more than two states. Here we are only interested in patterns designed 
  # for the Game of Life, which only has two states.
  #
  cell_list = g.load(seed_path)
  #
  # Make sure cell_list is not too small
  #
  too_small = 5
  #
  if (len(cell_list) <= too_small):
    return 0
  #
  # We can only handle cell_list if it has an even number of members.
  #
  if (len(cell_list) % 2 != 0):
    return 0
  #
  # See how big this pattern is.
  #
  min_x = cell_list[0]
  max_x = cell_list[0]
  min_y = cell_list[1]
  max_y = cell_list[1]
  #
  for i in range(0, len(cell_list), 2):
    pair = (cell_list[i], cell_list[i + 1])
    (x, y) = pair
    if (x < min_x):
      min_x = x
    if (x > max_x):
      max_x = x
    if (y < min_y):
      min_y = y
    if (y > max_y):
      max_y = y
  #
  # Make sure it's not too big.
  #
  if (max_x * max_y > max_area):
    return 0
  #
  # Make sure it's not too small.
  #
  if (max_x == 0) or (max_y == 0):
    return 0
  #
  # Passed all tests.
  #
  return 1
#
# load_designed_seed(g, seed_path) -- returns seed
#
def load_designed_seed(g, seed_path):
  """
  Given the path to a human-designed Game of Life pattern, load the
  file and convert it to a seed.
  """
  #
  # Golly has two kinds of cell lists, one that contains an even number 
  # of members and one that contains an odd number of members. The 
  # former is intended for two states (0 and 1) and the latter is intended 
  # for more than two states. Here we are only interested in patterns designed 
  # for the Game of Life, which only has two states.
  #
  cell_list = g.load(seed_path)
  #
  # Make sure that cell_list is the type of list that contains an even
  # number of members. Make sure cell_list is not unreasonably small.
  #
  assert len(cell_list) % 2 == 0
  assert len(cell_list) > 10
  #
  # Convert cell_list to a list of (x, y) pairs.
  #
  pair_list = []
  min_x = cell_list[0]
  max_x = cell_list[0]
  min_y = cell_list[1]
  max_y = cell_list[1]
  #
  for i in range(0, len(cell_list), 2):
    pair = (cell_list[i], cell_list[i + 1])
    pair_list.append(pair)
    (x, y) = pair
    if (x < min_x):
      min_x = x
    if (x > max_x):
      max_x = x
    if (y < min_y):
      min_y = y
    if (y > max_y):
      max_y = y
  #
  # Convert pair_list to a seed. Start with a seed full of
  # zeros and set the cells given in pair_list to ones.
  #
  assert min_x == 0
  assert min_y == 0
  assert max_x > 0
  assert max_y > 0
  #
  s_xspan = max_x + 1
  s_yspan = max_y + 1
  #
  seed = mclass.Seed(s_xspan, s_yspan, mparam.pop_size)
  #
  for pair in pair_list:
    (x, y) = pair
    seed.cells[x][y] = 1
  #
  # Count the initial number of living cells in the seed
  # and store the count.
  #
  seed.num_living = seed.count_ones()
  #
  assert seed.num_living > 0
  #
  return seed
#
# compare_random(g, random_seed, description, stats_hash)
# -- returns nothing: all the information is stored in the
#    hash table stats_hash
#
def compare_random(g, evolved_seed, description, stats_hash):
  """
  Calculate the fitness of evolved_seed by comparing it with randomly
  shuffled versions of itself (random_seed) in the Immigration Game.
  """
  #
  # Get the parameters for competitions in the Immigration Game.
  #
  width_factor = mparam.width_factor # (e.g., 6.0)
  height_factor = mparam.height_factor # (e.g., 3.0)
  time_factor = mparam.time_factor # (e.g., 6.0)
  num_trials = mparam.num_trials # (e.g., 2)
  # so that the noise level here is comparable to the noise level
  # in compare_generations.py, generate the same number of random
  # seeds as there are seeds in the elite pickles (num_runs * elite_size)
  num_runs = mparam.num_generations + 1 # number of pickles (e.g. 101)
  elite_size =  mparam.elite_size # number of seeds per pickle (e.g. 50)
  #
  # Run the competitions.
  #
  total_fitness = 0
  total_sample_size = 0
  for sample in range(num_runs * elite_size): # (e.g. 101 * 50 = 5,050)
    # make a copy of evolved_seed and randomly shuffle the cells
    # in the new seed, so that the new randomized seed has the
    # same dimensions and the same density as evolved_seed
    random_seed = evolved_seed.shuffle()
    # compare the input seed to the random seed -- score_pair()
    # will change the colour of the random seed and run the
    # Immigration Game
    [random_score, evolved_score] = score_pair(g, random_seed, \
      evolved_seed, width_factor, height_factor, time_factor, num_trials)
    total_fitness = total_fitness + evolved_score
    total_sample_size = total_sample_size + 1
  # calculate average fitness for the run
  avg_fitness = total_fitness / total_sample_size
  # add info to stats_hash
  stats_hash[description + " absolute fitness"] = avg_fitness
  #
  # return nothing -- all the information is stored in the
  # hash table stats_hash
  return
#
# change_live_state(seed, new_state) 
# -- return a modified copy of the input seed
#
def change_live_state(seed, new_state):
  """
  Given an input seed with live state 1, copy the seed and
  replace state 1 with new_state.
  """
  # let's not waste time by changing state 1 to state 1
  assert new_state != 1
  # copy the seed so that the original is not changed
  new_seed = copy.deepcopy(seed)
  # change state 1 in the given seed to new_state
  for x in range(new_seed.xspan):
    for y in range(new_seed.yspan):
      # update seed.cells -- state 0 remains state 0
      # and state 1 becomes new_state
      if (new_seed.cells[x][y] == 1): 
        new_seed.cells[x][y] = new_state
  #
  return new_seed
#
# change_colour(seed, old_colour, new_colour)
# -- return a modified copy of the input seed
#
def change_colour(seed, old_colour, new_colour):
  """
  Given an input seed, copy the seed and change every cell
  that is in state old_colour to the state new_colour.
  """
  # copy the seed so that the original is not changed
  new_seed = copy.deepcopy(seed)
  # change old_colour to new_colour
  for x in range(new_seed.xspan):
    for y in range(new_seed.yspan):
      if (new_seed.cells[x][y] == old_colour): 
        new_seed.cells[x][y] = new_colour
  #
  return new_seed
#
# join_seeds(part1, part2) -- returns whole
#
def join_seeds(part1, part2):
  """
  Given two seeds, part1 and part2, join them together, with
  part1 on the left and part2 on the right. Insert a gap of
  one column between them.
  """
  #
  # Calculate the dimensions of the new seed.
  #
  xspan = part1.xspan + part2.xspan + 1 # left width + right width + empty gap
  yspan = max(part1.yspan, part2.yspan) # the larger of the two heights
  whole = mclass.Seed(xspan, yspan, mparam.pop_size) # cells set to zero
  #
  # Copy part1 into the left side of whole.
  #
  for x in range(part1.xspan):
    for y in range(part1.yspan):
      whole.cells[x][y] = part1.cells[x][y] 
  #
  # Copy part2 into the right side of whole.
  #
  for x in range(part2.xspan):
    for y in range(part2.yspan):
      whole.cells[x + part1.xspan + 1][y] = part2.cells[x][y]
  #
  # Insert the purple border (buffer zone)
  #
  x = part1.xspan
  for y in range(yspan):
    whole.cells[x][y] = 5 # state 5 = purple
  #
  # Return the new seed.
  #
  return whole
#
# snap_photo(g, file_path, rule_name, seed,
#            steps, description, pause)
# -- returns nothing: photo is written to specified file_path
#
def snap_photo(g, file_path, rule_name, seed, \
               steps, description, pause):
  """
  Run Golly with the given seed and take a photo of the result.
  The photo will be stored in file_path (*.png).
  """
  #
  # Prevent Windows Screen Lock, which results in blank photos.
  #
  pyautogui.press('volumedown')
  pyautogui.press('volumeup')
  #
  # Initialize the game.
  #
  g.setalgo("QuickLife") # use "HashLife" or "QuickLife"
  g.autoupdate(False) # do not update the view unless requested
  g.new(rule_name) # initialize cells to state 0
  g.setrule(rule_name) # make an infinite plane
  #
  # Copy the seed into Golly.
  #
  for x in range(seed.xspan):
    for y in range(seed.yspan):
      state = seed.cells[x][y]
      g.setcell(x, y, state)
  #
  # Run the game for num_steps steps.
  #
  if (steps == 0):
    g.show(description)
    g.fit()
    g.update() # show the start state
  else:
    g.show(description)
    g.update() # show the start state
    g.run(steps) # run the Game of Life for steps
    g.fit()
    g.update() # show the end state
  #
  # Take the photo and save it.
  #
  time.sleep(pause)
  photo_object = pyautogui.screenshot()
  photo_object.save(file_path)
  #
  # Return nothing: the photo is written to "file_path".
  #
  return
#
# measure_growth_life(g, seed, num_steps) -- return growth
#
def measure_growth_life(g, seed, num_steps):
  """
  Given a Game of Life seed pattern (two states only, 0 and 1),
  run the pattern for num_steps and calculate its growth.
  """
  g.setalgo("QuickLife")
  g.autoupdate(False)
  g.new("Life")
  g.setrule("Life")
  # state 0 = white, state 1 = black
  g.setcolors([0,255,255,255,1,0,0,0])
  for x in range(seed.xspan):
    for y in range(seed.yspan):
      g.setcell(x, y, seed.cells[x][y])
  g.update()
  start_size = int(g.getpop())
  g.run(num_steps)
  g.update()
  end_size = int(g.getpop())
  growth = end_size - start_size
  return growth
#
# measure_consistent_growth(g, seed, test_num_steps)
# -- return score
#
def measure_consistent_growth(g, seed, test_num_steps):
  """
  Given a Game of Life seed pattern (two states only, 0 and 1),
  run the pattern for num_steps and calculate a score that
  rewards early consistent growth.
  """
  g.setalgo("QuickLife")
  g.autoupdate(False)
  g.new("Life")
  g.setrule("Life")
  # state 0 = white, state 1 = black
  g.setcolors([0,255,255,255,1,0,0,0])
  for x in range(seed.xspan):
    for y in range(seed.yspan):
      g.setcell(x, y, seed.cells[x][y])
  g.update()
  size_before = int(g.getpop())
  delta_positive = 0
  for step in range(test_num_steps):
    g.run(1)
    g.update()
    size_after = int(g.getpop())
    if (size_after > size_before):
      delta_positive += 1
    size_before = size_after
  return delta_positive / test_num_steps
#
# hash_seed(seed) -- returns a hash key for seed
#
def hash_seed(seed):
  """
  Given a seed, return a string that serves as a unique ID for the seed. 
  The purpose is to quickly identify duplicate seeds.
  """
  #
  # copy the matrix from the given seed
  # -- it will be a numpy matrix of integers
  #
  matrix = copy.deepcopy(seed.cells) 
  #
  # remove outer rows and columns that are entirely zeros, because
  # two seed patterns that only differ by empty outer rows and columns 
  # should really count as being identical
  #
  # however, we will count seeds that are rotations of each other as
  # being different, because Model-S is not likely to generate such seeds
  #
  # step 1: delete top rows that are all zero
  while (np.all(matrix[0] == 0)):
    matrix = np.delete(matrix, (0), axis = 0)
  #
  # step 2: delete bottom rows that are all zero
  while (np.all(matrix[-1] == 0)):
    matrix = np.delete(matrix, (-1), axis = 0)
  #
  # step 3: delete left columns that are all zero
  while (np.all(matrix[:,0] == 0)):
    matrix = np.delete(matrix, (0), axis = 1)
  #
  # step 4: delete right columns that are all zero
  while (np.all(matrix[:,-1] == 0)):
    matrix = np.delete(matrix, (-1), axis = 1)
  #
  # convert the matrix to a string, to serve as a hash key
  #
  seed_hash = ""
  num_rows = matrix.shape[0]
  num_cols = matrix.shape[1]
  # rotate matrix for consistency with Golly
  for y in range(num_cols): 
    for x in range(num_rows):
      seed_hash += str(matrix[x][y])
    seed_hash += "\n" # end of row marker
  #
  return seed_hash
#
# read_fusion_pickles(fusion_file_list) -- returns a list of fusion seeds
#
def read_fusion_pickles(fusion_file_list):
  """
  Given a list of fusion files, read the files and return a list of seeds.
  Note that a single fusion file may contain several seeds, so the final
  list of seeds will usually be longer than the list of fusion files.
  """
  # initialize the list of seeds
  fusion_list = []
  # loop through the file names and extract the seeds
  for fusion_file in fusion_file_list:
    # open the fusion pickle file -- "ab+" opens a file for 
    # both appending and reading in binary mode
    fusion_handle = open(fusion_file, "ab+")
    fusion_handle.seek(0) # start at the beginning of the file
    # step through all of the pickles in the file, adding them
    # to the list
    while True:
      try:
        item = pickle.load(fusion_handle)
        fusion_list.append(item)
      except (EOFError, pickle.UnpicklingError):
        break
    #
    fusion_handle.close()
  # note that fusion_list has the form [s2, s3, s4, n, s2, s3, s4, n, ...],
  # where s2 and s3 are two parts in the seed s4 and n is the birth
  # number of s4 -- however, we only need s4, so let's remove the other
  # items from the list
  seed_list = []
  # read four items at a time
  for (s2, s3, s4, n) in zip(*[iter(fusion_list)] * 4):
    seed_list.append(s4)
  # return the list of s4 items
  return seed_list
#
# growth_tensor(g, seed_list, step_size, max_seeds, num_steps, 
#               num_colours, num_parts)
# -- returns a tensor populated with counts of the growth of colours
#    generated by running the Management Game
#
def growth_tensor(g, seed_list, step_size, max_seeds, num_steps, 
                  num_colours, num_parts):
  """
  Given a list of seeds, fill a tensor with counts of the growth of colours
  generated by running the Management Game.
  """
  #
  # Make sure we have room in the tensor for all these seeds.
  #
  assert len(seed_list) < max_seeds
  #
  # Suppose the steps of the Management Game run from 0 to 1000, yielding
  # a total of 1001 steps. The number of the final step (1000) should be
  # divisible by step_size (e.g., 20).
  #
  assert ((num_steps - 1) % step_size) == 0
  #
  # Initialize the tensor.
  #
  tensor = np.zeros([max_seeds, num_steps, num_colours, num_parts])
  # 
  # Count the number of seeds we actually use -- note that the following
  # loop skips over some of the seeds in seed_list. Note that, if there
  # are N seeds, then seed_num ranges from 0 to N-1.
  #
  seed_num = 0
  #
  for seed in seed_list:
    #
    # make a matrix that is a map of the regions in seed; the map will
    # have the same size matrix as seed and each cell in the map will 
    # contain a number that uniquely defines the region it belongs to,
    # as determined by the purple borders in seed
    #
    seed_map = region_map(seed)
    num_rows = seed_map.shape[0]
    num_cols = seed_map.shape[1]
    #
    # find out how many different regions there are in the map
    # and then generate the p 1-vs-(p-1) colourings
    #
    num_regions = np.amax(seed_map)
    #
    # the actual number of parts in the current seed is given by
    # num_regions, but the desired number of parts is given by
    # num_parts (a parameter in growth_tensor() function)
    #
    # we skip over a seed if the actual number of parts
    # (num_regions) does not equal the desired number of parts
    # (num_parts)
    #
    if (num_regions != num_parts):
      continue
    #
    for target_region in range(1, num_regions + 1):
      # make a copy of seed -- we will create a new colouring
      # for the copy
      seed_colouring = copy.deepcopy(seed)
      # - if a cell in the region map seed_map has the value target_region
      #   and the corresponding cell in seed_colouring is 1 or 2 (red or blue),
      #   then the corresponding cell in seed_colouring will be set
      #   to red (state 1)
      # - if a cell in the region map seed_map does not have the value target_region
      #   and the corresponding cell in seed_colouring is 1 or 2 (red or blue),
      #   then the corresponding cell in seed_colouring will be set
      #   to blue (state 2)
      # - if a cell in the region map seed_map has the value -1,
      #   then the corresponding cell in seed_colouring should be
      #   purple (state 5 -- the colour of the borders); if it is not
      #   purple, then signal an error
      for x in range(num_rows): 
        for y in range(num_cols):
          # state 1 -- red -- target_region
          if ((seed_map[x][y] == target_region) and \
             ((seed_colouring.cells[x][y] == 1) or \
             (seed_colouring.cells[x][y] == 2))):
            seed_colouring.cells[x][y] = 1
          # state 2 -- blue -- not target_region
          elif ((seed_map[x][y] != target_region) and \
               ((seed_colouring.cells[x][y] == 1) or \
               (seed_colouring.cells[x][y] == 2))):
            seed_colouring.cells[x][y] = 2
          # state 5 -- purple -- the border between regions
          elif (seed_map[x][y] == -1):
            assert seed_colouring.cells[x][y] == 5
      # initialize Golly
      rule_name = "Management" # the Management Game
      g.setalgo("QuickLife") # use "HashLife" or "QuickLife"
      g.autoupdate(False) # do not update the view unless requested
      g.new(rule_name) # initialize cells to state 0
      g.setrule(rule_name) # make an infinite plane
      # initialize the counts for the five states:
      # [white, red, blue, orange, green]
      start_size = [0, 0, 0, 0, 0] 
      end_size = [0, 0, 0, 0, 0]
      # copy seed into Golly 
      for x in range(num_rows):
        for y in range(num_cols):
          state = seed_colouring.cells[x][y]
          # ignore purple colours (state 5)
          if (state < 5):
            g.setcell(x, y, state)
            # update start_size and end_size
            start_size[state] += 1
            end_size[state] += 1
      # record the initial growth (time step 0) in the tensor
      # -- the intitial growth is necessarily zero for all colours
      step_num = 0
      part_num = target_region - 1
      for colour_num in range(num_colours):
        tensor[seed_num, step_num, colour_num, part_num] = 0
      # iterate over the number of time steps -- we start the iteration
      # at step_size, because we already filled the tensor for step 0,
      # immediately above
      for step_num in range(step_size, num_steps, step_size):
        g.run(step_size)
        g.update()
        # update end_size
        boundary = g.getrect()
        if (len(boundary) == 0): # if no live cells ...
          end_size = [0, 0, 0, 0, 0]
        else:
          cell_list = g.getcells(boundary)
          # if cell_list ends in 0, then delete the 0 -- note that stateN
          # will never be zero, since dead cells (state 0) are not included
          # in cell_list
          if (cell_list[-1] == 0):
            cell_list.pop()
          # end_size = [white, red, blue, orange, green]
          end_size = [0, 0, 0, 0, 0] # initialize
          for (x, y, state) in zip(*[iter(cell_list)] * 3):
            end_size[state] += 1 # update count
        # update the tensor
        part_num = target_region - 1
        for colour_num in range(num_colours):
          tensor[seed_num, step_num, colour_num, part_num] = \
            end_size[colour_num] - start_size[colour_num]
        #       
      #
    # increment fusion number (seed number), so we're ready for the
    # next trip around the loop
    seed_num += 1
    #
  #
  # total number of seeds with specified number of parts (2, 3, 4)
  # -- now that we're exiting the loop, seed_num is the actual number
  # of seeds; it is no longer one less than the total
  num_seeds = seed_num
  #
  return [tensor, num_seeds]
#
# sort_deduplicate(mylist) -- returns mylist sorted with no duplicates
#
def sort_deduplicate(mylist):
  """
  Remove all duplicates from the given list and then sort the
  remaining items.
  """
  newlist = list(dict.fromkeys(mylist)) # remove duplicates
  newlist.sort() # sort in ascending order
  return newlist
#
# append_to_sublist(sublist, position, value) -- returns sublist
#
def append_to_sublist(sublist, position, value):
  """
  Suppose we have a list of lists, such as:
     sublist = [[]] * 5
     sublist == [[], [], [], [], []]
  Now suppose we want to append value 4 to position 2 in sublist:
     sublist[2] = [4]
     sublist == [[], [], [4], [], []}
  However, to append another value to a position that is not emtpy,
  we need to use a different method:
     sublist[2].append(6)
     sublist == [[], [], [4, 6], [], []}
  """
  if (len(sublist[position]) == 0):
    sublist[position] = [value]
  else:
    sublist[position].append(value)
  return sublist
#
# map_parent_to_child(seed_list, avoid_list) 
# -- returns a list that maps parent ID numbers to child ID numbers
#
def map_parent_to_child(seed_list, avoid_list):
  """
  Each seed has a unique numerical ID number. Each seed gives
  the ID numbers of its parents. Our task here is to create a
  reverse mapping from parent IDs to child IDs. (The parent of
  a child can be coded in the child when the child is born, but the
  child of a parent can only be determined some time after the
  parent is born. Coding the parent of a child in the child is
  therefore easier than coding the child of a parent in the
  parent.)
  """
  #
  # NOTE: each seed is one of the following types: "uniform_asexual",
  # "variable_asexual", "sexual", "fusion", "fission", or "random"
  # ("random" is for the initial first-generation random seeds)
  #
  # make a list of sublists for mapping parents to children
  # - the sublists are initially empty
  map_parent_to_children = [[]] * len(seed_list)
  # each seed might be both a parent and a child, so we need
  # to look at all possible pairs of seeds
  for parent_seed in seed_list:
    for child_seed in seed_list:
      # skip over certain types of children (e.g., "random",
      # "fusion", "fission") because they are not ordinarily
      # considered as children
      if (child_seed.birth_type in avoid_list):
        continue
      # ID numbers are assigned in order of birth, so a child
      # must have a larger ID number than its parents
      if (child_seed.unique_ID_num <= parent_seed.unique_ID_num):
        continue
      # if parent_seed is a parent of child_seed, then add
      # child_seed to the list of children of the parent
      # -- parent_A
      if (parent_seed.unique_ID_num == child_seed.parent_A_ID_num):
        # sanity check
        assert child_seed.unique_ID_num > child_seed.parent_A_ID_num
        # append_to_sublist(sublist, position, value)
        append_to_sublist(map_parent_to_children,
          parent_seed.unique_ID_num, child_seed.unique_ID_num)
      # -- parent B
      if (parent_seed.unique_ID_num == child_seed.parent_B_ID_num):
        # sanity check
        assert child_seed.unique_ID_num > child_seed.parent_B_ID_num
        # append_to_sublist(sublist, position, value)
        append_to_sublist(map_parent_to_children,
          parent_seed.unique_ID_num, child_seed.unique_ID_num)
        #
      #
    #
  #
  # remove duplicates from map_parent_to_children and sort
  #
  for seed in seed_list:
    messy_list = map_parent_to_children[seed.unique_ID_num]
    clean_list = sort_deduplicate(messy_list)
    map_parent_to_children[seed.unique_ID_num] = clean_list
  #
  # return the list of lists that maps parents to their children
  #
  return map_parent_to_children
#
# fusion_seed_IDs(map_parent_to_children)
# -- returns a list of all fusion seeds
#
def fusion_seed_IDs(seed_list):
  """
  Given a list of seeds, extract the IDs numbers of all fusion seeds.
  """
  fusion_seed_identifiers = []
  for seed in seed_list:
    if (seed.birth_type == "fusion"):
      fusion_seed_identifiers.append(seed.unique_ID_num)
  return fusion_seed_identifiers
#
# find_types(seed_list) 
# -- returns a mapping from seeds to their corresponding types
#
def find_types(seed_list):
  """
  Returns a list that maps seeds to their corresponding types.
  """
  seed_types = []
  for seed in seed_list:
    seed_types.append(seed.birth_type)
  return seed_types
#
# longest_paths(fusion_seed_ID, map_parent_to_children, 
#               map_seed_list_to_type, sample_exponent) 
# -- returns a list of the form:
#   [
#     [node X ID, node X type, node X prob, node X avg depth, node X num children],
#     [node Y ID, node Y type, node Y prob, node Y avg depth, node Y num children],
#     ...
#   ]
#
def longest_paths(fusion_seed_ID, map_parent_to_children,
                  map_seed_list_to_type, sample_exponent):
  """
  Given a fusion seed, randomly sample full paths (any path that joins
  the root fusion seed to a leaf in the tree of descendants). For each
  node in a full path, record (1) the ID of the node, (2) the type of the
  node, (3) the number of times that the node was found in a sample, and 
  (4) the depths that the node was found in the tree (root = depth 0, 
  one step from root = depth 1, ...). From (3) the number of times the0
  node was found, calculate the probability of the node. From (4) the 
  depths where the node was found, calculate the average depth.
  """
  #
  # the number of times a given node was found in a random sample
  # -- maps node ID number to count number
  dict_node_count = {}
  # the type of a given node ("uniform_asexual", "variable_asexual", etc.)
  # -- maps node ID number to string
  dict_node_type = {}
  # the depths that the node was found in the tree
  # -- maps node ID number to list of depths
  dict_node_depth = {}
  #
  sample_size = 10 ** sample_exponent
  #
  for i in range(sample_size):
    #
    # walk a random path
    #
    random_path = [fusion_seed_ID] # random path begins at root of tree
    next_descendants = map_parent_to_children[fusion_seed_ID] # children of root
    num_descendants = len(next_descendants)
    while (num_descendants > 0):
      # randomly select one of the children
      random_descendant = rand.choice(next_descendants)
      # add the chosen child to the path
      random_path.append(random_descendant)
      # look for the next step in the path
      next_descendants = map_parent_to_children[random_descendant]
      # count the descendants in the next depth level of the tree
      num_descendants = len(next_descendants)
    #
    # random_path is now a complete path from root to leaf
    # -- step through the depth levels of random_path
    #
    for depth in range(len(random_path)):
      # get the node at the given depth
      node = random_path[depth]
      # update dict_node_count
      # -- maps node ID number to count *number*
      if (node in dict_node_count):
        dict_node_count[node] += 1
      else:
        dict_node_count[node] = 1
      # update dict_node_type if this node has not been encountered yet
      # -- maps node ID number to *string*
      if node not in dict_node_type:
        dict_node_type[node] = map_seed_list_to_type[node]
      # update dict_node_depth
      # -- maps node ID number to *list* of depths
      if node not in dict_node_depth:
        # first observation of this node
        dict_node_depth[node] = [depth]
      else:
        # this node has been seen before
        dict_node_depth[node].append(depth)
    #
  #
  # calculate the probability for encountering each node
  #
  dict_node_prob = {}
  for node in dict_node_count:
    dict_node_prob[node] = dict_node_count[node] / sample_size
  #
  # calculate the average depth for each node
  #
  dict_node_avg_depth = {}
  for node in dict_node_depth:
    depth_list = dict_node_depth[node]
    sample_size = dict_node_count[node]
    assert len(depth_list) == sample_size
    avg_depth = sum(depth_list) / sample_size
    # round off avg_depth to two decimal places
    dict_node_avg_depth[node] = round(avg_depth, 2)
  #
  # calculate number of children for each node
  #
  dict_node_num_children = {}
  for node in dict_node_count:
    dict_node_num_children[node] = len(map_parent_to_children[node])
  #
  #
  # make a sorted list of lists of the following form:
  #
  # [
  #   [node X ID, node X type, node X prob, node X avg depth, node X num children],
  #   [node Y ID, node Y type, node Y prob, node Y avg depth, node Y num children],
  #   ...
  # ]
  #
  # sort the nodes in increasing order, based on ID number
  #
  sorted_keys = sorted(dict_node_count.keys())
  descendants_properties = []
  for key in sorted_keys:
    descendants_properties.append([key, dict_node_type[key], \
                           dict_node_prob[key], dict_node_avg_depth[key], \
                           dict_node_num_children[key]])
  #
  # return the list of lists
  #
  return descendants_properties
#
# shuffle_score(g, target_seed, num_trials) -- returns score
#
def shuffle_score(g, target_seed, num_trials):
  """
  Given a seed, measure the fitness of the seed by having the seed
  compete with shuffled versions of itself. The shuffled seed has
  exactly the same matrix size and the same number of zeros and ones
  as the given seed. The only difference is that the zeros and ones
  have been shuffled randomly, which destroys the structure of the
  seed.
  """
  #
  # first get some parameter settings from the parameter file
  #
  width_factor = mparam.width_factor
  height_factor = mparam.height_factor
  time_factor =  mparam.time_factor
  #
  # calculate average fitness of target_seed
  #
  total_fitness = 0
  total_sample_size = 0
  for trial in range(num_trials):
    random_seed = target_seed.shuffle()
    [random_score, target_score] = score_pair(g, random_seed, \
      target_seed, width_factor, height_factor, time_factor, num_trials)
    total_fitness = total_fitness + target_score
    total_sample_size = total_sample_size + 1
  avg_fitness = total_fitness / total_sample_size
  return avg_fitness
#
# write_fusion_tables(table_matrix, table_handle, table_title, 
#                     table_range, row_label, col_label)
# -- does not return anything; all information is written to table_handle
#
def write_fusion_tables(table_matrix, table_handle, table_title, \
  table_range, row_label, col_label):
  """
  This function writes out a table for displaying (1) managers vs workers in
  the management game, (2) insiders vs outsiders in the mutualism game, and
  (3) soloists vs ensembles in the interaction game. It is assumed that many
  tables will be written to a single file, so the file handle (table_handle) is
  assumed to be open and it will not be closed by this function.
  """
  max_parts = table_range - 1
  table_handle.write("\n" + table_title + "\n")
  for row in range(table_range):
    reverse_order_row = max_parts - row
    table_handle.write(row_label + "\t" + str(reverse_order_row))
    for col in range (table_range):
      table_handle.write("\t" + str(table_matrix[reverse_order_row, col]))
    table_handle.write("\n")
  table_handle.write("\t")
  for col in range (table_range):
    table_handle.write("\t" + str(col))
  table_handle.write("\n\t")
  for col in range (table_range):
    table_handle.write("\t" + col_label)
  table_handle.write("\n")
  return
#
# read_seed_storage(seed_file) -- returns seed_list
#
def read_seed_storage(seed_file):
  """
  Read a seed file and return a list of seeds.
  """
  # open the pickle file -- "ab+" opens a file for 
  # both appending and reading in binary mode
  seed_handle = open(seed_file, "ab+")
  # start at the beginning of the file
  seed_handle.seek(0)
  # initialize the list of seeds
  seed_list = []
  # step through all of the pickles in the file, adding them
  # to the list
  while True:
    try:
      seed = pickle.load(seed_handle)
      seed_list.append(seed)
    except (EOFError, pickle.UnpicklingError):
      break
  # close the file
  seed_handle.close()
  # return the seed list
  return seed_list
#
# check_for_borders(seed) -- returns True if a border is found, False otherwise
#
def check_for_borders(seed):
  """
  Check whether the given seed is a symbiote (it contains a purple border)
  or a singleton (it has no purple border). Return True if a border is found
  or return False if no border is found.
  """
  # Borders are purple cells, represented by state 5.
  border_colour = 5
  for x in range(seed.xspan):
    for y in range(seed.yspan):
      if (seed.cells[x][y] == border_colour):
        return True
  # If we make it here, we did not find a border.
  return False
#
#