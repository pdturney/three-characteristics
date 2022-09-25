#
# Photos Compete Seeds
#
# Peter Turney, September 16, 2022
#
# Choose a run and a generation and read the pickle
# file for that run and generation. Choose the number 
# of competitions to photograph. This program will 
# randomly select pairs of seeds from the given run 
# and generation. The pairs will then compete. For 
# each random pair of seeds, the program will take 
# two photographs, one of the two seeds at the start 
# of the competition and one of the final result of 
# the competition.
#
import golly as g
import model_classes as mclass
import model_functions as mfunc
import model_parameters as mparam
import numpy as np
import random
import copy
import time
import pickle
import os
import re
import sys
import pyautogui # tool for taking photos of the screen
#
# Set the path for the files where the runs are
# stored, relative to the location of this file
# (compete_seeds_photo.py).
#
run_dir = "../Experiments"
#
# The runs are numbered from "run1" to "run40".
# Choose a run here.
#
run_num = 32
#
# The generations are numbered from 0 to 100.
# Choose a generation here.
#
gen_num = 20
#
# It seems necessary to set the magnification manually, probably
# because the grid is a toroid, rather than the standard infinite
# grid. As the gen_num increases, it may be necessary to reduce
# the mag.
#
mag = 4
#
# Size of one generation.
#
pop_size = mparam.pop_size # 0 to 199
#
# Choose the number of competitions to photograph.
#
num_photos = 30
#
# Parameters for competitions between seeds.
#
width_factor = mparam.width_factor
height_factor = mparam.height_factor
time_factor = mparam.time_factor
#
# Have a short sleep while taking the photo, to make sure
# the photo is stable.
#
short_sleep = 1
#
# The above information should be sufficient to
# specify the location of the desired seeds.
#
seed_list = []
pickle_dir = run_dir + "/run" + str(run_num) # 1 to 40
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
# Run the competitions.
#
for i in range(num_photos):
  # prevent windows screen lock, which results in blank photos
  pyautogui.press('volumedown')
  pyautogui.press('volumeup')
  # randomly select a pair of seeds (without replacement)
  [seed1, seed2] = random.sample(seed_list, 2)
  # initialize scores
  score1 = 0.0
  score2 = 0.0
  # randomly rotate the seeds
  seed1 = seed1.random_rotate()
  seed2 = seed2.random_rotate()
  # the two seeds are red -- let's make the second one blue
  seed2.red2blue()
  # rule file
  rule_name = "Management"
  # make a toroidal universe for the competition
  [g_width, g_height, g_time] = mfunc.dimensions(seed1, seed2, \
    width_factor, height_factor, time_factor)
  g.setalgo("QuickLife") # use "HashLife" or "QuickLife"
  g.autoupdate(False) # do not update the view unless requested
  g.new(rule_name) # initialize cells to state 0
  g.setrule(rule_name + ":T" + str(g_width) + "," + str(g_height)) # make a toroid
  [g_xmin, g_xmax, g_ymin, g_ymax] = mfunc.get_minmax(g)
  g.setmag(mfunc.set_mag(g))
  # randomly position the two seeds, one on the left and 
  # the other on the right
  seed1.insert(g, g_xmin, -1, g_ymin, g_ymax)
  seed2.insert(g, +1, g_xmax, g_ymin, g_ymax)
  # take a photo of the two seeds together at the start
  seed_ID = str(seed1.unique_ID_num) + "_" + \
    str(seed2.unique_ID_num) + "_start"
  file_path = pickle_dir + "/photo_" + seed_ID + ".png"
  g.show(seed_ID)
  g.setmag(mag) 
  g.update()
  time.sleep(short_sleep)
  photo_object = pyautogui.screenshot()
  photo_object.save(file_path)
  # take a photo of the final result
  seed_ID = str(seed1.unique_ID_num) + "_" + \
    str(seed2.unique_ID_num) + "_stop"
  file_path = pickle_dir + "/photo_" + seed_ID + ".png"
  g.show(seed_ID)
  g.run(g_time)
  g.setmag(mag)
  g.update()
  time.sleep(short_sleep)
  photo_object = pyautogui.screenshot()
  photo_object.save(file_path)
#
#
# 