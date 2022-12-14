"""
Model Classes

Peter Turney, July 6, 2022
"""
import golly as g
import model_parameters as mparam
import model_functions as mfunc
import random as rand
import numpy as np
import copy
"""
Make a class for seeds.
"""
#
# Note: Golly locates cells by x (horizontal) and y (vertical) coordinates,
# usually given in the format (x, y). On the other hand, we are storing
# these cells in matrices, where the coordinates are usually given in the
# format [row][column], where row is a vertical coordinate and column
# is a horizontal coordinate. Although it may be somewhat confusing, we
# use [x][y] for our matrices (x = row index, y = column index). That is:
#
#     self.xspan = self.cells.shape[0]
#     self.yspan = self.cells.shape[1]
#
class Seed:
  """
  A class for seeds.
  """
  #
  # __init__(self, xspan, yspan, pop_size) -- returns NULL
  #
  def __init__(self, xspan, yspan, pop_size):
    """
    Make an empty seed (all zeros).
    """
    # width of seed on the x-axis
    self.xspan = xspan 
    # height of seed on the y-axis
    self.yspan = yspan 
    # initial seed of zeros, to be modified later
    self.cells = np.zeros((xspan, yspan), dtype=np.int) 
    # initial history of zeros (fitness results)
    self.history = np.zeros(pop_size, dtype=np.float) 
    # initial similarities of zeros
    self.similarities = np.zeros(pop_size, dtype=np.float) 
    # position of seed in the population array, to be modified later
    self.address = 0 
    # count of living cells (ones) in the seed, to be modified later
    self.num_living = 0
    # unique ID number for each seed, to be modified later
    self.unique_ID_num = 0
    # type of birth of given seed, to be modified later
    # type = "random", "uniform_asexual", "variable_asexual",
    #        "sexual", "fusion", "fission"
    self.birth_type = ""
    # unique ID number of parent A, to be modified later
    self.parent_A_ID_num = 0
    # unique ID number of parent B, to be modified later
    self.parent_B_ID_num = 0
  #
  # randomize(self, seed_density) -- returns NULL
  #
  def randomize(self, seed_density):
    """
    Randomly set some cells to state 1. It is assumed that the
    cells in the given seed are initially all in state 0. The
    result is a seed in which the fraction of cells in state 1
    is approximately equal to seed_density (with some random
    variation). Strictly speaking, seed_density is the
    expected value of the fraction of cells in state 1.
    """
    for x in range(self.xspan):
      for y in range(self.yspan):
        if (rand.random() <= seed_density):
          self.cells[x][y] = 1
  #
  # check_colour(self) -- returns True or False
  #
  def check_colour(self):
    """
    Verify that the seeds are limited to 0 (white), 1 (red or black),
    and 5 (purple). The function score_pair() assumes that the input
    seed does not contain 2 (blue), 3 (orange), or 4 (green).
    """
    for x in range(self.xspan):
      for y in range(self.yspan):
        if (self.cells[x][y] == 2): # blue
          return False
        if (self.cells[x][y] == 3): # orange
          return False
        if (self.cells[x][y] == 4): # green
          return False
    return True
  #
  # shuffle(self) -- returns a shuffled copy of the given seed
  #
  def shuffle(self):
    """
    Make a copy of the given seed and then shuffle the cells in 
    the seed. The new shuffled seed will have the same dimensions
    and the same density of 1s and 0s as the given seed, but the 
    locations of the 1s and 0s will be different. (There is a very 
    small probability that shuffling might not result in any change, 
    just as shuffling a deck of cards might not change the deck.)
    The density of shuffled_seed is exactly the same as the density
    of the given seed.
    """
    #
    shuffled_seed = copy.deepcopy(self)
    #
    # for each location [x0][y0], randomly choose another location
    # [x1][y1] and swap the values of the cells in the two locations.
    #
    for x0 in range(self.xspan):
      for y0 in range(self.yspan):
        x1 = rand.randrange(self.xspan)
        y1 = rand.randrange(self.yspan)
        temp = shuffled_seed.cells[x0][y0]
        shuffled_seed.cells[x0][y0] = shuffled_seed.cells[x1][y1]
        shuffled_seed.cells[x1][y1] = temp
    #
    return shuffled_seed
  #
  # red2blue(self) -- returns NULL
  #
  def red2blue(self):
    """
    Switch cells from state 1 (red) to state 2 (blue).
    """
    for x in range(self.xspan):
      for y in range(self.yspan):
        if (self.cells[x][y] == 1):
          self.cells[x][y] = 2
  #
  # insert(self, g, g_xmin, g_xmax, g_ymin, g_ymax) -- returns NULL
  #
  def insert(self, g, g_xmin, g_xmax, g_ymin, g_ymax):
    """
    Write the seed into the Golly grid at a random location
    within the given bounds.
    g = the Golly universe
    s = a seed
    """
    step = 1
    g_xstart = rand.randrange(g_xmin, g_xmax - self.xspan, step)
    g_ystart = rand.randrange(g_ymin, g_ymax - self.yspan, step)
    for s_x in range(self.xspan):
      for s_y in range(self.yspan):
        g_x = g_xstart + s_x
        g_y = g_ystart + s_y
        s_state = self.cells[s_x][s_y]
        g.setcell(g_x, g_y, s_state)
  #
  # random_rotate(self) -- returns new_seed
  #
  def random_rotate(self):
    """
    Randomly rotate and flip the given seed and return a new seed.
    """
    rotation = rand.randrange(0, 4, 1) # 0, 1, 2, 3
    flip = rand.randrange(0, 2, 1) # 0, 1
    new_seed = copy.deepcopy(self)
    # rotate by 90 degrees * rotation (0, 90, 180 270)
    new_seed.cells = np.rot90(new_seed.cells, rotation) 
    if (flip == 1):
      # flip upside down
      new_seed.cells = np.flipud(new_seed.cells)
    new_seed.xspan = new_seed.cells.shape[0]
    new_seed.yspan = new_seed.cells.shape[1]
    return new_seed
  #
  # fitness(self) -- returns fitness
  #
  def fitness(self):
    """
    Calculate a seed's fitness from its history. 
    """
    history = self.history
    fitness = sum(history) / len(history)
    # If mutual_benefit_flag == 1, then test to see whether each partner
    # in the symbiote experiences mutual benefit. If a partner does not
    # experience mutual benefit, then adjust its fitness according to
    # mutual_benefit_penalty_factor.
    if (mparam.mutual_benefit_flag == 1):
      # If any partner in s1 does not experience mutual benefit, then
      # adjust the fitness.
      if (mfunc.test_mutual_benefit(g, self) == False):
        fitness = fitness * mparam.mutual_benefit_penalty_factor
    # return fitness
    return fitness
  #
  # mutate(self, prob_grow, prob_flip, prob_shrink, seed_density, mutation_rate) 
  # -- returns mutant
  #
  def mutate(self, prob_grow, prob_flip, prob_shrink, seed_density, mutation_rate):
    """
    Make a copy of self and return a mutated version of the copy.
    """
    #
    mutant = copy.deepcopy(self)
    #
    # prob_grow     = probability of invoking grow()
    # prob_flip     = probability of invoking flip_bits()
    # prob_shrink   = probability of invoking shrink()
    # seed_density  = target density of ones in an initial random seed
    # mutation_rate = probability of flipping an individual bit
    #
    assert prob_grow + prob_flip + prob_shrink == 1.0
    #
    uniform_random = rand.uniform(0, 1)
    #
    if (uniform_random < prob_grow):
      # this will be invoked with a probability of prob_grow
      mutant.grow(seed_density) 
    elif (uniform_random < (prob_grow + prob_flip)):
      # this will be invoked with a probability of prob_flip
      mutant.flip_bits(mutation_rate)
    else:
      # this will be invoked with a probability of prob_shrink
      mutant.shrink()
    # erase the parent's history from the child
    pop_size = len(self.history)
    mutant.history = np.zeros(pop_size, dtype=np.float)
    return mutant
  #
  # flip_bits(self, mutation_rate) -- returns NULL
  #
  def flip_bits(self, mutation_rate):
    """
    Mutate a seed by randomly flipping bits. Assumes the seed
    contains 0s and 1s.
    """
    num_mutations = 0
    for s_x in range(self.xspan):
      for s_y in range(self.yspan):
        if (self.cells[s_x][s_y] == 5):
          # if the cell is in state 5 (purple), then mutation
          # is not allowed -- purple indicates the buffer zone
          continue
        if (rand.uniform(0, 1) < mutation_rate):
          # flip cell value: 0 becomes 1 and 1 becomes 0
          self.cells[s_x][s_y] = 1 - self.cells[s_x][s_y]
          # count the number of mutations so far
          num_mutations += 1
    # force a minimum of one mutation -- there is no value
    # in having duplicates in the population
    while (num_mutations == 0):
      s_x = rand.randrange(self.xspan)
      s_y = rand.randrange(self.yspan)
      # cannot flip if state 5 (purple)
      if (self.cells[s_x][s_y] != 5):
        self.cells[s_x][s_y] = 1 - self.cells[s_x][s_y]
        num_mutations += 1
  #
  # exposed_border(self, choice) -- returns True or False
  #
  def exposed_border(self, choice):
    """
    Given a seed, check the specified choice to see whether it is
    a purple border (state 5).
    """
    # choice == 0 --> first row of the matrix
    # choice == 1 --> last row of the matrix
    # choice == 2 --> first column of the matrix
    # choice == 3 --> last column of the matrix
    #
    # x = row index, y = column index
    # 
    # start by assuming the given choice is a purple border
    border = True
    # check the specified choice
    if (choice == 0):
      # first row of the matrix
      for y in range(self.yspan):
        # if first row is not all purple ...
        if (self.cells[0][y] != 5): 
          border = False
          break
    elif (choice == 1):
      # last row of the matrix
      for y in range(self.yspan):
        # if last row is not all purple ...
        if (self.cells[-1][y] != 5): 
          border = False
          break
    elif (choice == 2):
      # first column of the matrix
      for x in range(self.xspan):
        # if the first column is not all purple ...
        if (self.cells[x][0] != 5):
          border = False
          break
    elif (choice == 3):
      # last column of the matrix
      for x in range(self.xspan):
        # if the last column is not all purple ...
        if (self.cells[x][-1] != 5):
          border = False
          break
    # return border status
    return border    
  #
  # shrink(self) -- returns NULL
  #
  def shrink(self):
    """
    Randomly remove rows or columns from a seed.
    """
    # x = row index, y = column index
    #
    # first we need to decide how to shrink
    #
    # choice == 0 --> first row of the matrix
    # choice == 1 --> last row of the matrix
    # choice == 2 --> first column of the matrix
    # choice == 3 --> last column of the matrix
    choice = rand.choice([0, 1, 2, 3])
    # now do it
    if ((choice == 0) and (self.xspan > mparam.min_s_xspan)):
      # delete first row
      self.cells = np.delete(self.cells, (0), axis=0)
      # update row count
      self.xspan = self.cells.shape[0]
      # if we are allowed to delete another row ...
      if (self.xspan > mparam.min_s_xspan):
        # if deleting the first row exposes a purple border ...
        if self.exposed_border(choice):
          # ... then we should also delete the purple border (state 5)
          self.cells = np.delete(self.cells, (0), axis=0)
          # update row count
          self.xspan = self.cells.shape[0]
    elif ((choice == 1) and (self.xspan > mparam.min_s_xspan)):
      # delete last row
      self.cells = np.delete(self.cells, (-1), axis=0)
      # update row count
      self.xspan = self.cells.shape[0]
      # if we are allowed to delete another row ...
      if (self.xspan > mparam.min_s_xspan):
        # if deleting the last row exposes a purple border ...
        if self.exposed_border(choice):
          # ... then we should also delete the purple border (state 5)
          self.cells = np.delete(self.cells, (-1), axis=0)
          # update row count
          self.xspan = self.cells.shape[0]
    elif ((choice == 2) and (self.yspan > mparam.min_s_yspan)):
      # delete first column
      self.cells = np.delete(self.cells, (0), axis=1)
      # update column count
      self.yspan = self.cells.shape[1]
      # if we are allowed to delete another column ...
      if (self.yspan > mparam.min_s_yspan):
        # if deleting the first column exposes a purple border ...
        if self.exposed_border(choice):
          # ... then we should also delete the purple border (state 5)
          self.cells = np.delete(self.cells, (0), axis=1)
          # update column count
          self.yspan = self.cells.shape[1]
    elif ((choice == 3) and (self.yspan > mparam.min_s_yspan)):
      # delete last column
      self.cells = np.delete(self.cells, (-1), axis=1)
      # update coloumn count
      self.yspan = self.cells.shape[1]
      # if we are allowed to delete another column ...
      if (self.yspan > mparam.min_s_yspan):
        # if deleting the last column exposes a purple border ...
        if self.exposed_border(choice):
          # ... then we should also delete the purple border (state 5)
          self.cells = np.delete(self.cells, (-1), axis=1)
          # update column count
          self.yspan = self.cells.shape[1]
    #
  #
  # grow(self, seed_density) -- returns NULL
  #
  def grow(self, seed_density):
    """
    Randomly add or remove rows or columns from a seed. Assumes 
    the seed contains 0s and 1s.
    """
    # x = row index, y = column index
    #
    # first we need to decide how to grow
    #
    # choice == 0 --> first row of the matrix
    # choice == 1 --> last row of the matrix
    # choice == 2 --> first column of the matrix
    # choice == 3 --> last column of the matrix
    choice = rand.choice([0, 1, 2, 3])
    # now do it
    if (choice == 0):
      # add a new row before the first row
      self.cells = np.vstack([np.zeros(self.yspan, dtype=np.int), self.cells])
      # initialize the new row with a density of approximately seed_density
      for s_y in range(self.yspan):
        # populate the new row
        if (rand.uniform(0, 1) < seed_density):
          self.cells[0][s_y] = 1
        # if the old row has any purple cells, then we must create matching
        # purple cells in the new row, to continue the purple border
        if (self.cells[1][s_y] == 5):
          self.cells[0][s_y] = 5
      #
    elif (choice == 1):
      # add a new row after the last row
      self.cells = np.vstack([self.cells, np.zeros(self.yspan, dtype=np.int)])
      # initialize the new row with a density of approximately seed_density
      for s_y in range(self.yspan):
        if (rand.uniform(0, 1) < seed_density):
          self.cells[-1][s_y] = 1
        # if the old row has any purple cells, then we must create matching
        # purple cells in the new row, to continue the purple border
        if (self.cells[-2][s_y] == 5):
          self.cells[-1][s_y] = 5
      #
    elif (choice == 2):
      # add a new column before the first column
      self.cells = np.hstack([np.zeros((self.xspan, 1), dtype=np.int), self.cells])
      # initialize the new column with a density of approximately seed_density
      for s_x in range(self.xspan):
        if (rand.uniform(0, 1) < seed_density):
          self.cells[s_x][0] = 1
        # if the old column has any purple cells, then we must create matching
        # purple cells in the new column, to continue the purple border
        if (self.cells[s_x][1] == 5):
          self.cells[s_x][0] = 5
      #
    elif (choice == 3):
      # add a new column after the last column
      self.cells = np.hstack([self.cells, np.zeros((self.xspan, 1), dtype=np.int)])
      # initialize the new column with a density of approximately seed_density
      for s_x in range(self.xspan):
        if (rand.uniform(0, 1) < seed_density):
          self.cells[s_x][-1] = 1
        # if the old column has any purple cells, then we must create matching
        # purple cells in the new column, to continue the purple border
        if (self.cells[s_x][-2] == 5):
          self.cells[s_x][-1] = 5
      #
    #
    # now let's update xspan and yspan to the new size
    self.xspan = self.cells.shape[0]
    self.yspan = self.cells.shape[1]
    #
  #
  # count_ones(self) -- returns number of ones in a seed
  #
  def count_ones(self):
    """
    Count the number of ones in a seed.
    """
    count = 0
    for x in range(self.xspan):
      for y in range(self.yspan):
        if (self.cells[x][y] == 1):
          count = count + 1
    return count
  #
  # count_colour(self, colour) -- returns number of given colour in a seed
  #
  def count_colour(self, colour):
    """
    Count the number of cells of a given colour, where the colour
    is represented by a number (but not zero).
    """
    count = 0
    for x in range(self.xspan):
      for y in range(self.yspan):
        if (self.cells[x][y] == colour):
          count = count + 1
    return count
  #
  # density(self) -- returns density of ones in a seed
  #
  def density(self):
    """
    Calculate the density of ones in a seed.
    """
    return self.count_ones() / float(self.xspan * self.yspan)
  #
#
#
#
