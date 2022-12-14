#
# Model Parameters
#
# Peter Turney, September 2, 2022
#
# Set various parameters for running experiments
#
#
#
# Type of experiment:
#
# 1 = uniform asexual  = asexual with bit-flip mutation but no seed size change
# 2 = variable asexual = asexual with bit-flip mutation and seed size change
# 3 = sexual           = bit-flip mutation, seed size change, and crossover
# 4 = symbiotic        = mutation, size change, crossover, fission, and fusion
#
experiment_type_num = 4
#
if (experiment_type_num == 1):
  experiment_type_name = "uniform asexual"
elif (experiment_type_num == 2):
  experiment_type_name = "variable asexual"
elif (experiment_type_num == 3):
  experiment_type_name = "sexual"
else:
  assert experiment_type_num == 4
  experiment_type_name = "symbiotic"
#
# Set the random number generator seed here. If random_seed is negative,
# then Python will automatically set a random number seed. Note that, if
# random_seed is negative, then the experiment cannot be exactly repeated.
#
random_seed = -1
#
# Directory for log files. 
#
log_directory = "../Experiments Type 3/run15"
#
# Fixed population size. Steady-state model for evolution. For every
# birth, there is one death.
#
pop_size = 200
#
# The number of trials for a given pair of seeds. Each pair of seeds
# competes several times, with different rotations and different
# locations in Golly space. The final result for the pair is the 
# average of the trials.
#
num_trials = 2
#
# run_length: the number of children born in one run. Each child that
# is born will replace an existing member of the population, so the
# size of the population is constant.
#
# num_generations: one generation is when pop_size children are born.
#
# We want run_length to be evenly divisible by pop_size, because we
# take samples of the population at the end of each generation.
#
num_generations = 100
#
run_length = num_generations * pop_size
#
# Minimum seed sizes.
#
min_s_xspan = 5
min_s_yspan = 5
#
# Initial seed sizes.
#
s_xspan = 5
s_yspan = 5
#
assert s_xspan >= min_s_xspan
assert s_yspan >= min_s_yspan
#
# Maximum seed area: The maximum seed area increases linearly with
# each new child born. Here we set the desired maximum seed area
# for the first child and the last child. The motivation for this
# linear limit to the seed area is to prevent an explosive increase
# in seed area, which causes the simulation to run extremely
# slowly. This limit is due to a lack of patience on my part; it
# is not intended to model a natural phenomenon.
#
max_area_first = 120
max_area_last = 170
#
# Probability of state 1 for random initialization of a new seed.
# For the initial random seeds, use a random density of 37.5%. See:
#
# http://www.njohnston.ca/2009/06/longest-lived-soup-density-in-conways-game-of-life/
# http://www.njohnston.ca/2009/07/the-maximal-lifespan-of-patterns-in-conways-game-of-life/
#
seed_density = 0.375
#
# Multiply the sizes of the seeds by this factor to get the size
# of the toroid. Thus the toroid expands as the seeds get larger.
# The width is greater than the height because the two seeds are
# positioned side-by-side horizontally.
#
width_factor = 6.0
height_factor = 3.0
#
# Multiply the sizes of the toroid by this factor to get the
# number of generations for a run. Thus the running time
# increases as the toroid expands.
#
time_factor = 6.0
#
# The size of the random sample for a tournament. The most fit
# member of the tournament sample will be allowed to reproduce.
#
tournament_size = 2
#
# Probability for mutation in uniform asexual experiments
# (type 1; see above).
#
mutation_rate = 0.01
#
# Probabilities for the three kinds of mutation in variable
# asexual experiments (type 2; see above).
#
# prob_grow     = probability of invoking grow()
# prob_flip     = probability of invoking flip_bits()
# prob_shrink   = probability of invoking shrink()
#
# Constraints:
#
# - all parameters must lie in the range from 0.0 to 1.0
# - the parameters must sum to 1.0
#
prob_grow     = 0.2
prob_flip     = 0.6
prob_shrink   = 0.2
#
assert prob_grow + prob_flip + prob_shrink == 1.0
#
# Elite size: size of the sample that will be used to evaluate
# whether the best seeds (the elite) are getting better.
#
elite_size = pop_size
#
# Two seeds are allowed to mate (that is, they are considered as
# member of the same species) when they are highly similar but
# not identical. The following parameters define the desired
# region of similarity for mating.
#
min_similarity = 0.80
max_similarity = 0.99
#
# For symbiosis, set the probabilities of fission and fusion.
# Because fusion can result in large seeds, which will slow down 
# the simulation, the probability of fusion should be relatively 
# low.
#
prob_fission = 0.01
prob_fusion = 0.005
#
assert prob_fission + prob_fusion <= 1.0
#
# Fusion test flag: If this flag is 0, then everything proceeds
# as usual. If this flag is 1, then fusion is modified by
# shuffling one of the two seeds that are about to be fused.
# The purpose of shuffling is to demonstrate that the structure
# of the seeds is important for fitness with fusion. That is,
# shuffling is expected to reduce fitness with fusion. In most 
# cases, this test flag should be set to 0.
#
fusion_test_flag = 0
#
# Immediate symbiosis flag: If this flag is 0, then everything
# proceeds as usual. If this flag is 1, then fusion is modified
# by requiring a newly fused pair of seeds to be more fit than
# the fitnesses of the two members of the pair before they were
# fused together. If a fused seed fails this test, then one of
# members of the pair is passed on to Layer 3, the sexual
# layer.
#
immediate_symbiosis_flag = 0
#
# Mutual benefit flag: If this flag is 0, then everything
# proceeds as usual. If this flag is 1, then each partner in
# a symbiote is tested to see whether it experiences mutual
# benefit. A symbiote experiences mutual benefit if its growth
# alone (when it is removed from the symbiote) is less than its
# growth together (when it grows together with the other partners).
# When measuring the fitness of a symbiote, we first test to
# see whether all partners experience mutual benefit. If they
# all benefit from being together in a symbiotic relation, then
# we proceed to measure the fitness of the symbiote in the usual
# way, by a series of competitions with other symbiotes. If they
# do not all benefit from being together, then the fitness is
# reduced by applying the mutual benefit penalty factor.
#
mutual_benefit_flag = 0
#
# Mutual benefit penalty factor: If there is a partner in
# a symbiote that does not benefit, then the fitness of the
# symbiote is reduced by multiplying the fitness with the
# mutual benefit penalty factor.
#
mutual_benefit_penalty_factor = 1.0
#