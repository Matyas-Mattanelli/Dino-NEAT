import pickle
import re
from run_neat import sim

# Specify RUN
RUN = 'Run 20240523011128 fifth (3 obs)'

# Load the best net
with open(f'{RUN}/best_net', 'rb') as handle:
    net = pickle.load(handle)

# Load parameters
with open(f'{RUN}/params.txt', 'r') as handle:
    params = handle.readlines()
NUM_SIM = int(re.search('([0-9]*)$', params[0].strip()).groups()[0])
NUM_OBS = int(re.search('([0-9]*)$', params[2].strip()).groups()[0])

# Calculate fitness
fitness = sim(net, NUM_SIM, NUM_OBS)
print(f'Best fitness for {RUN}: {fitness}')

# Store fitness
with open(f'{RUN}/best_fitness.txt', 'w') as handle:
    handle.write(str(fitness))