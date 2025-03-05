import neat
import visualize
import numpy as np
import os
import datetime
import pickle
import shutil
import re

from game.dino import Game, DinoMove
from random import randrange

os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

# Specify the parameters
RUN = None
NUM_SIM = 20
NUM_GEN = 500
NUM_OBS = 3 # Number of obstacles to consider
SEED = 42
config_file = 'config-feedforward'

# Specify all the possible moves
MOVES = [DinoMove.NO_MOVE, DinoMove.UP, DinoMove.DOWN, DinoMove.RIGHT, DinoMove.LEFT, DinoMove.UP_RIGHT, DinoMove.UP_LEFT, DinoMove.DOWN_RIGHT, DinoMove.DOWN_LEFT]
#MOVES = [DinoMove.NO_MOVE, DinoMove.UP, DinoMove.DOWN, DinoMove.RIGHT, DinoMove.LEFT]
#MOVES = [DinoMove.NO_MOVE, DinoMove.UP, DinoMove.DOWN]

def get_move(neural_network: neat.nn.FeedForwardNetwork, game: Game, print_move: bool = False, num_obs: int = NUM_OBS) -> DinoMove:
    """
    Predict the next move using a neural network based on the position of dino and three closest obstacles, and speed of the game
    """
    # Check if any obstacles are available
    if len(game.obstacles):
    
        # Find the required number of closest obstacles
        obs_dists = np.zeros(len(game.obstacles))
        for idx, obs in enumerate(game.obstacles): # Loop through the obstacles and get their distance from dino
            obs_dists[idx] = abs(obs.rect.x - game.dino.x)
            # if obs_dists[idx] < 0:
            #     obs_dists[idx] = np.inf
        
        # Find the characteristics of the closest obstacles
        obs_dists = obs_dists.argsort() # Sort the obstacles by distance
        obs_inputs = [0] * num_obs * 3
        for i in range(min(num_obs, len(obs_dists))): # Loop through the required number of obstacles
            obs = game.obstacles[obs_dists[i]]
            obs_inputs[3*i] = obs.rect.x - game.dino.x
            obs_inputs[3*i+1] = obs.rect.top - game.dino.y
            obs_inputs[3*i+2] = obs.rect.bottom - game.dino.y
        
        # Get the inputs
        #inputs = [game.dino.x, game.dino.y, game.speed] + obs_inputs
        inputs = [game.speed] + obs_inputs

        # Get the move
        move = neural_network.activate(inputs)
        move = MOVES[np.argmax(move)]
    
    # If no obstacles, return no move
    else:
        move = DinoMove.NO_MOVE

    if print_move:
        print(move.name)

    return move


def sim(neural_network: neat.nn.FeedForwardNetwork, num_sim: int = NUM_SIM, num_obs: int = NUM_OBS, seeds: list | None = None) -> float:
    """
    Simulate dino games with the given neural network
    """
    # Specify the seeds
    if seeds is None:
        #seeds = [randrange(0, 999999999) for _ in range(num_sim)]
        seeds = [42]

    # CREATE GAME
    game: Game = Game(new_game=False)

    score = 0

    # SIM
    for seed in seeds:
        game.new_game(seed)
        while not game.game_over:
            # Perform tick
            move: DinoMove = get_move(neural_network, game, num_obs=num_obs)
            game.tick(move)

            # Break if score is too high
            if game.score > 6000:
                break

        # Save score
        score += game.score

    avg_score = score / num_sim
    
    return avg_score


def eval_genomes(genomes, config):
    seeds = [randrange(0, 999999999) for _ in range(NUM_SIM)]
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = sim(net, NUM_SIM, NUM_OBS, seeds)


def run(config_file, folder='', restore=False):

    if restore:
        checkpoint_files = [i for i in os.listdir(folder) if 'neat-checkpoint' in i]
        gen = [int(re.search('neat-checkpoint-([0-9]*)', i).groups()[0]) for i in checkpoint_files]
        p = neat.Checkpointer().restore_checkpoint(f'{folder}/neat-checkpoint-{max(gen)}')
    
    else:
        gen = [0]
        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(NUM_GEN // 10, filename_prefix=f'{folder}/neat-checkpoint-'))

    # Run for the given number of generations.
    winner = p.run(eval_genomes, NUM_GEN)

    # Store best network
    winner_net = neat.nn.FeedForwardNetwork.create(winner, p.config) # Need to get config for restoring
    with open(f'{folder}/best_net', 'wb') as out:
        pickle.dump(winner_net, out)

    # Calculate and store fitness
    fitness = sim(winner_net, NUM_SIM, NUM_OBS)
    with open(f'{folder}/best_fitness.txt', 'w') as handle:
        handle.write(str(fitness))

    #visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True, filename=folder+'/avg_fitness.svg')
    visualize.plot_species(stats, view=True, filename=folder+'/species.svg')
    visualize.draw_net(p.config, winner, True, f'{folder}/best_net.gv')


if __name__ == '__main__':
    if RUN is None:
        # Create folder for results
        folder = f'Run {datetime.datetime.now():%Y%m%d%H%M%S}'
        os.makedirs(folder)

        # Copy the configuration file
        shutil.copyfile(config_file, f'{folder}/{config_file}')

        # Store parameters
        with open(f'{folder}/params.txt', 'w') as handle:
            handle.write(f'{NUM_SIM=}\n')
            handle.write(f'{NUM_GEN=}\n')
            handle.write(f'{NUM_OBS=}\n')
    else:
        # Specify existing folder
        folder = RUN

        # Load parameters
        with open(f'{folder}/params.txt', 'r') as handle:
            params = handle.readlines()
        NUM_SIM = int(re.search('([0-9]*)$', params[0].strip()).groups()[0])
        #NUM_GEN = int(re.search('([0-9]*)$', params[1].strip()).groups()[0])
        NUM_OBS = int(re.search('([0-9]*)$', params[2].strip()).groups()[0])
    

    # Run the algorithm
    run(config_file, folder, RUN is not None)