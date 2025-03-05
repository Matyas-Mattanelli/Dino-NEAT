#!/usr/bin/env python3
from game.dino import *
from game.agent import Agent
import pickle
import numpy as np
from run_neat import get_move, MOVES
import re

# Specify parameters
folder = 'Run 20240523011128 fifth (3 obs)'

# Load parameters
with open(f'{folder}/params.txt', 'r') as handle:
    params = handle.readlines()
NUM_OBS = int(re.search('([0-9]*)$', params[2].strip()).groups()[0])

with open(f'{folder}/best_net', 'rb') as handle:
    neural_network = pickle.load(handle)

class MyAgent(Agent):
    """Reflex agent static class for Dino game."""

    @staticmethod
    def get_move(game: Game) -> DinoMove:
        """
        Note: Remember you are creating simple-reflex agent, that should not
        store or access any information except provided.
        """
        # # Check if any obstacles are available
        # if len(game.obstacles):
        
        #     # Find the required number of closest obstacles
        #     obs_dists = np.zeros(len(game.obstacles))
        #     for idx, obs in enumerate(game.obstacles): # Loop through the obstacles and get their distance from dino
        #         obs_dists[idx] = obs.rect.x - game.dino.x
        #         if obs_dists[idx] < 0:
        #             obs_dists[idx] = np.inf
            
        #     # Find the characteristics of the closest obstacles
        #     obs_dists = obs_dists.argsort() # Sort the obstacles by distance
        #     obs_inputs = [0] * NUM_OBS * 4#5
        #     for i in range(min(NUM_OBS, len(obs_dists))): # Loop through the required number of obstacles
        #         obs = game.obstacles[obs_dists[i]]
        #         obs_inputs[4*i] = obs.rect.x - game.dino.x
        #         obs_inputs[4*i+1] = obs.rect.y
        #         obs_inputs[4*i+2] = obs.type.width
        #         obs_inputs[4*i+3] = obs.type.height
        #         #obs_inputs[4*i+4] = obs.speed
            
        #     # Get the inputs
        #     inputs = [game.dino.y, game.speed] + obs_inputs

        #     # Get the move
        #     move = neural_network.activate(inputs)
        #     move = MOVES[np.argmax(move)]
        
        # # If no obstacles, return no move
        # else:
        #     move = DinoMove.NO_MOVE

        # print(move.name)
        # return move
        return get_move(neural_network, game, True, NUM_OBS)