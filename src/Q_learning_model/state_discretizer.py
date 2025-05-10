# state_discretizer.py

import numpy as np
import math


class IHT:
    def __init__(self, sizeval):
        self.size = sizeval
        self.dictionary = {}
        self.overfull_count = 0

    def getindex(self, obj, readonly=False):
        if obj in self.dictionary:
            return self.dictionary[obj]
        elif readonly:
            return None
        else:
            if len(self.dictionary) >= self.size:
                if self.overfull_count == 0:
                    print('IHT full, starting to allow collisions')
                self.overfull_count += 1
                return hash(obj) % self.size
            else:
                index = len(self.dictionary)
                self.dictionary[obj] = index
                return index


def tiles(iht, num_tilings, floats, ints=[]):
    """
    Returns a list of tiles corresponding to the given floats and ints.
    """
    qfloats = [math.floor(f * num_tilings) for f in floats]
    tiles = []
    for tiling in range(num_tilings):
        coords = [tiling]
        for i in range(len(qfloats)):
            coords.append((qfloats[i] + tiling) // num_tilings)
        coords.extend(ints)
        tile = iht.getindex(tuple(coords))
        tiles.append(tile)
    return tiles


class StateDiscretizer:
    def __init__(self, env, num_tilings=32, tiles_per_dim=8, iht_size=4096):
        self.env = env
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.iht_size = iht_size
        self.iht = IHT(self.iht_size)
        self.state_low = self.env.observation_space.low
        self.state_high = self.env.observation_space.high
        self.state_range = self.state_high - self.state_low

    def discretize(self, state):
        # Normalize continuous state variables (first 6 dimensions) to [0, 1]
        continuous_state = state[:6]
        scaled_state = (continuous_state - self.state_low[:6]) / self.state_range[:6]
        scaled_state = np.clip(scaled_state, 0, 0.9999)  # Avoid edge cases

        # Include leg contact information (last 2 dimensions) as integer variables
        leg_contact = state[6:].astype(int)

        # Get active tiles with both continuous and integer variables
        active_tiles = tiles(
            self.iht,
            self.num_tilings,
            scaled_state * self.tiles_per_dim,
            ints=leg_contact.tolist()
        )
        return active_tiles
