from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import numpy as np
from log_schema import Episode, Step, SCHEMA_VERSION

class Reader:
    def __init__(self, log_file):
        self._log_file = open(log_file, 'rb')

    def read(self):
        end = False
        Observation=[]
        Linear=[]
        Angular=[]

        while not end:
            try:
                log = pickle.load(self._log_file)
                for entry in log:
                    step = entry['step']
                    Observation.append(step[0])
                    action = step[1]
                    Linear.append(action[0])
                    Angular.append(action[1])

            except EOFError:
                end = True

        return Observation,Linear,Angular

    def modern_read(self):
        episode_data = None
        episode_index = 0
        episodes_size = 0
        observations_size = 0
        end = False
        Observation=[]
        Linear=[]
        Angular=[]
        while True:
            if episode_data is None:
                try:
                    episode_data = pickle.load(self._log_file)
                    episode_index = 0
                except EOFError:
                    print("End of log file!")
                    print("Size: ",len(Observation)," ",len(Linear)," ",len(Angular))
                    return Observation,Linear,Angular
            try:
                step = episode_data.steps[episode_index]
                episode_index+=1
                Observation.append(step.obs)
                Linear.append(step.action[0])
                Angular.append(step.action[1])
            except IndexError:
                episode_data=None
                continue

    def close(self):
        self._log_file.close()
