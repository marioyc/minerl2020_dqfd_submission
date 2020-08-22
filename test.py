
import json
import select
import time
import logging
import os
import pickle
import threading

from typing import Callable

import aicrowd_helper
import gym
import minerl
import abc
import numpy as np
import tqdm

import coloredlogs
coloredlogs.install(logging.DEBUG)

from torch import optim

import pfrl
from pfrl import explorers
from pfrl.wrappers import ContinuingTimeLimit, RandomizeAction
from pfrl.wrappers.atari_wrappers import LazyFrames

from dqfd import DQfD, PrioritizedDemoReplayBuffer
from env_wrappers import (
    ClusteredActionWrapper,
    MoveAxisWrapper, FrameSkip, FrameStack, ObtainPoVWrapper,
    PoVWithCompassAngleWrapper, FullObservationSpaceWrapper)
from q_functions import DuelingDQN

# All the evaluations will be evaluated on MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))

# Parallel testing/inference, **you can override** below value based on compute
# requirements, etc to save OOM in this phase.
EVALUATION_THREAD_COUNT = int(os.getenv('EPISODES_EVALUATION_THREAD_COUNT', 2))

class EpisodeDone(Exception):
    pass

class Episode(gym.Env):
    """A class for a single episode.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s,r,d,i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s,r,d,i



# DO NOT CHANGE THIS CLASS, THIS IS THE BASE CLASS FOR YOUR AGENT.
class MineRLAgentBase(abc.ABC):
    """
    To compete in the competition, you are required to implement a
    SUBCLASS to this class.
    
    YOUR SUBMISSION WILL FAIL IF:
        * Rename this class
        * You do not implement a subclass to this class 

    This class enables the evaluator to run your agent in parallel, 
    so you should load your model only once in the 'load_agent' method.
    """

    @abc.abstractmethod
    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_agent_on_episode(self, single_episode_env : Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs)) 
                ...
        
        NOTE: This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        raise NotImplementedError()


#######################
# YOUR CODE GOES HERE #
#######################

class MineRLMatrixAgent(MineRLAgentBase):
    """
    An example random agent. 
    Note, you MUST subclass MineRLAgentBase.
    """

    def load_agent(self):
        """In this example we make a random matrix which
        we will use to multiply the state by to produce an action!

        This is where you could load a neural network.
        """
        # Some helpful constants from the environment.
        flat_video_obs_size = 64*64*3
        obs_size = 64
        ac_size = 64
        self.matrix = np.random.random(size=(ac_size, flat_video_obs_size + obs_size))*2 -1
        self.flatten_obs = lambda obs: np.concatenate([obs['pov'].flatten()/255.0, obs['vector'].flatten()])
        self.act = lambda flat_obs: {'vector': np.clip(self.matrix.dot(flat_obs), -1,1)}


    def run_agent_on_episode(self, single_episode_env : Episode):
        """Runs the agent on a SINGLE episode.

        Args:
            single_episode_env (Episode): The episode on which to run the agent.
        """
        obs = single_episode_env.reset()
        done = False
        while not done:
            obs,reward,done,_ = single_episode_env.step(self.act(self.flatten_obs(obs)))


class MineRLRandomAgent(MineRLAgentBase):
    """A random agent"""
    def load_agent(self):
        pass # Nothing to do, this agent is a random agent.

    def run_agent_on_episode(self, single_episode_env : Episode):
        obs = single_episode_env.reset()
        done = False
        while not done:
            random_act = single_episode_env.action_space.sample()
            single_episode_env.step(random_act)


class DQfDAgent(MineRLAgentBase):
    def load_agent(self, n_clusters, use_noisy_net, frame_stack, gpu, path):
        q_func = DuelingDQN(n_clusters, n_input_channels=3 * frame_stack)

        if use_noisy_net == 'before-pretraining':
            pfrl.nn.to_factorized_noisy(q_func, sigma_scale=0.5)

        opt = optim.RMSprop(q_func.parameters(), lr=2.5e-4, alpha=0.95,
                            momentum=0.0, eps=1e-2)
        replay_buffer = PrioritizedDemoReplayBuffer()
        explorer = explorers.Greedy()

        def phi(x):
            return np.asarray(x, dtype=np.float32) / 255.0

        def reward_transform(x):
            return np.sign(x) * np.log(1 + np.abs(x))

        self.agent = DQfD(q_func, opt, replay_buffer, gamma=0.99,
                          explorer=explorer,
                          n_pretrain_steps=0,
                          demo_supervised_margin=0,
                          bonus_priority_agent=0,
                          bonus_priority_demo=0,
                          loss_coeff_nstep=0,
                          loss_coeff_supervised=0,
                          gpu=gpu,
                          replay_start_size=0,
                          target_update_interval=0,
                          clip_delta=0,
                          update_interval=1,
                          batch_accumulator="sum",
                          phi=phi, reward_transform=reward_transform,
                          minibatch_size=0)

        path = os.path.join(AGENT_PATH, "best")
        self.agent.load(path)

    def run_agent_on_episode(self, single_episode_env : Episode):
        obs = single_episode_env.reset()
        done = False
        while not done:
            action = self.agent.act(obs)
            obs, reward, done, _ = single_episode_env.step(action)


#####################################################################
# IMPORTANT: SET THIS VARIABLE WITH THE AGENT CLASS YOU ARE USING   # 
######################################################################
AGENT_TO_TEST = DQfDAgent
AGENT_PATH = "saved_agents/06e460d8f0306051bbd6a4b6fab91ac6f204c70d-00000000-da33b4ea/"

####################
# EVALUATION CODE  #
####################


def make_env(env, frame_skip, frame_stack, kmeans):
    env = ContinuingTimeLimit(env, max_episode_steps=18000)  # 18000: ObtainDiamond
    env = ObtainPoVWrapper(env)
    env = FrameSkip(env, skip=frame_skip)
    env = MoveAxisWrapper(env, source=-1, destination=0)
    env = FrameStack(env, frame_stack, channel_order='chw')
    env = ClusteredActionWrapper(env, clusters=kmeans.cluster_centers_)
    env = RandomizeAction(env, 0.001)
    return env

def main(frame_skip, frame_stack, gpu, n_clusters, use_noisy_net):
    logger = logging.getLogger(__name__)

    logger.debug("Loading kmeans")
    kmeans_file = open(os.path.join(AGENT_PATH, "kmeans.pkl"), "rb")
    kmeans = pickle.load(kmeans_file)
    if kmeans.cluster_centers_.shape[0] != n_clusters:
        raise Exception("Wrong number of clusters")

    logger.debug("Creating agent")
    agent = AGENT_TO_TEST()
    assert isinstance(agent, MineRLAgentBase)

    logger.debug("MINERL_MAX_EVALUATION_EPISODES: %s", MINERL_MAX_EVALUATION_EPISODES)
    assert MINERL_MAX_EVALUATION_EPISODES > 0
    logger.debug("EVALUATION_THREAD_COUNT: %s", EVALUATION_THREAD_COUNT)
    assert EVALUATION_THREAD_COUNT > 0

    # Create the parallel envs (sequentially to prevent issues!)
    logger.debug("Create the parallel envs")
    envs = []
    for i in range(EVALUATION_THREAD_COUNT):
        logger.debug("Creating env %s", i)
        env = gym.make(MINERL_GYM_ENV)
        env = make_env(env, frame_skip, frame_stack, kmeans)
        envs.append(env)

    logger.debug("Loading agent")
    agent.load_agent(n_clusters, use_noisy_net, frame_stack, gpu, AGENT_PATH)

    episodes_per_thread = [MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT for _ in range(EVALUATION_THREAD_COUNT)]
    episodes_per_thread[-1] += MINERL_MAX_EVALUATION_EPISODES - EVALUATION_THREAD_COUNT *(MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT)
    # A simple funciton to evaluate on episodes!
    def evaluate(i, env):
        logger.info("[{}] Starting evaluator.".format(i))
        for i in range(episodes_per_thread[i]):
            try:
                agent.run_agent_on_episode(Episode(env))
            except EpisodeDone:
                logger.info("[{}] Episode complete".format(i))
                pass
    
    logger.debug("Starting evaluation")
    evaluator_threads = [threading.Thread(target=evaluate, args=(i, envs[i])) for i in range(EVALUATION_THREAD_COUNT)]
    for thread in evaluator_threads:
        thread.start()

    # wait fo the evaluation to finish
    for thread in evaluator_threads:
        thread.join()

if __name__ == "__main__":
    main()
    

