#import matplotlib as mpl
#mpl.rcParams['backend.qt4'] = 'PySide'
#mpl.use('Qt4Agg')
#from pylab import *
#ion()

from ple import PLE
from ple.games.flappybird import FlappyBird
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import random


class QLearingAgent:
    alpha = 0.5
    gamma = 1
    epsilon = 0.1

    _q = {}

    def __init__(self):
        random.seed(42)
        return

    def reward_values(self):
        """ returns the reward values used for training

            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def maskState(self, s):
        return (int(s['player_y'] * 15 / 512), int(s['player_vel']), int(s['next_pipe_top_y'] * 15 / 512),int(s['next_pipe_dist_to_player'] * 15 / 512))

    def getQValue(self, stateActionPair):
        if not (stateActionPair in self._q):
            self._q[stateActionPair] = random.randint(-5,5)

        return self._q[stateActionPair]

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        maskS1 = self.maskState(s1)
        maskS2 = self.maskState(s2)

        currentQ = self.getQValue((maskS1, a))
        maxNextQ = 0
        if not end:
            maxNextQ = max([self.getQValue((maskS2, 0)), self.getQValue((maskS2, 1))])
        newQ = currentQ + self.alpha * (r+ self.gamma * maxNextQ - currentQ)

        self._q[(maskS1, a)] = newQ
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        randomNumber = random.randint(0,100)
        if self.epsilon * 100 < randomNumber:
            return random.randint(0, 1)

        return self.policy(state)

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        maskState = self.maskState(state)

        qAction0 = self.getQValue((maskState, 0))
        qAction1 = self.getQValue((maskState, 1))

        if qAction0 > qAction1:
            return 0

        return 1

    def printQ(self):
        x =[]
        y = []
        z = []

        for key, value in self._q.iteritems():
            z.append(value)
            state = key[0]
            x.append(key[0][3])
            y.append(key[0][2] - key[0][0])

        data = pd.DataFrame(data={'x-distance': x, 'y-distance': y, 'q-value': z})
        datapivot = pd.pivot_table(data, "q-value", "y-distance", "x-distance")
        sns.heatmap(data=datapivot, linewidths=.5, linecolor='lightgray')
        plt.show(block = True)
        return

def train_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """
    reward_values = agent.reward_values()

    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None, reward_values=reward_values)

    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        s1 = env.game.getGameState()
        action = agent.training_policy(s1)
        # print("reward=%s" % s1)
        # step the environment
        reward = env.act(env.getActionSet()[action])
        # print("reward=%d" % reward)
        s2 = env.game.getGameState()
        isGameOver = env.game_over()
        # for training let the agent observe the current state transition
        agent.observe(s1, action, reward, s2, isGameOver)

        score += reward

        # reset the environment if the game is over
        if isGameOver:

            if nb_episodes % 100 == 0:
                print("score for this episode: %d" % score)
                print("number of episodes left %d" % nb_episodes)
                #agent.printQ()
            env.reset_game()
            nb_episodes -= 1
            score = 0



def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}

    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
              reward_values=reward_values)
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        action = agent.policy(env.game.getGameState())

        # step the environment
        reward = env.act(env.getActionSet()[action])
        #print("reward=%d" % reward)

        score += reward

        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0

agent = QLearingAgent()
train_game(10000, agent)
run_game(1, agent)
agent.printQ()