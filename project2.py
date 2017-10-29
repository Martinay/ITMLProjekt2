from ple import PLE
from ple.games.flappybird import FlappyBird
import sys

import random


class MonteCarloAgent:
    _q = {}
    _returns = {}
    _visitedStatesInEpisode = {}

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


    def addToVisited(self, stateAction, r):
        #if not(stateAction in self._visitedStatesInEpisode):
            self._visitedStatesInEpisode[stateAction] = r

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """

        if(end):
            a = 0

        s1Masked = self.maskState(s1)
        if(not(end)):
            self.addToVisited((s1Masked, a), r)
            return

        self._visitedStatesInEpisode[(s1Masked, a)] = r
        #self._visitedStatesInEpisode[self.maskState(s2)] = 0

        G = sum(self._visitedStatesInEpisode.itervalues())

        for key, value in self._visitedStatesInEpisode.iteritems():
            if not (key in self._returns):
                self._returns[key] = [G]
            else:
                self._returns[key].append(G)

        for key, value in self._visitedStatesInEpisode.iteritems():
                self._q[key] = float(sum(self._returns[key]))/len(self._returns[key])

        self._visitedStatesInEpisode = {}
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """

        epsilon = 10
        if (random.randint(0,100) < epsilon):
            random.randint(0, 1)

        return self.policy(state)

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """

        stateMasked = self.maskState(state)

        action0InQ = (stateMasked, 0) in self._q
        action1InQ = (stateMasked, 1) in self._q

        if action0InQ and action1InQ:
            valueAction0 = self._q[(stateMasked, 0)]
            valueAction1 = self._q[(stateMasked, 1)]
            if(valueAction0 == valueAction1):
                return random.randint(0, 1)
            if (valueAction0 > valueAction1):
                return 0
            else:
                return 1

        if action0InQ and not action1InQ:
            return 0

        if not action0InQ and action1InQ:
            return 1

        return random.randint(0, 1)


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

            print("score for this episode: %d" % score)
            if nb_episodes % 100 == 0:
                print("number of episodes left %d" % nb_episodes)
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


agent = MonteCarloAgent()
train_game(3000, agent)
run_game(1, agent)