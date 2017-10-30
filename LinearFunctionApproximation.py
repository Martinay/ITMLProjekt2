import random

class LFA:
    alpha = 0.1
    gamma = 1
    epsilon = 0.1

    _thetaA0 = [0, 0, 0, 0]
    _thetaA1 = [0, 0, 0, 0]

    def __init__(self):
        random.seed(42)
        return

    def transfromState(s):
        return ( s['next_pipe_top_y'], s['player_y'], s['player_vel'], s['next_pipe_dist_to_player'])

    def calcQA0(self, state):
        return [self._thetaA0[0] * state[0], self._thetaA0[1] * state[1], self._thetaA0[2] * state[2], self._thetaA0[3] * state[3]]

    def calcQA1(self, state):
        return [self._thetaA1[0] * state[0], self._thetaA1[1] * state[1], self._thetaA1[2] * state[2], self._thetaA1[3] * state[3]]

    def reward_values(self):
        """ returns the reward values used for training

            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        if a == 0:
            theta = self._thetaA0
        else:
            theta = self._thetaA1

        theta[0] = theta[0] + self.alpha * ( r + self.gamma * )

        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        if self.epsilon > random.random():
            return random.randint(0, 1)
        return self.policy(state)

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        transformedState = self.transfromState(state)
        qAction0 = self.calcQA0(transformedState)
        qAction1 = self.calcQA1(transformedState)

        if qAction0 == qAction1:
            return random.randint(0, 1)
        if qAction0 > qAction1:
            return 0
        return 1
