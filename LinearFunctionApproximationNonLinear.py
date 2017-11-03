import random
import numpy as np

class LFA:
    alpha = 0.1
    gamma = 1
    epsilon = 0.1

    _thetaA0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    _thetaA1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    def __init__(self):
        random.seed(42)

        return

    def transfromState(self, s):
        features = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        delta_y = s['player_y'] - s['next_pipe_top_y']
        if delta_y < -250:
            features[0] = 1
        elif delta_y < -150:
            features[1] = 1
        elif delta_y < -110:
            features[2] = 1
        elif delta_y < -80:
            features[3] = 1
        elif delta_y < -50:
            features[4] = 1
        elif delta_y < -20:
            features[5] = 1
        elif delta_y < 0:
            features[6] = 1
        elif delta_y < 20:
            features[7] = 1
        elif delta_y < 40:
            features[8] = 1
        elif delta_y < 60:
            features[9] = 1
        elif delta_y < 80:
            features[10] = 1
        elif delta_y < 100:
            features[11] = 1
        elif delta_y < 120:
            features[12] = 1
        elif delta_y < 150:
            features[13] = 1
        elif delta_y < 180:
            features[14] = 1
        elif delta_y < 250:
            features[15] = 1
        elif delta_y < 350:
            features[16] = 1
        else:
            features[17] = 1

        player_vel = s['player_vel'] / 2
        if player_vel < -3:
            features[18] = 1
        elif player_vel < -2:
            features[19] = 1
        elif player_vel < -1:
            features[20] = 1
        elif player_vel < 0:
            features[21] = 1
        elif player_vel < 1:
            features[22] = 1
        elif player_vel < 2:
            features[23] = 1
        elif player_vel < 3:
            features[24] = 1
        elif player_vel < 4:
            features[25] = 1
        else:
            features[26] = 1

        distance = s['next_pipe_dist_to_player'] * 15 / 288

        if distance < 1:
            features[27] = 1
        elif distance < 2:
            features[28] = 1
        elif distance < 3:
            features[29] = 1
        elif distance < 4:
            features[30] = 1
        elif distance < 5:
            features[31] = 1
        elif distance < 6:
            features[32] = 1
        elif distance < 7:
            features[33] = 1
        elif distance < 8:
            features[34] = 1
        elif distance < 9:
            features[35] = 1
        elif distance < 10:
            features[36] = 1
        elif distance < 11:
            features[37] = 1
        elif distance < 12:
            features[38] = 1
        elif distance < 13:
            features[39] = 1
        elif distance < 14:
            features[40] = 1
        elif distance < 15:
            features[41] = 1
        else:
            features[42] = 1

        return features

    def calcQA0(self, state):
        return np.dot(self._thetaA0, state)

    def calcQA1(self, state):
        return np.dot(self._thetaA1, state)

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
        maxNextQ = 0
        if not end:
            maskS2 = self.transfromState(s2)
            qs2A0 = self.calcQA0(maskS2)
            qs2A1 = self.calcQA1(maskS2)
            if (qs2A0 > qs2A1):
                maxNextQ = qs2A0
            else:
                maxNextQ = qs2A1

        transformedS1 = self.transfromState(s1)

        if a == 0:
            theta = self._thetaA0
            currentQ = self.calcQA0(transformedS1)
        else:
            theta = self._thetaA1
            currentQ = self.calcQA1(transformedS1)


        factor = self.alpha * (r + self.gamma * maxNextQ - currentQ)
        newTheta = theta + np.multiply(transformedS1, factor)

        if a == 0:
            self._thetaA0 = newTheta
        else:
            self._thetaA1 = newTheta

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
