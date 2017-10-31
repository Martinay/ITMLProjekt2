import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class QLearingAgentOptimized:
    alpha = 0.1
    gamma = 1
    epsilon = 0.1

    _q = defaultdict(lambda: [0, 0])

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
        return {"positive": 1.0, "tick": 0.0, "loss": -100.0}

    def maskState(self, s):
        ydifference = s['player_y'] - s['next_pipe_top_y']
        if ydifference < 0:
            ydifference = -1
        else:
            ydifference *= 15 / 512.0
        return (int(ydifference), int(s['player_vel'] / 4), int(s['next_pipe_dist_to_player'] * 15 / 512))

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        maskS1 = self.maskState(s1)

        currentQ = self._q[maskS1][a]
        maxNextQ = 0
        if not end:
            maskS2 = self.maskState(s2)
            _qstate2 = self._q[maskS2]
            if(_qstate2[0] > _qstate2[1]):
                maxNextQ = _qstate2[0]
            else:
                maxNextQ = _qstate2[1]
        newQ = currentQ + self.alpha * (r + self.gamma * maxNextQ - currentQ)

        if a == 0:
            self._q[maskS1][0] = newQ
        else:
            self._q[maskS1][1] = newQ

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
        maskState = self.maskState(state)

        qValues = self._q[maskState]
        qAction0 = qValues[0]
        qAction1 = qValues[1]

        if qAction0 == qAction1:
            return random.randint(0, 1)
        if qAction0 > qAction1:
            return 0
        return 1

    def plotQ(self, what='v'):
        # This function assumes that q = { (s, [q(s,flap), q(s,noop)]) ,... },
        # that is, q is a dictionary where each entry is mapping from a state to
        # an array of q-values.
        # States are expected to be encoded as 4-tuples with the (discretized
        # versions of) the 4 values 'next_pipe_top_y', 'player_y', 'player_vel',
        # 'next_pipe_dist_to_player' (in this order).
        #
        # "what" defines which value is plotted and can be one of 'q_flap',
        # 'q_noop', 'v' or 'pi'

        # turn q into a list of records, one for each state
        data = [s + tuple(self._q[s]) for s in self._q.keys()]
        # turn this into a dataframe, giving the columns the right names
        df = pd.DataFrame(data=data,
                          columns=('next_pipe_top_y', 'player_y', 'player_vel',
                                   'next_pipe_dist_to_player', 'q_flap', 'q_noop')
                          )
        # add a few more columns that might come in handy
        df['delta_y'] = df['player_y'] - df['next_pipe_top_y']
        df['v'] = df[['q_noop', 'q_flap']].max(axis=1)
        df['pi'] = (df[['q_noop', 'q_flap']].idxmax(axis=1) == 'q_flap') * 1
        # group entries that have the same 'delta_y' and 'next_pipe_dist_to_player',
        # by taking the mean of the remaining values
        df = df.groupby(
            ['delta_y', 'next_pipe_dist_to_player'], as_index=False).mean()

        plt.figure()
        if what in ('q_flap', 'q_noop', 'v'):
            # for estimated values, use a range of -5 to 5
            ax = sns.heatmap(
                df.pivot('delta_y', 'next_pipe_dist_to_player', what),
                vmin=-5, vmax=5, cmap='coolwarm', annot=True, fmt='.2f')
        elif what == 'pi':
            # for the policy, use a range of 0 to 1
            ax = sns.heatmap(
                df.pivot('delta_y', 'next_pipe_dist_to_player', 'pi'),
                vmin=0, vmax=1, cmap='coolwarm')
        # invert the x axis such that states further from the next pipe are on the
        # left and states closer to the next pipe are on the right
        ax.invert_xaxis()
        ax.set_title(what)
        plt.show()