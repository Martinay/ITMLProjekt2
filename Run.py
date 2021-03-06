from ple import PLE
from ple.games.flappybird import FlappyBird
import matplotlib.pyplot as plt
import numpy as np

from LinearFunctionApproximation import LFA
from QLearningAgent import QLearingAgent
from QLearningAgentOptimizedReward import QLearingAgentOptimizedReward
from QLearningAgentOptimizedGamma import QLearingAgentOptimizedGamma
from QLearningAgentDynamicAlpha import QLearingAgentDynamicAlpha
from MonteCarloAgent import MCAgent
from MCAgentDynamicAlpha import MCAgentDynamicAlpha
from AgentBest import AgentBest
from test import QLearingAgentTest

################################
#agent = MCAgent()
#agent = QLearingAgent()
#agent = QLearingAgentOptimizedReward()
#agent = QLearingAgentOptimizedGamma()
#agent = QLearingAgentDynamicAlpha()
agent = LFA()
#agent = MCAgentDynamicAlpha()
#agent = AgentBest()
#agent = QLearingAgentTest()

printEveryIterations = 100
plotEveryNAverages = 20
################################
_scores = []

def plotAverage():
    addedScores = 0;
    averageScores = []
    splitEveryN = len(_scores) / plotEveryNAverages
    for idx, score in enumerate(_scores):
        addedScores += score
        if (idx + 1) % splitEveryN == 0:
            averageScores.append(addedScores / splitEveryN)
            addedScores = 0

    countEpisodes = range(splitEveryN, (len(averageScores) + 1) * splitEveryN, splitEveryN)

    plt.plot(countEpisodes, averageScores, 'o-', linewidth=2, label='Average training scores')
    plt.legend(loc='best')
    plt.xlabel("Episodes")
    plt.ylabel("Average score")
    plt.show()

def train_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """
    reward_values = agent.reward_values()
    game = FlappyBird()
    game.allowed_fps = None
    env = PLE(game, fps=30, display_screen=False, force_fps=True, rng=None, reward_values=reward_values)

    env.init()

    score = 0
    maxScore = 0
    numberOfFrames = 0
    realScore = 0

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
        numberOfFrames += 1

        if score > 99 and score <102:
            realScore += 1

        # reset the environment if the game is over
        if isGameOver:
            _scores.append(score)

            if nb_episodes % printEveryIterations == 0:
#                print("score for this episode: %d" % realScore)
                print("score for this episode: %d" % score)
                print("number of frame %d" % numberOfFrames)
                print("number of episodes left %d" % nb_episodes)
            env.reset_game()
            nb_episodes -= 1
            realScore = 0
            if(score > maxScore):
                maxScore = score
            score = 0
    print("best score: %d" % maxScore)

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
        score += reward
        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0

print "current agent : %s" % agent.__class__
while(True):
    choice = int(raw_input("1: Training \n2: Save Q \n3: Load Q \n4: Run "
                           "\n5: Plot Pi \n6: Plot Average \n0: Exit \n\nType in: "))
    if choice == 0:
        break
    if choice == 1:
        rounds = int(raw_input("Enter Trainrounds: "))
        train_game(rounds, agent)
    if choice == 2:
        name = raw_input("Enter Filename: ")
        np.save(name + 'Q.npy', dict(agent._q))
        np.save(name + 'S.npy', _scores)
    if choice == 3:
        name = raw_input("Enter Filename: ")
        # Load
        agent._q = np.load(name + 'Q.npy').item()
        _scores = np.load(name + 'S.npy').tolist()
    if choice == 4:
        rounds = int(raw_input("Enter Runrounds: "))
        run_game(rounds, agent)
    if choice == 5:
        agent.plotQ('pi')
    if choice == 6:
        plotAverage()