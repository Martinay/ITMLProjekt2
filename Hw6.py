episodes = [('a', 'up'), ('b', 'up'), ('c', 'up'), ('d', 'right'), ('e', 'right'), ('f', 'down')]

q = {}
a = 0.4
gamma = 0.9

def getReward(state):
    if state == episodes[-1]:
        return 10
    return 0

for step in episodes:
    q[step] = 0

for x in range(0, 10):
    for i, step in enumerate(episodes):
        currentQ = q[step]
        r = getReward(step)
        nextQ = 0
        if i + 1 != len(episodes):
            nextQ = q[episodes[i + 1]]
        newQ = currentQ + a *(r + gamma * nextQ-currentQ)
        q[step] = newQ
        print('Episode: {} (state, action): {} new Q value: {}'.format(x,step, newQ))