import numpy as np
import random
import matplotlib.pyplot as plt

class Auction:
    def __init__(self, n_bidders, value, fee, alpha, gamma, epsilon):
        self.n_bidders = n_bidders
        self.value = value
        self.fee = fee
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.q_table = np.zeros((n_bidders, value))
        self.bids = []
        self.rewardtable = []

    def run_auction(self):
        for i in range(0,self.n_bidders):        
            if np.random.rand() < self.epsilon:
                randomindex = np.random.randint(0,self.value)
                print("Agent " + str(i) + " has bid " + str(randomindex) + " randomly")
                self.bids.append(randomindex)
            else:
                self.bids.append(np.argmax(self.q_table[i]))
                print("Agent " + str(i) + " has bid " + str(np.argmax(self.q_table[i])) + " based on its Q-value")
        self.updateQtable(self.evaluateauction())

    def evaluateauction(self):
        winningbidder = np.argmax(self.bids)
        print("Winning bidder is: " + str(winningbidder))
        if isinstance(winningbidder, list):
            winningbidder = random.choice(winningbidder)
            print("Winning bidder is (chosen randomly): " + str(winningbidder))
        else:
            pass
        return(winningbidder)
    
    def updateQtable(self, winningbidder):
        for n in range(0,len(self.q_table)):
            if n == winningbidder:
                print("Winning bidder is bidder: " + str(n))
                reward = self.value - self.fee - self.bids[n]
                OldQValue = self.q_table[n,self.bids[n]]
                NewQValue = OldQValue + self.alpha * (reward + self.gamma * np.max(self.q_table[n]) - OldQValue)
                self.q_table[n,self.bids[n]] = NewQValue
                print("Bidder " + str(n) + " Q Value update from " + str(OldQValue) + " to " + str(NewQValue))
                self.rewardtable.append(reward)
            else:
                print("Losing bidder " + str(n))
                reward = -self.fee
                OldQValue = self.q_table[n,self.bids[n]]
                NewQValue = OldQValue + self.alpha * (reward + self.gamma * np.max(self.q_table[n]) - OldQValue)
                self.q_table[n,self.bids[n]] = NewQValue
                print("Bidder " + str(n) + " Q Value update from " + str(OldQValue) + " to " + str(NewQValue))
        self.bids = []
        return
n_bidders = 5
value = 5
fee = 1
alpha = 0.1
gamma = 0.99
epsilon = 0.1
numinstances = 100

A = Auction(n_bidders, value, fee, alpha, gamma, epsilon)
for k in range(0,numinstances):
    A.run_auction()
B = plt.plot(range(0,numinstances),A.rewardtable, ls = '-')

B.show()