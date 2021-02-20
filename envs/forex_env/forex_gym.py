import gym
import pandas as pd
import numpy as np
from collections import deque

class ForexEnv(gym.Env):

    def __init__(self, start_idx=0):
        self.start_idx = start_idx
        self.action_space = [0, 1, 2]  # [pass, buy, sell]
        self.df = None
        self.state_iter = None
        self.states = deque(maxlen = 60)
        self.state_idx = {'Open': 0, 'High':1, 'Low':2, 'Close':3, 'Volume':4, 'hour':5}
        self.holding = 0
        print("Initialized with starting index: {}!".format(self.start_idx))



    def step(self, action):
        assert action in [0,1,2], "Enter a valid action"      
        next_state = next(self.state_iter)[1]


        Close = next_state[self.state_idx["Close"]]
        Open = next_state[self.state_idx["Open"]]
        #Pass and not holding
        if action == 0 and self.holding == 0:
            reward = 0
            holding = 0
        # Pass and holding long 
        elif action == 0 and self.holding == 1:
            reward = (Close - Open)
            holding = 1
        # Pass and holidng short
        elif action == 0 and self.holding == -1:
            reward = (Open - Close)
            holding = -1
        # Buy and not holding
        elif action == 1 and self.holding == 0:
            reward = (Close - Open) - 0.08
            holding = 1
        # Buy and holding long
        elif action == 1 and self.holding == 1:
            reward = (Close - Open)
            holding = 1
        # Buy and holding short
        elif action == 1 and self.holding == -1:
            reward = 0
            holding = 0
        # Sell and not holding
        elif action == 2 and self.holding == 0:
            reward = (Open - Close) - 0.08
            holding = -1
        # Sell and holding long
        elif action == 2 and self.holding == 1:
            reward = 0
            holding = 0
        #sell and holding short
        elif action == 2 and self.holding == -1:
            reward = (Open - Close)
            holding = -1
        else:
            print("Something is wrong in the step() function!")
            return
        self.states.append(next_state)
        self.holding = holding
        done = False
        return self.states, reward, done


    def reset(self):
        self.df = load_data()
        self.state_iter = self.df.iterrows()
        for i in range(int(self.start_idx)):
            _ = next(self.state_iter)
        for i in range(60):
            state = next(self.state_iter)[1]
            self.states.append(state)
        self.holding = 0
        print("env reset")
        return self.states

    def render(self):
        print(self.states)

def load_data(path="C:/JURE_SNOJ/forex_gym/forex_gym_env/dataset/crypto/BTCUSD_M5.csv"):
    df = pd.read_csv(path, sep= ",", names = ["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df['200_EMA'] = df.iloc[:,1].ewm(span=200,adjust=True).mean()
    df['50_EMA'] = df.iloc[:,1].ewm(span=50,adjust=True).mean()
    df['9_EMA'] = df.iloc[:,1].ewm(span=9,adjust=True).mean()
    EMA26 = df.iloc[:,1].ewm(span=26,adjust=True).mean()
    EMA12 = df.iloc[:,1].ewm(span=12,adjust=True).mean()
    df["MACD-diff"] = EMA12 - EMA26
    df["MACD-signal"] = df["MACD-diff"].ewm(span=9,adjust=True).mean()
    L14 = df.loc[:,"low"].rolling(window=14).min()
    H14 = df.loc[:,"high"].rolling(window=14).max()
    C = df.loc[:, "close"]
    df["stochastic"] = 100* (C - L14)/(H14 - L14)
    df = df.iloc[50:, [1,4,5,6,7,8,9,10,11]]
    return df



