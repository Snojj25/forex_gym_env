from gym.envs.registration import register

register(
    id= "ForexGym-v0",
    entry_point= "envs.forex_env:ForexEnv")