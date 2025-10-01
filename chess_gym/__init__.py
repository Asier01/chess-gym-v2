from gymnasium.envs.registration import register

register(
    id='Chess-v0',
    entry_point='chess_gym.envs:ChessEnv',
    kwargs={'chess960': False},
    max_episode_steps=1000
)

register(
    id='Chess960-v0',
    entry_point='chess_gym.envs:ChessEnv',
    kwargs={'chess960': True},
    max_episode_steps=1000
)
