# Chess `Gym`

[![PyPI download month](https://img.shields.io/pypi/dm/chess-gym.svg)](https://pypi.python.org/pypi/chess-gym/)
[![PyPI - Status](https://img.shields.io/pypi/status/chess-gym)](https://pypi.python.org/pypi/chess-gym/)
[![PyPI](https://img.shields.io/pypi/v/chess-gym)](https://pypi.python.org/pypi/chess-gym/)
![GitHub](https://img.shields.io/github/license/Ryan-Rudes/chess-gym)


Gym Chess V2 is an environment for reinforcement learning updated to be compatible with the Farama gymnasium module, originaly made for the OpenAI Gym module by Ryan Rudes.

<a href="https://imgbb.com/"><img src="https://i.ibb.co/Fw4fhzK/Screen-Shot-2020-10-27-at-2-30-21-PM.png" alt="Screen-Shot-2020-10-27-at-2-30-21-PM" border="0"></a>

## Installation

1. Install [Gymnasium]([https://github.com/openai/gym](https://github.com/Farama-Foundation/Gymnasium)) and its dependencies. \
`pip install gymnasium`

2. Download and install `chess_gym`: \
`pip install git+https://github.com/Asier01/chess-gym-v2.git` \
 \

## Environments
<a href="https://ibb.co/dgLW9rH"><img src="https://i.ibb.co/NSmVhsG/Screen-Shot-2020-10-27-at-3-08-46-PM-copy.png" alt="Screen-Shot-2020-10-27-at-3-08-46-PM-copy" border="0"></a>

## Example
You can use the standard `Chess-v0` environment as so:
```python
import gymnasium as gym
import chess_gym

env = gym.make("Chess-v0")
env.reset()

terminal = False

while not terminal:
  action = env.action_space.sample()
  observation, reward, terminal, truncated, info = env.step(action)
  env.render()
  
env.close()
```

There is also an environment for the Chess960 variant; its identifier is `Chess960-v0`

## Further Info
This environment will return 0 reward until the game has reached a terminal state. In the case of a draw, it will still return 0 reward. Otherwise, the reward will be either 1 or -1, depending upon the winning player.
```python
observation, reward, terminal, truncated, info = env.step(action)
```
Here, `info` will be a dictionary containing the following information pertaining to the board configuration and game state:
* [`turn`](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.turn): The side to move (`chess.WHITE` or `chess.BLACK`).
* [`castling_rights`](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.castling_rights): Bitmask of the rooks with castling rights.
* [`fullmove_number`](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.fullmove_number): Counts move pairs. Starts at 1 and is incremented after every move of the black side.
* [`halfmove_clock`](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.halfmove_clock): The number of half-moves since the last capture or pawn move.
* [`promoted`](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.promoted): A bitmask of pieces that have been promoted.
* [`ep_square`](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.ep_square): The potential en passant square on the third or sixth rank or `None`.
