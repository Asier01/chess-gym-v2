import gymnasium as gym
from gymnasium import spaces

import chess
import chess.svg

import numpy as np

from io import BytesIO
import cairosvg
from PIL import Image

from IPython.display import clear_output
import matplotlib.pyplot as plt


#returns a list with all posible (legal and illegal) moves
def all_possible_moves(include_promotions=True, include_drops=False):
        all_moves = []
        squares = list(chess.SQUARES)
    
        for from_square in squares:
            for to_square in squares:
                if from_square == to_square:
                    continue
    
                # Add normal move
                all_moves.append(chess.Move(from_square, to_square))
    
                # Add promotion moves
                if include_promotions and chess.square_rank(to_square) in [0, 7]:
                    for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        all_moves.append(chess.Move(from_square, to_square, promotion=promo))
    
                # Add drops (for variants like Crazyhouse)
                if include_drops:
                    for piece_type in range(1, 7):
                        all_moves.append(chess.Move(None, to_square, drop=piece_type))
    
        return all_moves
        

ALL_POSSIBLE_MOVES = all_possible_moves()
MOVE_TO_INDEX = {m.uci(): i for i, m in enumerate(ALL_POSSIBLE_MOVES)}
#print(MOVE_TO_INDEX)
INDEX_TO_MOVE = {i: m for i, m in enumerate(ALL_POSSIBLE_MOVES)}
#print(INDEX_TO_MOVE)

ACTION_SPACE_SIZE = len(ALL_POSSIBLE_MOVES) 


#Gymnasium requires the action space to inherit spaces.Space class
class MoveSpace(gym.spaces.Discrete):
    def __init__(self, board):
        self.board = board
        super().__init__(n=ACTION_SPACE_SIZE)
        
    def sample(self):
        legal_moves = list(self.board.legal_moves)
        move = np.random.choice(legal_moves)
        
        return MOVE_TO_INDEX.get(move.uci())

class ChessEnv(gym.Env):
    """Chess Environment"""
    metadata = {'render_modes': ['rgb_array', 'human', 'training'], 'observation_modes': ['rgb_array', 'piece_map']}

    def __init__(self, render_size=512, render_mode=None, observation_mode='rgb_array', claim_draw=True, **kwargs):
        super(ChessEnv, self).__init__()
        self.step_counter = 0
        self.render_mode = render_mode

        if observation_mode == 'rgb_array':
            self.observation_space = spaces.Box(low = 0, high = 255,
                                                shape = (render_size, render_size, 3),
                                                dtype = np.uint8)
        elif observation_mode == 'piece_map':
            self.observation_space = spaces.flatten_space(spaces.Box(low = -6, high = 6,
                                                shape = (8, 8),
                                                dtype = np.int8))
            self.observation_space.n = self.observation_space.shape[0]
        else:
            raise Exception("observation_mode must be either rgb_array or piece_map")

        self.observation_mode = observation_mode

        # chess960 defines Fischer Random Chess, a chess variant that randomizes the starting position of the pieces on the back rank
        self.chess960 = kwargs['chess960']
        self.board = chess.Board(chess960 = self.chess960)

        if self.chess960:
            self.board.set_chess960_pos(np.random.randint(0, 960))

        self.render_size = render_size
        self.claim_draw = claim_draw

        self.viewer = None

        self.action_space = MoveSpace(self.board)

    def _get_image(self):
        out = BytesIO()
        bytestring = chess.svg.board(self.board, size = self.render_size).encode('utf-8')
        cairosvg.svg2png(bytestring = bytestring, write_to = out)
        image = Image.open(out)
        return np.asarray(image)

    def _get_piece_configuration(self):
        piece_map = np.zeros(64)

        for square, piece in zip(self.board.piece_map().keys(), self.board.piece_map().values()):
            piece_map[square] = piece.piece_type * (piece.color * 2 - 1)

        #return piece_map.reshape((8, 8))
        return piece_map
    def _observe(self):
        observation = (self._get_image() if self.observation_mode == 'rgb_array' else self._get_piece_configuration())
        return observation


    '''
    def _action_to_move(self, action):
        from_square = chess.Square(action[0])
        to_square = chess.Square(action[1])
        promotion = (None if action[2] == 0 else chess.Piece(chess.PieceType(action[2])), chess.Color(action[4]))
        drop = (None if action[3] == 0 else chess.Piece(chess.PieceType(action[3])), chess.Color(action[5]))
        move = chess.Move(from_square, to_square, promotion, drop)
        return move
    '''
    def _action_to_move(self, action):
        #Convert an integer action index to a python-chess Move.
        return INDEX_TO_MOVE[action]
        
    '''
    def _move_to_action(self, move):
        from_square = move.from_square
        to_square = move.to_square
        promotion = (0 if move.promotion is None else move.promotion)
        drop = (0 if move.drop is None else move.drop)
        return [from_square, to_square, promotion, drop]
    '''    
    def _move_to_action(self, move):
        #Convert a python-chess Move to an integer action.
        return MOVE_TO_INDEX.get(move.uci())

    def _get_legal_moves_index(self):
        legalMoveIndexList = []
        for move in MOVE_TO_INDEX:
                move = chess.Move.from_uci(move)
                if move in list(self.board.legal_moves):
                    legalMoveIndexList.append(MOVE_TO_INDEX.get(move.uci()))
        return legalMoveIndexList
                
    def step(self, action):

        print(self._action_to_move(action))
        self.step_counter += 1
        #if illegal action chosen, end the match as a loss
        if action not in self._get_legal_moves_index():     
                #Set the reward proportionally with the amount of legal moves done, for a max of -1
                reward = ((-1)/self.step_counter) - 1
                terminated = True
                truncated = False
           
        else:
                self.board.push(self._action_to_move(action))
                result = self.board.result()
                reward = (1 if result == '1-0' else -1 if result == '0-1' else 0)
                
                # is_game_over() checks for fifty-move rule or threefold repetition if claim_draw = true. Checking threefold repetition may be too slow
                terminated = self.board.is_game_over(claim_draw = self.claim_draw)
                truncated = terminated
        
        observation = self._observe()
        info = {'turn': self.board.turn,
                'castling_rights': self.board.castling_rights,
                'fullmove_number': self.board.fullmove_number,
                'halfmove_clock': self.board.halfmove_clock,
                'promoted': self.board.promoted,
                'chess960': self.board.chess960,
                'ep_square': self.board.ep_square}
                        
        return observation, reward, terminated, truncated, info

    #Gymnasium requires handling the 'seed' and 'options' arguments 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        self.step_counter = 0
        if self.chess960:
            self.board.set_chess960_pos(np.random.randint(0, 960))
            #self.board.set_chess960_pos(seed)

        return self._observe(), {}

    def render(self):
        img = self._get_image()

        if self.render_mode == 'training':
            return NaN
        elif self.render_mode == 'rgb_array':
            return img
        elif self.render_mode == 'human':
            plt.imshow(img)
            plt.show()
            
            ''' from gymnasium.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()

            self.viewer.imshow(img)
            return self.viewer.isopen
            '''
           
    def close(self):
        if not self.viewer is None:
            self.viewer.close()
