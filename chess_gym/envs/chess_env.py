import gymnasium as gym
from gymnasium import spaces

import chess
import chess.svg
import chess.engine

import numpy as np
import random

from io import BytesIO
import cairosvg
from PIL import Image

from IPython.display import clear_output
import matplotlib.pyplot as plt


# =====================================================
# Utils
# =====================================================

STOCKFISH_PATH = "/usr/stockfish/stockfish-ubuntu-x86-64-avx2"

MAX_MOVES = 75

# For material evaluation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

#Get the best next move decided by the stockfish engine 
def stockfish_next_move(board, time_limit = 0.0001):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    engine.quit()
    return result.move

#Evaluate current state using the stockfish chess engine
def stockfish_evaluation(board, time_limit = 0.01):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    result = engine.analyse(board, chess.engine.Limit(time=time_limit))
    engine.quit()
    return result['score'].relative.score()

#Evaluate current state using current available material
def material_evaluation(board):
    white_score = sum(PIECE_VALUES.get(piece.piece_type, 0) for piece in board.piece_map().values() if piece.color)
    black_score = sum(PIECE_VALUES.get(piece.piece_type, 0) for piece in board.piece_map().values() if not piece.color)
    # Normalize to [-1, 1]
    return (white_score - black_score) / 39.0  # 39 = total material available

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
                if include_promotions:
                    promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
                    # Handle promotion possibilities
                    from_rank = chess.square_rank(from_square)
                    to_rank = chess.square_rank(to_square)
            
                    # White pawn promotion
                    if from_rank == 6 and to_rank == 7:
                        for promo in promotion_pieces:
                            all_moves.append(chess.Move(from_square, to_square, promotion=promo))
                    
                    # Black pawn promotion
                    if from_rank == 1 and to_rank == 0:
                        for promo in promotion_pieces:
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




# =====================================================
# Action Space
# =====================================================

#Gymnasium requires the action space to inherit spaces.Space class
class MoveSpace(gym.spaces.Discrete):
    def __init__(self, board):
        self.board = board
        super().__init__(n=ACTION_SPACE_SIZE)
        
    def sample(self):
        legal_moves = list(self.board.legal_moves)
        move = np.random.choice(legal_moves)
        
        return MOVE_TO_INDEX.get(move.uci())

# =====================================================
# Enviroment
# =====================================================

class ChessEnv(gym.Env):
    """Chess Environment"""
    metadata = {'render_modes': ['rgb_array', 'human', 'training'], 'observation_modes': ['rgb_array', 'piece_map']}

    def __init__(self, render_size=512, render_mode=None, observation_mode='rgb_array', claim_draw=True,  logging = False, render_steps = False, steps_per_render = 50, reward_type = "sparse" , use_eval = None, rival_agent = "engine", **kwargs):
        super(ChessEnv, self).__init__()
        self.render_steps = render_steps
        self.steps_per_render = steps_per_render
        self.step_counter = 0
        self.render_mode = render_mode
        self.logging = logging
        self.terminated_episodes = 0
        self.use_eval = use_eval
        self.reward_type = reward_type
        self.last_reward = 0
        self.rival_agent = rival_agent
        if observation_mode == 'rgb_array':
            self.observation_space = spaces.Box(low = 0, high = 255,
                                                shape = (render_size, render_size, 3),
                                                #dtype = np.uint8)
                                                dtype = np.float64)
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

        self.color = None

        self.viewer = None

        self.action_space = MoveSpace(self.board)


    # =====================================================
    # Observation utils
    # =====================================================
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
            
        return piece_map
        
    def _observe(self):
        observation = (self._get_image() if self.observation_mode == 'rgb_array' else self._get_piece_configuration())
        return observation


    # =====================================================
    # Action utils
    # =====================================================
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
    
    def get_action_mask(self):
        legal_actions = self._get_legal_moves_index()
        all_actions = set(range(ACTION_SPACE_SIZE))
        mask = np.array([move in legal_actions for move in all_actions], dtype=bool)
        return mask

    # =====================================================
    # Core Gymnasium functions
    # =====================================================
    
    def step(self, action):

        #if illegal action chosen, end the match as a loss with worse reward
        if action not in self._get_legal_moves_index():  
                #reward = ((-1)/self.step_counter) - 1
                reward = -1
                terminated = True
                truncated = False
                print("ILLEGAL MOVE")
                
        else:
                
                self.board.push(self._action_to_move(action))
                result = self.board.result()
                
                # is_game_over() checks for fifty-move rule or threefold repetition if claim_draw = true. Checking threefold repetition may be too slow
                terminated = self.board.is_game_over(claim_draw = self.claim_draw)
                truncated = self.step_counter > MAX_MOVES

                #reward = (1 if result == '1-0' else -1 if result == '0-1' else 0)
                if terminated:
                    #Positive reward if the agents wins, independently of being white or black
                    #reward = (1 if result == '1-0' else 1 if result == '0-1' else 0)
                    reward = (1 if result == '1-0' else -1 if result == '0-1' else 0)
                    print("REWARD - TERMINATED - ",reward)
                    
                elif truncated:
                    match self.use_eval:
                    
                        #Use material left for intermediate evaluation
                        case "material":
                            reward = material_evaluation(self.board)
                            # If agent moves as black, set opposite reward
                            if self.color == "BLACK":
                                reward = -reward
                    
                        #Use Stockfish engine for intermediate evaluation    
                        case "stockfish":
                            eval_cp = stockfish_evaluation(self.board)
                            
                            #Stockfish evaluation returns a NoneType if it sees a mate
                            if eval_cp is None:
                                reward = -1
                            else:
                                reward = np.clip(eval_cp / 1000.0, -0.9, 0.9)  # normalize centipawns given by the engine
                                '''
                                # If agent moves as black, set opposite reward
                                if self.color == "BLACK":
                                    reward = -reward
                                '''
                                reward = -reward
                        case _:
                            reward = 0
                    print("REWARD - TRUNCATED - ",reward)
                else:
                    if self.reward_type == "dense":
                        match self.use_eval:
                            case "material":
                                reward = material_evaluation(self.board)

                                # If agent moves as black, set opposite reward
                                if self.color == "BLACK":
                                    reward = -reward

                            #Use Stockfish engine for intermediate evaluation    
                            case "stockfish":
                                eval_cp = stockfish_evaluation(self.board)
                                
                                #Stockfish evaluation returns a NoneType if it sees a mate
                                if eval_cp is None:
                                    reward = -1
                                else:
                                    reward = np.clip(eval_cp / 1000.0, -0.9, 0.9)  # normalize centipawns given by the engine
                                    '''
                                    # If agent moves as black, set opposite reward
                                    if self.color == "BLACK":
                                        reward = -reward
                                    '''
                                    reward = -reward
                            case _:
                                reward = 0
                    else:
                        reward = 0
                    #print(self.color," - REWARD - INTERMEDIATE - ",reward)

        # Optional render every few steps, second move render
        if self.step_counter % self.steps_per_render == 0 and self.render_steps:
            self.render()
        
        if not terminated or truncated:
            #Make the engine play the next move of the opposite color
            match self.rival_agent:
                case "engine":
                    self.board.push(stockfish_next_move(self.board))
                case "random":
                    self.board.push(np.random.choice(list(self.board.legal_moves)))
                case "human":
                    print("Write your next move")
                    self.board.push(chess.Move.from_uci(input()))
            terminated = self.board.is_game_over(claim_draw = self.claim_draw)
            if terminated:
                reward = -1
                #print("REWARD - LOSE - ",reward)
        
        observation = self._observe()
        info = {'turn': self.board.turn,
                'castling_rights': self.board.castling_rights,
                'fullmove_number': self.board.fullmove_number,
                'halfmove_clock': self.board.halfmove_clock,
                'promoted': self.board.promoted,
                'chess960': self.board.chess960,
                'ep_square': self.board.ep_square}    
        self.step_counter += 1

        '''
        #Set rewards as the difference of evaluation so it telescopes, avoiding accomulation in dense rewards
        if not terminated and self.reward_type=="dense":
                reward_delta = (reward - self.last_reward)
                self.last_reward = reward 
                reward = reward_delta #* 0.05
        '''     

        # Optional render every few steps
        if self.step_counter % self.steps_per_render == 0 and self.render_steps:
            self.render()
        
        
        return observation, reward, terminated, truncated, info

    
    #Gymnasium requires handling the 'seed' and 'options' arguments 
    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)
        self.board.reset()
        self.step_counter = 0
        if self.chess960:
            self.board.set_chess960_pos(np.random.randint(0, 960))
            #self.board.set_chess960_pos(seed)
        
        #Chose if agent plays whites or blacks
        if random.choice([0,1]) ==0:
            if self.step_counter % self.steps_per_render == 0 and self.render_steps:
                self.render()
                #if blacks, engine makes the first move
                self.color = "BLACK"
                match self.rival_agent:
                    case "engine":
                        self.board.push(stockfish_next_move(self.board))
                    case "random":
                        self.board.push(np.random.choice(list(self.board.legal_moves)))
                    case "human":
                        print("Playing as not",self.color," Write your next move")
                        self.board.push(chess.Move.from_uci(input()))
        else:
            self.color = "WHITE"
        #print("RESET - PLAYING AS - ",self.color)
        
        self.last_reward = 0

        
        return self._observe(), {}

    
    # ==========================================
    # Rendering
    # ==========================================
    def render(self):
        img = self._get_image()

        if self.render_mode == 'training':
            return NaN
        elif self.render_mode == 'rgb_array':
            return img
        elif self.render_mode == 'human':
            plt.clf()
            plt.imshow(img)
            plt.show(block=False)
            plt.pause(0.001)
            ''' from gymnasium.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()

            self.viewer.imshow(img)
            return self.viewer.isopen
            '''
           
    def close(self):
        if not self.viewer is None:
            self.viewer.close()
