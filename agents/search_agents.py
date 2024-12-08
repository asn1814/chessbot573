from abc import abstractmethod
from typing import Union

import chess
import chess.engine

import constants
from agents.agent import ChessAgent
from utils.utils import State


class ChessEvaluator:

    def __init__(self):
        super().__init__()

    @abstractmethod
    def getEvaluation(self, state: State) -> chess.engine.PovScore:
        raise NotImplementedError

    @abstractmethod
    def quit(self) -> None:
        raise NotImplementedError


class StockfishEvaluator(ChessEvaluator):
    def __init__(self, limit: chess.engine.Limit):
        super().__init__()
        self.engine = chess.engine.SimpleEngine.popen_uci(constants.STOCKFISH_PATH)
        self.limit = limit

    def getEvaluation(self, state: State) -> chess.engine.PovScore:
        score = self.engine.analyse(state.board, self.limit)["score"]
        return score

    def quit(self) -> None:
        self.engine.quit()


class MinimaxAgent(ChessAgent):
    def __init__(
        self,
        evaluator: ChessEvaluator,
        move_time_limit: float = 0.1,
        move_depth_limit: int = 20,
    ):
        super().__init__(move_time_limit, move_depth_limit)
        self.evaluator = evaluator

    def max_value(
        self, state: State, depth: int
    ) -> tuple[chess.engine.PovScore, chess.Move]:
        # check for terminal state
        if state.board.is_game_over() or depth <= 0:
            return self.evaluator.getEvaluation(state), None

        # update depth
        depth -= 1

        # Collect legal moves and successor states
        legalMoves = state.board.generate_legal_moves()

        # Choose one of the best actions
        scores = []
        moves = []
        for action in legalMoves:
            moves.append(action)
            state.board.push(action)
            score, move = self.min_value(state, depth)
            state.board.pop()
            scores.append(score.relative)
        bestScore: chess.engine.Score = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = bestIndices[0]

        return chess.engine.PovScore(bestScore, chess.WHITE), moves[chosenIndex]

    def min_value(self, state: State, depth: int):
        # check for terminal state
        if state.board.is_game_over() or depth <= 0:
            return self.evaluator.getEvaluation(state), None

        # update depth
        depth -= 1

        # Collect legal moves and successor states
        legalMoves = state.board.generate_legal_moves()

        # Choose one of the best actions
        scores = []
        moves = []
        for action in legalMoves:
            moves.append(action)
            state.board.push(action)
            score, move = self.max_value(state, depth)
            state.board.pop()
            scores.append(score.relative)
        bestScore: chess.engine.Score = min(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = bestIndices[0]

        return chess.engine.PovScore(bestScore, chess.BLACK), moves[chosenIndex]

    def getMove(self, state) -> Union[chess.Move, None]:
        if state.board.turn is chess.WHITE:
            centipawns, move = self.max_value(state, self.limit.depth)
        else:
            centipawns, move = self.min_value(state, self.limit.depth)
        return move

    def quit(self) -> None:
        self.evaluator.quit()
