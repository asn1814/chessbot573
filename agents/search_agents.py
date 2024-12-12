from abc import abstractmethod
from typing import Union

import chess
import chess.engine

import constants
from agents.agent import ChessAgent
from utils.utils import State, score_to_float


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


class SimpleEvaluator(ChessEvaluator):
    VALUE_PAWN = 100
    VALUE_KNIGHT = 310
    VALUE_BISHOP = 320
    VALUE_ROOK = 500
    VALUE_QUEEN = 900

    def __init__(self):
        super().__init__()

    def getEvaluation(self, state: State):
        if state.board.is_checkmate():
            return chess.engine.PovScore(
                relative=chess.engine.MateGiven, turn=state.board.turn
            )

        fen: str = state.board.fen().split()[0]
        if state.board.turn == chess.WHITE:
            centipawns = 50.0
        else:
            centipawns = -50.0
        centipawns += self.VALUE_PAWN * fen.count("P")
        centipawns += self.VALUE_KNIGHT * fen.count("N")
        centipawns += self.VALUE_BISHOP * fen.count("B")
        centipawns += self.VALUE_ROOK * fen.count("R")
        centipawns += self.VALUE_QUEEN * fen.count("Q")

        centipawns -= self.VALUE_PAWN * fen.count("p")
        centipawns -= self.VALUE_KNIGHT * fen.count("n")
        centipawns -= self.VALUE_BISHOP * fen.count("b")
        centipawns -= self.VALUE_ROOK * fen.count("r")
        centipawns -= self.VALUE_QUEEN * fen.count("q")

        return chess.engine.PovScore(
            relative=chess.engine.Cp(centipawns), turn=state.board.turn
        )

    def quit(self):
        return None


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


class AlphaBetaAgent(ChessAgent):
    def __init__(
        self,
        evaluator: ChessEvaluator,
        move_time_limit: float = 0.1,
        move_depth_limit: int = 20,
    ):
        super().__init__(move_time_limit, move_depth_limit)
        self.evaluator = evaluator

    def max_value(
        self, state: State, depth: int, alpha: float, beta: float
    ) -> tuple[chess.engine.PovScore, chess.Move]:
        # check for terminal state
        if state.board.is_game_over() or depth <= 0:
            return self.evaluator.getEvaluation(state), None

        # update depth
        depth -= 1

        # Collect legal moves and successor states
        legalMoves = state.board.generate_legal_moves()

        # Choose one of the best actions
        scores: list[float] = []
        moves = []
        for action in legalMoves:
            moves.append(action)
            state.board.push(action)
            score, move = self.min_value(state, depth, alpha, beta)
            state.board.pop()
            scores.append(score.relative)
            if score_to_float(score.relative, score.turn) > beta:
                break
            alpha = max(alpha, score_to_float(max(scores), score.turn))
        bestScore: chess.engine.Score = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = bestIndices[0]

        return (
            chess.engine.PovScore(chess.engine.Cp(bestScore), chess.WHITE),
            moves[chosenIndex],
        )

    def min_value(
        self, state: State, depth: int, alpha: float, beta: float
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
            score, move = self.max_value(state, depth, alpha, beta)
            state.board.pop()
            scores.append(score.relative)
            if score_to_float(score.relative, score.turn) < alpha:
                pass
            beta = min(beta, score_to_float(min(scores), score.turn))
        bestScore: float = min(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = bestIndices[0]

        return (
            chess.engine.PovScore(chess.engine.Cp(bestScore), chess.BLACK),
            moves[chosenIndex],
        )

    def getMove(self, state) -> Union[chess.Move, None]:
        if state.board.turn is chess.WHITE:
            centipawns, move = self.max_value(
                state, self.limit.depth, float("-inf"), float("inf")
            )
        else:
            centipawns, move = self.min_value(
                state, self.limit.depth, float("-inf"), float("inf")
            )
        return move

    def quit(self) -> None:
        self.evaluator.quit()


class QuiescenceAgent(ChessAgent):
    def __init__(
        self,
        evaluator: ChessEvaluator,
        move_time_limit: float = 0.1,
        move_depth_limit: int = 2,
        quiescence_depth_limit: int = 10,
    ):
        super().__init__(move_time_limit, move_depth_limit)
        self.evaluator = evaluator
        self.quiescence_depth_limit = quiescence_depth_limit

    def max_value(
        self, state: State, depth: int, alpha: float, beta: float
    ) -> tuple[chess.engine.PovScore, chess.Move]:
        # check for terminal state
        if state.board.is_game_over():
            return self.evaluator.getEvaluation(state), None
        if depth <= 0:
            quiescence_score, _ = self.quiescence_max_value(
                state, self.quiescence_depth_limit, float("-inf"), float("inf")
            )
            curr_score = self.evaluator.getEvaluation(state)
            if quiescence_score.relative <= curr_score.relative:
                return curr_score, None
            else:
                return quiescence_score, None

        # update depth
        depth -= 1

        # Collect legal moves and successor states
        legalMoves = state.board.generate_legal_moves()

        # Choose one of the best actions
        scores: list[float] = []
        moves = []
        for action in legalMoves:
            moves.append(action)
            state.board.push(action)
            score, move = self.min_value(state, depth, alpha, beta)
            state.board.pop()
            scores.append(score_to_float(score))
            if score_to_float(score) > beta:
                break
            alpha = max(alpha, max(scores))
        bestScore: chess.engine.Score = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = bestIndices[0]

        return (
            chess.engine.PovScore(chess.engine.Cp(bestScore), chess.WHITE),
            moves[chosenIndex],
        )

    def min_value(
        self, state: State, depth: int, alpha: float, beta: float
    ) -> tuple[chess.engine.PovScore, chess.Move]:
        # check for terminal state
        if state.board.is_game_over():
            return self.evaluator.getEvaluation(state), None
        if depth <= 0:
            quiescence_score, _ = self.quiescence_min_value(
                state, self.quiescence_depth_limit, float("-inf"), float("inf")
            )
            curr_score = self.evaluator.getEvaluation(state)
            if quiescence_score.relative >= curr_score.relative:
                return curr_score, None
            else:
                return quiescence_score, None

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
            score, move = self.max_value(state, depth, alpha, beta)
            state.board.pop()
            scores.append(score_to_float(score))
            if score_to_float(score) < alpha:
                break
            beta = min(beta, min(scores))
        bestScore: chess.engine.Score = min(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = bestIndices[0]

        return (
            chess.engine.PovScore(chess.engine.Cp(bestScore), chess.BLACK),
            moves[chosenIndex],
        )

    def quiescence_max_value(
        self, state: State, depth: int, alpha: float, beta: float
    ) -> tuple[chess.engine.PovScore, chess.Move]:
        # check for terminal state
        if state.board.is_game_over() or depth <= 0:
            return self.evaluator.getEvaluation(state), None

        # update depth
        depth -= 1

        # Collect legal moves and successor states
        legalMoves = state.board.generate_legal_moves()
        volatile_moves = []
        for move in legalMoves:
            if state.board.gives_check(move) or state.board.is_capture(move):
                volatile_moves.append(move)
        if len(volatile_moves) == 0:
            return self.evaluator.getEvaluation(state), None

        # Choose one of the best actions
        scores: list[float] = []
        moves = []
        for action in volatile_moves:
            moves.append(action)
            state.board.push(action)
            score, move = self.quiescence_min_value(state, depth, alpha, beta)
            state.board.pop()
            scores.append(score_to_float(score))
            if score_to_float(score) > beta:
                break
            alpha = max(alpha, max(scores))
        bestScore: chess.engine.Score = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = bestIndices[0]

        return (
            chess.engine.PovScore(chess.engine.Cp(bestScore), chess.WHITE),
            moves[chosenIndex],
        )

    def quiescence_min_value(
        self, state: State, depth: int, alpha: float, beta: float
    ) -> tuple[chess.engine.PovScore, chess.Move]:
        # check for terminal state
        if state.board.is_game_over() or depth <= 0:
            return self.evaluator.getEvaluation(state), None

        # update depth
        depth -= 1

        # Collect legal moves and successor states
        legalMoves = state.board.generate_legal_moves()
        volatile_moves = []
        for move in legalMoves:
            if state.board.gives_check(move) or state.board.is_capture(move):
                volatile_moves.append(move)
        if len(volatile_moves) == 0:
            return self.evaluator.getEvaluation(state), None

        # Choose one of the best actions
        scores = []
        moves = []
        for action in volatile_moves:
            moves.append(action)
            state.board.push(action)
            score, move = self.quiescence_max_value(state, depth, alpha, beta)
            state.board.pop()
            scores.append(score_to_float(score))
            if score_to_float(score) < alpha:
                break
            beta = min(beta, min(scores))
        bestScore: chess.engine.Score = min(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = bestIndices[0]

        return (
            chess.engine.PovScore(chess.engine.Cp(bestScore), chess.BLACK),
            moves[chosenIndex],
        )

    def getMove(self, state) -> Union[chess.Move, None]:
        if state.board.turn is chess.WHITE:
            centipawns, move = self.max_value(
                state, self.limit.depth, float("-inf"), float("inf")
            )
        else:
            centipawns, move = self.min_value(
                state, self.limit.depth, float("-inf"), float("inf")
            )
        return move

    def quit(self) -> None:
        self.evaluator.quit()
