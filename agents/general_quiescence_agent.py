from typing import Union

import chess
import chess.engine

from agents.agent import ChessAgent
from agents.search_agents import ChessEvaluator
from utils.utils import State, score_to_float


class GeneralQuiescenceAgent(ChessAgent):
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
            quiescence_score, move = self.quiescence_max_value(
                state, self.quiescence_depth_limit, alpha, beta
            )
            return quiescence_score, move

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
            chess.engine.PovScore(bestScore, chess.WHITE),
            moves[chosenIndex],
        )

    def min_value(
        self, state: State, depth: int, alpha: float, beta: float
    ) -> tuple[chess.engine.PovScore, chess.Move]:
        # check for terminal state
        if state.board.is_game_over():
            return self.evaluator.getEvaluation(state), None
        if depth <= 0:
            quiescence_score, move = self.quiescence_min_value(
                state, self.quiescence_depth_limit, alpha, beta
            )
            return quiescence_score, move

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
                break
            beta = min(beta, score_to_float(min(scores), score.turn))
        bestScore: chess.engine.Score = min(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = bestIndices[0]

        return (
            chess.engine.PovScore(bestScore, chess.BLACK),
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

        # test null move
        null_move_score = self.evaluator.getEvaluation(state)
        if score_to_float(null_move_score.relative, chess.WHITE) >= beta:
            return chess.engine.PovScore(null_move_score.relative, chess.WHITE), None
        alpha = max(alpha, score_to_float(null_move_score.relative, chess.WHITE))

        # Collect legal moves and successor states
        legalMoves = state.board.generate_legal_moves()
        volatile_moves = []
        if state.board.is_check():
            volatile_moves = legalMoves
        else:
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
            scores.append(score.relative)
            if score_to_float(score.relative, score.turn) > beta:
                break
            alpha = max(alpha, score_to_float(max(scores), score.turn))
        bestScore: chess.engine.Score = max(scores)
        if bestScore < null_move_score.relative:
            return chess.engine.PovScore(null_move_score.relative, chess.WHITE), None
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = bestIndices[0]

        return (
            chess.engine.PovScore(bestScore, chess.WHITE),
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

        # test null move score
        null_move_score = self.evaluator.getEvaluation(state)
        if score_to_float(null_move_score.relative, chess.BLACK) <= alpha:
            return chess.engine.PovScore(null_move_score.relative, chess.BLACK), None
        beta = min(beta, score_to_float(null_move_score.relative, chess.WHITE))

        # Collect legal moves and successor states
        legalMoves = state.board.generate_legal_moves()
        volatile_moves = []
        if state.board.is_check():
            volatile_moves = legalMoves
        else:
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
            scores.append(score.relative)
            if score_to_float(score.relative, score.turn) < alpha:
                break
            beta = min(beta, score_to_float(min(scores), score.turn))
        bestScore: chess.engine.Score = min(scores)
        if bestScore > null_move_score.relative:
            return chess.engine.PovScore(null_move_score.relative, chess.BLACK), None
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = bestIndices[0]

        return (
            chess.engine.PovScore(bestScore, chess.BLACK),
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
