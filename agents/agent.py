from abc import abstractmethod
from typing import Union

import chess
import chess.engine

import constants
from utils.utils import State


class ChessAgent:
    """Base class for chessplaying agents"""

    def __init__(self, move_time_limit: float = 0.1, move_depth_limit: int = 20):
        super().__init__()
        self.limit = chess.engine.Limit(time=move_time_limit, depth=move_depth_limit)

    @abstractmethod
    def getMove(self, state: State) -> Union[chess.Move, None]:
        """Gets a move

        Args:
            state (State): The board state to get a move for

        Raises:
            NotImplementedError: Not implemented

        Returns:
            Union[chess.Move, None]: A move, or None
        """
        raise NotImplementedError

    @abstractmethod
    def quit(self) -> None:
        """Closes the ChessAgent's processes

        Raises:
            NotImplementedError: Not implemented
        """
        raise NotImplementedError


class StockfishAgent(ChessAgent):
    def __init__(self):
        super().__init__()
        self.engine = chess.engine.SimpleEngine.popen_uci(constants.STOCKFISH_PATH)

    def getMove(self, state) -> Union[chess.Move, None]:
        result = self.engine.play(state.board, self.limit)
        return result.move

    def quit(self) -> None:
        self.engine.quit()
