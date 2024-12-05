from utils.utils import State
from abc import ABC, abstractmethod
import asyncio
import chess
import chess.engine
import paths

class ChessAgent(ABC):
  """Base class for chessplaying agents"""
  def __init__(self, move_time_limit: float = 0.1, move_depth_limit: int = 10):
    super().__init__()
    self.time_limit = move_time_limit
    self.depth_limit = move_depth_limit

  @abstractmethod
  def getMove(self, state: State) -> chess.Move:
    raise NotImplementedError
  
class StockFishAgent(ChessAgent):
  async def __init__(self):
    super().__init__()
    self.transport, self.engine = await chess.engine.popen_uci(paths.STOCKFISH_PATH)

  async def getMove(self, state) -> chess.Move:
    result = await self.engine.play(state.board, chess.engine.Limit(time=0.1))
    return result.move
    