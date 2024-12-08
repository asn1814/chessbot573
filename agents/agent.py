from utils.utils import State
from abc import ABC, abstractmethod
import chess
import chess.engine
import paths
import asyncio
from typing import Union

class ChessAgent(ABC):
  """Base class for chessplaying agents"""
  def __init__(self, move_time_limit: float = 0.1, move_depth_limit: int = 20):
    super().__init__()
    self.limit = chess.engine.Limit(time=move_time_limit, depth=move_depth_limit)

  @abstractmethod
  def getMove(self, state: State) -> Union[chess.Move, None]:
    raise NotImplementedError
  
class StockFishAgent(ChessAgent):
  def __init__(self):
    super().__init__()
    self.transport, self.engine = asyncio.run(self.__ainit__())
  
  async def __ainit__(self):
    return await chess.engine.popen_uci(paths.STOCKFISH_PATH)

  async def getMove(self, state) -> Union[chess.Move, None]:
    self.transport, self.engine = await chess.engine.popen_uci(paths.STOCKFISH_PATH)
    result = await self.engine.play(state.board, self.limit)
    return result.move
    