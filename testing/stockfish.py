import asyncio

import chess
import chess.engine

import paths


async def main() -> None:
    transport, engine = await chess.engine.popen_uci(paths.STOCKFISH_PATH)

    board = chess.Board()
    info = await engine.analyse(board, chess.engine.Limit(time=0.1))
    print(info["score"])
    # Score: PovScore(Cp(+20), WHITE)

    board = chess.Board(
        "r1bqkbnr/p1pp1ppp/1pn5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 4"
    )
    info = await engine.analyse(board, chess.engine.Limit(depth=20))
    print(info["score"])
    # Score: PovScore(Mate(+1), WHITE)

    await engine.quit()


asyncio.run(main())
