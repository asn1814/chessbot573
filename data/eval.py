import agents.agent as agent
from data.dataset import get_splits
import paths
from utils.utils import State
import fire
from tqdm import tqdm
import asyncio

def eval(agent: agent.ChessAgent, use_test = False) -> float:
    print("Getting splits")
    train, val, test = get_splits(paths.TACTICS_DATA_ALL)
    eval = val
    if use_test:
        eval = test
    print("Done")

    correct = 0
    total = 0
    for data in tqdm(eval, "Evaluating"):
        move = asyncio.run(agent.getMove(State(data.fen)))
        if move is None:
            uci = "None"
        else:
            uci = move.uci()
        if data.best_move == uci:
            correct += 1
        total += 1

    print(f"Accuracy: {1.0 * correct / total}\nCorrect: {correct}\t Total: {total}")


def run_eval():
    model = agent.StockFishAgent()
    eval(model)

if __name__ == "__main__":
    fire.Fire(run_eval)
