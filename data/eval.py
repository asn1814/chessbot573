import chess.engine
import fire
from tqdm import tqdm

import agents.agent as agent
import agents.search_agents as search_agents
import constants
from agents.dp_g_q_agent import DPGeneralQuiescenceAgent
from agents.general_quiescence_agent import GeneralQuiescenceAgent
from data.dataset import get_splits
from utils.utils import State


def eval(agent: agent.ChessAgent, use_test=False) -> float:
    print("Getting splits")
    train, val, test = get_splits(constants.TACTICS_DATA_ALL)
    eval = val
    if use_test:
        eval = test
    print("Done")

    correct = 0
    total = 0
    for data in tqdm(eval, "Evaluating"):
        move = agent.getMove(State(data.fen))
        if move is None:
            uci = "None"
        else:
            uci = move.uci()
        if data.best_move == uci:
            correct += 1
        total += 1

    print(f"Accuracy: {1.0 * correct / total}\nCorrect: {correct}\t Total: {total}")


def run_eval():
    # model = agent.StockfishAgent(move_depth_limit=25)
    """model = search_agents.MinimaxAgent(
        search_agents.SimpleEvaluator(), move_depth_limit=1
    )"""
    """model = search_agents.AlphaBetaAgent(
        evaluator=search_agents.StockfishEvaluator(
            limit=chess.engine.Limit(time=0.01, depth=2)
        ),
        move_depth_limit=2,
    )"""
    """model = search_agents.AlphaBetaAgent(
        evaluator=search_agents.SimpleEvaluator(),
        move_depth_limit=3,
    )"""
    """model = search_agents.BruteQuiescenceAgent(
        evaluator=search_agents.SimpleEvaluator(),
        move_depth_limit=2,
        quiescence_depth_limit=3,
    )"""
    model = GeneralQuiescenceAgent(
        evaluator=search_agents.SimpleEvaluator(),
        move_depth_limit=1,
        quiescence_depth_limit=7,
    )
    """model = DPGeneralQuiescenceAgent(
        evaluator=search_agents.SimpleEvaluator(),
        move_depth_limit=2,
        quiescence_depth_limit=3,
    )"""
    eval(model)
    model.quit()


if __name__ == "__main__":
    fire.Fire(run_eval)
