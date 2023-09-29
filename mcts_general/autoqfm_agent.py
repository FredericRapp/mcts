from autoqfm_config import MCTSAgentConfig
from autoqfm_game import DeepCopyableGame
from mcts_autqfm import MCTS, select_action


class MCTSAgent:

    def __init__(self, config: MCTSAgentConfig):
        self.mcts = MCTS(config)
        self.result_node = None

    @property
    def config(self):
        return self.mcts.config

    def step(self, game_state: DeepCopyableGame, observations, reward, done, output_debug_info=False):
        self.result_node, info = self.mcts.run(
            observation=observations,
            reward=reward,
            game=game_state,
            add_exploration_noise=False,
            override_root_with=self.result_node if self.config.reuse_tree else None,
            done=done
        )
        action = select_action(self.result_node, temperature=self.config.temperature)
        if output_debug_info:
            return action, info
        else:
            return action

