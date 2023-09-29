class MCTSAgentConfig:

    def __init__(self):
        self.pb_c_base = 1000
        self.pb_c_init = 1.25
        self.discount = 0.997
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.num_simulations = 100000
        self.reuse_tree = False
        self.temperature = 0.35
        self.do_roll_outs = False
        self.number_of_roll_outs = 1
        self.max_roll_out_depth = 20
        self.do_roll_out_steps_with_simulation_true = False


