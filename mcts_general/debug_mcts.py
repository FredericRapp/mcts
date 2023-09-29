import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load data
housing = fetch_california_housing()
X = housing.data
y = housing.target

X = MinMaxScaler().fit_transform(X)
# Define the desired number of data points
desired_num_samples = 400  # Adjust this as needed

# Randomly sample data points
# set the seed to ensure reproducibility
np.random.seed(42)
random_indices = np.random.choice(len(X), desired_num_samples, replace=False)
X_subset = X[random_indices]
y_subset = y[random_indices]
# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X_subset, y_subset, test_size=0.5, random_state=42)

from autoqfm_env import KernelPropertiyEnv
num_qubits = 8
env = KernelPropertiyEnv(training_data=X_train, training_labels=Y_train, num_qubits=num_qubits, num_features=8)
env.reset()

from autoqfm_agent import MCTSAgent
from autoqfm_config import MCTSAgentConfig
from autoqfm_game import DiscreteGymGame

config = MCTSAgentConfig()
agent = MCTSAgent(config=config)

game = DiscreteGymGame(env=env)
state = game.reset()

done = False
reward = 0

# run a trajectory
while not done:
    action = agent.step(game, state, reward, done)
    state, reward, done = game.step(action)
    print("Action: ", action, "Reward: ", reward, "Done: ", done)

game.close()