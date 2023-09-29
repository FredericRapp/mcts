import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from scipy.linalg import eigh
import scipy
from scipy.linalg import sqrtm
from qiskit import Aer

# lets try a QML approach based on a random feature map 
from squlearn.feature_map.layered_feature_map import LayeredFeatureMap
from squlearn.kernel.matrix import FidelityKernel, ProjectedQuantumKernel
from squlearn.util import Executor

max_num_gates = 10
max_steps = 12


actions = [         # action integer number
    "",             #0
    "H",            #1
    f"Rx(x;=(x),{{x}})",        #2
    f"Ry(x;=(x),{{x}})",        #3
    f"Rz(x;=(x),{{x}})",        #4
    f"Rx(p,x;=y*(x),{{y,x}})",        #5
    f"Ry(p,x;=y*(x),{{y,x}})",        #6
    f"Rz(p,x;=y*(x),{{y,x}})",        #7
    #"cz", #8
    #"cy",  #9
    'cx',           #10    
    f'Rx(p,x;=y*np.cos(x),{{y,x}})', #11 # changed arccos to cos due to no rescaling
    'X',            #12
    'Y',            #13
    'Z',            #14
    'S',            #15
    'T'             #16
   ]




class KernelPropertiyEnv(gym.Env):
    def __init__(self, training_data, training_labels, num_qubits, num_features):
        """
        Initialization function of the environment
        """

        super(KernelPropertiyEnv, self).__init__()
        # Define action and observation space (integer arrays)

        # Defines the possible actions
        self.action_space = spaces.Discrete(len(actions))

        # Observations feature map converted to integer string
        self.observation_space = spaces.Box(low=0, high=len(actions)-1,
                                            shape=(max_num_gates,),dtype=int)

        self.best_overall_pred_error = np.inf
        self.best_overall_fm = ""
        self.training_data  = training_data
        self.training_labels = training_labels
        # set a random seed
        self.seed = np.random.randint(0,1000)

        #self.seed = 42
        # Dictionary for string to integer labels
        self.action_dic={}
        i=0
        for a in actions:
            self.action_dic[a]=i
            i+=1

        # try to calculate the best possible classical kernel matrix in the init -> then it just has to be
        # evaluated once as long as the training data stays the same
        self._num_qubits = num_qubits
        self._num_features = num_features

        print('MCTS autoqfm')
        #print('legal autoqfm actions:', self.legal_actions())
    def reset(self):
        """
        Reset function of the environment
        Resets the feature map to inital state
        """

        self.fm_str = ""
        self.best_fm_str = "" 
        self.last_action = ""
        self.done = False
        self.steps_done = 0
        self.best_spec_quotient = 0.0
        self.last_pred_error = 0.0 
        self.best_pred_error = np.inf
        self.reward = 0.0
        self.observations = self.text_to_int(self.fm_str)

        return self.observations

    def step(self,action):
        """
        Function that performs an action and returns the reward and the resulting
        feature-map as observation space
        """
        self.steps_done += 1
        
        pred_error = np.inf
        var_kernel = 10.0
        punish_exp_conc = 0.0
        reward_pred_error = 0.0
        punish_x = 0.0 
        punish_action = 0.0
        reward_overall_best = 0.0


        # Add gates to the feature map
        self.fm_str = self.fm_str + actions[action] + "-"

        print('step in the AQGM envi with respective feature map:', self.steps_done, self.fm_str)
        
        if "(x)" not in self.fm_str:
            # Capture the case, that there is no X in the circuit
            punish_x = -5.0
            print("No X in the circuit!",self.fm_str)
        else:
            # Calculate the quantum kernel and validate its properties
            q_kernel_matrix = self.return_kernel_matrix(x_train=self.training_data)

            #geom_diff, g_tra = self.geometric_difference(self.classical_kernel_matrix, q_kernel_matrix)

            pred_error = self.prediction_error_bound(k=q_kernel_matrix, y=self.training_labels)
            pred_error = np.real_if_close(pred_error, tol=1e-7)

            # Calculate the kernel target alignment
            #kta = self.compute_kernel_target_alignment(q_kernel_matrix, labels=self.training_labels)
            #print('prediction error bound in a NORMAL step:', pred_error, self.steps_done)
            # Calculate the kernel variance -> indicator for exponential concentration
            var_kernel = self.matrix_variance_exclude_diagonal(q_kernel_matrix)

            if var_kernel < 1e-7:
                print("kernel variance is very small, exponential concentration!",self.fm_str,var_kernel)
                punish_exp_conc = -50.0
            else:
                # reward improvement of the prediction error bound
                
                if pred_error <= (self.best_pred_error-0.01) and pred_error <= (self.last_pred_error-0.01) and self.last_pred_error != 0.0:
                    print("better prediction error bound!",self.fm_str,pred_error, self.best_pred_error, self.steps_done)
                    self.best_fm_str = self.fm_str
                    self.best_pred_error = pred_error
                    # change here! more reward for improvements in prediction error than kta! 
                    reward_pred_error = 100.0

                    # Reward for surpassing overall best prediction error
                    if pred_error < self.best_overall_pred_error:
                        reward_overall_best = 500.0  # You can adjust the reward value
                        self.best_overall_pred_error = pred_error
                        self.best_overall_fm = self.fm_str
                        print('best overall prediction error and FM:', self.best_overall_pred_error, self.best_overall_fm)

                else: 
                    punish_action = -1.0
            
            self.last_pred_error = pred_error

             
        # probably the split of the reward in that way makes no sense because both of them dont happen at the same time anyway
        self.reward =  punish_exp_conc + reward_pred_error + punish_x + reward_overall_best + punish_action #+ reward_improved_kta

        self.last_action = action
        # Create observation integer array
        self.observations = self.text_to_int(self.fm_str)

        # If too many steps are done, finish this environment
        if self.steps_done >= max_steps:
            self.done = True

        # If the feature map is to long, finish this environment
        if np.count_nonzero(self.observations) >= max_num_gates:
            self.done = True
        info = {}
        # Return featuremap as observation, current reward for action, done (=true if environment is done)
        return self.observations,self.reward,self.done,info

    def return_kernel_matrix(self, x_train):
        """
        Create Kernel Ridge class from feature map
        """
        lfm = LayeredFeatureMap.from_string(self.fm_str,num_qubits=self._num_qubits,num_features=self._num_features)
        np.random.seed(42)
        param_ini = np.random.uniform(-1,1,lfm.num_parameters)

        #kernel_matrix = ProjectedQuantumKernel(lfm,Executor("statevector_simulator"),initial_parameters=initial_parameters,gamma=0.25)
        quantum_kernel = ProjectedQuantumKernel(lfm,Executor("statevector_simulator"),initial_parameters=param_ini)
        
        quantum_kernel.assign_parameters(param_ini)
        # Do Kernel Ridge Regression
        
        kernel_matrix = quantum_kernel.evaluate(x_train)
        return kernel_matrix


    def matrix_variance_exclude_diagonal(self,matrix):
        # Exclude diagonal elements
        flattened_matrix = matrix.flatten()
        diagonal_indices = np.arange(0, len(flattened_matrix), matrix.shape[1] + 1)
        flattened_matrix_without_diagonal = np.delete(flattened_matrix, diagonal_indices)
        
        variance = np.var(flattened_matrix_without_diagonal)
        return variance
    
    
    def prediction_error_bound(
        self,
        k: np.ndarray,
        y: np.ndarray,
        lambda_reg: float = 0.001,
        ) -> float:
        """
        Function for calculating the prediction error s_K(N) bound for a given kernel matrix K and labels y

        s_k(N) = sum_ij [sqrt(K)*(K+lambda_reg*I)_ij)^-2*sqrt(K)]_ij * y_i * y_j

        See supplementary material of DOI:10.1038/s41467-021-22539-9 for details

        Args:
            k (np.ndarray): Kernel matrix
            y (np.ndarray): Labels
            lambda_reg (float): Regularization parameter lambda_reg

        """
        
        try:
            sqrt_k = sqrtm(k)
        except Exception as e:
            print("Error: Failed to find square root of matrix k:", e)
            return 1e6
        
        k_shift_inv = np.linalg.inv(k+np.eye(k.shape[0])*lambda_reg)
        k_shift_inv_squared = np.matmul(k_shift_inv,k_shift_inv)
        kk = np.matmul(sqrt_k,np.matmul(k_shift_inv_squared,sqrt_k))
        return np.real_if_close(np.dot(y,np.dot(kk,y)))

    def text_to_int(self,text:str):
        """
        Helper function for generating the observation integer array from the feature_map string
        """
        text_array = text.split('-')
        text_array.pop()
        observation = np.zeros(max_num_gates,dtype=int)

        i = 0
        for t in text_array:
            if t != "":
                observation[i] = self.action_dic[t]
                i=i+1
        return observation
    
    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(len(actions)))  