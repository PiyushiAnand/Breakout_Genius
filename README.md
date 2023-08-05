The code implements DQN to train an agent to play Breakout Game.The code is implemted in Python language using Pytorch,numpy etc.Here we are using Convulolutional Neural Networks since the inputs are images and the agents learns from them.The code also includes testing of the agent once the training finshes.
The complete breakdown of the code is given as follows:

Environment Setup: The make_env function is defined to create the Gym environment for the Atari game. It applies various wrappers such as AtariPreprocessing, FrameStack, and TransformReward to preprocess the environment. The function returns the configured environment object.

DQNAgent Class: This class is derived from the nn.Module class in PyTorch and represents the neural network model for the DQN agent. The __init__ method initializes the layers of the network, including convolutional layers and fully connected layers. The forward method performs the forward pass of the network to compute Q-values. The get_qvalues method returns the Q-values as a numpy array. The sample_actions method selects actions based on Q-values and exploration probability.

Evaluation Function: The evaluate function is used to evaluate the trained agent by playing multiple games in the environment. It returns the average sum of rewards across the specified number of games.

ReplayBuffer Class: This class implements the replay buffer used for experience replay. The __init__ method initializes an empty buffer with a maximum size. The add method adds a new experience to the buffer, discarding old experiences if the buffer is full. The sample method retrieves a random batch of experiences from the buffer.

Play and Record Function: The play_and_record function makes the agent interact with the environment and stores the experiences in the replay buffer. It takes a start state, the agent, the environment, the replay buffer, and the number of steps to play.

Compute TD Loss Function: The compute_td_loss function calculates the TD loss for training the agent. It takes the agent, target network, discount factor (gamma), device, batch size, and replay buffer. It samples a batch of experiences from the replay buffer, computes the Q-values and target Q-values, and calculates the mean squared error loss between them.

Main Loop: The main training loop begins by setting the random seed and initializing the environment, action space, and observation space. The DQN agent and target network are created using the DQNAgent class, and the target network is initialized with the agent's weights. The replay buffer is populated with initial experiences by playing random actions in the environment.

Training Loop: The main training loop iterates for the specified number of steps. It updates the exploration probability, plays and records experiences in the replay buffer, computes the TD loss, performs backpropagation and network updates, and logs the loss and mean reward. The target network is updated periodically by loading the agent's weights.

Saving and Evaluation: The code saves the model and reward history at regular intervals. After the training loop, the final score is evaluated by playing one game using the trained agent.

===============================================================================================

 While learning about CNN and RL i made handwritten notes and are uploaded in this repository.The demo video has been shown in the video.
  
