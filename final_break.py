
import random
import numpy as np
import torch
import torch.nn as nn
import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, gaussian

import os
import io
import base64
import time
import glob
from IPython.display import HTML
import torch.nn.functional as F
from gym.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack
from gym.wrappers import TransformReward


def make_env(env_name, clip_rewards = True, seed = None):
	# complete this function which returns an object 'env' using gym module
	# Use AtariPreprocessing, FrameStack, TransformReward(based on the clip_rewards variable passed in the arguments of the function), check their usage from internet
	# Use FrameStack to stack 4 frames
	# TODO
  env = gym.make(env_name)
  env = AtariPreprocessing(env)
  env = FrameStack(env, num_stack=4)
  if clip_rewards:
        env = TransformReward(env, lambda r: np.sign(r))
  if seed is not None:
        env.seed(seed)
  return env

# Initialize the device based on CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Next we create a class DQNAgent which is the class containing the neural network, This class is derived from nn.Module

class DQNAgent(nn.Module):
	def __init__(self, state_shape, n_actions, epsilon):
		super(DQNAgent, self).__init__()  # Call the nn.Module constructor
    # Calculate the number of input channels for the first convolutional layer
		num_frames = state_shape[0]
    # First Convolutional Layer
		self.conv1 = nn.Conv2d(num_frames, 16, kernel_size=8, stride=4)
		self.relu1 = nn.ReLU()
    # Second Convolutional Layer
		self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
		self.relu2 = nn.ReLU()

    # Flatten the output for the linear layer
		self.flatten = nn.Flatten()

    # Linear Layer
		conv_out_size = self.calculate_conv_output_size(state_shape)
		self.fc1 = nn.Linear(conv_out_size, 256)
		self.relu3 = nn.ReLU()

    # Final Linear Layer
		self.fc2 = nn.Linear(256, n_actions)

    # Epsilon for exploration in epsilon-greedy policy
		self.epsilon = epsilon
		# TODO
		# Here state_shape is the input shape to the neural network.
		# n_Actions is the number of actions
		# epsilon is the probability to explore, 1-epsilon is the probabiltiy to stick to the best actions
		# initialise a neural network containing the following layers:
		# 1)a convulation layer which accepts size = state_shape, in_channels = 4( state_shape is stacked with 4 frames using FrameStack ), out_channels = 16, kernel_size = 8, stride = 4 followed by ReLU activation
		# 2)a convulation layer, in_channels = 16, out_channels = 32, kernel_size = 4, stride = 2 followed by ReLU activation function
		# 3)layer to convert the output to a 1D output which is fed into a linear Layer with output size = 256 followed by ReLU actiovation
		# 4) linear Layer with output size = 'number of actions'(the qvalues of actions)

	def calculate_conv_output_size(self, state_shape):
		dummy_input = torch.zeros(1, *state_shape)
		dummy_output = self.conv1(dummy_input)
		dummy_output = self.conv2(dummy_output)
		conv_output_size = dummy_output.view(dummy_output.size(0), -1).size(1)
		return conv_output_size

	def forward(self, state_t):
		state_t = self.relu1(self.conv1(state_t))
		state_t = self.relu2(self.conv2(state_t))
		state_t = self.flatten(state_t)
		state_t = self.relu3(self.fc1(state_t))
		q_values = self.fc2(state_t)
		return q_values
		# return qvalues generated from the neural network

	def get_qvalues(self, state_t):
		q_values = self.forward(state_t)
		q_values_np = q_values.detach().cpu().numpy()
		return q_values_np
		# returns the numpy array of qvalues from the neural network

	def sample_actions(self, qvalues):
		batch_size = qvalues.shape[0]
		actions = []

		for _ in range(batch_size):
			if random.random() < self.epsilon:
				action = random.randint(0, qvalues.shape[0] - 1)
			else:
				action = qvalues[_].argmax().item()
			actions.append(action)
		return actions
		#TODO
		# sample_Actions based on the qvalues
		# Use epsilon for choosing between best possible current actions of the give batch_size(can be found from the qvalues object passed in argument) based on qvalues vs explorations(random action)
		# return actions
		# pass



def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
	# used for evaluationing the trained agent for number of games = n_games and step in each game = t_max
	# returns the mean of sum of all rewards across n_games
	#TODO
    total_rewards = 0
    for i in range(n_games):
        state = env.reset()
        done = False
        t = 0

        while not done and t < t_max:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = agent.get_qvalues(state_t)
            if greedy:
                action = q_values.argmax().item()
            else:
                action = agent.sample_actions(q_values)[0]

            next_state, reward, done, i = env.step(action)
            total_rewards += reward
            state = next_state
            t += 1

    return total_rewards / n_games

# Now we create a class ReplayBuffer. The object of this class is responsible for storing the buffer information based on the agent's action when we play the agent(i.e, current_State -> action -> next_state -> done_flag ->reward)
# For Deep Q Learning we sample information of size = 'batch_size' from the ReplayBuffer and return that information for training
# This buffer has a fixed size, set that to 10**6. remove previous information as new information is passed in the buffer


class ReplayBuffer:
	def __init__(self, size):
		#TODO
		# size is the maximum size that the buffer can hold
		self.size=size
		self.buffer=[]
		self.position=0


	def __len__(self):
		# no need to change
		return len(self.buffer)

	def add(self, state, action ,reward, next_state, done):
		experience=(state, action ,reward, next_state, done)
		if len(self.buffer)<self.size:
			self.buffer.append(experience)
		else:
			self.buffer[self.position] = experience

		self.position = (self.position + 1) % self.size
		#TODO
		# store the information passed in one call to add as 1 unit of informmation




	def sample(self, batch_size):
		#TODO
		# return a random sampling of 'batch_size' units of information
		batch=random.sample(self.buffer,min(batch_size,len(self.buffer)))
		return batch


def play_and_record(start_state, agent, env, exp_replay, n_steps = 1):
	state = start_state
	for _ in range(n_steps):
		state_t = torch.tensor([state], dtype=torch.float32, device=device)
		qvalues_t = agent(state_t)
		qvalues = qvalues_t.cpu().detach().numpy()[0]
		action = agent.sample_actions(qvalues)[0]
		next_state, reward, done, _ = env.step(action)
		exp_replay.add(state, action, reward, next_state, done)
		if done:
			state = env.reset()
		else:
			state = next_state



	# use this function to make the agent play on the env and store the information in exp_replay which is an object of class ReplayBuffer
	# n_steps is the number of steps to be played in this function on one call
	#TODO
	# pass


def compute_td_loss(agent, target_network, device, batch_size, exp_replay ,gamma = 0.99,):
	states, actions, rewards, next_states, dones = zip(*exp_replay.sample(batch_size))
	states = torch.tensor(states,dtype=torch.float32).to(device)
	actions = torch.tensor(actions,dtype=torch.long).to(device)
	rewards = torch.tensor(rewards,dtype=torch.float32).to(device)
	next_states = torch.tensor(next_states,dtype=torch.float32).to(device)
	dones = torch.tensor(dones,dtype=torch.float32).to(device)

    # Compute the predicted Q-values of the actions using the agent
	q_values = agent(states)
	predicted_qvalues = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
	gamma=torch.tensor(gamma,dtype=torch.float32)

    # Compute the target Q-values of the actions using the target network
	with torch.no_grad():
		target_q_values = target_network(next_states)
		target_qvalues_of_actions = rewards + torch.mul(gamma,target_q_values.max(dim=1)[0]) * torch.logical_not(dones)

    # Compute the TD loss (Mean Squared Error)
	loss = torch.nn.MSELoss()(predicted_qvalues, target_qvalues_of_actions)
	return loss

	# Here agent is the one playing on the game and target_network is updates using agent after some fixed steps as is done in Deep Q Learning
	# sample 'batch_size' units of info stored in the exp_replay
	# Find the predicted_qvalues_of_actions using agent and target_qvalues_of_actions using target_network, find the loss based on these Mean Squared Error of these two
	# IMPORTANT NOTE : check the type of objects, U need to convert the actions, rewards, etc, to toch.tensors for backward propogation using pytorch
	#TODO
	# pass


############# MAIN LOOP ###############
from tqdm import trange
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch.optim as optim

seed = 108
random.seed(108)
np.random.seed(108)
torch.manual_seed(108)



##  setup environment using make_env function defined above
# find action_space and observation_space of the atari
# Use env_name = "BreakoutNoFrameskip-v4"
# Reset the environment before starting to train the agent and everytime the game ends (U will get a done flag which is a boolean representing whether the game has ended or not)
# TODO
env_name = "BreakoutNoFrameskip-v4"
env = make_env(env_name)
action_space = env.action_space
observation_space = env.observation_space
state_shape = observation_space.shape
n_actions = action_space.n
state = env.reset()
done = False



# create agent from DQNAgent class the online network
# create target_network from DQNAgent class is updated after some fixed steps from agent
# Note initialise target network values from agent
# Create the online network (agent) and target network objects

agent = DQNAgent(observation_space.shape, action_space.n, epsilon=0.1)
target_network = DQNAgent(observation_space.shape, action_space.n, epsilon=0.1)


# Initialize the target network with the agent's values
target_network.load_state_dict(agent.state_dict())
agent.to(device)
target_network.to(device)
# TODO


# created a ReplayBuffer object and saved some information in the object by playing the agent. It is better to populate some information in the Buffer, hence this step
#filling experience replay with some samples using full random policy
exp_replay = ReplayBuffer(10**6)
for i in range(200):
    play_and_record(state, agent, env, exp_replay, n_steps=10**2)
    print( "Replay Buffer : i : ", i)
    if len(exp_replay) == 10**6:
        break
print(len(exp_replay))



#setup some parameters for training
timesteps_per_epoch = 2
batch_size = 32

total_steps = 2 * 10**6

#Optimizer
optimizer = torch.optim.Adam(agent.parameters(), lr=2e-5)
# TODO - use Adam optimiser from torch with learning rate (lr) = 2*1e-5


#setting exploration epsilon
start_epsilon = 0.1
end_epsilon = 0.05
eps_decay_final_step = 1 * 10**5

# setup spme frequency for logginf and updating target network
loss_freq = 20
refresh_target_network_freq = 100
eval_freq = 10000

# to clip the gradients
max_grad_norm = 5000

mean_rw_history = []
td_loss_history = []

SAVE_INTERVAL = 50000

from numpy import asarray
from numpy import savetxt


def epsilon_schedule(start_eps, end_eps, step, final_step):
    return start_eps + (end_eps-start_eps)*min(step, final_step)/final_step




# TODO - reset the state of the environment before starting
state=env.reset()
## MAIN LOOP STARTING

for step in trange(total_steps + 1):

	#TODO update the exploration probability (epsilon) as time passes
		epsilon = epsilon_schedule(start_epsilon, end_epsilon, step, eps_decay_final_step)
		agent.epsilon = epsilon
	#TODO taking timesteps_per_epoch and update experience replay buffer, (use play_and_record)
		play_and_record(state, agent, env, exp_replay, n_steps=timesteps_per_epoch)
	#TODO compute loss
		loss = compute_td_loss(agent, target_network, device=device, batch_size=batch_size, exp_replay=exp_replay,gamma=0.99)
	#TODO Backward propogation and updating the network parameters
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)  # Clip gradients to avoid exploding gradients
		optimizer.step()
	# IMPORTANT NOTE : You only need to update the parameters of agent and not of target_network, that will be done according to refresh_target_network_freq. But Backward Propogation will take into account the target_network parameters as well. So use detach() method on target_network while calculating the loss. Google what it does and how to use !!


		if step % loss_freq == 0:
			td_loss_history.append(loss.data.cpu().item())


		if step % refresh_target_network_freq == 0:
        #TODO Load agent weights into target_network
			target_network.load_state_dict(agent.state_dict())

		if step % eval_freq == 0:
			mean_reward = evaluate(make_env(env_name, seed=step), agent, n_games=3, greedy=True, t_max=6000)
			mean_rw_history.append(mean_reward)

		print("mean_reward : ", mean_reward)

		clear_output(True)
		print("buffer size = %i, epsilon = %.5f" %
				(len(exp_replay), agent.epsilon))


		if step % SAVE_INTERVAL == 0 and step!= 0:
			print('Saving...')
			device = torch.device('cpu')
			torch.save(agent.state_dict(), f'model_{step}.pth')
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		savetxt(f'reward_{step}.csv', np.array(mean_rw_history))



# savetxt('reward_final.csv', np.array(mean_rw_history))

final_score = evaluate(
  make_env(env_name),
  agent, n_games=1, greedy=True, t_max=10000
)
print('final score:', final_score)


