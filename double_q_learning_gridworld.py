import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import random


from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# from collections import defaultdict
# from lib.envs.gridworld import GridworldEnv
# from lib import plotting

# env = GridworldEnv()


from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
env = CliffWalkingEnv()

# matplotlib.style.use('ggplot')



def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn




def double_q_learning(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):

	#Off Policy TD - Find Optimal Greedy policy while following epsilon-greedy policy

	Q_A = defaultdict(lambda : np.zeros(env.action_space.n))

	Q_B = defaultdict(lambda : np.zeros(env.action_space.n))

	Total_Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	# state = 0
	# actions_init = 0
	# Total_Q[state][actions_init] = Q_A[state][actions_init] + Q_B[state][actions_init]

	#choose a based on Q_A for now
	policy = make_epsilon_greedy_policy(Total_Q, epsilon, env.action_space.n)


	for i_episode in range(num_episodes):

		state = env.reset()

		for t in itertools.count():

			#choose a from policy derived from Q1 + Q2 (epsilon greedy here)
			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)			
			# with taken aciton, observe the reward and the next state
			next_state, reward, done, _, = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			#choose randomly either update A or update B
			#randmly generate a for being 1 or 2
			random_number = random.randint(1,2)

			if random_number == 1:
				best_action_Q_A = np.argmax(Q_A[next_state])
				TD_Target_A = reward + discount_factor * Q_B[next_state][best_action_Q_A]
				TD_Delta_A = TD_Target_A - Q_A[state][action]
				Q_A[state][action] += alpha * TD_Delta_A

			elif random_number ==2:
				best_action_Q_B = np.argmax(Q_B[next_state])
				TD_Target_B = reward + discount_factor * Q_A[next_state][best_action_Q_B]
				TD_Delta_B = TD_Target_B - Q_B[state][action]
				Q_B[state][action] += alpha * TD_Delta_B


			if done:
				break

			state = next_state
			Total_Q[state][action] = Q_A[state][action] + Q_B[state][action]


	return Total_Q, stats


def plot_episode_stats(stats1, stats2, stats3, stats4,  smoothing_window=200, noshow=False):

	#higher the smoothing window, the better the differences can be seen

    # Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed_1 = pd.Series(stats1.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # rewards_smoothed_5 = pd.Series(stats5.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(rewards_smoothed_1, label="Double Q-Learning, Epsilon = 0.1")
    cum_rwd_2, = plt.plot(rewards_smoothed_2, label="Double Q-Learning, Epsilon = 0.5")
    cum_rwd_3, = plt.plot(rewards_smoothed_3, label="Double Q-Learning, Epsilon = 0.7")
    cum_rwd_4, = plt.plot(rewards_smoothed_4, label="Double Q-Learning, Epsilon = 0.9")
    # cum_rwd_5, = plt.plot(rewards_smoothed_5, label="Double Q Learning")

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Double Q-Learning for Epsilon Greedy Exploration")
    plt.show()


    return fig





def main():

	Number_Episodes = 1500


	print "Double Q Learning"
	Doube_Q, stats_Double_Q = double_q_learning(env, Number_Episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1)
	Doube_Q, stats_Double_Q2 = double_q_learning(env, Number_Episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.5)
	Doube_Q, stats_Double_Q3 = double_q_learning(env, Number_Episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.7)
	Doube_Q, stats_Double_Q4 = double_q_learning(env, Number_Episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.9)

	plot_episode_stats(stats_Double_Q, stats_Double_Q2, stats_Double_Q3, stats_Double_Q4)




if __name__ == '__main__':
	main()





