import numpy as np
import time
import matplotlib.pyplot as plt

class LearningEnvironment:
    def __init__(self, num_states, num_actions, num_agents):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.transition_probabilities = self.initialize_transition_probabilities()
        self.rewards = self.initialize_rewards()

#action = stay/study more, previous course, additional content, Do assignment/quiz, move to exams, move to high-level course, move to low-level course, move to social media
    def initialize_transition_probabilities(self):
        transitions = np.zeros((self.num_states, self.num_actions, self.num_states))

        # State 0 (Start)
        transitions[0, 0, 0] = 0.1  # Stay in Start
        transitions[0, 0, 1] = 0.2  # Move to Reading
        transitions[0, 2, 1] = 0.7  # Move to reading (additional content)

        # State 1 (Reading)
        transitions[1, 4, 3] = 0.1  # Move to Exams
        transitions[1, 0, 2] = 0.7
        transitions[1, 3, 0] = 0.2  # Move back to Start

        # State 2 (Watching Video Lesson)
        transitions[2, 3, 6] = 0.6  # Move to Writing
        transitions[2, 3, 1] = 0.3  # Move to Reading
        transitions[2, 3, 3] = 0.1  # Stay in Watching Video Lesson

        # State 3 (Entertaining)
        transitions[3, 6, 1] = 0.4  # Move to Reading
        transitions[3, 6, 7] = 0.3  # Stay in Playing Game
        transitions[3, 6, 8] = 0.3  # Move to Idle Time

        # State 4 (Getting Bored/Frustration)
        transitions[4, 6, 6] = 0.7  # Move to Writing
        transitions[4, 6, 4] = 0.3  # Stay in Frustration
        transitions[4, 7, 7] = 0.4  # Play a game
        transitions[4, 7, 8] = 0.4  # Move to idle time
        transitions[4, 7, 10] = 0.2  # Quit

        # State 5 (Rest/Sleep)
        transitions[5, 1, 9] = 0.2  # Move to High-Level Course
        transitions[5, 1, 0] = 0.6  # Go back to Start
        transitions[5, 1, 1] = 0.2  # Move to Reading


        # State 6 (Writing)
        transitions[6, 3, 3] = 0.5 # Move to Assignment/Quiz
        transitions[6, 5, 9] = 0.5  # move to take an exam


        # State 7 (Playing Game)
        transitions[7, 7, 7] = 0.5  # Stay in Playing Game
        transitions[7, 7, 8] = 0.3  # Move to idle time
        transitions[7, 7, 0] = 0.2  # Move back to Start
        transitions[7, 6, 1] = 0.6  # Move to Reading
        transitions[7, 6, 8] = 0.4  # Move to idle time

        # State 8 (Idle Time)
        transitions[8, 7, 10] = 0.8  # Quit Study
        transitions[8, 7, 0] = 0.2  # Move back to Start

        # State 9 (Completing Course)
        transitions[9, 5, 0] = 0.2  # Move to Start
        transitions[9, 5, 5] = 0.8  # Stay in High-Level Course

        # State 10 (Quit Study)
        transitions[10, 0, 1] = 0.2  # Move to Reading
        transitions[10, 0, 0] = 0.8  # Go back to Start

        # Fill uninitialized actions with uniform probabilities
        for s in range(self.num_states):
            for a in range(self.num_actions):
                if np.sum(transitions[s, a, :]) == 0:
                    transitions[s, a, :] = 1.0 / self.num_states

        # Normalize all probabilities to ensure they sum to 1
        for s in range(self.num_states):
            for a in range(self.num_actions):
                total = np.sum(transitions[s, a, :])
                if total > 0:
                    transitions[s, a, :] /= total

        return transitions



    def initialize_rewards(self):
        reward_matrix = np.array([
            [10, 0, 50, 0, 0, 0, 0, -10],  # S0
            [10, 0, 0, 0, 0, 0, 0, -10],  # S1
            [10, 0, 50, 60, 0, 0, 0, -10],  # S2
            [0, 20, 0, 60, 0, 0, 0, -10],  # S3
            [0, 20, 50, 60, 100, 0, 100, 0],  # S4
            [0, 20, 50, 60, 0, 0, 0, -10],  # S5
            [10, 0, 50, 0, 100, 0, 0, 0],  # S6
            [10, 0, 50, 60, 0, 20, 70, 0],  # S7
            [0, 20, 0, 60, 0, 0, 0, -20],  # S8
            [0, 20, 0, 60, 0, 70, 0, 0],  # S9
            [70, 20, 50, 0, 0, 0, 0, -10],  # S10
        ])
        return np.tile(reward_matrix, (self.num_agents, 1, 1)) / 10

    def get_next_state_and_reward(self, agent_id, state, action):
        probabilities = self.transition_probabilities[state, action, :]
        next_state = np.random.choice(range(self.num_states), p=probabilities)
        # next_state = (state + action) % self.num_states

        reward = self.rewards[agent_id, state, action]
        return next_state, reward



# Multi-agent Q-learning
class MultiAgentQLearning:
    def __init__(self, num_agents, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99):
        self.num_agents = num_agents
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        self.Q_tables = [np.zeros((num_states, num_actions)) for _ in range(num_agents)]

    def choose_action(self, agent_id, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.num_actions))  # Explore
        return np.argmax(self.Q_tables[agent_id][state, :])  # Exploit

    def update_q_value(self, agent_id, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q_tables[agent_id][next_state, :])
        td_target = reward + self.gamma * self.Q_tables[agent_id][next_state, best_next_action]
        self.Q_tables[agent_id][state, action] += self.alpha * (td_target - self.Q_tables[agent_id][state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def train_multi_agents(env, agents, num_episodes, max_steps):
    rewards_log = {agent_id: [] for agent_id in range(env.num_agents)}
    for episode in range(num_episodes):
        for agent_id in range(env.num_agents):
            state = np.random.choice(range(env.num_states))
            total_reward = 0
            for _ in range(max_steps):
                action = agents.choose_action(agent_id, state)
                next_state, reward = env.get_next_state_and_reward(agent_id, state, action)
                agents.update_q_value(agent_id, state, action, reward, next_state)
                state = next_state
                total_reward += reward
            rewards_log[agent_id].append(total_reward)
        agents.decay_epsilon()
    return rewards_log


# Plot function
def plot_rewards(rewards_log, num_agents):
    plt.figure(figsize=(10, 6))
    for agent_id in range(num_agents):
        plt.plot(rewards_log[agent_id], label=f'Agent {agent_id}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve for Multi-Agent Q-Learning')
    plt.legend()
    plt.show()

def get_optimal_path(agent_id, start_state, q_table, env, max_steps=50):
    state = start_state
    path = [state]
    for _ in range(max_steps):
        action = np.argmax(q_table[state, :])
        next_state, _ = env.get_next_state_and_reward(agent_id, state, action)
        path.append(next_state)
        state = next_state
        if state == 10:
            break
    return path

if __name__ == "__main__":
    num_states = 11
    num_actions = 8
    num_agents = 3

    # Initialize environment and agents
    env = LearningEnvironment(num_states=num_states, num_actions=num_actions, num_agents=num_agents)
    agents = MultiAgentQLearning(num_agents=num_agents, num_states=num_states, num_actions=num_actions)

    # Train agents
    start_time = time.time()

    rewards_log = train_multi_agents(env, agents, num_episodes=2000, max_steps=100)
    training_time = time.time() - start_time


    # Plot learning curves
    plot_rewards(rewards_log, num_agents)


    for agent_id in range(num_agents):
      print(f"Optimal Path for Agent {agent_id} starting from State 0:")
      optimal_path = get_optimal_path(agent_id, start_state=0, q_table=agents.Q_tables[agent_id], env=env)
      print(optimal_path)

    # Display final Q-tables
    for agent_id in range(num_agents):
        print(f"Final Q-Table for Agent {agent_id}:")
        print(agents.Q_tables[agent_id])

        # Extract optimal policy from Q-table
        optimal_policy = {state: np.argmax(agents.Q_tables[agent_id][state, :]) for state in range(num_states)}
        print(f"Optimal Policy for Agent {agent_id}:")
        for state, action in optimal_policy.items():
            print(f"  State {state} → Action {action}")