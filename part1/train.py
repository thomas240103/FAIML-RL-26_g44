"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
from agent import Agent, Policy

def main():
    render = True

    if render:
        env = gym.make('Hopper-v4', render_mode='human')
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array')

    policy=Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])
    agent = Agent(policy)


    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    n_episodes = 50
    for episode in range(n_episodes):
        done = False
        current_state, info = env.reset()  # Reset environment to initial state

        while not done:

            if len(agent.states) == 0:
                action = env.action_space.sample()
                action_log_prob = 0.0

            else:
                action, action_log_prob = agent.get_action(current_state)

            state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep

            done = terminated or truncated

            agent.store_outcome(state=current_state, next_state=state, reward=reward, action_log_prob=action_log_prob, done=done)
            current_state = state

            if render:
                env.render()
        
        agent.update_policy()
    
    env.close()


if __name__ == '__main__':
    main()