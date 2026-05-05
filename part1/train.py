"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent, Policy


def evaluate_policy(agent, eval_env, n_eval_episodes=5):
    """Run deterministic evaluation episodes and return mean cumulative reward."""
    eval_returns = []

    for _ in range(n_eval_episodes):
        done = False
        state, _ = eval_env.reset()
        episode_return = 0.0

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            action_np = action.detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = eval_env.step(action_np)
            done = terminated or truncated
            episode_return += reward
            state = next_state

        eval_returns.append(episode_return)

    return float(np.mean(eval_returns))


def main():
    render = False

    if render:
        env = gym.make('Hopper-v4', render_mode='human')
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array')

    eval_env = gym.make('Hopper-v4', render_mode='rgb_array')

    policy=Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])
    agent = Agent(policy)


    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    n_episodes = 3000
    print_every = 10
    eval_every = 100
    n_eval_episodes = 5

    train_returns = []
    eval_episodes = []
    eval_returns = []

    for episode in range(n_episodes):
        done = False
        current_state, info = env.reset()  # Reset environment to initial state
        episode_return = 0.0

        while not done:

            action, action_log_prob,vote = agent.get_action(current_state)
            action_np = action.detach().cpu().numpy()
            state, reward, terminated, truncated, _ = env.step(action_np)  # Step the simulator to the next timestep

            done = terminated or truncated
            episode_return += reward

            agent.store_outcome(state=current_state, next_state=state, reward=reward, action_log_prob=action_log_prob, done=done, state_value=vote)
            current_state = state

            if render:
                env.render()

        train_returns.append(episode_return)
        if (episode + 1) % print_every == 0:
            recent_mean = np.mean(train_returns[-print_every:])
            print(f"Episode {episode + 1}/{n_episodes} | Return: {episode_return:.2f} | Mean({print_every}): {recent_mean:.2f}")

        if (episode + 1) % eval_every == 0:
            eval_mean_return = evaluate_policy(agent, eval_env, n_eval_episodes=n_eval_episodes)
            eval_episodes.append(episode + 1)
            eval_returns.append(eval_mean_return)
            print(f"[EVAL] Episode {episode + 1} | Mean return over {n_eval_episodes} episodes: {eval_mean_return:.2f}")

        agent.update_policy()

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_returns, label='Train return per episode', alpha=0.4)

    window = 100
    if len(train_returns) >= window:
        moving_avg = np.convolve(train_returns, np.ones(window) / window, mode='valid')
        plt.plot(
            np.arange(window - 1, len(train_returns)),
            moving_avg,
            label=f'Train moving average ({window})',
            linewidth=2
        )

    if eval_returns:
        plt.plot(eval_episodes, eval_returns, marker='o', linestyle='-', label='Evaluation mean return')

    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.title('Learning curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('part1/learning_curve.png', dpi=150)
    plt.show()

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
