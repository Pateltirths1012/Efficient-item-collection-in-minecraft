import gym
import os



# Create the environmentap
env = gym.make('CartPole-v1')  # Replace 'CartPole-v1' with the environment you want to use

# Reset the environment
observation = env.reset()

# Run the simulation for a few steps
for _ in range(1000):  # Run for 100 steps
    # Choose a random action (replace with your agent's logic)
    action = env.action_space.sample()

    # Take the action
    new_observation, reward, done, info = env.step(action)

    # Render the environment (optional, but useful for visualization)
    env.render()

    # Print some information (optional)
    # print(f"Observation: {new_observation}, Reward: {reward}, Done: {done}")

    # Update the observation
    observation = new_observation

    # Check if the episode is done
    if done:
        # print("Episode finished")
        observation = env.reset()  # Reset for a new episode

# Close the environment
env.close()


