import gym_cutting_stock
import gymnasium as gym
import numpy as np
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":
    # # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test GreedyPolicy
    gd_policy = GreedyPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = gd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # Uncomment the following code to test your policy
    # Reset the environment
    observation, info = env.reset(seed=42)
    # print(info)
    # observation, info = env.reset(seed=1)
    policy2210xxx = Policy2210xxx(policy_id=1)
    ep = 0
    while ep < NUM_EPISODES:
        action = policy2210xxx.get_action(observation, info)
        # action = gd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        # print(info)

        if terminated or truncated:
            area = 0
            for sid, stock in enumerate(observation["stocks"]):
                if np.any((stock != -1) & (stock != -2)):
                    stock_w = np.sum(np.any(stock != -2, axis=1))
                    stock_h = np.sum(np.any(stock != -2, axis=0))
                    area += stock_w * stock_h
                
            print (area)
            print (info)
            observation, info = env.reset(seed=ep)
            ep += 1

env.close()
