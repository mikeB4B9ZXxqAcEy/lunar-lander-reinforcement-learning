import numpy as np
import gym
from collections import deque
import csv

from model import Model
from K import K
from agent import Agent
from replay_buffer import ReplayBuffer

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    memory = ReplayBuffer(
        action_size=K.ACTION_SIZE,
        buffer_size=K.MEMORY_SIZE,
        batch_size=K.BATCH_SIZE,
        seed=0)
    recent_scores = deque(maxlen=100)

    model_A = Model.build_model()
    model_B = Model.build_model()

    all_scores = []

    agent = Agent(env, model_A, model_B, memory)

    for e in range(K.EPISODES):
        score = agent.run_episode()
        model_B.set_weights(model_A.get_weights())
        recent_scores.append(score)
        all_scores.append((score, agent.epsilon, len(memory)))

        if np.mean(recent_scores) >= 200:
            model_A.save(K.MODEL_EXPORT_NAME, overwrite=True)
            break
        if e % K.PRINT_EVERY == 0:
            avg_score = np.mean(recent_scores)
            print(f"Episode: {e}\tCurrent Epsilon: {agent.epsilon}\tCurrent Score: {avg_score}")

with open('results/train_scores.csv','w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['score', 'epsilon', 'mem_size'])
    for row in all_scores:
        csv_out.writerow(row)
