import csv

import numpy as np
import gym
from keras.models import load_model
import random
from K import K

SHOW_ANIMATION = True


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    trained_model = load_model(K.MODEL_EXPORT_NAME)
    scores = []
    choices = []
    state = env.reset()

    NUM_GAMES = 1000
    MAX_NUM_STEPS_PER_GAME = 500

    for each_game in range(NUM_GAMES):
        score = 0
        prev_obs = []
        for step_index in range(MAX_NUM_STEPS_PER_GAME):
            # Enable the boolean flag show animation to render the GUI
            if SHOW_ANIMATION:
                env.render()

            if len(prev_obs) == 0:
                action = random.randrange(0, K.ACTION_SIZE - 1)
            else:
                action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])

            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            score += reward
            if done:
                break

        env.reset()
        scores.append(score)

with open('results/play_scores.csv','w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['score'])
    for row in scores:
        csv_out.writerow([row])
