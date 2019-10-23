# Reinforcement Learning with Duelling Neural Networks

![preview of landing](https://i.imgur.com/cPSV7CX.gif)

This repo demonstrates how duelling neural networks can be applied to a reinforcement learning problem. More specifically, we complete the [Open AI](https://gym.openai.com/envs/LunarLander-v2/) problem which challenges the agent ("spaceship") to land a spaceship in a designated landing zone. The model landed successfully for the first time on attempt #383 and completed the challenge on attempt #589. The results of this challenge revealed a successful yet inconsistent model and examined the effects of changing two key inputs to the models.

## Usage

```bash
git clone https://github.com/mikeroher/lunar-lander-reinforcement-learning llrl/
cd llrl

# Change `K.py` to adjust any parameters or leave file as-is to use defaults
# Default parameters are:
#   EPISODES = 3000
#   LEARNING_RATE = 0.25e-3
#   MEMORY_SIZE = 100_000
#   EPSILON_DECAY = 0.99995
#   EPSILON_MIN = 0.05
#   EPSILON_INIT = 1
#   BATCH_SIZE = 64
#   DISCOUNT_FACTOR = 0.99
#   STATE_SIZE = 8
#   ACTION_SIZE = 4
#   PRINT_EVERY = 100

# To run the simulation
python main.py

# To generate plots
python plot.py
```

## Results

### Training & Testing

<div>
<img src="https://i.imgur.com/v40bFX5.jpg"  width="400px" />
</div>

The model solved the Lunar Lander environment in 589 attempts. Solving the environment requires the agent to achieve an average score of 200 over its 100 most recent attempts. The model’s learning performance was plotted in the above plot ("Learning Results") with a moving average (MA) window size of five attempts for smoothing purposes. The red line indicates the minimum score of 200 needed for success. The green line highlights the 383rd attempt as it was the first attempt where the agent first lands successfully.

It can be observed in the above plot that the general trend of the plot shows the model’s performance improving over time. Spikes during the initial training (i.e. before attempt 350) are not concerning as they can be explained by the model’s initial learning. However, a concerning aspect of the plots are the large spikes around attempt number 450 and attempt 525. These spikes are unexpected as the training is nearing completion, yet the model is still experiencing large flucuations. It is hypothesized that removing the average score stopping condition of the model and instead training for a fixed number of attempts (i.e., 2000), would result in significantly smaller spikes. This was not tested due to the limited computational resources available.

<div>
<img src="https://i.imgur.com/g2yHU2I.jpg"  width="400px" />
</div>

Once the initial training was completed, the trained model was evaluated by running it on 1000 additional attempts. The results of this evaluation are plotted above ("Playing Results") with a moving average window of 20 attempts for smoothing. The red line indicates the minimum score of 200 required for success. The results of this evaluation were underwhelming: the model averaged a score of 114 with very inconsistent landings. This would make the model impractical as a real-life lunar landing algorithm as its large flucuations would result in a significant number of likely fatal crashes. An ideal model would be extremely cautious, sometimes over-cautious, with an approximately zero crash rate (i.e. very small flucuations).

### Comparison to Other Approaches

![comparison to other approaches](https://i.imgur.com/ZdE1hUl.png)

The results of comparing the DNN approach of solving the lunar lander challenge to other solutions is shown in the above table. While the consistency of the DNN approach did not meet expectations, the speed at which the model solved the Lunar Lander problem was excellent. The DNN model per- formed better than all approaches, except for B.S. Haney’s Proximal Policy Optimization 1 approach. The number of attempts is used as the comparison, rather than accuracy as most approaches terminated after achieving an average score of 200, rather than continue training to achieve the maximum possible score. The comparators sourced were disappointing as there was minimal variety in approaches. All implemented some form of a RL algorithm with the bulk of the differences occurring on how the agent makes decisions (i.e., hill climbing, multiple state-action functions, prioritizing most effective decisions over most recent, etc.). It would have been interesting to compare the results to a genetic algorithm or an image vision approach. It is also important to note that the results in the above table are self-reported by the respective authors. Care was taken where possible to verify the code operated as described.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)

## References

+ [John McDonald's Agent Implementation](https://github.com/johnptmcdonald/openAI-gym-lunar-lander/)
+ [ReplayBuffer from Udacity's Starter](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/exercise/dqn_agent.py)

