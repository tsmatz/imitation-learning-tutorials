# Imitation Learning Algorithms Tutorial (Python)

This repository shows you the implementation examples of imitation learning (IL) from scratch in Python, with theoretical aspects behind code.

## Table of Contents

- [Behavior Cloning (BC)](01_bc.ipynb)
- [Dataset Aggregation (DAgger)](02_dagger.ipynb)
- [Maximum Entropy Inverse Reinforcement Learning (max-ent IRL)](03_maxent_irl.ipynb)
- [Relative Entropy Inverse Reinforcement Learning (rel-ent IRL)](04_relent_irl.ipynb)
- [Generative Adversarial Imitation Learning (GAIL)](05_gail.ipynb)

In this repository, I focus on above 5 model-free IL methods (which affected other works a lot in history), but this might also help you learn other recent IL algorithms (such as, [rank-game](https://www.microsoft.com/en-us/research/blog/unifying-learning-from-preferences-and-demonstration-via-a-ranking-game-for-imitation-learning/), etc).

In this repository, I'll often use basic terminologies for behavioral learning - such as, "discount", "policy", "advantages", ... If you're new to learn behavioral learning, I recommend you to learn [reinforcement learning (RL)](https://github.com/tsmatz/reinforcement-learning-tutorials) briefly at first.

> Note : Also, this repository only focuses on action-state learning, and trajectory learning (which is sometimes applied in robotics) is out of scope.<br>
> In trajectory learning, the trajectory is modeled by [GMM](https://github.com/tsmatz/gmm), [HMM](https://github.com/tsmatz/hmm-lds-em-algorithm), or MP (Movement Primitive), etc. (See [here](https://arxiv.org/abs/1811.06711) for details.)

## Imitation Learning - What's and How ?

Such like [reinforcement learning](https://github.com/tsmatz/reinforcement-learning-tutorials), imitation learning is an approach to learn how the agent takes the action to get optimal results. However, unlike reinforcement learning, imitation learning never use prior reward's functions, but do use expert's behaviors instead.<br>

There exist two main approaches for imitation learning - Behavior Cloning (BC) and Inverse Reinforcement Learning (IRL).

**Behavioral Cloning (BC)** directly learns expert's (demonstrated) behaviors without reward functions, in which the optimal mapping from states to actions is explored. It simply finds optimal solution by solving a regression or classification problem using expert's behaviors (dataset) as a supervised learning problem.<br>
When you want to refine the policy optimized by Behavioral Cloning (BC), you can also apply regular reinforcement learning method after that.<br>
The methods of [Behavior Cloning (BC)](01_bc.ipynb) and [Dataset Aggregation (DAgger)](02_dagger.ipynb) belongs to this approach.

**Inverse Reinforcement Learning (IRL)**, on the other hand, is a method to learn a cost function, i.e, recovering the unknown reward function from expert's behaviors, and then extract a policy
from the generated cost function with reinforcement learning. In complex systems, it'll often be difficult to design manual reward functions. In such cases, Inverse Reinforcement Learning (IRL) will come into play.<br>
The methods of [Maximum Entropy Inverse Reinforcement Learning](03_maxent_irl.ipynb) and [Relative Entropy Inverse Reinforcement Learning](04_relent_irl.ipynb) belongs to this approach.

Finally, [Generative Adversarial Imitation Learning (GAIL)](05_gail.ipynb) is a method inspired by Generative Adversarial Networks (GANs) and IRL, but unlike IRL method, it constrains the behavior of the agent to be approximately optimal without explicitly recovering the reward's (or cost's) function. (Hence GAIL is also applied in complex systems, unlike BC + RL.)<br>
GAIL is one of today's state-of-the-art (SOTA) imitation learning algorithm.

Reinforcement learning (RL) has achieved a great success in a wide variety of agentic and autonomous tasks. However, it's sometimes time-consuming and hard to learn from scratch in case of some complex tasks.<br>
The imitation learning makes sense in such systems, and a lot of prior successful works show us the benefits to provide prior knowledge by imitation learning, before applying reinforcement learning directly.

> Note : There also exist a lot of works to learn policy from expert's behaviors in gaming - such as, [1](https://www.nature.com/articles/nature16961), [2](https://openai.com/blog/vpt/), or [3](https://developer.nvidia.com/blog/building-generally-capable-ai-agents-with-minedojo/).

## Environment and Expert Dataset

This repository includes expert dataset (```./expert_data/ckpt0.pkl```), which is trained by PPO (state-of-the-art RL algorithm) to solve GridWorld environment.

GridWorld is a primitive environment, but widely used for behavioral training - such as, reinforcement learning or imitation learning.<br>
The following is the game rule of GridWorld environment used in this repository. (This definition is motivated by the paper "[Relative Entropy Inverse Reinforcement Learning](https://proceedings.mlr.press/v15/boularias11a/boularias11a.pdf)".)

- It has 50 x 50 grids (cells) and the state corresponds to the location of the agent on the grid.
- The agent has four actions to move in one of the directions of the compass.
- When the agent reaches to the goal state (located on the bottom-right corner), a reward ```10.0``` is given.
- For the remaining states, the reward was randomly set to ```0.0``` with probability 2/3 and to ```−1.0``` with probability 1/3.
- The duration of each trajectory has maximum ```200``` time-step.
- If the agent tries to exceed the border, the fail reward (i.e, reward=```-1.0```) is given and the agent keeps the same state.

The following picture shows GridWorld environment used in this repository (which is generated with a fixed seed value, ```1000```).<br>
When the agent is on the gray-colored states, the agent can reach to the goal state without losing any rewards. The initial state is sampled from a uniform distribution on the gray-colored states, and then maximum total reward in a single episode always becomes ```10.0```.

![GridWorld game difinition](./assets/gridworld_definition.png)

The expert dataset ```./expert_data/ckpt0.pkl``` includes the following entities.

| name          | description |
| ------------- | ------- |
| states        | Numpy array of visited states.<br>The state is an integer - in which, the left-top corner is ```0``` and the right-bottom corner is ```2499```. |
| actions       | Numpy array of corresponding actions to be taken.<br>The action is also an integer - in which, 0=UP 1=DOWN 2=LEFT 3=RIGHT. |
| rewards       | Numpy array of corresponding rewards to be obtained.<br>**This is never used in imitation learning.** (This is for reference.) |
| timestep_lens | Numpy array of time-step length.<br>Thus, the length of this array becomes the number of episodes. |

This repository also has the script [00_generate_expert_trajectories.ipynb](./00_generate_expert_trajectories.ipynb) which is used to create expert model and dataset.<br>
By modifying and running this script, you can also customize and build your own expert demonstrations.

> Note : By setting ```transition_prob=True``` in environment's constructor, you can apply the transition probability - in which, the action succeeds with probability `0.7`, a failure results in a uniform random transition to one of the adjacent states (i.e, `0.1`, `0.1`, `0.1` respectively).<br>
> Dataset in this repository (```./expert_data/ckpt0.pkl```) is generated without transition probability (i.e, always transit to the selected direction deterministically).

*Tsuyoshi Matsuzaki @ Microsoft*
