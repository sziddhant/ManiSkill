# ManiSkill Benchmark

![](readme_files/teaser.jpg)

SAPIEN Manipulation Skill Benchmark (abbreviated as **ManiSkill**, pronounced as "Many Skill") is a large-scale learning-from-demonstrations benchmark for articulated object manipulation with visual input (point cloud and image). ManiSkill supports object-level variations by utilizing a rich and diverse set of articulated objects, and each task is carefully designed for learning manipulations on a single category of objects. We equip ManiSkill with high-quality demonstrations to facilitate learning-from-demonstrations approaches and perform evaluations on common baseline algorithms. We believe ManiSkill can encourage the robotic learning community to explore more on learning generalizable object manipulation skills.

Currently, ManiSkill has released 4 different tasks: OpenCabinetDoor, OpenCabinetDrawer, PushChair and MoveBucket

[Click here for our website and paper.](https://sapien.ucsd.edu/challenges/maniskill2021)

This README describes how to install ManiSkill, how to run a basic example, and relevant environment details.

**Table of Contents:**
- [ManiSkill Benchmark](#maniskill-benchmark)
  - [Updates and Announcements](#updates-and-announcements)
  - [Preliminary Knowledge](#preliminary-knowledge)
  - [What Can I Do with ManiSkill? (TL;DR)](#what-can-i-do-with-maniskill-tldr)
    - [For Computer Vision People](#for-computer-vision-people)
    - [For Reinforcement Learning People](#for-reinforcement-learning-people)
    - [For Robot Learning People](#for-robot-learning-people)
  - [Getting Started](#getting-started)
    - [System Requirements](#system-requirements)
    - [Installation](#installation)
    - [Basic Example](#basic-example)
    - [Viewer Tutorial](#viewer-tutorial)
    - [Utilize Demonstrations](#utilize-demonstrations)
    - [Baselines](#baselines)
  - [Environment Details](#environment-details)
    - [Tasks](#tasks)
    - [Robots and Actions](#robots-and-actions)
    - [Observations](#observations)
      - [Observation Structure for Each Mode](#observation-structure-for-each-mode)
      - [Segmentation Masks](#segmentation-masks)
    - [Rewards](#rewards)
    - [Termination](#termination)
    - [Evaluation](#evaluation)
    - [Visualization](#visualization)
    - [Available Environments](#available-environments)
  - [Advanced Usage](#advanced-usage)
    - [Custom Split](#custom-split)
    - [Visualization inside Docker](#visualization-inside-docker)
  - [Conclusion](#conclusion)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)


## Updates and Announcements
- July 29, 2021: The initial version of ManiSkill is released!

## Preliminary Knowledge
ManiSkill environment is built on the [gym](https://gym.openai.com/docs/) interface. 
If you have used OpenAI gym or have worked on RL previously, you can skip this section.

In ManiSkill environments, you are controlling a robot to accomplish certain predefined tasks in simulated environments. 
The tasks are defined by rewards.
Every timestep you are given an observation (e.g., point cloud / image) from the environment, and you are required to output an action (a vector) to control the robot.

![](https://gym.openai.com/assets/docs/aeloop-138c89d44114492fd02822303e6b4b07213010bb14ca5856d2d49d6b62d88e53.svg) (* image credits to OpenAI Gym)

Explanations about the above terminologies: 
- Observation: a description about the current state of the simulated environment, which may not contain complete information. Observation is usually represented by several arrays (e.g., point clouds, images, vectors).
- Action: how agent interact with the environment, which is usually represented by a vector.
- Reward: used to define the goal of the agent, which is usually represented by a scalar value.

## What Can I Do with ManiSkill? (TL;DR)

### For Computer Vision People
Based on the current visual observations (point clouds / RGBD images), your job is to output an action (a vector).
We provide supervision for actions, so training an agent is just supervised learning (more specifically, imitation learning).
You can start with little knowledge with robotics and policy learning.

### For Reinforcement Learning People
ManiSkill is designed for generalization of policies and learning from demonstrations methods.
Some topics you might be interested:
- how to combine offline RL and online RL
- generalize a manipulation policy to unseen objects
- ...

### For Robot Learning People
In simulated environment, large-scale learning and planning is feasible.
We provide four meaningful daily-life tasks for you as a good test bed.

## Getting Started
This section introduces benchmark installation, basic examples, demonstrations,
and baselines we provide.

### System Requirements
Minimum requirements
- Ubuntu 18.04 / Ubuntu 20.04 or equivalent Linux distribution. 16.04 is **not** supported.
- Nvidia GPU with > 6G memory
- Nvidia Graphics Driver 460+ (lower versions may work but are untested)

### Installation
First, clone this repository anc `cd` into it.
```
git clone https://github.com/haosulab/sapien-rl-benchmark.git
cd sapien-rl-benchmark
```

Second, install dependencies listed in `environment.yml`. 
It is recommended to use the latest (mini)[conda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) to manage the environment, but you can also choose to manually install the dependencies.
```
conda env create -f environment.yml
conda activate mani_skill
```

Last, install ManiSkill.
```
pip install -e .
```

### Basic Example
Here is a basic example for making an environment in ManiSkill, and running a random policy in it. You can also run the full script at `basic_example.py`. 
ManiSkill environment is built on the [gym](https://gym.openai.com/docs/) interface. If you have not used OpenAI Gym
before, we strongly recommend reading their documentation first.

```python
import gym
import mani_skill.env

env = gym.make('OpenCabinetDoor-v0')
# full environment list can be found in available_environments.txt

env.set_env_mode(obs_mode='state', reward_type='sparse')
# obs_mode can be 'state', 'pointcloud' or 'rgbd'
# reward_type can be 'sparse' or 'dense'
print(env.observation_space) # this shows the structure of the observation, openai gym's format
print(env.action_space) # this shows the action space, openai gym's format

for level_idx in range(0, 5): # level_idx is a random seed
    obs = env.reset(level=level_idx)
    print('#### Level {:d}'.format(level_idx))
    for i_step in range(100000):
        # env.render('human') # a display is required to use this function, rendering will slower the running speed
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action) # take a random action
        print('{:d}: reward {:.4f}, done {}'.format(i_step, reward, done))
        if done:
            break
env.close()

```
### Viewer Tutorial
The `env.render('human')` line above opens the SAPIEN viewer for interactively
debugging the environment. Here is a short tutorial.

Navigation:
- Use `wasd` keys to move around (just like in FPS games).
- Hold `Right Mouse Button` to rotate the view.
- Click any object to select it. Now press `f` to enter focus mode. In focus
  mode, hold `Right Mouse Button` to rotate around the object origin. Use `wasd`
  to exit focus mode.
- **Important limitation: do not reset a level while an object is selected,
  otherwise the program will crash.**

Inspection:
- `Pause` will keep the rendering running in a loop and pause the simulation. You can
  look around with the navigation keys when paused.
- You can use the Scene Hierarchy to select objects.
- Use Actor/Entity tab and Articulation tab to view properties of the selected
  object.

You can find a more detailed tutorial
[here](https://sapien.ucsd.edu/docs/latest/tutorial/basic/viewer.html).

### Utilize Demonstrations
We provide demonstration datasets for each task to facilitate learning-from-demonstrations approaches.
Please refer to the documentation [here](https://github.com/haosulab/ManiSkill-Learn#demonstrations).

### Baselines
We provide imitation learning and inverse RL baselines, as well as relevant development kits 
in [ManiSkill-Learn](https://github.com/haosulab/ManiSkill-Learn) framework. Try it out!

In our challenge, ManiSkill (this repo) contains the environments you need to work on, 
and [ManiSkill-Learn](https://github.com/haosulab/ManiSkill-Learn) framework contains the baselines provided by us. 
ManiSkill is a required component for this challenge, but ManiSkill-Learn is not required.
However, we encourage you to use ManiSkill-Learn to develop your algorithms and it will help you start quicker and easier.

## Environment Details
This section describes some details of the environments in the
ManiSkill Benchmark. ManiSkill environment is built on the
[gym](https://gym.openai.com/docs/) interface. If you have not used OpenAI Gym
before, we strongly recommend reading their documentation first.

### Tasks
The challenge contains 4 tasks `OpenCabinetDoor`, `OpenCabinetDrawer`,
`PushChair`, `MoveBucket`.

`OpenCabinetDoor` and `OpenCabinetDrawer` are examples of manipulating
articulated objects with revolute and prismatic joints respectively. The agent
is required to open the target door or drawer through the coordination between
arm and body.

`PushChair` exemplifies the ability to manipulate complex underactuated systems.
The agent needs to push a swivel chair to a target location. Each chair is
typically equipped with several omni-directional wheels and a rotating seat.

`MoveBucket` is an example of manipulation that heavily relies on two-arm
coordination. The agent is required to lift a bucket with a ball in it from the
ground onto a platform.

These environments can be constructed by changing the environment name passed to
`gym.make`. Keep reading for more details.

### Robots and Actions
The state of the robot is vector, and the action is also a vector. We have implemented the modules compiling the state of the robot into a vector, and the modules converting the action vector into the robot control signals. 
As a machine learning research, you do not need to worry about them. 
The details are provided below in case of you are curious.
All the tasks in ManiSkill use similar robots, which are composed of three parts: moving platform, Sciurus robot body, and one or two Franka Panda arm(s). The moving platform can move and rotate on the ground plane with its height adjustable. The robot body is fixed on top of the platform, providing support for the arms. Depending on the task, one or two robot arm(s) are connected to the robot body. There are 22 joints in the dual-arm robot and 13 for the single-arm robot.
To match with the realistic robotics setup, we use PID controllers to control the joints of the robots. The robot fingers use position controllers, while all other joints, including moving platform joints and arm joints, use velocity controllers. The controllers are internally implemented as augmented PD and PID controllers. The action space corresponds to the normalized target values of all controllers.

### Observations
ManiSkill supports three observation modes: `state`, `pointcloud` and `rgbd`, which can be set by `env.set_env_mode(obs_mode=obs_mode)`.
For all observation modes, they consist of three components: 1) A vector that describes the current state of the robot, including pose, velocity, angular velocity of the moving platform of the robot, joint angles and joint velocities of all robot joints, as well as the states of all controllers; 2) A vector that describes task-relevant information, if necessary; 3) Perception of the scene, which has different representations according to the observation modes. In `state` mode, the perception information is a vector that encodes the full physical state (e.g. pose of the manipulated objects); in `pointcloud` mode, the perception information is a point cloud captured from the mounted cameras on the robot; in `rgbd` mode, it is the RGB-D images captured from the cameras.

#### Observation Structure for Each Mode

The following script shows the structure of the observations in different observation modes.
```python
# Observation structure for pointcloud mode
obs = {
    'agent': ... , # a vector that describes agent's state, including pose, velocity, angular velocity of the moving platform of the robot, joint angles and joint velocities of all robot joints, positions and velocities of end-effectors
    'pointcloud': {
        'rgb': ... , # (N, 3) array, RGB values for each point
        'xyz': ... , # (N, 3) array, position for each point, recorded in world frame
        'seg': ... , # (N, k) array, k task-relevant segmentation masks, e.g. handle of a cabinet door, each mask is a binary array
    }
}

# Observation structure for rgbd mode
obs = {
    'agent': ... , # a vector that describes agent's state, including pose, velocity, angular velocity of the moving platform of the robot, joint angles and joint velocities of all robot joints, positions and velocities of end-effectors
    'rgbd': {
        'rgb': ... , # (160, 400, 3*3) array, three RGB images concatenated on the last dimensions, captured by three cameras on robot
        'depth': ... , # (160, 400, 3) array, three depth images concatenated on the last dimensions
        'seg': ... , # (160, 400, k*3) array, k task-relevant segmentation masks, e.g. handle of a cabinet door, each mask is a binary array
    }
}

# Observation structure for state mode
obs = ... # a vector that describes agent's state, task-relevant information, and object-relevant information; the object-relevant information includes pose, velocity, angular velocity of the object, as well as joint angles and joint velocities if it is an articulated object (e.g, cabinet). 
# State mode is commonly used when training and test on the same object, but is not suitable for studying the generalization to unseen objects, as different objects may have completely different state representations. 

```

The observations `obs` are typically obtained when resetting and stepping the environment as shown below
```python
# reset
obs = env.reset(level=level_idx)

# step
obs, reward, done, info = env.step(action)
```

#### Segmentation Masks
As mentioned in above codes, we provide task-relevant segmentation masks in `pointcloud` and `rgbd` modes.
We describe what segmentation masks we provide for each task here:
- `OpenCabinetDoor`: handle of the target door, target door, robot (3 masks in total)
- `OpenCabinetDrawer`: handle of the target drawer, target drawer, robot (3 masks in total)
- `PushChair`: robot (1 mask in total)
- `MoveBucket`: robot (1 mask in total)

Basically, we provide the robot mask and the masks which are necessary for specifying the target.

### Rewards
Reward is also return by `obs, reward, done, info = env.step(action)`. 
ManiSkill supports two kinds of rewards: `sparse` and `dense`. The sparse reward is a binary signal which is equivalent to the task-specific success condition. Learning with sparse reward is very difficult. To alleviate such difficulty, we carefully designed well-shaped dense reward functions for each task. The type of reward can be configured by `env.set_env_mode(reward_type=reward_type)`.

### Termination
Agent-environment interaction usually breaks naturally into subsequences, which we call episodes, e.g., plays of a game, trips through a maze. 

In ManiSkill tasks, an episode will be terminated if either of the following conditions is satisfied:
- Go beyond time limit
  - In all tasks, the time limit for each episode is 200, which should be sufficient to solve the task.
- Task is solved
  - We design several success metrics for each task, which can be accessed from `info['eval_info']`.
  - Each metric will be `True` if and only if some certain conditions are satisfied for 10 consecutive steps.
  - The task is regarded as solved when all the metrics are `True` at the same time.

### Evaluation
We evaluate the performance of a policy (agent) on each task by **mean success rate**.
A formal description about the challenge submission and evaluation process can be found [here](https://github.com/haosulab/ManiSkill-Submission-Example). 

Users can also evaluate their policies using the evaluation tools provided by us.
Please go through the following steps:

- Implement your solution following [this example](https://github.com/haosulab/ManiSkill-Submission-Example)
  - If your codes include file paths, please use relative paths w.r.t your code file. (Check [this example](https://stackoverflow.com/questions/10174211/how-to-make-an-always-relative-to-current-module-file-path))
- Name your solution file to `user_solution.py`
- Run `PYTHONPATH=YOUR_SOLUTION_DIRECTORY:$PYTHONPATH python mani_skill/tools/evaluate_policy.py --env ENV_NAME`
    - `YOUR_SOLUTION_DIRECTORY` is the directory containing your `user_solution.py`
    - Specify the levels on which you want to evaluate: `--level-range 100-200`
    - Note that you should using a python environment supporting your `user_solution.py`
- Result will be exported to `./eval_results.csv`



### Visualization
The environment normally runs in off-screen mode. If you want to visualize the
scene in a window and interactively inspect the scene, you need to call 

```python
env.render("human")
```

This function requires your machine to be connected to a display screen, or more
specifically, a running x-server. It opens a visualization window that you can
interact with. Do note that using this visualization can add additional helper
objects or change the appearance of objects in the scene, so you should **not**
generate any data for training purposes while the visualizer is open.

The visualizer is based on the SAPIEN viewer, and it provides a lot of debug
functionalities. You can read more about how to use this viewer
[here](https://sapien.ucsd.edu/docs/latest/tutorial/basic/viewer.html). **Note:
the render function must be called repeatedly to interact with the viewer, and
the viewer will not run by itself when the program is paused.**


### Available Environments
We registered two kinds of environments:
1. Random-object environment
  - If you call `env.reset()`, you may get a different object instance (e.g., a different chair in PushChair task)
  - Environment names: `OpenCabinetDoor-v0`, `OpenCabinetDrawer-v0`, `PushChair-v0`, `MoveBucket-v0`
2. Fixed-object environment
  - Only one object instance will be presented in the environment, it will never be replaced by other object instances
  - These environments are registered as simpler versions of the multi-object environments, and they can be used for debugging
  - Environment name examples: `PushChair_3000-v0`, `OpenCabinetDoor_1000_link_0-v0`, ... 

The full list of available environments can be found in `available_environments.txt`. 
(Note: OpenCabinetDoor and OpenCabinetDrawer also have Single-link environments, in which the target door or drawer is fixed.)


## Advanced Usage
### Custom Split
If you want to select some objects to be used in a task (e.g., create training/validation split), we provide an example for you.
Let us take PushChair task as an example, you can create such a file (`mani_skill/assets/config_files/chair_models_custom_split_example.yml`) to specify the objects you want to use. And modify [these lines](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/env/__init__.py#L110-L118) accordingly to register new environments.

### Visualization inside Docker
If you want to run the environment in a Docker while still visualizing it, it is
possible by giving the Docker container access to the graphics card and letting
the Docker container access your local x-server. When starting the Docker, make
sure to pass `--gpus all -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -e XAUTHORITY
-e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix` as
arguments. For example,
```bash
docker run -i -d --gpus all --name maniskill_container  \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix  \
  DOCKER_IMAGE_NAME
```
Next, connect the x-server by
```bash
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' maniskill_container`
```
You can replace the `maniskill_container` with your container name.

## Conclusion
Now you have familiarized yourself with the ManiSkill benchmark and you should
be able to execute policies on ManiSkill environments and visualize the
policies. Next you may want to play with [our baselines](https://github.com/haosulab/ManiSkill-Learn) 
and get started with learning from demonstrations. 

## Acknowledgements
We thank Qualcomm for sponsoring the associated challenge, Sergey Levine and Ashvin Nair for insightful discussions during the whole development process, Yuzhe Qin for the suggestions on building robots, Jiayuan Gu for providing technical support on SAPIEN, and Rui Chen, Songfang Han, Wei Jiang for testing our system.

## Citation
TBD