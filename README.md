### This repository is my implementation of paper "Virtual-to-real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation"
* I using a pioneer robot equip with SICK lidar navigative in office area ( ~10mx10m) in Coppeliasim. Agent learning to avoid collision and reaching target position.
* This project have used SAC implementation from [1](https://github.com/JM-Kim-94/rl-pendulum) and some parts from my previous projects [2](https://github.com/phuongboi/fastslam-with-coppeliasim) and [3](https://github.com/phuongboi/land-following-with-reinforcement-learning) and reference some hyper-parameter settings from
[4](https://github.com/m5823779/motion-planner-reinforcement-learning)
#### [15/3/2024] First commit
* Static environment ~ (10mx10m)
* Sampling 10 among 270 sensor of SICK TIM310
* State format: 10 laser range + 3 robot pose + 2 target position + 2 current twist (17)
* RL agent: Soft Actor critic
* Linear velocity range: (0,0.5), angular velocity range: (-1, 1), speed up v_left, v_right 4 time

#### TODO
* Navigate in dynamic environment
#### Note
* This repo have some limitations, the goal position are not encode as relative position to robot, so robot only work well in pre-define region. The scenario setting is too easy which don't have obstacles.
##### CoppeliaSim simulation
* The video show robot reach 2 pre-defined goal before return to initial position.
![alt text](https://github.com/phuongboi/mapless-navigation-in-indoor-environments/blob/main/result/202403160010.gif)

### Requirements
* CoppeliaSim v4.5.1 linux
* ROS Noetic, rospy
### Setup
* Launch `roscore` in one terminal before launch Coppeliasim in another terminal to make sure that CoppeliaSim can load ROS plugin properly
* Open vrep_scenario/room_d1.ttt in CoppeliaSim
* Training using SAC `python train_sac.py`
* Test pretrained model `python test_sac.py`
### Note
* It took near 20 hour to complete 600k step on my laptop for both simulation and training neural net, model start converge from step 300k
* I got issue of action saturation when experimenting with SAC, after a few modify, issue have been fixed https://github.com/m5823779/motion-planner-reinforcement-learning/issues/1

### Reference
* [1] https://github.com/JM-Kim-94/rl-pendulum
* [2] https://github.com/phuongboi/fastslam-with-coppeliasim
* [3] https://github.com/phuongboi/land-following-with-reinforcement-learning
* [4] https://github.com/m5823779/motion-planner-reinforcement-learning
* [5] Tai, Lei, Giuseppe Paolo, and Ming Liu. "Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation." 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017.
