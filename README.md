### This repository is my implementation of paper "Virtual-to-real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation"
* I using a pioneer robot equip with SICK lidar navigative in office area ( ~10mx10m) in Coppeliasim. Agent learning to avoid collision and reaching target position.
* This project have used SAC implementation from [] and some parts from my previous projects [] and []

#### [15/3/2024] First commit
* Static environment ~ (10mx10m)
* Sampling 10 among 270 sensor of SICK TIM310
* State format: 10 laser range + 3 robot pose + 2 target position + 2 current twist (17)
* RL agent: Soft Actor critic
*

#### TODO
* Navigate in dynamic environment

##### CoppeliaSim simulation
* This is no speed up video 
![alt text]()

### Requirements
* CoppeliaSim v4.5.1 linux
* ROS Noetic, rospy
### Setup
* Launch `roscore` in one terminal before launch Coppeliasim in another terminal to make sure that CoppeliaSim can load ROS plugin properly
* Open vrep_scenario/room_d.ttt in CoppeliaSim and modify child_script of Pioneer_p3dx by v_rep_scenario/rosInterface_slam.lua
* Start CoppeliaSim simulation, make sure topics is work as expect by `rostopic list`
* Run `python train_qlearning.py`
* Test `python test_qlearning.py`
### Note
*
*

### Reference
* [1] https://github.com/andriusbern/slam
* [2] https://github.com/kunnnnethan/FastSLAM
* [3] Montemerlo, Michael, et al. "FastSLAM: A factored solution to the simultaneous localization and mapping problem." Aaai/iaai 593598 (2002).
* [4] Grisetti, Giorgio, Cyrill Stachniss, and Wolfram Burgard. "Improved techniques for grid mapping with rao-blackwellized particle filters." IEEE transactions on Robotics 23.1 (2007): 34-46.
