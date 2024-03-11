### This repository is my implementation of paper "Virtual-to-real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation"
* I using a pioneer robot equip with SICK lidar navigative in office area ( ~10mx10m) in Coppeliasim. Agent learning to avoid collision and reaching target position.
* This project have used some parts from my previous projects https://github.com/phuongboi/fastslam-with-coppeliasim and https://github.com/phuongboi/land-following-with-reinforcement-learning
#### [3/11/2023] First commit
* Static environment
* Sampling 27 among 270 sensor of SICK TIM310

#### TODO
* Navigate in dynamic environment

##### CoppeliaSim simulation
![alt text]()
##### Output Map
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
* I have do many experiment with q learning and SAC[]. Q learning gave quite straight forward outcome while I face some problem to fine tune hyper-parameter for SAC.
* I don't normalize laser ranges and use control command as input like original work. Paper's author only use sample 10 laser ray, in some enviroment like  
* I do some expreriment with reward factors, I encounter that if following target reward is to small compare to collision or reaching goal, agent tend to try to avoid collision and don't care about target


### Reference
* [1] https://github.com/andriusbern/slam
* [2] https://github.com/kunnnnethan/FastSLAM
* [3] Montemerlo, Michael, et al. "FastSLAM: A factored solution to the simultaneous localization and mapping problem." Aaai/iaai 593598 (2002).
* [4] Grisetti, Giorgio, Cyrill Stachniss, and Wolfram Burgard. "Improved techniques for grid mapping with rao-blackwellized particle filters." IEEE transactions on Robotics 23.1 (2007): 34-46.
