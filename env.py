import rospy
import math
import random
import numpy as np
from collections import deque
from std_msgs.msg import Float32MultiArray, Float32, Bool, String
from geometry_msgs.msg import Transform
from utils import absolute2relative, relative2absolute, degree2radian, visualize
class VrepEnvironment_Q():
    def __init__(self, speed, turn, rate):
        #self.scan_sub = rospy.Subscriber('scanData', Float32MultiArray, self.scan_callback)
        self.left_pub = rospy.Publisher('leftMotorSpeed', Float32, queue_size=1, latch=True)
        self.right_pub = rospy.Publisher('rightMotorSpeed', Float32, queue_size=1, latch=True)
        self.reset_pub = rospy.Publisher('resetRobot', Bool, queue_size=1)
        #self.fifo = []
        rospy.init_node('pioneer_controller')
        self.v_forward = speed
        self.v_turn = turn
        self.rate = rospy.Rate(rate) # frequence of motor speed publisher
        self.target_pos = []
        self.prev_dist = 0
        self.terminal_area =[-6, -3, 2, 3]


    def reset(self):
        self.reset_pub.publish(Bool(True))
        # transform = rospy.wait_for_message('transformData', Transform)
        # scan = rospy.wait_for_message('scanData',Float32MultiArray).data
        # p= transform.translation
        # rot = transform.rotation
        # robot_state = np.array((p.x, p.y, rot.z))

        #using default position due to coppeliasim error data cache
        robot_state = np.array((0.5305, -0.2, -np.pi/2))
        laser_ranges = [1.1659228953134664, 1.2443354568654792, 1.4033980500417036, \
        1.6774612965423652, 2.5263631882626227, 2.3779080599469484, 2.420433168625554, \
        2.8102441450949476, 2.558661097124145, 1.8819450128288457, 2.0970726220125244, \
        7.330948842232143, 6.3149659627299375, 6.3073673248291104, 6.314861815008193, \
        6.372795553325116, 7.530248620208432, 6.965292611107916, 2.5269491702281206, \
        3.733117544856479, 7.007458933744258, 6.884622624934687, 2.4326257023275684, \
        7.385658617072893, 6.179130153263128, 5.512183445108169, 5.164955177361274]

        # random target point
         # (7, -2) : (0:-1)
         # (6.25, 3.25) : (2.25:2)
         # (2.25:2.5): (0.5, -1.5)
         # (-2.5:1): (-7:-6.75)
         # (6, -3.5) : (-1.25, -6)
        x1 = np.random.uniform(low=0, high=7)
        y1 = np.random.uniform(low=-2, high=-1)
        x2 = np.random.uniform(low=2.25, high=6.25)
        y2 = np.random.uniform(low=2, high=3.25)
        x3 = np.random.uniform(low=0.5, high=2.25)
        y3 = np.random.uniform(low=-1.5, high=2.5)
        # x4 = np.random.uniform(low=-7, high=-2.5)
        # y4 = np.random.uniform(low=-6.75, high=1)
        x5 = np.random.uniform(low=-1.25, high=6)
        y5 = np.random.uniform(low=-6, high=-3.5)
        region_list = [(x1, y1), (x2, y2), (x3, y3), (x5, y5)]
        idx = np.random.randint(4)
        self.target_pos = np.round(region_list[idx], 4)

        target_pos_r = self.dist2target(robot_state)
        self.prev_dist = target_pos_r[0]
        state = laser_ranges + target_pos_r #+ action

        return np.array(state, dtype=np.float32)

    def step(self, action):
        #v_left, v_right = self.convert2leftright_vel(action)
        if action == 0:
            self.left_pub.publish(self.v_forward-self.v_turn)
            self.right_pub.publish(self.v_forward+self.v_turn)
            self.rate.sleep()
        elif action == 1:
            self.left_pub.publish(self.v_forward)
            self.right_pub.publish(self.v_forward)
            self.rate.sleep()
        elif action == 2:
            self.left_pub.publish(self.v_forward+self.v_turn)
            self.right_pub.publish(self.v_forward-self.v_turn)
            self.rate.sleep()

        transform = rospy.wait_for_message('transformData', Transform)
        scan = rospy.wait_for_message('scanData',Float32MultiArray).data
        p= transform.translation
        rot = transform.rotation
        robot_state = np.round(np.array((p.x, p.y, rot.z)), 4)

        laser_ranges = self.scan2range(scan, robot_state) # 10
        target_pos_r = self.dist2target(robot_state)
        state = laser_ranges + target_pos_r #+ action

        reward, done = self.compute_reward(laser_ranges, target_pos_r)
        info = {"reward": reward,
                "target":list(self.target_pos),
                #"collision": collision,
                "robot": list(robot_state[:2]),
                #"laser": laser_ranges
                }
        # robot go to the yard
        if self.terminal_area[0] < robot_state[0] < self.terminal_area[1] and \
        self.terminal_area[2] < robot_state[1] < self.terminal_area[3]:
            done = True
            print("Terminal area!!! \n")
        return np.array(state, dtype=np.float32), reward, done, info

    def convert2leftright_vel(action):
         # convert linear velocity and
         d = 0.381/2 # distance between to wheel of pinoneer robot
         v_left = action[0] - d * action[1]
         v_right = action[0] + d * action[1]
         return v_left, v_right

    def dist2target(self, robot_state):
        distance = np.linalg.norm(robot_state[:2] - self.target_pos)
        angle = math.atan2(self.target_pos[1] - robot_state[1], self.target_pos[0] - robot_state[0])
        t_pos_r = [distance, angle]
        return t_pos_r

    def scan2range(self, scan, robot_state):
        scan = np.reshape(scan, (-1,3))
        sample_index = np.array(np.linspace(0, len(scan)-1, 27).round(), dtype= np.int32)
        sample_scan = scan[sample_index]
        laser_ranges = []
        for end_ray_r in sample_scan:
            end_ray_w = relative2absolute((end_ray_r[0], end_ray_r[1]), robot_state)
            ray_range = np.linalg.norm(robot_state[:2] - end_ray_w[:2])
            laser_ranges.append(ray_range)

        return laser_ranges

    def compute_reward(self, laser_ranges, target_pos_r):
        done = False
        if target_pos_r[0] < 0.1: # arrived
            reward = 50
            done = True
        elif min(laser_ranges) < 0.4:
            reward = -20
            done = True
        else:
            reward = np.round((self.prev_dist - target_pos_r[0]) * 100, 4)
            self.prev_dist = target_pos_r[0]

        return reward, done



class VrepEnvironment_SAC():
    def __init__(self, rate, is_testing, fix_pos):
        #self.scan_sub = rospy.Subscriber('scanData', Float32MultiArray, self.scan_callback)
        self.left_pub = rospy.Publisher('leftMotorSpeed', Float32, queue_size=1, latch=True)
        self.right_pub = rospy.Publisher('rightMotorSpeed', Float32, queue_size=1, latch=True)
        self.reset_pub = rospy.Publisher('resetRobot', Bool, queue_size=1)
        #self.fifo = []
        rospy.init_node('pioneer_controller')
        self.rate = rospy.Rate(rate) # frequence of motor speed publisher
        self.target_pos = []
        self.prev_dist = 0
        self.fix_pos = fix_pos
        self.is_testing = is_testing

    def reset(self):
        self.reset_pub.publish(Bool(True))
        # transform = rospy.wait_for_message('transformData', Transform)
        # scan = rospy.wait_for_message('scanData',Float32MultiArray).data
        # p= transform.translation
        # rot = transform.rotation
        # robot_state = np.array((p.x, p.y, rot.z))

        #using default position due to coppeliasim error data cache
        robot_state = np.array((0.5305, -0.2, -np.pi/2))
        laser_ranges = np.array([1.088317397388187, 1.4385752712785067, 2.41956903948799, 5.174541473583594, \
                 7.350649816272867, 6.45006540999269, 7.329295108881135, 6.878220439558997, \
                 7.0089747456474045, 5.025298054485153])

        laser_ranges_norm = (laser_ranges - min(laser_ranges)) / (max(laser_ranges) - min(laser_ranges))

        x_goal = np.random.uniform(low=-1,  high=5)
        y_goal = np.random.uniform(low=0, high=-6)

        self.target_pos = np.round((x_goal, y_goal), 4)
        if self.is_testing:
            self.target_pos = np.array(self.fix_pos)


        target_pos_r = self.dist2target(robot_state)
        target_pos_r_norm = [target_pos_r[0]/(10 *np.sqrt(2)), target_pos_r[1] /(2*np.pi)]
        self.prev_dist = target_pos_r[0]
        action = [0, 0]
        yaw = [robot_state[2]/(np.pi)]
        state = list(laser_ranges) + list(robot_state) + list(self.target_pos) + action

        return np.array(state, dtype=np.float32)
        
    def step(self, action):
        v_left, v_right = self.convert2leftright_vel(action)
        self.left_pub.publish(v_left*4)
        self.right_pub.publish(v_right*4)
        #self.rate.sleep()

        transform = rospy.wait_for_message('transformData', Transform)
        scan = rospy.wait_for_message('scanData',Float32MultiArray).data
        p= transform.translation
        rot = transform.rotation
        robot_state = np.round(np.array((p.x, p.y, rot.z)), 4)

        laser_ranges = self.scan2range(scan, robot_state) # 27
        # normalization
        laser_ranges_norm = (laser_ranges - min(laser_ranges)) / (max(laser_ranges) - min(laser_ranges))
        target_pos_r = self.dist2target(robot_state)
        target_pos_r_norm = [target_pos_r[0]/(10 *np.sqrt(2)), target_pos_r[1] /(2*np.pi)]
        state = list(laser_ranges) + list(robot_state) + list(self.target_pos) + list(action)
        reward, done = self.compute_reward(laser_ranges, target_pos_r)
        info = {"reward": reward,
                "target":list(self.target_pos),
                #"collision": collision,
                "robot": list(robot_state[:2]),
                #"laser": laser_ranges
                }

        return np.array(state, dtype=np.float32), reward, done, info

    def convert2leftright_vel(self,action):
         # convert linear velocity and
         d = 0.381/2 # distance between to wheel of pinoneer robot
         v_left = action[0] - d * action[1]
         v_right = action[0] + d * action[1]
         return v_left, v_right

    def dist2target(self, robot_state):
        distance = np.linalg.norm(robot_state[:2] - self.target_pos)
        angle = math.atan2(self.target_pos[1] - robot_state[1], self.target_pos[0] - robot_state[0])
        t_pos_r = [distance, angle]
        return t_pos_r

    def scan2range(self, scan, robot_state):
        scan = np.reshape(scan, (-1,3))
        sample_index = np.array(np.linspace(0, len(scan)-1, 10).round(), dtype= np.int32)
        sample_scan = scan[sample_index]
        laser_ranges = []
        for end_ray_r in sample_scan:
            end_ray_w = relative2absolute((end_ray_r[0], end_ray_r[1]), robot_state)
            ray_range = np.linalg.norm(robot_state[:2] - end_ray_w[:2])
            laser_ranges.append(ray_range)

        return laser_ranges

    def compute_reward(self, laser_ranges, target_pos_r):
        done = False
        goal_dist = 0.2
        if self.is_testing:
            goal_dist = 0.4
        if target_pos_r[0] < goal_dist: # arrived
            reward = 15
            done = True
        elif min(laser_ranges) < 0.4:
            reward = -15
            done = True
        else:
            reward = np.round((self.prev_dist - target_pos_r[0]) * 500, 4)
            self.prev_dist = target_pos_r[0]

        return reward, done



    # def scan_callback(self, msg):
    #     # FIFO queue storing DVS data during one step
    #     self.fifo.append(msg.data)
    #     self.fifo.popleft()
    #     return


    # if action == 0:
    #     self.left_pub.publish(self.v_forward-self.v_turn)
    #     self.right_pub.publish(self.v_forward+self.v_turn)
    #     self.rate.sleep()
    # elif action == 1:
    #     self.left_pub.publish(self.v_forward)
    #     self.right_pub.publish(self.v_forward)
    #     self.rate.sleep()
    #
    # elif action == 2:
    #     self.left_pub.publish(self.v_forward+self.v_turn)
    #     self.right_pub.publish(self.v_forward-self.v_turn)
    #
    #     self.rate.sleep()
