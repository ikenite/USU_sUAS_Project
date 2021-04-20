#!/usr/bin/env python

import rospy
import numpy as np
from mat import mat
from algorithms import Algorithms
from table_dp import Table
from sua.msg import AttitudeController, StateData, Waypoints, PathParameters
from mavros_test_common import MavrosTestCommon
from std_msgs.msg import Float32, Int32MultiArray
from threading import Thread
from time import sleep
from ast import literal_eval
from geodesy.utm import fromLatLong
import rospkg
import json
import sys
import os

# set default plan if not passed in on command line
rospack = rospkg.RosPack()
PLAN_FILE = rospack.get_path('sua') + "/scripts/missions/path_manager.plan"

pi = np.pi


class UAV(MavrosTestCommon):
    def setUp(self, path_follower):
        super(UAV, self).setUp()

        # SUA Algorithm tuning parameters
        # ----------------------------
        self.chi_inf = pi / 2
        self.k_path = 0.0125
        self.k_orbit = 3.5
        self.R = 50  # fillet radius (m) (This should be set to the min radius)
        # ----------------------------

        self.path_follower = path_follower
        self.chi = 0.0
        self.chi_c = 0.0
        self.h_c = 0.0
        self.Va_c = 18  # start with a constant airspeed
        self.current_waypoint = 0
        self.tab = Table()
        self.dp_list = []
        self.alg = Algorithms()
        self.output = AttitudeController()
        self.e_crosstrack = Float32()
        self.waypoints = Waypoints()
        self.path_params = PathParameters()
        self.position = mat([0, 0, 0]).T
        self.W = None
        self.Chi_waypoint = None

        rospy.Subscriber('attitude_bridge/state_data', StateData, self.att_state_callback)
        self.pub = rospy.Publisher('attitude_bridge/commanded', AttitudeController, queue_size=1)
        self.crosstrack_pub = rospy.Publisher('crosstrack_error', Float32, queue_size=1)
        self.waypoint_pub = rospy.Publisher('waypoints', Waypoints, queue_size=10)
        self.path_pub = rospy.Publisher('path_params', PathParameters, queue_size=10)

        # read waypoint file
        self.read_plan()
        self.calc_Chi_waypoint()

        if self.path_follower:
            self.path_params.header.stamp = rospy.Time.now()
            self.path_params.r = traj.r
            self.path_params.q = traj.q
            self.path_params.c = traj.c
            self.path_params.rho = traj.rho
            self.path_params.lamb = traj.lamb
            path_params_rate = rospy.Rate(0.5)
            for i in range(0, 3):
                self.path_pub.publish(self.path_params)
                path_params_rate.sleep()

        # start running algorithms
        self.pub_thread = Thread(target=self.waypoint_publisher, args=())
        self.pub_thread.daemon = True
        self.pub_thread.start()

    def tearDown(self):
        super(UAV, self).tearDown()

    def read_plan(self):
        """ read .plan file from QGroundControl and build W member """
        with open(PLAN_FILE, 'r') as f:
            d = json.load(f)
            if 'mission' in d:
                d = d['mission']

            if 'plannedHomePosition' in d:
                home_lat = d['plannedHomePosition'][0]
                home_lon = d['plannedHomePosition'][1]
                utm_home = fromLatLong(home_lat, home_lon, 0)
                geo_home = utm_home.toPoint()
            else:
                raise KeyError("No home position in .plan file")

            if 'items' in d:
                for wp in d['items']:
                    lat = float(wp['params'][4])
                    lon = float(wp['params'][5])
                    alt = float(wp['params'][6])
                    utm_wp = fromLatLong(lat, lon, alt)
                    geo_wp = utm_wp.toPoint()

                    # make local by subtracting home position
                    N = geo_wp.y - geo_home.y
                    E = geo_wp.x - geo_home.x
                    D = -(geo_wp.z - geo_home.z)

                    if self.W is None:
                        self.W = mat([N, E, D]).T
                    else:
                        self.W = mat(np.hstack((self.W, mat([N, E, D]).T)))
            else:
                raise KeyError("No waypoints in .plan file")

        self.waypoints.header.stamp = rospy.Time.now()
        self.waypoints.x = Int32MultiArray(data=self.W[0])
        self.waypoints.y = Int32MultiArray(data=self.W[1])
        self.waypoints.z = Int32MultiArray(data=self.W[2])

        waypoint_rate = rospy.Rate(0.5)
        for i in range(0, 3):
            self.waypoint_pub.publish(self.waypoints)
            waypoint_rate.sleep()

    def calc_Chi_waypoint(self):
        """ read .plan file from QGroundControl and build Chi_waypoint member """
        # lines used for debugging
        # self.position = mat([10, 0, 50]).T
        # self.W = mat([[0, -10, -10, 0], [10, 10, 0, 0], [50, 50, 50, 50]])
        m, n = self.W.shape
        self.Chi_waypoint = mat(np.zeros(n)).T

        # define chi zero vector
        v0 = mat([0, 1, 0]).T

        # set chi for other waypoints
        for i in xrange(0, n):

            # define vectors between waypoints
            if i == 0:
                v1 = mat([float(self.position[1]), float(self.position[0]), 0]).T - \
                    mat([self.W[1, i], self.W[0, i], 0]).T
                v2 = mat([self.W[1, i + 1], self.W[0, i + 1], 0]).T - \
                    mat([self.W[1, i], self.W[0, i], 0]).T

            elif i == n - 1:
                v1 = mat([self.W[1, i - 1], self.W[0, i - 1], 0]).T - \
                    mat([self.W[1, i], self.W[0, i], 0]).T
                v2 = -v1

            else:
                v1 = mat([self.W[1, i - 1], self.W[0, i - 1], 0]).T - \
                    mat([self.W[1, i], self.W[0, i], 0]).T
                v2 = mat([self.W[1, i + 1], self.W[0, i + 1], 0]).T - \
                    mat([self.W[1, i], self.W[0, i], 0]).T

            # normailze vectors
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)

            # define chi sign
            sign = int(np.sign(v2[0] - v1[0]))
            if sign == 0:
                sign = 1

            bisector = v1 + v2

            # check if waypoints are in a straight line
            if bisector[0] == 0 and bisector[1] == 0:
                self.Chi_waypoint[i] = sign * self.angle(v0, v2)
            else:
                # define both chi options
                opt1 = mat([float(bisector[1]), float(-bisector[0]), 0]).T
                opt2 = mat([float(-bisector[1]), float(bisector[0]), 0]).T

                # determine which option is closer to the out vector
                opt1_dist = self.angle(opt1, v2)
                opt2_dist = self.angle(opt2, v2)

                # assign chi accordingly
                if abs(opt1_dist) < abs(opt2_dist):
                    self.Chi_waypoint[i] = sign * self.angle(v0, opt1)
                else:
                    self.Chi_waypoint[i] = sign * self.angle(v0, opt2)

    def att_state_callback(self, msg):
        """ brings in state data from bridge node """
        self.chi = msg.chi
        # convert from ENU to NED
        self.position = mat([msg.position.y, msg.position.x, -msg.position.z]).T

    def waypoint_publisher(self):
        """ executes algorithms and publishes to bridge node """
        rate = rospy.Rate(100)
        self.output.header.frame_id = ""

        while not rospy.is_shutdown():
            self.output.header.stamp = rospy.Time.now()
            self.output.chi_c = self.chi_c
            self.output.h_c = self.h_c
            self.output.Va_c = self.Va_c

            # publish message
            self.pub.publish(self.output)
            self.crosstrack_pub.publish(self.e_crosstrack)

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def run_algorithms(self):
        """ executes waypoint algorithms """
        newpath = 1

        self.wait_for_topics(10)
        self.set_mode("OFFBOARD", 5)
        self.set_arm(True, 5)

        # run dubins algorithm
        while not rospy.is_shutdown():
            try:
                if self.path_follower:
                    flag, r, q, c, rho, lamb, i = self.set_path_follower_params(int(sys.argv[1]))
                else:
                    flag, r, q, c, rho, lamb, i, dp = self.alg.followWppDubins(self.W, self.Chi_waypoint, self.position, self.R, newpath)

                # print out current waypoint (not completely working yet)
                if self.current_waypoint != i:
                    self.current_waypoint = i
                    print("Achieved waypoint " + str(i))

                    # if dubins path
                    if dp:
                        self.dp_list.append(dp)

                # feed dubins output to straight line and orbit follower
                self.e_crosstrack.data, chi_c, h_c = self.alg.pathFollower(flag, r, q, self.position, self.chi, self.chi_inf, self.k_path, c, rho, lamb, self.k_orbit)

                # format chi interval to -pi < chi < pi
                self.chi_c = self.format_chi(chi_c)
                self.h_c = h_c

                if newpath == 1:
                    newpath = 0

            except IndexError:
                break

        # write dubins path parameters to a file
        if self.dp_list:
            self.tab.write_dp(self.dp_list)

    def set_path_follower_params(self, path):
        flag = path
        r = traj.r
        q = traj.q
        c = traj.c
        rho = traj.rho
        lamb = traj.lamb
        i = 0
        return flag, r, q, c, rho, lamb, i

    def format_chi(self, chi_c):
        """ makes sure chi is in the correct interval """
        chi_c = float(chi_c)
        while(chi_c > pi):
            chi_c -= 2 * pi
        while(chi_c < -pi):
            chi_c += 2 * pi
        assert (chi_c >= -pi and chi_c <= pi)
        return chi_c

    def angle(self, v1, v2):
        """ returns the angle between two vectors """
        return np.arctan2(np.linalg.norm(np.cross(v1.T, v2.T)), np.dot(v1.T, v2))


if __name__ == '__main__':
    # set default to dubins paths
    path_follower = False

    # check input arguments
    if len(sys.argv) == 2:
        if sys.argv[1][-5:] == '.plan':
            PLAN_FILE = sys.argv[1]
        elif sys.argv[1] == '1' or sys.argv[1] == '2':
            from path_follower_trajectories import Traj
            traj = Traj()
            path_follower = True
        else:
            print("Invalid input argument. Expecting valid '.plan' file or a path_follower number")
            print("'1' for straight line and '2' for orbit")
            os._exit(1)

    # create node and class instance
    rospy.init_node("waypoint_node")
    uav = UAV()
    uav.setUp(path_follower)
    uav.run_algorithms()
