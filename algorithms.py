import numpy as np
from numpy import linalg as LA
from mat import mat
from utils import in_half_plane, s_norm, Rz, angle, i2p
import math


class Algorithms:
    def __init__(self):
        self.i = 0
        self.state = 0

    def pathFollower(self, flag, r, q, p, chi, chi_inf,
                     k_path, c, rho, lamb, k_orbit):
        """
        Input:
            flag = 1 for straight line, 2 for orbit
            r = origin of straight-line path in NED (m)
            q = direction of straight-line path in NED (m)
            p = current position of uav in NED (m)
            chi = course angle of UAV (rad)
            chi_inf = straight line path following parameter
            k_path = straight line path following parameter
            c = center of orbit in NED (m)
            rho = radius of orbit (m)
            lamb = direction of orbit, 1 clockwise, -1 counter-clockwise
            k_orbit = orbit path following parameter

        Outputs:
            e_crosstrack = crosstrack error (m)
            chi_c = commanded course angle (rad)
            h_c = commanded altitude (m)

        Example Usage
            e_crosstrack, chi_c, h_c = pathFollower(path)

        Reference: Beard, Small Unmanned Aircraft,
                   Chapter 10, Algorithms 3 and 4
        Copyright 2018 Utah State University
        """

        # Unpack universal variables
        P = np.squeeze(np.asarray(p))
        p_n, p_e, p_d = P[0], P[1], P[2]

        if flag == 1:  # straight line

            # Unpack line path variables
            Q = np.squeeze(np.asarray(q))
            q_n, q_e, q_d = Q[0], Q[1], Q[2]
            R = np.squeeze(np.asarray(r))
            r_n, r_e, r_d = R[0], R[1], R[2]

            # Calculations for h_c
            epi = [(p_n - r_n), (p_e - r_e), (p_d - r_d)]
            EPI = np.squeeze(np.asarray(epi))
            k = np.array([0, 0, 1])  # Unit down vector (NED Inertial Frame)

            # ... Unit vector normal to q-k plane
            n = np.cross(Q, k)/LA.norm(np.cross(Q, k))

            # ... Position vector of MAV projected to q-k plane
            s_i = epi - np.dot(EPI, n)*n

            # ... Unpack s_i vector
            s_n, s_e, s_d = s_i[0], s_i[1], s_i[2]

            # ... Equation 10.5
            h_c = -r_d + (math.sqrt(s_n**2 + s_e**2)*q_d /
                          math.sqrt(q_n**2 + q_e**2))

            # Algorithm 3
            chi_q = math.atan2(q_e, q_n)
            while (chi_q - chi) < -math.pi:
                chi_q = chi_q + 2*math.pi
            while (chi_q - chi) > math.pi:
                chi_q = chi_q - 2*math.pi
            e_crosstrack = (-math.sin(chi_q)*(p_n - r_n) +
                            math.cos(chi_q)*(p_e - r_e))
            chi_c = chi_q - chi_inf*(2/math.pi)*math.atan(k_path*e_crosstrack)

        elif flag == 2:  # orbit following

            # Unpack orbit path variables
            C = np.squeeze(np.asarray(c))
            c_n, c_e, c_d = C[0], C[1], C[2]

            # Algorithm 4
            h_c = -c_d
            d = math.sqrt((p_n - c_n)**2 + (p_e - c_e)**2)
            phi = math.atan2((p_e - c_e), (p_n - c_n))
            while (phi-chi) < -math.pi:
                phi = phi + 2*math.pi
            while (phi-chi) > math.pi:
                phi = phi - 2*math.pi
            chi_c = phi + lamb*((math.pi/2) + math.atan(k_orbit*((d-rho)/rho)))
            e_crosstrack = d - rho

        else:
            raise Exception("Invalid path type")

        return e_crosstrack, chi_c, h_c

    # followWpp algorithm left here for reference
    # It is not used in the final implementation
    def followWpp(self, w, p, newpath):
        """
        followWpp implements waypoint following via connected straight-line
        paths.

        Inputs:
            w = 3xn matrix of waypoints in NED (m)
            p = position of MAV in NED (m)
            newpath = flag to initialize the algorithm or define new waypoints

        Outputs
            r = origin of straight-line path in NED (m)
            q = direction of straight-line path in NED (m)

        Example Usage;
            r, q = followWpp(w, p, newpath)

        Reference: Beard, Small Unmanned Aircraft, Chapter 11, Algorithm 5
        Copyright 2018 Utah State University
        """

        if self.i is None:
            self.i = 0

        if newpath:
            # initialize index
            self.i = 1

        # check sizes
        m, N = w.shape
        assert N >= 3
        assert m == 3

        # calculate the q vector
        r = w[:, self.i - 1]
        qi1 = s_norm(w[:, self.i], -w[:, self.i - 1])

        # Calculate the origin of the current path
        qi = s_norm(w[:, self.i + 1], -w[:, self.i])

        # Calculate the unit normal to define the half plane
        ni = s_norm(qi1, qi)

        # Check if the MAV has crossed the half-plane
        if in_half_plane(p, w[:, self.i], ni):
            if self.i < (N - 2):
                self.i += 1
        q = qi1

        return r, q

    # followWppFillet algorithm left here for reference.
    # It is not used in the final implementation
    def followWppFillet(self, w, p, R, newpath):
        """
        followWppFillet implements waypoint following via straightline paths
        connected by fillets

        Inputs:
            W = 3xn matrix of waypoints in NED (m)
            p = position of MAV in NED (m)
            R = fillet radius (m)
            newpath = flag to initialize the algorithm or define new waypoints

        Outputs
            flag = flag for straight line path (1) or orbit (2)
            r = origin of straight-line path in NED (m)
            q = direction of straight-line path in NED (m)
            c = center of orbit in NED (m)
            rho = radius of orbit (m)
            lamb = direction or orbit, 1 clockwise, -1 counter clockwise

        Example Usage
            [flag, r, q, c, rho, lamb] = followWppFillet( w, p, R, newpath )

        Reference: Beard, Small Unmanned Aircraft, Chapter 11, Algorithm 6
        Copyright 2018 Utah State University
        """

        if self.i is None:
            self.i = 0
            self.state = 0
        if newpath:
            # Initialize the waypoint index
            self.i = 2
            self.state = 1

            # Check size of waypoints matrix
            m, N = w.shape  # 'N' is number of waypoints and 'm' dimensions
            assert N >= 3
            assert m == 3
        else:
            [m, N] = w.shape
            assert N >= 3
            assert m == 3
        # Calculate the q vector and fillet angle
        qi1 = mat(s_norm(w[:, self.i], -w[:, self.i - 1]))
        qi = mat(s_norm(w[:, self.i + 1], -w[:, self.i]))
        e = acos(-qi1.T * qi)

        # Determine if the MAV is on a straight or orbit path
        if self.state == 1:
            c = mat([0, 0, 0]).T
            rho = 0
            lamb = 0

            flag = 1
            r = w[:, self.i - 1]
            q = q1
            z = w[:, self.i] - (R / (np.tan(e / 2))) * qi1
            if in_half_plane(p, z, qi1):
                self.state = 2

        elif self.state == 2:
            r = [0, 0, 0]
            q = [0, 0, 0]

            flag = 2
            c = w[:, self.i] - (R / (np.sin(e / 2))) * s_norm(qi1, -qi)
            rho = R
            lamb = np.sign(qi1(1) * qi(2) - qi1(2) * qi(1))
            z = w[:, self.i] + (R / (np.tan(e / 2))) * qi

            if in_half_plane(p, z, qi):
                if self.i < (N - 1):
                    self.i = self.i + 1
                self.state = 1

        else:
            # Fly north as default
            flag = -1
            r = p
            q = mat([1, 0, 0]).T
            c = np.nan(3, 1)
            rho = np.nan
            lamb = np.nan

        return flag, r, q, c, rho, lamb

    def findDubinsParameters(self, p_s, chi_s, p_e, chi_e, R):
        """
        findDubinsParameters determines the dubins path parameters

        Inputs:
        p_s = start position (m)
        chi_s = start course angle (rad)
        p_e = end position (m)
        chi_e = end course angle (rad)
        R = turn radius (m)

        Outputs
        dp.L = path length (m)
        dp.c_s = start circle origin (m)
        dp.lamb_s = start circle direction (unitless)
        dp.c_e = end circle origin (m)
        dp.lamb_e = end circle direction (unitless)
        dp.z_1 = Half-plane H_1 location (m)
        dp.q_12 = Half-planes H_1 and H_2 unit normals (unitless)
        dp.z_2 = Half-plane H_2 location (m)
        dp.z_3 = Half-plane H_3 location  (m)
        dp.q_3 = Half-plane H_3 unit normal (m)
        dp.case = case (unitless)

        Example Usage
        dp = findDubinsParameters( p_s, chi_s, p_e, chi_e, R )

        Reference: Beard, Small Unmanned Aircraft, Chapter 11, Algorithm 7
        Copyright 2018 Utah State University
        """

        # Unpack variables
        e_s = mat([[1], [0], [0]]).T
        P_s = np.squeeze(np.asarray(p_s))
        p_s_n, p_s_e, p_s_d = P_s[0], P_s[1], P_s[2]
        P_e = np.squeeze(np.asarray(p_e))
        p_e_n, p_e_e, p_e_d = P_e[0], P_e[1], P_e[2]

        # Algorithm 7

        # Require ||p_s - p_e|| >= 3R
        R_check = np.array([p_s_n-p_e_n, p_s_e-p_e_e])
        assert LA.norm(R_check) >= 3*R

        pi = math.pi

        # Compute circle centers
        c_chi_s = math.cos(chi_s)
        s_chi_s = math.sin(chi_s)
        c_chi_e = math.cos(chi_e)
        s_chi_e = math.sin(chi_e)
        c_rs = p_s + R*Rz(pi/2)*mat([[c_chi_s], [s_chi_s], [0]]).T
        c_ls = p_s + R*Rz(-pi/2)*mat([[c_chi_s], [s_chi_s], [0]]).T
        c_re = p_e + R*Rz(pi/2)*mat([[c_chi_e], [s_chi_e], [0]]).T
        c_le = p_e + R*Rz(-pi/2)*mat([[c_chi_e], [s_chi_e], [0]]).T

        # ... convert to arrays...
        C_rs = np.squeeze(np.asarray(c_rs))
        C_ls = np.squeeze(np.asarray(c_ls))
        C_re = np.squeeze(np.asarray(c_re))
        C_le = np.squeeze(np.asarray(c_le))

        # Compute path lengths

        # ... Case 1: R-S-R
        th = np.angle(C_le - C_rs)
        L1 = (LA.norm(C_rs - C_re)
              + R*(2*pi + (th - pi/2) % (2*pi)
              - (chi_s - pi/2) % (2*pi)) % (2*pi)
              + R*(2*pi + (chi_e - pi/2) % (2*pi)
              - (th - pi/2) % (2*pi)) % (2*pi))

        # ... Case 2: R-S-L
        th = np.angle(C_le - C_rs)
        ell = LA.norm(C_le - C_rs)
        th2 = th - pi/2 + math.asin(2*R/ell)
        if ~np.isreal(th2):
            L2 = math.nan
        else:
            L2 = (math.sqrt(ell**2 - 4*(R**2))
                  + R*(2*pi + th2 % (2*pi)
                       - (chi_s - pi/2) % (2*pi)) % (2*pi)
                  + R*(2*pi + (th2 + pi) % (2*pi)
                       - (chi_e + pi/2) % (2*pi)) % (2*pi))

        # ... Case 3: L-S-R
        th = np.angle(C_re - C_ls)
        ell = np.norm(C_re - C_ls)
        th2 = math.acos(2*R/ell)
        if ~np.isreal(th2):
            L3 = math.nan
        else:
            L3 = (math.sqrt(ell**2 - 4*(R**2))
                  + R*(2*pi + (chi_s + pi/2) % (2*pi)
                       - (th + th2) % (2*pi)) % (2*pi)
                  + R*(2*pi + (chi_e - pi/2) % (2*pi)
                       - (th + th2 - pi/2) % (2*pi)) % (2*pi))

        # ... Case 4: L-S-L
        th = np.angle(C_le - C_ls)
        L4 = (np.norm(C_ls - C_le)
              + R*(2*pi + (chi_s + pi/2) % (2*pi)
                   - (th + pi/2) % (2*pi)) % (2*pi)
              + R*(2*pi + (th + pi/2) % (2*pi)
                   - (chi_e + pi/2) % (2*pi)) % (2*pi))

        # Define parameters for minimum length path

        c_s = mat([[math.nan], [math.nan], [math.nan]]).T
        lamb_s = math.nan
        c_e = mat([[math.nan], [math.nan], [math.nan]]).T
        lamb_e = math.nan
        q_1 = mat([[math.nan], [math.nan], [math.nan]]).T
        z_1 = mat([[math.nan], [math.nan], [math.nan]]).T
        z_2 = mat([[math.nan], [math.nan], [math.nan]]).T

        L_list = [L1, L2, L3, L4]
        L_min = min(L_list)

        if L_min == L1:
            i_min = 1
            c_s = c_rs
            lamb_s = 1
            c_e = c_re
            lamb_e = 1
            q_1 = (c_e - c_s)/np.norm(C_re - C_rs)
            z_1 = c_s + R*Rz(-pi/2)*q_1
            z_2 = c_e + R*Rz(-pi/2)*q_1

        elif L_min == L2:
            i_min = 2
            c_s = c_rs
            lamb_s = 1
            c_e = c_le
            lamb_e = -1
            ell = np.norm(C_le - C_rs)
            th = np.angle(C_le - C_rs)
            th2 = th - pi/2 + math.asin(2*R/ell)
            q_1 = Rz(th2+pi/2)*e_s
            z_1 = c_s + R*Rz(th2)*e_s
            z_2 = c_e + R*Rz(th2 + pi)*e_s

        elif L_min == L3:
            i_min = 3
            c_s = c_ls
            lamb_s = -1
            c_e = c_re
            lamb_e = 1
            ell = np.norm(C_re - C_ls)
            th = np.angle(C_re - C_ls)
            th2 = math.acos(2*R/ell)
            q_1 = Rz(th+th2-pi/2)*e_s
            z_1 = c_s + R*Rz(th + th2)*e_s
            z_2 = c_e + R*Rz(th + th2 - pi)*e_s

        elif L_min == L4:
            i_min = 4
            c_s = c_ls
            lamb_s = -1
            c_e = c_le
            lamb_e = -1
            q_1 = (c_e - c_s)/np.norm(C_le - C_ls)
            z_1 = c_s + R*Rz(pi/2)*q_1
            z_2 = c_e + R*Rz(pi/2)*q_1

        z_3 = p_e
        q_3 = Rz(chi_e)*e_s

        # package output into DubinsParameters class
        dp = DubinsParameters()

        dp.L = L_min
        dp.c_s = c_s
        dp.lamb_s = lamb_s
        dp.c_e = c_e
        dp.lamb_e = lamb_e
        dp.z_1 = z_1
        dp.q_1 = q_1
        dp.z_2 = z_2
        dp.z_3 = z_3
        dp.q_3 = q_3
        dp.case = i_min
        dp.lengths = np.array([[L1, L2, L3, L4]])
        dp.theta = th
        dp.ell = ell
        dp.c_rs = c_rs
        dp.c_ls = c_ls
        dp.c_re = c_re
        dp.c_le = c_le

        return dp

    def followWppDubins(self, W, Chi, p, R, newpath):
        """
        followWppDubins implements waypoint following via Dubins paths

        Inputs:
            W = list of waypoints in NED (m)
            Chi = list of course angles at waypoints in NED (rad)
            p = mav position in NED (m)
            R = fillet radius (m)
            newpath = flag to initialize the algorithm or define new waypoints

        Outputs
            flag = flag for straight line path (1) or orbit (2)
            r = origin of straight-line path in NED (m)
            q = direction of straight-line path in NED (m)
            c = center of orbit in NED (m)
            rho = radius of orbit (m)
            lamb = direction or orbit, 1 clockwise, -1 counter clockwise
            self.i = waypoint number
            dp = dubins path parameters

        Example Usage
            flag, r, q, c, rho, lamb = followWppDubins(W, Chi, p, R, newpath)

        Reference: Beard, Small Unmanned Aircraft, Chapter 11, Algorithm 8
        Copyright 2018 Utah State University
        """

        if self.i is None:
            self.i = 0
            self.state = 0
        if newpath:
            # Initialize the waypoint index
            self.i = 2
            self.state = 1

            # Check size of waypoints matrix
            m, N = W.shape  # 'N' is number of waypoints and 'm' dimensions
            assert N >= 3
            assert m == 3
        else:
            [m, N] = W.shape
            assert N >= 3
            assert m == 3

        # Determine the Dubins path parameters
        dp = findDubinsParameters(W[:, self.i - 1], Chi[:, self.i - 1],
                                  W[:, self.i], Chi[:, self.i], R)

        # ... Follow start orbit until on the correct side of H1
        if self.state == 1:
            flag = 2
            c = dp.c_s
            rho = R
            lamb = dp.lamb_s
            r = mat([[math.nan], [math.nan], [math.nan]]).T
            q = mat([[math.nan], [math.nan], [math.nan]]).T
            if in_half_plane(p, dp.z_1, -dp.q_1):
                self.state = 2

        # ... Continue following the start orbit until in H1
        elif self.state == 2:
            flag = 2
            c = dp.c_s
            rho = R
            lamb = dp.lamb_s
            r = mat([[math.nan], [math.nan], [math.nan]]).T
            q = mat([[math.nan], [math.nan], [math.nan]]).T
            if in_half_plane(p, dp.z_1, dp.q_1):
                self.state = 3

        # ... Transition to straight-line path until H2
        elif self.state == 3:
            flag = 1
            r = dp.z_1
            q = dp.q_1
            c = mat([[math.nan], [math.nan], [math.nan]]).T
            lamb = math.nan
            rho = math.nan
            if in_half_plane(p, dp.z_2, dp.q_1):
                self.state = 4

        # ... Follow the end orbit until on the correct side of H3
        elif self.state == 4:
            flag = 2
            c = dp.c_e
            rho = R
            lamb = dp.lamb_e
            r = mat([[math.nan], [math.nan], [math.nan]]).T
            q = mat([[math.nan], [math.nan], [math.nan]]).T
            if in_half_plane(dp, dp.z_3, -dp.q_3):
                self.state = 5

        # state = 5
        else:
            flag = 2
            c = dp.c_e
            rho = R
            lamb = dp.lamb_e
            r = mat([[math.nan], [math.nan], [math.nan]]).T
            q = mat([[math.nan], [math.nan], [math.nan]]).T
            # ... Continue following the end oribit until in H3
            if in_half_plane(p, dp.z_3, dp.q_3):
                self.state = 1
                if(self.i < N):
                    self.i = self.i + 1
                dp = findDubinsParameters(W[:, self.i - 1], Chi[:, self.i - 1],
                                          W[:, self.i], Chi[:, self.i], R)

        return flag, r, q, c, rho, lamb, self.i, dp


class DubinsParameters:
    def __init__(self):
        """
        Member Variables:
            L = path length (m)
            c_s = start circle origin (m)
            lamb_s = start circle direction (unitless)
            c_e = end circle origin (m)
            lamb_e = end circle direction (unitless)
            z_1 = Half-plane H_1 location (m)
            q_1 = Half-planes H_1 and H_2 unit normals (unitless)
            z_2 = Half-plane H_2 location (m)
            z_3 = Half-plane H_3 location  (m)
            q_3 = Half-plane H_3 unit normal (m)
            case = case (unitless)
        """

        self.L = 0
        self.c_s = mat([0, 0, 0]).T
        self.lamb_s = -1
        self.c_e = mat([0, 0, 0]).T
        self.lamb_e = 1
        self.z_1 = mat([0, 0, 0]).T
        self.q_1 = mat([0, 0, 0]).T
        self.z_2 = mat([0, 0, 0]).T
        self.z_3 = mat([0, 0, 0]).T
        self.q_3 = mat([0, 0, 0]).T
        self.case = 0
        self.lengths = np.array([[0, 0, 0, 0]])
        self.theta = 0
        self.ell = 0
        self.c_rs = mat([0, 0, 0]).T
        self.c_ls = mat([0, 0, 0]).T
        self.c_re = mat([0, 0, 0]).T
        self.c_le = mat([0, 0, 0]).T
