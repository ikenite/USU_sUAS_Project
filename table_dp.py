import numpy as np
from mat import mat
import matplotlib.pyplot as plt
import rospkg

rospack = rospkg.RosPack()
save_dir = rospack.get_path('sua') + "/grading_scripts/output/path_manager/"


class Table:
    def __init__(self):
        self.L = []
        self.c_s = []
        self.lamb_s = []
        self.c_e = []
        self.lamb_e = []
        self.z_1 = []
        self.q_1 = []
        self.z_2 = []
        self.z_3 = []
        self.q_3 = []
        self.case = []
        self.lengths = []
        self.theta = []
        self.ell = []
        self.c_rs = []
        self.c_ls = []
        self.c_re = []
        self.c_le = []

    def write_dp(self, dp_list):
        fig, ax = plt.subplots()

        # remove duplicate first entry since current self.i solution not working
        # properly. Remove this line one bug is fixed
        dp_list = dp_list[1:]

        for dp in dp_list:
            self.L.append(dp.L)
            self.c_s.append([round(x, 1) for x in dp.c_s])
            self.lamb_s.append(dp.lamb_s)
            self.c_e.append([round(x, 1) for x in dp.c_e])
            self.lamb_e.append(dp.lamb_e)
            self.z_1.append([round(x, 1) for x in dp.z_1])
            self.q_1.append([round(x, 1) for x in dp.q_1])
            self.z_2.append([round(x, 1) for x in dp.z_2])
            self.z_3.append([round(x, 1) for x in dp.z_3])
            self.q_3.append([round(x, 1) for x in dp.q_3])
            self.case.append(dp.case)
            self.lengths.append([round(x, 1) for x in dp.lengths])
            self.theta.append(dp.theta)
            self.ell.append(dp.ell)
            self.c_rs.append([round(x, 1) for x in dp.c_rs])
            self.c_ls.append([round(x, 1) for x in dp.c_ls])
            self.c_re.append([round(x, 1) for x in dp.c_re])
            self.c_le.append([round(x, 1) for x in dp.c_le])

        data = [self.case, self.L, self.c_s, self.lamb_s, self.c_e, self.lamb_e, self.z_1, self.q_1, self.z_2,
                self.z_3, self.q_3, self.lengths, self.theta, self.ell, self.c_rs, self.c_ls, self.c_re, self.c_le]
        label = ("case", "L", "c_s", "lamb_s", "c_e", "lamb_e", "z_1", "q_1", "z_2", "z_3", "q_3", "lengths",
                 "theta", "ell", "c_rs", "c_ls", "c_re", "c_le")
        ax.axis('tight')
        ax.axis('off')

        color = ['xkcd:sky blue'] * 18
        the_table = ax.table(cellText=data, rowLabels=label, rowColours=color, loc='center')
        plt.title('Dubins Parameters')

        plt.savefig(save_dir + 'a_dubins_parameters.png', dpi=300)
