from mat import mat


class Traj:
    def __init__(self):
        # self.r = mat([-200, 0, -30]).T
        # self.q = mat([0.7044, 0.7044, -0.0872]).T  # must be unit vector
        # self.c = mat([150, 150, -40]).T
        # self.rho = 50
        # self.lamb = -1

        self.r = mat([-100, 0, -40]).T
        self.q = mat([0, 1, 0]).T  # must be unit vector
        self.c = mat([-150, 150, -40]).T
        self.rho = 50
        self.lamb = 1
