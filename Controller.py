import numpy as np


class BotController:
    def __init__(self, k_P, k_I, k_D):
        self.k_P = k_P
        self.k_I = k_I
        self.k_D = k_D
        self.sum_error = 0.0
        self.time = 0.0
        self.prev_time = 0.0

        self.wheel_radius = 0.08
        self.base_length = 0.258
        self.base_velocity = 0.006
        self.pwm_lower = 70.0
        self.pwm_upper = 210.0
        self.omega_lower = 0.0
        self.omega_upper = 2.35

        self.error = 0.0
        self.prev_error = 0.0
        self.prev_error_1 = 0.0
        self.prev_error_2 = 0.0
        self.last_output = 0.0
        self.T = 0.3
        # Integral part
        self.I = 0.0

    def unicycle_2_differential(self, omega):
        a = 2 * self.base_velocity / self.wheel_radius
        b = self.base_length * omega / self.wheel_radius
        l_w = (a + b) / 2
        r_w = (a - b) / 2

        if (l_w > self.omega_upper) or (r_w > self.omega_upper):
            if l_w > self.omega_upper:
                temp = l_w - self.omega_upper
                l_w = self.omega_upper
                r_w -= temp
                if r_w < self.omega_lower:
                    r_w = self.omega_lower
            else:
                temp = r_w - self.omega_upper
                r_w = self.omega_upper
                l_w -= temp
                if l_w < self.omega_lower:
                    l_w = self.omega_lower
        elif (l_w < self.omega_lower) or (r_w < self.omega_lower):
            if l_w < self.omega_lower:
                temp = self.omega_lower - l_w
                l_w = self.omega_lower
                r_w += temp
                if r_w > self.omega_upper:
                    r_w = self.omega_upper
            else:
                temp = self.omega_lower - r_w
                r_w = self.omega_lower
                l_w += temp
                if l_w > self.omega_upper:
                    l_w = self.omega_upper
        l_w = self.pwm_lower + (((self.pwm_upper - self.pwm_lower) / (self.omega_upper - self.omega_lower)) * l_w)
        r_w = self.pwm_lower + (((self.pwm_upper - self.pwm_lower) / (self.omega_upper - self.omega_lower)) * r_w)

        return int(l_w), int(r_w)

    def pid_control(self, value):
        print value
        self.error = value

        alpha = 2 * self.T * self.k_P + self.k_I * self.T * self.T + 2 * self.k_D
        beta = self.T * self.T * self.k_I - 4 * self.k_D - 2 * self.T * self.k_P
        gamma = 2 * self.k_D

        output = (alpha * self.error + beta * self.prev_error_1 + gamma * self.prev_error_2 + 2 * self.T * self.last_output) / (2 * self.T)
        self.last_output = output
        self.prev_error_2 = self.prev_error_1
        self.prev_error_1 = self.error

        if output >= 255:
            return 255
        elif output <= -255:
            return -255

        return output

    def go_to_angle(self, angle):
        pass
        # self.error = value
        # self.I += self.error
        # signal = (self.error * self.k_P) + (np.fabs(self.error - self.prev_error) * self.k_D * self.T) + (self.I *
        #                                                                                                   self.k_I /
        #                                                                                                   self.T)
        # self.prev_error = self.error
        # if signal >= 255:
        #     return 255
        #
        # return signal

    def reset(self):
        self.error = 0
        self.prev_error_1 = 0
        self.prev_error_2 = 0
        self.last_output = 0