from copy import deepcopy
import numpy as np


def unit_cv_polar(x_in, y_in, theta_in, speed_in, dt=1):
    """

    :param x_in:
    :param y_in:
    :param theta_in:
    :param speed_in:
    :param dt:
    :return:
    """

    dx = speed_in * np.cos(theta_in) * dt
    dy = speed_in * np.sin(theta_in) * dt

    x_out = x_in + dx
    y_out = y_in + dy

    theta_out = deepcopy(theta_in)
    speed_out = deepcopy(speed_in)

    return x_out, y_out, theta_out, speed_out


def cv_polar(state_v_in, dt=1):
    """

    :param state_v_in:
    :param dt:
    :return:
    """

    x_in, y_in, theta_in, speed_in = (
        state_v_in[0],
        state_v_in[1],
        state_v_in[2],
        state_v_in[3],
    )
    x_out, y_out, theta_out, speed_out = unit_cv_polar(x_in, y_in, theta_in, speed_in, dt)
    state_v_out = np.array([x_out, y_out, theta_out, speed_out])

    return state_v_out


