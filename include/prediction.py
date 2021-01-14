from numpy import array, dot, vstack, hstack
from math import sin, cos


def compute_F_matrix_linear_kalman(dt=1):
    return array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])


def compute_command_vector_linear_kalman(spd, theta):
    vx = spd * cos(theta)
    vy = spd * sin(theta)
    return vstack([0, 0, vx, vy])


def compute_prediction_linear_kalman(state_vector, P_mat, Q_mat, dt):
    state_in = vstack(state_vector)
    F_mat = compute_F_matrix_linear_kalman(dt=dt)
    state_out = dot(F_mat, state_in)
    cov_mat_out = dot(dot(F_mat, P_mat), F_mat.T) + Q_mat

    return hstack(state_out), cov_mat_out


def compute_prediction_linear_kalman_with_imu(state_vector, P_mat, Q_mat, spd, theta, dt):
    state_in = vstack(state_vector)
    F_mat = compute_F_matrix_linear_kalman(dt=dt)
    U_vect = compute_command_vector_linear_kalman(spd=spd, theta=theta)
    state_out = dot(F_mat, state_in) + U_vect
    cov_mat_out = dot(dot(F_mat, P_mat), F_mat.T) + Q_mat

    return hstack(state_out), cov_mat_out
