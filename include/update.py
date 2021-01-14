from numpy import array, dot, linalg, vstack, eye, hstack


def compute_update_linear_kalman_gps_pos(state_vector, meas_vector, P_mat, R_mat):
    H_mat = array([[1, 0, 0, 0], [0, 1, 0, 0]])

    # Compute innovation
    inov = vstack(meas_vector) - dot(H_mat, vstack(state_vector))

    # Compute S matrix
    S_mat = dot(H_mat, dot(P_mat, H_mat.T)) + R_mat

    # Compute Kalman gain
    K_mat = dot(P_mat, dot(H_mat.T, linalg.inv(S_mat)))

    # Update state vector
    state_vect_out = vstack(state_vector) + dot(K_mat, inov)

    # Update covariance matrix
    P_mat_out = dot(eye(4) - dot(K_mat, H_mat), P_mat)

    return hstack(state_vect_out), P_mat_out


def compute_update_linear_kalman_gps(state_vector, meas_vector, P_mat, R_mat):
    H_mat = array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Compute innovation
    inov = vstack(meas_vector) - dot(H_mat, vstack(state_vector))

    # Compute S matrix
    S_mat = dot(H_mat, dot(P_mat, H_mat.T)) + R_mat

    # Compute Kalman gain
    K_mat = dot(P_mat, dot(H_mat.T, linalg.inv(S_mat)))

    # Update state vector
    state_vect_out = vstack(state_vector) + dot(K_mat, inov)

    # Update covariance matrix
    P_mat_out = dot(eye(4) - dot(K_mat, H_mat), P_mat)

    return hstack(state_vect_out), P_mat_out
