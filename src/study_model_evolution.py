import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import cv2DataFrame, ctrv2DataFrame


def run_batch_cv_kalman(ax):
    init = np.array([0, 0, np.deg2rad(183), .0])
    accel = pd.read_csv('include/ref_accel.csv')
    gyro = pd.read_csv('include/ref_gyro.csv')
    em_cv = batch_cv_integration(state_init=init,
                                 imu_accel=accel['ref_accel_x (m/s^2)'],
                                 imu_gyro=np.deg2rad(gyro['ref_gyro_z (deg/s)']))
    ax.plot(em_cv['x'], em_cv['y'], '--', label='CV model')
    return None


def run_batch_ctrv_kalman(ax):
    init = np.array([0, 0, np.deg2rad(183), .0])
    accel = pd.read_csv('../data/simu_dataset_1/ref_accel.csv')
    gyro = pd.read_csv('../data/simu_dataset_1/ref_gyro.csv')
    em_cv = batch_ctrv_integration(state_init=init,
                                   imu_accel=accel['ref_accel_x (m/s^2)'],
                                   imu_gyro=np.deg2rad(gyro['ref_gyro_z (deg/s)']))
    ax.plot(em_cv['x'], em_cv['y'], marker='o', linestyle='--', fillstyle='none', label='CTRV model')
    return None


def main():
    fig, ax = plt.subplots()
    plot_ref_gps(ax=ax)
    run_batch_cv_kalman(ax=ax)
    run_batch_ctrv_kalman(ax=ax)
    ax.legend()
    plt.show()
    return None
