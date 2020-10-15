import time
import numpy as np
from numpy import pi
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from mpl_toolkits import mplot3d


def MeasurementOutput_two(x):
    # x is the current robot position
    # output y is the array described in Kalman_filter_one
    sigrel = 0.05
    sigb = 0.12

    beaconrefs = np.array([[4.5, 3], [-4.5, 3], [4.5, -3], [-4.5, -3]])
    y = np.zeros((4, 2))
    for s in range(4):
        pos = x[0:2]
        zmu = beaconrefs[s] - pos
        dist = np.linalg.norm(zmu)
        direc = np.arctan2(zmu[0], zmu[1])
        yabs = np.array([dist, direc]) + np.array(
            [np.random.normal(0, sigrel) * dist, np.random.normal(0, sigb) * direc])
        y[s, 0] = yabs[0]
        y[s, 1] = yabs[1] - x[2]
    return y


def jacobian(f, x, rows=None, dx=1e-3):
    rows = rows if rows is not None else len(x)
    J = np.empty((rows, len(x)))
    xpdx, xmdx = np.empty_like(x), np.empty_like(x)
    for j in range(len(x)):
        xpdx[:] = xmdx[:] = x
        xpdx[j] += dx
        xmdx[j] -= dx
        f_j = f(xpdx) - f(xmdx)
        J[:, j] = f_j / (2 * dx)
    return J


def ang(theta):
    return (theta + pi) % (2 * pi) - pi


def predict(f, x, u, P, Q):
    F = jacobian(lambda x_: f(x_, u), x)
    x = f(x, u)
    P = F @ P @ F.T + Q
    return x, P


def update(x, P, R, h, z):
    H = jacobian(h, x, len(z))
    y = z - h(x)
    y[1] = ang(y[1])
    S = (H @ P @ H.T) + R
    K = P @ H.T @ np.linalg.inv(S)
    x += K @ y
    P = (np.eye(3) - K @ H) @ P
    return x, P


def Kalman_filter_two(muold, Sigmaold, ut, yt):
    # muold should contain position and orientation estimate at previous time (vector of length 3)
    # Sigmaold should contain covariance matrix at previous time
    # ut contains the input command at time t: first component is 0 or 1, second component is a real value
    # yt contains the measurement data (4 x 2 array: distance, direction for each of the four beacons)
    #     see just below the order of the beacons:
    beaconrefs = np.array([[4.5, 3], [-4.5, 3], [4.5, -3], [-4.5, -3]])
    # motion standard devs  (you can change these to play around -- do so also in virtual motion generator)
    sigth0 = 0.1
    sigx0 = 0.05
    sigth1 = 0.02
    sigth2 = 0.02
    sigv1 = 0.03
    # measurement standard devs  (you can change these to play around -- do so also in virtual meas. generator)
    sigrel = 0.05
    sigb = 0.12
    # HERE THE SOLUTIONSigmaold
    if ut[0] == 0:
        f = lambda x, u: np.array([x[0], x[1], ang(x[2] + u)])
        if ut[1] < 0.001:
            Q = np.zeros((3, 3))
        else:
            Q = np.array([[sigx0, 0, 0],
                          [0, sigx0, 0],
                          [0, 0, sigth0]])
    else:
        f = lambda x, u: x + np.array([np.cos(x[2]), np.sin(x[2]), 0]) * u
        Q = np.array([[np.cos(muold[2]) * sigv1, 0, 0],
                      [0, np.sin(muold[2]) * sigv1, 0],
                      [0, 0, sigth1 + sigth2]])

    mubar, Sigmabar = predict(f, muold, ut[1], Sigmaold, Q)
    R = np.array([[sigrel, 0], [0, sigb]])

    for s in range(4):
        h = lambda x: np.array([np.linalg.norm(beaconrefs[s] - x[:2]),
                                np.arctan2(*(beaconrefs[s] - x[:2])) - x[2]])
        mubar, Sigmabar = update(mubar, Sigmabar, R, h, yt[s])

    return muold, Sigmabar


def MotionGeneration_one(xold, ut):
    # xold contains x1,x2 and theta at previous time, in this order
    # ut contains the input command at time t: first component is 0 or 1, second component is a real value
    # output contains x1,x2 and theta after applying ut
    signth = 0.08
    signx0 = 0.03
    signth1 = 0.08
    signth2 = 0.08
    signv1 = 0.08
    x = xold
    if ut[1] < 0.001:
        # case of not motion command
        x = xold
    elif ut[0] < 0.01:
        # case of rotation command
        x[0] = xold[0] + np.random.normal(0, signx0)
        x[1] = xold[1] + np.random.normal(0, signx0)
        x[2] = xold[2] + (1 + np.random.normal(0, signth)) * ut[1]
    else:
        # case of translation command
        nth1 = np.random.normal(0, signth1)
        nth2 = np.random.normal(0, signth2)
        trans = (1 + np.random.normal(0, signv1)) * ut[1]
        x[0] = xold[0] + trans * np.cos(xold[2] + nth1)
        x[1] = xold[1] + trans * np.sin(xold[2] + nth1)
        x[2] = xold[2] + nth1 + nth2
    return x


## This simulation & visualization program includes plotting the orientation of the robot
#  You are encouraged to copy-paste it and adapt it in further cells.

# initial true pose, estimated pose, and uncertainty on initial estimation (this last one be big)
x = np.array([0, 0, 0])
mu = np.array([0, 0, 0])
Sig = 100 * np.identity(3)

# sequence of input commands (change this as you want. NB: now first component is 0 or 1)
u_seq = np.array([[0, 0.5], [0, 0.5], [1, 0.5], [1, 0.5], [0, 1.5], [1, 0.5], [1, 0.5]])

# buffers for the graphs
xonetrue = []
xtwotrue = []
thetatrue = []
xoneestimate = []
xtwoestimate = []
thetaestimate = []
Ellh = []
Ellv = []
Ellangle = []
xonetrue.append(x[0])
xtwotrue.append(x[1])
thetatrue.append(x[2])
xoneestimate.append(mu[0])
xtwoestimate.append(mu[1])
thetaestimate.append(mu[2])
PosSig = Sig[0:2, 0:2]
w, v = np.linalg.eigh(PosSig)
Ellh.append(6 * w[0]);
Ellv.append(6 * w[1]);
Ellangle.append(np.arctan2(v[1, 0], v[0, 0]))

# Running the simulation
for t in range(len(u_seq)):
    muold = mu
    Sigold = Sig
    xold = x
    unow = u_seq[t]
    # 1. Real Robot simulation
    x = MotionGeneration_one(xold, unow)
    xonetrue.append(x[0])
    xtwotrue.append(x[1])
    thetatrue.append(x[2])
    # !!!! change next line when using the more realistic sensor model of Ex.2
    ynow = MeasurementOutput_two(x)
    # 2. Kalman Filter estimate
    # !!!! change next line when using the EKF of Ex.2
    mu = Kalman_filter_two(muold, Sigold, unow, ynow)[0]
    xoneestimate.append(mu[0])
    xtwoestimate.append(mu[1])
    thetaestimate.append(mu[2])
    # !!!! change next line when using the EKF of Ex.2
    Sig = Kalman_filter_two(muold, Sigold, unow, ynow)[1]
    PosSig = Sig[0:2, 0:2]
    w, v = np.linalg.eigh(PosSig)
    Ellh.append(6 * w[0]);
    Ellv.append(6 * w[1]);
    Ellangle.append(np.arctan2(v[1, 0], v[0, 0]))

# The following code plots the result of the simulation.
# True robot trajectory is represented as connected blue circles,
# Kalman filter estimate as red crosses with uncertainty ellipse (except at time zero -- too big),
# beacons are green crosses.
# orientation is represented with a small line, in blue or red respectively

fig, axnstd = plt.subplots(figsize=(10, 6))
pl.plot([-4.5, 4.5, -4.5, 4.5], [-3, -3, 3, 3], 'g+')
pl.plot(xonetrue, xtwotrue, 'bo')
pl.plot(xonetrue, xtwotrue, 'b-')
pl.plot(xoneestimate, xtwoestimate, 'rx')
axnstd.set_aspect("equal")

for t in range(len(u_seq)):
    zelinex = np.array([xonetrue[t + 1], xonetrue[t + 1] + 0.3 * np.cos(thetatrue[t + 1])])
    zeliney = np.array([xtwotrue[t + 1], xtwotrue[t + 1] + 0.3 * np.sin(thetatrue[t + 1])])
    pl.plot(zelinex, zeliney, 'b-')
    zelinex = np.array([xoneestimate[t + 1], xoneestimate[t + 1] + 0.3 * np.cos(thetaestimate[t + 1])])
    zeliney = np.array([xtwoestimate[t + 1], xtwoestimate[t + 1] + 0.3 * np.sin(thetaestimate[t + 1])])
    pl.plot(zelinex, zeliney, 'r-')
    e1 = Ellipse([xoneestimate[t + 1], xtwoestimate[t + 1]], Ellh[t + 1], Ellv[t + 1], Ellangle[t + 1], linewidth=2,
                 fill=False)
    axnstd.add_patch(e1)

plt.show()