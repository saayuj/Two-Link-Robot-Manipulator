# importing required libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# defining a few constants
g = 9.81
t1 = np.linspace(0, 40, 1001)


### change variable values here ###
m1 = 1
m2 = 1
l1 = 1
l2 = 1
c6 = m2 * (l2 ** 2) / 3
Kp = 100
Ki = 100
Kd = 0.1
###################################


# implementing a pid controller to follow trajectory
def dSdt(t, S):
    theta1, w1, theta2, w2 = S

    theta1_given = np.cos(t)
    theta2_given = np.sin(t)
    w1_given = - np.sin(t)
    w2_given = np.cos(t)
    alpha1_given = - np.cos(t)
    alpha2_given = - np.sin(t)

    return [w1,
            (Kd * alpha1_given + Kp * w1_given + Ki * theta1_given - (Kp + 1) * w1 - Ki * theta1) / Kd,
            w2,
            (Kd * alpha2_given + Kp * w2_given + Ki * theta2_given - (Kp + 1) * w2 - Ki * theta2) / Kd]

sol = odeint(dSdt, y0=[0, 0, 0, 0], t=t1, tfirst=True)

theta1 = sol.T[0]
w1 = sol.T[1]
theta2 = sol.T[2]
w2 = sol.T[3]

theta1_given = np.cos(t1)
theta2_given = np.sin(t1)
w1_given = - np.sin(t1)
w2_given = np.cos(t1)
alpha1_given = - np.cos(t1)
alpha2_given = - np.sin(t1)

alpha1 = (Kd * alpha1_given + Kp * w1_given + Ki * theta1_given - (Kp + 1) * w1 - Ki * theta1) / Kd
alpha2 = (Kd * alpha2_given + Kp * w2_given + Ki * theta2_given - (Kp + 1) * w2 - Ki * theta2) / Kd


# function to find torques at each instant
def find_torques(m1, m2, l1, l2, theta1, theta2, w1, w2, alpha1, alpha2):
    c1 = m1 * (l1 ** 2) / 3 + m2 * (l1 ** 2) + m2 * (l2 ** 2) / 3 + m2 * l1 * l2 * np.cos(theta2)
    c2 = m2 * (l2 ** 2) / 3 + m2 * l1 * l2 * np.cos(theta2) / 2
    c3 = - m2 * l1 * l2 * np.sin(theta2) / 2
    c4 = (m1 / 2 + m2) * g * l1 * np.cos(theta1)
    c5 = m2 * g * l2 * np.cos(theta1 + theta2) / 2

    tau1 = c1 * alpha1 + c2 * alpha2 + 2 * c3 * w1 * w2 + c3 * (w2 ** 2) + c4 + c5
    tau2 = c2 * alpha1 + c6 * alpha2 - c3 * (w1 ** 2) + c5

    return (tau1, tau2)

tau1, tau2 = find_torques(m1, m2, l1, l2, theta1, theta2, w1, w2, alpha1, alpha2)


# obtaining the coordinates of the endpoints of the links
def get_coordinates(t, theta1, theta2, l1, l2):
    return (l1 * np.cos(theta1),
            l1 * np.sin(theta1),
            l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2),
            l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2))

x1, y1, x2, y2 = get_coordinates(t1, theta1, theta2, l1, l2)


# creating a GIF
def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])

fig, ax = plt.subplots(1,1, figsize=(8,8))
ln1, = plt.plot([], [], 'b-', lw=3)
ax.set_ylim(-4,4)
ax.set_xlim(-4,4)
ani = FuncAnimation(fig, animate, frames=1000, interval=50)
ani.save('./Two Link Planar Robot Manipulator/Trajectory/two_link_robot_manipulator_trajectory.gif', writer='pillow', fps=15)


# plotting the torques required as functions of time
plt.clf()
plt.plot(t1, tau1, label='Torque 1')
plt.plot(t1, tau2, label='Torque 2')
plt.xlabel('Time')
plt.ylabel('Torque')
plt.title('Torques vs Time')
plt.legend()
plt.savefig('./Two Link Planar Robot Manipulator/Trajectory/torques_required.png')


# plotting actual and expected trajectories
plt.clf()
plt.plot(t1, theta1, 'b-', label='actual')
plt.plot(t1, theta1_given, 'r--', label='expected')
plt.xlabel('Time')
plt.ylabel('Theta')
plt.title('Theta 1 vs Time')
plt.legend()
plt.savefig('./Two Link Planar Robot Manipulator/Trajectory/theta1_comparison.png')

plt.clf()
plt.plot(t1, theta2, 'b-', label='actual')
plt.plot(t1, theta2_given, 'r--', label='expected')
plt.xlabel('Time')
plt.ylabel('Theta')
plt.title('Theta 2 vs Time')
plt.legend()
plt.savefig('./Two Link Planar Robot Manipulator/Trajectory/theta2_comparison.png')


# plotting total energy by KE+PE method
k1 = m1 * (l1 ** 2) / 6 + m2 * (l2 ** 2) / 6 + m2 * (l1 ** 2) / 2 + m2 * l1 * l2 * np.cos(theta2) / 2
k2 = m2 * (l2 ** 2) / 3 + m2 * l1 * l2 * np.cos(theta2) / 2
KE = k1 * (w1 ** 2) + c6 * (w2 ** 2) / 2 + k2 * w1 * w2

PE = (m1 / 2 + m2) * g * l1 * np.sin(theta1) + m2 * g * l2 * np.sin(theta1 + theta2) / 2

TE1 = KE + PE

plt.clf()
plt.plot(t1, TE1)
plt.xlabel('Time')
plt.ylabel('Total Energy')
plt.title('Total Energy (KE+PE) vs Time')
plt.savefig('./Two Link Planar Robot Manipulator/Trajectory/total_energy_KE-PE.png')


# plotting total energy by power integral method
tw1 = tau1 * w1
tw2 = tau2 * w2

TE2 = np.zeros(1001)
for i in range(1001):
    TE2[i] = np.trapz(tw1[:i+1], x=t1[:i+1]) + np.trapz(tw2[:i+1], x=t1[:i+1])

plt.clf()
plt.plot(t1, TE2)
plt.xlabel('Time')
plt.ylabel('Total Energy')
plt.title('Total Energy (Power Integral) vs Time')
plt.savefig('./Two Link Planar Robot Manipulator/Trajectory/total_energy_power.png')


# plotting error in energy found by KE+PE and power integral methods
error = TE1 - TE2
plt.clf()
plt.plot(t1, error)
plt.xlabel('Time')
plt.ylabel('Error in Energy')
plt.title('Error in Energy (KE+PE & Power Integral) vs Time')
plt.savefig('./Two Link Planar Robot Manipulator/Trajectory/error_energy_KE-PE_power.png')
