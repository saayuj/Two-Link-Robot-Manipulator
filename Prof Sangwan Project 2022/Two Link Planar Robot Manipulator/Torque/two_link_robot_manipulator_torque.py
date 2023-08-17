# importing required libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


### change variable values here ###
m1 = 1
m2 = 1
l1 = 1
l2 = 1
###################################


# defining a few constants
g = 9.81
c6 = m2 * (l2 ** 2) / 3


# function to calculate derivatives
def dSdt(t, S):
    theta1, w1, theta2, w2 = S
    c1 = m1 * (l1 ** 2) / 3 + m2 * (l1 ** 2) + m2 * (l2 ** 2) / 3 + m2 * l1 * l2 * np.cos(theta2)
    c2 = m2 * (l2 ** 2) / 3 + m2 * l1 * l2 * np.cos(theta2) / 2
    c3 = - m2 * l1 * l2 * np.sin(theta2) / 2
    c4 = (m1 / 2 + m2) * g * l1 * np.cos(theta1)
    c5 = m2 * g * l2 * np.cos(theta1 + theta2) / 2
    tau1 = np.cos(theta1)
    tau2 = np.sin(theta2)

    return [w1,
            (- c2 * c3 * (w1 ** 2) - 2 * c3 * c6 * w1 * w2 - c3 * c6 * (w2 ** 2) - c4 * c6 - c5 * c6 + c2 * c5 + tau1 * c6 - tau2 * c2) / (c1 * c6 - (c2 ** 2)),
            w2,
            (c1 * c3 * (w1 ** 2) + 2 * c2 * c3 * w1 * w2 + c2 * c3 * (w2 ** 2) + c2 * c4 + c2 * c5 - c1 * c5 - tau1 * c2 + tau2 * c1) / (c1 * c6 - (c2 ** 2))]


# solving the differential equations
t1 = np.linspace(0, 40, 1001)
sol = odeint(dSdt, y0=[0, 0, 0, 0], t=t1, tfirst=True)
# sol = odeint(dSdt, y0=np.random.randint(4, size=4), t=t1, tfirst=True)

theta1 = sol.T[0]
w1 = sol.T[1]
theta2 = sol.T[2]
w2 = sol.T[3]


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
ani.save('./Two Link Planar Robot Manipulator/Torque/two_link_robot_manipulator_torque.gif', writer='pillow', fps=15)


# plotting total energy by KE+PE method
tau1 = np.cos(theta1)
tau2 = np.sin(theta2)

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
plt.savefig('./Two Link Planar Robot Manipulator/Torque/total_energy_KE-PE.png')


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
plt.savefig('./Two Link Planar Robot Manipulator/Torque/total_energy_power.png')


# plotting total energy by work done by torque method
TE3 = np.zeros(1001)
for i in range(1001):
    TE3[i] = np.trapz(tau1[:i+1], x=theta1[:i+1]) + np.trapz(tau2[:i+1], x=theta2[:i+1])

plt.clf()
plt.plot(t1, TE3)
plt.xlabel('Time')
plt.ylabel('Total Energy')
plt.title('Total Energy (Torque Work) vs Time')
plt.savefig('./Two Link Planar Robot Manipulator/Torque/total_energy_torque.png')


# plotting error in energy
error1 = TE1 - TE1[0]
plt.clf()
plt.plot(t1, error1)
plt.xlabel('Time')
plt.ylabel('Error in Energy')
plt.title('Error in Energy vs Time')
plt.savefig('./Two Link Planar Robot Manipulator/Torque/error_energy.png')


# plotting error in energy found by KE+PE and power integral methods
error2 = TE1 - TE2
plt.clf()
plt.plot(t1, error2)
plt.xlabel('Time')
plt.ylabel('Error in Energy')
plt.title('Error in Energy (KE+PE & Power Integral) vs Time')
plt.savefig('./Two Link Planar Robot Manipulator/Torque/error_energy_KE-PE_power.png')


# plotting error in energy found by KE+PE and work done by torque methods
error3 = TE1 - TE3
plt.clf()
plt.plot(t1, error3)
plt.xlabel('Time')
plt.ylabel('Error in Energy')
plt.title('Error in Energy (KE+PE & Torque Work) vs Time')
plt.savefig('./Two Link Planar Robot Manipulator/Torque/error_energy_KE-PE_torque.png')


# plotting error in energy found by power integral and work done by torque methods
error4 = TE2 - TE3
plt.clf()
plt.plot(t1, error4)
plt.xlabel('Time')
plt.ylabel('Error in Energy')
plt.title('Error in Energy (Power Integral & Torque Work) vs Time')
plt.savefig('./Two Link Planar Robot Manipulator/Torque/error_energy_power_torque.png')
