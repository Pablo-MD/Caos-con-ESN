import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from math import fabs, log


rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0
t=0.0
f=open("datos.txt", "a")
def f(state, t):
    x, y, z = state  # Desempaqueta el vector de estado
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivadas

state0 = [1.0, 1.0, 1.0]
state1 = [1.01, 1.01, 1.01]
t = np.arange(0.0, 50.0, 0.01)

states = odeint(f, state0, t)
states2 = odeint(f, state1, t)

a =np.fabs(states-states2)


sol1 =a/0.01

sol2 = np.log(sol1)
np.savetxt('datos.txt', sol2)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(states[:, 0], states[:, 1], states[:, 2])
plt.show()
