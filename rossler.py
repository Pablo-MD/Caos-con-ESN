import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

rho = 0.2
sigma = 0.2
beta = 5.7
t=0.0
f=open("datos.txt", "a")
def f(state, t):
    x, y, z = state  # Desempaqueta el vector de estado
    return (-y - z), x + (rho * y), sigma + z * (x - beta)  # Derivadas

state0 = [1.0, 1.0, 1.0]
state1 = [1.01, 1.01, 1.01]
t = np.arange(0.0, 200.0, 0.01)

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
