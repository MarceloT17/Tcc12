import numpy
import math
import matplotlib.pyplot as plt
import seaborn

Z = 0.2
R = 0.03
ra = 0.01

nr = 20
nz = 20

T = numpy.zeros([nr, nz], dtype=float)

for i in range(0, nr, 1):
    for j in range(0, nz, 1):
        T[i, j] = (math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z))

print(T)


A_sol12 = T.copy()
'''
ncolunas1 = (nr * nz)
eixoD1 = numpy.linspace(ra, R, ncolunas1)
A_1 = A_sol12.copy()
A_1_1 = numpy.reshape(A_1, ncolunas1)
plt.plot(eixoD1, A_1_1)
plt.show()
'''
