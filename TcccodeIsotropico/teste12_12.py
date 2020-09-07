import numpy
import math
import matplotlib.pyplot as plt
import seaborn
import Tcccode12_1_SA
import Tcccode12_Num

nr = 52 #valores pares
R = 0.03
ra = 0.01

eixor = numpy.linspace(ra, R, nr)


DSa3 = Tcccode12_Num.DSA
T4 = Tcccode12_1_SA.T3
D5 = []

for i in range(len(eixor)):
    D5.append((T4[i] - DSa3[i])/ T4[i] * 100)

PDiff = D5
plt.ylabel('Valores percentuais')
plt.plot(eixor, PDiff)

plt.show()
