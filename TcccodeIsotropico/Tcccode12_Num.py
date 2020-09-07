import numpy
import math
import matplotlib.pyplot as plt
import seaborn

#Numerica 1D
nr = 52#valores pares

T = numpy.zeros([nr], dtype=float)

nz = 52
Z = 0.2
R = 0.03
ra = 0.01

ro = 1050
c = 3617
Ta = 37

h = 10.0
Tinf = 23.0

for i in range(0, nr, 1):

    ks = 0.49
    ke = 0.49
    kw = 0.49
    kn = 0.49
    gm = 991.9
    w = 0.0006722

    ae = ((((i + 1) * (R - ra) / nr) + ra) * ke * (Z / nz)) / ((R - ra) / nr)
    aw = (((i * (R - ra) / nr) + ra) * kw * (Z / nz)) / ((R - ra) / nr)

    sc = (gm * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

    if (i == 0):
        ap = ae
        T[i] = ((ae * (T[i + 1])) / ap) + (sc / ap)

    if (i == nr - 1):
        q1 = 400
        q2 = h / (1 + (h * ((R - ra) / (nr * 2)) / ke))
        ap = aw + (
                    (((i + 1.0) * (R - ra) / nr) + ra) * (Z / nz) * (q2))

        T[i] = ((aw * (T[i - 1])) / ap) + (sc / ap) + (
                    ((((i + 1.0) * (R - ra) / nr) + ra) * (Z / nz) * (q1 + (q2 * Tinf))) / ap)

    elif (i != nr -1 and i != 0):
        ap = ae + aw

        T[i] = ((aw * (T[i - 1])) / ap) + (
                    (ae * (T[i + 1])) / ap) + (sc / ap)

    print(T)

D = T.copy()
print(D)


def calculo(E):
    for i in range(0, nr, 1):

        ks = 0.49
        ke = 0.49
        kw = 0.49
        kn = 0.49
        gm = 991.9
        w = 0.0006722

        ae = ((((i + 1) * (R - ra) / nr) + ra) * ke * (Z / nz)) / ((R - ra) / nr)
        aw = (((i * (R - ra) / nr) + ra) * kw * (Z / nz)) / ((R - ra) / nr)

        sc = (gm * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

        if (i == 0):
            ap = ae
            E[i] = ((ae * (E[i + 1])) / ap) + (sc / ap)

        if (i == nr - 1):
            q1 = 400
            q2 = h / (1 + (h * ((R - ra) / (nr * 2)) / ke))
            ap = aw + (
                    (((i + 1.0) * (R - ra) / nr) + ra) * (Z / nz) * (q2))

            E[i] = ((aw * (E[i - 1])) / ap) + (sc / ap) + (
                    ((((i + 1.0) * (R - ra) / nr) + ra) * (Z / nz) * (q1 + (q2 * Tinf))) / ap)

        elif (i != nr - 1 and i != 0):
            ap = ae + aw

            E[i] = ((aw * (E[i - 1])) / ap) + (
                    (ae * (E[i + 1])) / ap) + (sc / ap)

    print(E)
    return E

for a in range(0, 100000, 1):
    D = calculo(D)

DSA = D.copy()
print(DSA)


'''
eixor = numpy.linspace(ra, R, nr)
plt.plot(eixor, DSA, '^k:')

import Tcccode12_1_SA
eixor = numpy.linspace(ra, R, nr)
plt.plot(eixor, Tcccode12_1_SA.T3)

plt.show()
'''





