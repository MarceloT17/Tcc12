
import numpy
import math
import matplotlib.pyplot as plt
import seaborn

nr = 52 #valores pares
nz = 52

T = numpy.zeros([nr, nz], dtype=float)

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

    for j in range(0, nz, 1):

        an = ((((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * kn * ((R - ra) / nr))/ (Z / nz)
        asul = ((((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ks * ((R - ra) / nr)) / (Z / nz)
        ae = ((((i + 1) * (R - ra) / nr) + ra) * ke * (Z / nz)) / ((R - ra) / nr)
        aw = (((i * (R - ra) / nr) + ra) * kw * (Z / nz)) / ((R - ra) / nr)

        sc = (gm * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr))) + (ro * c * w * Ta * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))


        if (i == 0 and j ==0):
            ap = an + ae + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

            T[i, j] = ((an * (T[i, j + 1])) / ap) + ((ae * (T[i + 1, j])) / ap) + (sc/ap)

        elif (i == 0 and j != 0 and j != nz - 1):
            ap = an + ae + asul + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

            T[i, j] = ((an * (T[i, j + 1])) / ap) + ((asul * (T[i, j - 1])) / ap) + ((ae * (T[i + 1, j])) / ap) + (sc/ap)

        elif (i == 0 and j == nz - 1):
            ap = ae + asul + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

            T[i, j] = ((asul * (T[i, j - 1])) / ap) + ((ae * (T[i + 1, j]))/ ap) + (sc/ap)


        elif (i != 0 and i != nr - 1 and j == nz - 1):
            ap = ae + aw + asul + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

            T[i, j] = ((asul * (T[i, j - 1])) / ap) + ((aw * (T[i - 1, j])) / ap) + (ae * (T[i + 1, j]) / ap) + (sc/ap)


        elif (i == nr - 1 and j == nz - 1):
            q1 = 400
            q2 = h / (1 + (h * ((R - ra) / (nr * 2)) / ke))
            ap = aw + asul + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr))) + ((((i + 1.0) * (R - ra) / nr) + ra) * (Z/nz) * (q2))

            T[i, j] = ((asul * (T[i, j - 1])) / ap) + ((aw * (T[i - 1, j])) / ap) + (sc/ap) + (((((i + 1.0) * (R - ra) / nr) + ra) * (Z/nz) * (q1 + (q2 * Tinf)))/ ap)


        elif (i == nr - 1 and j != 0 and j != nz - 1):
            q1 = 400
            q2 = h / (1 + (h * ((R - ra) / (nr * 2)) / ke))
            ap = an + aw + asul + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr))) + ((((i + 1.0) * (R - ra) / nr) + ra) * (Z/nz) * (q2))

            T[i, j] = ((an * (T[i, j + 1])) / ap) + ((asul * (T[i, j - 1])) / ap) + ((aw * (T[i - 1, j])) / ap) + (sc/ap) + (((((i + 1.0) * (R - ra) / nr) + ra) * (Z/nz) * (q1 + (q2 * Tinf)))/ ap)


        elif (i == nr - 1 and j == 0):
            q1 = 400
            q2 = h / (1 + (h * ((R - ra) / (nr * 2)) / ke))
            ap = an + aw + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr))) + ((((i + 1.0) * (R - ra) / nr) + ra) * (Z/nz) * (q2))

            T[i, j] = ((an * (T[i, j + 1])) / ap) + ((aw * (T[i - 1, j])) / ap) + (sc/ap) + (((((i + 1.0) * (R - ra) / nr) + ra) * (Z/nz) * (q1 + (q2 * Tinf)))/ ap)


        elif (i != 0 and i != nr - 1 and j == 0):
            ap = an + ae + aw + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

            T[i, j] = ((an * (T[i, j + 1])) / ap) + ((aw * (T[i - 1, j])) / ap) + ((ae * (T[i + 1, j])) / ap) + (sc/ap)


        elif (i != 0 and i != nr - 1 and j != 0 and j != nz - 1):
            ap = an + ae + aw + asul + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

            T[i, j] = ((an * (T[i, j + 1])) / ap) + ((asul * (T[i, j - 1])) / ap) + ((aw * (T[i - 1, j])) / ap) + ((ae * (T[i + 1, j])) / ap) + (sc/ap)

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

        for j in range(0, nz, 1):

            an = ((((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * kn * ((R - ra) / nr)) / (Z / nz)
            asul = ((((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ks * ((R - ra) / nr)) / (Z / nz)
            ae = ((((i + 1) * (R - ra) / nr) + ra) * ke * (Z / nz)) / ((R - ra) / nr)
            aw = (((i * (R - ra) / nr) + ra) * kw * (Z / nz)) / ((R - ra) / nr)

            sc = (gm * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr))) + (
                        ro * c * w * Ta * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

            if (i == 0 and j == 0):
                ap = an + ae + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

                E[i, j] = ((an * (E[i, j + 1])) / ap) + ((ae * (E[i + 1, j])) / ap) + (sc / ap)

            elif (i == 0 and j != 0 and j != nz - 1):
                ap = an + ae + asul + (
                            ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

                E[i, j] = ((an * (E[i, j + 1])) / ap) + ((asul * (E[i, j - 1])) / ap) + ((ae * (E[i + 1, j])) / ap) + (
                            sc / ap)

            elif (i == 0 and j == nz - 1):
                ap = ae + asul + (
                            ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

                E[i, j] = ((asul * (E[i, j - 1])) / ap) + ((ae * (E[i + 1, j])) / ap) + (sc / ap)


            elif (i != 0 and i != nr - 1 and j == nz - 1):
                ap = ae + aw + asul + (
                            ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

                E[i, j] = ((asul * (E[i, j - 1])) / ap) + ((aw * (E[i - 1, j])) / ap) + (ae * (E[i + 1, j]) / ap) + (
                            sc / ap)


            elif (i == nr - 1 and j == nz - 1):
                q1 = 400
                q2 = h / (1 + (h * ((R - ra) / (nr * 2)) / ke))
                ap = aw + asul + (
                            ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr))) + (
                                 (((i + 1.0) * (R - ra) / nr) + ra) * (Z / nz) * (q2))

                E[i, j] = ((asul * (E[i, j - 1])) / ap) + ((aw * (E[i - 1, j])) / ap) + (sc / ap) + (
                            ((((i + 1.0) * (R - ra) / nr) + ra) * (Z / nz) * (q1 + (q2 * Tinf))) / ap)


            elif (i == nr - 1 and j != 0 and j != nz - 1):
                q1 = 400
                q2 = h / (1 + (h * ((R - ra) / (nr * 2)) / ke))
                ap = an + aw + asul + (
                            ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr))) + (
                                 (((i + 1.0) * (R - ra) / nr) + ra) * (Z / nz) * (q2))

                E[i, j] = ((an * (E[i, j + 1])) / ap) + ((asul * (E[i, j - 1])) / ap) + ((aw * (E[i - 1, j])) / ap) + (
                            sc / ap) + (((((i + 1.0) * (R - ra) / nr) + ra) * (Z / nz) * (q1 + (q2 * Tinf))) / ap)


            elif (i == nr - 1 and j == 0):
                q1 = 400
                q2 = h / (1 + (h * ((R - ra) / (nr * 2)) / ke))
                ap = an + aw + (
                            ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr))) + (
                                 (((i + 1.0) * (R - ra) / nr) + ra) * (Z / nz) * (q2))

                E[i, j] = ((an * (E[i, j + 1])) / ap) + ((aw * (E[i - 1, j])) / ap) + (sc / ap) + (
                            ((((i + 1.0) * (R - ra) / nr) + ra) * (Z / nz) * (q1 + (q2 * Tinf))) / ap)


            elif (i != 0 and i != nr - 1 and j == 0):
                ap = an + ae + aw + (
                            ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

                E[i, j] = ((an * (E[i, j + 1])) / ap) + ((aw * (E[i - 1, j])) / ap) + ((ae * (E[i + 1, j])) / ap) + (
                            sc / ap)


            elif (i != 0 and i != nr - 1 and j != 0 and j != nz - 1):
                ap = an + ae + aw + asul + (
                            ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

                E[i, j] = ((an * (E[i, j + 1])) / ap) + ((asul * (E[i, j - 1])) / ap) + ((aw * (E[i - 1, j])) / ap) + (
                            (ae * (E[i + 1, j])) / ap) + (sc / ap)


    print(E)
    return E


for a in range(0, 10000, 1):
    D = calculo(D)


#fazer convergencia

'''
DSA = D[:, 3]
print(DSA)
eixor = numpy.linspace(ra, R, nr)
plt.plot(eixor, DSA)
plt.show()
'''


eixoz = numpy.linspace(0, Z)
eixor = numpy.linspace(ra, R, nr)
eixox = D


numpy.meshgrid(eixoz, eixor)
ax = seaborn.heatmap(eixox)
ax.invert_yaxis()
plt.show()



