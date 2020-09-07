import numpy
import math
import matplotlib.pyplot as plt
import seaborn

nr = 20 #valores pares
nz = 20

T = numpy.zeros([nr, nz], dtype=float)

Z = 0.2
R = 0.03
ra = 0.01

ro = 1050
c = 3617
Ta = 37

h = 100
b = Z/2
cvetorq = 0.01

h2 = 10.0
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

        scmms = (((-1 * (ro * c * w)) * (Ta - (
                    (math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (
                math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z))))) - gm - (
                         (1 / (((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra)) * kn * (
                     math.sin((math.pi) / Z * (j + (1.0 / 2.0)) * Z / nz)) * (math.pi) / (R - ra) * (
                             math.cos((math.pi) * (((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) / (R - ra)))) + (
                         kn * ((math.pi) / (R - ra)) * ((math.pi) / (R - ra)) * (math.sin(
                     (math.pi) * (j + (1.0 / 2.0)) * Z / nz / Z)) * (math.sin(
                     (math.pi) * (((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) / (R - ra)))) + (
                         kn * ((math.pi) / Z) * ((math.pi) / Z) * (
                     math.sin((math.pi) / Z * (j + (1.0 / 2.0)) * Z / nz)) * (math.sin(
                     (math.pi) * (((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) / (R - ra))))) * (
                        Z / nz * (((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (R - ra) / nr)

        if (i == 0 and j == 0):
            ap = an + ae + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
            scaw = - (((i * (R - ra) / nr) + ra) * (Z/nz) * kw * (math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z)) * ((math.pi) / (R-ra)) * (math.cos((math.pi) * ra / (R-ra))))
            scasul = ((-1) * ks * ((math.pi) / Z) * (math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * ((R - ra) / nr))

            T[i, j] = ((an * (T[i, j + 1])) / ap) + ((ae * (T[i + 1, j])) / ap) + (sc / ap) + (scmms / ap) + (scaw / ap) + (scasul / ap)

        elif (i == 0 and j != 0 and j != nz - 1):
            ap = an + ae + asul + (
                        ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
            scaw = - (((i * (R - ra) / nr) + ra) * (Z / nz) * kw * (
                math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z)) * ((math.pi) / (R - ra)) * (
                          math.cos((math.pi) * ra / (R - ra))))

            T[i, j] = ((an * (T[i, j + 1])) / ap) + ((asul * (T[i, j - 1])) / ap) + ((ae * (T[i + 1, j])) / ap) + (
                        sc / ap) + (scmms / ap) + (scaw / ap)

        elif (i == 0 and j == nz - 1):
            ap = ae + asul + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
            scaw = - (((i * (R - ra) / nr) + ra) * (Z / nz) * kw * (
                math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z)) * ((math.pi) / (R - ra)) * (
                          math.cos((math.pi) * ra / (R - ra))))
            scan1 = ((-1) * kn * ((math.pi) / Z) * (math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * ((R - ra) / nr))

            T[i, j] = ((asul * (T[i, j - 1])) / ap) + ((ae * (T[i + 1, j])) / ap) + (sc / ap) + (scmms / ap) + (scaw / ap) + (scan1 / ap)


        elif (i != 0 and i != nr - 1 and j == nz - 1):
            ap = ae + aw + asul + (
                        ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
            scan1 = ((-1) * kn * ((math.pi) / Z) * (
                math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (
                                 ((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * ((R - ra) / nr))

            T[i, j] = ((asul * (T[i, j - 1])) / ap) + ((aw * (T[i - 1, j])) / ap) + (ae * (T[i + 1, j]) / ap) + (
                        sc / ap) + (scmms / ap) + (scan1 / ap)


        elif (i == nr - 1 and j == nz - 1):
            ap = aw + asul + (
                        ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
            scan1 = ((-1) * kn * ((math.pi) / Z) * (
                math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (
                             ((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * ((R - ra) / nr))
            scae = ((((i + 1) * (R - ra) / nr) + ra) * ke * (Z / nz) * (math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z)) * ((math.pi) / (R - ra)) * (math.cos((math.pi) * R / (R-ra))))

            T[i, j] = ((asul * (T[i, j - 1])) / ap) + ((aw * (T[i - 1, j])) / ap) + (sc / ap) + (scmms / ap) + (scan1 / ap) + (scae / ap)


        elif (i == nr - 1 and j != 0 and j != nz - 1):
            ap = an + aw + asul + (
                        ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
            scae = ((((i + 1) * (R - ra) / nr) + ra) * ke * (Z / nz) * (
                math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z)) * ((math.pi) / (R - ra)) * (
                        math.cos((math.pi) * R / (R - ra))))

            T[i, j] = ((an * (T[i, j + 1])) / ap) + ((asul * (T[i, j - 1])) / ap) + ((aw * (T[i - 1, j])) / ap) + (
                        sc / ap) + (scmms / ap) + (scae / ap)


        elif (i == nr - 1 and j == 0):
            ap = an + aw + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
            scae = ((((i + 1) * (R - ra) / nr) + ra) * ke * (Z / nz) * (
                math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z)) * ((math.pi) / (R - ra)) * (
                        math.cos((math.pi) * R / (R - ra))))
            scasul = ((-1) * ks * ((math.pi) / Z) * (
                math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (
                                  ((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * ((R - ra) / nr))

            T[i, j] = ((an * (T[i, j + 1])) / ap) + ((aw * (T[i - 1, j])) / ap) + (sc / ap) + (scmms / ap) + (scae / ap) + (scasul / ap)


        elif (i != 0 and i != nr - 1 and j == 0):
            ap = an + ae + aw + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
            scasul = ((-1) * ks * ((math.pi) / Z) * (
                math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (
                              ((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * ((R - ra) / nr))

            T[i, j] = ((an * (T[i, j + 1])) / ap) + ((aw * (T[i - 1, j])) / ap) + ((ae * (T[i + 1, j])) / ap) + (
                        sc / ap) + (scmms / ap) + (scasul / ap)


        elif (i != 0 and i != nr - 1 and j != 0 and j != nz - 1):
            ap = an + ae + aw + asul + (
                        ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

            T[i, j] = ((an * (T[i, j + 1])) / ap) + ((asul * (T[i, j - 1])) / ap) + ((aw * (T[i - 1, j])) / ap) + (
                        (ae * (T[i + 1, j])) / ap) + (sc / ap) + (scmms / ap)

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

            scmms = (((-1 * (ro * c * w)) * (Ta - (
                    (math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (
                math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z))))) - gm - (
                             (1 / (((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra)) * kn * (
                         math.sin((math.pi) / Z * (j + (1.0 / 2.0)) * Z / nz)) * (math.pi) / (R - ra) * (
                                 math.cos((math.pi) * (((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) / (R - ra)))) + (
                             kn * ((math.pi) / (R - ra)) * ((math.pi) / (R - ra)) * (math.sin(
                         (math.pi) * (j + (1.0 / 2.0)) * Z / nz / Z)) * (math.sin(
                         (math.pi) * (((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) / (R - ra)))) + (
                             kn * ((math.pi) / Z) * ((math.pi) / Z) * (
                         math.sin((math.pi) / Z * (j + (1.0 / 2.0)) * Z / nz)) * (math.sin(
                         (math.pi) * (((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) / (R - ra))))) * (
                            Z / nz * (((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (R - ra) / nr)

            if (i == 0 and j == 0):
                ap = an + ae + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
                scaw = - (((i * (R - ra) / nr) + ra) * (Z / nz) * kw * (
                    math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z)) * ((math.pi) / (R - ra)) * (
                              math.cos((math.pi) * ra / (R - ra))))
                scasul = ((-1) * ks * ((math.pi) / Z) * (
                    math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (
                                      ((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * ((R - ra) / nr))

                E[i, j] = ((an * (E[i, j + 1])) / ap) + ((ae * (E[i + 1, j])) / ap) + (sc / ap) + (scmms / ap) + (
                            scaw / ap) + (scasul / ap)

            elif (i == 0 and j != 0 and j != nz - 1):
                ap = an + ae + asul + (
                        ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
                scaw = - (((i * (R - ra) / nr) + ra) * (Z / nz) * kw * (
                    math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z)) * ((math.pi) / (R - ra)) * (
                              math.cos((math.pi) * ra / (R - ra))))

                E[i, j] = ((an * (E[i, j + 1])) / ap) + ((asul * (E[i, j - 1])) / ap) + ((ae * (E[i + 1, j])) / ap) + (
                        sc / ap) + (scmms / ap) + (scaw / ap)

            elif (i == 0 and j == nz - 1):
                ap = ae + asul + (
                            ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
                scaw = - (((i * (R - ra) / nr) + ra) * (Z / nz) * kw * (
                    math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z)) * ((math.pi) / (R - ra)) * (
                              math.cos((math.pi) * ra / (R - ra))))
                scan1 = ((-1) * kn * ((math.pi) / Z) * (
                    math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (
                                     ((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * ((R - ra) / nr))

                E[i, j] = ((asul * (E[i, j - 1])) / ap) + ((ae * (E[i + 1, j])) / ap) + (sc / ap) + (scmms / ap) + (
                            scaw / ap) + (scan1 / ap)


            elif (i != 0 and i != nr - 1 and j == nz - 1):
                ap = ae + aw + asul + (
                        ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
                scan1 = ((-1) * kn * ((math.pi) / Z) * (
                    math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (
                                 ((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * ((R - ra) / nr))

                E[i, j] = ((asul * (E[i, j - 1])) / ap) + ((aw * (E[i - 1, j])) / ap) + (ae * (E[i + 1, j]) / ap) + (
                        sc / ap) + (scmms / ap) + (scan1 / ap)


            elif (i == nr - 1 and j == nz - 1):
                ap = aw + asul + (
                        ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
                scan1 = ((-1) * kn * ((math.pi) / Z) * (
                    math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (
                                 ((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * ((R - ra) / nr))
                scae = ((((i + 1) * (R - ra) / nr) + ra) * ke * (Z / nz) * (
                    math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z)) * ((math.pi) / (R - ra)) * (
                            math.cos((math.pi) * R / (R - ra))))

                E[i, j] = ((asul * (E[i, j - 1])) / ap) + ((aw * (E[i - 1, j])) / ap) + (sc / ap) + (scmms / ap) + (
                            scan1 / ap) + (scae / ap)


            elif (i == nr - 1 and j != 0 and j != nz - 1):
                ap = an + aw + asul + (
                        ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
                scae = ((((i + 1) * (R - ra) / nr) + ra) * ke * (Z / nz) * (
                    math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z)) * ((math.pi) / (R - ra)) * (
                            math.cos((math.pi) * R / (R - ra))))

                E[i, j] = ((an * (E[i, j + 1])) / ap) + ((asul * (E[i, j - 1])) / ap) + ((aw * (E[i - 1, j])) / ap) + (
                        sc / ap) + (scmms / ap) + (scae / ap)


            elif (i == nr - 1 and j == 0):
                ap = an + aw + (ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
                scae = ((((i + 1) * (R - ra) / nr) + ra) * ke * (Z / nz) * (
                    math.sin(((j + (1.0 / 2.0)) * (Z / nz)) * (math.pi) / Z)) * ((math.pi) / (R - ra)) * (
                            math.cos((math.pi) * R / (R - ra))))
                scasul = ((-1) * ks * ((math.pi) / Z) * (
                    math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (
                                  ((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * ((R - ra) / nr))

                E[i, j] = ((an * (E[i, j + 1])) / ap) + ((aw * (E[i - 1, j])) / ap) + (sc / ap) + (scmms / ap) + (
                            scae / ap) + (scasul / ap)


            elif (i != 0 and i != nr - 1 and j == 0):
                ap = an + ae + aw + (
                            ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))
                scasul = ((-1) * ks * ((math.pi) / Z) * (
                    math.sin((((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * (math.pi) / (R - ra))) * (
                                  ((i + (1.0 / 2.0)) * ((R - ra) / nr)) + ra) * ((R - ra) / nr))

                E[i, j] = ((an * (E[i, j + 1])) / ap) + ((aw * (E[i - 1, j])) / ap) + ((ae * (E[i + 1, j])) / ap) + (
                        sc / ap) + (scmms / ap) + (scasul / ap)


            elif (i != 0 and i != nr - 1 and j != 0 and j != nz - 1):
                ap = an + ae + aw + asul + (
                        ro * c * w * ((Z / nz) * (((i + (1.0 / 2.0)) * (R - ra) / nr) + ra) * ((R - ra) / nr)))

                E[i, j] = ((an * (E[i, j + 1])) / ap) + ((asul * (E[i, j - 1])) / ap) + ((aw * (E[i - 1, j])) / ap) + (
                        (ae * (E[i + 1, j])) / ap) + (sc / ap) + (scmms / ap)

    print(E)
    return E


for a in range(0, 10000, 1):
    D = calculo(D)

B_sol12 = D.copy()



#rever esses plots

#transformar array em vetor nrxnz
ncolunas1 = (nr * nz)
D1 = numpy.reshape(B_sol12, ncolunas1)
eixoD1 = numpy.linspace(ra, R, ncolunas1)
plt.plot(eixoD1, D1)


#SolTmms12
import solTmms12
A_1 = solTmms12.A_sol12.copy()
A_1_1 = numpy.reshape(A_1, ncolunas1)
plt.plot(eixoD1, A_1_1)


plt.show()







