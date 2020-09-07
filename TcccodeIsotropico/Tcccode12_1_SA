import numpy
import math
import matplotlib.pyplot as plt
import seaborn

nr = 52 #valores pares

T = []

Z = 0.2
R = 0.03
ra = 0.01

ks = 0.49
ke = 0.49
kw = 0.49
kn = 0.49
gm = 991.9
w = 0.0006722

h = 10.0
Tinf = 23.0

eixox = numpy.linspace(ra, R, nr)


for i in range(len(eixox)):

    # colocando todos com ke porque e isotropico
    q1 = 400
    q2 = h / (1 + (h * ((R - ra) / (nr * 2)) / ke))
    c1 = gm * (ra * ra) / 2
    c2 = (q1 + (q2 * Tinf) - (q2 * ((c1 * (math.log(R)) / ke) - (gm * (R * R) / (4 * ke)))) - (c1 / ke) + (
                gm * R / 2)) * (1 / q2) * ke
    T.append((c1 / ke * (math.log(eixox[i]))) + (c2 / ke) - (gm * ((eixox[i]) * (eixox[i])) / (4 * ke)))
print(T)

T3 = T.copy()

