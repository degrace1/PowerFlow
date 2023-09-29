import numpy as np
import matplotlib.pyplot as plt
from classes import *
from functions import *


def main():
    busnum = int(input("Enter number of buses: "))
    knowns = [VarMat() for i in range(int(busnum))]
    xmat = [VarMat() for j in range(int(busnum))]
    startMats = getInitMats(busnum, xmat, knowns)
    knowns = startMats[0]
    xmat = startMats[1]

    printMat(busnum, xmat)

    printMat(busnum, knowns)

    xmat = setInitGuess(busnum, xmat)

    printMat(busnum, xmat)

    yBus = [[VarMat() for i in range(int(busnum))] for j in range(int(busnum))]
    zBus = [[complex(0, 0) for i in range(int(busnum))] for j in range(int(busnum))]
    yz = getZYbus(busnum, yBus, zBus)
    yBus = yz[0]
    zBus = yz[1]
    print(zBus)

    yBus = calcYbus(busnum, yBus, zBus)
    printMultiMat(busnum, yBus)


if __name__ == "__main__":
    main()