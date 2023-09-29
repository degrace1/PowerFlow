# Functions file for Power FLow 4205 Project


def getInitMats(busnum, xmat, knowns):
    for i in range(busnum):

        knowns[i].name = input("Enter known #" + str(i + 1) + ": ")
        knowns[i].val = input("Enter associated known value: ")
        numvar = str(knowns[i].name)[1]

        if "p" in knowns[i].name or "P" in knowns[i].name:
            xmat[i].name = "T" + numvar
            knowns[i].name = "P" + numvar
        elif "q" in knowns[i].name or "Q" in knowns[i].name:
            xmat[i].name = "V" + numvar
            knowns[i].name = "Q" + numvar
    return knowns, xmat


def setInitGuess(busnum, xmat):
    for i in range(int(busnum)):
        if "V" in xmat[i].name:
            xmat[i].val = 1
        elif "T" in xmat[i].name:
            xmat[i].val = 0
    return xmat


def printMat(busnum, xmat):
    for i in range(int(busnum)):
        print(xmat[i].name + ", " + str(xmat[i].val))


def printMultiMat(busnum, mat):
    for i in range(int(busnum)):
        for j in range(int(busnum)):
            print(mat[i][j].name + ", " + str(mat[i][j].val))


def getZYbus(busnum, yBus, zBus):
    for i in range(int(busnum)):
        for j in range(int(busnum)):
            yBus[i][j].name = "Y" + str(i + 1) + str(j + 1)
            # ask for "zbus" values
            if j != i:
                if zBus[j][i] != complex(0, 0) or zBus[i][j] != 0:
                    zBus[i][j] = zBus[i][j]
                else:
                    print("Please enter zero if there is no bus")
                    a = float(input("Enter z" + str(i + 1) + str(j + 1) + " a value: "))
                    b = float(input("Enter z" + str(i + 1) + str(j + 1) + " b value: "))
                    zBus[i][j] = complex(a, b)
    return yBus, zBus


def calcYbus(busnum, yBus, zBus):
    num = int(busnum)
    for i in range(num):
        for j in range(num):
            # calc diagnol
            if i == j:
                sum = 0
                for k in range(num):
                    if k != i and zBus[i][k] != complex(0, 0) and zBus[i][j] != 0:
                        sum += 1 / zBus[i][k]
                if sum == 0:
                    yBus[i][j].val = complex(0, 0)
                else:
                    yBus[i][j].val = sum
            else:
                if zBus[i][k] != complex(0, 0) and zBus[i][j] != 0:
                    yBus[i][j].val = -1 / zBus[i][j]
                else:
                    yBus[i][j].val = complex(0, 0)
    return yBus