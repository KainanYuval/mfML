import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


def plotAlpha(pathToMotherDir):
    alphaPerLayerIndex = getAlphaDict(pathToMotherDir)
    fig, ax = plt.subplots()
    x = []
    y = []
    x = sorted(alphaPerLayerIndex.keys())
    for key in x:
        y.append(alphaPerLayerIndex[key])
    ax.plot(x, y)

    ax.set(xlabel='layer_index', ylabel='alpha_sobolov',
           title=pathToMotherDir.split('//')[-1])
    ax.grid()

    fig.savefig("test.png")
    plt.show()

def getAlphaDict(pathToMotherDir):
    dirs = os.listdir(pathToMotherDir)
    alphaPerLayerIndex = {}
    for dir1 in dirs:
        if '.txt' in dir1:
            continue
        layerDir = pathToMotherDir + '//' + dir1 + '//results'
        iterationsDir = os.listdir(layerDir)
        alpha = 0.0
        if '.txt' in layerDir:
            continue
        count = 0
        for dir2 in iterationsDir:
            if '.txt' in layerDir + '//' + dir2:
                continue

            alphaFile = layerDir + '//' + dir2 + '//alphaMterm.txt'
            try:
                count += 1
                file = open(alphaFile)
                line = file.readline()
                alpha += float(line)
            except:
                print('faulty')
        layerIndex = dir1.split('layer_')[1].split('_')[0]
        if count > 0:
            alphaPerLayerIndex[int(layerIndex)] = alpha/float(count)
    return alphaPerLayerIndex

def plotAlpha(pathToMotherDir1, pathToMotherDir2):
    alphaPerLayerIndex1 = getAlphaDict(pathToMotherDir1)
    alphaPerLayerIndex2 = getAlphaDict(pathToMotherDir2)

    fig, ax = plt.subplots()
    y = []
    x = sorted(alphaPerLayerIndex1.keys())
    for key in x:
        y.append(alphaPerLayerIndex1[key])
    TMP, = ax.plot(x, y, color ='b', label=pathToMotherDir1.split('//')[-1])
    y = []
    x = sorted(alphaPerLayerIndex2.keys())
    for key in x:
        y.append(alphaPerLayerIndex2[key])
    ax.plot(x, y, color ='r', label=pathToMotherDir2.split('//')[-1])

    ax.set(xlabel='layer_index', ylabel='alpha_sobolov')
    ax.grid()

    fig.savefig("test.png")
    plt.legend(handler_map={TMP: HandlerLine2D(numpoints=4)})

    plt.show()


pathToMotherDir1 = 'C:\CIFAR//residual//18ResBlocksNet//18ResBlocksNet - adam - 30 epochs'
pathToMotherDir2 = 'C:\CIFAR//residual//layer_output'
plotAlpha(pathToMotherDir1, pathToMotherDir2)
