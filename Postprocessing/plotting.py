from matplotlib import cm
import matplotlib.pyplot as plt
#import matplotlib_inline.backend_inline
import numpy as np
from pyspark.sql import SparkSession
from matplotlib import cbook
from matplotlib.colors import LightSource
from matplotlib.ticker import LinearLocator
from matplotlib.patches import Rectangle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.transforms import Bbox

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Read Parquet File") \
    .getOrCreate()

limitsPath = "/Users/axega337/Documents/PhD/SparksInTheDark/GOAR-ML-Project/SparksInTheDark/output/output_3d/limits/"
valuesPath = "/Users/axega337/Documents/PhD/SparksInTheDark/GOAR-ML-Project/SparksInTheDark/output/output_3d/plotValues/"

samplePath = "/Users/axega337/Documents/PhD/SparksInTheDark/GOAR-ML-Project/SparksInTheDark/output/output_3d/sample/"

def plotDensity(pointsPerAxis, z_max, limitsPath, valuesPath):

    #matplotlib_inline.backend_inline.set_matplotlib_formats('png2x')

    limits = np.array(spark.read.parquet(limitsPath).collect())[-1,-1]
    values = np.array(spark.read.parquet(valuesPath).collect())[-1,-1]

    print(limits)

    print(values)
    x4_min = limits[0]
    x4_max = limits[1]
    x6_min = limits[2]
    x6_max = limits[3]

    x4_width = (x4_max - x4_min) / pointsPerAxis
    x6_width = (x6_max - x6_min) / pointsPerAxis

    x = np.arange(x4_min, x4_max, x4_width)
    y = np.arange(x6_min, x6_max, x6_width)
    x, y = np.meshgrid(x, y, indexing='ij')

    z = np.empty((pointsPerAxis,pointsPerAxis))
    for i in range(pointsPerAxis):
        for j in range(pointsPerAxis):
            z[i,j] = values[i*pointsPerAxis + j]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=1, antialiased=False)
    # surf = ax.plot_surface(x, y, z, cmap=cm.gist_earth, linewidth=0, antialiased=False)
    
    # Customize the z axis.
    #ax.set_zlim(0.0, z_max)
    #ax.set_xlim(x4_min, x4_max)
    #ax.set_ylim(x6_min, x6_max)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('f_n(X1,X2)')
    ax.invert_xaxis()
    plt.savefig("test1.pdf",format="pdf")
    plt.show()


plotDensity(256,0.0002, limitsPath, valuesPath)

def scatterPlot(dimensions, alph, limitsPath, samplePath):

    limits = np.array(spark.read.parquet(limitsPath).collect())[-1,-1]
    values = np.array(spark.read.parquet(samplePath).collect())[-1,-1]
    print(limits)

    print(values)
    print(np.shape(values))
    #fig, ax = plt.subplots()

    fig, axs = plt.subplots(2, 2)

    length = int(len(values) / 3)

    xs = np.ndarray(shape=(dimensions, length))
    for i in range(dimensions):
        for j in range(length):
            xs[i,j] = values[3*j + i]

    axs[0,0].scatter(xs[0,], xs[1,], alpha = alph)
    axs[0, 0].set_title('X2,X4')
    #axs[0,0].set_xlabel('X2')
    #axs[0,0].set_ylabel('X4')
    axs[0,1].scatter(xs[0,], xs[2,], alpha = alph)
    axs[0, 1].set_title('X2,X5')
    #axs[0,1].set_xlabel('X2')
    #axs[0,1].set_ylabel('X5')
    axs[1,1].scatter(xs[1,], xs[2,], alpha = alph)
    axs[1, 1].set_xlabel('X4,X5')
    #axs[1,1].set_xlabel('X4')
    #axs[1,1].set_ylabel('X5')

    # Customize the z axis.
    #ax.set_xlim(x4_min, x4_max)
    #ax.set_ylim(x6_min, x6_max)
    #ax.set_xlabel('X1')
    #ax.set_ylabel('X2')

    plt.show()

#scatterPlot(3, 1, limitsPath, samplePath)