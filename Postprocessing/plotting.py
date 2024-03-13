from matplotlib import cm
import matplotlib.pyplot as plt
#import matplotlib_inline.backend_inline
import numpy as np
from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Read Parquet File") \
    .getOrCreate()

limitsPath = "/Users/axega337/Documents/PhD/SparksInTheDark/GOAR-ML-Project/SparksInTheDark/output/limits/"
valuesPath = "/Users/axega337/Documents/PhD/SparksInTheDark/GOAR-ML-Project/SparksInTheDark/output/plotValues/"

def plotDensity(pointsPerAxis, z_max, limitsPath, valuesPath):

    #matplotlib_inline.backend_inline.set_matplotlib_formats('png2x')

    limits = np.array(spark.read.parquet(limitsPath).collect())[-1,-1]
    values = np.array(spark.read.parquet(valuesPath).collect())[-1,-1]
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
    surf = ax.plot_surface(x, y, z, cmap=cm.gist_earth, linewidth=1, antialiased=False)
    # surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(0.0, z_max)
    ax.set_xlim(x4_min, x4_max)
    ax.set_ylim(x6_min, x6_max)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('f_n(X1,X2)')
    #plt.savefig("test.pdf",format="pdf")
    plt.show()


plotDensity(128,0.01, limitsPath, valuesPath)