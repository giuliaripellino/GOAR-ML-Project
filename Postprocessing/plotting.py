from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pyspark.sql import SparkSession
from matplotlib import cbook
from matplotlib.colors import LightSource
from matplotlib.ticker import LinearLocator
from matplotlib.patches import Rectangle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.transforms import Bbox
import pandas as pd
import sys

try:
    rootPath = sys.argv[1]
except:
    rootPath = "output"
    print(f"Failed to fetch rootPath from SparksInTheDark.main... using '{rootPath}' instead")

limitsPath = f"../SparksInTheDark/{rootPath}/limits/"
valuesPath = f"../SparksInTheDark/{rootPath}/plotValues/"
samplePath = f"../SparksInTheDark/{rootPath}/sample/"

savePath = f"../SparksInTheDark/{rootPath}"
saveFileName = "figures.pdf"

def save_plots_to_pdf(file_path, plot_functions):
    with PdfPages(file_path) as pdf:
        for plot_function, arguments in plot_functions:
            plt.figure()
            plot_function(*arguments)
            pdf.savefig()
            plt.close()
    print(f"Plots saved as {file_path}")

def plotDensity(pointsPerAxis, z_max, limitsPath, valuesPath):

    limits = np.array(pd.read_parquet(limitsPath))[-1,-1]
    values = np.array(pd.read_parquet(valuesPath))[-1,-1]

    for i in range(0, len(limits), 2):
        print(f"Variable {i//2 + 1} limits: ({limits[i]}, {limits[i+1]})")

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
    #plt.savefig("test1.pdf",format="pdf")
    #plt.show()

def scatterPlot(dimensions, alph, limitsPath, samplePath):

    limits = np.array(pd.read_parquet(limitsPath))[-1,-1]
    values = np.array(pd.read_parquet(valuesPath))[-1,-1]

    for i in range(0, len(limits), 2):
        print(f"Variable {i//2 + 1} limits: ({limits[i]}, {limits[i+1]})")

    fig, axs = plt.subplots(1, 3,figsize=(12,4))

    length = int(len(values) / 3)

    xs = np.ndarray(shape=(dimensions, length))
    for i in range(dimensions):
        for j in range(length):
            xs[i,j] = values[3*j + i]

    # Assuming `color_values` is a list of colors for each point
    color_values = np.random.rand(length)  # Example random color values

    # Scatter plots with color values and colorbar
    sc0 = axs[0].scatter(xs[0, :], xs[1, :], c=color_values, alpha=alph)
    axs[0].set_title('X1,X2')
    cbar0 = fig.colorbar(sc0, ax=axs[0])
    cbar0.set_label('f_n(X1,X2)')

    sc1 = axs[1].scatter(xs[0, :], xs[2, :], c=color_values, alpha=alph)
    axs[1].set_title('X1,X3')
    cbar1 = fig.colorbar(sc1, ax=axs[1])
    cbar1.set_label('f_n(X1,X3)')

    sc2 = axs[2].scatter(xs[1, :], xs[2, :], c=color_values, alpha=alph)
    axs[2].set_title('X2,X3')
    cbar2 = fig.colorbar(sc2, ax=axs[2])
    cbar2.set_label('f_n(X2,X3)')

    #plt.show()

plot_functions = [
    (plotDensity,(256,0.0002, limitsPath, valuesPath)),
    (scatterPlot,(3, 1, limitsPath, samplePath)),
]

save_plots_to_pdf(savePath + saveFileName, plot_functions)


