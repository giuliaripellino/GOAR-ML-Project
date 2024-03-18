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
    print(f"Using rootPath: {rootPath}")
except:
    rootPath = "output/HPS_count100_1e-6"
    print(f"Failed to fetch rootPath from SparksInTheDark.main... using '{rootPath}' instead")

limitsPath = f"../SparksInTheDark/{rootPath}/limits/"
valuesPath = f"../SparksInTheDark/{rootPath}/plotValues/"
samplePath = f"../SparksInTheDark/{rootPath}/sample/"

savePath = f"../SparksInTheDark/{rootPath}"
saveFileName = "figures.pdf"

variable_list = {"X1":r"$\Delta R(l,b_2)$","X2":r"$m_{J^{lep}} + m_{J^{had}}$", "X3":r"$m_{bb\Delta R_{min}}$"}

def save_plots_to_pdf(file_path, plot_functions):
    with PdfPages(file_path) as pdf:
        for plot_function, arguments in plot_functions:
            plt.figure()
            plot_function(*arguments)
            pdf.savefig()
            plt.close()
    print(f"Plots saved as {file_path}")

def plotDensity(pointsPerAxis, z_max, limitsPath, valuesPath,variable_list):
    limits = np.array(pd.read_parquet(limitsPath))[-1,-1]
    values = np.array(pd.read_parquet(valuesPath))[-1,-1]

    for i in range(0, len(limits), 2):
        print(f"Variable {i//2 + 1} limits: ({limits[i]}, {limits[i+1]})")

    x1_min, x1_max = limits[0], limits[1]
    x2_min, x2_max = limits[2], limits[3]
    x3_min, x3_max = limits[4], limits[5]

    values_3d = values.reshape((pointsPerAxis, pointsPerAxis, pointsPerAxis))
    z_x1_x2 = values_3d[:, :, 0]
    z_x1_x3 = values_3d[:, 0, :]
    z_x2_x3 = values_3d[0, :, :]

    combination_list = {"X1_X2":[x1_min, x1_max, x2_min, x2_max],
                        "X1_X3":[x1_min, x1_max, x3_min, x3_max],
                        "X2_X3":[x2_min, x2_max, x3_min, x3_max]}

    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 5))
    z_list = [z_x1_x2,z_x1_x3,z_x2_x3]
    for i, (combination, minmaxlist) in enumerate(combination_list.items()):
        xlabel, ylabel = combination.split("_")[0], combination.split("_")[1]

        ax = axes[i]

        x_width = (minmaxlist[1] - minmaxlist[0]) / pointsPerAxis
        y_width = (minmaxlist[3] - minmaxlist[2]) / pointsPerAxis

        x = np.arange(minmaxlist[0], minmaxlist[1], x_width)
        y = np.arange(minmaxlist[2], minmaxlist[3], y_width)

        x, y = np.meshgrid(x, y, indexing='ij')

        # Plot the surface.
        surf = ax.plot_surface(x, y, z_list[i], cmap=cm.coolwarm, linewidth=1, antialiased=False)
        xlabel = variable_list[xlabel]
        ylabel = variable_list[ylabel]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(r'$f_n$')
        ax.invert_xaxis()


def plotDensity2D(pointsPerAxis, z_max, limitsPath, valuesPath,variable_list):
    limits = np.array(pd.read_parquet(limitsPath))[-1,-1]
    values = np.array(pd.read_parquet(valuesPath))[-1,-1]

    for i in range(0, len(limits), 2):
        print(f"Variable {i//2 + 1} limits: ({limits[i]}, {limits[i+1]})")

    x1_min, x1_max = limits[0], limits[1]
    x2_min, x2_max = limits[2], limits[3]
    x3_min, x3_max = limits[4], limits[5]

    # Loop to fill z with values from values array
    z_full = np.zeros((pointsPerAxis, pointsPerAxis, pointsPerAxis))
    index = 0
    for i in range(pointsPerAxis):
        for j in range(pointsPerAxis):
            for k in range(pointsPerAxis):
                z_full[i, j, k] = values[index]
                index += 1

    combination_list = {"X1_X2": [x1_min, x1_max, x2_min, x2_max],
                        "X1_X3": [x1_min, x1_max, x3_min, x3_max],
                        "X2_X3": [x2_min, x2_max, x3_min, x3_max]}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (combination, minmaxlist) in enumerate(combination_list.items()):
        xlabel, ylabel = combination.split("_")[0], combination.split("_")[1]

        ax = axes[i]

        x_width = (minmaxlist[1] - minmaxlist[0]) / pointsPerAxis
        y_width = (minmaxlist[3] - minmaxlist[2]) / pointsPerAxis

        x = np.arange(minmaxlist[0], minmaxlist[1], x_width)
        y = np.arange(minmaxlist[2], minmaxlist[3], y_width)

        x, y = np.meshgrid(x, y, indexing='ij')
        if combination == "X1_X2":
            z = z_full[:,:,0]
            ax.set_ylim(-10,600)
        elif combination == "X1_X3":
            z = z_full[:,0,:]
            ax.set_ylim(0,500)
        elif combination == "X2_X3":
            z = z_full[0,:,:]
            ax.set_xlim(0,500)
            ax.set_ylim(0,400)

        xlabel = variable_list[xlabel]
        ylabel = variable_list[ylabel]

        im = ax.contourf(x, y, z, cmap='coolwarm')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{xlabel} vs {ylabel}')
        fig.colorbar(im, ax=ax)

def scatterPlot(dimensions, alph, limitsPath, samplePath,variable_list):

    limits = np.array(pd.read_parquet(limitsPath))[-1,-1]
    values = np.array(pd.read_parquet(samplePath))[-1,-1]

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
    (plotDensity,(256,0.0002, limitsPath, valuesPath,variable_list)),
    (plotDensity2D,(256,0.0002, limitsPath, valuesPath,variable_list)),
    #(scatterPlot,(3, 1, limitsPath, samplePath,variable_list)),
]

save_plots_to_pdf(savePath + saveFileName, plot_functions)


