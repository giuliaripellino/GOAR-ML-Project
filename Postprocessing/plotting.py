import numpy as np
import pandas as pd
import sys
import re

import matplotlib.pyplot as plt
from matplotlib import cm, cbook
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LightSource
from itertools import combinations
from matplotlib.patches import Rectangle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d

try:
    rootPath = sys.argv[1]
    pointsPerAxis = int(sys.argv[2])
    dimensions = int(sys.argv[3])
    inputDataPath = sys.argv[4]
    colStrings = sys.argv[5]
    print(f"Using rootPath: '{rootPath}', ppAxis={pointsPerAxis}, dim={dimensions} & inputDataPath='{inputDataPath}' from 'SparksInTheDarkMain.scala'")
except:
    rootPath = "output/4D_plotting_tailProb100_bkg_count1_res1e-7"
    pointsPerAxis = 64
    dimensions = 4
    print(f"Failed to fetch rootPath from SparksInTheDarkMain.scala... using '{rootPath}' instead")
    print(f"Failed to fetch pointsPerAxis from SparksInTheDarkMain.scala... using pointsPerAxis={pointsPerAxis}")
    print(f"Failed to fetch dimensions from SparksInTheDarkMain.scala... using dim={dimensions} instead")
def extract_column_names(s): return re.findall(r'\b\w+\b', s)
colList = extract_column_names(colStrings)
limitsPath = f"../SparksInTheDark/{rootPath}/limits/"
valuesPath = f"../SparksInTheDark/{rootPath}/plotValues/"
samplePath = f"../SparksInTheDark/{rootPath}/sample/"
savePath = f"../SparksInTheDark/{rootPath}/"
saveFileName = f"figures_{colList[1]}_{colList[2]}.pdf"

variable_list = {"deltaRLep2ndClosestBJet":r"$\Delta R(l,b_2)$","LJet_m_plus_RCJet_m_12":r"$m_{J^{lep}} + m_{J^{had}}$", "bb_m_for_minDeltaR":r"$m_{bb\Delta R_{min}}$","HT":r"$H_T$"}
variable_list2 = [r"$\Delta R(l,b_2)$", r"$m_{J^{lep}} + m_{J^{had}}$", r"$m_{bb\Delta R_{min}}$", r"$H_T$"]

def save_plots_to_pdf(file_path, plot_functions):
    with PdfPages(file_path) as pdf:
        for plot_function, arguments in plot_functions:
            plt.figure()
            plot_function(*arguments)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"Plots saved as {file_path}")

def plotDensity(pointsPerAxis, z_max, limitsPath, valuesPath,colStrings):
    limits = np.array(pd.read_parquet(limitsPath))[-1,-1]
    values = np.array(pd.read_parquet(valuesPath))[-1,-1]

    # Extract limits for x and y
    x_min, x_max, y_min, y_max = limits[0], limits[1], limits[2], limits[3]

    # Compute widths and ranges for x and y axes
    x_width = (x_max - x_min) / pointsPerAxis
    y_width = (y_max - y_min) / pointsPerAxis

    # Generate meshgrid for the axes
    x = np.arange(x_min, x_max, x_width)
    y = np.arange(y_min, y_max, y_width)
    x, y = np.meshgrid(x, y)

    z = np.empty((pointsPerAxis,pointsPerAxis))
    for i in range(pointsPerAxis):
        for j in range(pointsPerAxis):
            z[i,j] = values[i*pointsPerAxis + j]

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Density Estimates")
    # Add 3D subplot for surface
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax3d.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10,label=r"$f_d$")
    ax3d.set_xlabel(variable_list[colStrings[1]])
    ax3d.set_ylabel(variable_list[colStrings[2]])
    ax3d.invert_xaxis()

    # Add 2D subplot for heatmap
    ax2d = fig.add_subplot(1, 2, 2)
    heatmap = ax2d.imshow(z, cmap=cm.coolwarm, extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
    fig.colorbar(heatmap, ax=ax2d, shrink=0.5, aspect=10,label=r"$f_d$")
    ax2d.set_xlabel(variable_list[colStrings[1]])
    ax2d.set_ylabel(variable_list[colStrings[2]])

def scatterPlot(dimensions, alph, limitsPath, samplePath,colStrings):

    limits = np.array(pd.read_parquet(limitsPath))[-1,-1]
    values = np.array(pd.read_parquet(samplePath))[-1,-1]
    scaled_input_data = pd.read_parquet(inputDataPath)

    length = int(len(values) / dimensions)

    xs = np.ndarray(shape=(dimensions, length))
    for i in range(dimensions):
        for j in range(length):
            xs[i,j] = values[dimensions*j + i]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    xbins_original = np.linspace(scaled_input_data[colStrings[1]].min(), scaled_input_data[colStrings[1]].max(), 50)
    ybins_original = np.linspace(scaled_input_data[colStrings[2]].min(), scaled_input_data[colStrings[2]].max(), 50)

    h_original, xedges_original, yedges_original, img_original = axs[0].hist2d(scaled_input_data[colStrings[1]], scaled_input_data[colStrings[2]], bins=[xbins_original, ybins_original], cmap='coolwarm')
    fig.colorbar(img_original, ax=axs[0], label='Counts')
    axs[0].set_xlabel(variable_list[colStrings[1]])
    axs[0].set_ylabel(variable_list[colStrings[2]])
    axs[0].set_title("Original Distribution")


    xbins = np.linspace(xs[0, :].min(), xs[0, :].max(), 50)
    ybins = np.linspace(xs[1, :].min(), xs[1, :].max(), 50)
    h, xedges, yedges, img = axs[1].hist2d(xs[0, :], xs[1, :], bins=[xbins, ybins], cmap='coolwarm')
    fig.colorbar(img, ax=axs[1], label='Counts')
    axs[1].set_xlabel(variable_list[colStrings[1]])
    axs[1].set_ylabel(variable_list[colStrings[2]])
    axs[1].set_xlim(0,1)
    axs[1].set_ylim(0,1)
    axs[1].set_title("Distribution from method")

plot_functions = [
    (plotDensity,(pointsPerAxis,0.0002, limitsPath, valuesPath,colList)),
    (scatterPlot,(dimensions,1,limitsPath,samplePath,colList)),
]

save_plots_to_pdf(savePath + saveFileName, plot_functions)


