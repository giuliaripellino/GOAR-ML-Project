import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from matplotlib import cm, cbook
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LightSource
from itertools import combinations

try:
    rootPath = sys.argv[1]
    pointsPerAxis = sys.argv[2]
    dimensions = sys.argv[3]
    print(f"Using rootPath: '{rootPath}', ppAxis={pointsPerAxis} & dim={dimensions} from 'SparksInTheDarkMain.scala'")
except:
    rootPath = "output/4D_plotting"
    pointsPerAxis = 64
    dimensions = 4
    print(f"Failed to fetch rootPath from SparksInTheDarkMain.scala... using '{rootPath}/' instead")
    print(f"Failed to fetch pointsPerAxis from SparksInTheDarkMain.scala... using pointsPerAxis={pointsPerAxis}")
    print(f"Failed to fetch dimensions from SparksInTheDarkMain.scala... using dim={dimensions} instead")

limitsPath = f"../SparksInTheDark/{rootPath}/limits/"
valuesPath = f"../SparksInTheDark/{rootPath}/plotValues/"
samplePath = f"../SparksInTheDark/{rootPath}/sample/"

# Path to original parquet file -----------------------------------------
inputDataPath = f"../SparksInTheDark/output/ntuple_em_v2_scaled.parquet"
# -----------------------------------------------------------------------

savePath = f"../SparksInTheDark/{rootPath}/"
saveFileName = "figures.pdf"

variable_list = {"X1":r"$\Delta R(l,b_2)$","X2":r"$m_{J^{lep}} + m_{J^{had}}$", "X3":r"$m_{bb\Delta R_{min}}$","X4":r"$H_T$"}
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

def plotDensity(pointsPerAxis, z_max, limitsPath, valuesPath, variable_list):
    limits = np.array(pd.read_parquet(limitsPath))[-1,-1]
    values = np.array(pd.read_parquet(valuesPath))[-1,-1]

    for i in range(0, len(limits), 2):
        print(f"Variable {i//2 + 1} limits: ({limits[i]}, {limits[i+1]})")

    x1_min, x1_max = limits[0], limits[1]
    x2_min, x2_max = limits[2], limits[3]
    x3_min, x3_max = limits[4], limits[5]
    x4_min, x4_max = limits[6], limits[7]

    z_full = np.zeros((pointsPerAxis, pointsPerAxis, pointsPerAxis, pointsPerAxis))
    index = 0
    for i in range(pointsPerAxis):
        for j in range(pointsPerAxis):
            for k in range(pointsPerAxis):
                for l in range(pointsPerAxis):
                    z_full[i, j, k, l] = values[index]
                    index += 1

    combination_list = {
        "X1_X2": [x1_min, x1_max, x2_min, x2_max],
        "X1_X3": [x1_min, x1_max, x3_min, x3_max],
        "X1_X4": [x1_min, x1_max, x4_min, x4_max],  # New
        "X2_X3": [x2_min, x2_max, x3_min, x3_max],
        "X2_X4": [x2_min, x2_max, x4_min, x4_max],  # New
        "X3_X4": [x3_min, x3_max, x4_min, x4_max]   # New
    }

    fig, axes = plt.subplots(2, 3, subplot_kw={"projection": "3d"}, figsize=(12, 7))
    axes = axes.flatten()

    for i, (combination, minmaxlist) in enumerate(combination_list.items()):
        xlabel, ylabel = combination.split("_")
        ax = axes[i]

        x_width = (minmaxlist[1] - minmaxlist[0]) / pointsPerAxis
        y_width = (minmaxlist[3] - minmaxlist[2]) / pointsPerAxis

        x = np.arange(minmaxlist[0], minmaxlist[1], x_width)
        y = np.arange(minmaxlist[2], minmaxlist[3], y_width)

        x, y = np.meshgrid(x, y, indexing='ij')

        if combination == "X1_X2":
            z = z_full[:,:,0,0]
        elif combination == "X1_X3":
            z = z_full[:,0,:,0]
        elif combination == "X1_X4":
            z = z_full[:,0,0,:]
        elif combination == "X2_X3":
            z = z_full[0,:,:,0]
        elif combination == "X2_X4":
            z = z_full[0,:,0,:]
        elif combination == "X3_X4":
            z = z_full[0,0,:,:]

        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        xlabel = variable_list.get(xlabel, xlabel)
        ylabel = variable_list.get(ylabel, ylabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()

def plotDensity2D(pointsPerAxis, z_max, limitsPath, valuesPath,variable_list):
    limits = np.array(pd.read_parquet(limitsPath))[-1,-1]
    values = np.array(pd.read_parquet(valuesPath))[-1,-1]

    for i in range(0, len(limits), 2):
        print(f"Variable {i//2 + 1} limits: ({limits[i]}, {limits[i+1]})")

    x1_min, x1_max = limits[0], limits[1]
    x2_min, x2_max = limits[2], limits[3]
    x3_min, x3_max = limits[4], limits[5]
    x4_min, x4_max = limits[6], limits[7]

    z_full = np.zeros((pointsPerAxis, pointsPerAxis, pointsPerAxis, pointsPerAxis))
    index = 0
    for i in range(pointsPerAxis):
        for j in range(pointsPerAxis):
            for k in range(pointsPerAxis):
                for l in range(pointsPerAxis):
                    z_full[i, j, k, l] = values[index]
                    index += 1

    combination_list = {
        "X1_X2": [x1_min, x1_max, x2_min, x2_max],
        "X1_X3": [x1_min, x1_max, x3_min, x3_max],
        "X1_X4": [x1_min, x1_max, x4_min, x4_max],
        "X2_X3": [x2_min, x2_max, x3_min, x3_max],
        "X2_X4": [x2_min, x2_max, x4_min, x4_max],
        "X3_X4": [x3_min, x3_max, x4_min, x4_max]
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    axes = axes.flatten()

    for i, (combination, minmaxlist) in enumerate(combination_list.items()):
        xlabel, ylabel = combination.split("_")
        ax = axes[i]

        x_width = (minmaxlist[1] - minmaxlist[0]) / pointsPerAxis
        y_width = (minmaxlist[3] - minmaxlist[2]) / pointsPerAxis

        x = np.arange(minmaxlist[0], minmaxlist[1], x_width)
        y = np.arange(minmaxlist[2], minmaxlist[3], y_width)

        x, y = np.meshgrid(x, y, indexing='ij')

        if combination == "X1_X2":
            z = z_full[:,:,0,0]
        elif combination == "X1_X3":
            z = z_full[:,0,:,0]
        elif combination == "X2_X3":
            z = z_full[0,:,:,0]
        elif combination == "X1_X4":
            z = z_full[:,0,0,:]
        elif combination == "X2_X4":
            z = z_full[0,:,0,:]
        elif combination == "X3_X4":
            z = z_full[0,0,:,:]

        xlabel = variable_list.get(xlabel, xlabel)
        ylabel = variable_list.get(ylabel, ylabel)

        im = ax.contourf(x, y, z, cmap='coolwarm')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{xlabel} vs {ylabel}')
        fig.colorbar(im, ax=ax)

def plotHeatmaps(dimensions, limitsPath, samplePath, bins=100, cmap='coolwarm',variables = variable_list2):

    limits = np.array(pd.read_parquet(limitsPath))[-1, -1]
    values = np.array(pd.read_parquet(samplePath))[-1, -1]

    num_plots = int(dimensions * (dimensions - 1) / 2)
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(15,5))
    fig.suptitle("DISTRIBUTIONS FROM METHOD")

    length = int(len(values) / dimensions)
    xs = np.ndarray(shape=(dimensions, length))
    for i in range(dimensions):
        for j in range(length):
            xs[i, j] = values[dimensions * j + i]

    arrays_dict = {}
    for dimension in range(dimensions):
        arrays_dict[dimension] = xs[dimension,:]

    axis_limits = {
            r"$\Delta R(l,b_2)$": [0, 0.8],
            r"$m_{J^{lep}} + m_{J^{had}}$": [0, 0.4],
            r"$m_{bb\Delta R_{min}}$": [0, 0.23],
            r"$H_T$": [0,0.26]
        }

    pairs = [
        ((arrays_dict[0], arrays_dict[1]), (r"$\Delta R(l,b_2)$", r"$m_{J^{lep}} + m_{J^{had}}$")),
        ((arrays_dict[0], arrays_dict[2]), (r"$\Delta R(l,b_2)$", r"$m_{bb\Delta R_{min}}$")),
        ((arrays_dict[0], arrays_dict[3]), (r"$\Delta R(l,b_2)$", r"$H_T$")),
        ((arrays_dict[1], arrays_dict[2]), (r"$m_{J^{lep}} + m_{J^{had}}$", r"$m_{bb\Delta R_{min}}$")),
        ((arrays_dict[1], arrays_dict[3]), (r"$m_{J^{lep}} + m_{J^{had}}$", r"$H_T$")),
        ((arrays_dict[2], arrays_dict[3]), (r"$m_{bb\Delta R_{min}}$", r"$H_T$"))
    ]

    for i, ((x, y), (xlabel, ylabel)) in enumerate(pairs):
            mesh = axs[i // 3, i % 3].hist2d(x, y, bins=bins, cmap=cmap)

            if xlabel in axis_limits:
                axs[i // 3, i % 3].set_xlim(*axis_limits[xlabel])
            if ylabel in axis_limits:
                axs[i // 3, i % 3].set_ylim(*axis_limits[ylabel])

            axs[i // 3, i % 3].set_xlabel(xlabel)
            axs[i // 3, i % 3].set_ylabel(ylabel)

            fig.colorbar(mesh[3], ax=axs[i // 3, i % 3])

def plot_all_permutations(dimensions=4, bins=100, cmap='coolwarm'):
    scaled_input_data = pd.read_parquet(inputDataPath)

    x1 = scaled_input_data["deltaRLep2ndClosestBJet"]
    x2 = scaled_input_data["LJet_m_plus_RCJet_m_12"]
    x3 = scaled_input_data["bb_m_for_minDeltaR"]
    x4 = scaled_input_data["HT"]

    num_plots = int(dimensions * (dimensions - 1) / 2)
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5))
    fig.suptitle("ORIGINAL DISTRIBUTIONS")

    axis_limits = {
        r"$\Delta R(l,b_2)$": [0, 0.8],
        r"$m_{J^{lep}} + m_{J^{had}}$": [0, 0.4],
        r"$m_{bb\Delta R_{min}}$": [0, 0.23],
        r"$H_T$": [0,0.26]
    }

    pairs = [
        ((x1, x2), (r"$\Delta R(l,b_2)$", r"$m_{J^{lep}} + m_{J^{had}}$")),
        ((x1, x3), (r"$\Delta R(l,b_2)$", r"$m_{bb\Delta R_{min}}$")),
        ((x1, x4), (r"$\Delta R(l,b_2)$", r"$H_T$")),
        ((x2, x3), (r"$m_{J^{lep}} + m_{J^{had}}$", r"$m_{bb\Delta R_{min}}$")),
        ((x2, x4), (r"$m_{J^{lep}} + m_{J^{had}}$", r"$H_T$")),
        ((x3, x4), (r"$m_{bb\Delta R_{min}}$", r"$H_T$"))
    ]

    for i, ((x, y), (xlabel, ylabel)) in enumerate(pairs):
        mesh = axs[i // 3, i % 3].hist2d(x, y, bins=bins, cmap=cmap)

        if xlabel in axis_limits:
            axs[i // 3, i % 3].set_xlim(*axis_limits[xlabel])
        if ylabel in axis_limits:
            axs[i // 3, i % 3].set_ylim(*axis_limits[ylabel])

        axs[i // 3, i % 3].set_xlabel(xlabel)
        axs[i // 3, i % 3].set_ylabel(ylabel)

        fig.colorbar(mesh[3], ax=axs[i // 3, i % 3])


plot_functions = [
    (plotDensity,(pointsPerAxis,0.0002, limitsPath, valuesPath,variable_list)),
    (plotDensity2D,(pointsPerAxis,0.0002, limitsPath, valuesPath,variable_list)),
    (plot_all_permutations,()),
    (plotHeatmaps,(dimensions,limitsPath,samplePath)),
]

save_plots_to_pdf(savePath + saveFileName, plot_functions)


