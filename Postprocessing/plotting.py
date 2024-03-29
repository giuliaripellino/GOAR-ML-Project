import numpy as np
import pandas as pd
import sys
import csv
from re import findall

import matplotlib.pyplot as plt
from matplotlib import cm, cbook
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LightSource
from itertools import combinations
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.ticker import FixedLocator

try:
    rootPath = sys.argv[1]
    pointsPerAxis = int(sys.argv[2])
    dimensions = int(sys.argv[3])
    inputDataPath = sys.argv[4]
    colStrings = sys.argv[5]
    print(f"Using rootPath: '{rootPath}', ppAxis={pointsPerAxis}, dim={dimensions} & inputDataPath='{inputDataPath}' from 'SparksInTheDarkMain.scala'")
except:
    print("CURRENTLY NOT RUNNABLE ON ITS OWN")
    sys.exit(1)
    rootPath = "output/4D_plotting_tailProb100_bkg_count1_res1e-7"
    pointsPerAxis = 64
    dimensions = 4
    print(f"Failed to fetch rootPath from SparksInTheDarkMain.scala... using '{rootPath}' instead")
    print(f"Failed to fetch pointsPerAxis from SparksInTheDarkMain.scala... using pointsPerAxis={pointsPerAxis}")
    print(f"Failed to fetch dimensions from SparksInTheDarkMain.scala... using dim={dimensions} instead")

def extract_column_names(s): return findall(r'\b\w+\b', s)
colList = extract_column_names(colStrings)

limitsPath = f"../SparksInTheDark/{rootPath}/limits/"
valuesPath = f"../SparksInTheDark/{rootPath}/plotValues/"
samplePath = f"../SparksInTheDark/{rootPath}/sample/"
savePath = f"../SparksInTheDark/{rootPath}/"
if 'SU2L' in inputDataPath:
    scaling_factors_path = f"../scaling_factors_signal.csv"
else: scaling_factors_path = f"../scaling_factors_bkg.csv"

saveFileName = f"figures_{colList[1]}_{colList[2]}.pdf"
variable_list = {"deltaRLep2ndClosestBJet":r"$\Delta R(l,b_2)$","LJet_m_plus_RCJet_m_12":r"$m_{J^{lep}} + m_{J^{had}}$ [GeV]", "bb_m_for_minDeltaR":r"$m_{bb_{\Delta R_{min}}}$ [GeV]","HT":r"$H_T$ [GeV]"}

def UnScaling(scaling_factors_path,variables):
    scaling_factors = {}
    with open(scaling_factors_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            feature = row['Variable']
            min_val = float(row['Min'])
            max_val = float(row['Max'])
            scaling_factors[feature] = {'min': min_val, 'max': max_val}
    return scaling_factors

scaling_factors = UnScaling(scaling_factors_path,variable_list.keys())

def original_value(min_val, max_val, normalized_val): return min_val + (max_val - min_val) * normalized_val
normalized_ticks = np.linspace(0, 1, 10)


def save_plots_to_pdf_and_eps(file_path, plot_functions):
    base_file_path = file_path.rsplit('.', 1)[0]
    with PdfPages(file_path) as pdf:
        for index, (plot_function, arguments) in enumerate(plot_functions):
            plt.figure()
            plot_function(*arguments)
            #plt.tight_layout()
            pdf.savefig()
            eps_file_path = f"{base_file_path}_{index+1}.eps"
            plt.savefig(eps_file_path, format='eps',bbox_inches='tight')
            plt.close()
    print(f"Plots saved as {file_path} and individual .eps files")

def plotDensity3D(pointsPerAxis, limitsPath, valuesPath, colStrings):
    limits = np.array(pd.read_parquet(limitsPath))[-1, -1]
    values = np.array(pd.read_parquet(valuesPath))[-1, -1]

    x_min, x_max, y_min, y_max = limits[0], limits[1], limits[2], limits[3]
    x_width = (x_max - x_min) / pointsPerAxis
    y_width = (y_max - y_min) / pointsPerAxis
    x = np.arange(x_min, x_max, x_width)
    y = np.arange(y_min, y_max, y_width)
    x, y = np.meshgrid(x, y)

    z = np.empty((pointsPerAxis, pointsPerAxis))
    for i in range(pointsPerAxis):
        for j in range(pointsPerAxis):
            z[i, j] = values[i * pointsPerAxis + j]
    z = z.T

    x_ticks = [original_value(scaling_factors[colStrings[1]]['min'], scaling_factors[colStrings[1]]['max'], val) for val in normalized_ticks]
    y_ticks = [original_value(scaling_factors[colStrings[2]]['min'], scaling_factors[colStrings[2]]['max'], val) for val in normalized_ticks]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, ax=ax, label=r"$f_n$", location='left', aspect=30,pad=0.01,shrink=0.7)
    ax.set_xlabel(variable_list[colStrings[1]])
    ax.set_ylabel(variable_list[colStrings[2]])
    ax.xaxis.set_major_locator(FixedLocator(normalized_ticks))
    ax.yaxis.set_major_locator(FixedLocator(normalized_ticks))
    ax.set_xticklabels([f'{x:.0f}' for x in x_ticks], fontsize=8, rotation=45)
    ax.set_yticklabels([f'{y:.0f}' for y in y_ticks], fontsize=8, rotation=-45)
    ax.tick_params(axis='x', labelsize=8, pad=0.01)
    ax.tick_params(axis='y', labelsize=8, pad=0.01)
    ax.tick_params(axis='z', labelsize=8, pad=0.5)
    ax.invert_xaxis()

def plotDensity2D(pointsPerAxis, limitsPath, valuesPath, colStrings):
    limits = np.array(pd.read_parquet(limitsPath))[-1, -1]
    values = np.array(pd.read_parquet(valuesPath))[-1, -1]

    x_min, x_max, y_min, y_max = limits[0], limits[1], limits[2], limits[3]
    x_width = (x_max - x_min) / pointsPerAxis
    y_width = (y_max - y_min) / pointsPerAxis
    x = np.arange(x_min, x_max, x_width)
    y = np.arange(y_min, y_max, y_width)

    z = np.empty((pointsPerAxis, pointsPerAxis))
    for i in range(pointsPerAxis):
        for j in range(pointsPerAxis):
            z[i, j] = values[i * pointsPerAxis + j]
    z = z.T

    # Preparing the ticks for the x and y axes based on the scaling factors and normalized ticks
    x_ticks = [original_value(scaling_factors[colStrings[1]]['min'], scaling_factors[colStrings[1]]['max'], val) for val in normalized_ticks]
    y_ticks = [original_value(scaling_factors[colStrings[2]]['min'], scaling_factors[colStrings[2]]['max'], val) for val in normalized_ticks]

    fig, ax = plt.figure(figsize=(8, 6),tight_layout=True), plt.gca()
    heatmap = ax.imshow(z, cmap='coolwarm', extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
    fig.colorbar(heatmap, ax=ax, label=r"$f_n$")

    # Setting the major locator for x and y axes
    ax.xaxis.set_major_locator(FixedLocator(normalized_ticks))
    ax.yaxis.set_major_locator(FixedLocator(normalized_ticks))

    # Adjusting the tick labels for readability
    ax.set_xticklabels([f'{x:.0f}' for x in x_ticks], fontsize=8)
    ax.set_yticklabels([f'{y:.0f}' for y in y_ticks], fontsize=8)

    # Setting the labels for the axes
    ax.set_xlabel(variable_list[colStrings[1]])
    ax.set_ylabel(variable_list[colStrings[2]])


def scatterPlot(dimensions, limitsPath, samplePath,colStrings):
    limits = np.array(pd.read_parquet(limitsPath))[-1,-1]
    values = np.array(pd.read_parquet(samplePath))[-1,-1]
    scaled_input_data = pd.read_parquet(inputDataPath)

    length = int(len(values) / dimensions)

    xs = np.ndarray(shape=(dimensions, length))
    for i in range(dimensions):
        for j in range(length):
            xs[i,j] = values[dimensions*j + i]

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    x_ticks = [original_value(scaling_factors[colStrings[1]]['min'], scaling_factors[colStrings[1]]['max'], val) for val in normalized_ticks]
    y_ticks = [original_value(scaling_factors[colStrings[2]]['min'], scaling_factors[colStrings[2]]['max'], val) for val in normalized_ticks]

    xbins_original = np.linspace(scaled_input_data[colStrings[1]].min(), scaled_input_data[colStrings[1]].max(), 50)
    ybins_original = np.linspace(scaled_input_data[colStrings[2]].min(), scaled_input_data[colStrings[2]].max(), 50)
    h_original, xedges_original, yedges_original, img_original = axs[0].hist2d(scaled_input_data[colStrings[1]], scaled_input_data[colStrings[2]], bins=[xbins_original, ybins_original], cmap='coolwarm')
    fig.colorbar(img_original, ax=axs[0], label='Counts')
    axs[0].set_xlabel(variable_list[colStrings[1]])
    axs[0].set_ylabel(variable_list[colStrings[2]])

    axs[0].xaxis.set_major_locator(FixedLocator(normalized_ticks))
    axs[0].yaxis.set_major_locator(FixedLocator(normalized_ticks))
    axs[0].set_xticklabels([f'{x:.1f}' for x in x_ticks],fontsize=8)
    axs[0].set_yticklabels([f'{y:.1f}' for y in y_ticks],fontsize=8)


    axs[0].set_title("Original Distribution")


    xbins = np.linspace(xs[0, :].min(), xs[0, :].max(), 50)
    ybins = np.linspace(xs[1, :].min(), xs[1, :].max(), 50)
    h, xedges, yedges, img = axs[1].hist2d(xs[0, :], xs[1, :], bins=[xbins, ybins], cmap='coolwarm')
    fig.colorbar(img, ax=axs[1], label='Counts')
    axs[1].set_xlabel(variable_list[colStrings[1]])
    axs[1].set_ylabel(variable_list[colStrings[2]])
    axs[1].set_xlim(0,1)
    axs[1].set_ylim(0,1)
    axs[1].xaxis.set_major_locator(FixedLocator(normalized_ticks))
    axs[1].yaxis.set_major_locator(FixedLocator(normalized_ticks))
    axs[1].set_xticklabels([f'{x:.1f}' for x in x_ticks],fontsize=8)
    axs[1].set_yticklabels([f'{y:.1f}' for y in y_ticks],fontsize=8)
    axs[1].set_title("Distribution from method")

plot_functions = [
    (plotDensity3D,(pointsPerAxis, limitsPath, valuesPath,colList)),
    (plotDensity2D,(pointsPerAxis, limitsPath, valuesPath,colList)),
    (scatterPlot,(dimensions,limitsPath,samplePath,colList)),
]

save_plots_to_pdf_and_eps(savePath + saveFileName, plot_functions)