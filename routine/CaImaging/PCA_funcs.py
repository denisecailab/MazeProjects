import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from IPython import display


def PCAthroughTime(projections, samplingRate=1/30, color='salmon', saveSeries=False, path='.'):
    '''
    Parameters
    ==========
    projections : 2D numpy array
        Numpy array with at least 3 rows corresponding with projections of rates onto principal components, columns corresponding with values for each projection
    samplingRate : double
        Value representing the sampling rate (in seconds) of the recording. Default is 1/30.
    color : string or list/1D array
        If color is a string, all points will plot that color. If color is a list or 1D array, the color of each data point will correspond with the value at that index of color. This is useful when you would like to correlate position in state space with another modality. Length of color in this case should be equal to number of columns in projections. Default is 'salmon'.
    saveSeries : boolean
        Should the images in this animation series be saved? Default is False.
    path : str
        This is only used if saveSeries is True. This should be a string to the directory where you would like the images to be saved. The files will be saved as 001.png, 002.png, etc. Default path is '.'.
    
    Returns
    =======
    Series of images plotted dynamically to appear as a video of progression through state space in time.
    '''
    for i in range(projections.shape[1]-1):
        try:
            # Make plot
            fig = plt.figure(figsize=(22,10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points
            ax.plot(projections[0,:i+1], projections[1,:i+1], projections[2,:i+1], c='grey', alpha=0.4, linewidth=1) # grey line
            ax.scatter(projections[0,0], projections[1,0], projections[2,0], marker='*', c='green', s=100) # start timepoint as green star
            if isinstance(color,str):
                scatterThroughTime = ax.scatter(projections[0,:i+1], projections[1,:i+1], projections[2,:i+1], '.', c=color, s=30, alpha=0.4) # add each next point
            else:
                scatterThroughTime = ax.scatter(projections[0,:i+1], projections[1,:i+1], projections[2,:i+1], '.',
                                                c=color[:i+1], cmap='viridis', vmax=color.max(), s=30, alpha=0.4) # add each next point
                fig.colorbar(scatterThroughTime)
            ax.scatter(projections[0,i], projections[1,i], projections[2,i], '.', c='darkred', s=30) # mark current point as dark red
            
            # Add current time and coordinates
            timepatch = mpatches.Patch(color='white',label=("Time: " + str(np.round(i*samplingRate, decimals=3)) + 'sec'))
            coordpatch = mpatches.Patch(color='white', label="Coord: " + str(np.round(projections[0,i+1], decimals=3)) + ',' +
                                        str(np.round(projections[1,i+1], decimals=3)) + ',' +
                                        str(np.round(projections[2,i+1], decimals=3)))
            plt.legend(handles = [coordpatch, timepatch],loc='upper right',fontsize=15)
            
            # Fix axes in place from the start to the min and max of each axes
            ax.set_xlim(projections[0,:].min(),projections[0,:].max())
            ax.set_ylim(projections[1,:].min(),projections[1,:].max())
            ax.set_zlim(projections[2,:].min(),projections[2,:].max())
            ax.set_xlabel('First Principal Component', fontsize=10)
            ax.set_ylabel('Second Principal Component', fontsize=10)
            ax.set_zlabel('Third Principal Component', fontsize=10)
            
            if saveSeries:
                fig.savefig(path + '/' + str(i).zfill(len(str(projections.shape[1]))) + '.png', dpi=80)
            
            # Clear figure
            display.clear_output(wait=True)
            display.display(pl.gcf())
            time.sleep(0.00001)
            plt.close(fig)
            
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
            break


def runPCA(rates, numProjections=[0,1,2], numVectors=[0,1,2], plotEigenvalues=True, plotEigenvectors=True):
    '''
    Parameters
    ==========
    rates : 2D numpy array
        Array where each row will be collapsed. If you would like each column to be collapsed instead, provide the rates transposed (i.e. rates.T).
    numProjections : list
        List of which dimensions to use in the projection. Default is [0,1,2].
    numVectors : list
        List of which eigenvectors to use for plotting. This is only used if plotEigenvectors is True. Default is [0,1,2].
    plotEigenvalues : boolean
        Should the eigenvalues be plotted as a scree plot?
    plotEigenvectors : boolean
        Should the eigenvectors be plotted as a stem plot?
    
    Returns
    =======
    evalues : 1D numpy array
        1D array the same length as number of columns in rates. These correspond to the eigenvalues of the PCA. They are sorted from largest to smallest.
    evectors : 2D numpy array
        2D square array whose rows and columns are the number of columns in rates. Each column is an eigenvector of the PCA. They are sorted based on evalues.
    projections : 2D numpy array
        2D array of the projected rates onto the principal components specified in numProjections.
    '''
    rates -= np.mean(rates, axis=0)
    covMat = (1.0/(rates.shape[0]-1))*(rates.T @ rates) # covariance matrix
    evectors, evalues, V = np.linalg.svd(covMat)
    projections = np.zeros((len(numProjections),rates.shape[0]))
    for n,x in enumerate(numProjections):
        projections[n,:] = np.dot(rates, evectors.T[x]) # projections of principal components onto firing rates
    
    if plotEigenvalues:
        plt.figure(figsize=(20,8))
        plt.scatter(x=np.arange(len(evalues)), y=evalues, marker='o', s=20)
        plt.xlabel('Eigenvalues (Principal Components)', fontsize=20)
        plt.ylabel('Explained Variance', fontsize=20)
        plt.title('Explained Variance Per Principal Component', fontsize=30)
        plt.show()
    if plotEigenvectors:
        plt.figure(figsize=(30,10), dpi=80)
        for i,n in enumerate(numVectors):
            plt.scatter(range(evectors.shape[1]), evectors.T[n] - i, s=10)
            plt.vlines(x=range(evectors.shape[1]), ymin=-i, ymax=-i+evectors.T[n], linewidth=2, alpha=0.5)
            plt.hlines(y=-i, xmin=0, xmax=evectors.shape[1], linewidth=1, alpha=0.5)
            plt.text(x=-4, y=-i, s='PC ' + str(n+1), fontsize=15, c='black')
        plt.vlines(x=range(evectors.shape[1]), ymin=-len(numVectors)+1, ymax=0, linestyles=':', alpha=0.2)
        unitnames = [str(x) for x in range(evectors.shape[1])]
        for txt in range(evectors.shape[1]):
            plt.text(y = -len(numVectors)/2, x = np.arange(evectors.shape[1])[txt], s = unitnames[txt], fontsize=7, horizontalalignment='center')
        plt.ylabel('Loadings', fontsize=20)
        plt.xlabel('Units', fontsize=20)
        plt.title('Eigenvectors for first ' + str(len(numVectors)) + " PCs", fontsize=30)
        plt.show()
    
    return evalues, evectors, projections

def projectionPlot3D(projections, color='salmon'):
    '''
    Parameters
    ==========
    projections : 2D numpy array
        2D array of projected rates onto principal components
    color : str or list/1D array
        If color is a string, all points will plot that color. If color is a list or 1D array, the color of each data point will correspond with the value at that index of color. This is useful when you would like to correlate position in state space with another modality. Length of color in this case should be equal to number of columns in projections. Default is 'salmon'.
    
    Returns
    =======
        3D plot of projections colored by specified colors
    '''
    fig = plt.figure(figsize=(25,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(projections[0,:], projections[1,:], projections[2,:], c='grey', alpha=0.4, linewidth=1)
    ax.scatter(projections[0,0], projections[1,0], projections[2,0], marker='*', c='green', s=100) # start timepoint
    scatterThroughTime = ax.scatter(projections[0,:], projections[1,:], projections[2,:], '.', c = color, cmap='viridis', s=30, alpha=0.4)
    fig.colorbar(scatterThroughTime)
    ax.set_xlabel('First Principal Component', fontsize=10)
    ax.set_ylabel('Second Principal Component', fontsize=10)
    ax.set_zlabel('Third Principal Component', fontsize=10)
    fig.show()
