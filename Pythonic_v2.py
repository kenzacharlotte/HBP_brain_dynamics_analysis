import seaborn as sns
import numpy as np
import yanat
import os
import math
import re
import scipy.stats as st
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from scipy.spatial.distance import pdist, squareform


def read_ftract(path: str)->  np.array:
    """
    Function that loads an np.matrix from a fTRACT file

    PARAMETERS :
    - path : (str) path to the file

    RETURNS :
    - M : (np.array)
    """
    data = []
    f = open(path,"r")
    for lines in f:
        if lines.split()[0]!='#':
            row = list(map(float,lines.split()))
            data.append(row)
    M = np.array(data)
    return M
    
def read_structural_connectivity(path: str, res: int)-> np.array:
    """"
    Load as an np.array  the structural connectivity from a matlab file
    """
    Lausanne = scipy.io.loadmat(path,simplify_cells=True, squeeze_me=True, chars_as_strings=True,)
    return Lausanne["LauConsensus"]["Matrices"][res][0]

def read_functional_connectivity(path: str,res: int)-> np.array:
    """"
    Load as an np.array  the functionnal connectivity from a matlab file
    """
    Lausanne = scipy.io.loadmat(path,simplify_cells=True, squeeze_me=True, chars_as_strings=True,)
    return Lausanne["LauConsensus"]["Matrices"][res][2]

def read_coordinates(path: str, res: int)-> np.array:
    """"
    Load as an np.array the 3D coordinates of the network from a matlab file
    """
    Lausanne = scipy.io.loadmat(path,simplify_cells=True, squeeze_me=True, chars_as_strings=True,)
    return Lausanne["LauConsensus"]["Matrices"][res][3]

def read_fiber_length(path: str, res: int)-> np.array:
    """
    Load as an np.array the fiber length  of the network from a matlab file
    """
    Lausanne = scipy.io.loadmat(path,simplify_cells=True, squeeze_me=True, chars_as_strings=True,)
    return Lausanne["LauConsensus"]["Matrices"][res][1]
    
def calculate_euclidean_distances(coordinates: np.array)-> np.array:
    """
    From the coordinates calculates the euclidian distance and return the matrix
    """
    return squareform(pdist(coordinates))
    
def add_numeration(names: str)-> str:
    """
    Function that add a numerotation at the end of a label when it appears more than one time in the list 
    """
    from collections import defaultdict
    # This dictionary will keep track of the counts of each name
    name_count = defaultdict(int)
    # List to hold the new names with numerations
    new_names = []
    for name in names:
        # Increment the count for this name
        name_count[name] += 1
        # Append the current count to the name to make it unique
        if name_count[name] > 1:
            # Append count only if it's a duplicate
            new_name = f"{name}_{name_count[name]}"
        else:
            new_name = name
        # Add the new name to the list
        new_names.append(new_name)
    return new_names

def reorder_matrix(matlabfile: str, ftractmat: str, ftractlabels: str, res: int=0, connectivity: int=0)-> np.array:
    """
    Function that 
    (1) Loads the SC from Lausanne and its labels
    (2) Loads the Ftract Matrix and its labels

    PARAMETERS:
    matlabfile, ftractmat, ftractlabels: (str) path to the different files
    res: (int) Lausanne resolution index
    connectivity: 0 for structural connectivity, 2 for functionnal connectivity
    RETURNS:
    - Ftract_reloc_pdf: (np.array) Probability matrix reordered according to Lausanne labelling 

    """
    # I-SC Lausanne
    # Loading the mat
    Lausanne = scipy.io.loadmat(matlabfile,simplify_cells=True, squeeze_me=True, chars_as_strings=True,)
    # Reading the labels
    Lausanne_sc = Lausanne["LauConsensus"]["Matrices"][res][connectivity]
    Llabels = Lausanne["LauConsensus"]["Matrices"][res][4][:, 0]
    hemispheres = Lausanne["LauConsensus"]["Matrices"][res][4][:, 3]
    LausanneLabels = []
    for i, j in enumerate(range(Lausanne_sc.shape[0])):
        LausanneLabels.append(str(hemispheres[j]+'.'+Llabels[j]))
    LausanneLabels = add_numeration(LausanneLabels)
    # Create pdf
    Lausanne_pdf = pd.DataFrame(Lausanne_sc, index=LausanneLabels, columns=LausanneLabels)

    # II-Ftract
    # Loading the mat
    Ftract = read_ftract(ftractmat)
    # Reading the labels
    FtractLabels = []
    Flabels = open(ftractlabels,"r")
    for lines in Flabels:
        FtractLabels.append(lines[:-1])
    for i in range(len(FtractLabels)):
        if "_" in FtractLabels[i]:
            FtractLabels[i] = (re.sub(r"_?[0-9]","",FtractLabels[i]))
    FtractLabels = add_numeration(FtractLabels)
    # Create pdf
    Ftract_pdf = pd.DataFrame(Ftract, index=FtractLabels, columns=FtractLabels)
    
    # Reordering Ftract matrix according to Lausanne labelling
    Ftract_reloc_pdf = Ftract_pdf.loc[LausanneLabels, LausanneLabels]
    Ftract_reloc_pdf.fillna(0, inplace=True)
    return Ftract_reloc_pdf.to_numpy()

def condition_function(value)-> bool:
    """
    Return True if the value is equal to 0 or NaN, False otherwise.
    """
    return value == 0 or np.isnan(value)
        
def mask_function(input_matrix: np.ndarray)-> np.array:
    """
    Creates a mask from a given matrix

    Args:
    - input_matrix (np.ndarray): Input matrix of shape (N, N).

    Returns:
    - mask (np.ndarray): Boolean matrix of shape (N, N), where True values indicate that the condition is met.
    """
    vectorized_function = np.vectorize(condition_function)
    mask = vectorized_function(input_matrix)
    return mask

def display_mask(model_matrix: np.ndarray, ftract_matrix: np.ndarray)-> np.array:
    """
    Apply a mask to an input matrix and return the masked given matrix

    Args: 
    - ref_matrix (np.ndarray): Reference matrix for the mask of shape (N, N).
    - input_matrix (np.ndarray): Input matrix of shape (N, N) where N is the number of nodes.
    
    Returns:
    - masked_matrix (np.ndarray): Input matrix of shape (N, N) masked according to the given ref_matrix
    """
    if np.shape(model_matrix)!=np.shape(ftract_matrix):
        raise Exception("Sorry, the 2 matrices don't have the same shape") 
    np.fill_diagonal(model_matrix, 0)
    np.fill_diagonal(ftract_matrix, 0)
    mask_model = mask_function(model_matrix)
    mask_ftract = mask_function(ftract_matrix)
    # Combine the 2 masks
    mask = mask_model|mask_ftract
    masked_model = np.copy(model_matrix)
    masked_model[mask]=0
    masked_ftract= np.copy(ftract_matrix)
    masked_ftract[mask]=0
    
    return masked_model, masked_ftract
    
def get_nonzero_flat_vector(mat: np.array)-> np.array:
    """
    Flattens the matrix, drops all zero entries, and returns a vector that is ready for use in Spearman correlation calculations.
    
    Args:
    - mat (np.array): a 2D matrix 

    Returns:
    - vect (np.array): a 1D numpy array after removing zeros, prepared specifically for Spearman correlation calculation.
    """
    vect = mat.flatten()
    vect = vect[vect != 0]
    return vect


def calculate_spearmanr(mat1: np.array, mat2: np.array)-> Tuple[float, float]:
    """
    Calculates the spearmanr correlation of the 2 matrices and return a tuple: rho and the pvalue associated

    Args:
    - mat1, mat2 (np.array): two 2D natrices

    Returns:
    - rho, pvalue (float): 
    """
    vect1 = get_nonzero_flat_vector(mat1)
    vect2 = get_nonzero_flat_vector(mat2)
    rho, pval = st.spearmanr(vect1,vect2)
    return rho, pval

def plot_spearmanr(model: np.array,ftract: np.array, xlab, ylab):
    """
    Display the scatter plot of the 2 matrices, and print the rho & pvalue
    """
    rho,pval = calculate_spearmanr(model,ftract)
    vect_model = get_nonzero_flat_vector(model)
    vect_ftract = get_nonzero_flat_vector(ftract)

    ax = plt.axes()
    sns.scatterplot(x=yanat.utils.log_normalize(vect_model),y=vect_ftract,s=5,ax=ax,color='k')
    ax.annotate(f"$\\rho = {np.round(rho, 3)}$\np-value = {np.round(pval, 3)}",xy=(1.05, 0.1),fontsize=12,xycoords="axes fraction",)
    ax.set(xlabel= xlab+' log', ylabel=ylab)
    #ax.set_title('Correlation model and ftract')
    sns.despine()

def find_optimal_alpha(structural_connectivity: np.ndarray, ftract: np.ndarray, alpha_interval: np.ndarray = np.linspace(0.0001, 1, 100)) -> float:
    """
    Returns the alpha that maximizes the Spearman correlation score for the LAM model.
    
    Args:
    - structural_connectivity (np.ndarray): The structural connectivity matrix
    - ftract (np.ndarray): Ftract dataset (Probability for example)
    - alpha_interval (np.ndarray): Array of alpha values to test.

    Returns:
    - float: alpha-LAM parameter that results in the highest Spearman correlation.
    """
    rho_values = [calculate_spearmanr(*display_mask(ftract, yanat.core.lam(structural_connectivity, alpha)))[0] 
                  for alpha in alpha_interval]
    optimal_index = np.argmax(rho_values)
    return alpha_interval[optimal_index]
    
def calculate_spearman_scores(structural_connectivity: np.ndarray, ftract: np.ndarray,alpha_interval: np.ndarray = np.linspace(0.0001, 1, 100)) -> np.ndarray:
    """
    Calculates Spearman correlation scores for a range of alpha values in the LAM model.
    Ready for plotting
    
    Parameters:
        structural_connectivity (np.ndarray): The structural connectivity matrix.
        ftract (np.ndarray): Ftract dataset (Probability for example)
        alpha_interval (np.ndarray): Array of alpha values to test.

    Returns:
        np.ndarray: An array of Spearman correlation scores corresponding to each alpha value.
    """
    rho_values = [calculate_spearmanr(*display_mask(ftract, yanat.core.lam(structural_connectivity, alpha)))[0] 
                  for alpha in alpha_interval]

    return np.array(rho_values),alpha_interval

def compute_models(matlabfile_path: str, ftract_path: str, ftractlabels_path: str, res: int)-> dict:
    """
    Given a structural connectivity matrix compute different models and their spearmanr correlation:
    - 'SC': structural connectivity with a spectral normalisation
    - 'FC': load the functional connectivity matrix from a mathlab file
    - 'LAM': compute LAM model using the alpha parameter that maximizes the spearmanr correlation score with the empirical data (ftract)
    - 'CMY': compute communicability 

    The matrices are stored in a dictionnary with the spearmanr correlation score and the pvalue associated

    Args:
    - matlabfile_path, ftract_path, ftractlabels_path (str): path to 1. the matlabfile, 2. the empirical matrix, 3. the parcellation names of the empirical matrix
    - res (int): resolution chosen for the analysis

    Returns:
    - models (dict): dictionnay containing all the masked matrices for each models described above and the spearmanr values with the empirical dataset 

    Note: the key 'ftract' contains the ftract masked for SC and the full ftract
    """
    
    ftract = reorder_matrix(matlabfile_path, ftract_path, ftractlabels_path, res,0)    
    # Load SC
    structural_connectivity = read_structural_connectivity(matlabfile_path,res)
    structural_connectivity = yanat.utils.spectral_normalization(1,structural_connectivity)
    models= {"SC": [structural_connectivity]}
    
    # Load FC
    functional_connectivity = read_functional_connectivity(matlabfile_path,res)
    models["FC"]=[functional_connectivity]

    # Load Euclidean distances
    coordinates = read_coordinates(matlabfile_path,res)
    models["ED"]=[calculate_euclidean_distances(coordinates)]

    # Load Fiber length 
    FL=read_fiber_length(matlabfile_path, res)
    models["FL"]=[FL]
    
    # Compute LAM
    alpha = find_optimal_alpha(structural_connectivity,ftract)
    LAM = yanat.core.lam(structural_connectivity,alpha)
    models["LAM"]=[LAM]
    
    # Compute communicability 
    CMY = yanat.core.communicability(structural_connectivity,1)
    models["CMY"]=[CMY]
    
    for key in models:
        masked_model, masked_ftract = display_mask(models[key][0],ftract)
        models[key][0] = masked_model
        models[key].append(calculate_spearmanr(masked_model,masked_ftract))

    models["ftract"] = [ftract]
    models["ftract"].append(display_mask(models['SC'][0],ftract)[1])#SC masked ftract
    models["ftract"].append(display_mask(models['FC'][0],ftract)[1])#FC masked ftract
    # regler le problemme de FC, ftract masked de FC devrait etre le meme aue le full ftract :3
    return models

def compute_models_direct_connections(matlabfile_path: str, ftract_path: str, ftractlabels_path: str, res: int)-> dict:
    """
    Take in account only the directed connections 
    Given a structural connectivity matrix compute different models and their spearmanr correlation:
    - 'SC': structural connectivity with a spectral normalisation
    - 'FC': load the functional connectivity matrix from a mathlab file
    - 'LAM': compute LAM model using the alpha parameter that maximizes the spearmanr correlation score with the empirical data (ftract)
    - 'CMY': compute communicability 

    The matrices are stored in a dictionnary with the spearmanr correlation score and the pvalue associated

    Args:
    - matlabfile_path, ftract_path, ftractlabels_path (str): path to 1. the matlabfile, 2. the empirical matrix, 3. the parcellation names of the empirical matrix
    - res (int): resolution chosen for the analysis

    Returns:
    - models (dict): dictionnay containing all the masked matrices for each models described above and the spearmanr values with the empirical dataset 

    Note: the key 'ftract' contains the ftract masked for SC and the full ftract
    """
    
    ftract = reorder_matrix(matlabfile_path, ftract_path, ftractlabels_path, res,0)    
    # Load SC
    structural_connectivity = read_structural_connectivity(matlabfile_path,res)
    structural_connectivity = yanat.utils.spectral_normalization(1,structural_connectivity)
    models= {"SC": [structural_connectivity]}

    masked_ftract_sc = display_mask(structural_connectivity,ftract)[1]
    
    # Load FC
    functional_connectivity = read_functional_connectivity(matlabfile_path,res)
    models["FC"]=[functional_connectivity]
    
    # Load Euclidean distances
    coordinates = read_coordinates(matlabfile_path,res)
    models["ED"]=[calculate_euclidean_distances(coordinates)]

    # Load fiber length 
    FL=read_fiber_length(matlabfile_path, res)
    models["FL"]=[FL]
    
    # Compute LAM
    alpha = find_optimal_alpha(structural_connectivity,ftract)
    LAM = yanat.core.lam(structural_connectivity,alpha)
    models["LAM"]=[LAM]
    
    # Compute communicability 
    CMY = yanat.core.communicability(structural_connectivity,1)
    models["CMY"]=[CMY]
    
    for key in models:
        masked_model, masked_SC = display_mask(models[key][0],masked_ftract_sc)
        models[key][0] = masked_model
        models[key].append(calculate_spearmanr(masked_model,masked_ftract_sc))

    models["ftract"] = [masked_ftract_sc]
    # regler le problemme de FC, ftract masked de FC devrait etre le meme aue le full ftract :3
    return models

def display_mask_indirect_connections(model_matrix: np.ndarray, ftract_matrix: np.ndarray,  structural_connectivity: np.ndarray)-> np.array:
    """
    Apply a mask to an input matrix and return the masked given matrix

    Args: 
    - ref_matrix (np.ndarray): Reference matrix for the mask of shape (N, N).
    - input_matrix (np.ndarray): Input matrix of shape (N, N) where N is the number of nodes.
    
    Returns:
    - masked_matrix (np.ndarray): Input matrix of shape (N, N) masked according to the given ref_matrix
    """
    if np.shape(model_matrix)!=np.shape(ftract_matrix) or np.shape(structural_connectivity)!=np.shape(ftract_matrix):
        raise Exception("Sorry, the 2 matrices don't have the same shape") 
    np.fill_diagonal(model_matrix, 0)
    np.fill_diagonal(ftract_matrix, 0)
    np.fill_diagonal(structural_connectivity, 0)
    mask_model = mask_function(model_matrix)
    mask_ftract = mask_function(ftract_matrix)
    mask_direct_connections = ~mask_function(structural_connectivity)
    # Combine the masks
    mask = mask_model|mask_ftract|mask_direct_connections
    masked_model = np.copy(model_matrix)
    masked_model[mask]=0
    masked_ftract= np.copy(ftract_matrix)
    masked_ftract[mask]=0
    return masked_model, masked_ftract
    
def compute_models_indirect_connections(matlabfile_path: str, ftract_path: str, ftractlabels_path: str, res: int)-> dict:
    """
    Given a structural connectivity matrix compute the correlaion, taking in account only the indirect connections
    
    indirect model s
    
    different models and their spearmanr correlation:
    - 'SC': structural connectivity with a spectral normalisation
    - 'FC': load the functional connectivity matrix from a mathlab file
    - 'LAM': compute LAM model using the alpha parameter that maximizes the spearmanr correlation score with the empirical data (ftract)
    - 'CMY': compute communicability 

    The matrices are stored in a dictionnary with the spearmanr correlation score and the pvalue associated

    Args:
    - matlabfile_path, ftract_path, ftractlabels_path (str): path to 1. the matlabfile, 2. the empirical matrix, 3. the parcellation names of the empirical matrix
    - res (int): resolution chosen for the analysis

    Returns:
    - models (dict): dictionnay containing all the masked matrices for each models described above and the spearmanr values with the empirical dataset 

    Note: the key 'ftract' contains the ftract masked for SC and the full ftract
    """
    
    ftract = reorder_matrix(matlabfile_path, ftract_path, ftractlabels_path, res,0)    
    # Load SC
    structural_connectivity = read_structural_connectivity(matlabfile_path,res)
    structural_connectivity = yanat.utils.spectral_normalization(1,structural_connectivity)
    #models= {"SC": [structural_connectivity]}
    
    # Load FC
    functional_connectivity = read_functional_connectivity(matlabfile_path,res)
    models= {"FC":[functional_connectivity]}
    
    # Load Euclidean distances
    coordinates = read_coordinates(matlabfile_path,res)
    models["ED"]=[calculate_euclidean_distances(coordinates)]

    # Load Fiber length 
    FL=read_fiber_length(matlabfile_path, res)
    models["FL"]=[FL]
    
    # Compute LAM
    alpha = find_optimal_alpha(structural_connectivity,ftract)
    LAM = yanat.core.lam(structural_connectivity,alpha)
    models["LAM"]=[LAM]
    
    # Compute communicability 
    CMY = yanat.core.communicability(structural_connectivity,1)
    models["CMY"]=[CMY]
    
    for key in models:
        masked_model, masked_ftract = display_mask_indirect_connections(models[key][0],ftract,structural_connectivity)
        models[key][0] = masked_model
        models[key].append(calculate_spearmanr(masked_model,masked_ftract))

    models["ftract"] = [masked_ftract]
    #models["ftract"].append(display_mask(models['SC'][0],ftract)[1])#SC masked ftract
    models["ftract"].append(display_mask(models['FC'][0],ftract)[1])#FC masked ftract
    # regler le problemme de FC, ftract masked de FC devrait etre le meme aue le full ftract :3
    return models