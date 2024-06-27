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
from typing import Callable
from scipy.spatial.distance import pdist, squareform
import warnings
import PySide2
from collections import OrderedDict
import statsmodels.api as sm
from tqdm import tqdm
from visual_config import *
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import pickle
for font in font_manager.findSystemFonts("/home/kenza/Desktop/Data-fTRACT/Atkinson-Hyperlegible-Font-Print-and-Web-2020-0514"):
    font_manager.fontManager.addfont(font)

set_visual_style()
def Load_Matrix_Names(HBP_Dataset_path):
    """
    Function that returns the paths of the HBP matrices for all resolution and to the label file associated 
    
    PARAMETERS:
    - HBP_Dataset_path (str): path containing Human Brain Projet matrices

    Returns:
    - 2-D (list): contains the paths leading to the different matrix resolution and the labels associated
    """
    os.chdir(HBP_Dataset_path)
    ftract_path = []
    label_ftract_path = []
    parcellations = ["33","60","125","250","500"]
    for res in range(len(parcellations)):
        ftract_path.append("Lausanne2008-"+parcellations[res]+"/probability.txt") 
        label_ftract_path.append("Lausanne2008-"+parcellations[res]+"/Lausanne2008-"+parcellations[res]+".txt")
    return [ftract_path,label_ftract_path]

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
    
def read_functional_connectivity(path: str,res: int)-> np.array:
    """"
    Load as an np.array  the functionnal connectivity from a matlab file
    """
    Lausanne = scipy.io.loadmat(path,simplify_cells=True, squeeze_me=True, chars_as_strings=True,)
    return Lausanne["LauConsensus"]["Matrices"][res][2]
def read_structural_connectivity(path: str, res: int)-> np.array:
    """"
    Load as an np.array  the structural connectivity from a matlab file
    """
    Lausanne = scipy.io.loadmat(path,simplify_cells=True, squeeze_me=True, chars_as_strings=True,)
    return Lausanne["LauConsensus"]["Matrices"][res][0]

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


def compute_models(matlabfile_path: str, ftract_path: str, ftractlabels_path: str, res: int, normalization_func: Callable = None) -> dict:
    """
    Compute different models 
    - 'SC': structural connectivity with a spectral normalisation
    - 'FC': load the functional connectivity matrix from a mathlab file
    - 'LAM': compute LAM model using the alpha parameter that maximizes the spearmanr correlation score with the empirical data (ftract)
    - 'CMY': compute communicability 

    The matrices are stored in a dictionary

    Args:
    - matlabfile_path, ftract_path, ftractlabels_path (str): path to 1. the matlabfile, 2. the empirical matrix, 3. the parcellation names of the empirical matrix
    - res (int): resolution chosen for the analysis
    - normalization_func (Callable, optional): A callable function for normalization. Defaults to None, which uses the spectral normalization.

    Returns:
    - models (dict): dictionary containing all the models described above 

    Note: the key 'ftract' contains the empirical dataset
    """
    # Load ftract
    ftract = reorder_matrix(matlabfile_path, ftract_path, ftractlabels_path, res,0)
    models = {"ftract": ftract}
    
    # Load SC with normalization if provided
    structural_connectivity = read_structural_connectivity(matlabfile_path, res)
    if normalization_func:
        structural_connectivity = normalization_func(structural_connectivity)
    else:
        structural_connectivity = yanat.utils.spectral_normalization(1, structural_connectivity)
    models["SC"] = structural_connectivity
    
    # Load FC
    functional_connectivity = read_functional_connectivity(matlabfile_path, res)
    models["FC"] = functional_connectivity
    
    # Load Euclidean distances
    coordinates = read_coordinates(matlabfile_path,res)
    models["ED"]=calculate_euclidean_distances(coordinates)

    # Load fiber length 
    FL=read_fiber_length(matlabfile_path, res)
    models["FL"]=FL
    
    # Compute LAM
    alpha = find_optimal_alpha(structural_connectivity, ftract)
    LAM = yanat.core.lam(structural_connectivity, alpha)
    models["LAM"] = LAM
    
    # Compute communicability 
    CMY = yanat.core.communicability(structural_connectivity, 1)
    models["CMY"] = CMY
    
    return models
def compute_models_spectral_then_strength_norm(matlabfile_path: str, ftract_path: str, ftractlabels_path: str, res: int) -> dict:
    """
    Compute different models 
    - 'SC': structural connectivity with a spectral normalisation
    - 'FC': load the functional connectivity matrix from a mathlab file
    - 'LAM': compute LAM model using the alpha parameter that maximizes the spearmanr correlation score with the empirical data (ftract)
    - 'CMY': compute communicability 

    The matrices are stored in a dictionary

    Args:
    - matlabfile_path, ftract_path, ftractlabels_path (str): path to 1. the matlabfile, 2. the empirical matrix, 3. the parcellation names of the empirical matrix
    - res (int): resolution chosen for the analysis
    - normalization_func (Callable, optional): A callable function for normalization. Defaults to None, which uses the spectral normalization.

    Returns:
    - models (dict): dictionary containing all the models described above 

    Note: the key 'ftract' contains the empirical dataset
    """
    # Load ftract
    ftract = reorder_matrix(matlabfile_path, ftract_path, ftractlabels_path, res,0)
    models = {"ftract": ftract}
    
    # Load SC with normalization if provided
    structural_connectivity = read_structural_connectivity(matlabfile_path, res)
    structural_connectivity = yanat.utils.strength_normalization(yanat.utils.spectral_normalization(1, structural_connectivity))
    models["SC"] = structural_connectivity
    
    # Load FC
    functional_connectivity = read_functional_connectivity(matlabfile_path, res)
    models["FC"] = functional_connectivity
    
    # Load Euclidean distances
    coordinates = read_coordinates(matlabfile_path,res)
    models["ED"]=calculate_euclidean_distances(coordinates)

    # Load fiber length 
    FL=read_fiber_length(matlabfile_path, res)
    models["FL"]=FL
    
    # Compute LAM
    alpha = find_optimal_alpha(structural_connectivity, ftract)
    LAM = yanat.core.lam(structural_connectivity, alpha)
    models["LAM"] = LAM
    
    # Compute communicability 
    CMY = yanat.core.communicability(structural_connectivity, 1)
    models["CMY"] = CMY
    
    return models

def COMPUTE_spectral_then_strength_norm(HBP_Dataset_path,Lausanne_path):
    """
    Compute all the communication models for all resolution.
    
    PARAMETERS:
    - HBP_Dataset_path, Lausanne_path (str): path of the HBP dataset and Lausanne consensus connectome file 

    RETURN:
    - MODELS (dic): a dictionnary containing matrices for all resolution & all models. To select a particular matrix follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': describes the type of connection
            - 2nd level correspond to the resolution
                - 3rd level correspond to the type of the model
    """
    MODELS=[]
    ftract_path, label_ftract_path =Load_Matrix_Names(HBP_Dataset_path)
    for res in range(5):
        MODELS.append(compute_models_spectral_then_strength_norm(Lausanne_path,ftract_path[res],label_ftract_path[res],res))
    return MODELS

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
    models= {"SC": structural_connectivity}

    masked_ftract_sc = display_mask(structural_connectivity,ftract)[1]
    
    # Load FC
    functional_connectivity = read_functional_connectivity(matlabfile_path,res)
    models["FC"]=functional_connectivity
    
    # Load Euclidean distances
    coordinates = read_coordinates(matlabfile_path,res)
    models["ED"]=calculate_euclidean_distances(coordinates)

    # Load fiber length 
    FL=read_fiber_length(matlabfile_path, res)
    models["FL"]=FL
    
    # Compute LAM
    alpha = find_optimal_alpha(structural_connectivity,ftract)
    LAM = yanat.core.lam(structural_connectivity,alpha)
    models["LAM"]=LAM
    
    # Compute communicability 
    CMY = yanat.core.communicability(structural_connectivity,1)
    models["CMY"]=CMY
    
    for key in models:
        masked_model, masked_SC = display_mask(models[key],masked_ftract_sc)
        models[key] = masked_model

    models["ftract"] = masked_ftract_sc
    # regler le problemme de FC, ftract masked de FC devrait etre le meme aue le full ftract :3
    return models
    
    
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
    models= {"FC":functional_connectivity}
    
    # Load Euclidean distances
    coordinates = read_coordinates(matlabfile_path,res)
    models["ED"]=calculate_euclidean_distances(coordinates)

    # Load Fiber length 
    FL=read_fiber_length(matlabfile_path, res)
    models["FL"]=FL
    
    # Compute LAM
    alpha = find_optimal_alpha(structural_connectivity,ftract)
    LAM = yanat.core.lam(structural_connectivity,alpha)
    models["LAM"]=LAM
    
    # Compute communicability 
    CMY = yanat.core.communicability(structural_connectivity,1)
    models["CMY"]=CMY
    
    for key in models:
        masked_model, masked_ftract = display_mask_indirect_connections(models[key],ftract,structural_connectivity)
        models[key] = masked_model


    models["ftract"] = [masked_ftract]
    #models["ftract"].append(display_mask(models['SC'][0],ftract)[1])#SC masked ftract
    models["ftract"].append(display_mask(models['FC'],ftract)[1])#FC masked ftract
    # regler le problemme de FC, ftract masked de FC devrait etre le meme aue le full ftract :3
    return models
    
    
def all_connections(models: dict)-> dict:
    """
    Apply a mask on the model matrices and returns the matrices with only indirect connections entries
    (ie. ftract mask & model mask minest direct connections = structural connectivity) 

    Args: 
    - models (dict): a dictionnary containing all the models

    Returns:
    - all_connections (dict): a dictionnary containing all model matrices with only indirect connections entries (minest structural connectivity)
    Each key contains a tuple: (masked model, masked ftract)
        > masked model: np.array of the model masked according to ftract dataset
        > masked ftract: np.array of ftract masked according to the model
    """
    all_connections = {}
    for k in ["SC","FC","LAM","CMY"]:
            all_connections_model, all_connections_ftract = display_mask(models[k],models["ftract"])
            all_connections[k] = (all_connections_model,all_connections_ftract)
    return all_connections

def try_all_masks(models: dict)-> dict:
    """
    Apply masks of ftract/model for all, direct and indirect connections. Store everything in a big dictionnary

    Args: 
    - models (dict): a dictionnary containing all the models

    Returns:
    - Data (dict): for each models (= keys) contains the matrices with   

    """
    Data = {} 
    all = all_connections(models)
    direct = only_direct_connections(models)
    indirect = only_indirect_connections(models)
    
    Data["SC"] = all["SC"]
    Data["ftract"] = [direct["ftract"], indirect["ftract"]]
    for model in ["FC","LAM","CMY"]:
        Data[model] = [all[model],direct[model],indirect[model]]
    return Data
    

def spearman_dictionary(Data: dict)-> dict:
    """
    Given a dictionnary containing all the masked matrices, structured as follow:
    Data[model][0] = tuple with all connections
    Data[model][1] = only direct connection
    Data[model][2] = only indirect connections
    
    Calculates the spearman correlation btw the model and 
    """
    SPEARMAN = pd.DataFrame(columns=['SC','FC','LAM','CMY'], index=['all','direct','indirect'])
    pval = pd.DataFrame(columns=['SC','FC','LAM','CMY'], index=['all','direct','indirect'])

    # spearman for all connections
    for model in ["SC","FC","LAM","CMY"]:
        SPEARMAN[model]['all']=[calculate_spearmanr(Data[model][0])][0]
        pval[model]['all']=[calculate_spearmanr(Data[model][0])][1]
    for model in ["FC","LAM","CMY"]:
    # spearman for direct connections
        SPEARMAN[model]['direct'] = calculate_spearmanr(Data[model][1], Data["ftract"][1])[0]
        pval[model]['direct'] = calculate_spearmanr(Data[model][1], Data["ftract"][1])[1]
    #spearman for indirect connections
        SPEARMAN[model]['indirect'] = calculate_spearmanr(Data[model][2], Data["ftract"][2])[0]
        pval[model]['indirect'] = calculate_spearmanr(Data[model][2], Data["ftract"][2])[1]
    return SPEARMAN,pval

def COMPUTE(HBP_Dataset_path,Lausanne_path,normalization_func: Callable = None):
    """
    Compute all the communication models for all resolution.
    
    PARAMETERS:
    - HBP_Dataset_path, Lausanne_path (str): path of the HBP dataset and Lausanne consensus connectome file 

    RETURN:
    - MODELS (dic): a dictionnary containing matrices for all resolution & all models. To select a particular matrix follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': describes the type of connection
            - 2nd level correspond to the resolution
                - 3rd level correspond to the type of the model
    """
    MODELS=[]
    ftract_path, label_ftract_path =Load_Matrix_Names(HBP_Dataset_path)
    for res in range(5):
        MODELS.append(compute_models(Lausanne_path,ftract_path[res],label_ftract_path[res],res,normalization_func))
    return MODELS

def VECTOR(MODELS,res):
    """
    Function that returns the flattened matrices of all the models applying the 3 different masks.
    This function is useful for plotting.

    PARAMETERS:
    - MODELS (dict): A dictionary containing matrices for all resolutions and all models.
    - res (int): Level of resolution.

    RETURNS:
    - VECT (dict): A dictionary containing the vectors for each model (masked or not) for the given resolution.
      To select a particular vector, follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': Describes the type of connection.
        - 2nd level of keys: 'SC', 'FC', 'ED', 'FL', 'LAM', 'CMY': Describes the models.
          Corresponds to a list: [model_vector, ftract_vector]
    """
    ALL_vect = {}
    DIRECT_vect = {}
    INDIRECT_vect = {}
    VECT = {"ALL": ALL_vect, "DIRECT": DIRECT_vect, "INDIRECT": INDIRECT_vect}
    for model in ['SC','FL','ED','FC','LAM', 'CMY']:
        m, P = display_mask(MODELS[res][model],MODELS[res]['ftract'])
        vectm = get_nonzero_flat_vector(m)
        vectP = get_nonzero_flat_vector(P)
        ALL_vect[model]=[vectm,vectP]
        # DIRECT
    for model in ['SC','FL','ED','FC','LAM', 'CMY']:
        f = display_mask(MODELS[res]['ftract'],MODELS[res]['SC'])[0]
        m, P = display_mask(MODELS[res][model],f)
        vectm = get_nonzero_flat_vector(m)
        vectP = get_nonzero_flat_vector(P)
        DIRECT_vect[model]=[vectm,vectP]
        # INDIRECT
    for model in ['ED','FC','LAM', 'CMY']:
        m, P = display_mask_indirect_connections(MODELS[res][model],MODELS[res]['ftract'],MODELS[res]['SC'])
        vectm = get_nonzero_flat_vector(m)
        vectP = get_nonzero_flat_vector(P)
        INDIRECT_vect[model]=[vectm,vectP]
    return VECT


def SPEARMAN(MODELS):
    """
    Function that calculates Spearman values for all the models, all resolutions, applying the 3 different masks.

    PARAMETERS:
    - MODELS (dict): A dictionary containing matrices for all resolutions and all models.

    RETURNS:
    - SPEARMAN (dict): A dictionary containing rho and pval for each model (masked or not) for all resolutions.
      To select a particular vector, follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': Describes the type of connection.
        - 2nd level of keys: 'SC', 'FC', 'ED', 'FL', 'LAM', 'CMY': Describes the models.
          Corresponds to a list: [df_rho, df_pval]
          All the values are stored in a pandas DataFrame.
    """
    parcellation = ["33","60","125","250","500"]
    ALL_spearman = pd.DataFrame(index=parcellation,columns=['SC', 'FL','ED','FC', 'LAM', 'CMY'])
    ALL_pval = pd.DataFrame(index=parcellation,columns=['SC', 'FL','ED','FC', 'LAM', 'CMY'])
    DIRECT_spearman = pd.DataFrame(index=parcellation,columns=['SC', 'FL','ED','FC', 'LAM', 'CMY'])
    DIRECT_pval = pd.DataFrame(index=parcellation,columns=['SC', 'FL','ED','FC', 'LAM', 'CMY'])
    INDIRECT_spearman = pd.DataFrame(index=parcellation,columns=['ED','FC','LAM', 'CMY'])
    INDIRECT_pval = pd.DataFrame(index=parcellation,columns=['ED','FC','LAM', 'CMY'])

    SPEARMAN = {"ALL":[ALL_spearman,ALL_pval],"DIRECT":[DIRECT_spearman,DIRECT_pval],"INDIRECT":[INDIRECT_spearman,INDIRECT_pval]}
    #ALL
    for res in range(len(parcellation)):
        for model in ['SC', 'FL','ED','FC', 'LAM', 'CMY']:
            m, P = display_mask(MODELS[res][model],MODELS[res]['ftract'])
            vectm = get_nonzero_flat_vector(m)
            vectP = get_nonzero_flat_vector(P)
            rho,pval=st.spearmanr(vectm,vectP)
            ALL_spearman.loc[parcellation[res],model] = rho
            ALL_pval.loc[parcellation[res],model] = f"{pval:.2e}"
           
        # DIRECT
        for model in ['SC', 'FL','ED','FC', 'LAM', 'CMY']:
            f = display_mask(MODELS[res]['ftract'],MODELS[res]['SC'])[0]
            m, P = display_mask(MODELS[res][model],f)
            vectm = get_nonzero_flat_vector(m)
            vectP = get_nonzero_flat_vector(P)
            rho,pval = st.spearmanr(vectm,vectP)
            DIRECT_spearman.loc[parcellation[res],model] = rho
            DIRECT_pval.loc[parcellation[res],model] = f"{pval:.2e}"
            
        # INDIRECT
        for model in ['ED','FC','LAM', 'CMY']:
            m, P = display_mask_indirect_connections(MODELS[res][model],MODELS[res]['ftract'],MODELS[res]['SC'])
            vectm = get_nonzero_flat_vector(m)
            vectP = get_nonzero_flat_vector(P)
            rho,pval = st.spearmanr(vectm,vectP)
            INDIRECT_spearman.loc[parcellation[res],model] = rho
            INDIRECT_pval.loc[parcellation[res],model] = f"{pval:.2e}"

    return SPEARMAN

def PLOT_SPEARMAN(SPEARMAN):
    """
    Function that plot the spearman value for all connections & models given the spearman dictionnary.

    PARAMETERS:
    - SPEARMAN (dict): A dictionary containing rho and pval for each model (masked or not) for all resolutions.
      To select a particular vector, follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': Describes the type of connection.
        - 2nd level of keys: 'SC', 'FC', 'ED', 'FL', 'LAM', 'CMY': Describes the models.
          Corresponds to a list: [df_rho, df_pval]
          All the values are stored in a pandas DataFrame.
    RETURN:
    A multiplot that displays spearman score accross all models and resolution for the 3 types of connections (all, direct, indirect)
    """
    parcellations = list(SPEARMAN["ALL"][0].index)
    connections = ["ALL","DIRECT","INDIRECT"]
    plots = ["A","B","C"]
    fig, axes = plt.subplot_mosaic([plots],figsize=(17*CM,5*CM),dpi=150,)
    color_dic = {"SC":'#A5D8FF',"FL":'#339AF0',"ED":'#0D47A1',"FC":'#12B886',"LAM":'#FFEC99',"CMY":'#FFC078',
                 "LAM_spect":'#FFEC99',"CMY_spect":'#FFC078',
                 "LAM_strength":'#FCC419',"CMY_strength":'#E8590C'}
    axes['A'].set_ylabel('spearman')
    for plot,co in zip(plots,connections):
        models = list(SPEARMAN[co][0].columns)
        axes[plot].set_xlabel('parcellation')
        axes[plot].set_title(co)
        for model in models:
            axes[plot].scatter(parcellations,abs(SPEARMAN[co][0].loc[:,model]),label=model,
                               linewidth=0.5,edgecolor='black',s=25,zorder=2,
                               color=color_dic[model])
            axes[plot].plot(parcellations,abs(SPEARMAN[co][0].loc[:,model]),
                            linewidth=1.5,zorder=1,
                            color=color_dic[model])
    axes['A'].legend(loc="upper left", frameon=False, fontsize=8,bbox_to_anchor=(3.5, 1))


    sns.despine(fig=fig, trim=False)
    fig.tight_layout(pad=0.5)

def SCATTERPLOT(VECTOR,SPEARMAN,res):
    """
    Creates a multiplot for all type of connections & all models for a given resolution.

    PARAMETERS:
    - VECTOR (dict): A dictionary containing the vectors for each model (masked or not) for the given resolution.
      To select a particular vector, follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': Describes the type of connection.
        - 2nd level of keys: 'SC', 'FC', 'ED', 'FL', 'LAM', 'CMY': Describes the models.
          Corresponds to a list: [model_vector, ftract_vector]
    - SPEARMAN (dict): A dictionary containing rho and pval for each model (masked or not) for all resolutions.
      To select a particular vector, follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': Describes the type of connection.
        - 2nd level of keys: 'SC', 'FC', 'ED', 'FL', 'LAM', 'CMY': Describes the models.
          Corresponds to a list: [df_rho, df_pval]
          All the values are stored in a pandas DataFrame.
          
    - res (str): resolution used for the representation

    RETURN:
    A multiplot with the scatter plot of all models vs ftract & rho value associated for a given resolution
    """
    ALL = ["A", "B", "C", "F", "E", "D"]
    DIRECT = ["AA", "BB", "CC", "FF", "EE", "DD"]
    INDIRECT = ["CCC", "FFF", "EEE", "DDD"]
    PLOTS=[ALL,DIRECT,INDIRECT]
    connections = ["ALL","DIRECT","INDIRECT"]

    color_dic = {"SC":'#A5D8FF',"FL":'#339AF0',"ED":'#0D47A1',"FC":'#12B886',"LAM":'#FFEC99',"CMY":'#FFC078'}

    fig, axes = plt.subplot_mosaic([
        ["A", "B", "C", "F", "E", "D"],
        ["AA", "BB", "CC", "FF", "EE", "DD"],
        [".", ".", "CCC", "FFF", "EEE", "DDD"]],
    figsize=(23*CM, 10*CM),
    dpi=150)

    for i in range(3):
        models = VECTOR[connections[i]].keys()
        axes[PLOTS[i][0]].set_ylabel(connections[i])
        for plot,model in zip(PLOTS[i],models): 
            m,P = VECTOR[connections[i]][model]
            rho = SPEARMAN[connections[i]][0].loc[res,model]
            if model=='FC':
                sns.scatterplot(x=m,y=P,ax=axes[plot],
                            s=3,color=color_dic[model],
                            linewidth=0.09,edgecolor='black',
                            rasterized=True)
            
                axes[plot].annotate(f"$\\rho_{{{model}}} = {np.round(rho, 2)}$", 
                                xy=(1, 0.08), fontsize=8,xycoords="axes fraction", 
                                bbox=dict(boxstyle='round,pad=0.2', edgecolor='white', facecolor='white',alpha=0.5,linewidth=0.5 ),
                                ha='right')
            else:
                sns.scatterplot(x=yanat.utils.log_normalize(m),y=P,ax=axes[plot],
                        s=3,color=color_dic[model],linewidth=0.09,edgecolor='black',rasterized=True)
                axes[plot].annotate(f"$\\rho_{{{model}}} = {np.round(rho, 2)}$", 
                                xy=(1, 0.08), fontsize=8,xycoords="axes fraction", 
                                bbox=dict(boxstyle='round,pad=0.2', edgecolor='white', facecolor='white',alpha=0.5,linewidth=0.5 ),
                                ha='right')
    axes["AA"].set_xlabel('[a.u.]')
    axes["BB"].set_xlabel('[a.u.]')
    axes["DDD"].set_xlabel('[a.u.]')
    axes["CCC"].set_xlabel('[a.u.]')
    axes["EEE"].set_xlabel('[a.u.]')
    axes["FFF"].set_xlabel('[a.u.]')

    for plot,model in zip(ALL,list(VECTOR["ALL"].keys())):
        axes[plot].set_title(model)
    sns.despine(fig=fig, trim=False)
    fig.tight_layout(pad=0.5)
    plt.show()

def PLOT_HEATMAP(MODELS):
    """
    Function that plot the heatmap of few models for all the resolutions available(ftract, SC, FC, CMY) from the model dictionnary

    PARAMETERS:
    - MODELS  (dic): a dictionnary containing matrices for all resolution & all models. To select a particular matrix follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': describes the type of connection
            - 2nd level correspond to the resolution
                - 3rd level correspond to the type of the model

    RETURNS:
    multiplot of the heatmap 5x4 figures
    """
    SC_color = sns.color_palette('light:'+HALF_BLACK, as_cmap=True)
    colors = [DEEP_BLUE, 'white' ,RED]
    cmap_name = 'deep_blue_white_red'
    FC_color = LinearSegmentedColormap.from_list(cmap_name, colors)
    
    color_dic = {"SC":SC_color,"ftract":FC_color,"FC":FC_color,"CMY":FC_color}
    models = ['ftract','SC','FC','CMY']
    parcellations = ['33', '60', '125', '250', '500']
    ftract =  ["A","B","C","D","E"]
    SC = ["AA","BB","CC","DD","EE"]
    FC = ["AAA","BBB","CCC","DDD","EEE"]
    CMY = ["AAAA","BBBB","CCCC","DDDD","EEEE"]
    
    PLOT = [ftract,SC,FC,CMY]
    fig, axes = plt.subplot_mosaic(PLOT,figsize=(10* CM, 7 * CM),dpi=150,)
    
    for res in range(5):
        for model,plot in zip(models,PLOT):
            if model == 'CMY':
                sns.heatmap(yanat.utils.log_minmax_normalize(MODELS[res][model]),ax=axes[plot[res]],
                        square=True,xticklabels=False,cmap=color_dic[model],
                        yticklabels=False,linewidths=0,rasterized=True,cbar=False,center=0)
            elif model == 'SC':
                sns.heatmap(yanat.utils.log_minmax_normalize(MODELS[res][model]),ax=axes[plot[res]],
                        square=True,xticklabels=False,cmap=color_dic[model],
                        yticklabels=False,linewidths=0,rasterized=True,cbar=False)
            else:
                sns.heatmap(MODELS[res][model],ax=axes[plot[res]],
                        square=True,xticklabels=False,cmap=color_dic[model],
                        yticklabels=False,linewidths=0,rasterized=True,cbar=False,center=0)
    for plot,res in zip(ftract,parcellations):
        axes[plot].title.set_text(res)
    
    for plot in range(len(PLOT)):
        axes[PLOT[plot][0]].set_ylabel(models[plot])
    fig.tight_layout(pad=0.3)
    sns.despine(fig=fig, top=False, right=False, left=False, bottom=False)
        
def COMPARE_NORMALIZATION_PLOT_SPEARMAN(COMPARE_NORM):
    """
    Function that plot the spearman value for all connections & models given the spearman dictionnary.

    PARAMETERS:
    - COMPARE_NORM (dict): A dictionary containing rho and pval for each model (masked or not) for all resolutions.
      To select a particular vector, follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': Describes the type of connection.
        - 2nd level of keys: 'ED', 'FL', 'LAM', 'CMY' (strength and spectral): Describes the models.
          Corresponds to a list:  df_rho
          All the values are stored in a pandas DataFrame.
    RETURN:
    A multiplot that displays spearman score accross all models and resolution for the 3 types of connections (all, direct, indirect)
    """
    parcellations = list(COMPARE_NORM["ALL"].index)
    connections = ["ALL","DIRECT","INDIRECT"]
    plots = ["A","B","C"]
    fig, axes = plt.subplot_mosaic([plots],figsize=(17*CM,5*CM),dpi=150,)
    color_dic = {"SC":'#A5D8FF',"FL":'#339AF0',"FL/ED":'#0D47A1',"ED":'#0D47A1',"FC":'#12B886',"LAM":'#FFEC99',"CMY":'#FFC078',
                 "LAMspect":'#FFEC99',"CMYspect":'#FFC078',
                 "LAMstrength":'#BE4BDB',"CMYstrength":'#FCC2D7'}
    handles = []
    labels = []
    axes['A'].set_ylabel('spearman')
    for plot,co in zip(plots,connections):
        models = list(COMPARE_NORM[co].columns)
        axes[plot].set_xlabel('parcellation')
        axes[plot].set_title(co)
        for model in models:
            scatter =axes[plot].scatter(parcellations,abs(COMPARE_NORM[co].loc[:,model]),label=model,
                               linewidth=0.5,edgecolor='black',s=25,zorder=2,
                               color=color_dic[model])
            axes[plot].plot(parcellations,abs(COMPARE_NORM[co].loc[:,model]),
                            linewidth=1.5,zorder=1,
                            color=color_dic[model])
            if model not in labels:
                handles.append((scatter))
                labels.append(model)
    fig.legend(handles, labels, loc="upper left", frameon=False, fontsize=8, bbox_to_anchor=(1.05, 1))        
    
    sns.despine(fig=fig, trim=False)
    fig.tight_layout(pad=0.5)

def SPEARMAN_NORM(SPEARMAN_STRENGTH,SPEARMAN_SPECTRAL):
    df_all = pd.DataFrame(index=['33', '60', '125', '250', '500'],columns=['FL','LAMstrength','CMYstrength','LAMspect','CMYspect'])
    df_direct = pd.DataFrame(index=['33', '60', '125', '250', '500'],columns=['FL','LAMstrength','CMYstrength','LAMspect','CMYspect'])
    df_indirect = pd.DataFrame(index=['33', '60', '125', '250', '500'],columns=['ED','LAMstrength','CMYstrength','LAMspect','CMYspect'])
    D = {"ALL":df_all,"DIRECT":df_direct,"INDIRECT":df_indirect}

    for co in D.keys():
        if co == "INDIRECT":
            D[co].loc[:,'ED']=SPEARMAN_STRENGTH[co][0].loc[:,'ED']
        else:
            D[co].loc[:,'FL']=SPEARMAN_STRENGTH[co][0].loc[:,'FL']
        for spect,strength,model in zip(['LAMspect','CMYspect'],['LAMstrength','CMYstrength'],['LAM','CMY']):
            D[co].loc[:,spect]=SPEARMAN_SPECTRAL[co][0].loc[:,model]
            D[co].loc[:,strength]=SPEARMAN_STRENGTH[co][0].loc[:,model]

    return D
    
def VECTOR_NORM(VECTOR_STRENGTH,VECTOR_SPECTRAL):
    ALL_vect = {}
    DIRECT_vect = {}
    INDIRECT_vect = {}
    VECT = {"ALL": ALL_vect, "DIRECT": DIRECT_vect, "INDIRECT": INDIRECT_vect}
    for model,co in zip(["FL","FL","ED"],list(VECT.keys())):
        VECT[co][model]=VECTOR_STRENGTH[co][model]
    for co in list(VECT.keys()):
        for model,stren,spec in zip(['LAM','CMY'],['LAMstrength','CMYstrength'],['LAMspect','CMYspect']):
            VECT[co][stren]=VECTOR_STRENGTH[co][model]
            VECT[co][spec]=VECTOR_SPECTRAL[co][model]
    
    new_orderFL = ["FL",'LAMstrength','CMYstrength','LAMspect','CMYspect']
    new_orderED = ["ED",'LAMstrength','CMYstrength','LAMspect','CMYspect']
    # Créer un OrderedDict avec le nouvel ordre
    ALL_vect = OrderedDict((key, ALL_vect[key]) for key in new_orderFL)
    DIRECT_vect = OrderedDict((key, DIRECT_vect[key]) for key in new_orderFL)
    INDIRECT_vect = OrderedDict((key, INDIRECT_vect[key]) for key in new_orderED)
    VECT = {"ALL": ALL_vect, "DIRECT": DIRECT_vect, "INDIRECT": INDIRECT_vect}        #co[model]=VECTOR_STRENGTH[co][model]
    return VECT
    
def  COMPARE_NORMALIZATION_SCATTERPLOT(VECTOR,SPEARMAN,res):
    """
    Creates a multiplot for all type of connections & all models for a given resolution.

    PARAMETERS:
    - VECTOR (dict): A dictionary containing the vectors for each model (masked or not) for the given resolution.
      To select a particular vector, follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': Describes the type of connection.
        - 2nd level of keys: 'SC', 'FC', 'ED', 'FL', 'LAM', 'CMY': Describes the models.
          Corresponds to a list: [model_vector, ftract_vector]
    - SPEARMAN (dict): A dictionary containing rho and pval for each model (masked or not) for all resolutions.
      To select a particular vector, follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': Describes the type of connection.
        - 2nd level of keys: 'SC', 'FC', 'ED', 'FL', 'LAM', 'CMY': Describes the models.
          Corresponds to a list: [df_rho, df_pval]
          All the values are stored in a pandas DataFrame.
          
    - res (str): resolution used for the representation

    RETURN:
    A multiplot with the scatter plot of all models vs ftract & rho value associated for a given resolution
    """
    ALL = ["A", "B", "C", "D", "E"]
    DIRECT = ["AA", "BB", "CC", "DD", "EE"]
    INDIRECT = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    PLOTS=[ALL,DIRECT,INDIRECT]
    connections = ["ALL","DIRECT","INDIRECT"]

    color_dic = {"SC":'#A5D8FF',"FL":'#339AF0',"FL/ED":'#0D47A1',"ED":'#0D47A1',"FC":'#12B886',"LAM":'#FFEC99',"CMY":'#FFC078',
                 "LAMspect":'#FFEC99',"CMYspect":'#FFC078',
                 "LAMstrength":'#BE4BDB',"CMYstrength":'#FCC2D7'}
    fig, axes = plt.subplot_mosaic(PLOTS,figsize=(23*CM, 10*CM),dpi=150)
    for i in range(3):
        models = VECTOR[connections[i]].keys()
        axes[PLOTS[i][0]].set_ylabel(connections[i])
        for plot,model in zip(PLOTS[i],models): 
            m,P = VECTOR[connections[i]][model]
            rho = SPEARMAN[connections[i]].loc[res,model]
            sns.scatterplot(x=yanat.utils.log_normalize(m),y=P,ax=axes[plot],
                        s=3,color=color_dic[model],linewidth=0.09,edgecolor='black',rasterized=True)
            axes[plot].annotate(f"$\\rho_{{{{{model}}}}} = {np.round(rho, 2)}$", 
                                xy=(1, 0.08), fontsize=8,xycoords="axes fraction", 
                                bbox=dict(boxstyle='round,pad=0.2', edgecolor='white', facecolor='white',alpha=0.5,linewidth=0.5 ),
                                ha='right')
    axes["AAA"].set_xlabel('[a.u.]')
    axes["BBB"].set_xlabel('[a.u.]')
    axes["DDD"].set_xlabel('[a.u.]')
    axes["CCC"].set_xlabel('[a.u.]')
    axes["AAA"].set_title('ED')
    for plot,model in zip(ALL,list(VECTOR["ALL"].keys())):
        axes[plot].set_title(model)
    sns.despine(fig=fig, trim=False)
    fig.tight_layout(pad=0.5)
    plt.show()

def SC_THRESHOLD(structural_connectivity,T):
    """
    Applies a threshold for a given SC

    PARAMETERS:
    - structural connectivity (np.array): a structural connectivity matrix
    - T (float): a threshold, T<0.54 (max value of SC)

    RETURNS:
    - structural_connectivity_copy (np.array):thresholded mat
    """
    structural_connectivity_copy = structural_connectivity.copy()
    structural_connectivity_copy = np.where(structural_connectivity_copy < T, 0, structural_connectivity_copy)
    return structural_connectivity_copy

def compute_models_thresholdSC(structural_connectivity,ftract) -> dict:
    """
    Compute different models given a SC thresholded
    - 'LAM': compute LAM model using the alpha parameter that maximizes the spearmanr correlation score with the empirical data (ftract)
    - 'CMY': compute communicability 

    The matrices are stored in a dictionary
    Args:
    - structural connectivity (np.array): a structural connectivity matrix
    - ftract (np.array): probability matrix of ftract HBP 

    Returns:
    - models (dict): dictionary containing all the models described above 

    Note: the key 'ftract' contains the empirical dataset
    """
    models = {"ftract":ftract}
    models["SC"]=structural_connectivity
    # Compute LAM
    alpha = find_optimal_alpha(structural_connectivity, ftract)
    LAM = yanat.core.lam(structural_connectivity, alpha)
    models["LAM"] = LAM
    
    # Compute communicability 
    CMY = yanat.core.communicability(structural_connectivity, 1)
    models["CMY"] = CMY
    return models

def VECTOR_SC_THRESHOLD(SC_THRESHOLDED):
    """
    Function that returns the flattened matrices of all the models applying the 3 different masks.
    This function is useful for plotting.

    PARAMETERS:
    - MODELS (dict): A dictionary containing matrices for all resolutions and all models.
    - t (int): index of the threshold
    RETURNS:
    - VECT (dict): A dictionary containing the vectors for each model (masked or not) for the given resolution.
      To select a particular vector, follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': Describes the type of connection.
        - 2nd level of keys: 'SC','LAM', 'CMY': Describes the models.
          Corresponds to a list: [model_vector, ftract_vector]
    """
    ALL_vect = {}
    DIRECT_vect = {}
    INDIRECT_vect = {}
    VECT = {"ALL": ALL_vect, "DIRECT": DIRECT_vect, "INDIRECT": INDIRECT_vect}
    
    for model in ['SC','LAM', 'CMY']:
            m, P = display_mask(SC_THRESHOLDED[model],SC_THRESHOLDED['ftract'])
            vectm = get_nonzero_flat_vector(m)
            vectP = get_nonzero_flat_vector(P)
            ALL_vect[model]=[vectm,vectP]
            # DIRECT
    for model in ['SC','LAM', 'CMY']:
            f = display_mask(SC_THRESHOLDED['ftract'],SC_THRESHOLDED['SC'])[0]
            m, P = display_mask(SC_THRESHOLDED[model],f)
            vectm = get_nonzero_flat_vector(m)
            vectP = get_nonzero_flat_vector(P)
            DIRECT_vect[model]=[vectm,vectP]
            # INDIRECT
    for model in ['LAM', 'CMY']:
            m, P = display_mask_indirect_connections(SC_THRESHOLDED[model],SC_THRESHOLDED['ftract'],SC_THRESHOLDED['SC'])
            vectm = get_nonzero_flat_vector(m)
            vectP = get_nonzero_flat_vector(P)
            INDIRECT_vect[model]=[vectm,vectP]
    return VECT

def SPEARMAN_SC_THRESHOLD(SC_THRESHOLDED,THRESHOLDS):
    """
    Function that calculates Spearman values for all the models, all resolutions, applying the 3 different masks.

    PARAMETERS:
    - SC_THRESHOLDED (dict): A dictionary containing matrices for different threshold T and the following models: 'SC','LAM','CMY'.
    - THRESHOLDS (list): a list containing all the thresholds used
    
    RETURNS:
    - SPEARMAN (dict): A dictionary containing rho and pval for each model (masked or not) for all resolutions.
      To select a particular vector, follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': Describes the type of connection.
        - 2nd level of keys: 'SC','LAM', 'CMY': Describes the models.
          Corresponds to a list: [df_rho, df_pval]
          All the values are stored in a pandas DataFrame.
    """
    ALL_spearman = pd.DataFrame(index=THRESHOLDS,columns=['SC', 'LAM', 'CMY'])
    ALL_pval = pd.DataFrame(index=THRESHOLDS,columns=['SC', 'LAM', 'CMY'])
    DIRECT_spearman = pd.DataFrame(index=THRESHOLDS,columns=['SC', 'LAM', 'CMY'])
    DIRECT_pval = pd.DataFrame(index=THRESHOLDS,columns=['SC', 'LAM', 'CMY'])
    INDIRECT_spearman = pd.DataFrame(index=THRESHOLDS,columns=['LAM', 'CMY'])
    INDIRECT_pval = pd.DataFrame(index=THRESHOLDS,columns=['LAM', 'CMY'])

    SPEARMAN = {"ALL":[ALL_spearman,ALL_pval],"DIRECT":[DIRECT_spearman,DIRECT_pval],"INDIRECT":[INDIRECT_spearman,INDIRECT_pval]}
    #ALL
    for res in range(len(THRESHOLDS)):
        for model in ['SC', 'LAM', 'CMY']:
            m, P = display_mask(SC_THRESHOLDED[res][model],SC_THRESHOLDED[res]['ftract'])
            vectm = get_nonzero_flat_vector(m)
            vectP = get_nonzero_flat_vector(P)
            rho,pval=st.spearmanr(vectm,vectP)
            ALL_spearman.loc[THRESHOLDS[res],model] = rho
            ALL_pval.loc[THRESHOLDS[res],model] = f"{pval:.2e}"
           
        # DIRECT
        for model in ['SC', 'LAM', 'CMY']:
            f = display_mask(SC_THRESHOLDED[res]['ftract'],SC_THRESHOLDED[res]['SC'])[0]
            m, P = display_mask(SC_THRESHOLDED[res][model],f)
            vectm = get_nonzero_flat_vector(m)
            vectP = get_nonzero_flat_vector(P)
            rho,pval = st.spearmanr(vectm,vectP)
            DIRECT_spearman.loc[THRESHOLDS[res],model] = rho
            DIRECT_pval.loc[THRESHOLDS[res],model] = f"{pval:.2e}"
            
        # INDIRECT
        for model in ['LAM', 'CMY']:
            m, P = display_mask_indirect_connections(SC_THRESHOLDED[res][model],SC_THRESHOLDED[res]['ftract'],SC_THRESHOLDED[res]['SC'])
            vectm = get_nonzero_flat_vector(m)
            vectP = get_nonzero_flat_vector(P)
            rho,pval = st.spearmanr(vectm,vectP)
            INDIRECT_spearman.loc[THRESHOLDS[res],model] = rho
            INDIRECT_pval.loc[THRESHOLDS[res],model] = f"{pval:.2e}"

    return SPEARMAN

def PLOT_SPEARMAN_SC_THRESHOLD(SPEARMAN):
    """
    Function that plot the spearman value for all connections & models given the spearman dictionnary conttaining the thresholded SC.

    PARAMETERS:
    - SPEARMAN (dict): A dictionary containing rho and pval for each model (masked or not) for different level of threshold.
      To select a particular vector, follow this hierarchy:
        - 1st level of keys: 'ALL', 'DIRECT', 'INDIRECT': Describes the type of connection.
        - 2nd level of keys: 'SC','LAM', 'CMY': Describes the models.
          Corresponds to a list: [df_rho, df_pval]
          All the values are stored in a pandas DataFrame.
          
    RETURN:
    A multiplot that displays spearman score accross all models and thresholds for the 3 types of connections (all, direct, indirect)
    """
    parcellations = list(SPEARMAN["ALL"][0].index)
    connections = ["ALL","DIRECT","INDIRECT"]
    plots = ["A","B","C"]
    fig, axes = plt.subplot_mosaic([plots],figsize=(17*CM,5*CM),dpi=150,)
    color_dic = {"SC":'#A5D8FF',"FL":'#339AF0',"ED":'#0D47A1',"FC":'#12B886',"LAM":'#FFEC99',"CMY":'#FFC078',
                 "LAM_spect":'#FFEC99',"CMY_spect":'#FFC078',
                 "LAM_strength":'#FCC419',"CMY_strength":'#E8590C'}
    axes['A'].set_ylabel('spearman')
    for plot,co in zip(plots,connections):
        models = list(SPEARMAN[co][0].columns)
        axes[plot].set_xlabel('threshold')
        axes[plot].set_title(co)
        for model in models:
            axes[plot].scatter(parcellations,abs(SPEARMAN[co][0].loc[:,model]),label=model,
                               linewidth=0.5,edgecolor='black',s=25,zorder=2,
                               color=color_dic[model])
            axes[plot].plot(parcellations,abs(SPEARMAN[co][0].loc[:,model]),
                            linewidth=1.5,zorder=1,
                            color=color_dic[model])
    axes['A'].legend(loc="upper left", frameon=False, fontsize=8,bbox_to_anchor=(3.5, 1))


    sns.despine(fig=fig, trim=False)
    fig.tight_layout(pad=0.5)