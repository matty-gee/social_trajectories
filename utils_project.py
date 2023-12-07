#-----------------------------------------------------------------------------------------------------------
# libraries
#-----------------------------------------------------------------------------------------------------------

import sys, warnings, os, glob, copy, itertools, json, shutil, math
if not sys.warnoptions: warnings.simplefilter("ignore")
from datetime import date
from pathlib import Path
from collections import Counter
from re import search
from six.moves import cPickle as pickle

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, Rectangle
from mpl_toolkits import axes_grid1
import matplotlib.backends.backend_pdf

import numpy.lib.recfunctions as rfn
import numpy as np
from numpy.linalg import norm

import pandas as pd
import nibabel as nib
from nilearn import plotting, image

import statsmodels.api as sm
from statsmodels.multivariate.cancorr import CanCorr

import scipy
from scipy import optimize
from scipy.stats import ( 
    zscore, chi2_contingency, wilcoxon, kendalltau,
    pearsonr, spearmanr, kendalltau, ttest_1samp, wilcoxon
    )
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cosine as cosd

from sklearn.base import (
    BaseEstimator, ClassifierMixin, 
    RegressorMixin, clone, TransformerMixin
    )
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer, MinMaxScaler, 
    OneHotEncoder, StandardScaler)
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.cluster import (
    AgglomerativeClustering, KMeans, SpectralClustering, DBSCAN
    )
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
    )
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import (
    KernelDensity, KNeighborsClassifier, 
    KNeighborsRegressor, NeighborhoodComponentsAnalysis, 
    RadiusNeighborsClassifier, RadiusNeighborsRegressor
    )
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.manifold import (
    MDS, Isomap, LocallyLinearEmbedding, 
    SpectralEmbedding, TSNE
    )
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, 
    f1_score, matthews_corrcoef, pairwise_distances, adjusted_mutual_info_score, adjusted_rand_score
    )
from sklearn.model_selection import (
    GridSearchCV, KFold, LeaveOneOut, StratifiedKFold, 
    StratifiedShuffleSplit, cross_val_predict, cross_val_score, train_test_split
    )

#-----------------------------------------------------------------------------------------------------------
# maybe not used?
#-----------------------------------------------------------------------------------------------------------

import networkx as nx
from ripser import ripser, Rips
from persim import plot_diagrams
import kmapper as km
from tqdm import tqdm

# persistent homology stuff:
# from gtda.homology import (
#     VietorisRipsPersistence, EuclideanCechPersistence, 
#     WeightedRipsPersistence, CubicalPersistence
#     )
# from gtda.diagrams import (
#     PersistenceImage, PersistenceLandscape, 
#     BettiCurve, HeatKernel, 
#     PersistenceEntropy, NumberOfPoints, 
#     Amplitude, ComplexPolynomial, Scaler
#     )
# from gtda.plotting import plot_diagram
# from gtda.mapper import (
#     CubicalCover, Projection,
#     make_mapper_pipeline,
#     plot_static_mapper_graph, plot_interactive_mapper_graph,
#     MapperInteractivePlotter)
# from umap import UMAP

#-----------------------------------------------------------------------------------------------------------
# custom code
#-----------------------------------------------------------------------------------------------------------

# detect user & set paths
user = os.path.expanduser('~')
if user == '/hpc/users/schafm03': # minerva 
    base_dir = '/sc/arion/projects/OlfMem/mgs/2D_place'
    sys.path.insert(0, f'{base_dir}/Code/toolbox') 
    sys.path.insert(0, f'{base_dir}/Code/social_navigation_analysis/social_navigation_analysis') 
else:

    if user == '/Users/matty_gee': # laptop
        base_dir = '/Users/matty_gee/Desktop/projects/SNT_trajectory'
    elif user == '/Users/matthew': 
        if os.path.exists('/Volumes/synapse/projects/SocialSpace/Projects/SNT-fmri_place/Trajectory_analyses'): # mt sinai desktop
            base_dir = '/Volumes/synapse/projects/SocialSpace/Projects/SNT-fmri_place/Trajectory_analyses'
        else: # columbia desktop
            base_dir  = '/Users/matthew/Desktop/projects/trajectory'
    data_dir = base_dir

    sys.path.insert(0, f'{user}/Dropbox/Projects/toolbox/toolbox')
    sys.path.insert(0, f'{user}/Dropbox/Projects/fmri_tools/quality_control')
    sys.path.insert(0, f'{user}/Dropbox/Projects/social_navigation_analysis/social_navigation_analysis')
    sys.path.insert(0, f'{user}/Dropbox/Projects/SNT_code/')
print(f'Base directory: {base_dir}')
fig_dir = f'{base_dir}/figures'

from general_utils import *
from images import (
    get_nifti_info, resample_nifti, get_timeseries, save_as_nifti)
from regression import run_ols
from classification import KDEClassifier, TreeEmbeddingLogisticRegression
from mvpa import (
    get_corrdist_matrix, fit_huber, calc_ps, fit_mds)
from geometry import * 
# from trajectories import *
# from circ_stats import (
#     circular_hist, calculate_angle, 
#     angle_between_vectors, circ_corrcl, circ_corrcc, circ_wwtest,
#     circ_rtest, circ_vtest, circ_mean, circ_cstd
#     )
from func_conn import compute_region_to_voxel_fc
from plotting import *
from socialspace import * 

from utils_timeseries import load_ts_mat
from preprocess import ComputeBehavior2

# from dim_reduction import Decompose, CrossDecompose
# tdqm.auto issue on minerva... not sure why? must be some newer package...
# https://github.com/swansonk14/p_tqdm/issues/16
# from tda import calc_mapper_graph

#-----------------------------------------------------------------------------------------------------------
# load data
#-----------------------------------------------------------------------------------------------------------

def update_sample_dict(data):
    return {'Initial': {'data': data[data['sample']=='Initial'].reset_index(drop=True), 'color': 'Blue'},
            'Validation': {'data': data[data['sample']=='Validation'].reset_index(drop=True), 'color': 'Green'}}

try:
    data = pd.read_excel(f'/{data_dir}/All-data_summary_n105.xlsx')
    # incl_info = pd.read_excel(f'/{data_dir}/participants_info_n105.xlsx')
    # incl = incl_info[incl_info['map_incl'] == 1]['sub_id'].values.astype(int)
    # data = data[data['sub_id'].isin(incl)]
    data = data[data['map_incl'] == 1]
    data = data.reset_index(drop=True)

    # most basic inclusion:
    fd_mask = data['fd_mean'] < 0.3
    data = data[fd_mask]
    incl = data['sub_id'].values
    print(f'Included n={len(data)}')

    sample_dict = update_sample_dict(data)
    samples = list(sample_dict.keys())
    sample_colors = [sample_dict[s]['color'] for s in samples]
except:
    print('Could not load behavioral summary data')

if user != '/hpc/users/schafm03':

    # masks
    mask_dir = f'{base_dir}/../Masks'
    mask_files = glob.glob(f'{mask_dir}/*.nii')
    print(f'Found {len(mask_files)} mask nifties')

# subject exclusions
repl_excl = ['18006', '20002', '20003', '20005', '21002', '21008', '21012', '21020', '22001'] 
orig_excl = ['13', '11', '4']
excl      = [int(e) for e in orig_excl + repl_excl]

demo_controls = ['sex', 'age_years']

# fmri info (all resampled to validation sample)
# - initial shape: (53, 63, 52, 784) (x,y,z,tr)
# - validation shape: (75, 90, 74, 1570)
affine = np.array([[  -2.0999999,    0.       ,    0.       ,   78.       ],
                    [   0.       ,    2.0999999,    0.       , -112.       ],
                    [   0.       ,    0.       ,    2.0999999,  -70.       ],
                    [   0.       ,    0.       ,    0.       ,    1.       ]])
vox_size = (2.1, 2.1, 2.1)
dims = (75, 90, 74)


#-----------------------------------------------------------------------------------------------------------
# project data helpers
#-----------------------------------------------------------------------------------------------------------

def get_cols(substr):
    # use wildcards to find columns in a dataframe
    return data.filter(regex=substr)

def get_fnames(wildcard, glm='lsa_decision_128hpf', ftype='embeddings'):
    fnames = glob.glob(f'{base_dir}/{glm}/{ftype}/*{wildcard}*')
    fnames = [f for f in fnames if int(f.split('/')[-1].split('_')[0]) in incl]
    print(f'Found {len(fnames)} {ftype} files')
    subs = [int(f.split('/')[-1].split('_')[0]) for f in fnames]
    missing = [s for s in incl if s not in subs]
    if len(missing) > 0:
        print(f'Missing subs: {missing}')
    return fnames, missing

def preprocess_betas(betas, zscore=False, neutrals=True):
    # simple preprocessing for betas
    if not neutrals: betas = remove_neutrals(betas) # remove neutral trials
    betas = VarianceThreshold().fit_transform(betas) # remove voxels with 0 variance
    if zscore: betas = scipy.stats.zscore(betas, axis=0) # zscore
    return betas

def get_behav_trajectories(decision_data=None):
    
    # get decision data
    if decision_data is None:
        decision_data = get_cols('decision_')
    else:
        decision_data = decision_data.filter(regex='decision_')
    decision_cols = [c for c in decision_data.columns if ('fd_' not in c) & (len(c.split('_')) == 2)]
    assert len(decision_cols) == 63
    decision_data = decision_data.loc[:, decision_cols].values

    # get character trajectories
    trajectories  = []
    for c in range(1, 6): 

        char_dims  = decision_trials[decision_trials['char_role_num'] == c]['dimension']
        affil_mask = char_dims.values == 'affil'
        power_mask = char_dims.values == 'power'

        character_decisions = decision_data[:, char_dims.index]
        character_coords = np.zeros((character_decisions.shape[0], 12, 2))
        character_coords[:, affil_mask, 0] = character_decisions[:, affil_mask]
        character_coords[:, power_mask, 1] = character_decisions[:, power_mask]

        trajectories.append(np.cumsum(character_coords, axis=1))
    trajectories = np.array(trajectories) 
    trajectories = np.transpose(trajectories, (1, 0, 2, 3)) # returns shape of (n_subjects, n_characters, n_trials, n_dimensions)
    return trajectories

def load_behavior(sub_id, neutrals=True):
    behav_files = glob.glob(f'{base_dir}/behavioral_data/behavior/*.xlsx')
    try:
        behav_fname = [f for f in behav_files if f'SNT_{sub_id}' in f][0]
        df = pd.read_excel(behav_fname)
    except:
        print(f'No behavior file for {sub_id}')
        return None
    if not neutrals:
        df = df[df['char_role_num'] != 6].reset_index(drop=True)
    return df

def load_choices(sub_id, neutrals=True):
    behav_files = glob.glob(f'{base_dir}/behavioral_data/choices/*.xlsx')
    try:
        behav_fname = [f for f in behav_files if f'SNT_{sub_id}' in f][0]
        df = pd.read_excel(behav_fname)
    except:
        print(f'No behavior file for {sub_id}')
        return None
    if not neutrals:
        df = df[df['char_role_num'] != 6].reset_index(drop=True)
    return df


#-----------------------------------------------------------------------------------------------------------
# data object helpers
#-----------------------------------------------------------------------------------------------------------


def assign_nested_dict_value(dictionary, keys_list, value):
    ''' Assign a value to a nested dictionary using a list of keys '''
    for key in keys_list[:-1]:
        if key not in dictionary:
            dictionary[key] = {}
        dictionary = dictionary[key]
    dictionary[keys_list[-1]] = value

def create_nested_dict_keys(dictionary, keys_list):

    exists = 1
    current_dict = dictionary
    for key in keys_list:
        if key not in current_dict:
            current_dict[key] = {}
            exists = 0
        current_dict = current_dict[key]

    return exists, dictionary

def check_key_exists(my_dict, key_name):
    # check if a key is in dictioanry
    return key_name in my_dict

def incl_df(df, incl_subs):
    # mask a dataframe based on sub_ids (integers)
    return df[np.isin(df['sub_id'].values.astype(int), incl_subs)]

def sort_reset(df, by='sub_id'):
    df.sort_values(by, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def df_subset_on_cols(df, subset_dict):
    for col,value in subset_dict.items():
        df = df[df[col] == value]
    if 'sub_id' in df.columns:
        df = df.sort_values('sub_id').reset_index(drop=True)
    return df

def digitize_df(df, cols=None, n_bins=10, zscored=True):
    if cols is None:
        cols = []
    df_ = df.copy()
    if len(cols) == 0:
        cols = df_.columns
    for col in cols:
        bins_ = stats.mstats.mquantiles(df[col], np.linspace(0., 1.0, num=n_bins, endpoint=False)) 
        df_[col] = digitized = np.digitize(df[col], bins_, right=True)
    return df_
    

#-----------------------------------------------------------------------------------------------------------
# minerva helpers
#-----------------------------------------------------------------------------------------------------------


def make_dirs(dirs_):
    '''makes directorys if doesnt exist'''
    [os.makedirs(d, exist_ok=True) for d in dirs_]

def get_sub_info(sub_dir):
    ''' parse the sub_dir string into sub_id and sample
        assumes minerva directory structure
    '''
    sub_id = sub_dir.split('/')[-1]
    sample = sub_dir.split('/')[-5]
    return sub_id, sample
    
def get_mask_name(mask_path):
    mask_nii  = mask_path.split('/')[-1]
    region    = ('_').join(mask_nii.split('_')[1:3])
    threshold = mask_nii.split('-thr')[-1].replace('-1mm.nii', '')
    return f'{region}_thr{threshold}'

def get_lsa_beta_img(sub_dir):
    ''' returns beta image fname for LSA model 
        resamples images if needed, returns beta image fname
    '''
    sub_id = sub_dir.split('/')[-1]
    img_names = ['spmT_decisions', 'mask', 'beta_decisions']
    for img_name in img_names:
        img_fname = f'{sub_dir}/{sub_id}_{img_name}.nii'
        img_shape = get_nifti_info(img_fname)[0]

        if img_shape[:3] != (75, 90, 74):
            img_fname = f'{sub_dir}/{sub_id}_{img_name}_resampled.nii' 
            if not os.path.isfile(img_fname): 
                print(f'Resampling {img_name}...')
                img_resampled = resample_nifti(f'{sub_dir}/{sub_id}_{img_name}.nii', 
                                               affine, dims, interpolation='nearest')
                save_as_nifti(img_resampled.get_data(), img_fname, affine, vox_size)

    return img_fname 

def run_beta_extraction(sub_dir, mask_dir):

    # outputs a .pkl with the extracted beta series for each ROI from mask_dir into sub_dir
    # sub_dir needs to have an lsa image, with specific naming convention

    ##-----------------------------------------------------------------
    # print('Running beta extraction')
    ##-----------------------------------------------------------------

    # parse sub_dir string
    strsplit = sub_dir.split('/')
    sub_id = strsplit[-1]
    print(f'Subject: {sub_id}, glm: {strsplit[-2]}')

    # load image & pkl
    beta_img_fname = get_lsa_beta_img(sub_dir)
    beta_fname = f'{sub_dir}/{sub_id}_beta_decisions_roi.pkl'
    beta_dict  = pd.read_pickle(beta_fname) if os.path.exists(beta_fname) else {}

    # extract betas
    for mask_fname in glob.glob(f'{mask_dir}/*.nii'):
        mask_name = get_mask_name(mask_fname)
        if mask_name not in beta_dict.keys():
            print(f'Extracting betas from {mask_name}', end='\r')
            betas = get_timeseries(beta_img_fname, mask=mask_fname, mask_type='roi')[0].T
            beta_dict[mask_name] = betas.astype(np.float32)
            pd.to_pickle(beta_dict, beta_fname) # interim save
        else:
            print(f'Betas for {mask_name} already extracted', end='\r')
 
    return beta_dict 


#-----------------------------------------------------------------------------------------------------------
# generic helpers
#-----------------------------------------------------------------------------------------------------------


def circular_shifts(arr, axis=None):
    """
        Generate all unique circular shifts of a 1D or 2D array along the specified axis.

        Parameters:
        arr (numpy.ndarray): Input array. Should be either 1D or 2D.
        axis (int, optional): Axis along which to shift. Must be 0 or 1 for 2D array. Ignored for 1D array.

        Returns:
        list: List of unique circularly shifted arrays.

    """
    arr = np.array(arr)
    if len(arr.shape) == 1:
        return [np.roll(arr, i) for i in range(1, len(arr))]
    elif len(arr.shape) == 2 and axis in [0, 1]:
        return [np.roll(arr, i, axis=axis) for i in range(1, arr.shape[axis])]
    else:
        raise ValueError("Invalid input. Input must be a 1D or 2D array and axis must be 0 or 1.")


#-----------------------------------------------------------------------------------------------------------
# social location analyses
# - distance binned pattern similarity
# - place rsa (huber regression)
# - classifier confusability
#-----------------------------------------------------------------------------------------------------------


class DistanceBinnedPs:

    # pattern similarity binned by distances

    def __init__(self, n_bins=3, masking=None):

        # distances are binned to get pattern similarity

        self.n_bins  = n_bins
        self.masking = masking

    def make_mask(self, masking=None):

        if masking is None: masking = self.masking
        
        if masking in ['between', 'within']:
            char_trials = np.array(decision_trials['char_role_num'])[:, np.newaxis]
            if self.distances.shape == (1770, 1): # excluded neutral
                char_trials = char_trials[char_trials != 6][:, np.newaxis]

            char_rdv = symm_mat_to_ut_vec(pairwise_distances(char_trials, metric='euclidean'))
            char_rdv[char_rdv > 0] = 1 # make categorical

            if masking == 'between': # diff characters 
                self.mask = char_rdv == 1
            elif masking == 'within': # same character
                self.mask = char_rdv == 0

        elif masking is None:   

            self.mask = np.ones(len(self.pattern_sim)).astype(bool) # include all trials

        else: 
            
            self.mask = masking

    def bin_distances(self, n_bins=None, strategy='quantile'):

        ''' 
            quantile bins are more even in count/size
            kmeans bins are more even in distance
        '''
        
        if n_bins is None: n_bins = self.n_bins

        binner          = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        self.bin_ixs    = binner.fit_transform(self.distances[self.mask]).flatten()
        self.bin_counts = [np.sum(self.bin_ixs==b) for b in range(n_bins)]
        self.mean_dists = np.array([np.mean(self.distances[self.mask][self.bin_ixs==b]) for b in range(n_bins)])

    def calc_pattern_similarity(self):

        self.ps_bins     = [self.pattern_sim[self.mask][self.bin_ixs==b] for b in range(self.n_bins)] # list of binned ps arrays
        self.mean_ps     = [np.mean(np.vstack(self.ps_bins))]
        self.binned_ps   = [np.mean(bin_) for bin_ in self.ps_bins]
        self.demeaned_ps = list(self.binned_ps - self.mean_ps[0])

        self.df = pd.DataFrame(np.hstack([self.binned_ps, self.mean_ps, self.demeaned_ps, self.bin_counts])[np.newaxis]) # length==n_bin * 3 + 1
        self.df.columns = [f'ps_bin{b+1:02d}' for b in range(self.n_bins)] + \
                          ['ps_bins_mean'] + [f'ps_bin{b+1:02d}_demeaned' for b in range(self.n_bins)] + \
                          [f'ps_bin{b+1:02d}_pairs' for b in range(self.n_bins)]

    def run(self, distances, pattern_sim):

        if distances.ndim == 1: distances = distances[:, np.newaxis]
        if pattern_sim.ndim == 1: pattern_sim = pattern_sim[:, np.newaxis]
        self.distances = distances
        self.pattern_sim = pattern_sim
        self.make_mask()
        self.bin_distances()
        self.calc_pattern_similarity()
        return self.df

class ClassifierConfusability:

    def __init__(self, rdm, clf_output):

        self._check_inputs(rdm, clf_output)

        # maybe just hard code the character labels... shouldnt depend on outside file probably
        self.character_labels = remove_neutrals(decision_trials['char_role_num'].values)
        self.df = pd.DataFrame()
        self._sorted = False

    def _check_inputs(self, rdm, clf_output):

        if rdm.shape != (5,5):
            raise Exception(f'rdm shape {rdm.shape}!= (5,5)')
        if clf_output.shape != (60,6):
            raise Exception(f'clf_output shape {clf_output.shape}!= (60,6)')
        
        self.rdm = rdm
        self.clf_ouput_raw = clf_output

    def run(self):
        
        self.sort()
        self.scale()
        self.similarity()
        self.difference()
        return self.df

    def sort(self):
        
        self._sorted = True

        # sort trials by trial num
        self.clf_ouput = self.clf_ouput_raw.copy()
        self.clf_ouput.sort_values(by='index', inplace=True) # so it needs these columns....
        self.clf_ouput.reset_index(drop=True, inplace=True)
        self.clf_ouput.drop(["index"], axis=1, inplace=True)
        self.clf_ouput.insert(1, 'char_role_num', self.character_labels)

        self.characterwise_unsorted = self.clf_ouput.groupby('char_role_num').mean().values

        # sort trialwise output by the end-of-task distances 
        rdm = self.rdm.copy()
        rdm = rdm + 1 # break distance ties w/ diagonal (each character w/ itself)
        np.fill_diagonal(rdm, 0) 
        self.sorter = rdm.argsort() # first col should be: 0,1,2,3,4 (the diagonals are the correct characters)

        trialwise_sorted = np.zeros((60, 5))
        for r,row in self.clf_ouput.iterrows():
            char_ix = int(row['char_role_num'] - 1) # which row of matrix
            trial_  = row[['probas_class01', 'probas_class02', 'probas_class03', 'probas_class04', 'probas_class05']].values.astype(float)
            trialwise_sorted[r,:] = trial_[self.sorter[char_ix]]

        self.trialwise_sorted = pd.DataFrame(trialwise_sorted)
        
        self.characterwise_sorted = self.trialwise_sorted.copy()
        self.characterwise_sorted.insert(0, 'char_role_num', self.character_labels)
        self.characterwise_sorted = self.characterwise_sorted.groupby('char_role_num').mean().values # mean of sorted probas within each character
                
    def scale(self):

        if not self._sorted: self.sort()

        # scale probabilities by % of incorrect
        probas_incorr      = np.mean(self.trialwise_sorted, axis=0)[1:5] 
        self.probas_scaled = probas_incorr/np.sum(probas_incorr)

        # scale mispredictions
        pred_counts       = Counter([np.where(row==np.amax(row))[0][0] for row in self.trialwise_sorted.values])
        self.preds_scaled = np.array([pred_counts[c] for c in range(1, 5)])/(60 - pred_counts[0])

        self.df.loc[0, ['probas_class02', 'probas_class03', 'probas_class04', 'probas_class05',
                        'preds_class02',  'preds_class03',  'preds_class04', 'preds_class05']] = list(self.probas_scaled) + list(self.preds_scaled)

    def similarity(self):

        if not self._sorted: self.sort()

        # kendalls tau: as distances between the characters get bigger, do confusion probas get smaller? (negative tau)
        probas       = symm_mat_to_ut_vec(self.characterwise_sorted)
        rdv          = symm_mat_to_ut_vec(self.rdm)
        self.tau, _  = kendalltau(rdv, probas)
        self.df.loc[0, ['confusability_tau'] + [f'proba_rdv{d:02d}' for d in np.arange(1,11)]] = [self.tau] + list(probas)

    def difference(self):

        if not self._sorted: self.sort()

        prefs_diff   = np.mean(self.preds_scaled[:2])  - np.mean(self.preds_scaled[2:])
        probas_diff  = np.mean(self.probas_scaled[:2]) - np.mean(self.probas_scaled[2:])
        self.df.loc[0, ['probas_diff', 'preds_diff']] =  [probas_diff, prefs_diff]

class ClassifierOptimization:

    __slots__ = ['X', 'y', 'n_classes', 'clf', 'pl_steps', 'gridsearch_params',
                 'cv', 'folds', 'n_folds', 'pipeline', 'results', 'fitted']

    def __init__(self, X, y, clf, gridsearch_params, pl_steps=None, cv=6):

        if X.shape[0] != y.shape[0]:
            raise Exception(f"X & y dont same the number of rows: X shape={X.shape}, y shape{y.shape}")
        
        self.X         = VarianceThreshold().fit_transform(X) # remove 0 variance features
        self.y         = y
        self.n_classes = len(np.unique(y))
        self.clf       = clf
        self.pl_steps  = pl_steps
        self.gridsearch_params = gridsearch_params

        # specify the cross-validation
        self.cv = cv
        if isinstance(self.cv, int): # stratified to balance each fold by class (i.e., character) 
            self.folds   = StratifiedKFold(n_splits=self.cv, random_state=76, shuffle=True).split(self.X, self.y)
            self.n_folds = self.cv
        elif (isinstance(self.cv, str)) & (self.cv=='loo'): # leave one out
            self.folds   = LeaveOneOut().split(self.X, self.y)
            self.n_folds = len(self.X)
        else: # iterator
            self.folds = self.cv

        self.fitted = False

    def make_pipeline(self):
        
        pl_dict = {'scaler': StandardScaler(), 
                    'selector': VarianceThreshold(threshold=0.2), 
                    'pca': PCA(n_components=25)}
        if self.pl_steps is None: self.pl_steps = []
        self.pipeline = Pipeline([(step, pl_dict[step]) for step in self.pl_steps] + [('classifier', self.clf)])

        # add to the params
        params = list(self.gridsearch_params.keys())
        for param in params:
            self.gridsearch_params[f'classifier__{param}'] = self.gridsearch_params[param]
            del self.gridsearch_params[param]
    
    def fit_predict(self):
        
        self.make_pipeline()

        folds = []
        eval = np.zeros(shape=(self.n_folds,), dtype=[('cv_split', 'uint8'), ('best_gs_param', 'float32'), 
                                                        ('accuracy', 'float32'), ('balanced_accuracy', 'float32'), 
                                                        ('f1', 'float32'), ('phi', 'float32')])
        for k, (train, test) in enumerate(self.folds):

            # inner loop: optimize pipeline w/ grid search
            grid = GridSearchCV(self.pipeline, self.gridsearch_params, cv=2, n_jobs=-1) # n_jobs=-1 : use all available cores in parallel
            grid.fit(self.X[train], self.y[train])

            # outer loop: test best pipeline
            preds = grid.predict(self.X[test])

            fold   = np.vstack([test, np.repeat(k+1, len(preds)), preds, self.y[test], (preds == self.y[test]) * 1]).T 
            dtypes = [('index', 'uint8'), ('cv_split', 'uint8'), ('predicted_class', 'uint8'), ('actual_class', 'uint8'), ('correct', 'bool')]
            if hasattr(grid, 'predict_proba'):
                fold = np.hstack([fold, grid.predict_proba(self.X[test])])
                dtypes.extend([(f'probas_class{p+1:02d}', 'float32') for p in range(self.n_classes)])

            folds.append(fold)
                    
            # evaluate performance
            eval[k]['cv_split'] = k + 1
            eval[k]['best_gs_param'] = list(grid.best_params_.values())[0]
            eval[k]['accuracy'] = grid.score(self.X[test],self. y[test])
            eval[k]['balanced_accuracy'] = balanced_accuracy_score(self.y[test], preds)
            eval[k]['f1'] = f1_score(self.y[test], preds, average='weighted')
            eval[k]['phi'] = matthews_corrcoef(self.y[test], preds)
        
        folds = np.sort(rfn.unstructured_to_structured(np.vstack(folds), np.dtype(dtypes)))

        # output dict
        self.results = {'pipeline': self.pipeline,
                        'cross-validation': self.cv, 
                        'predictions': pd.DataFrame(folds),
                        'evaluation': pd.DataFrame(eval)} 

        self.fitted = True
        self.confusion_matrix()

    def confusion_matrix(self):

        if not self.fitted: self.fit_predict()
            
        pred = self.results['predictions']['predicted_class'].values
        y    = self.results['predictions']['actual_class'].values
        conf_matrix = confusion_matrix(y, pred)
        self.results['confusion_matrix'] = conf_matrix / np.sum(conf_matrix, 1) * 100

# wrappers
def run_clf_conf(clf, X, y, cv=6, standardize=False):

    # TODO: accept pipelines...
    
    eval_df  = pd.DataFrame()
    pred_dfs = []

    if isinstance(cv, int): # stratified to balance each fold by class (i.e., character) 
        folds = StratifiedKFold(n_splits=cv, random_state=76, shuffle=True).split(X, y)
    elif (isinstance(cv, str)) & (cv=='loo'): # leave one out
        folds = LeaveOneOut().split(X, y)
    else: # iterator
        folds = cv

    # drop features with no variation
    X = VarianceThreshold().fit_transform(X) 

    # cross-validated decoding
    for k, (train, test) in enumerate(folds):

        # if standardizing, fit a scaling model on training folds
        if standardize: 
            scaler  = StandardScaler().fit(X[train])     
            X_train = scaler.transform(X[train])
            X_test  = scaler.transform(X[test])
        else:
            X_train = X[train].copy()
            X_test  = X[test].copy()

        # fit classifier on training folds
        decoder = clone(clf)
        decoder.fit(X_train, y[train]) 

        # predict on held out fold
        y_preds = decoder.predict(X_test)
        pred_df = pd.DataFrame(np.vstack([test, y_preds, y[test], (y_preds == y[test]) * 1]).T,
                               columns=['index', 'predicted', 'actual', 'correct'])
        pred_df.insert(0, 'split', k)

        # evaluate performance
        eval_df.loc[k, 'split']    = k
        eval_df.loc[k, 'accuracy'] = decoder.score(X_test, y[test])

        # get probabilities
        if hasattr(decoder, 'predict_proba'): 
            y_probas = decoder.predict_proba(X_test)
            for p, y_probas_ in enumerate(y_probas.T):
                pred_df[f'probas_class{p+1:02d}'] = y_probas_

        pred_dfs.append(pred_df)
    pred_df = pd.concat(pred_dfs)
    pred_df.reset_index(inplace=True, drop=True)

    # output dict
    clf_dict = {'cross-validation': cv, 
                'predictions': pred_df,
                'evaluation': eval_df} 
    
    return clf_dict

def run_rsa_analyses(sub_dir, base_dir=None, model_name='place_2d'):

    # if base_dir is None: base_dir = '/sc/arion/projects/OlfMem/mgs/2D_place' 
    masks = glob.glob(f'{base_dir}/Masks/*.nii')

    # sub info & directories
    sub_id, sample = get_sub_info(sub_dir)
    sample_dir = f'{base_dir}/Samples/{sample}'
    print(f'Running LSA pipeline for {sub_id} from {sample} sample')

    for dir_ in ['Results', 'Betas', 'Func_conn']: 
        if not os.path.exists(f'{sample_dir}/{dir_}'): os.makedirs(f'{sample_dir}/{dir_}')

    ##---------------------------------------------------------------------------------------------------------
    # print('Loading data')
    ##---------------------------------------------------------------------------------------------------------

    # load the model-based behavioral rdvs
    beh_rdvs   = pd.read_excel(f'{sample_dir}/SNT/RDVs/snt_{sub_id}_rdvs.xlsx')
    rdvs_z     = zscore(beh_rdvs[['time1','time2','time3',
                                  'reaction_time','slide',
                                  'scene','familiarity', model_name]])
    other_rdvs = beh_rdvs[['button_press','char1','char2','char3','char4','char5']]
    model_rdvs = pd.concat([rdvs_z, other_rdvs], axis=1)
        
    # get rdvs for end trials only
    beh_end_rdv = pd.read_excel(f'{sample_dir}/SNT/RDVs/snt_{sub_id}_end_rdvs.xlsx', 
                                usecols=[model_name]).values
    beh_end_rdm = ut_vec_to_symm_mat(beh_end_rdv.flatten()) 
    
    # load the lsa image
    lsa_img_fname = get_lsa_img(sub_dir) # will resample if needed
    image_shape = get_nifti_info(lsa_img_fname)[0]

    # perform some checks
    if model_rdvs.shape != (1953, 14): 
        raise Exception(f'rdv shape is {model_rdvs.shape} and should be (1953, 14)')


    ##---------------------------------------------------------------------------------------------------------
    # print('Running ROI analyses')
    ##---------------------------------------------------------------------------------------------------------


    # dictionaries to save the results
    results_dict = {'betas':  {'fname': f'{sample_dir}/Betas/{sub_id}_betas.pkl', 'dict': {}},
                    'huber':  {'fname': f'{sample_dir}/Results/{sub_id}_huber.pkl', 'dict': {}},
                    'lr':     {'fname': f'{sample_dir}/Results/{sub_id}_lr.pkl', 'dict': {}},
                    'mds':    {'fname': f'{sample_dir}/Results/{sub_id}_mds.pkl', 'dict': {}}}
    for key in results_dict.keys():
        if os.path.isfile(results_dict[key]['fname']): 
            results_dict[key]['dict'] = pd.read_pickle(results_dict[key]['fname'])
            print(f'Loaded {key} results for {sub_id}')

    beta_dict = results_dict['betas']['dict']
    results, mask_names = [], []

    # loop over the roi masks
    for mask_path in masks: 

        ##-----------------------------------------------------------------
        # preallocate a structued array to hold results summary
        # - float32 is prob. sufficient precision, saves space & time
        ##-----------------------------------------------------------------

        float_dtype = 'float32'
        ps_bins, mds_dims  = 3, 10
        dtypes = [('mask_shapeX', 'uint8'), ('mask_shapeY', 'uint8'), 
                  ('mask_shapeZ', 'uint8'), ('mask_voxels', 'uint32')]
        dtypes += [('huber_beta', float_dtype)]
        dtypes += [('ps_bins_mean', float_dtype)]
        dtypes += [(f'ps_bin0{b}', float_dtype) for b in range(1, ps_bins+1)]
        dtypes += [(f'ps_bin0{b}_demeaned', float_dtype) for b in range(1, ps_bins+1)] 
        dtypes += [(f'ps_bin0{b}_pairs', float_dtype) for b in range(1, ps_bins+1)]
        dtypes += [('clf_acc', float_dtype), ('confusability_tau', float_dtype), 
                   ('probas_diff', float_dtype), ('preds_diff', float_dtype)]
        dtypes += [(f'probas_class0{b}', float_dtype) for b in range(2, 6)]
        dtypes += [(f'preds_class0{b}', float_dtype) for b in range(2, 6)]
        dtypes += [(f'proba_rdv{b:02d}', float_dtype) for b in range(1, 11)]
        dtypes += [('ps_mean', float_dtype), ('ps_std', float_dtype)]
        dtypes += [(f'mds_{d}d_stress', float_dtype) for d in range(2, mds_dims+1)]
        out = np.zeros((1,), dtype=dtypes)

        ##-----------------------------------------------------------------
        # get data
        ##-----------------------------------------------------------------

        # get the mask
        mask_shape = get_nifti_info(mask_path)[0]
        mask_nii   = mask_path.split('/')[-1]
        region     = ('_').join(mask_nii.split('_')[1:3])
        threshold  = mask_nii.split('-thr')[-1].replace('-1mm.nii', '')
        mask_name  = f'{region}_thr{threshold}' 
        mask_names.append(mask_name)

        # extract betas from masked region
        if mask_name not in beta_dict.keys():
            betas = get_timeseries(lsa_img_fname, mask=mask_path, mask_type='roi')[0].T
            beta_dict[mask_name] = betas
        else:
            betas = beta_dict[mask_name]

        out[['mask_shapeX', 'mask_shapeY', 'mask_shapeZ', 'mask_voxels']] = \
            (mask_shape[0], mask_shape[1], mask_shape[2], betas.shape[1])
        
        # turn beta patterns into dissimilarity matrix
        beta_rdm = get_rdm(betas)
        beta_rdv = symm_mat_to_ut_vec(beta_rdm)
        if beta_rdv.shape != (1953,): 
            raise Exception(f'rdv shape {beta_rdv.shape} != (1953,)')

        ##-----------------------------------------------------------------
        # print('Running distance-binned pattern similarity analysis')
        ##-----------------------------------------------------------------

        dist_ps = DistanceBinnedPs(masking='between', n_bins=ps_bins)
        dist_df = dist_ps.run(model_rdvs[model_name], 1-beta_rdv)        
        out[dist_df.columns] = (*(dist_df.values).flatten(), ) # needs tuple


        ##-----------------------------------------------------------------
        # print('Running huber regression analysis')
        ##-----------------------------------------------------------------

        huber_dict = results_dict['huber']['dict']
        if mask_name not in huber_dict.keys():
            huber_coefs = fit_huber(model_rdvs, beta_rdv) # X, y
            huber_df = pd.DataFrame(huber_coefs[np.newaxis,:], columns=model_rdvs.columns)
            huber_dict[mask_name] = huber_df 
        else:
            huber_df = huber_dict[mask_name]
        
        out['huber_beta'] = huber_df[model_name].values[0]

        ##-----------------------------------------------------------------
        # print('Running confusability analysis')
        ##-----------------------------------------------------------------

        # logisitc regression
        lr_dict = results_dict['lr']['dict']
        if mask_name not in lr_dict.keys():
            X = remove_neutrals(betas)
            y = remove_neutrals(decision_trials['char_role_num'].values)
            lr = LogisticRegression(penalty='l2', C=10, multi_class='multinomial', 
                                    solver='lbfgs', max_iter=10000, random_state=0, n_jobs=-1)
            clf = run_clf(lr, X, y, cv=6, standardize=False)
            lr_dict[mask_name] = clf
        else:
            clf = lr_dict[mask_name]

        # compare probas to behavior
        lr_probas = clf['predictions'][['index'] + [f'probas_class{c+1:02d}' for c in range(5)]]
        clf_conf  = ClassifierConfusability(beh_end_rdm, lr_probas).run()

        out['clf_acc'] = np.mean(clf['evaluation']['accuracy']) 
        out[clf_conf.columns] = (*(clf_conf.values).flatten(), )

        # [if validation sample] run with dots
            # load in dots xlsx
            # make sure its correct character order
            # reshape into 5,2 array
            # get distances between dots locations
            # rerun with these distances

        ##-----------------------------------------------------------------
        # print('Running average pattern similarity analysis')
        ##-----------------------------------------------------------------

        out[['ps_mean', 'ps_std']] = (*calc_ps(betas), )

        ##-----------------------------------------------------------------
        # print('Running mds long axis analysis')
        ##-----------------------------------------------------------------
        
        mds_dict = results_dict['mds']['dict']
        if mask_name not in mds_dict.keys():
            mds_dict[mask_name] = {}            
            for n_dim in range(2, mds_dims + 1):
                embedding, stress = fit_mds(beta_rdm, n_components=n_dim)
                mds_dict[mask_name][f'{n_dim}d'] = {'embedding': embedding, 'stress': stress}
                out[f'mds_{n_dim}d_stress'] = stress
        else:
            for n_dim in range(2, mds_dims + 1):
                stress = mds_dict[mask_name][f'{n_dim}d']['stress']
                out[f'mds_{n_dim}d_stress'] = stress

        ##-----------------------------------------------------------------
        # print('Outputting results')
        ##-----------------------------------------------------------------
        
        results.append(out)

        # for each dictionary, save the results in ongoing manner w/ each ROI
        for key in results_dict.keys():
            pd.to_pickle(results_dict[key]['dict'], results_dict[key]['fname'])

    results = pd.DataFrame(np.hstack(results))
    results.insert(0, 'roi', mask_names)
    results.to_excel(f'{sample_dir}/Results/{sub_id}_results.xlsx', index=False)


#-----------------------------------------------------------------------------------------------------------
# embedding wrappers
#-----------------------------------------------------------------------------------------------------------


def pca_dim_reduction(data, cumvar_threshold=0.8, dr_f=None, **kwargs):

    # perform PCA and retain components that explain the desired percentage of variance
    pca = PCA(svd_solver='full', random_state=0)
    pca.fit(data)
    cumvar_explained = np.cumsum(pca.explained_variance_ratio_) # cumulative sum of variance explained
    n_comps = np.argmax(cumvar_explained > cumvar_threshold) + 1 # first component that exceeds the threshold
    comps = pca.components_[:n_comps, :] 
    pca_projected = (data - pca.mean_) @ (comps.T) # project the data onto the selected eigenvectors
    pca_recon = pca_projected @ comps + pca.mean_ # inverse transform to get denoised reconstruction
    
    # apply the specified dimensionality reduction technique on the PCA embedded data
    X_reduced = dr_f(**kwargs).fit_transform(pca_projected)

    return {'pca': {'X_pca': pca_projected, 
                    'cumvar_threshold': cumvar_threshold,
                    'n_comps': n_comps,
                    'cumvar_explained': cumvar_explained},
                     'X_reduced': X_reduced}

def pca_reconstruction(data, pca=None, n_comps=3):
    # https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
    if pca is None:
        pca = PCA(svd_solver='full', random_state=0)
        pca.fit(data)       
    pca_embedding = pca.transform(data)
    return np.dot(pca_embedding[:,:n_comps], pca.components_[:n_comps,:]) + np.mean(X, axis=0)

def rotate_with_pca(embedding):

    # rotate with pca
    pca = PCA(svd_solver='full', random_state=85)
    embedding_rotated = pca.fit_transform(embedding)

    # check rotation against original: pairwise distances should be the same
    distances = pairwise_distances(embedding)
    distances_rotated = pairwise_distances(embedding_rotated)
    assert np.allclose(distances, distances_rotated)
    
    return embedding_rotated

def calc_pca_embedding(data, n_components=3):
  pca = PCA(n_components=n_components)
  emb = pca.fit_transform(data)
  recon = pca.inverse_transform(emb) # find reconstruvtion in lowr dimensions
  return {'embedding': emb, 
          'inverse_transform': recon,
          'explained_variance': pca.explained_variance_ratio_, 
          'params': pca.get_params()}

def calc_mds_embedding(data, n_components=3, dissimilarity='correlation', **kwargs):
    if dissimilarity == 'correlation':
        dissimilarity = 'precomputed'
        X = 1 - np.corrcoef(data)
    mds = MDS(n_components=n_components, dissimilarity=dissimilarity, random_state=0, **kwargs)
    emb = mds.fit_transform(data)
    return {'embedding': emb, 'raw_stress': mds.stress_, 
            'kruskal_stress': convert_stress_raw_to_kruskal(data, mds.stress_), 
            'params': mds.get_params()}

def convert_stress_raw_to_kruskal(dm, raw_stress):
    ''' 
        a scaled version of the raw stress
        dm = orig. dissimilarity matrix
        - raw stress can be misleading b/c it depends on normalization of disparities
        - kruskal stress is value in range [0,1], where smaller is better
        - divide demoninator by two to make it only half of the symmetric matrix

        in sklearn, need to set metric to false to get normalized stress; but metric=True and this is v. similar
    '''
    return np.sqrt(raw_stress / (0.5 * np.sum(dm ** 2)))

def calc_isomap_embedding(data, n_components=3, n_neighbors=10, **kwargs):
    isomap = Isomap(n_components=n_components, 
                    n_neighbors=n_neighbors,
                    **kwargs)
    emb = isomap.fit_transform(data)
    return {'embedding': emb, 'geodesic_distances': isomap.dist_matrix_,
            'params': isomap.get_params()}

def calc_lle_embedding(data, n_components=3, n_neighbors=10, **kwargs):
    lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors,
                                 eigen_solver='dense',
                                 **kwargs)
    emb = lle.fit_transform(data)
    return {'embedding': emb, 
            'reconstruction_error': lle.reconstruction_error_, 
            'params': lle.get_params()}

def calc_umap_embedding(data, n_components=3, n_neighbors=10, **kwargs):
    # fitted object is small in memory; but gives issues pickling it on minerva
    # transform and inverse transform take time - best to do on minerva
    umap = UMAP(n_components=n_components, 
                 n_neighbors=n_neighbors,
                 **kwargs)
    umap.fit(data)
    emb = umap.transform(data)
    recon = umap.inverse_transform(emb)
    # 'obj': umap, - not sure how to pickle this?
    return {'embedding': emb, 
            'inverse_transform': recon,
            'params': umap.get_params()}

def calc_spec_embedding(data, n_components=3, n_neighbors=10, **kwargs):
    spec = SpectralEmbedding(n_components=n_components, 
                            n_neighbors=n_neighbors,
                            **kwargs)
    emb = spec.fit_transform(data)
    return {'embedding': emb, 'affinity_matrix': spec.affinity_matrix_, 
            'params': spec.get_params()}

def calc_tsne_embedding(data, n_components=3, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=n_components, 
                perplexity=perplexity, init='pca',
                n_iter=n_iter, random_state=0)
    return {'embedding': tsne.fit_transform(data), 'params': tsne.get_params()}

def calc_tphate_embedding(data, n_components=3, n_neighbors=10, 
                          mds_solver='smacof', mds='metric', **kwargs):
    tphate_operator = tphate.TPHATE(n_components=n_components, 
                                    knn=n_neighbors,
                                    mds_solver=mds_solver, # 'sgd' or 'smacof'
                                    mds=mds, # 'metric' or 'nonmetric' or 'classical'
                                    verbose=0)
    return {'embedding': tphate_operator.fit_transform(data)}

def run_calc_embeddings(data_fname,
                        rois=None, 
                        algos=None, 
                        n_components=None, 
                        n_neighbors=None):
    ''' 
        calculate embedding on decision trials

        arguments 
        ---------
        data_fname: path to betas .pkl or timeseries .mat file
        rois: list of rois to run on (if None, run on all)
        algos: list of algorithms to run (if None, run on all)
        n_components: list of n_components to run (if None, run on all)
        n_neighbors: list of n_neighbors to run (if None, run on all)
    '''

    ##-----------------------------------------------------------------
    # print('Running manifold learning analyses')
    ##-----------------------------------------------------------------

    # input
    sub_id = data_fname.split('/')[-1].split('_')[0]
    if data_fname.endswith('.pkl'): # betas
        base_dir = ('/').join(data_fname.split('/')[:-3])
        dt = 'betas'
        data_dict = pd.read_pickle(data_fname)
    elif data_fname.endswith('.mat'): # timeseries (w/ preprocesing)
        base_dir = ('/').join(data_fname.split('/')[:-2])
        dt = 'timeseries'
        tr = 1.0 if len(sub_id) > 2 else 2.0
        data_dict = load_ts_mat(data_fname, 
                                preprocess=True, 
                                decisions_only=True, 
                                high_pass=1/250,
                                tr=tr)
    
    # output
    print(f'Subject: {sub_id}, base dir: {base_dir}')
    out_dir = f'{base_dir}/embeddings'
    if not os.path.exists(out_dir): os.mkdir(out_dir)

    # run over parameter grid
    algo_dict = {'isomap': calc_isomap_embedding,
                 'lle': calc_lle_embedding,
                 'spec': calc_spec_embedding,
                 'umap': calc_umap_embedding,
                 'tsne': calc_tsne_embedding,
                 'pca': calc_pca_embedding,
                 'mds': calc_mds_embedding}
    algos = algos or ['umap', 'lle', 'pca']
    if dt == 'betas':
        n_neighbors  = n_neighbors or [5, 10, 20]
        n_components = n_components or [2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif dt == 'timeseries':
        n_neighbors  = n_neighbors or [25, 50, 100]
        n_components = n_components or [3, 10]
    rois = rois or ['L_HPC_thr25', 'R_HPC_thr25', 
                    'L_HPC-DG_thr25', 'R_HPC-DG_thr25', 'L_HPC-CA_thr25', 'R_HPC-CA_thr25', 'L_HPC-sub_thr25', 'R_HPC-sub_thr25', 
                    'L_HPC-ant_thr25', 'R_HPC-ant_thr25', 'L_HPC-mid_thr25', 'R_HPC-mid_thr25', 'L_HPC-post_thr25', 'R_HPC-post_thr25',  
                    'L_ERC_thr25', 'R_ERC_thr25', 'L_mpfc_thr25', 'R_mpfc_thr25', 'L_M1-p_thr25', 'R_M1-p_thr25']
    # rois = rois or list(data_dict.keys())
    param_grid = list(itertools.product(rois, n_components, n_neighbors))

    for algo in algos:

        # check for results
        results_fname = f'{out_dir}/{sub_id}_{algo}-embeddings.pkl'
        if os.path.exists(results_fname):
            print(f'Loading {algo} results')
            results_dict = pd.read_pickle(results_fname)
        else:
            results_dict = {}

        # define parameter grid
        if algo in ['pca', 'mds', 'tsne']:
            param_grid = list(itertools.product(rois, n_components))
        else:
            param_grid = list(itertools.product(rois, n_components, n_neighbors))

        # calculate embeddings
        for params in param_grid: 
            roi, nc = params[0], params[1] # all have roi & n_components
            key_list = [roi, f'{nc}d'] if algo in ['pca', 'mds', 'tsne'] else [roi, f'{nc}d', f'{params[2]}nn']
            exists, results_dict = create_nested_dict_keys(results_dict, key_list) # should return dictionary w/ keys, & whether they already existed
            if not exists:
                print(f'Calculating {algo} in {roi}')
                if algo in ['pca', 'mds', 'tsne']: 
                    results_dict[roi][f'{nc}d'] = algo_dict[algo](data_dict[roi], n_components=nc)
                else: # algos that also have a nearest neighbor parameter
                    nn = params[2]
                    results_dict[roi][f'{nc}d'][f'{nn}nn'] = algo_dict[algo](data_dict[roi], n_components=nc, n_neighbors=nn)
            else:
                print(f'{algo} {roi} already calculated')

            # save each emedding separately
            pd.to_pickle(results_dict, results_fname)


#-----------------------------------------------------------------------------------------------------------
# co-clustering analysis
#-----------------------------------------------------------------------------------------------------------


def calc_kmeans_clustering(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=23).fit(X)
    return {'labels': kmeans.labels_,
            'inertia': kmeans.inertia_,
            'cluster_centers_': kmeans.cluster_centers_, 
            'params': kmeans.get_params()}
    
def calc_ward_clustering(X, n_clusters=3):
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(X)
    return {'labels': ward.labels_, 'n_leaves': ward.n_leaves_,
            'n_connected_components': ward.n_connected_components_,
            'children': ward.children_, 'params': ward.get_params()}

def calc_spectral_clustering(X, n_clusters=3):
    # n_neighbors is ignored w/ rbf kernel, which is default
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=23).fit(X)
    return {'labels': spectral.labels_, 'affinity_matrix': spectral.affinity_matrix_, 
            'params': spectral.get_params()}

class ClusterClassifier:

    # cluster data and then classify data into same clusters

    def __init__(self, 
                 X_cluster_f=None, y_cluster_f=None, 
                 n_clusters=None, n_neighbors=None, 
                 overlap_f=None,
                 clf_f=None, 
                 random_state=None):

        # unsuperivsed clustering
        self.n_clusters  = n_clusters or 5
        self.X_cluster_f = X_cluster_f or SpectralClustering(n_clusters=self.n_clusters, 
                                                             affinity='rbf', # rbf or nearest_neighbors
                                                             assign_labels='kmeans',
                                                             random_state=0)
        self.y_cluster_f = y_cluster_f or KMeans(n_clusters=self.n_clusters, random_state=0) # or BisectingKMeans, but will allow mre diff. shapes of clusters
        
        # overlap measure to maximize
        self.overlap_f = overlap_f or adjusted_mutual_info_score

        # supervised classification: needed for non-linear clustering...
        self.n_neighbors = n_neighbors or 5
        self.clf_f = clf_f or KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                                   weights='distance', metric='euclidean',
                                                   algorithm='brute')
        
        
        self.rs = check_random_state(random_state) or 0

    def fit(self, X, y=None):

        # cluster states & build classification model
        # - can pass in y to choose cluster number that maximizes cluster overlap (mutual information) between X and y

        X = check_array(X)
        if y is not None:
            assert y.shape[0] == X.shape[0], 'X and y must have the same number of observations'
            self.cluster_max_overlap(X, y) # find the optimal number of clusters
            
        # cluster & fit classification model
        self.labels_ = self.X_cluster_f.fit_predict(X)
        self.clf_f.fit(X, self.labels_)

        return self

    def predict(self, X):

        check_is_fitted(self)
        X = check_array(X)
        return self.clf_f.predict(X)

    def cluster_max_overlap(self, X, y, X_cluster_f=None, y_cluster_f=None, overlap_f=None):
        
        # find the number of clusters that maximizes some measure of overlap between X & y clusterings
        X_cluster_f = X_cluster_f or self.X_cluster_f
        y_cluster_f = y_cluster_f or self.y_cluster_f
        overlap_f   = overlap_f or self.overlap_f
        n_clusters = (
            self.n_clusters if isinstance(self.n_clusters, range)
            else np.arange(2, self.n_clusters+1)
        )
        
        overlap = []
        for n_c in n_clusters:
            X_cluster_f.set_params(n_clusters=n_c)
            y_cluster_f.set_params(n_clusters=n_c)
            overlap.append(self.overlap_f(X_cluster_f.fit_predict(X), 
                                          y_cluster_f.fit_predict(y)))
        self.overlap_value = np.max(overlap)
        self.n_clusters = n_clusters[np.argmax(overlap)]
        self.X_cluster_f.set_params(n_clusters=self.n_clusters) # reassign n_clusters in X_cluster_f

def predict_behavioral_trajectory(train_states, train_xy, test_states, test_xy):

    n_states = len(np.unique(train_states))
    
    #--------------------------------------------------------------------------------
    # parameterize the states w/ mean behavior (xy) for each state cluster
    #--------------------------------------------------------------------------------

    # train_xy = train_behav[['affil_coord', 'power_coord']].values
    train_states_xy = np.vstack([np.mean(train_xy[train_states==c], axis=0).astype(float) for c in range(n_states)])
    
    #--------------------------------------------------------------------------------
    # predict the behavior for test character
    #--------------------------------------------------------------------------------

    # test_xy = test_behav[['affil_coord', 'power_coord']].values # actual xy locations
    pred_xy = np.cumsum(train_states_xy[test_states], axis=0) # sum up xy averages for each state

    # scale both observed and predicted trajectories to between [-1,+1] by dividing by the max poss. abs. value on x and y 
    test_xy = test_xy / 6 # each dimension has 6 decisions w/ 1 unit move
    pred_xy = pred_xy / (np.max(np.abs(train_states_xy), axis=0) * len(test_states)) # max(abs(xy)) * 12

    # null behavior-only model: average scaled location for train characters
    # train_mean_xy = np.mean(train_xy / 6, axis=0) # max possible (abs) value is 6 for each dim
    # train_mean_xy = np.repeat(train_mean_xy[np.newaxis,:], 12, axis=0) 
        
    # output predictions and parameterized states
    pred_df  = pd.DataFrame(np.hstack([test_xy, pred_xy]), columns=['test_x', 'test_y', 'pred_x', 'pred_y'])
    param_df = pd.DataFrame(train_states_xy, columns=['cluster_x', 'cluster_y'], index=range(n_states))
    return pred_df, param_df

def predict_trajectory_from_state_clusters(emb_coords, 
                                           behav_coords, 
                                           n_clusters=8, 
                                           n_shifts=None,
                                           verbose=False):

    ''' combines ClusterClassifier with behavioral parameerization to predict state trajectories
        clusters are selected that maximizes overlap (adjusted mutual information) w/ behavior clusters
        uses leave-one-character-out cross-validation
    '''

    results = {}

    # leave one character out cross-validation
    for i, (train_ix, test_ix) in enumerate(character_cv()):

        if verbose: print(f'Character {i+1}/5', end='\r')

        # get train & test data
        train_emb, test_emb     = emb_coords[train_ix, :], emb_coords[test_ix, :]
        train_behav, test_behav = behav_coords.iloc[train_ix,:].values, behav_coords.iloc[test_ix,:].values
        assert train_emb.shape[0] == train_behav.shape[0], 'train embedding and behav must have same number of rows'
        assert test_emb.shape[0]  == test_behav.shape[0],  'test embedding and behav must have same number of rows'

        #--------------------------------------------------------------------------------
        # run full analysis
        #--------------------------------------------------------------------------------
        
        
        # run clustering maxing overlap of neural & behavior clusters
        cluster_clf = ClusterClassifier(n_clusters=n_clusters)
        cluster_clf.fit(train_emb, train_behav)
        train_states = cluster_clf.labels_

        # assign held out points to clusters w/ knn classifier
        test_states = cluster_clf.predict(test_emb) 

        # predict behavioral trajectory
        pred_df, param_states = predict_behavioral_trajectory(train_states, train_behav, test_states, test_behav)
        
        #--------------------------------------------------------------------------------
        # permutation testing to get null distribution of predictions
        # - circularly shifted embeddings
        #--------------------------------------------------------------------------------
        
        perm_xys, perm_param_states = [], []
 
        train_emb_shifts = circular_shifts(train_emb, axis=0)
        if n_shifts is None:
            n_shifts = len(train_emb_shifts)
        else:
            train_emb_shifts = random.sample(train_emb_shifts, n_shifts) # if want an upper limit to the number of shifts
        for p in tqdm(range(n_shifts)):
            if verbose: print(f'Character {i+1}/5: permutation {p+1}/{n_shifts}', end='\r')
            cluster_clf = ClusterClassifier(n_clusters=n_clusters)
            cluster_clf.fit(train_emb_shifts[p], train_behav) 
            perm_preds, perm_states = predict_behavioral_trajectory(cluster_clf.labels_, train_behav, 
                                                                    cluster_clf.predict(test_emb), test_behav)
            perm_xys.append(perm_preds[['pred_x', 'pred_y']])
            perm_param_states.append(perm_states)

        perm_param_states = np.mean(perm_states, axis=0)
        perm_xy = np.mean(perm_xys, axis=0)  
        pred_df['perm_x'] = perm_xy[:,0]
        pred_df['perm_y'] = perm_xy[:,1]
        
        results[f'char_0{i+1}'] = {'params': {'n_clusters': n_clusters, 'n_perms': n_shifts},
                                   'train_states': train_states, 'test_states': test_states,
                                   'train_states_xy': param_states, # parameterized states for training data
                                   'perm_states_xy': perm_param_states, # parameterized states for permuted data
                                   'predictions': pred_df}
    return results

def run_state_cluster_analysis(emb_fname,
                               rois=None, 
                               n_clusters=15, 
                               n_shifts=None,
                               overwrite=False,
                               verbose=False):

    # load data
    algo = emb_fname.split('/')[-1].split('_')[1].split('-')[0]
    sub_id = emb_fname.split('/')[-1].split('_')[0]
    emb_dict = pd.read_pickle(emb_fname)
    behav_coords = load_behavior(sub_id)[['affil_coord', 'power_coord']]

    # output directory & results
    out_dir = f"{('/').join(emb_fname.split('/')[:-2])}/state_clusters"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    if n_shifts is None:
        n_shifts = 47
    results_fname = f'{out_dir}/{sub_id}_{algo}-trajectory_predictions_{n_clusters}ul_{n_shifts}perms.pkl'
    results_dict = pd.read_pickle(results_fname) if os.path.exists(results_fname) else {}

    # define parameter grid
    rois = rois or ['L_HPC_thr25', 'R_HPC_thr25', 'L_ERC_thr25', 'R_ERC_thr25',
                    'L_mpfc_thr25', 'R_mpfc_thr25', 'L_M1-p_thr25', 'R_M1-p_thr25']
    rois = [r for r in rois if r in emb_dict.keys()]
    n_dims = ['5d', '6d', '7d']
    n_neighbors  = ['10nn', '20nn']
    if algo in ['pca', 'mds', 'tsne']:
        param_grid = list(itertools.product(rois, n_dims))
    else:
        param_grid = list(itertools.product(rois, n_dims, n_neighbors))
    if verbose: print(f'Running {len(param_grid)} parameter combinations')

    # loop over parameter grid
    for params in param_grid:
        roi, dims = params[0], params[1] # all have roi & n_components
        key_list = [roi, dims] if algo in ['pca', 'mds', 'tsne'] else [roi, dims, f'{params[2]}']
        exists, results_dict = create_nested_dict_keys(results_dict, key_list)
        if exists and not overwrite: continue
        if algo in ['pca', 'mds', 'tsne']:
            try: 
                if verbose: print(f'Running {roi}, {dims}')
                results_dict[roi][dims] = predict_trajectory_from_state_clusters(emb_dict[roi][dims]['embedding'], 
                                                                                  behav_coords, n_clusters=n_clusters, 
                                                                                  n_shifts=n_shifts, verbose=verbose)
            except KeyError:
                print(f'No embedding found for {roi}, {dims}')
                continue
        else:
            try:
                if verbose: print(f'Running {roi}, {dims}, {params[2]}')
                results_dict[roi][dims][params[2]] = predict_trajectory_from_state_clusters(emb_dict[roi][dims][params[2]]['embedding'], 
                                                                                            behav_coords, n_clusters=n_clusters, 
                                                                                            n_shifts=n_shifts, verbose=verbose)
            except KeyError:
                print(f'No embedding found for {roi}, {dims}, {params[2]}')
                continue

        pd.to_pickle(results_dict, results_fname)


#-----------------------------------------------------------------------------------------------------------
# spline decoding analysis
# - w. different randomization functions
#   - shuffle embedding order within character: break trajectory-ness
#   - random sequences: break character-ness
#   - simulate random trajectories: test if trajectory match is better than random choice based trajectory
#-----------------------------------------------------------------------------------------------------------


def break_sequenceness(trial_seq):
    # shuffle the trial_seq until there are no sequential numbers
    while True:
        np.random.shuffle(trial_seq)
        if np.all(np.diff(trial_seq) != 1) & np.all(np.diff(trial_seq) != -1):
            break
    return trial_seq
    
def random_task_sequences():

    # shuffle the decision trials into random sequences of trials
    # not fully random; constraints: 6 affil, 6 power; at least 2 trials from each character, 1 affil, 1 power; remaining 2 are from random & different characters

    # organize the characters into lists of affil and power
    trials_dict = {
        c: {'affil': np.random.permutation(decision_trials[(decision_trials['char_role_num'] == c) & 
                                                        (decision_trials['dimension'] == 'affil')]['decision_num'].values).tolist(),
            'power': np.random.permutation(decision_trials[(decision_trials['char_role_num'] == c) & 
                                                        (decision_trials['dimension'] == 'power')]['decision_num'].values).tolist()
        } for c in range(1, 6)
    }

    # assign first 10 trials in each random sequence
    random_dict = {
        s: [item for c in range(1, 6)  # 5 characters
            for item in [trials_dict[c]['affil'].pop(), trials_dict[c]['power'].pop()]]
        for s in range(1, 6)  # 5 sequences
    }

    # remaining trials on each dimension (ensure each character is split into dif sequences)
    affil = np.array([1, 2, 3, 4, 5])
    power = np.array([1, 2, 3, 4, 5])
    while np.any(affil == power):
        np.random.shuffle(affil)
        np.random.shuffle(power)
    for s in range(1,6):
        random_dict[s].append(trials_dict[affil[s-1]]['affil'].pop())
        random_dict[s].append(trials_dict[power[s-1]]['power'].pop())

    # get decision df for each random sequence
    return [sorted(d) for s, d in random_dict.items()] # return decision numbers for each sequence

def random_task_trajectories(behav, recalc_locs=True):
    # recalc_locs: if True, re-calculate locations from choices
    # if not true, then jsut use the real locations
    sequences = random_task_sequences() # get random sequences of decision numbers

    # compute trajectories with them
    trajectories = []
    for seq in sequences:
        behav_ = behav[behav['decision_num'].isin(seq)]
        if recalc_locs: # calculate new locations
            xy = np.cumsum(behav_[['affil_decision', 'power_decision']].values, axis=0)
        else: # use real locations
            xy = behav_[['affil_coord', 'power_coord']].values
        trajectories.append(pd.DataFrame(np.concatenate([np.array(seq)[:, np.newaxis], xy], axis=1),
                            columns=['decision_num', 'affil_coord', 'power_coord']))
    
    return add_dists(trajectories)

def simulate_trajectories():

    ''' generate random decisions witin the task structures of character and dimension '''

    decisions = np.zeros((63, 2))
    decisions[decision_trials['dimension']=='affil', 0] = np.array([np.random.choice([-1,1]) for _ in range(30)])
    decisions[decision_trials['dimension']=='power', 1] = np.array([np.random.choice([-1,1]) for _ in range(30)])

    # combine decisions with decision_trials[['char_role_num', 'char_decision_num', 'dimension']]
    decisions_df = pd.DataFrame(decisions, columns=['affil_decision', 'power_decision'])
    decisions_df = pd.concat([decision_trials[['decision_num','char_role_num', 'char_decision_num', 'dimension']], decisions_df], axis=1)

    # for each character, get the cumulative sum of affil and power
    decisions_df['affil_coord'] = decisions_df.groupby(['char_role_num'])['affil_decision'].cumsum()
    decisions_df['power_coord'] = decisions_df.groupby(['char_role_num'])['power_decision'].cumsum()
    trajectories = [decisions_df[['decision_num', 'affil_coord', 'power_coord']][decisions_df['char_role_num']==c] for c in range(1,6)]

    return add_dists(trajectories)

def add_dists(trajs):
    # helper for adding distances to randomized/simulated trajectory dfs
    for traj in trajs: 
        neu_dists = [norm(coords) for coords in traj[['affil_coord', 'power_coord']].values]
        pov_dists = [norm(coords-[6,0]) for coords in traj[['affil_coord', 'power_coord']].values]
        traj.insert(3, 'neu_2d_dist', neu_dists)
        traj.insert(4, 'pov_2d_dist', pov_dists)
    return trajs

def get_behav_params(params_name, behav):
    return (
        behav[['affil_coord', 'power_coord']].values if params_name == 'xy'
        else behav[params_name].values
    )

def scale_behav_params(behav):
    behav_scaled = behav.copy()
    for param in ['affil', 'power', 'affil_coord', 'power_coord', 'pov_2d_dist', 'neu_2d_dist']:
        if param not in behav_scaled.columns: continue
        behav_params = behav_scaled[param].values[:, np.newaxis]
        behav_scaled[param] = MinMaxScaler(feature_range=(0, 1)).fit_transform(behav_params).squeeze()
    return behav_scaled

class SplineDecoder:

    def __init__(self, 
                 s=0,
                 k=3,
                 w=None, 
                 t=None, 
                 extrapolate=False, 
                 periodic=False, 
                 sort=False,
                 regressor_f=None, **kwargs):

        #------------------------------------------------
        # spline fitting parameters
        #------------------------------------------------

        self.k = k # degree
        self.s = s # smoothing factor
            # Positive smoothing factor used to choose the number of knots. Number of 
            # knots will be increased until the smoothing condition is satisfied:
            # sum((w[i]*(y[i]-s(x[i])))**2,axis=0) <= s
        self.w = w # weights for weighted least-squares spline fit, length of the data; defaults to 1s        
        self.t = t # knots
        if t is None:
            self.task = 0 # find t & c for a given s 
        else:
            self.task = -1 # find weighted least sqr spline for t
            assert len(t) >= 2*k+2, \
                f'Not enough knots, need at least {2*k+2}'
        self.extrapolate = extrapolate
        self.periodic = periodic
        self.sort = sort 

        #------------------------------------------------
        # regression parameters
        #------------------------------------------------

        self.regressor_f = regressor_f or KNeighborsRegressor(n_neighbors=50)
        if kwargs:
            self.regressor_f = self.regressor_f(**kwargs)
        self.regressor_params = self.regressor_f.get_params()
        # self.__dict__['kwargs'] = kwargs
        
    def fit(self, X):
        ''' fit the spline object '''

        self.X = X
        n_points = self.X.shape[1]

        # if periodic spline, add a point at the end that is the same as the first
        if self.periodic:
            self.X = np.hstack((self.X, self.X[:, 0].reshape(-1, 1)))

        if self.sort: # sort points smallest to largest x, else leave as is
            self.X = self.X[:, np.argsort(self.X[0, :])]
        ndim = self.X.shape[0] # dimensionality of input space

        #-------------------------------------------------------------------------------------------
        # fit a parameterized cubic B-spline ("basis spline") to a set of data points
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html

        # B-spline: a piecewise polynomial function of degree k
        # - fitted w/ least squares: min. sum of squared residuals of spline approx. to data
        # - 'knots' connect the pieces, define 'control points' that control curves

        # inputs:
        # - points: array of data points (up to 10D)
        # - k : degree of the spline (3 = cubic)
        # - s : smoothing factor, balances fit against smoothness
        # - w : weights for weighted least-squares spline fit

        # returns:
        # - tck : tuple of knots, coefficients, and degree k of the spline
        # -- knots : points at which pieces of curve meet 
        # -- coefs : coefficients of the spline pieces 
        # - u : parameter values corresponding to x, y, [z]
        #-------------------------------------------------------------------------------------------

        tck, self.u_params = splprep(self.X, 
                                     s=self.s, 
                                     k=self.k, 
                                     w=self.w, 
                                     t=self.t,
                                     per=self.periodic,
                                     task=self.task,
                                     full_output=False)
        self.knots, self.coefs, _ = tck 

        #-------------------------------------------------------------------------------------------
        # knots: breakpoints
        #-------------------------------------------------------------------------------------------

        self.m = len(self.knots)
        if (self.s == 0) & (not self.periodic): # only true if s=0
            assert self.m == n_points + self.k + 1, \
                        f'm != len(x) + k + 1: {self.m} != {n_points + self.k + 1}'

        # evaluate the spline for the knots to get x,y,z coordinates of knots
        knots_coords = splev(self.knots, tck)
        self.knots_coords = np.vstack(knots_coords).T
        if (self.s == 0) & (not self.periodic):
            assert all(
                len(knots_coords[i]) == len(self.knots) == n_points + self.k + 1
                for i in range(ndim)
            ), f'knots != len(x) + k + 1: {len(knots_coords)} != {n_points + self.k + 1}'

        # internal & clamped knots (not sure if this is true for periodic too?)
        # - internal knots: not at the beginning or end of the spline
        # - clamped knots: of number k at the beginning or end of the spline
        self.knots_internal = self.knots_coords[self.k+1 : self.m-self.k-1]
        self.knots_clamped = (
            self.knots_coords[: self.k + 1],
            self.knots_coords[-self.k - 1 :],
        )

        #-------------------------------------------------------------------------------------------
        # coefficients for control points
        # - num of control points should equal number of coefficients
        #-------------------------------------------------------------------------------------------

        self.coefs = np.asarray(self.coefs)
        assert len(self.coefs) == ndim, \
                    f'coefs != ndim: {len(self.coefs)} != {ndim}'

        for i in range(len(self.coefs)):
            ni = len(self.coefs[i])
            assert ni == len(self.knots) - 1 - self.k, \
                        f'coefs != len(knots) - 1 - k: {ni} != {len(self.knots) - 1 - self.k}'
        self.n = len(self.coefs[0]) 

        #-------------------------------------------------------------------------------------------
        # define coordinates of control points in knot space
        # http://vadym-pasko.com/examples/spline-approx-scipy/example1.html
        #-------------------------------------------------------------------------------------------

        n = len(self.knots) - 1 - self.k
        self.control_points = [float(sum(self.knots[i+j] for j in range(1, self.k+1))) / self.k 
                                    for i in range(n)]
        
        #-------------------------------------------------------------------------------------------
        # check u parameter values
        #-------------------------------------------------------------------------------------------

        if not self.periodic:
            assert len(self.u_params) == n_points, f'u != len(x): {len(self.u_params)} != {n_points}'

        #-------------------------------------------------------------------------------------------
        # check p = degree of spline = m [len(knots)] - n [len(coefs)] - 1
        #-------------------------------------------------------------------------------------------

        assert self.m - self.n - 1 == self.k, f'p != k: {self.m - self.n - 1} != {self.k}'

        #-------------------------------------------------------------------------------------------
        # create BSpline object
        # - BSpline by default extrapolates the first and last polynomial pieces of B-spline functions active on base interval
        # - note: backwards compatability issue between splprep & BSpline means have to transpose coefs - https://github.com/scipy/scipy/issues/10389
        #-------------------------------------------------------------------------------------------

        self.spline_f = BSpline(self.knots, self.coefs.T, self.k, extrapolate=self.extrapolate, axis=0)
        check = np.sum(self.spline_f(self.u_params) - self.X.T) # should be ~equal to inputted points
        assert np.allclose(check, 0, atol=0.01), f'check != 0: {check} != 0'

        # return self.spline_f, self.u_params
            
    def transform(self, n=1000):
        # evaluate spline function: find a set of locations [0,1]
        # self.spline_f.__call__(x)
        self.spline = self.spline_f(np.linspace(0, 1, n)) # generate points in domain to evaluate the spline
        return self.spline

    def fit_transform(self, X, n=1000):
        # fit the spline object and evaluate the spline function
        self.fit(X)
        return self.transform(n=n)

    def inverse_transform(self):
        # TODO evaluate the inverse spline object
        return self

    def predict(self, X):
        # fit a simple reressor (eg KNN) & predict the location on the spline
        if X.ndim == 1: X = X[np.newaxis, :]
        self.regressor_f.fit(self.spline, np.arange(len(self.spline)))
        return self.regressor_f.predict(X)
        
    def fit_predict(self, X, n=1000):
        # fit the spline object and predict the location on the spline
        # check array
        
        self.fit_transform(X, n=n)
        return self.predict(X)
    
    def dummy_predict(self, X, strategy='mean'):
        # fit a dummy regressor
        if X.ndim == 1: X = X[np.newaxis, :]
        dummy_reg = DummyRegressor(strategy=strategy)
        dummy_reg.fit(self.spline, np.arange(len(self.spline)))
        return dummy_reg.predict(X)

    def distance_to_origin(self):

        #-------------------------------------------------------------------------------------------
        # distance between points along curve/trajectory
        # TODO: integrate between all points in points?
        # approximation: sum distances between all consecutive points
        #-------------------------------------------------------------------------------------------
        
        X = np.vstack(X).T
        return np.cumsum([np.linalg.norm(X[i] - X[i+1]) for i in range(len(X)-1)])
    
class DistanceRegressor(BaseEstimator, RegressorMixin):
    """
        A regressor that uses smallest distance between points in a given metric to predict value of the target variable
    """

    def __init__(self, metric='euclidean'):
        self.metric = metric

    def fit(self, X, y):
        self.X = X
        self.y = y # just for compatibility with sklearn
        
    def predict(self, x):
        dists = pairwise_distances(self.X, x, metric=self.metric)
        return [np.argmin(dists)]

class SplineParameterInference:

    # infer parameter value of location on the spline

    def __init__(self, spline_obj, params, n=1000):
        self.spline_obj = spline_obj # spline object
        self.params = params # parameter array
        self.n = n # number of points used to evaluate the spline

    def fit(self):
        # parametrize spline locations: spline is defined on [0, n]
        self.u_locs = self.spline_obj.u_params * self.n 
        assert np.all(np.diff(self.u_locs) > 0), 'u_locs not ascending'
    
    def predict(self, loc):
        # infer parameter value for predicted location (in terms of knots)
        # TODO check tat it can accomodate mutliple X
        
        # check inputs etc
        if loc.ndim == 0: loc = np.array([loc])[np.newaxis]
        loc = check_array(loc)

        # calculate % of distance between knots for the predicted location
        pred_ix = np.searchsorted(self.u_locs, loc) # where would this location fall between u_locs?
        knot_dist = self.u_locs[pred_ix] - self.u_locs[pred_ix-1] # distance between knots before & after predicted location
        pred_change = loc - self.u_locs[pred_ix-1] # distance between predicted location & knot right before it in trajetory
        pred_perc_change = pred_change / knot_dist # as % change 

        # convert % change to parameter change
        pred_param_change = pred_perc_change * (self.params[pred_ix] - self.params[pred_ix-1])
        pred_param = self.params[pred_ix-1] + pred_param_change # add the change in parameter value to last parameter value
        return pred_ix[0][0], pred_param[0][0] # TODO - fix why its nested
    
    def fit_predict(self, loc):
        self.fit()
        return self.predict(loc)

def spline_analysis(char_embs, 
                    char_trajs, 
                    k=1, 
                    s=0,
                    train_shift=None):

    ''' for each character, use leave one out C.V. to fit a spline & decode location 
        pass in train_ixs if there is a specific way to train the spline (ie, circleshift)
    
    '''

    # hold results
    fitted_dict = {}
    decoded_df  = pd.DataFrame(columns=['decision_num', 'character', 'familiarity', 'pred_loc', 'dummy_loc'])
    paramz_df   = pd.DataFrame(columns=['param_name', 'decision_num', 'character', 'familiarity', 'param',
                                        'pred_familiarity', 'pred_loc', 'pred_param', 
                                        'dummy_familiarity', 'dummy_loc', 'dummy_param'])

    # loop over each character
    for char_num in tqdm(range(1, 6)):

        fitted_dict[f'char_{char_num}'] = {}

        # get character's data
        char_traj = char_trajs[char_num-1]
        char_emb  = char_embs[char_num-1]
        char_ixs  = decision_trials[decision_trials['char_role_num'] == char_num]['decision_num'].values-1 # ixs of trials
        
        # loop over each trial as test trial
        for test_ix in range(12):

            # get test & training data
            char_emb_test  = char_emb[test_ix, :]
            train_ixs      = np.delete(np.arange(12), test_ix)
            if train_shift is not None: # circular shift the data
                train_ixs = np.roll(train_ixs, train_shift, axis=0)
            char_emb_train = char_emb[train_ixs, :]

            # fit spline & decode held-out location
            spl_decoder = SplineDecoder(s=s, k=k, sort=False)
            spl_decoder.fit_transform(char_emb_train.T)
            pred_loc  = spl_decoder.predict(char_emb_test.T)[0] # predicted location
            dummy_loc = spl_decoder.dummy_predict(char_emb_test.T)[0]

            fitted_dict[f'char_{char_num}'][f'trial_{test_ix+1}'] = {'X_train': char_emb_train, 'train_ixs': char_ixs[train_ixs], # ixs are in terms of the actual trials
                                                                     'X_test': char_emb_test, 'test_ixs': char_ixs[test_ix], 
                                                                     'regressor_params': spl_decoder.regressor_params}
            decoded_df.loc[len(decoded_df)] = [char_ixs[test_ix], char_num, test_ix+1, pred_loc, dummy_loc]

            # parameterize spline & decode held-out location's parameter value
            for param_name in ['xy']: # 'pov_2d_angle', 'neu_2d_angle', 'pov_2d_dist', 'neu_2d_dist'

                # organize behavioral parameter values
                char_params = get_behav_params(param_name, char_traj)
                train_params, test_param = char_params[np.delete(np.arange(12), test_ix)], char_params[test_ix] # split parameter values into train & test

                # decode held out trials location 
                pred_ix, pred_param   = SplineParameterInference(spl_decoder, train_params).fit_predict(pred_loc)
                dummy_ix, dummy_param = SplineParameterInference(spl_decoder, train_params).fit_predict(dummy_loc) # dummy model: predicts middle

                paramz_df.loc[len(paramz_df)] = [param_name, char_ixs[test_ix], char_num, test_ix+1, test_param,
                                                 pred_ix+1, pred_loc, pred_param, 
                                                 dummy_ix+1, dummy_loc, dummy_param]
            
    return fitted_dict, decoded_df, paramz_df

def run_spline_analysis(emb_fname,
                        rois=None, 
                        s=0, 
                        k=1, 
                        traj='real', 
                        scale='within',
                        n_shuffles=None,
                        overwrite=False,
                        verbose=False):

    ''' run full character-specific spline analysis for each roi & output in a pickle '''

    #-------------------------------------------------------------------------------------------
    # load data
    #-------------------------------------------------------------------------------------------

    # load previous results if exist
    fname_split = emb_fname.split('/')
    sub_id, algo = fname_split[-1].split('_')[0], fname_split[-1].split('_')[1].split('-')[0]
    k_name = 'linear' if k == 1 else 'cubic'
    fname_pre = f'{sub_id}_{algo}_{k_name}'
    if verbose: print(f'Running {k_name} spline analysis for {sub_id}')

    if algo == 'pca':
        out_dir = f"{('/').join(fname_split[:-2])}/splines_scaled-{scale}/pca"
    else:
        out_dir = f"{('/').join(fname_split[:-2])}/splines_scaled-{scale}/{traj}"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    fnames = {'parameterized': f'{out_dir}/{fname_pre}-spline_{traj}.pkl'}
    if n_shuffles is not None: fnames['parameterized'] = f'{out_dir}/{fname_pre}-spline_{traj}_{n_shuffles}shuffles.pkl'
    paramz_dict = pd.read_pickle(fnames['parameterized']) if os.path.exists(fnames['parameterized']) else {}

    # get subject data
    emb_dict = pd.read_pickle(emb_fname)
    behavior = load_behavior(sub_id)
    if verbose: print(f'Loaded embeddings & behavior for {sub_id}')

    # define ROIs to run analysis in
    rois = rois or list(emb_dict.keys())
    rois = [r for r in rois if r in emb_dict.keys()]
    if verbose: print(f'ROIs to fit splines in: {rois}')

    #-------------------------------------------------------------------------------------------
    # run the analysis
    #-------------------------------------------------------------------------------------------

    # for schemes that use real behavior
    if traj == 'real':
        # real: use real social locations (main analysis)
        char_trajs = organize_by_character(behavior)
        n_shuffles = 1
    elif traj == 'circleshift':
        # circleshift: for each fold, circle shift trials 
        # a randomization that respects temporal autocorrelation
        # randomization happens inside training loop
        # basic structure:
            # run 10 circle shifted spline analyses
            # for shift in shifts: 
                # for character in characters:
                    # for test_trial in trials:
                        # other trials are training trials: circle shift them
                        # fit spline to training trials
                        # predict spline location of test trial
                        # predict parameter given location
                        # store results
        char_trajs = organize_by_character(behavior)
        n_shuffles = 10
    elif traj == 'other-characters':
        char_trajs_unshuffled = organize_by_character(behavior)
        char_ixs_random = [(2,3,4,5,1), (3,4,5,1,2), (4,5,1,2,3), (5,1,2,3,4)]  
        n_shuffles = len(char_ixs_random)
    else: # for all others, just shuffle in some way
        if n_shuffles is None:
            n_shuffles = 50
            
    # scale the behavioral parameters between 0 & 1 for methods with same behavior across shuffles
    if traj in ['real', 'circleshift']:
        if scale == 'within':
            char_trajs = [scale_behav_params(char_traj) for char_traj in char_trajs]
        elif scale == 'across':
            char_trajs = np.array_split(scale_behav_params(pd.concat(char_trajs)), 5)

    for roi in rois:
        
        # run in available embeddings
        dims = list(emb_dict[roi].keys())
        if algo in ['pca', 'mds', 'tsne']: 
            param_grid = dims
        else:
            nns = ['10nn','20nn']
            param_grid = list(itertools.product(dims, nns))
        
        for params in param_grid:

            # checks if already been run
            key_list = [roi, params] if algo in ['pca', 'mds', 'tsne'] else [roi, *params]
            exists, paramz_dict = create_nested_dict_keys(paramz_dict, key_list) 
            if (exists) & (not overwrite): continue

            # load roi embedding
            try: 
                if algo in ['pca', 'mds', 'tsne']:
                    embedding = emb_dict[roi][params]['embedding']
                else:
                    embedding = emb_dict[roi][params[0]][params[1]]['embedding']
            except: 
                print(f'Embedding not found: {roi} {params}')
                continue
            char_embs = organize_by_character(embedding)
            assert behavior.shape[0] == embedding.shape[0], 'Mismatched shapes'

            # loop over shuffle count
            paramz_dfs  = []
            train_shift = None # dont shift the training indices in any way
            for shuff in range(n_shuffles):

                if verbose: print(f'Running: {traj} {roi} {params} - {shuff+1}/{n_shuffles}', end='\r')

                # diff. randomization schemes
                if traj == 'circleshift':
                    train_shift = shuff+1

                elif traj == 'other-characters':
                    # shuffle characters
                    shuff_ixs  = char_ixs_random[shuff]
                    char_trajs = [char_trajs_unshuffled[char_ixs_random[shuff][c]-1] for c in range(5)]
                    # char_trajs = char_trajs_unshuffled[char_ixs_random[shuff]]

                elif traj == 'shuffle-choices':
                    # shuffle-choices: shuffle choices within each character
                    # a very conservative null: maintains everything - including the end location - except for the precise path taken
                    char_choices = organize_by_character(load_choices(sub_id)[['affil', 'power']])
                    char_choices = [char_choices[c].sample(frac=1) for c in range(5)] # shuffle each set of char_choices
                    char_trajs   = [np.cumsum(char_choices[c], axis=0) for c in range(5)] # cumulative sum to turn into trajectories
                    char_trajs   = [char_traj.rename(columns={'affil': 'affil_coord', 'power': 'power_coord'}) for char_traj in char_trajs]

                elif traj == 'shuffle-locations':
                    # shuffle character's sequence of trials for spline fitting
                    # ie, the relationship between neural state & social location is preserved, but break the spline transitions
                    char_trajs = [behavior[behavior['char_role_num'] == c].iloc[break_sequenceness(np.arange(0, 12)),:].reset_index(drop=True) for c in range(1,6)]

                elif traj == 'pseudo-choices':
                    # use 12 random trials for each trajectory (maintaining temporal seq. & dimension, etc...)
                    # calculate social locations from choices
                    char_trajs = random_task_trajectories(behavior, recalc_locs=True)

                elif traj == 'pseudo-locations':
                    # use 12 random trials for each trajectory (maintaining temporal seq. & dimension, etc...)
                    # use 'real' social locations
                    char_trajs = random_task_trajectories(behavior, recalc_locs=False)

                elif traj == 'random-sim':
                    # simulated trajectories using random choosing
                    # a conservative null: maintains everything except the choices
                    char_trajs = simulate_trajectories()

                # scale the behavioral parameters between 0 & 1 for methods that re-compute behavior on each shuffle
                if traj not in ['real', 'circleshift']:
                    
                    if scale == 'within':
                        char_trajs = [scale_behav_params(char_traj) for char_traj in char_trajs]
                    elif scale == 'across':
                        char_trajs = np.array_split(scale_behav_params(pd.concat(char_trajs)), 5)

                # run analysis
                _, _, paramz_df = spline_analysis(char_embs, char_trajs, k=k, s=s, train_shift=train_shift)
                if traj != 'real': paramz_df['perm'] = shuff
                paramz_dfs.append(paramz_df)
            paramz_df = pd.concat(paramz_dfs)
            assign_nested_dict_value(paramz_dict, key_list, paramz_df)
            pd.to_pickle(paramz_dict, fnames['parameterized'])


#-----------------------------------------------------------------------------------------------------------
# standardize the plotting
#-----------------------------------------------------------------------------------------------------------

tab10_colors = sns.color_palette("tab10").as_hex()
general_pal = [tab10_colors[0], tab10_colors[2], tab10_colors[4], tab10_colors[5], tab10_colors[7]]
roi_pal, distance_pal = sns.color_palette("Paired"), sns.color_palette("Purples")

alphas = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001] # for stars
tick_fontsize, label_fontsize, title_fontsize = 10, 13, 15
legend_title_fontsize, legend_label_fontsize = 12, 10
legend_title_fontsize = label_fontsize-1
legend_fontsize = tick_fontsize-.5
suptitle_fontsize = title_fontsize * 1.5
ec, lw = 'black', 1
bw = 0.15 # this is a proportion of the total??
figsize, facet_figsize = (5, 5), (5, 7.5)

def add_pval(num, ax, fontsize=25):
    ax.annotate('*'*num, xy=(.5, .925), xycoords='axes fraction', fontsize=fontsize, ha='center', va='center')
    
def add_sample_legend(ax, loc='upper left', 
                      title_fontsize=legend_title_fontsize, 
                      label_fontsize=legend_label_fontsize, 
                      **kwargs):
    handles = [mpatches.Patch(color=sample_dict[sample]['color'], label=sample) for sample in sample_dict.keys()]
    ax.legend(handles=handles, loc=loc, title='Sample', frameon=False,
              title_fontsize=title_fontsize, fontsize=label_fontsize, **kwargs)
    return ax

# more specific plotting functions
def plot_coords(coords, polar_coords, kde=True, \
                colors=None, annot=False, connect=False, jitter=False):

    fig, axs = plt.subplots(1,2, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 2]})

    #------------------------------------------------------------
    # plot cartesian + circular fit

    ax = axs[0]
    plot_circle_fit(ax, coords, colors=colors, connect=connect, jitter=jitter)
    if kde: 
         sns.kdeplot(ax=ax, x=coords[:, 0], y=coords[:, 1], fill=True, thresh=0, levels=100, cmap='Blues', zorder=-1)
    if annot:
            # add text for each quadrant in the plot
            ax.text(0.975, 0.975, 'Q1 [friendly, dominant]', 
                    verticalalignment='top', horizontalalignment='right',
                    transform=ax.transAxes, color='black', fontsize=8)
            ax.text(0.025, 0.975,  'Q2 [unfriendly, dominant]', 
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes, color='black', fontsize=8)
            ax.text(0.025, 0.025, 'Q3 [unfriendly, submissive]', 
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes, color='black', fontsize=8)
            ax.text(0.975, 0.025, 'Q4 [friendly, submissive]', 
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes, color='black', fontsize=8)
    ax.set_aspect('equal')

    #------------------------------------------------------------
    # plot a phasor representation
    ax = axs[1]

    # generate phasor representations
    phasor_lead  = [magnitude_phasor(crd[0], np.rad2deg(crd[1]), 0) for crd in polar_coords]
    phasor_trail = [magnitude_phasor(crd[0], np.rad2deg(crd[1]), 90) for crd in polar_coords]
    degs         = np.rad2deg(polar_coords[:, 1])

    # plot the waves
    if jitter:
        jittered_degs  = add_jitter(degs, jitter=0.001)
        jittered_lead  = add_jitter(phasor_lead, jitter=0.05)
        jittered_trail = add_jitter(phasor_trail, jitter=0.05)
        ax.scatter(jittered_degs, jittered_trail, 
                    s=30, c=colors, edgecolors='k', linewidths=0.5)
    else:
        ax.scatter(degs, phasor_trail, 
                    s=30, c=colors, edgecolors='k', linewidths=0.5)
    
    # add spline fit
    # degs, phasor_trail = zip(*sorted(zip(degs, phasor_trail)))
    # spline = fit_spline(degs, phasor_trail, s=1, k=3)
    # ax.plot(degs, spline(degs), 'k', lw=3, zorder=-1)

    # make pretty
    ax.set_xlim(-8, 360)
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    m = max(np.abs(ax.get_ylim()))
    ax.set_ylim(-m, m)

    # add vertcal lnes at 90,180,270
    for deg in [90, 180, 270]:
            ax.axvline(deg, c='k', linewidth=0.5, zorder=-1)
            
    if annot:
        # add quadrant text between the lines
        ax.text(45/360, 0.975, 'Q1 [friendly, dominant]',
                verticalalignment='top', horizontalalignment='center',
                transform=ax.transAxes, color='black', fontsize=8)
        ax.text(135/360, 0.975, 'Q2 [unfriendly, dominant]',
                verticalalignment='top', horizontalalignment='center',
                transform=ax.transAxes, color='black', fontsize=8)
        ax.text(225/360, 0.975, 'Q3 [unfriendly, submissive]',
                verticalalignment='top', horizontalalignment='center',
                transform=ax.transAxes, color='black', fontsize=8)
        ax.text(315/360, 0.975, 'Q4 [friendly, submissive]',
                    verticalalignment='top', horizontalalignment='center',
                    transform=ax.transAxes, color='black', fontsize=8)

    ax.axhline(0, c='k', linewidth=.5, zorder=-1)
    ax.set_xlabel('Phase (angle [deg])')
    ax.set_ylabel('Magnitude * sine(phase)')

    # if colors is not None:
    #     for color in colors:
    #         mask = colors == color
    #         ax.plot(jittered_degs[mask], jittered_lead[mask], color=color, linewidth=.5, alpha=.5)
            # ax.plot(jittered_degs[mask], jittered_trail[mask], color=color, linewidth=.5, alpha=.5)

    plt.show()

def plot_embedding_circle(emb, ax):

    trial_radius = (np.arange(1, len(emb)+1) / 7) ** 2
    if len(emb) == 60: colors = remove_neutrals(character_colors_trials)
    else:              colors = character_colors_trials
    sns.kdeplot(ax=ax, x=emb[:, 0], y=emb[:, 1], 
                fill=True, thresh=0, levels=100, cmap='Blues')
    ax.scatter(emb[:,0], emb[:,1], s=trial_radius, c=colors,      
               alpha=0.75, edgecolor='k', linewidth=0.5, zorder=2)

def plot_parameterization(ax, params, n=1000, 
                         step=5, cmap='Reds', lw=3, zorder=-1):
    params = (((params - params.min()) / (params.max() - params.min())) * (n-1)).astype(int)  # rescale from 0 to n-1
    colors = [plt.get_cmap(cmap)(float(ii) / (n)) for ii in range(n)]
    for i in range(0, n-1, step):
        segment = spl[i : i+step+1]
        params_ = int(np.mean(params[i : i+step+1]))
        ax.plot(segment[:,0], segment[:,1], c=colors[params_], lw=lw, zorder=zorder)

def plot_embedded_decision_trajectory(emb_data, scatter=True, 
                                      color_character=False, 
                                      view_init=(30, 45),
                                      ax=None, title='', figsize=(5, 5), 
                                      alpha=0.6, linewidth=0.5):

    # should only be decision trials
    n_tps = emb_data.shape[0]
    n_decision_tps = n_tps / 63
    character_labels_ = np.repeat(character_labels, n_decision_tps)

    if ax is None: 
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(*view_init)
    ax.set_title(title, fontsize=16)

    # loop through characters
    for i in range(5):

        c = character_colors[i] if color_character else 'k'
        char_ixs = np.where(character_labels_ == i+1)[0]
       
       # plot trajectory of each trial
        for ixs in np.split(char_ixs, n_decision_tps):
            if scatter: ax.scatter3D(*emb_data[ixs, :].T, c=c, alpha=min(1, alpha*2), s=10)
            ax.plot3D(*emb_data[ixs, :].T, c=c, alpha=alpha, linewidth=linewidth)

    if ax is None: 
        return fig
    
def plot_trajectory_with_plane(emb, title, ax):
    _ = plot_embedded_decision_trajectory(emb, title=title, ax=ax)
    X, Y, Z = fit_plane(emb, order=1)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2, color='grey')
    X, Y, Z = fit_plane(emb, order=2)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2, color='red')

