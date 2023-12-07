import os, sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nilearn

#-------------------------------------------------------------------------------------------
# my modules
#-------------------------------------------------------------------------------------------

user = os.path.expanduser('~')
# if user == '/hpc/users/schafm03':
#     base_dir = '/sc/arion/projects/OlfMem/mgs/2D_place'
#     sys.path.insert(0, f'{base_dir}/Code/toolbox') 
#     sys.path.insert(0, f'{base_dir}/Code/social_navigation_analysis/social_navigation_analysis') 
# else:
#     if user == '/Users/matthew':
#         base_dir = '/Volumes/synapse/projects/SocialSpace/Projects/SNT-fmri_place/New_analyses' 
#     elif user == '/Users/matty_gee':
#         base_dir = '/Users/matty_gee/Desktop/SNT-fMRI'
#     sys.path.insert(0, f'{user}/Dropbox/Projects/toolbox/toolbox')
#     sys.path.insert(0, f'{user}/Dropbox/Projects/fmri_tools/qc')
#     sys.path.insert(0, f'{user}/Dropbox/Projects/social_navigation_analysis/social_navigation_analysis')
from info import decision_trials, character_roles, task


#-------------------------------------------------------------------------------------------
# load data etc
#-------------------------------------------------------------------------------------------


# add an argument to only get certain ROIs, to speed up loading
def load_ts_mat(mat_file, preprocess=True, decisions_only=False, **kwargs):
    """
        Load timseries mat file and return a nested dictionary
        Options to preprocess and restrict to decision period
    """
    data = loadmat(mat_file)['data'][0]
    ts_dict = {}
    for r in range(len(data)):
        roi_name, roi_ts = data[r][0][0], data[r][2]
        if preprocess:
            roi_ts = clean_ts(roi_ts, **kwargs)
        if decisions_only:
            roi_ts = get_decision_ts(roi_ts)
        ts_dict[roi_name] = roi_ts
    return ts_dict

def digitize_matrix(matrix, n_bins=10): 
    '''
        Digitize an input matrix to n bins (10 bins by default)
        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    matrix_bins = [np.percentile(np.ravel(matrix), 100/n_bins * i) for i in range(n_bins)] # compute the bins 
    matrix_vec_digitized = np.digitize(np.ravel(matrix), bins = matrix_bins) * (100 // n_bins) # compute the vector digitized value 
    matrix_digitized = np.reshape(matrix_vec_digitized, np.shape(matrix)) # reshape to matrix
    matrix_digitized = (matrix_digitized + matrix_digitized.T) / 2  # force symmetry in the plot
    return matrix_digitized


#-------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------


def get_task_trs(tr=1.0):

    # double & triple check timing....
    # new data: ... TRs, older data: ... TRs
    
    if tr == 1.0:
        onset, offset = 'cogent_onset', 'cogent_offset'
    elif tr == 2.0:
        onset, offset = 'cogent_onset_2015', 'cogent_offset_2015'

    other_rows = ['trial_type', 'slide_num', 'scene_num', 'dimension', 'char_role_num', 'char_decision_num']
    out = []
    for _, row in task.iterrows():
        on, off = np.round(row[onset]), np.round(row[offset])
        sec_ix = np.arange(on, off, tr)[:, np.newaxis] # range of indices
        other  = np.vstack([np.repeat(r, len(sec_ix)) for r in row[other_rows]]).T
        out.append(np.hstack([sec_ix, other]))

    out_df = pd.DataFrame(np.vstack(out), columns=['onset(s)'] + other_rows)
    out_df['onset(s)'] = out_df['onset(s)'].astype(float).astype(int)
    out_df.insert(0, 'TR', out_df.index + 1)

    # add rows if needed
    total_secs = out_df['onset(s)'].values[-1] # should be 1570
    missing_secs = 1570 - total_secs

    # add rows to tr_df
    extra_trs_df = pd.DataFrame(np.arange(total_secs+1, 1570), columns=['onset(s)'])
    extra_trs_df['TR'] = np.arange(total_secs+1, 1570) / tr
    extra_trs_df[other_rows] = np.nan

    out_df = pd.concat([out_df, extra_trs_df], axis=0)
    return out_df.reset_index(drop=True)


#-------------------------------------------------------------------------------------------
# timeseries cleaning
#-------------------------------------------------------------------------------------------


def clean_ts(ts, tr=1.0, detrend=True, 
             standardize='zscore', filter='cosine',
             low_pass=None, high_pass=1/128):
    # default: 
    # 1 - detrend to remove linear trends
    # 2 - high_pass filter to filter out low freq. signals (default is 1/128s = 0.0078125Hz)
    # 3 - standardize
    
    return nilearn.signal.clean(ts,
                                t_r=tr,
                                detrend=detrend,
                                standardize=standardize,
                                filter=filter,
                                low_pass=low_pass,
                                high_pass=high_pass)

def get_decision_ts(ts, pre_trs=0): 
    """
        Returns trial-ordered decision periods for timeseries

        tc : timeseries (num_trs x num_voxels)
        onsets : 'new' or 'old' (new is the 2019 onsets, old is the 2015 onsets)
        tr : TR in seconds
        pre_trs : number of TRs before the onset to include in the epoch
    """
    if ts.shape[0] == 1570:   
        col_name, tr = 'cogent_onset', 1.0
    # elif ts.shape[0] == 784: 
    else:
        col_name, tr = 'cogent_onset_2015', 2.0
    # else:
    #     raise Exception(f'Make sure timeseries has correct shape: {ts.shape}')

    onset_secs = decision_trials[col_name].values # in seconds
    onset_trs  = (np.round(onset_secs) / tr).astype(int) - pre_trs # convert to TRs, subtract to get pre-onset TRs
    decision_windows = np.vstack([np.arange(on_tr, on_tr + (12/tr) + pre_trs) for on_tr in onset_trs]) # num_trials x num_trs
    assert decision_windows.shape == (len(onset_trs), (12/tr) + pre_trs), f'Epochs shape is {decision_windows.shape}'
    return np.vstack([ts[int(w[0]) : int(w[-1])+1, :] for w in decision_windows])

def preprocess_ts(ts, tr=1.0, detrend=True,
                  standardize='zscore', filter='cosine',
                  low_pass=None, high_pass=1/128, onsets='new'):
    # clean and return only dcision periods
    cleaned_ts = clean_ts(ts, tr=tr, detrend=detrend,
                          standardize=standardize, filter=filter,
                          low_pass=low_pass, high_pass=high_pass)
    cleaned_ts = np.vstack(get_decision_ts(cleaned_ts, onsets=onsets))
    return cleaned_ts

def maskout_pretrial_trs(ts):

    # mask out the extra TRs included prior to the actual decision period volumes
    extra_trs = (ts.shape[0] / 63) - 12 # TRs include per trial (may be pre-decision TRs in it)
    trial_mask = np.concatenate([np.zeros(int(extra_trs), dtype=bool), np.ones(12, dtype=bool)])
    tc_mask = np.repeat(trial_mask, 63)
    assert tc_mask.shape[0] == ts.shape[0]
    ts_masked = ts[tc_mask, :]
    assert ts_masked.shape == (63*12, ts.shape[1])
    return ts_masked

def get_roi_decision_ts(roi, ts_dict):

    ts = ts_dict[roi]
    ts = clean_ts(ts, tr=1.0)
    ts = np.vstack(get_decision_ts(ts, onsets='new', tr=1.0))

    return ts


#-------------------------------------------------------------------------------------------
# plotting 
#-------------------------------------------------------------------------------------------


def plot_ts(ts, figsize=(16, 8)):

    f, axs = plt.subplots(1,2, figsize=figsize, gridspec_kw={'width_ratios': [1, .5]})
    ax = axs[0]
    ax.imshow(ts.T, interpolation='nearest', cmap='magma_r', aspect='auto')
    ax.set_title('ROI timseries', fontsize=15)
    ax.set_ylabel('Voxels/Components', fontsize=12)
    ax.set_xlabel('Timepoints', fontsize=12)

    ax = axs[1]
    ax.imshow(np.corrcoef(ts), cmap='magma_r')
    # add a box around every decision period
    # for i in range(0, ts.shape[0], 12):
    #     rect = patches.Rectangle((i,i), 12, 12, 
    #                              linewidth=2, edgecolor='white', facecolor='none')
    #     ax.add_patch(rect)
    ax.set_title('TR-TR correlation matrix', fontsize=15)
    ax.set_xlabel('TR', fontsize=12)
    ax.set_ylabel('TR', fontsize=12)

    return f

def plot_ts_corrmat(ts, digitize=False, 
                    ax=None, cmap='magma_r', figsize=(10,10)):
    """
        Plot the correlation matrix of the timeseries
    """
    if ax is None:
        f, ax = plt.subplots(1,1, figsize=figsize)
    cm = np.corrcoef(ts)
    if digitize: cm = digitize_matrix(cm)
    ax.imshow(np.corrcoef(ts), cmap=cmap)
    ax.set_title('TR-TR correlation matrix', fontsize=15)
    ax.set_xlabel('TR', fontsize=12)
    ax.set_ylabel('TR', fontsize=12)
    return ax

def plot_ts_similarity_matrix(ax, ts, bounds, 
                              digitize=False, lw=2, cmap='magma_r'):

    # plot a TR-TR correlation matrix, with bounds highlighted
    # tc should be n_TRs x n_voxels
    # bounds is boudnaries to highligh tin TRs
    n_TRs = ts.shape[0]
    cm = np.corrcoef(ts)
    if digitize: cm = digitize_matrix(cm)
    ax.imshow(cm, cmap=cmap)
    ax.set_xlabel('TR')
    ax.set_ylabel('TR')

    # plot the boundaries 
    bounds_aug = np.concatenate(([0], bounds, [n_TRs]))
    for i in range(len(bounds_aug)-1):
        rect = patches.Rectangle(
            (bounds_aug[i],bounds_aug[i]),
            bounds_aug[i+1]-bounds_aug[i],
            bounds_aug[i+1]-bounds_aug[i],
            linewidth=lw, edgecolor='w', facecolor='none'
        )
        ax.add_patch(rect)