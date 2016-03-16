import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from shapely.geometry import Point, LineString
from scipy import signal


# Loading functions from *.mat files
def load_csc(matfile):
    loading_csc = sio.loadmat(matfile)
    csc = dict(time=[])
    csc['data'] = loading_csc['csc_data'][0]
    for val in range(len(loading_csc['csc_tvec'])):
        csc['time'].append(loading_csc['csc_tvec'][val][0])
    csc['type'] = loading_csc['csc_type'][0]
    csc['label'] = loading_csc['csc_label'][0][0][0]
    return csc


def load_position(matfile):
    loading_pos = sio.loadmat(matfile)
    pos = dict()
    pos['x'] = loading_pos['pos_datax'][0]
    pos['y'] = loading_pos['pos_datay'][0]
    pos['time'] = loading_pos['pos_tvec'][0]
    pos['type'] = loading_pos['pos_type'][0]
    pos['label'] = loading_pos['pos_label'][0][0][0]
    return pos

# This data had issues with the feeder lights contaminating the position tracking, so those contaminating
# signals were removed.

def load_videotrack(matfile):
    loading_vt = sio.loadmat(matfile)
    vt = dict()
    vt['time'] = loading_vt['pos_tsd'][0][0][1][0]
    vt['x'] = loading_vt['pos_tsd'][0][0][1][0]
    vt['y'] = loading_vt['pos_tsd'][0][0][1][1]
    # vt['x'] = loading_vt['pos_tsd'][0][0][2][0]
    # vt['y'] = loading_vt['pos_tsd'][0][0][2][1]

    nan_idx = np.isnan(vt['x']) | np.isnan(vt['y'])
    vt['time'] = vt['time'][~nan_idx]
    vt['x'] = vt['x'][~nan_idx]
    vt['y'] = vt['y'][~nan_idx]
    return vt


def load_events(matfile):
    loading_events = sio.loadmat(matfile)
    events = dict()
    events['led1'] = loading_events['evt_led1id'][0]
    events['led2'] = loading_events['evt_led2id'][0]
    events['ledoff'] = loading_events['evt_ledoff'][0]
    events['pb1'] = loading_events['evt_pb1id'][0]
    events['pb2'] = loading_events['evt_pb2id'][0]
    events['pboff'] = loading_events['evt_pboff'][0]
    events['feeder1'] = loading_events['evt_feeder1id'][0]
    events['feeder2'] = loading_events['evt_feeder2id'][0]
    events['feederoff'] = loading_events['evt_feederoff'][0]
    events['type'] = loading_events['evt_type'][0]
    events['label'] = loading_events['evt_label'][0][0][0]
    return events


def load_spikes(matfile):
    loading_spikes = sio.loadmat(matfile)
    spikes = dict()
    spikes['time'] = loading_spikes['spikes_times'][0]
    spikes['type'] = loading_spikes['spikes_type'][0]
    spikes['label'] = loading_spikes['spikes_label'][0][0][0]
    return spikes


# Some useful functions
def find_nearest_idx(array, val):
    return (np.abs(array-val)).argmin()


def time_slice(spikes, t_start, t_stop):
    if t_start is None:
        t_start = -np.inf
    if t_stop is None:
        t_stop = np.inf
    indices = (spikes >= t_start) & (spikes <= t_stop)
    sliced_spikes = spikes[indices]
    return sliced_spikes


def convert_to_cm(path_pts, xy_conversion):
    for key in path_pts:
        path_pts[key][0] = path_pts[key][0] / xy_conversion[0]
        path_pts[key][1] = path_pts[key][1] / xy_conversion[1]
    return path_pts


def linear_trajectory(pos, ideal_path, trial_start, trial_stop):
    t_start_idx = find_nearest_idx(np.array(pos['time']), trial_start)
    t_end_idx = find_nearest_idx(np.array(pos['time']), trial_stop)

    pos_trial = dict()
    pos_trial['x'] = pos['x'][t_start_idx:t_end_idx]
    pos_trial['y'] = pos['y'][t_start_idx:t_end_idx]
    pos_trial['time'] = pos['time'][t_start_idx:t_end_idx]

    z = dict(position=[])
    z['time'] = pos_trial['time']
    for point in range(len(pos_trial['x'])):
        position = Point(pos_trial['x'][point], pos_trial['y'][point])
        z['position'].append(ideal_path.project(position))
    return z


def raster_plot(spikes, colour='k'):
    for neuron in range(len(spikes)):
        plt.plot(spikes[neuron], np.ones(len(spikes[neuron]))+neuron+1,
                 '|', color=colour)


def tuning_curve(position_z, spike_times, num_bins=100, sampling_rate=1/30.0):
    linear_start = np.min(position_z['position'])
    linear_stop = np.max(position_z['position'])
    bin_edges = np.linspace(linear_start, linear_stop, num=num_bins)
    bin_centers = np.array((bin_edges[1:] + bin_edges[:-1]) / 2.)
    tc = []
    occupancy = np.zeros(len(bin_centers))
    for pos in position_z['position']:
        pos_idx = find_nearest_idx(bin_centers, pos)
        occupancy[pos_idx] += sampling_rate
    occupied_idx = occupancy > 0
    for neuron in range(len(spike_times)):
        spike_z = np.zeros(len(bin_centers))
        for spike_time in spike_times[neuron][0]:
            assigned_bin = find_nearest_idx(np.array(position_z['time']), spike_time)
            which_bin = find_nearest_idx(bin_centers, position_z['position'][assigned_bin])
            spike_z[which_bin] += 1
        firing_rate = np.zeros(len(bin_centers))
        firing_rate[occupied_idx] = spike_z[occupied_idx] / occupancy[occupied_idx]
        tc.append(firing_rate)
    return tc


# Gaussian filter to convolve tuning curves, could be added to the tuning curve function directly.
def filter_tc(tuning_curve, filter_type='gaussian', gaussian_std=3.):
    if filter_type == 'gaussian':
        gauss_tc = []
        # Normalizing gaussian filter
        gaussian_filter = signal.get_window(('gaussian', gaussian_std), gaussian_std*6.)
        normalized_gaussian = gaussian_filter / np.sum(gaussian_filter)
        for firing_rate in tuning_curve:
            gauss_tc.append(np.convolve(firing_rate, normalized_gaussian, mode='same'))
        return gauss_tc


def sort_spikes(tuning_curves, spike_times):
    tc_max_idx = []
    for tc in tuning_curves:
        tc_max_idx.append(np.argmax(tc))
    sort_idx = np.argsort(tc_max_idx)
    sorted_spikes = []
    for sort_tc in sort_idx:
        sorted_spikes.append(spike_times[sort_tc])
    return sorted_spikes


def idx_in_pos(position, index):
    pos = dict()
    pos['x'] = position['x'][index]
    pos['y'] = position['y'][index]
    pos['time'] = position['time'][index]
    return pos


def get_trial_idx(low_priority, mid_priority, high_priority, feeder1_times, feeder2_times, phase_start, phase_stop):
    start_trials = []
    stop_trials = []

    high_priority_time = []
    mid_priority_time = []
    low_priority_time = []

    f1_idx = 0
    f2_idx = 0

    while f1_idx < len(feeder1_times) and f2_idx < len(feeder2_times):
        if f1_idx == len(feeder1_times):
            start_trial = feeder2_times[f2_idx]
            stop_trial = phase_stop
        elif f2_idx == len(feeder2_times):
            start_trial = feeder1_times[f1_idx]
            stop_trial = phase_stop
        else:
            start_trial = min(feeder1_times[f1_idx], feeder2_times[f2_idx])
            if start_trial in feeder1_times:
                f1_idx += 1
                stop_trial = feeder2_times[f2_idx]
            elif start_trial in feeder2_times:
                f2_idx += 1
                stop_trial = feeder1_times[f1_idx]
        start_trials.append(start_trial)
        stop_trials.append(stop_trial)

        for element in high_priority:
            if np.logical_and(start_trial <= element, element < stop_trial):
                high_priority_time.append(start_trial)
                break
        if start_trial not in high_priority_time:
            for element in mid_priority:
                if np.logical_and(start_trial <= element, element < stop_trial):
                    mid_priority_time.append(start_trial)
                    break
        if start_trial not in high_priority_time and start_trial not in mid_priority_time:
            for element in low_priority:
                if np.logical_and(start_trial <= element, element < stop_trial):
                    low_priority_time.append(start_trial)
                    break

    high_priority_trials = []
    mid_priority_trials = []
    low_priority_trials = []

    for trial in high_priority_time:
        high_priority_trials.append((find_nearest_idx(np.array(start_trials), trial), 'novel'))
    for trial in mid_priority_time:
        mid_priority_trials.append((find_nearest_idx(np.array(start_trials), trial), 'shortcut'))
    for trial in low_priority_time:
        low_priority_trials.append((find_nearest_idx(np.array(start_trials), trial), 'u'))

    trials_idx = dict()
    trials_idx['novel'] = high_priority_trials
    trials_idx['shortcut'] = mid_priority_trials
    trials_idx['u'] = low_priority_trials
    trials_idx['start_trials'] = start_trials
    trials_idx['stop_trials'] = stop_trials

    return trials_idx


def bytrial_counts(togethers, min_length=100):
    u_bytrial = np.zeros(min_length)
    shortcut_bytrial = np.zeros(min_length)
    novel_bytrial = np.zeros(min_length)
    for session in togethers:
        if len(session) < min_length:
            min_length = len(session)
    for single_trial in range(min_length):
        for session in togethers:
            if session[single_trial][1] == 'u':
                u_bytrial[single_trial] += 1
            if session[single_trial][1] == 'shortcut':
                shortcut_bytrial[single_trial] += 1
            if session[single_trial][1] == 'novel':
                novel_bytrial[single_trial] += 1

    bytrial = dict()
    bytrial['u'] = u_bytrial
    bytrial['shortcut'] = shortcut_bytrial
    bytrial['novel'] = novel_bytrial

    return bytrial
