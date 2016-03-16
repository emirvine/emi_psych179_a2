from shortcut_general import *
import os

# Load the data

# os.chdir('C:\Users\Emily\Desktop\R063-2015-03-22_recording')
csc = load_csc('C:\Users\Emily\Desktop\R063-2015-03-22_recording\emi_inputs_csc.mat')
pos = load_videotrack('C:\Users\Emily\Desktop\R063-2015-03-22_recording\emi_inputs_vt.mat')
events = load_events('C:\Users\Emily\Desktop\R063-2015-03-22_recording\emi_inputs_event.mat')
spikes = load_spikes('C:\Users\Emily\Desktop\R063-2015-03-22_recording\emi_inputs_spike.mat')

# Experimental session-specific task times for R063 day 2
task_times = dict()
task_times['prerecord'] = [837.4714, 1143.1]
task_times['phase1'] = [1207.9, 2087.5]
task_times['pauseA'] = [2174.3, 2800.8]
task_times['phase2'] = [2836.2, 4034.1]
task_times['pauseB'] = [4051.3, 6185.6]
task_times['phase3'] = [6249.5, 9373.7]
task_times['postrecord'] = [9395.4, 9792.5]

pxl_to_cm = (7.3452, 7.2286)

fs = 2000

good_lfp = []

# Session-specific path trajectory points
path_pts = dict()
path_pts['feeder1'] = [547, 469]
path_pts['point1'] = [542, 396]
path_pts['turn1'] = [538, 377]
path_pts['point2'] = [511, 380]
path_pts['added1'] = [442, 396]
path_pts['point3'] = [389, 402]
path_pts['added2'] = [332, 415]
path_pts['point4'] = [248, 370]
path_pts['added3'] = [208, 385]
path_pts['turn2'] = [225, 352]
path_pts['point5'] = [217, 316]
path_pts['point6'] = [236, 84]
path_pts['turn3'] = [249, 59]
path_pts['point7'] = [289, 51]
path_pts['point8'] = [532, 47]
path_pts['feeder2'] = [654, 56]
path_pts['shortcut1'] = [446, 391]
path_pts['point9'] = [438, 334]
path_pts['point10'] = [465, 297]
path_pts['point11'] = [471, 277]
path_pts['point12'] = [621, 269]
path_pts['point13'] = [649, 263]
path_pts['added4'] = [653, 280]
path_pts['point14'] = [660, 240]
path_pts['shortcut2'] = [654, 56]
path_pts['novel1'] = [247, 61]
path_pts['point15'] = [135, 55]
path_pts['point16'] = [128, 64]
path_pts['point17'] = [132, 83]
path_pts['novel2'] = [130, 266]

u_trajectory = [path_pts['feeder1'], path_pts['point1'], path_pts['turn1'],
                path_pts['point2'], path_pts['added1'], path_pts['point3'],
                path_pts['added2'], path_pts['point4'], path_pts['added3'],
                path_pts['turn2'], path_pts['point5'], path_pts['point6'],
                path_pts['turn3'], path_pts['point7'], path_pts['point8'],
                path_pts['feeder2']]

shortcut_trajectory = [path_pts['shortcut1'],path_pts['point9'], path_pts['point10'],
                       path_pts['point11'], path_pts['point12'],
                       path_pts['point13'], path_pts['added4'],
                       path_pts['point14'], path_pts['shortcut2']]

novel_trajectory = [path_pts['novel1'], path_pts['point15'], path_pts['point16'],
                    path_pts['point17'], path_pts['novel2']]

path_pts = convert_to_cm(path_pts, pxl_to_cm)

