function [ linear ] = linear_shortcut(position, expkeys, boundary1, boundary2)
% Requires:
% - LinearizePos.m *Returns linearized position
% - emi_makecoord.m *Returns coord
% - restrict.m *Returns tsd data restricted to specific parts
%
% * Example usage:
% **** todo. ****
% 'feeder1' and 'feeder2' for u; 'shortcut1' and 'shortcut2' for shortcut;
% 'novel1' and 'novel2' for novel

% Get conversion factor of pixel-to-cm from expkeys
conv_cm = expkeys.pxl_to_cm;

% Get trajectories from expkeys for each path in cm
trajectory_pts(:,1) = expkeys.trajectory_pts(:,1) ./ conv_cm(1);
trajectory_pts(:,2) = expkeys.trajectory_pts(:,2) ./ conv_cm(2);
trajectory_labels = expkeys.trajectory_labels;
trajectory_idx = expkeys.trajectory_idx;

linear_path = short_path(trajectory_pts, trajectory_labels,...
    trajectory_idx, boundary1, boundary2);

% Set coordinates
coordinates = shortcut_makecoord(getd(position,'x'), getd(position,'y'), linear_path);

% Restrict to experimental Phase
% position = restrict(position, start_phase, end_phase);
position = rm_nan(position);

% Get linearized coordinates
cfg.Coord = coordinates;
linear = LinearizePos(cfg, position);
end

function [ all_path ] = short_path(trajectory_pts, trajectory_labels, ...
    trajectory_idx, start_label, end_label)
start_path = trajectory_idx(strcmp(trajectory_labels,start_label));
end_path = trajectory_idx(strcmp(trajectory_labels,end_label));
all_path = trajectory_pts(start_path:end_path, :);
end

function [ pos_tsd ] = rm_nan(pos_tsd)
keep = ~isnan(pos_tsd.data(1,:)) & ~isnan(pos_tsd.data(2,:));
pos_tsd.data = pos_tsd.data(:,keep);
pos_tsd.tvec = pos_tsd.tvec(keep);
end