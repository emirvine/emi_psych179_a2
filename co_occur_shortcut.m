% This code conducts a co-occurrence analysis for the "shortcut" experiment.
% The critical aspects of this experiment for this analysis are that 1)
% there are 3 separate track trajectories: U, shortcut, and novel. 2) The
% shortcut and novel track segments are introduced in pauseA, present
% during phase2 and pauseB, and traversed during phase3. 3) The recording
% electrodes were in the dCA1 region of the hippocampus, where we can take
% advantage of Place Cells and Sharp-Wave Ripple (SWR) events. 
cd('C:\Users\Emily\Desktop\R063-2015-03-22_recording');
LoadExpKeys;


%% Extract times associated with putative sharp-wave ripple (swr) events.
cfg = [];
cfg.fc = ExpKeys.good_lfp;
csc = LoadCSC(cfg);

% csc = restrict(csc, ExpKeys.pauseA(1), ExpKeys.pauseA(2));
csc = restrict(csc, ExpKeys.pauseB(1), ExpKeys.pauseB(2));

cfg = [];
cfg.load_questionable_cells = 1;
spikes = LoadSpikes(cfg);

load('R063-2015-03-22-emi-vt.mat');

pos_tsd.data(1,:) = pos_tsd.data(1,:)./ExpKeys.pxl_to_cm(1); 
pos_tsd.data(2,:) = pos_tsd.data(2,:)./ExpKeys.pxl_to_cm(2);

% swr events occur at frequencies between 140Hz and 250Hz
cfg = [];
cfg.type = 'fdesign';
cfg.f = [140, 250];
csc_filtered = FilterLFP(cfg, csc);

filtering = abs(csc_filtered.data);

% Convolve with a Gaussian kernel
gauss_x = 60;
gauss_sigma = 20;
kernel = gausskernel(gauss_x, gauss_sigma);
filtering = conv(filtering, kernel, 'same');

swr = tsd(csc.tvec, filtering);

% Extract times associated with swr events
cfg = [];
cfg.method = 'zscore';
cfg.threshold = 3;
cfg.minlen = 0.02;
cfg.merge_thr = 0;
swr_evt = TSDtoIV(cfg, swr);

clearvars -except pos_tsd swr_evt ExpKeys spikes


%% Load track spikes .mat (generated with Python's Shapely package)
% load('spike_pos_14_R063d3.mat');
load('spike_pos_R063d3.mat');
% load('saved_struct.mat');

tracks = {'novel', 'shortcut', 'u'};
for side = 1:length(tracks)
    spikes_track.(tracks{side}) = ts;
    spikes_track.(tracks{side}).usr = spikes.usr;
    spikes_track.(tracks{side}).t = spike_pos.(tracks{side});
end

%% Extract running positions to estimate place fields
linspeed = getLinSpd([], pos_tsd);

cfg = [];
cfg.method = 'raw';
cfg.dcn = '>';
cfg.threshold = 3.5;
iv_fast = TSDtoIV(cfg, linspeed);

pos_fast = restrict(pos_tsd, iv_fast); 

for side = 1:length(tracks)
    spikes_fast.(tracks{side}) = ts;
    spikes_fast.(tracks{side}).usr = spikes.usr;
    spikes_fast.(tracks{side}) = restrict(spikes_track.(tracks{side}), iv_fast);
end


%% Estimating place fields based on linear trajectories

linear.novel = linear_shortcut(pos_fast, ExpKeys, 'novel1', 'novel2');
linear.shortcut = linear_shortcut(pos_fast, ExpKeys, 'shortcut1', 'shortcut2');
linear.u = linear_shortcut(pos_fast, ExpKeys, 'feeder1', 'feeder2');

for side = 1:length(tracks)
    cfg = [];
    cfg.binSize = 1;
    tc.(tracks{side}) = MakeTC(cfg, spikes_fast.(tracks{side}), linear.(tracks{side}));
    fields.(tracks{side}) = tc.(tracks{side}).field_template_idx;
end

%% *Optional* check point. Should see tuning curves for all neurons.
figure(1); clf; hold on;
tc_plot = tc.novel.tc;
for tuning = 1:size(tc_plot, 1)
    plot(tc_plot(tuning,:), 'b', 'LineWidth', 2);
end
set(gca,'FontSize',14); xlabel('Linearized position (cm)'); ylabel('Firing rate (Hz)');


%% For this analysis, we only want to include place cells that belong to 
% an individual track. Here, we remove any neuron that have place fields
% that are on multiple tracks. This is an area that should be further 
% analyzed -- bimodal place fields may be one of the ways that the neurons
% adapt to the changing environment.
[~, remove.u, remove.shortcut] = intersect(fields.u, fields.shortcut);
u_shortcut = {'u', 'shortcut'};
for side = 1:length(u_shortcut)
    fields.(u_shortcut{side})(remove.(u_shortcut{side})) = [];
    fields.(u_shortcut{side}) = unique(fields.(u_shortcut{side}));
    spikes_side.(u_shortcut{side}) = SelectTS([], spikes, fields.(u_shortcut{side}));
end

[~, remove.u, remove.novel] = intersect(fields.u, fields.novel);
u_novel = {'u', 'novel'};
for side = 1:length(u_novel)
    fields.(u_novel{side})(remove.(u_novel{side})) = [];
    fields.(u_novel{side}) = unique(fields.(u_novel{side}));
    spikes_side.(u_novel{side}) = SelectTS([], spikes, fields.(u_novel{side}));
end

[~, remove.shortcut, remove.novel] = intersect(fields.shortcut, fields.novel);
shortcut_novel = {'shortcut', 'novel'};
for side = 1:length(shortcut_novel)
    fields.(shortcut_novel{side})(remove.(shortcut_novel{side})) = [];
    fields.(shortcut_novel{side}) = unique(fields.(shortcut_novel{side}));
    spikes_side.(shortcut_novel{side}) = SelectTS([], spikes, fields.(shortcut_novel{side}));
end


%% Construct Q matrices for each track.
for side = 1:length(tracks)
    cfg = [];
    cfg.win = 0.1;
    q.(tracks{side}) = MakeQfromS2(cfg, spikes_side.(tracks{side}), swr_evt);
end



%% *Optional* Test code with toys that show A) No evidence of co-activation beyond chance
%  and B) No difference between the track segments.
% ----------------------- not yet implemented -------------------------


%% Co-activation probabilities
% We are currently using 10000 shuffles. Tests with more shuffles yielded 
% similar results. 
for side = 1:length(tracks)
    cfg = [];
    cfg.nShuffle = 10000;
    cfg.useMask = 1;
    cfg.outputFormat = 'vectorU';
    cooccur.(tracks{side}) = CoOccurQ2(cfg, q.(tracks{side}));
end


%% Plotting
p_list = {'p0', 'p3', 'p4'};
titles = {'Activation probability (p0)', 'Observed coactivity (p3)', 'Coactivation above chance levels (p4)'};
ylabels = {{'Proportion of'; 'SWRs active'}, {'Cell pair'; 'joint probability'}, {'SWR coactivation'; 'z-scored'}};
tracks = {'novel', 'shortcut', 'u'};
colors = flipud(linspecer(3));
location = [0.0, 0.75, 1.5];
xticklabel = {'Novel', 'Shortcut', 'U'};
 
figure;
for prob = length(p_list):-1:1
    p_data(prob, 1:length(tracks)) = [nanmean(cooccur.novel.(p_list{prob})), nanmean(cooccur.shortcut.(p_list{prob})), nanmean(cooccur.u.(p_list{prob}))];
    fig(prob) = subplot(1, 3, prob);
    for side = 1:length(tracks)
        bar(location(side), p_data(prob, side), 0.6, 'FaceColor', colors(side,:), 'Edgecolor', 'none');
        hold on;
        title(titles{prob});
        ylabel(ylabels{prob});
        xlabel('Field location');
        % todo: Include how many neurons and how many pairs were analyzed.
    end
end

set(fig, 'XLim', [location(1)-1, location(3)+1], 'XTick', location, 'XTickLabel', xticklabel, 'FontSize', 12);
set(fig, 'PlotBoxAspectRatio', [1, 1, 1]);
% todo: Stats.
maximize
savefig('C:\Users\Emily\Desktop\cooccurrance_r063d3_pB.png')