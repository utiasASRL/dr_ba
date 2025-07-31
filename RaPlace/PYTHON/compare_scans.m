%% visualize_loops.m
% Visual inspection of RaPlace loop candidates

%--- CONFIGURATION ---
csvFile      = 'raplace_loops_June_12_17_00.csv';           % your CSV output
% radarFolder  = '/home/liweican-2025-research/data/boreas-2021-04-08-12-44/radar/cart';
radarFolder  = '/home/liweican-2025-research/root/dro/output/boreas-2021-04-08-12-44_May/local_maps';
scoreThresh  = 20500;                          % only show loops with score >= this
maxPairs     = Inf;                           % or set to, e.g., 20
pauseBetween = true;                          % press a key to advance

%--- LOAD THE CSV ---
T = readtable(csvFile); 

% If your CSV uses different column names, adjust these:
% e.g. T.scan_i_name, T.scan_j_name
scanI = T.scan_i_name;    % cell array of filenames, e.g. '1617900271271733.png'
scanJ = T.scan_j_name;
score = T.score;     % numeric

%--- FILTER BY SCORE (optional) ---
mask = score >= scoreThresh;
scanI = scanI(mask);
scanJ = scanJ(mask);
score = score(mask);

nPairs = min(numel(scanI), maxPairs);
fprintf('Displaying %d loop pairs (score >= %.2f)\n', nPairs, scoreThresh);

%--- MAIN LOOP ---
for k = 1:nPairs
    fname1 = scanI{k};
    fname2 = scanJ{k};
    img1 = imread(fullfile(radarFolder, fname1));
    img2 = imread(fullfile(radarFolder, fname2));
    
    % SIDE-BY-SIDE
    figure(1); clf;
    subplot(1,2,1);
    imshow(img1, []); title(sprintf('Scan %s', fname1),'Interpreter','none');
    subplot(1,2,2);
    imshow(img2, []); title(sprintf('Scan %s', fname2),'Interpreter','none');
    sgtitle(sprintf('Loop #%d  Score=%.3f', k, score(k)));
    
    % OVERLAY
    figure(2); clf;
    % convert to RGB for blending
    rgb1 = ind2rgb(img1, gray(256));
    rgb2 = ind2rgb(img2, gray(256));
    overlay = imfuse(rgb1, rgb2, 'blend', 'Scaling','joint');
    imshow(overlay); title('Alpha Blend Overlay');
    
    if pauseBetween
        fprintf('Pair %d/%d displayed. Press any key to continue...\n', k, nPairs);
        pause;
    else
        pause(0.5);
    end
end

fprintf('Done.\n');
