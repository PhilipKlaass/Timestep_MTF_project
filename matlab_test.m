function edge_analysis()
% EDGE_ANALYSIS 
% Main entry point. Uncomment the function you wish to run.
%
%   edge_analysis() will call the “reorder” function by default.
%
% (See the local functions below for open_images, flatfield_correction, etc.)

    reorder();
    % main();
    % intro();
end

%% ----------------------- Utilities -----------------------

function [imgROI, roi_rows, roi_cols] = open_images(filename, roi_select, row0, row1, col0, col1)
% OPEN_IMAGES Reads an image from a subfolder, rotates it 90° CCW,
% and either returns a preset ROI or allows the user to select one.
%
%   [imgROI, roi_rows, roi_cols] = open_images(filename, roi_select, row0, row1, col0, col1)
%
% Note: In MATLAB indices start at 1. Adjust your ROI values accordingly.

    if nargin < 2, roi_select = true; end
    if nargin < 3, row0 = 1; end
    if nargin < 4, row1 = 1; end
    if nargin < 5, col0 = 1; end
    if nargin < 6, col1 = 1; end

    % Determine the script directory and construct full file path.
    script_dir = fileparts(mfilename('fullpath'));
    rel_path = fullfile('images', filename);
    abs_file_path = fullfile(script_dir, rel_path);
    
    % Read image; if RGB, convert to grayscale.
    img = imread(abs_file_path);
    if ndims(img) == 3
        img = rgb2gray(img);
    end
    % Rotate 90° counterclockwise.
    img = rot90(img);
    
    if ~roi_select
        roi_rows = [row0, row1];
        roi_cols = [col0, col1];
        imgROI = img(row0:row1, col0:col1);
        return;
    end

    validROI = false;
    while ~validROI
        imshow(img, []); 
        title('Select ROI. Enter coordinates as "row_start:row_end, col_start:col_end"');
        drawnow;
        roi_coords = input('Enter the coordinates for the wanted ROI (e.g., 1:100,1:100): ', 's');
        coords = strsplit(roi_coords, ',');
        roi_rows = str2num(coords{1});  %#ok<ST2NM>
        roi_cols = str2num(coords{2});  %#ok<ST2NM>
        if (roi_rows(2)-roi_rows(1)) == (roi_cols(2)-roi_cols(1))
            tempROI = img(roi_rows(1):roi_rows(2), roi_cols(1):roi_cols(2));
            imshow(tempROI, []);
            title('Is this the intended ROI? (Y/N)');
            drawnow;
            confirmation = input('Is this the intended ROI? (Y/N): ', 's');
            if strcmpi(confirmation, 'Y')
                validROI = true;
            end
        else
            disp('Please enter a square ROI.');
        end
    end
    imgROI = img(roi_rows(1):roi_rows(2), roi_cols(1):roi_cols(2));
end

function save_as_csv(array, filename)
% SAVE_AS_CSV Saves a 2D numeric array to a CSV-like text file.
    fid = fopen(filename, 'a');
    [r, c] = size(array);
    for i = 1:r
        for j = 1:c
            fprintf(fid, '%f   ', array(i,j));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end

function arr = get_array(filename, sz)
% GET_ARRAY Reads a text file from images_csv and returns a sz-by-sz array.
    script_dir = fileparts(mfilename('fullpath'));
    abs_file_path = fullfile(script_dir, 'images_csv', filename);
    fid = fopen(abs_file_path, 'r');
    arr = ones(sz, sz);
    m = 1;
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            line = strtrim(line);
            parts = strsplit(line, '   ');
            parts = parts(~cellfun('isempty',parts));
            n = 1;
            for k = 1:length(parts)
                arr(m, n) = str2double(parts{k});
                n = n + 1;
            end
            m = m + 1;
        end
    end
    fclose(fid);
end

function [X, Y] = make_scatter(array)
% MAKE_SCATTER Splits a two‐column array into X and Y vectors.
    X = array(:,1);
    Y = array(:,2);
end

%% -------------------- Simulation Functions --------------------

function roi = make_object_plane(theta, roi_height, roi_width, dark, bright)
% MAKE_OBJECT_PLANE Simulates an object-plane with a tilted edge.
    roi = zeros(roi_height, roi_width);
    edge_start = floor(roi_width/2);
    tan_theta = tan(deg2rad(theta));
    for y = 1:roi_height
        for x = 1:roi_width
            % Adjust y by subtracting 1 to mimic 0-indexing
            if x < edge_start || x < edge_start - tan_theta*(y-1)
                roi(y,x) = dark;
            else
                roi(y,x) = bright;
            end
        end
    end
end

function out = make_PSFimage(sz, dark, light)
% MAKE_PSFIMAGE Creates a simulated PSF image.
    out = zeros(sz, sz);
    for i = 1:sz
        for j = 1:(i-1)
            out(i,j) = dark;
        end
    end
    center = floor(sz/2);
    if center < sz/2
        out(center, center) = light;
    else
        out(center, center) = light;
        out(center+1, center) = light;
        out(center, center+1) = light;
        out(center+1, center+1) = light;
    end
end

function image_plane = make_image_plane(object_plane, size_ip)
% MAKE_IMAGE_PLANE Creates an image plane by averaging blocks from the object plane.
    [op_rows, ~] = size(object_plane);
    block_size = op_rows / size_ip;  % assumes division without remainder
    image_plane = zeros(size_ip, size_ip);
    for x = 1:size_ip
        for y = 1:size_ip
            block = object_plane((x-1)*block_size+1 : x*block_size, (y-1)*block_size+1 : y*block_size);
            image_plane(x,y) = mean(block(:));
        end
    end
end

function kernel = make_kernal(xscaling_factor, yscaling_factor, kernel_size)
% MAKE_KERNAL Creates a Gaussian kernel whose values are obtained by
% integrating over each pixel.
    f = @(x,y) exp(-xscaling_factor*x.^2 - yscaling_factor*y.^2);
    half = kernel_size/2;
    total = integral2(f, -half, half, -half, half);
    kernel = zeros(kernel_size, kernel_size);
    for j = 1:kernel_size
        for i = 1:kernel_size
            x_low = -half + (i-1);
            x_high = x_low + 1;
            y_low = -half + (j-1);
            y_high = y_low + 1;
            kernel(j,i) = integral2(f, x_low, x_high, y_low, y_high) / total;
        end
    end
end

function output = convolve_image(kernel, image)
% CONVOLVE_IMAGE Convolves an image with a kernel using border replication.
    [iH, iW] = size(image);
    [~, kW] = size(kernel);
    pad = floor((kW-1)/2);
    padded = padarray(image, [pad pad], 'replicate', 'both');
    output = zeros(iH, iW);
    for j = 1:iH
        for i = 1:iW
            roi = padded(j:j+2*pad, i:i+2*pad);
            output(j,i) = sum(sum(roi .* kernel));
        end
    end
end

function result = add_poisson(edge, density)
% ADD_POISSON Adds Poisson noise to an image.
    result = edge + poissrnd(density, size(edge));
end

function [dist, intensity] = make_lsf(theta, xscaling_factor, yscaling_factor)
% MAKE_LSF Simulates a line spread function.
    dist = linspace(-11.94, 9.95, 1000);
    intensity = exp(-xscaling_factor * dist.^2);
    intensity = intensity / max(intensity);
end

function [dist, intensity] = make_erf(theta, xscaling_factor, yscaling_factor)
% MAKE_ERF Simulates an edge spread function using error function integration.
    dist = linspace(-5, 5, 100);
    deltax = 10 / 100;
    f = @(x) exp(-yscaling_factor*x.^2 - xscaling_factor*(tan(deg2rad(theta)))^2);
    intensity = zeros(size(dist));
    for i = 1:length(dist)
        intensity(i) = integral(f, dist(i)-deltax/2, dist(i)+deltax/2);
    end
    intensity = intensity / max(intensity);
end

function [freqs, mtf] = fft_mtf(a, ~, theta)
% FFT_MTF Computes a modulus transfer function by evaluating a Fourier integral.
    freqs = linspace(0, 2, 20);
    mtf = zeros(size(freqs));
    for i = 1:length(freqs)
        mtf(i) = abs(ft_integral(a, freqs(i)));
    end
    mtf = mtf / max(mtf);
end

function fhat = ft_integral(a, freq)
% FT_INTEGRAL Computes the Fourier transform integral
    integrand = @(x) exp(-a*x.^2 + 2i*pi*x*freq);
    fhat = integral(integrand, -Inf, Inf);
end

%% ------------------- ROI and Flatfield Correction -------------------

function norm_img = flatfield_correction(light, dark, image)
% FLATFIELD_CORRECTION Corrects an image using light and dark frames.
    N = size(image,1);
    avg = sum(double(light(:) - dark(:))) / numel(image);
    corrected = avg * (double(image) - double(dark)) ./ (double(light) - double(dark));
    norm_img = corrected / max(corrected(:));
end

%% -------------------- Edge Detection --------------------

function edge_points = detect_edge_points(array, threshold)
% DETECT_EDGE_POINTS Finds edge points based on a threshold relative to half the maximum.
    light_val = max(array(:));
    [rows, cols] = size(array);
    edge_points = zeros(rows, cols);
    for j = 1:rows
        if sum(edge_points(j,:)) < 5
            for i = cols:-1:1
                if (0.5-threshold)*light_val <= array(j,i) && array(j,i) <= (0.5+threshold)*light_val
                    edge_points(j,i) = 1;
                end
            end
        end
    end
end

function lines = hough_transform(array, threshold1, do_plot)
% HOUGH_TRANSFORM Performs a Hough transform on a binary edge image and
% returns a list of candidate lines.
%
% Returns an N-by-3 matrix where each row is [theta, rho, rho/cos(theta)].
%
% (MATLAB’s hough, houghpeaks, and houghlines functions are used.)
    [H, thetaVals, d] = hough(array, 'Theta', -90:0.25:89.75);
    peaks = houghpeaks(H, 5, 'Threshold', threshold1);
    linesStruct = houghlines(array, thetaVals, d, peaks);
    lines = [];
    for k = 1:length(linesStruct)
        angle = linesStruct(k).theta;
        rho = linesStruct(k).rho;
        line_param = rho / cosd(angle);
        lines = [lines; angle, rho, line_param];  %#ok<AGROW>
        if do_plot
            imshow(array, []);
            hold on;
            xy = [linesStruct(k).point1; linesStruct(k).point2];
            line(xy(:,1), xy(:,2), 'LineWidth',2, 'Color','green');
            hold off;
            pause(0.5);
        end
    end
    disp('Lines:');
    disp(lines);
end

function [esf_x, esf_y] = get_esf(array, theta, r, sampling_frequency, sample_number)
% GET_ESF Extracts the edge spread function (ESF) from an image.
%
%   [esf_x, esf_y] = get_esf(array, theta, r, sampling_frequency, sample_number)
%
% Note: Adjustments have been made for MATLAB’s 1-indexing.
    theta_rad = deg2rad(theta);
    num_rows = size(array,1);
    num_cols = size(array,2);
    x_intercept = r / cos(theta_rad) - num_cols * 0.5 * tan(theta_rad);
    esf_x = [];
    esf_y = [];
    for y_edge = 1:num_rows
        y_adj = y_edge - 0.5;
        x_edge = -y_adj * tan(theta_rad) + r / cos(theta_rad);
        for i = 0:(sample_number-1)
            x_sample1 = x_intercept + i / sampling_frequency;
            x_sample2 = x_intercept - i / sampling_frequency;
            y_sample1 = tan(theta_rad) * (x_sample1 - x_edge) + y_adj;
            y_sample2 = tan(theta_rad) * (x_sample2 - x_edge) + y_adj;
            if y_sample1 >= 1 && y_sample1 <= num_rows && x_sample1 >= 1 && x_sample1 <= num_cols
                idx1 = floor(y_sample1); idx1 = max(idx1,1);
                jdx1 = floor(x_sample1); jdx1 = max(jdx1,1);
                intensity1 = double(array(idx1, jdx1));
                dist1 = sign(x_sample1-x_edge) * sqrt((y_adj-y_sample1)^2 + (x_edge-x_sample1)^2);
                esf_x(end+1) = dist1 - 1;  %#ok<AGROW>
                esf_y(end+1) = intensity1;  %#ok<AGROW>
            end
            if y_sample2 >= 1 && y_sample2 <= num_rows && x_sample2 >= 1 && x_sample2 <= num_cols
                idx2 = floor(y_sample2); idx2 = max(idx2,1);
                jdx2 = floor(x_sample2); jdx2 = max(jdx2,1);
                intensity2 = double(array(idx2, jdx2));
                dist2 = sign(x_sample2-x_edge) * sqrt((y_adj-y_sample2)^2 + (x_edge-x_sample2)^2);
                esf_x(end+1) = dist2 - 1;  %#ok<AGROW>
                esf_y(end+1) = intensity2;  %#ok<AGROW>
            end
        end
    end
end

function [bin_esfx, bin_esfy] = esf_bin_smooth(esf_dist, esf_intensity, binsize)
% ESF_BIN_SMOOTH Averages the ESF intensity over distance bins.
    min_d = min(esf_dist);
    range_d = max(esf_dist) - min_d;
    num_bins = floor(range_d / binsize);
    bin_esfx = zeros(1, num_bins);
    bin_esfy = zeros(1, num_bins);
    for i = 1:num_bins
        tot_intensity = 0;
        count = 0;
        bin_center = min_d + (i-1)*binsize;
        for j = 1:length(esf_dist)
            if esf_dist(j) > (bin_center - binsize/2) && esf_dist(j) <= (bin_center + binsize/2)
                tot_intensity = tot_intensity + esf_intensity(j);
                count = count + 1;
            end
        end
        if count > 0
            bin_esfx(i) = bin_center;
            bin_esfy(i) = tot_intensity / count;
        end
    end
end

function [avg_x, avg_y] = average_filter(esfx, esfy, window_size)
% AVERAGE_FILTER Smooths the data by averaging over a moving window.
    avg_x = esfx;
    avg_y = esfy;
    half_win = ceil(window_size/2);
    N = length(esfy);
    for i = half_win:(N-half_win)
        if i < 0.35 * N || i > 0.65 * N
            avg_y(i) = mean(esfy(i-half_win+1:i+half_win));
        end
    end
    avg_y(1:half_win) = avg_y(half_win);
    avg_y(end-half_win+1:end) = avg_y(end-half_win+1);
end

function [med_x, med_y] = median_filter(esfx, esfy, window_size)
% MEDIAN_FILTER Applies a median filter to the data.
    med_x = esfx;
    out_y = esfy;
    half_win = ceil(window_size/2);
    N = length(esfy);
    for i = half_win:(N-half_win)
        if i < 0.35 * N || i > 0.65 * N
            temp = esfy(i-floor(window_size/2):i+floor(window_size/2));
            out_y(i) = median(temp);
        end
    end
    out_y(1:half_win) = out_y(half_win);
    out_y(end-half_win+1:end) = out_y(end-half_win+1);
    med_y = out_y;
end

function [x_out, y_out] = get_derivative(x, y)
% GET_DERIVATIVE Approximates the derivative of a 1-D function.
    dy = diff(y);
    x_vals = x(2:end);
    if length(dy) > 2
        y_out = dy(2:end-1);
        x_out = x_vals(2:end-1);
    else
        y_out = dy;
        x_out = x_vals;
    end
    m = max(abs(y_out));
    if m ~= 0
        y_out = y_out / m;
    end
end

function [freq, X] = FFT_custom(lsf_dist, lsf_inten)
% FFT_CUSTOM Computes the FFT of a 1-D signal and returns spatial frequencies.
    N = length(lsf_inten);
    X = fft(lsf_inten);
    X = abs(X);
    X = X / max(X);
    R = max(lsf_dist) - min(lsf_dist);
    sr = N / R;
    freq = (0:N-1) * sr / N;
end

%% -------------------- Main Routines --------------------

function intro()
% INTRO A demonstration routine that follows the Python “intro” function.
    [ROI, rows, cols] = open_images('image0008.bmp', true, 400, 600, 400, 600);
    [light, ~, ~] = open_images('image0008_light.bmp', false, rows(1), rows(2), cols(1), cols(2));
    [dark, ~, ~] = open_images('image0008_dark.bmp', false, rows(1), rows(2), cols(1), cols(2));
    corrected_ROI = flatfield_correction(light, dark, ROI);
    
    figure; imshow(corrected_ROI, []); colorbar; title('Corrected ROI');
    pause;
    
    threshold = (rows(2)-rows(1)) * 0.85;
    edge_pts = detect_edge_points(corrected_ROI, 0.2);
    lines = hough_transform(edge_pts, threshold, true);
    
    % Choose (for example) the second detected line.
    r = lines(2,2);
    theta = lines(2,1);
    
    [erf_x, erf_y] = get_esf(corrected_ROI, theta, r, 0.9, 20);
    figure; scatter(erf_x, erf_y, '.'); title('ESF Scatter');
    pause;
    
    [bin_esfx, bin_esfy] = esf_bin_smooth(erf_x, erf_y, 0.1);
    [avg_esfx, avg_esfy] = average_filter(bin_esfx, bin_esfy, 5);
    [med_esfx, med_esfy] = median_filter(avg_esfx, avg_esfy, 5);
    figure; scatter(med_esfx, med_esfy, '.'); title('Median Applied');
    pause;
    
    X_interp = linspace(min(med_esfx), max(med_esfx), 1000);
    Y_interp = interp1(med_esfx, med_esfy, X_interp, 'pchip');
    Yhat = sgolayfilt(Y_interp, 2, 51);
    figure; plot(X_interp, Yhat); title('Savitzky-Golay Applied');
    pause;
    
    [lsf_x, lsf_y] = get_derivative(X_interp, Yhat);
    figure; plot(lsf_x, lsf_y); title('LSF'); pause;
    
    [mtf_x, mtf_y] = FFT_custom(lsf_x, lsf_y);
    figure; scatter(mtf_x, mtf_y, '.'); xlim([0 1]); title('MTF');
    freq_res = (mtf_x(end)-mtf_x(1)) / length(mtf_x);
    disp(freq_res);
end

function reorder()
% REORDER Demonstration routine that follows the Python “reorder” function.
    [ROI, rows, cols] = open_images('image0009.bmp', false, 1200, 1500, 250, 550);
    [light, ~, ~] = open_images('image0009_light.bmp', false, rows(1), rows(2), cols(1), cols(2));
    [dark, ~, ~] = open_images('image0008_dark.bmp', false, rows(1), rows(2), cols(1), cols(2));
    corrected_ROI = flatfield_correction(light, dark, ROI);
    
    figure; imshow(corrected_ROI, []); colorbar; title('Corrected ROI');
    pause;
    
    threshold = size(ROI,2) * 0.75;
    edge_pts = detect_edge_points(corrected_ROI, 0.2);
    lines = hough_transform(edge_pts, threshold, true);
    
    % For example, choose the first line.
    r = lines(1,2);
    theta = lines(1,1);
    
    [erf_x, erf_y] = get_esf(corrected_ROI, theta, r, 0.9, 15);
    [binx, biny] = esf_bin_smooth(erf_x, erf_y, 0.1);
    figure; scatter(binx, biny, '.'); title('Binned'); pause;
    
    % (Optional average and median filters can be applied here.)
    Yhat = sgolayfilt(biny, 2, 51);
    Yhat = Yhat / max(Yhat);
    figure; scatter(binx, Yhat, '.'); title('Savitzky-Golay'); pause;
    
    [lsf_x, lsf_y] = get_derivative(binx, biny);
    lsf_y = lsf_y / max(lsf_y);
    [lsf_x, lsf_y] = average_filter(lsf_x, lsf_y, 20);
    figure; scatter(lsf_x, lsf_y, '.'); title('Averaged LSF'); pause;
    
    [mtf_x, mtf_y] = FFT_custom(lsf_x, lsf_y);
    figure; plot(mtf_x, mtf_y); xlim([0 1]); title('MTF Processed');
    freq_res = (mtf_x(end)-mtf_x(1)) / length(mtf_x);
    disp(freq_res);
end

function main()
% MAIN Demonstration routine that saves the corrected ROI.
    [ROI, rows, cols] = open_images('image0009.bmp', false, 1200, 1500, 250, 550);
    [light, ~, ~] = open_images('image0009_light.bmp', false, rows(1), rows(2), cols(1), cols(2));
    [dark, ~, ~] = open_images('image0008_dark.bmp', false, rows(1), rows(2), cols(1), cols(2));
    corrected_ROI = flatfield_correction(light, dark, ROI);
    imwrite(corrected_ROI, 'corrected_ROI.bmp');
end
