% test_scattering.m
clc; clear; close all;

% Load LUT data
% Read signal from CSV file (1 row, many columns)
csv_data = readmatrix('D:\DroneRF\AR drone\RF Data_10100_H\10100H_0.csv');
x = csv_data(:); % Ensure it's a column vector

% Normalize to [-1, 1]
x = x / max(abs(x));

% Parameters
segment_length = 2.5e6;
num_segments = ceil(length(x) / segment_length);

LUT = load_fft_lut_once();      % Load FFT lookup table (from disk)
LUT_gpu = prepare_LUT_gpu(LUT); % Prepare LUT on GPU

% Start timing
tic;

% Initialize result container
S_all = [];

% Loop over segments
for i = 1:num_segments
    start_idx = (i - 1) * segment_length + 1;
    end_idx = min(i * segment_length, length(x));
    x_seg = x(start_idx:end_idx);
    x_seg = x_seg'; % Convert to row vector

    % Run scattering transform on segment
    S = scattering_transform_optimized_1(x_seg, LUT_gpu);
    S_all = [S_all; S]; % Stack vertically (along dimension 1)

    disp(['Finished processing segment ' num2str(i) ' / ' num2str(num_segments)]);
end

% Stop timing
elapsed_time = toc;

% Display results
disp(['Total segments: ' num2str(num_segments)]);
disp(['Total processing time: ' num2str(elapsed_time) ' seconds']);
