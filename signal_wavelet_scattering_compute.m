% === Setup ===
fileH = 'D:\DroneRF\10100L_0.csv'; % File path
numSamples = 10e6; % Total number of samples to read from file
segmentSize = 2.5e6; % Size of the processing segment

% --- Optional: select starting point of the segment ---
startSample = 2.5e6; % <-- Change this value to choose a different segment
endSample = startSample + segmentSize - 1;

% Validate range
if endSample > numSamples
    error('Segment exceeds data limit (endSample > numSamples)');
end

% === Load and Normalize Data ===
% Read data from file
data = readmatrix(fileH);
data = data(1:numSamples); % Trim to the required number of samples

% Extract the desired segment
segment = data(startSample:endSample);

% Transfer to GPU
segmentGPU = gpuArray(segment);

% --- Normalize by max ---
segmentGPU = segmentGPU / max(abs(segmentGPU)); % Normalize to [-1, 1]

% === Wavelet Scattering ===
Fs = 40e6; % Sampling frequency
sf = waveletScattering('SignalLength', segmentSize, ...
    'SamplingFrequency', Fs);
[S, U] = scatteringTransform(sf, segmentGPU);

% === Plotting ===
% Plot original signal
figure;
plot((0:segmentSize-1)/Fs, gather(segmentGPU));
xlabel('Time (s)');
ylabel('Amplitude');
title(['Signal Segment (Start at sample ', num2str(startSample), ')']);

% Order 1 scattergram
figure;
scattergram(sf, U, 'FilterBank', 1);
title('Wavelet Scattering Coefficients - Order 1');

% Order 2 scattergram
figure;
scattergram(sf, S, 'FilterBank', 2);
title('Wavelet Scattering Coefficients - Order 2');
