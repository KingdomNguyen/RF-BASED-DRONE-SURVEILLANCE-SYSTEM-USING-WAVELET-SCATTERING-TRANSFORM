fileH = 'D:\DroneRF\00000L_0.csv';
% Load the first 10,000,000 samples from the file
numSamples = 10e6; % 10 million samples
data = readmatrix(fileH);
data = data(1:numSamples); % Take first 10M samples
% Move data to GPU for faster processing
dataGPU = gpuArray(data);
% Define sampling frequency
Fs = 40e6; % 40 MHz
% Split data into 4 segments of 2.5M samples each
segmentSize = 2.5e6;
numSegments = numSamples / segmentSize;
% Reshape into a matrix where each column is a segment
dataSegments = reshape(dataGPU, segmentSize, numSegments);
% Create a figure
figure;
for i = 1:numSegments
 % Extract segment and gather from GPU
 segment = gather(dataSegments(:, i));

 % Time vector for the segment
 t = (0:segmentSize-1) / Fs;

 % Plot in subplot
 subplot(numSegments, 1, i);
 plot(t, segment);
 xlabel('Time (s)');
 ylabel('Amplitude');
 title(['Segment ' num2str(i) ' (2.5M samples)']);
 grid on;
end
