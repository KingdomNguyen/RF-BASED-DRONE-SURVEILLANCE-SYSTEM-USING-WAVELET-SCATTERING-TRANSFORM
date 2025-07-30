fileH = 'D:\DroneRF\11000H_0.csv'; % High-frequency signal (Bebop)
fileL = 'D:\DroneRF\11000L_0.csv'; % Low-frequency signal (AR)
% Sampling frequency
fs = 40e6; % 40 MHz
% Number of samples to read
N = 10e6; % 10 million samples
% Read data from CSV files
signalH = csvread(fileH, 0, 0, [0 0 N-1 0]); % Bebop
signalL = csvread(fileL, 0, 0, [0 0 N-1 0]); % AR
% Normalize signals to [-1, 1]
signalH = 2 * (signalH - min(signalH)) / (max(signalH) - min(signalH)) - 1;
signalL = 2 * (signalL - min(signalL)) / (max(signalL) - min(signalL)) - 1;
% Time vector
t = (0:N-1) / fs; % Time in seconds
% Offset for separation
offset = 1.5;
% Plot signals
figure;
plot(t, signalH, 'b', 'DisplayName', 'Phantom drone high'); hold on;
plot(t, signalL - offset, 'r', 'DisplayName', 'Bebop high');
% Labels and title
xlabel('Time (s)');
ylabel('Normalized Amplitude');
title('Drone RF Signals');
% Legend and grid
legend;
grid on;
hold off;
