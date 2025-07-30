function generate_fft_lut_from_existing()
    % Load existing wavelet LUTs
    load('D:\wavelet_lut_2.mat'); % File containing phi, psi1_lut, psi2_lut
    downsample = 4;

    % Determine FFT computation size
    N = 2500000;
    L = 2^nextpow2(2*N - 1); % Zero-padding length

    % Compute FFT for the scaling function
    phi_fft = fft(phi, L);

    % Compute FFT for first-order wavelets (process in parts if needed)
    psi1_fft_lut = zeros(size(psi1_lut,1), L, 'single'); % Use 'single' to save memory
    for i = 1:size(psi1_lut,1)
        psi1_fft_lut(i,:) = fft(psi1_lut(i,:), L);
        if mod(i,10) == 0
            fprintf('Processing first-order wavelet: %d/%d\n', i, size(psi1_lut,1));
        end
    end

    % Compute FFT for second-order wavelets
    psi2_fft_lut = zeros(size(psi2_lut,1), L, 'single');
    for j = 1:size(psi2_lut,1)
        psi2_fft_lut(j,:) = fft(psi2_lut(j,:), L);
        if mod(j,10) == 0
            fprintf('Processing second-order wavelet: %d/%d\n', j, size(psi2_lut,1));
        end
    end

    % Save FFT LUT as a MAT-file (version 7.3 to support large variables)
    save('wavelet_fft_lut.mat', 'phi_fft', 'psi1_fft_lut', 'psi2_fft_lut', ...
        'scales_1', 'scales_2', 'downsample', 'L', '-v7.3');

    disp('FFT LUT has been successfully generated and saved (supports large variables).');
end
