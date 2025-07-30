function LUT = load_fft_lut_once()
 persistent cached_LUT
 if isempty(cached_LUT)
 tmp = load('wavelet_fft_lut.mat', ...
 'phi_fft', 'psi1_fft_lut', 'psi2_fft_lut', ...
 'scales_1', 'scales_2', 'downsample', 'L');
 cached_LUT = tmp;
 end
 LUT = cached_LUT;
end
function LUT_gpu = prepare_LUT_gpu(LUT)
 LUT_gpu = LUT;
 LUT_gpu.phi_fft = gpuArray(LUT.phi_fft);
 LUT_gpu.psi1_fft_lut = gpuArray(LUT.psi1_fft_lut);
 LUT_gpu.psi2_fft_lut = gpuArray(LUT.psi2_fft_lut);

end
