function x = initializeWeightsGaussian(sz)
x = randn(sz,"single") .* 0.01;
end