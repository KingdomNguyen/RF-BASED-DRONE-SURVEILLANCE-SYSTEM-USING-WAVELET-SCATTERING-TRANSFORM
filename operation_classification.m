clear; clc; close all;
%% Load Low-Frequency Data
lowFreqData = load('D:/DroneRF/Operation_low.mat');
lowFreqDataStruct = lowFreqData.newStruct;
lowFreqScatData = lowFreqDataStruct.Data;
lowFreqLabels = categorical(string(lowFreqDataStruct.Label));

%% Load High-Frequency Data
highFreqData = load('D:/DroneRF/Operation_high.mat');
highFreqDataStruct = highFreqData.newStruct;
highFreqScatData = highFreqDataStruct.Data;
highFreqLabels = categorical(string(highFreqDataStruct.Label));
%% Split into Training and Testing
percent_train = 70;
num_samples = size(lowFreqScatData, 3); % Assuming both datasets have the
same number of samples
num_train = round(percent_train / 100 * num_samples);
rng(1); % For reproducibility
shuffled_indices = randperm(num_samples);
train_indices = shuffled_indices(1:num_train);
test_indices = shuffled_indices(num_train + 1:end);
%% Prepare Low-Frequency Data
trainData_low = reshape(lowFreqScatData(:,:,train_indices), [], num_train)';
testData_low = reshape(lowFreqScatData(:,:,test_indices), [], num_samples -
num_train)';
trainLabels_low = lowFreqLabels(train_indices);
testLabels_low = lowFreqLabels(test_indices);
%% Prepare High-Frequency Data
trainData_high = reshape(highFreqScatData(:,:,train_indices), [],
num_train)';
testData_high = reshape(highFreqScatData(:,:,test_indices), [], num_samples -
num_train)';
trainLabels_high = highFreqLabels(train_indices);
testLabels_high = highFreqLabels(test_indices);
%% Train Bagging (GBM) for Low-Frequency Data
gbm_low = fitcensemble(trainData_low, trainLabels_low, ...
 'Method', 'Bag', ...
 'NumLearningCycles', 50, ...
 'Learners', 'Tree');
%% Train Bagging (GBM) for High-Frequency Data
gbm_high = fitcensemble(trainData_high, trainLabels_high, ...
 'Method', 'Bag', ...
 'NumLearningCycles', 50, ...
 'Learners', 'Tree');
%% Test Base Models (Get Predictions and Confidence Scores)
[predTest_gbm_low, score_low] = predict(gbm_low, testData_low);
[predTest_gbm_high, score_high] = predict(gbm_high, testData_high);

%% Combine Based on Raw Confidence Scores
weight_low = 1.1; % Weight for low-frequency model
weight_high = 1.75; % Weight for high-frequency model
final_pred = categorical(strings(size(testLabels_low)));
for i = 1:length(testLabels_low)
 weighted_conf_low = max(score_low(i, :)) * weight_low;
 weighted_conf_high = max(score_high(i, :)) * weight_high;
 if weighted_conf_low > weighted_conf_high
 final_pred(i) = predTest_gbm_low(i);
 else
 final_pred(i) = predTest_gbm_high(i);
 end
end
%% Calculate Accuracy
ensemble_accuracy = sum(final_pred == testLabels_low) / numel(testLabels_low)
* 100;
fprintf("Ensemble Accuracy without Regularization: %2.2f%%\n",
ensemble_accuracy);
%% Confusion Matrix
figure;
confusionchart(testLabels_low, final_pred, 'Title', 'Combined model');
figure;
confusionchart(testLabels_low, predTest_gbm_low, 'Title', 'Low frequency');
figure;
confusionchart(testLabels_low, predTest_gbm_high, 'Title', 'High frequency');
%% Determine if each prediction is correct for both models
correct_low = (predTest_gbm_low == testLabels_low);
correct_high = (predTest_gbm_high == testLabels_high);
result_low = categorical(correct_low, [1, 0], {'True', 'False'});
result_high = categorical(correct_high, [1, 0], {'True', 'False'});
test_samples = (1:length(testLabels_low))';
confidenceTable = table(test_samples, ...
 max(score_low, [], 2), result_low, ...
 max(score_high, [], 2), result_high, ...
 'VariableNames', {'SampleIndex', 'LowFreq_Confidence', 'LowFreq_Result',
...
 'HighFreq_Confidence', 'HighFreq_Result'});
disp(confidenceTable);
%% Determine Final Decision Confidence and Classification Result
final_confidence = zeros(length(testLabels_low), 1);

final_result = categorical(repmat("False", length(testLabels_low), 1),
["True", "False"]);
for i = 1:length(testLabels_low)
 weighted_conf_low = max(score_low(i, :)) * weight_low;
 weighted_conf_high = max(score_high(i, :)) * weight_high;
 if weighted_conf_low > weighted_conf_high
 final_confidence(i) = weighted_conf_low;
 final_prediction = predTest_gbm_low(i);
 else
 final_confidence(i) = weighted_conf_high;
 final_prediction = predTest_gbm_high(i);
 end

 if final_prediction == testLabels_low(i)
 final_result(i) = "True";
 else
 final_result(i) = "False";
 end
end
finalTable = table(test_samples, ...
 max(score_low, [], 2), result_low, ...
 max(score_high, [], 2), result_high, ...
 final_confidence, final_result, ...
 'VariableNames', {'SampleIndex', 'LowFreq_Confidence', 'LowFreq_Result',
...
 'HighFreq_Confidence', 'HighFreq_Result', ...
 'Final_Decision_Confidence', 'Final_Result'});
disp(finalTable);
%% Analyze Cases Where Lower Confidence Score is Correct but Higher
Confidence Score is Wrong
low_conf_but_correct = 0;
high_conf_but_correct = 0;
low_conf_model = [];
for i = 1:length(testLabels_low)
 weighted_conf_low = max(score_low(i, :)) * weight_low;
 weighted_conf_high = max(score_high(i, :)) * weight_high;
 is_low_correct = (predTest_gbm_low(i) == testLabels_low(i));
 is_high_correct = (predTest_gbm_high(i) == testLabels_low(i));
 if weighted_conf_low < weighted_conf_high && is_low_correct &&
~is_high_correct
 low_conf_but_correct = low_conf_but_correct + 1;
 elseif weighted_conf_high < weighted_conf_low && is_high_correct &&
~is_low_correct
 high_conf_but_correct = high_conf_but_correct + 1;
 low_conf_model = [low_conf_model; "High"];
 end
end
fprintf("\nSamples where lower weighted confidence score classified correctly
while higher confidence score failed: %d\n", ...
 low_conf_but_correct + high_conf_but_correct);
fprintf(" - Low-Frequency Model had lower confidence but was correct: %d
times\n", low_conf_but_correct);
fprintf(" - High-Frequency Model had lower confidence but was correct: %d
times\n", high_conf_but_correct);
lowConfTable = table((1:length(low_conf_model))', low_conf_model, ...
 'VariableNames', {'SampleIndex', 'LowerConfidenceModel'});
disp(lowConfTable);
disp('Unique Labels in testLabels_low:');
disp(unique(testLabels_low));
disp('Unique Labels in final_pred:');
disp(unique(final_pred));
mismatches = testLabels_low(final_pred ~= testLabels_low);
disp('Misclassified Labels:');
disp(mismatches);
