% Clear workspace
clear; clc; close all;
%% Load Low-Frequency Data
lowFreqData = load('D:/DroneRF/combined_data_12_low.mat');
lowFreqDataStruct = lowFreqData.struct;
lowFreqScatData = lowFreqDataStruct.Data;
lowFreqLabels = categorical(string(lowFreqDataStruct.Label));
%% Load High-Frequency Data
highFreqData = load('D:/DroneRF/combined_data_12.mat');
highFreqDataStruct = highFreqData.struct;
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
%% **Train TreeBagger (Random Forest) for Low-Frequency Data**
numTrees = 100; % Number of trees
maxSplits = 20; % Max number of splits
treeBagger_low = TreeBagger(numTrees, trainData_low, trainLabels_low, ...
 'Method', 'classification', ...
 'MaxNumSplits', maxSplits, ...
 'OOBPrediction', 'on');
%% **Train TreeBagger (Random Forest) for High-Frequency Data**
treeBagger_high = TreeBagger(numTrees, ...
 gather(trainData_high), ...
 gather(trainLabels_high), ...
 'OOBPrediction','on', ...
 'Method','classification');
%% **Test TreeBagger Models**
[predLabels_tree_low, scores_low] = predict(treeBagger_low, testData_low);
predLabels_tree_low = categorical(predLabels_tree_low);
tree_accuracy_low = sum(predLabels_tree_low == testLabels_low) /
numel(testLabels_low) * 100;
fprintf("Test Accuracy (TreeBagger - Low Frequency): %2.2f%%\n",
tree_accuracy_low);
[predLabels_tree_high, scores_high] = predict(treeBagger_high,
testData_high);
predLabels_tree_high = categorical(predLabels_tree_high);
tree_accuracy_high = sum(predLabels_tree_high == testLabels_high) /
numel(testLabels_high) * 100;
fprintf("Test Accuracy (TreeBagger - High Frequency): %2.2f%%\n",
tree_accuracy_high);
%% **Confusion Matrices**
figure;
confusionchart(testLabels_low, predLabels_tree_low, 'Title', 'Confusion
Matrix - Low Frequency Data');
figure;
confusionchart(testLabels_high, predLabels_tree_high, 'Title', 'Confusion
Matrix - High Frequency Data');
%% **Ensemble Learning (Combine Low & High Frequency Predictions using
Confidence Scores)**
final_pred = categorical(strings(size(testLabels_low))); % Initialize final
prediction
for i = 1:numel(testLabels_low)
 [conf_low, idx_low] = max(scores_low(i, :)); % Get confidence for low
 [conf_high, idx_high] = max(scores_high(i, :)); % Get confidence for high

 % Choose model with higher confidence
 if conf_low > conf_high
 final_pred(i) = predLabels_tree_low(i);
 else
 final_pred(i) = predLabels_tree_high(i);
 end
end
ensemble_accuracy = sum(final_pred == testLabels_low) / numel(testLabels_low)
* 100;
fprintf("Ensemble Model Accuracy (Confidence-Based Combination - TreeBagger):
%2.2f%%\n", ensemble_accuracy);
%% **Final Confusion Matrix**
figure;
confusionchart(testLabels_low, final_pred, 'Title', 'Confusion Matrix -
Confidence-Based Ensemble Model');
% Determine if each prediction is correct for both models
correct_low = (predLabels_tree_low == testLabels_low); % True if correct,
False if wrong
correct_high = (predLabels_tree_high == testLabels_high);
% Convert logical values (1 = True, 0 = False) to categorical for better
visualization
result_low = categorical(correct_low, [1, 0], {'True', 'False'});
result_high = categorical(correct_high, [1, 0], {'True', 'False'});
% Create a table with sample index, confidence scores, and classification
results
test_samples = (1:length(testLabels_low))'; % Sample indices
confidenceTable = table(test_samples, ...
 max(scores_low, [], 2), result_low, ...
 max(scores_high, [], 2), result_high, ...
 'VariableNames', {'SampleIndex', 'LowFreq_Confidence', 'LowFreq_Result',
...
 'HighFreq_Confidence', 'HighFreq_Result'});
% Display the table
disp(confidenceTable);

%% Determine Final Decision Confidence and Classification Result
final_confidence = zeros(length(testLabels_low), 1);
final_result = categorical(repmat("False", length(testLabels_low), 1),
["True", "False"]);
for i = 1:length(testLabels_low)
 % Choose the model with the higher confidence score
 if max(scores_low(i, :)) > max(scores_high(i, :))
 final_confidence(i) = max(scores_low(i, :));
 final_prediction = predLabels_tree_low(i);
 else
 final_confidence(i) = max(scores_high(i, :));
 final_prediction = predLabels_tree_high(i);
 end

 % Compare with true label
 if final_prediction == testLabels_low(i)
 final_result(i) = "True";
 else
 final_result(i) = "False";
 end
end
%% Create and Display Final Table
finalTable = table(test_samples, ...
 max(scores_low, [], 2), result_low, ...
 max(scores_high, [], 2), result_high, ...
 final_confidence, final_result, ...
 'VariableNames', {'SampleIndex', 'LowFreq_Confidence', 'LowFreq_Result',
...
 'HighFreq_Confidence', 'HighFreq_Result', ...
 'Final_Decision_Confidence', 'Final_Result'});
disp(finalTable);
