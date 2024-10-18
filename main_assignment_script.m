%% First Machine Learning Assignment %%

% Main script that runs whole program without user intervention

% Read the weather data from text file converted into integers by 'weather_data_processing.m' script
processedWeatherData = readmatrix('processed_weather_data.txt');

% Shuffle the processed matrix so everytime the training and test datasets are different
rng("shuffle");
n_row = size(processedWeatherData, 1);
indices = randperm(n_row);

% Split the processed dataset into training data and test data
% this can be made dynamic
% I was thinking of using 70% for training and rest for test, but task requires 10
trainSize = 10; 
trainingData = processedWeatherData(indices(1:trainSize), :);
testData = processedWeatherData(indices((trainSize + 1):end), :);

% Call naive bayes classifier without laplace smoothing function
[predictions, errorRate] = naive_bayes_classifier(trainingData, testData);

% Task 3: Improve the classifier with Laplace (additive) smoothing
% Determine number of unique levels for each feature
numFeatures = size(processedWeatherData, 2) - 1; % Exclude target class
numLevels = zeros(1, numFeatures); % Initialize the numLevel matrix
for j = 1:numFeatures
    numLevels(j) = length(unique(processedWeatherData(:, j))); % Count unique values for each feature
end
% Call naive bayes classifier with laplace smoothing function
[predictions_laplace, errorRate_laplace] = naive_bayes_classifier_laplace(trainingData, testData, numLevels);

% Display the predictions and error rate without laplace smoothing
disp('Predictions for Test Data without laplace smoothing:');
disp(predictions);
if ~isnan(errorRate)
    disp(['Error Rate without laplace smoothing: ' num2str(errorRate * 100) '%']);
end

% Display the predictions and error rate with laplace smoothing
disp('Predictions for Test Data with laplace smoothing:');
disp(predictions_laplace);
if ~isnan(errorRate_laplace)
    disp(['Error Rate with laplace smoothing: ' num2str(errorRate_laplace * 100) '%']);
end