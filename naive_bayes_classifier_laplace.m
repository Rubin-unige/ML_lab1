%% First Machine Learning Assignment %%

% Task 3: Make the classifier robust to missing data with Laplace (additive) smoothing

% naive bayes classifier with laplace smoothing function
function [predictions_laplace, errorRate_laplace] = naive_bayes_classifier_laplace(trainingData, testData, numLevels, alphaLaplace)

    % Check dimensions
    [n_train, d_train] = size(trainingData);
    [m_test, c_test] = size(testData);
    % Check number of columns
    if ~(c_test == d_train || c_test == d_train - 1)
        error('Check number of columns in training data.')
    end
    % Check for invalid data entries
    if any(trainingData(:) < 1) || any(testData(:) < 1)
        error('All entries must be >= 1.')
    end

    % Extract features and class
    trainingFeatures = trainingData(:, 1:end-1);
    trainingClass = trainingData(:, end);

    fprintf('\nNaive Bayes Classifier with Laplace smoothing\n');

    % Calculate prior probabilities
    classLabels = unique(trainingClass);
    prior = zeros(length(classLabels), 1); % Initialize prior probabilities matrix with zeros
    for i = 1:length(classLabels)
        count_class_occurance = sum(trainingClass == classLabels(i)); % Count how many times the class appears
        prior(i) = count_class_occurance / n_train; % Calculate prior probability 
    end

    % Display prior probabilities
    fprintf('Prior Probabilities:\n');
    fprintf('%-10s %-10s\n', 'Class', 'Probability');
    fprintf('-------------------------\n');
    for i = 1:length(classLabels)
        fprintf('%-10d %-10.4f\n', classLabels(i), prior(i));
    end

    % Calculate likelihoods with laplace smoothing
    numFeatures = size(trainingFeatures, 2); % Number of features
    likelihoods = cell(length(classLabels), numFeatures); % Initialize likelihoods as cell array

    % Display likelihoods
    fprintf('\nLikelihoods:\n');
    fprintf('%-10s %-10s %-10s %-10s\n', 'Class', 'Feature', 'Value', 'Likelihood');
    fprintf('-----------------------------------------------------\n');

    for i = 1:length(classLabels)
        classData = trainingFeatures(trainingClass == classLabels(i), :);
        for j = 1:numFeatures % For each feature            
            % Laplace smoothing added
            likelihoods{i, j} = zeros(numLevels(j), 1); % Adjust size according to numLevels
            for v = 1:numLevels(j) % Now iterating through numLevels
                count_value = sum(classData(:, j) == v); 
                likelihood = (count_value + alphaLaplace)/(size(classData, 1)+(alphaLaplace*numLevels(j))); % Laplace smoothing
                likelihoods{i, j}(v) = likelihood; 
                fprintf('%-10d %-10d %-10d %-10.4f\n', classLabels(i), j, v, likelihood); % Print likelihood
            end
        end
    end

    % Calculate prosterior probabilities and predictions
    predictions_laplace = zeros(m_test, 1); % Initialize predictions vector
    for i = 1:m_test
        posterior = zeros(length(classLabels), 1); % Initialize posterior probabilities
        for j = 1:length(classLabels) % For each class (Yes, No)
            posterior(j) = prior(j); % Start with the prior probability         
            for k = 1:numFeatures % For each feature
                featureValue = testData(i, k); % Get the feature value for the test instance          
                % Check if the feature value is within the bounds of the likelihood array
                if featureValue <= length(likelihoods{j, k}) 
                    likelihood = likelihoods{j, k}(featureValue); 
                    posterior(j) = posterior(j) * likelihood; % Update the posterior probability
                else
                    posterior(j) = 0; % If the feature value is out of bounds, set to 0
                end
            end
        end

        % Display posterior probabilities for the test instance
        fprintf('\nTest Instance %d:\n', i);
        fprintf('%-10s %-10s\n', 'Class', 'Posterior Probability');
        fprintf('-----------------------------------\n');
        for j = 1:length(classLabels)
            fprintf('%-10d %-10.4f\n', classLabels(j), posterior(j));
        end
    
        % Compare posterior probability and take higher probability
        [~, predictedClassIndex] = max(posterior); 
        predictions_laplace(i) = classLabels(predictedClassIndex); % Store the predicted class
    end

    % Compute error rate if the test set has the target column
    if c_test == d_train
        trueLabels = testData(:, end); % Get true labels from the test set
        errorRate_laplace = sum(predictions_laplace ~= trueLabels) / m_test; % Calculate error rate
    else
        errorRate_laplace = NaN; % No error rate if no target column
    end
end