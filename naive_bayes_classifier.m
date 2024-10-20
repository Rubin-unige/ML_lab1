%% First Machine Learning Assignment %%

% Task 2: Build a naive Bayes classifier

% naive bayes classifier function
function [predictions, errorRate] = naive_bayes_classifier(trainingData, testData)

    % Check dimensions
    [n_train, d_train] = size(trainingData);
    [m_test, c_test] = size(testData);
    % Check number of columns
    if ~(c_test == d_train || c_test == (d_train - 1))
        error('Check number of columns in training data.')
    end
    % Check for invalid data entries
    if any(trainingData(:) < 1) || any(testData(:) < 1)
        error('All entries must be >= 1.')
    end

    % Extract features and class
    trainingFeatures = trainingData(:, 1:end-1);
    trainingClass = trainingData(:, end);

    % Calculate prior probabilities
    classLabels = unique(trainingClass);
    prior = zeros(length(classLabels), 1); % Initialize prior probabilities matrix with zeros
    for i = 1:length(classLabels)
        count_class_occurance = sum(trainingClass == classLabels(i)); % Count how many times the class appears
        prior(i) = count_class_occurance / n_train; % Calculate prior probability 
    end
    % Create a table for prior probabilities
    priorTable = table(classLabels, prior, 'VariableNames', {'Class', 'PriorProbability'});
    disp('Prior Probabilities Table:');
    disp(priorTable);


    % Calculate likelihoods without laplace smoothing
    numFeatures = size(trainingFeatures, 2); % Number of features
    % Initialize likelihoods as cell array
    % cell so that it can hold all the possible feature values
    likelihoods = cell(length(classLabels), numFeatures); 
    for i = 1:length(classLabels)
        classData = trainingFeatures(trainingClass == classLabels(i), :);
        for j = 1:numFeatures % For each feature
            uniqueValues = unique(classData(:, j)); % Get unique values for the feature
            likelihoods{i, j} = zeros(length(uniqueValues), 1); % Initialize likelihoods for this feature
            for v = uniqueValues' % Iterate through unique values
                count_value = sum(classData(:, j) == v); % Count occurrences of value v in class data
                likelihood = count_value / size(classData, 1); % Calculate likelihood
                likelihoods{i, j}(v) = likelihood; % Store likelihood in the corresponding index
            end 
        end
    end
    % Example for the first feature
    likelihoodTableFeature1 = array2table(likelihoods{1, 1}, 'VariableNames', {'Feature1_Likelihood'});
    disp('Likelihood Table for Feature 1 (Class 1):');
    disp(likelihoodTableFeature1);


    % Calculate prosterior probabilities and predictions
    predictions = zeros(m_test, 1); % Initialize predictions vector
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
        disp('Posterior probabilities without laplace');
        disp(posterior);
    
        % Compare posterior probability and take higher probability
        [~, predictedClassIndex] = max(posterior); 
        predictions(i) = classLabels(predictedClassIndex); % Store the predicted class
    end
    % Create a table for posterior probabilities for each test instance
    posteriorTable = array2table(posterior, 'VariableNames', {'Posterior_Class1', 'Posterior_Class2'});
    disp('Posterior Probabilities Table:');
    disp(posteriorTable);

    % Compute error rate if the test set has the target column
    if c_test == d_train
        trueLabels = testData(:, end); % Get true labels from the test set
        errorRate = sum(predictions ~= trueLabels) / m_test; % Calculate error rate
    else
        errorRate = NaN; % No error rate if no target column
    end
end