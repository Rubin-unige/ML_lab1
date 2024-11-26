%% First Machine Learning Assignment %%

% Task 2: Build a naive Bayes classifier

% naive bayes classifier function
function [predictions, errorRate] = naive_bayes_classifier(trainingData, testData)

    [n_train, d_train] = size(trainingData);
    [m_test, c_test] = size(testData);

    if ~(c_test == d_train || c_test == (d_train - 1))
        error('Check number of columns in training data.');
    end
    if any(trainingData(:) < 1) || any(testData(:) < 1)
        error('All entries must be >= 1.');
    end

    trainingFeatures = trainingData(:, 1:end-1);
    trainingClass = trainingData(:, end);

    classLabels = unique(trainingClass);
    prior = zeros(length(classLabels), 1);

    for i = 1:length(classLabels)
        count_class_occurance = sum(trainingClass == classLabels(i));
        prior(i) = count_class_occurance / n_train;
    end

    % Save prior probabilities to CSV
    priorTable = table(classLabels, prior, ...
        'VariableNames', {'Class', 'Probability'});
    writetable(priorTable, fullfile('result', 'prior_probabilities.csv'));

    numFeatures = size(trainingFeatures, 2);
    likelihoods = cell(length(classLabels), numFeatures);

    likelihoodData = []; % To store likelihoods for saving
    for i = 1:length(classLabels)
        classData = trainingFeatures(trainingClass == classLabels(i), :);
        for j = 1:numFeatures
            uniqueValues = unique(classData(:, j));
            likelihoods{i, j} = zeros(length(uniqueValues), 1);
            for v = uniqueValues'
                count_value = sum(classData(:, j) == v);
                likelihood = count_value / size(classData, 1);
                likelihoods{i, j}(v) = likelihood;

                % Append to likelihoodData for saving
                likelihoodData = [likelihoodData; classLabels(i), j, v, likelihood];
            end
        end
    end

    % Save likelihoods to CSV
    likelihoodTable = array2table(likelihoodData, ...
        'VariableNames', {'Class', 'Feature', 'Value', 'Likelihood'});
    writetable(likelihoodTable, fullfile('result', 'likelihoods.csv'));

    predictions = zeros(m_test, 1);
    posteriorData = []; % To store posterior probabilities for saving

    for i = 1:m_test
        posterior = zeros(length(classLabels), 1);
        for j = 1:length(classLabels)
            posterior(j) = prior(j);
            for k = 1:numFeatures
                featureValue = testData(i, k);
                if featureValue <= length(likelihoods{j, k})
                    likelihood = likelihoods{j, k}(featureValue);
                    posterior(j) = posterior(j) * likelihood;
                else
                    posterior(j) = 0;
                end
            end
        end

        % Save posterior probabilities for this test instance
        for j = 1:length(classLabels)
            posteriorData = [posteriorData; i, classLabels(j), posterior(j)];
        end

        [~, predictedClassIndex] = max(posterior);
        predictions(i) = classLabels(predictedClassIndex);
    end

    % Save posterior probabilities to CSV
    posteriorTable = array2table(posteriorData, ...
        'VariableNames', {'TestInstance', 'Class', 'PosteriorProbability'});
    writetable(posteriorTable, fullfile('result', 'posterior_probabilities.csv'));

    if c_test == d_train
        trueLabels = testData(:, end);
        errorRate = sum(predictions ~= trueLabels) / m_test;
    else
        errorRate = NaN;
    end
end
