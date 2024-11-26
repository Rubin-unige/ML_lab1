
%% First Machine learning assignment %%

% Task 1: Data preprocessing 

addpath("data\");

% Read weather data from text file 
weatherData = readtable('data\weather_data.txt');

% Preallocate matrix space in memory
n_row = height(weatherData);
processedWeatherData = zeros(n_row, width(weatherData)); 

% Convert categorical data into integer >= 1
for i = 1:n_row
    switch weatherData.Outlook{i} % convert outlook
        case 'overcast'
            processedWeatherData(i, 1) = 1;
        case 'rainy'
            processedWeatherData(i, 1) = 2;
        case 'sunny'
            processedWeatherData(i, 1) = 3;
    end
    switch weatherData.Temperature{i} % convert temperature
        case 'hot'
            processedWeatherData(i, 2) = 1;
        case 'cool'
            processedWeatherData(i, 2) = 2;
        case 'mild'
            processedWeatherData(i, 2) = 3;
    end
    switch weatherData.Humidity{i} % convert humidity
        case 'high'
            processedWeatherData(i, 3) = 1;
        case 'normal'
            processedWeatherData(i, 3) = 2;
    end
    switch weatherData.Windy{i} % convert windy
        case 'TRUE'
            processedWeatherData(i, 4) = 1;
        case 'FALSE'
            processedWeatherData(i, 4) = 2;
    end
    switch weatherData.Play{i} % convert play
        case 'yes'
            processedWeatherData(i, 5) = 1;
        case 'no'
            processedWeatherData(i, 5) = 2;
    end 
end

% Display processed data
disp(processedWeatherData);

% Save the processed data in text or csv file
writematrix(processedWeatherData, 'data/datprocessed_weather_data.txt');


