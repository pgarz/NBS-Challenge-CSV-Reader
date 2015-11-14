%--------- TRAIN  ---------%

Data = csvread('randomtestdata.csv');

% set X and Y
% n = number of features, m = number of training examples
[m, cols] = size(Data);
n = cols - 1;
X = Data(:, 1:n);
Y = Data(:, cols);

%Statistics Toolbox - Robust Fit
B_rob = robustfit(X,Y, 'welsch');

%Linear Regression
X_plus = [ones(m,1) X]; %adds x_0 (to calculate intercepts)
B_regress = regress(Y,X_plus);

%--------- TEST  ----------%
X_test = X_plus;
Y_actual = Y;

Y_rob = X_test*B_rob;
Error = Y_actual - Y_rob;
avgerror_rob = sum(Error)/length(Error)

Y_regress = X_test*B_regress;
Error = Y_actual - Y_regress;
avgerror_regress = sum(Error)/length(Error)
