Data = csvread('randomtestdata.csv');

% set X and Y
% n = number of features, m = number of training examples
[m, cols] = size(Data);
n = cols - 1;
X = Data(:, 1:n);
X = [ones(m,1) X];
Y = Data(:, cols);

X_test = X;
Y_actual = Y;

tau = .8;

Y_predicted = zeros(size(X_test,1),1);
for i = 1:size(X_test, 1)
    xi = X_test(i,:);
    xiMatrix = repmat(xi, m, 1);
    
    W = exp(- sum((xiMatrix - X).^2,2) ./ (2*tau^2));
    W = diag(W);
    Theta = (X'*W*X)\(X'*W*Y);
    Y_predicted(i,1) = xi*Theta;
end

%------ Error ------%
Error = Y_actual - Y_predicted;
avgerror_lwlr = sum(Error)/length(Error)