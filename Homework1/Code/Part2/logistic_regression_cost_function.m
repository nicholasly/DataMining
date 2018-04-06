function J = logistic_regression_cost_function(theta, X, y)
    [~, sample_num] = size(X);
    sum = 0;
    for k = 1:sample_num
        temp = y(k) * log(logistic_regression_function(theta, X(:, k))) ...
            + (1 - y(k)) * log(1 - logistic_regression_function(theta, X(:, k)));
        sum = sum + temp;
    end
    J = -sum / sample_num;
end