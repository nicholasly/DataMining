function h = logistic_regression_function(theta, x)
    h = 1 / (1 + exp(-x * theta));
end