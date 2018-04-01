%% function calculating the cost function for each theta
function J = cost_function(theta, X, y, lamda)
    [sample_num, ~] = size(X);
    temp = X * theta - y;
    J = 1 / (2 * sample_num) * temp' * temp + lamda * 0.5 * theta' * theta;
end
