function theta = logistic_regression_gradient_descent(X, y, alpha, step, lamda)
    [sample_num, theta_num] = size(X);
    theta = zeros(theta_num, 1);
    temp = zeros(theta_num, 1);
    for i = 1:step
        for j = 1:theta_num
            sum = 0;
            for k = 1:sample_num
                if j == 1
                    sum = sum + (logistic_regression_function(theta, X(k, :)) ...
                    - y(k)) * X(k, j);
                else
                    sum = sum + (logistic_regression_function(theta, X(k, :)) ...
                    - y(k)) * X(k, j) + lamda * theta(j);
                end
                
            end
            temp(j) = theta(j) - alpha * sum / sample_num;
        end
        for j = 1:theta_num
            theta(j) = temp(j);
        end
    end
end
