%% function doing gradient descent
function thetas = gradient_descent(X, y, alpha, step)
    % initialization
    [sample_num, theta_num] = size(X);
    theta = zeros(theta_num, 1);
    temp = zeros(theta_num, 1);
    thetas = [];
    % gradient descent
    for i = 1:step
        for j = 1:theta_num
            sum = 0;
            for k = 1:sample_num
                sum = sum + (X(k, :) * theta - y(k)) * X(k, j);
            end
            temp(j) = theta(j) - alpha * sum / sample_num;
        end
        for j = 1:theta_num
            theta(j) = temp(j);
        end
        thetas = [thetas, theta];
    end
end
