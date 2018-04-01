%% function using normal equation to find theta for X and y
function theta = finding_solution_using_normal_equation(X, y, lamda)
    [~,theta_num] = size(X);
    regulation_matrix = eye(theta_num);
    regulation_matrix(1, 1) = 0;
    theta = inv(X' * X + lamda * regulation_matrix) * X' * y;
end
