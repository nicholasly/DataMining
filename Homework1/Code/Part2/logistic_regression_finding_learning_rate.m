function descent = logistic_regression_finding_learning_rate(X, y, alpha, step, lamda)
    alphas = [];
    descent = [];
    [theta_num, ~] = size(X);
    theta = zeros(1, theta_num);
    theta = theta';
    for i = 1:20
        thetas = logistic_regression_gradient_descent(X, y, alpha, step, lamda);
        J(i) = logistic_regression_cost_function(thetas(:,step), X, y);
        alphas(i) = alpha;
        alpha = alpha / 10;
    end
    J = J';
    alphas = alphas';
    descent = [J, alphas];
end