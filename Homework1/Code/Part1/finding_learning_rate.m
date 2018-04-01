%% function finding the appropriate learning rate
function [Js, descent] = finding_learning_rate(X, y, alpha, step)
    Js = [];
    alphas = [];
    descent = [];
    [~,theta_num] = size(X);
    theta = zeros(theta_num, 1);
    initial = cost_function(theta, X, y, 0);
    for j = 1:100
        thetas = gradient_descent(X, y, alpha, step);
        steps = ones(1,step);
        for i = 1:step
            steps(i) = i;
            J(i) = cost_function(thetas(:,i), X, y, 0);
        end
        % difference
        descent = [descent, initial - J];
        Js = [Js, J];
        alphas = [alphas, alpha];
        alpha = alpha / 10;
    end
    descent = descent';
    Js = Js';
    alphas = alphas';
    descent = [alphas, descent];
end