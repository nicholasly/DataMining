clear;
%% data mean nornalization
X = [87 89 89 92 93; 72 76 74 71 76; 83 88 82 91 89; 90 93 91 89 94];
x_mean = mean(X');
x_s = std(X');
for i = 1:4
    norm_X(i,:) = (X(i,:) - x_mean(i)) / x_s(i);
end
norm_X = norm_X';
temp = ones(5,1);
norm_X = [temp norm_X];
X = X';
temp = ones(5,1);
X = [temp X];
y = [89	91 93 95 97];
y = y';
%% Problem 1
fprintf('\nProblem 1\n');
theta = gradient_descent(norm_X, y, 1, 1);
fprintf('The theta after the first iteration is\n');
display(theta);
%% Problem 2
fprintf('\nProblem 2\n');
initial_theta = zeros(5,1);
J0 = cost_function(initial_theta, norm_X, y, 0);
J1 = cost_function(theta, norm_X, y, 0);
fprintf('The cost before the first iteration is\n');
display(J0);
fprintf('The cost after the first iteration is\n');
display(J1);
%% Problem 3
fprintf('\nProblem 3\n');
% test begin with 100 and 300
[Js1, descent1] = finding_learning_rate(norm_X, y, 100, 1);
[Js2, descent2] = finding_learning_rate(norm_X, y, 300, 1);
%% Problem 4
fprintf('\nProblem 4\n');
theta_norm = finding_solution_using_normal_equation(X, y, 0);
fprintf('The theta obtained by normal equation is\n');
display(theta_norm);
testX = [88 73 87 92];
testX = [1 testX];
predict_y = testX * theta_norm;
fprintf('The math grade is\n');
display(predict_y);
predict_J = cost_function(theta_norm, norm_X, y, 0);
fprintf('The cost is\n');
display(predict_J);
%% Problem 5
fprintf('\nProblem 5\n');
theta_norm_la = finding_solution_using_normal_equation(X, y, 1);
fprintf('The theta obtained by normal equation is\n');
display(theta_norm_la);
predict_y_la = testX * theta_norm_la;
fprintf('The math grade is\n');
display(predict_y_la);
predict_J_la = cost_function(theta_norm_la, norm_X, y, 0);
fprintf('The cost is\n');
display(predict_J_la);