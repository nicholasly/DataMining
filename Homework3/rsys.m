r = [1 1 0 0 1 1 1 0;
    1 1 0 1 0 0 0 0;
    0 1 1 0 0 1 1 1;
    1 0 1 1 0 1 1 0;
    1 0 1 1 1 0 0 1;
    1 0 1 0 0 1 0 0;
    0 1 0 1 0 1 1 0;];
y = [4 4 0 0 1 1 5 0;
    5 5 0 1 0 0 0 0;
    0 4 1 0 0 1 5 4;
    5 0 2 5 0 1 2 0;
    1 0 5 4 5 0 0 1;
    1 0 5 0 0 4 0 0;
    0 1 0 5 0 5 1 0;];
[m,n] = size(r);
maxIter = 100000000;
lamda = 0.1;
alpha = 0.01;
c = abs(randn(1,1));
x = zeros(7,4);
theta = zeros(8,4);
x(:) = c;
theta(:) = c;
%epsilon = 1e-7;
for iter = 1:maxIter
   predict = x * theta';
   loss = (predict - y) .* r;
   x_grad = loss * theta + lamda .* x;
   theata_grad = loss' * x + lamda .* theta;
   x = x - alpha .* x_grad;
   theta = theta  - alpha .* theata_grad;
   temp = loss.^ 2;  
   %J = sum(sum(temp(r == 1)))/2 + lamda/2 .* sum(sum(theta.^2)) + ...
       %lamda/2 .* sum(sum(x.^2));
   %if (J < epsilon)
    %   break;
   %end
end
loss = sum(sum(((x * theta' - y) .* r) .^ 2));
