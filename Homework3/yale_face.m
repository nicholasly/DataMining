close all;
load('yale_face.mat');
% Task 1
X = X';
[m, n] = size(X);
k = 5;
xmean = mean(X, 1);
xstd = std(X, 1);
for i = 1:m
    X(i,:) = (X(i,:) - xmean) ./ xstd;
end
t = reshape(xmean(1,:),[64,64]);
imshow(t,[]);
imwrite(uint8(t), 'mean.bmp');
Sigma = (1/m) * X' * X;
tic
[U,S,V] = svd(Sigma);
toc
Ureduce = U(:,1:k);
for i = 1:k
    t = reshape(Ureduce(:,i),[64,64]);
    minD = min(t(:));
    maxD = max(t(:));
    mapped_image = (double(t) - minD) ./ (maxD - minD);
    ncmap = size(colormap, 1);
    mapped_image = mapped_image .* ncmap;
    if (ncmap == 2) 
        mapped_image = mapped_image >= 0.5;
    else if (ncmap <= 256)
            mapped_image = uint8(mapped_image);
        else
            mapped_image = uint16(mapped_image);
        end
    end
    figure;
    imshow(t,[]);
    imwrite(mapped_image, strcat(num2str(i),'.bmp'));
end
% Task 2
tic
[Q,D] = eig(Sigma);
d = zeros(1, n);  
for i = 1:n
    d(1,i) = D(i,i);
end
[D,index] = sort(d, 'descend');
toc
z = Q(:,index(1:k));
for i = 1:k
    t = reshape(z(:,i),[64,64]);
    minD = min(t(:));
    maxD = max(t(:));
    mapped_image = (double(t) - minD) ./ (maxD - minD);
    ncmap = size(colormap, 1);
    mapped_image = mapped_image .* ncmap;
    if (ncmap == 2) 
        mapped_image = mapped_image >= 0.5;
    else if (ncmap <= 256)
            mapped_image = uint8(mapped_image);
        else
            mapped_image = uint16(mapped_image);
        end
    end
    figure;
    imshow(t,[]);
    imwrite(mapped_image, strcat(num2str(i),'eigen.bmp'));
end
% Task 3
[U,S,V] = svd(Sigma);
s = zeros(1, n);  
for i = 1:n
    s(1,i) = S(i,i);
end
[S,index] = sort(s, 'descend');
x1 = X(1,:);
x2 = X(2,:);
x3 = X(3,:);

k = 10;
retained_var1 = sum(s(1:k))/sum(s(1:n));
Ureduce = U(:, 1:k);
Y = X * Ureduce;
Y = Y * Ureduce';
y1 = Y(1,:);
y2 = Y(2,:);
y3 = Y(3,:);

k = 100;
retained_var2 = sum(s(1:k) .* s(1:k))/sum(s(1:n) .* s(1:n));
Ureduce = U(:, 1:k);
Z = X * Ureduce;
Z = Z * Ureduce';
z1 = Z(1,:);
z2 = Z(2,:);
z3 = Z(3,:);

figure;
subplot(3,3,1);imshow(reshape(x1,[64,64]), []);
subplot(3,3,2);imshow(reshape(x2,[64,64]), []);
subplot(3,3,3);imshow(reshape(x3,[64,64]), []);
subplot(3,3,4);imshow(reshape(y1,[64,64]), []);
subplot(3,3,5);imshow(reshape(y2,[64,64]), []);
subplot(3,3,6);imshow(reshape(y3,[64,64]), []);
subplot(3,3,7);imshow(reshape(z1,[64,64]), []);
subplot(3,3,8);imshow(reshape(z2,[64,64]), []);
subplot(3,3,9);imshow(reshape(z3,[64,64]), []);
