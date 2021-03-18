close all; clear all;
[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
[images_test, labels_test] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
%% 
len = size(images, 1) * size(images, 2); % # of pixels in an img
n = size(images, 3); % # of imgs

Xbig = double(reshape(images, len, n));
% each observation is a column in X, an element of R^{len}

% doing SVD would require 26.8 GB memory. Let's subsample.
% Let's make the sampling deterministic for easier replication.
rng(482); % param
entries = randperm(n);
pca_sample_factor = 6; % param
num_samples = floor(n/pca_sample_factor);
entries = entries(1:num_samples);

X = zeros(len, num_samples);
for j = 1:num_samples
    i = entries(j);
    X(:,j) = Xbig(:,i);
end

%% 
% Let's try visually identifying how many modes it takes, without the PCA
%  method of zero-mean.
[U,S,V] = svd(X, 'econ');

figure(1)
subplot(2,4,8);
imshow(uint8(reshape(X(:,1), [28 28])));
title("True image", "FontSize", 16);

%modes = [inf 600 300 100 50 25 10 5];
modes = [5 10 25 50 100 300 600 inf];
for j = 1:7
    m = modes(j);
    subplot(2,4,j);
    approx = U(:,1:m) * S(1:m, 1:m) * V(:,1:m)';
    imshow(uint8(reshape(approx(:,1), [28 28])));
    title(strcat(int2str(m), " modes"), "FontSize", 16);
end

set(gcf, 'Position', [600 500 750 400]);
saveas(gcf, 'approximations.png');

clear approx;
%% Now let's do PCA
% Center the pixel values on each row
mn = mean(X,2);
X = X - repmat(mn, 1, num_samples);

[U,S,V] = svd(X'/sqrt(num_samples-1), 'econ');
lambda = diag(S).^2;
%clear U X; % we only need the projection vectors V and the singular values


figure(2)
% Let's also look at what the modes themselves are, the eigen-digits.
for i = 1:10
    subplot(2,5,i);
    digit = reshape(V(:,i), [28 28]);
    s = pcolor(digit);
    s.EdgeColor = 'none';
    title(int2str(i), 'FontSize', 16);
    ax = gca;
    ax.YTick = [];
    ax.XTick = [];
end

set(gcf, 'Position', [600 500 750 400]);
saveas(gcf, 'eigendigits.png');



% Plot the singular values
figure(3)
subplot(1,2,1);
semilogy(1:25, lambda(1:25), 'x', 26:len, lambda(26:end), 'x');
title("Singular values of X", 'FontSize', 12);
xlabel("index", 'FontSize', 12);
ylabel("value (logarithmic scale)", 'FontSize', 12);

subplot(1,2,2);
semilogy(1:25, lambda(1:25), 'x');
ylim([lambda(25)*0.9 lambda(1)*1.1]);
xlabel("index", 'FontSize', 12);
ylabel("value (logarithmic scale)", 'FontSize', 12);
title("First 25 singular values", 'FontSize', 12);

set(gcf, 'Position', [600 500 900 250]);
saveas(gcf, 'singular_values.png');

% There are 677 "non-zero" singular values in this set. That means the 
% pixels are in a space of about R^674.

%% Let's try plotting all images onto three projection coefficients
cols = [1 2 3]; % param
coeffs = zeros(n, 4);  % three coeffs and the label
v1 = V(:,cols(1))';
v2 = V(:,cols(2))';
v3 = V(:,cols(3))';
for j=1:n
    img = Xbig(:,j);
    coeffs(j, 1) = v1*img;
    coeffs(j, 2) = v2*img;
    coeffs(j, 3) = v3*img;
    coeffs(j, 4) = labels(j);
end

X = sortrows(coeffs, 4);

first = zeros(10,1);
last = zeros(10,1);

for digit = 0:9
    rows = X(:,4) == digit;
    first(digit+1) = find(rows, 1, 'first');
    last(digit+1) = find(rows, 1, 'last');
end

figure(4);
newcolors = {'#67E','#F70','#E44','#5AA','#182','#EC0','#C1E','#964','#1D3','#FBF'};
colororder(newcolors)
symb = '*';
plot3(X(first(1):last(1),1), X(first(1):last(1),2), X(first(1):last(1),3), symb, ...
      X(first(2):last(2),1), X(first(2):last(2),2), X(first(2):last(2),3), symb, ...
      X(first(3):last(3),1), X(first(3):last(3),2), X(first(3):last(3),3), symb, ...
      X(first(4):last(4),1), X(first(4):last(4),2), X(first(4):last(4),3), symb, ...
      X(first(5):last(5),1), X(first(5):last(5),2), X(first(5):last(5),3), symb, ...
      X(first(6):last(6),1), X(first(6):last(6),2), X(first(6):last(6),3), symb, ...
      X(first(7):last(7),1), X(first(7):last(7),2), X(first(7):last(7),3), symb, ...
      X(first(8):last(8),1), X(first(8):last(8),2), X(first(8):last(8),3), symb, ...
      X(first(9):last(9),1), X(first(9):last(9),2), X(first(9):last(9),3), symb, ...
      X(first(10):last(10),1), X(first(10):last(10),2), X(first(10):last(10),3), symb);
%title(strcat("Projection onto first three right singular vectors"))
legend("0","1","2","3","4","5","6","7","8","9");

xlabel("$\vec{v}_1$", 'Interpreter', 'latex', 'FontSize', 14);
ylabel("$\vec{v}_2$", 'Interpreter', 'latex', 'FontSize', 14);
zlabel("$\vec{v}_3$", 'Interpreter', 'latex', 'FontSize', 14);

set(gcf, 'Position', [600 500 600 400]);
saveas(gcf, '10projections.png');

%% Prepare for ML methods
proj = V(:,1:20);

img_train = proj' * Xbig;
label_train = labels;
img_test = proj' * double(reshape(images_test, [784 10000]));
label_test = labels_test;

%% LDA2
errors = zeros(10);
for a = 0:8
    for b = (a+1):9
        [performance, w, threshold] = lda2(a, b, img_train, label_train, img_test, label_test, false);
        errors(a+1, b+1) = performance;
        errors(b+1, a+1) = performance;
    end
end

% Worst is 7 and 9, 6.09%
%          4 and 9, 6.07%
% Best is 0 and 1, 0.19%
%         0 and 4, 0.20%

%% LDA3
errors = zeros(10,10,10);
for a = 0:7
    for b = (a+1):8
        for c = (b+1):9
            % abc acb bac bca cab cba
            try 
                [performance, w, threshold] = lda3(a, b, c, img_train, label_train, img_test, label_test, false);
            catch ME
                disp(strcat("Didn't work for", ...
                    int2str(a), ...
                    int2str(b), ...
                    int2str(c)))
            end
            errors(a+1, b+1, c+1) = performance;
        end
    end
end

% best is 0, 1, 7 with 8.62% error

%%
close all;
figure(5);

subplot(1,2,1);
a = 0; b = 1;
[performance, w, threshold] = lda2(a, b, img_train, label_train, img_test, label_test, true);
%title(strcat(int2str(a), " and ", int2str(b)), 'FontSize', 13);
ax = gca;
l = ax.Legend;
l.Position = [0.1454 0.7658 0.1422 0.1325];
l.FontSize = 10;

subplot(1,2,2);
a = 7; b = 9;
[performance, w, threshold] = lda2(a, b, img_train, label_train, img_test, label_test, true);
%title(strcat(int2str(a), " and ", int2str(b)), 'FontSize', 13);
ax = gca;
l = ax.Legend;
l.FontSize = 10;

set(gcf, 'Position', [600 500 950 400]);
saveas(gcf, 'bestworst.png');

%%
close all;
figure(6);

subplot(1,2,1);
a = 0; b = 1; c=7;
[performance, w, threshold] = lda3(a, b, c, img_train, label_train, img_test, label_test, true);
%title(strcat(int2str(a), " and ", int2str(b)), 'FontSize', 13);
ax = gca;
l = ax.Legend;
l.Position = [0.1454 0.7658 0.1422 0.1325];
l.FontSize = 10;

subplot(1,2,2);
a = 2; b = 3; c=6;
[performance, w, threshold] = lda3(a, b, c, img_train, label_train, img_test, label_test, true);
%title(strcat(int2str(a), " and ", int2str(b)), 'FontSize', 13);
ax = gca;
l = ax.Legend;
l.FontSize = 10;

set(gcf, 'Position', [600 500 950 400]);
saveas(gcf, 'bestworsttriple.png');



%%
tree = fitctree(img_train', label_train, 'MaxNumSplits', 10, 'CrossVal', 'on');
%view(tree.Trained{1},'Mode','graph');
%classError = kfoldLoss(tree);

error = evaluateClassifier(tree, img_test', label_test)
%%
n = 1000;
Mdl = fitcsvm(img_train(:,1:n)', label_train(1:n));
result = predict(Mdl,img_test');
score = sum(result == label_test);
error = score / n;