%% Read experiment 1 data
clear all;

experiment = 1;
a = (150)^2;  % controls width of gaussian tunnel vision while tracking

% Load the data
for cam = 1:3
    filename = strcat('cam', string(cam), '_', string(experiment), '.mat');
    load(filename)
end

% Set the starting parameters
switch experiment
    case 1
        start1=1; start2=10; start3=1;
        init = [228 324 179 274 270 320];
        v1 = vidFrames1_1; v2 = vidFrames2_1; v3 = vidFrames3_1;
        exp_title = "Experiment 1 (control)";
    case 2
        start1=1; start2=18; start3=5;
        init = [306 324 125 369 245 363];
        v1 = vidFrames1_2; v2 = vidFrames2_2; v3 = vidFrames3_2;
        exp_title = "Experiment 2 (shaky camera)";
    case 3
        start1=5; start2=27; start3=1;
        init = [283 320 190 336 233 354];
        v1 = vidFrames1_3; v2 = vidFrames2_3; v3 = vidFrames3_3;
        exp_title = "Experiment 3 (swaying oscillator)";
    case 4
        start1=1; start2=2; start3=3;
        init = [273 377 247 241 352 222];
        v1 = vidFrames1_4; v2 = vidFrames2_4; v3 = vidFrames3_4;
        exp_title = "Experiment 4 (rotating & swaying oscillator)";
    otherwise
        disp("Invalid experiment id");
        return;
end

% Trim the data, convert to bw
v1 = v1(:,:,:,start1:end);
v2 = v2(:,:,:,start2:end);
v3 = v3(:,:,:,start3:end);

num_frames = min([size(v1,4) size(v2,4) size(v3,4)]);

v1 = v1(:,:,:,1:num_frames);
v2 = v2(:,:,:,1:num_frames);
v3 = v3(:,:,:,1:num_frames);

% Now we can convert to bw
bw1 = rgbvid2grayvid(v1);
bw2 = rgbvid2grayvid(v2);
bw3 = rgbvid2grayvid(v3);
bw = cat(4, bw1, bw2, bw3);

clear filename;
clear vidFrames* v1 v2 v3;
clear bw1 bw2 bw3;
clear start*;
%% 

frame_size = [size(bw, 1) size(bw, 2)];
big_gauss = ones(frame_size*2);
gauss_f = @(x) exp(-1/a*x^2);
xmax = frame_size(1);
ymax = frame_size(2);
for i = 1:2*xmax
    for j = 1:2*ymax
        big_gauss(i,j) = gauss_f(i-xmax)*gauss_f(j-ymax);
    end
end

location = zeros(6, num_frames);
location(:,1) = init;

for j = 2:num_frames
    for cam = 1:3
        frame = double(bw(:,:,j,cam));
        x0 = location(cam*2-1, j-1);
        y0 = location(cam*2, j-1);
        cone = big_gauss((481-x0):(960-x0), (641-y0):(1280-y0));
        frame = frame .* cone;
        
        [M, I] = max(frame(:));
        [x, y] = ind2sub(frame_size, I);
        location(cam*2-1, j) = x;
        location(cam*2, j) = y;
    end
end

clear i j big_gauss cam cone frame M I;
clear x0 y0 x y xmax ymax gauss_f;
%%
% location is [y1 x1 y2 x2 y3 x3]'
% let's make X [x1 y1 x2 y2 x3 y3]'
X = [location(2,:)
    location(1,:)
    location(4,:)
    location(3,:)
    location(6,:)
    location(5,:)];

[m,n] = size(X);
mn = mean(X,2);
X = X-repmat(mn, 1, n);

[u,s,v] = svd(X'/sqrt(n-1));
lambda = diag(s).^2;

% coefficients of projection onto first component
figure(1);
Y = v'*X;
Y = Y / max(abs(Y), [], 'all');
figure(1)
subplot(2,1,1);
plot((1:n)/20, Y(1,:), ...
     (1:n)/20, Y(2,:), ... 
     (1:n)/20, Y(3,:) );%, ...
     %(1:n)/20, Y(4,:), ...
     %(1:n)/20, Y(5,:), ...
     %(1:n)/20, Y(6,:));
xlabel("time (seconds)");
ylabel("Relative displacement");
set(gca, 'ytick', -1:1);
title("Projection coeffs. onto first three right singular vectors (normalized)", "FontSize", 13);
subplot(2,1,2);
plot(1, lambda(1), '*', ...
         2, lambda(2), '*', ...
         3, lambda(3), '*', ...
         4, lambda(4), '*', ...
         5, lambda(5), '*', ...
         6, lambda(6), '*', 'MarkerSize', 12);
set(gca, 'xtick',1:6, 'FontSize', 13);
xlim([0.5 6.5]);
title("Singular values", 'FontSize', 13);
     
sgtitle(exp_title, 'FontSize', 18);

%saveas(gcf, strcat('fig', string(experiment), '.png'));