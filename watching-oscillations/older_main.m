% For experiment 1, the can light is clearly visible

% For experiment 2, the can light is clearly visible, but with lots of
% shaking. We're going to see how well the previous method works.

% For experiment 3, the light rotates out of frame in camera 1.
%                   in camera 2, the light sways between a blue/white
%                   background

% For experiment 4, the light is rotating all over the place.
%% Read experiment 1 data
clear all;
load('cam1_1.mat')
load('cam2_1.mat')
load('cam3_1.mat')
%% Load these three vars with the particular experiment you want 
v1 = vidFrames1_1;
v2 = vidFrames2_1;
v3 = vidFrames3_1;

% You MUST sync the starting frame as well
% Frame 12 for 1_1
% Frame 21 for 2_1
% Frame 12 for 3_1
start1 = 12;
start2 = 21;
start3 = 12;

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
%%
% Because the camera is steady, we can identify a bucket's location using 
% the change between two frames. For example
frame = 158;
% This compares the absolute difference between two successive frames
% We scale to find *relatively* prominent features.
diff = imscale(abs(bw(:,:,frame,1) - bw(:,:,frame+1,1)));
clear frame;
%imshow(diff);

% So let's calculate the diffs
% Some frames don't have significant change, so let's also examine the max 
% diff between frames to get some insight.

diffs = zeros(size(bw), 'uint8');
diffs = diffs(:,:,1:(num_frames-1),:);
max_diff = zeros(3,num_frames-1);

for j = 1:(num_frames-1)
    for cam = 1:3
        curr = int16(bw(:,:,j,cam));
        next = int16(bw(:,:,j+1,cam));
        diff = uint8(abs(next-curr));
        
        diffs(:,:,j,cam) = diff;
        max_diff(cam, j) = max(diff, [], 'all');
    end
end
clear curr next;


figure(2);
sgtitle("Max absolute difference between successive frames");
subplot(2,2,1);
plot(1:(num_frames-1), max_diff(1,1:(num_frames-1)));
title('Perspective 1');
subplot(2,2,2);
plot(1:(num_frames-1), max_diff(2,1:(num_frames-1)));
title("Perspective 2");
subplot(2,2,3);
plot(1:(num_frames-1), max_diff(3,1:(num_frames-1)));
title("Perspective 3");
ylim([0 210]);


%%
% TODO if I notice jags in the resulting motion tracking, it's probably
% the thing being distracted on other features of the bucket. I will remedy
% by applying a Gaussian centered at (x_j, y_j) onto frame j+1 before
% identifying (x_{j+1}, y_{j+1}). But until then...

%% Let's try getting the coordinates
threshold = 50;  % We will need to interpolate for the junk frames
a = 200^2;  % to control the width of the last-known-gaussian filter

% We will store coordinates as [x_a y_a x_b y_b x_c y_c]'

location = zeros(6, num_frames-1, 'double');  % type double because it's 
                                              % gonna go through PCA anyway
                                              
diffs = double(diffs);
frame_size = [size(bw, 1) size(bw, 2)];

big_gauss = ones(frame_size*2);
gauss_f = @(x) exp(-1/a*x^2);
xmax = frame_size(1);
ymax = frame_size(2);
for a = 1:2*xmax
    for b = 1:2*ymax
        big_gauss(a,b) = gauss_f(a-xmax)*gauss_f(b-ymax);
    end
end

diff_f = zeros(size(diffs));

% Keep running track of the bucket's current location
x0 = -1;
y0 = -1;
for j = 1:num_frames-1
    for cam = 1:3
        diff = diffs(:,:,j,cam);
        
        % Let's put a Gaussian over the last known appearance
        if x0 ~= -1
            gauss_filter = big_gauss((481-x0):(960-x0), (641-y0):(1280-y0));
            diff = diff.*gauss_filter;
        end
        
        % diff indicates regions of change
        % diff doesn't recognize brightness of light
        % regions of interest <- bright regions of change
        diff = diff .* double(bw(:,:,j,cam));
        diff = diff / max(double(bw(:,:,j,cam)), [], 'all');
        
        % maybe i need to blur so that pants outlines are less important
        % and bigger regions are of interest
        diff = imgaussfilt(diff, 4);
        
        diff_f(:,:,j,cam) = diff;
        
        [M, I] = max(diff(:));
        if M < threshold
            x = -1;
            y = -1;
        else
            [x, y] = ind2sub(frame_size, I);
            x0 = x;  % for next gauss filter
            y0 = y; 
        end
        location(cam*2 - 1, j) = x;  % store it
        location(cam*2, j) = y;
    end
end

% Junk frames have been marked as x_i, y_i = -1, -1. Let's interpolate
for j = 1:num_frames - 1
    for cam = 1:3
        if location(cam*2, j) == -1
            if j == 1  % can't interpolate location at the first diff
                x_new = location(cam*2-1, j+1);
                y_new = location(cam*2, j+1);
            elseif j == num_frames - 1  % or the last diff
                x_new = location(cam*2-1, j-1);
                y_new = location(cam*2, j-1);
            else
                x_new = mean([location(cam*2-1, j-1) location(cam*2-1, j+1)]);
                y_new = mean([location(cam*2, j-1) location(cam*2, j+1)]);
            end
            location(cam*2-1, j) = x_new;
            location(cam*2, j) = y_new;
        end
    end
end

figure(3);
plot(1:(num_frames-1), location(1,:));

% Let's just try SVD
A = location;
[m,n] = size(A);
mn = mean(location, 2);
A = A - repmat(mn, 1, n);

A = location' / sqrt((size(location, 2) - 1));
[U,S,V] = svd(A);
lambda = diag(S).^2  % variances
proj = V*A';


X = location;
[m,n] = size(X);
mn = mean(X,2);
X = X-repmat(mn, 1, n);

[u,s,v] = svd(X'/sqrt(n-1));
lambda = diag(s).^2;
Y=v'*X;

figure(1)
row = 1;
plot(1:(num_frames-1), Y(row,:), 1:(num_frames-1), X(row,:))

%%
% A is short and wide. A' is tall and skinny.
% Then U is huge
plot(1:(num_frames-1), proj(1,:), 1:(num_frames-1), location(1,:));

%%
frame = 97;
x0 = 302;
y0 = 322;
img = diffs(:,:,frame,2);
img = img .* big_gauss((481-x0):(960-x0), (641-y0):(1280-y0));
img = imgaussfilt(img, 4);

imshow(imscale(img));

