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

max_len = min([size(v1,4) size(v2,4) size(v3,4)]);

v1 = v1(:,:,:,1:max_len);
v2 = v2(:,:,:,1:max_len);
v3 = v3(:,:,:,1:max_len);

% Now we can convert to bw
bw1 = rgbvid2grayvid(v1);
bw2 = rgbvid2grayvid(v2);
bw3 = rgbvid2grayvid(v3);
bw = [bw1 bw2 bw3];
%%
% Because the camera is steady, we can identify a bucket's location using 
% the change between two frames. For example
frame = 78;
% This compares the absolute difference between two successive frames
% We scale to find *relatively* prominent features.
diff = imscale(abs(bw1(:,:,frame) - bw1(:,:,frame+1)));
imshow(diff);

%%
% Now some frames don't have significant change, as can be seen between 
% frame=78,79. Thus let's examine the max diff between frames to get some
% insight.

num_frames = max_len;
max_diff = zeros(3,num_frames-1);
for j = 1:(num_frames-1)
    diff = abs(bw1(:,:,j) - bw1(:,:,j+1));
    max_diff(1,j) = max(diff, [], 'all');
    
    diff = abs(bw2(:,:,j) - bw2(:,:,j+1));
    max_diff(2,j) = max(diff, [], 'all');
    
    diff = abs(bw3(:,:,j) - bw3(:,:,j+1));
    max_diff(3,j) = max(diff, [], 'all');
end

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
% The peaks have varying height. Some are quite low, but the duds are very
% obvious, with a value < 50. Fortunately, I don't see any pairs of duds 
% next to each other, so I think I can interpolate between adjacent frames.
bw1_i = int16(zeros(size(bw1)));

threshold = 50;
for j = 2:(num_frames-1)
    if max_diff(1,j) < threshold
        bw1_i(:,:,j) = uint8((uint16(bw1(:,:,j-1)) + uint16(bw1(:,:,j+1)))./2);
    else
        bw1_i(:,:,j) = bw1(:,:,j);
    end
end

%% Let's just look at the changed bw1 and call it a night.
figure(3);
for j = 1:num_frames
    X = imscale(bw1_i(:,:,j));
    imshow(X); drawnow
end



%% Here be dragons below


























































%%
% First challenge: Identify the location of the bucket in a frame.
% My great idea is that I can take a clear snapshot of the bucket, subtract
% the mean value, and convolve. The filter would have positive values where
% the bucket is white, and negative values where the bucket is dark.

% Thought experiment 1: I convolve over a dark region. All image values are
% close to zero, and the feature is close to zero.

% Thought experiment 2: I convolve over a bright region. Some image values
% are bright, and the feature is quite high. Adding dark edges anywhere
% would only reduce the compatibility.

% From thought experiment 2, I conclude the source image should also be
% averaged. It's a good thing our bucket has brights and darks.

% Plan of attack: We will operate on the greyscaled image, but perform our
% analysis after centering about the mean. Get the subimage of the can at
% frame 70, coords (314,333) to (381,430)
% (I may re-center this image about its mean. Don't know. Try later.)
% Then convolve to get the features. I will re-scale this feature to
% (0,255) so that it is like an image again. I will then look at feature as
% an image representing confidence of can location.

% Let's get the can.
% img_f = int16(bw1(:,:,70));
img_f = v1(:,:,3,70);
imshow(img_f);
%%
can = int16(img_f(330:430, 314:381));
figure(2);
imshow(can);

% And normalize our can and our image

img_f = img_f - mean(img_f, 'all');
can_f = can - mean(can, 'all');

%% Now we convolve
result = conv2(img_f,can_f, 'same');
% And rescale to view
result = imscale(result);
figure(1);
imshow(img_f);
figure(2);
imshow(result)

