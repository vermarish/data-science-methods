%% Clean workspace
clear all; close all; clc;
load ../data/subdata.mat;

%% 
L = 10;  % spatial domain
n = 64;  % Fourier modes (data size)
x2 = linspace(-L,L,n+1); x = x2(1:n); y=x; z=x;
k = (2*pi/(2*L))*[0:(n/2-1) -n/2 : -1];
ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);  % spatial coordinate grid
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);  % spatial frequency-domain grid

%% Let's average our data in the frequency domain

realize = 49; % number of realizations

aveUt = zeros(n,n,n);
for j = 1:realize
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Ut = fftshift(fftn(Un));
    aveUt = aveUt + Ut;
end

absUt = abs(aveUt)/ max(abs(aveUt), [], 'all');

close all, isosurface(Kx,Ky,Kz,absUt,0.30)
axis([-20 20 -20 20 -20 20]), grid on, drawnow

%% I see a 3-D signal being emitted. Let's identify it.
[M,I] = max(aveUt, [], 'all', 'linear');
dim = [n n n];
[x0,y0,z0] = ind2sub(dim,I);

% Our signal has the frequencies:
ks(x0)
ks(y0)
ks(z0)
% In units of m^-1

%% We now want a 3-D Gaussian filter centered at frequencies x,y,z
% After filtering, we can do:
%     signal -> FFT -> filter -> IFFT
% and the submarine should be visible in each sample.

% Build the filter
filter = ones(n,n,n);
tau = 0.4;
fx = @(x) exp(-tau*(x-x0)^2);
fy = @(y) exp(-tau*(y-y0)^2);
fz = @(z) exp(-tau*(z-z0)^2);
fxyz = @(x,y,z) fx(x)*fy(y)*fz(z);
for x = 1:n
    for y = 1:n
        for z = 1:n
            filter(x,y,z) = fxyz(x,y,z);
        end
    end
end

% signal -> FFT -> filter -> IFFT
subdata_f = zeros(49, 64,64,64);  % filtered data (still in spatial domain)
for j = 1:realize
    signal = reshape(subdata(:,j),n,n,n);
    signal = fftn(signal);
    signal = signal .* fftshift(filter);
    signal = ifftn(signal);
    signal = abs(signal) / max(abs(signal), [], 'all');
    subdata_f(j,:,:,:) = signal;
end

%% View a Slideshow
frame_duration = 0.5;  % # of seconds each isosurface appears for
iso_strength = 0.8;
for j=1:realize
    frame(:,:,:) = subdata_f(j,:,:,:);
    frame = abs(frame)/max(abs(frame), [], 'all');

    close all, isosurface(X,Y,Z,frame, iso_strength)
    axis([-20 20 -20 20 -20 20]), grid on, drawnow
    pause(frame_duration)
end

%% View One frame
j=43;  % the frame index
frame(:,:,:) = subdata_f(j,:,:,:);
frame = abs(frame)/max(abs(frame), [], 'all');

close all, isosurface(X,Y,Z,frame, iso_strength)
axis([-20 20 -20 20 -20 20]), grid on, drawnow

%% Get and plot the path of the submarine.
% subdata_f has size 49,64,64,64
path = zeros(49,3);
for j = 1:realize
    [M,I] = max(subdata_f(j,:,:,:), [], 'all', 'linear');
    dim = [64 64 64];
    [xj,yj,zj] = ind2sub(dim, I);
    path(j,:) = [ks(xj), ks(yj), ks(zj)];
end

plot3(path(:,1), path(:,2), path(:,3))
xlabel('x')
ylabel('y')
zlabel('z')
title("Submarine path");
