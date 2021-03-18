%% load data
clear all;

monte_carlo_low = 'data/monte_carlo_low.mp4';
monte_carlo_high = 'data/monte_carlo.mov';
ski_drop_low = 'data/ski_drop_low.mp4';

% params
choice = ski_drop_low;
portion = 1;

v = VideoReader(choice);
n = portion*v.NumFrames;

n = floor(n);
frames = read(v, [1 n]);
dim_h = size(frames, 1);
dim_w = size(frames, 2);
Xdup = zeros(dim_h*dim_w, n);

% convert rgb to grayscale
for j = 1:n
    frame = rgb2gray(frames(:,:,:,j));
    Xdup(:,j) = double(reshape(frame, [dim_h*dim_w 1]));
end
clear frames frame;

% remove duplicate frames
k = 1;
X = [];
maxdiffs = [];
for j = 1:n-1
    maxdiff = max(abs(Xdup(:,j)-Xdup(:,j+1)), [], 'all') ;
    maxdiffs(1,j) = maxdiff;
    if maxdiff > 15
        X(:,k) = Xdup(:,j);
        k = k + 1;
    end
end
n = k-1;
clear k maxdiff Xdup

figure(1);
stem(1:length(maxdiffs), maxdiffs, 'x'); 

X1 = X(:,1:end-1);
X2 = X(:,2:end);
dt = 1/v.FrameRate;

% X has columns of data [x1 x2 x3 ... xn] representing states
% I hypothesize that we can use a linear state transformation, a Koopman
% operator A such that X ~= [x1 Ax1 AAx1 ... A^(n-1)x1]
% We form S from X1,X2, and note that the same operations, when applied
%    to A, result in S = U'*A*U, so these matrices are similar.
% If y is an eigenvector of S, then Uy is an eigenvector of A.
% Thus we can get the eigenvectors of A, or the DMD modes, and their
% eigenvalues. 
[U,Sigma,V] = svd(X1, 'econ');
S = U'*X2*V*diag(1./diag(Sigma));

clear X2 Sigma V
[evecs, evals] = eig(S); 
mu = diag(evals);  % the eigenvalues
omega = log(mu)/dt;
phi = U*evecs; % the eigenvectors of A

% Now each x(t) is representated in a basis of phi_k * exp(omega_k*t)
% What are the linear combination coefficients? Notice that at t=0, we have
% the first state x1 and the exponential cancels. Thus x1 = phi_k*b.
% Let's use the pseudoinverse!
b = phi \ X1(:,1);
disp("Calculated b");
clear U S X1 

%% Did we do well with the DMD deconstruction?
figure(2);
subplot(1,2,1);
plot(real(mu), imag(mu), 'r.', 'Markersize', 15);
xlabel('Re(\mu)', 'FontSize', 13);
ylabel('Im(\mu)', 'FontSize', 13);
xline(0, 'k');
yline(0, 'k');
title("\mu_k", 'FontSize', 13);

subplot(1,2,2);
plot(real(omega), imag(omega), 'r.', 'Markersize', 15);
xlabel('Re(\omega)', 'FontSize', 13);
ylabel('Im(\omega)', 'FontSize', 13);
xline(0, 'k');
yline(0, 'k');
title("\omega_k", 'FontSize', 13);

set(gcf, 'Position', [600 500 800 350]);
saveas(gcf, 'fig_2.png');
%% Now let's build our DMD reconstructions.
modes = zeros(length(b), n);
for iter = 1:n
    t = (iter-1)*dt;
    modes(:,iter) = b .* exp(omega*t);
end

inds = find(abs(omega) < 1e-1);
background = zeros(dim_h*dim_w,n);
for i = 1:length(inds)
    j = inds(i);
    background = background + phi(:,j)*modes(j,:);
end
background = abs(background);

foreground = X(:,1:end-1) - abs(background(:,end-1));
R = foreground .* (foreground < 0);

foreground = foreground - 0*R;
background = background(:,1:n-1) + 0*R;

disp("Launching movie player...");


vid_orig = reshape(mat2gray(X(:,1:n-1)), [dim_h dim_w n-1]);
vid_back = reshape(mat2gray(background), [dim_h dim_w n-1]);
vid_fore = reshape(mat2gray(foreground), [dim_h dim_w n-1]);

% delete(findall(0));
% implay(vid_orig);
% implay(vid_back);
% implay(vid_fore);
%%
inds = floor(linspace(1,n,6));
inds = inds(2:end-1);
figure(3);

for j = 1
    ind = inds(j+1);
    
    subplot_tight(3,3,j*3-2, [0.01]);
    imshow(vid_orig(:,:,ind));
    title("Original video", "FontSize", 13);
    
    subplot_tight(3,3,j*3-1, [0.01]);
    imshow(vid_back(:,:,ind));
    title("Background video", "FontSize", 13);
    
    subplot_tight(3,3,j*3, [0.01]);
    imshow(vid_fore(:,:,ind));
    title("Foreground video", "FontSize", 13);
end

for j = 2:3
    ind = inds(j+1);
    
    subplot_tight(3,3,j*3-2, [0.01]);
    imshow(vid_orig(:,:,ind));
    
    subplot_tight(3,3,j*3-1, [0.01]);
    imshow(vid_back(:,:,ind));
    
    subplot_tight(3,3,j*3, [0.01]);
    imshow(vid_fore(:,:,ind));
end

set(gcf, 'Position', [2203 168 744 683]);
saveas(gcf, 'fig_3.png');

% background is the DMD mode that does not change in time...
% how'd you pick the rank?
% Why are we interested in the omegas close to zero? Because e^0=1 and that
% corresponds to the mode which does not change in time

