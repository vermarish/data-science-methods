figure(1)
[y_gnr, Fs_gnr] = audioread('GNR.m4a');
y_gnr = y_gnr/max(abs(y_gnr));
tr_gnr=length(y_gnr)/Fs_gnr; % song duration in seconds
plot((1:length(y_gnr))/Fs_gnr,y_gnr);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Sweet Child O Mine intro');
% p8 = audioplayer(y_gnr, Fs_gnr); play(p8);

figure(2)
[y_pf, Fs_pf] = audioread('Floyd.m4a');
y_pf = y_pf/max(abs(y_pf));
y_pf = y_pf(1:end-1);
tr_pf=length(y_pf)/Fs_pf; % song duration in seconds
plot((1:length(y_pf))/Fs_pf,y_pf);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Comfortably Numb solo');
% p8 = audioplayer(y_pf, Fs_pf); playblocking(p8);

y_gnr = cast(y_gnr, 'single');
y_pf = cast(y_pf, 'single');

fontsize=20;

%% Let's score Guns N Roses

figure(3);
frequency_step = 8;  % only record every step-th bin to save memory
song_portion=1;  % portion of the song to display
a=150;
num_gabors = 200;

threshold = 3;  % if max < threshold at some time, we will record a rest
max_freq = 20000;  % to avoid higher notes while scoring

Fs = Fs_gnr;
y = y_gnr;
tr = tr_gnr;

y_f = gnrlowpass48(y);
% p8 = audioplayer(y_f, Fs); play(p8);

track = y_f(1:floor(end*song_portion));
sz = length(track);
tau = linspace(0, tr*song_portion, num_gabors);  % gabor centers, unit seconds

t = (1:sz)/Fs; % song sample time-values
t = cast(t, 'single');
Sgt_spec = cast(zeros(floor(sz/frequency_step),num_gabors), 'single');

gauss = @(t, tau) exp(-a*(t-tau).^2);
% Perform gabor transform, store in Sgt_spec
for j = 1:num_gabors
    g = gauss(t, tau(j));
    g = cast(reshape(g,sz,1), 'single');
    Sg = g.*track;
    Sgt = fft(Sg);
    Sgt = Sgt(1:frequency_step:end);
    % % (un)-comment the line below to resolve off-by-one errors
    Sgt = Sgt(1:end-1);
    Sgt_spec(:,j) = cast(fftshift(log(1+abs(Sgt))), 'single');
end
clear Sgt Sg g;

% Display spectrogram
subplot(2,1,1);
ks = linspace(-Fs/2, Fs/2, sz/frequency_step);
pcolor(tau, ks, Sgt_spec)
shading interp
set(gca, 'ylim', [200, 900], 'Fontsize', fontsize)
colormap(hot)
%colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
% title("First " + floor(song_portion*tr) + " seconds with " + num_gabors + " a = " + a + " STFTs");
title("Spectrogram of Sweet Child O Mine's intro. Computed with " + num_gabors + " STFTs using a=" + a);

% Let's get the notes.
bandwidth = (length(ks)/2)*max_freq/(Fs/2); % half bandwidth, measured as # of bins below max_freq
% for LPF centered at 0, to avoid transcribing higher notes
minbin = floor(length(ks)/2 - bandwidth);
maxbin = floor(length(ks)/2 + bandwidth);

freqs = zeros(1, num_gabors);
for j = 1:num_gabors
    [M,I] = max(Sgt_spec(minbin:maxbin,j)); I = I + minbin;
    % [M,I] = max(Sgt_spec(:,j));
    if M > threshold
        freqs(j) = abs(ks(I));
    else
        freqs(j) = 0;
    end
end
subplot(2,1,2);
stairs((1:length(freqs))*(song_portion*tr)/length(freqs), freq2midi(freqs));
xlabel("Time (s)");
ylabel("MIDI note");
title("MIDI score");
set(gca, 'Fontsize', fontsize);



%% Let's isolate the Floyd bassline. Let's start with LPF above 250 Hz
y_pf_f = y_pf;
y_pf_f = lowpassbass(y_pf_f);

sz = size(y_pf);
sz = sz(1);
ks = linspace(-Fs_pf/2, Fs_pf/2, sz);
% a=3e-6 sounds best, but too many overtones for analysis
a=1e-6; 
gauss = @(k, f0) exp(-a*(k-f0).^2);
%p8 = audioplayer(y_pf_f, Fs_pf); play(p8);


freqs = [125 110 100 80]./2;
filter = zeros(1,length(y_pf_f));
for j = 1:length(freqs)
    freq = freqs(j);
    for order = 1:20
        filter = filter + 1/sqrt(order)*(gauss(ks, order*freq) + gauss(ks, -order*freq));
    end
end
filter = filter/max(filter);
filter = reshape(filter, sz, 1);

fy = fft(y_pf_f);
fy = fftshift(filter) .* fy;
y_pf_f = real(ifft(fy));

% y_pf_f = lowpass48(y_pf_f);

p8 = audioplayer(real(y_pf_f), Fs_pf); play(p8);

%% Now let's get a spectrogram and score the bass in y_pf.
figure(3);
frequency_step = 16;  % only record every step-th bin to save memory
song_portion=1;  % portion of the song to display
a=50;
num_gabors = 150;

Fs = Fs_pf;
y = y_pf_f;
tr = tr_pf;

threshold = 3;  % if max < threshold at some time, we will record a rest
max_freq = 130;  % to avoid higher notes while scoring

gauss = @(t, tau) exp(-a*(t-tau).^2);
track = y(1:floor(end*song_portion));
sz = length(track);
tau = linspace(0, tr*song_portion, num_gabors);  % gabor centers, unit seconds

t = (1:sz)/Fs; % song sample time-values
t = cast(t, 'single');
Sgt_spec = cast(zeros(floor(sz/frequency_step),num_gabors), 'single');

% Perform gabor transform, store in Sgt_spec
for j = 1:num_gabors
    g = gauss(t, tau(j));
    g = cast(reshape(g,sz,1), 'single');
    Sg = g.*track;
    Sgt = fft(Sg);
    Sgt = Sgt(1:frequency_step:end);
    % % (un)-comment the line below to resolve off-by-one errors
    % Sgt = Sgt(1:end-1);
    Sgt_spec(:,j) = cast(fftshift(log(1+abs(Sgt))), 'single');
end
clear Sgt Sg g;

% Display spectrogram
subplot(2,1,1);
ks = linspace(-Fs/2, Fs/2, sz/frequency_step + 1);
ks = ks(1:end-1);
pcolor(tau, ks, Sgt_spec)
shading interp
set(gca, 'ylim', [20, 150], 'Fontsize', fontsize)
colormap(hot)
% colorbar
xlabel('time (t)'), ylabel('frequency (Hz)');
title("Spectrogram of Comfortably Numb's solo bassline. Computed with " + num_gabors + " STFTs using a="+a)

% Let's get the notes.
bandwidth = (length(ks)/2)*max_freq/(Fs/2); % half bandwidth, measured as # of bins below max_freq
% for LPF centered at 0
minbin = floor(length(ks)/2 - bandwidth);
maxbin = floor(length(ks)/2 + bandwidth);

notes = strings(1,num_gabors);
freqs = zeros(1, num_gabors);
for j = 1:num_gabors
    [M,I] = max(Sgt_spec(minbin:maxbin,j)); I = I + minbin;
    % [M,I] = max(Sgt_spec(:,j));
    if M > threshold
        freqs(j) = abs(ks(I));
    else
        freqs(j) = 0;
    end
end
subplot(2,1,2);
stairs((1:length(freqs))*(song_portion*tr_pf)/length(freqs), freq2midi(freqs));
xlabel("Time (s)");
ylabel("MIDI note");
ylim([39 48]);
title("MIDI score");
set(gca, 'Fontsize', fontsize);


%% And now let's try the Floyd solo
y = y_pf;
Fs = Fs_pf;
tr = tr_pf;
figure(4);
frequency_step = 16;  % only record every step-th bin to save memory
song_portion=0.25;  % portion of the song to display
a=50;
num_gabors = 200;

threshold = 4.2;  % if max < threshold at some time, we will record a rest
max_freq = 10000;  % to avoid higher notes while scoring

y_f = soloFilter(y);

% track = y_f(1:floor(end*song_portion));
track = y_f(floor(end*(1-song_portion)):end);
sz = length(track);
p8 = audioplayer(track, Fs); play(p8);
tau = linspace(0, tr*song_portion, num_gabors);  % gabor centers, unit seconds

t = (1:sz)/Fs; % song sample time-values
t = cast(t, 'single');
Sgt_spec = cast(zeros(floor(sz/frequency_step),num_gabors), 'single');

gauss = @(t, tau) exp(-a*(t-tau).^2);
% Perform gabor transform, store in Sgt_spec
for j = 1:num_gabors
    g = gauss(t, tau(j));
    g = cast(reshape(g,sz,1), 'single');
    Sg = g.*track;
    Sgt = fft(Sg);
    Sgt = Sgt(1:frequency_step:end);
    % % (un)-comment the line below to resolve off-by-one errors
    Sgt = Sgt(1:end-1);
    Sgt_spec(:,j) = cast(fftshift(log(1+abs(Sgt))), 'single');
end
clear Sgt Sg g;

% Display spectrogram
subplot(2,1,1);
ks = linspace(-Fs/2, Fs/2, sz/frequency_step);
pcolor(tau+45, ks, Sgt_spec)
shading interp
set(gca, 'ylim', [350, 1200], 'Fontsize', fontsize)
colormap(hot)
%colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
title("Spectrogram of Comfortably Numb's guitar solo. Computed with " + num_gabors + " STFTs using a=" + a);

% Let's get the notes.
bandwidth = (length(ks)/2)*max_freq/(Fs/2); % half bandwidth, measured as # of bins below max_freq
% for LPF centered at 0, to avoid transcribing higher notes
minbin = floor(length(ks)/2 - bandwidth);
maxbin = floor(length(ks)/2 + bandwidth);

notes = strings(1,num_gabors);
freqs = zeros(1, num_gabors);
for j = 1:num_gabors
    [M,I] = max(Sgt_spec(minbin:maxbin,j)); I = I + minbin;
    % [M,I] = max(Sgt_spec(:,j));
    if M > threshold
        freqs(j) = abs(ks(I));
    else
        freqs(j) = 0;
    end
end

subplot(2,1,2);

stairs((1:length(freqs))*(song_portion*tr)/length(freqs)+45, freq2midi(freqs));

xlabel("Time (s)");
ylabel("MIDI note");
title("MIDI score")
set(gca, 'Fontsize', fontsize);
%% trash this too??
track = y(1:floor(end*song_portion));
sz = length(track);
tau = linspace(0, tr*song_portion, num_gabors);  % gabor centers, unit seconds

t = (1:sz)/Fs; % song sample time-values
t = cast(t, 'single');
Sgt_spec = cast(zeros(floor(sz/frequency_step),num_gabors), 'single');

gauss = @(t, tau) exp(-a*(t-tau).^2);
% Perform gabor transform, store in Sgt_spec
for j = 1:num_gabors
    g = gauss(t, tau(j));
    g = cast(reshape(g,sz,1), 'single');
    Sg = g.*track;
    Sgt = fft(Sg);
    Sgt = Sgt(1:frequency_step:end);
    % (un)-comment the line below to resolve off-by-one errors
    Sgt = Sgt(1:end-1);
    Sgt_spec(:,j) = cast(fftshift(log(1+abs(Sgt))), 'single');
end
clear Sgt Sg g;

% Display spectrogram
subplot(2,1,1);
ks = linspace(-Fs/2, Fs/2, sz/frequency_step);
pcolor(tau, ks, Sgt_spec)
shading interp
set(gca, 'ylim', [600, 6000], 'Fontsize', 12)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
title("Spectrogram of Comfortably Numb's guitar solo. Computed with " + num_gabors + " STFTs using a=" + a);
title("First " + floor(song_portion*tr) + " seconds with " + num_gabors + " a = " + a + " STFTs");

% Let's get the notes.
bandwidth = (length(ks)/2)*max_freq/(Fs/2); % half bandwidth, measured as # of bins below max_freq
% for LPF centered at 0
minbin = floor(length(ks)/2 - bandwidth);
maxbin = floor(length(ks)/2 + bandwidth);

notes = strings(1,num_gabors);
freqs = zeros(1, num_gabors);
for j = 1:num_gabors
    [M,I] = max(Sgt_spec(minbin:maxbin,j)); I = I + minbin;
    % [M,I] = max(Sgt_spec(:,j));
    if M > threshold
        freqs(j) = abs(ks(I));
    else
        freqs(j) = 0;
    end
end
subplot(2,1,2);
stairs((1:length(freqs))*(song_portion*tr)/length(freqs), freq2midi(freqs));
xlabel("Time (s)");
ylabel("MIDI note");
title("MIDI score")
