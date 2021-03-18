function y = steeplowpass48(x)
%DOFILTER Filters input x and returns output y.

% MATLAB Code
% Generated by MATLAB(R) 9.8 and DSP System Toolbox 9.10.
% Generated on: 10-Feb-2021 14:25:19

%#codegen

% To generate C/C++ code from this function use the codegen command.
% Type 'help codegen' for more information.

persistent Hd;

if isempty(Hd)
    
    % The following code was used to design the filter coefficients:
    %
    % Fpass = 2000;   % Passband Frequency
    % Fstop = 3000;   % Stopband Frequency
    % Apass = 1;      % Passband Ripple (dB)
    % Astop = 30;     % Stopband Attenuation (dB)
    % Fs    = 48000;  % Sampling Frequency
    %
    % h = fdesign.lowpass('fp,fst,ap,ast', Fpass, Fstop, Apass, Astop, Fs);
    %
    % Hd = design(h, 'butter', ...
    %     'MatchExactly', 'stopband', ...
    %     'SystemObject', true);
    
    Hd = dsp.BiquadFilter( ...
        'Structure', 'Direct form II', ...
        'SOSMatrix', [1 2 1 1 -1.84263965357109 0.922141186071747; 1 2 1 1 ...
        -1.71455392825387 0.788529143422525; 1 2 1 1 -1.61606140155038 ...
        0.685787111506432; 1 2 1 1 -1.54688892743441 0.613630159286771; 1 2 1 1 ...
        -1.50599743867472 0.570974388500297; 1 1 0 1 -0.746240680120137 0], ...
        'ScaleValues', [0.0198753831251651; 0.0184938037921628; ...
        0.0174314274890133; 0.0166853079630898; 0.0162442374563946; ...
        0.126879659939932; 1]);
end

s = double(x);
y = step(Hd,s);
