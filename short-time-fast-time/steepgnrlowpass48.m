function y = steepgnrlowpass48(x)
%DOFILTER Filters input x and returns output y.

% MATLAB Code
% Generated by MATLAB(R) 9.8 and DSP System Toolbox 9.10.
% Generated on: 10-Feb-2021 14:32:53

%#codegen

% To generate C/C++ code from this function use the codegen command.
% Type 'help codegen' for more information.

persistent Hd;

if isempty(Hd)
    
    % The following code was used to design the filter coefficients:
    %
    % Fpass = 200;    % Passband Frequency
    % Fstop = 400;    % Stopband Frequency
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
        'SOSMatrix', [1 2 1 1 -1.98401169650101 0.984872445577703; 1 2 1 1 ...
        -1.95835558078796 0.959205199145036; 1 2 1 1 -1.94384291426632 ...
        0.944686236408346], ...
        'ScaleValues', [0.000215187269173392; 0.000212404589269023; ...
        0.00021083053550577; 1]);
end

s = double(x);
y = step(Hd,s);
