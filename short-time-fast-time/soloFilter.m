function y = soloFilter(x)
%DOFILTER Filters input x and returns output y.

% MATLAB Code
% Generated by MATLAB(R) 9.8 and DSP System Toolbox 9.10.
% Generated on: 10-Feb-2021 17:44:29

%#codegen

% To generate C/C++ code from this function use the codegen command.
% Type 'help codegen' for more information.

persistent Hd;

if isempty(Hd)
    
    % The following code was used to design the filter coefficients:
    %
    % Fstop1 = 390;    % First Stopband Frequency
    % Fpass1 = 420;    % First Passband Frequency
    % Fpass2 = 1250;   % Second Passband Frequency
    % Fstop2 = 1350;   % Second Stopband Frequency
    % Astop1 = 50;     % First Stopband Attenuation (dB)
    % Apass  = 1;      % Passband Ripple (dB)
    % Astop2 = 30;     % Second Stopband Attenuation (dB)
    % Fs     = 48000;  % Sampling Frequency
    %
    % h = fdesign.bandpass('fst1,fp1,fp2,fst2,ast1,ap,ast2', Fstop1, Fpass1, ...
    %                      Fpass2, Fstop2, Astop1, Apass, Astop2, Fs);
    %
    % Hd = design(h, 'butter', ...
    %     'MatchExactly', 'stopband', ...
    %     'SystemObject', true);
    
    Hd = dsp.BiquadFilter( ...
        'Structure', 'Direct form II', ...
        'SOSMatrix', [1 0 -1 1 -1.96730760390539 0.994367288405901; 1 0 -1 1 ...
        -1.99515429290365 0.998125950648714; 1 0 -1 1 -1.95642152404577 ...
        0.983236522401752; 1 0 -1 1 -1.99140580645179 0.994382319954892; 1 0 -1 ...
        1 -1.94586595705659 0.972347933253846; 1 0 -1 1 -1.98764265858871 ...
        0.990634549025947; 1 0 -1 1 -1.93570953081642 0.961773565832257; 1 0 -1 ...
        1 -1.98385245989824 0.986870500458523; 1 0 -1 1 -1.9260171047124 ...
        0.951582007107515; 1 0 -1 1 -1.98002251288357 0.983077903495418; 1 0 -1 ...
        1 -1.9168499618947 0.941838546869789; 1 0 -1 1 -1.9761396816077 ...
        0.979244238309871; 1 0 -1 1 -1.90826603446109 0.932605379932817; 1 0 -1 ...
        1 -1.97219026535745 0.97535663101112; 1 0 -1 1 -1.90032014865575 ...
        0.923941838220364; 1 0 -1 1 -1.96815988195771 0.9714017659861; 1 0 -1 1 ...
        -1.89306427310489 0.915904637475242; 1 0 -1 1 -1.96403337088188 ...
        0.967365827003285; 1 0 -1 1 -1.88654774638318 0.908548117150699; 1 0 -1 ...
        1 -1.95979473357158 0.963234486020474; 1 0 -1 1 -1.88081744930206 ...
        0.901924441838969; 1 0 -1 1 -1.9554271398669 0.958992970251814; 1 0 -1 1 ...
        -1.87591787006082 0.896083716163767; 1 0 -1 1 -1.95091304730361 ...
        0.954626255717376; 1 0 -1 1 -1.94623450720515 0.950119461826758; 1 0 -1 ...
        1 -1.87189098383245 0.891073939303454; 1 0 -1 1 -1.94137377137859 ...
        0.945458559295324; 1 0 -1 1 -1.86877582905618 0.88694068645071; 1 0 -1 1 ...
        -1.93631436854851 0.940631554523269; 1 0 -1 1 -1.86660760793301 ...
        0.88372634900371; 1 0 -1 1 -1.93104288867454 0.935630374092734; 1 0 -1 1 ...
        -1.86541607011565 0.881468693252933; 1 0 -1 1 -1.92555178101227 ...
        0.930453726150908; 1 0 -1 1 -1.8652228713309 0.880198420906166; 1 0 -1 1 ...
        -1.9198434937506 0.925111215134173; 1 0 -1 1 -1.86603757711058 ...
        0.879935374119185; 1 0 -1 1 -1.91393616149072 0.919628836564306; 1 0 -1 ...
        1 -1.86785210373735 0.880683116174922; 1 0 -1 1 -1.90787062094942 ...
        0.914055530191003; 1 0 -1 1 -1.87063381471092 0.882421997950965; 1 0 -1 ...
        1 -1.90171764580568 0.908469591925594; 1 0 -1 1 -1.87431838077624 ...
        0.885101666131816; 1 0 -1 1 -1.89558301455791 0.902982557276843; 1 0 -1 ...
        1 -1.87880478884192 0.888635252232103; 1 0 -1 1 -1.88960704522185 ...
        0.897737379447277; 1 0 -1 1 -1.88395586535564 0.892898583572083], ...
        'ScaleValues', [0.0550784200040117; 0.0550784200040117; ...
        0.0548736097764453; 0.0548736097764453; 0.0546724772978888; ...
        0.0546724772978888; 0.0544758575924469; 0.0544758575924469; ...
        0.0542845390044662; 0.0542845390044662; 0.0540992625188714; ...
        0.0540992625188714; 0.0539207215552862; 0.0539207215552862; ...
        0.053749562166664; 0.053749562166664; 0.0535863835750648; ...
        0.0535863835750648; 0.0534317389802508; 0.0534317389802508; ...
        0.0532861365806253; 0.0532861365806253; 0.0531500407504578; ...
        0.0531500407504578; 0.0530238733220901; 0.0530238733220901; ...
        0.0529080149267237; 0.0529080149267237; 0.0528028063522915; ...
        0.0528028063522915; 0.0527085498817079; 0.0527085498817079; ...
        0.0526255105793858; 0.0526255105793858; 0.0525539174982476; ...
        0.0525539174982476; 0.0524939647835147; 0.0524939647835147; ...
        0.0524458126533229; 0.0524458126533229; 0.0524095882396796; ...
        0.0524095882396796; 0.0523853862764848; 0.0523853862764848; ...
        0.0523732696242982; 0.0523732696242982; 1]);
end

s = double(x);
y = step(Hd,s);
