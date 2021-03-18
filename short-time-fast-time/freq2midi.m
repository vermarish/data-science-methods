function [midi] = freq2midi(freq)
% Input vector of frequencies
%   Not guaranteed to work with array
% Outputs the 7-bit MIDI channel representation of pitch
%   Does so by comparing against C0.
%   From a base frequency, I multiply 2^{1/12} until I get up to the target
%       frequency. Thus compute log_1.0595 (target/base) to get the num of 
%       semitones above base. Then compare mod 12 against the map to get
%       the note. Then do /12 to find out how many octaves to increase by.
%       Counting system works best with C0 as the base; otherwise a 
%       semitone offset is necessary.
    c0 = 16.35160;
    thresh = 0.5;
    semitones = log(freq./c0)/log(2^(1/12));
    % semitones = cast(floor(semitones+thresh), "uint8");
    semitones = floor(semitones + thresh);
    
    midi = semitones + 12;
end

