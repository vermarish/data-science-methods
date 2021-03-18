function [note] = freq2note(freq)
% Input vector of frequencies
% Outputs 'B3' or whatever the note is for each frequency as row vector
%   Does so by comparing against C0.
%   From a base frequency, I multiply 2^{1/12} until I get up to the target
%       frequency. Thus compute log_1.0595 (target/base) to get the num of 
%       semitones above base. Then compare mod 12 against the map to get
%       the note. Then do /12 to find out how many octaves to increase by.
%       Counting system works best with C0 as the base. Otherwise a
%       semitone offset is necessary.
    c0 = 16.35160;
    thresh = 0.5;
    pitchSet = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
    
    freq = reshape(freq, 1, length(freq));
    note = strings(1, length(freq));
    
    for j = 1:length(freq)
        if freq(j) == 0
            note(j) = "R";
        else
            semitones = log(freq(j)/c0)/log(2^(1/12));
            semitones = floor(semitones + thresh);
    
            pitch = pitchSet(mod(semitones, 12)+1); % 1-based indexing
                                                    % hurts
            octave = string(floor(semitones/12));
    
            note(j) = strcat(pitch, octave);
        end
    end
end

