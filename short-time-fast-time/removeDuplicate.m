function [output] = removeDuplicate(vec)
% Input vector of notes
% Output will remove duplicates
    output = strings(1,length(vec));
    i = 1;
    offset = 0;
    while i < length(vec)
        output(i-offset) = vec(i);
        if vec(i) == vec(i+1)
            offset = offset + 1;
        end
        i = i + 1;
    end
    output = output(1:end-offset);
end