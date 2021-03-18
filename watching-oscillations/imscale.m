function [I] = imscale(data)
% To represent 2-D data graphically, it must be scaled from 0 to 255 and
% represented as uint8
    I = double(data);
    I = I - min(I, [], 'all');
    I = I * 255 / max(I, [], 'all');
    I = uint8(I);
end

