function [I] = rgbvid2grayvid(RGB)
%   Convert an RGB video (4-D array) to a grayscale video frame-by-frame
    sz = size(RGB);
    bw_sz = [sz(1) sz(2) sz(4)];
    I = uint8(zeros(bw_sz));
    for j = 1:sz(4)
        I(:,:,j) = rgb2gray(RGB(:,:,:,j));
    end
end


