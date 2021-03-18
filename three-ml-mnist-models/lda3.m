function [performance, w, threshold] = lda3(a, b, c, img_train, label_train, img_test, label_test, verbose)
% INPUT
% a, b are the digits to try and classify
% V are the singular projection vectors (pre-computed). Should be 784 x modes.
% img data should be reshaped to modes x n, so a column is a PCA'd image.
% if verbose, then it will place a double histogram in the current figure.
%
% OUTPUT
% performance is the percentage of images correctly classified.
% w spans the projection space
% threshold gives the value to project on

    digits = [a b c];
    digits = sort(digits);
    a = digits(1);
    b = digits(2);
    c = digits(3);
    

    img_train_a = img_train(:,label_train==a);
    img_train_b = img_train(:,label_train==b);
    img_train_c = img_train(:,label_train==c);

    % Compute means of each group for variance calculation.
    ma = mean(img_train_a, 2);
    mb = mean(img_train_b, 2);
    mc = mean(img_train_c, 2);
    mu = mean([img_train_a img_train_b img_train_c], 2);
    
    % Sb: between-class scatter matrix. Get the variance between the two means.
    Sb = (ma-mu)*(ma-mu)' + ...
         (mb-mu)*(mb-mu)' + ...
         (mc-mu)*(mc-mu)';
    
    % Sw: within-class scatter matrix. Get the variance within each group.
    Sw = zeros(size(img_train_a, 1));
    
    na = size(img_train_a, 2);
    nb = size(img_train_b, 2);
    nc = size(img_train_c, 2);

    for k = 1:na
        Sw = Sw + (img_train_a(:,k) - ma)*(img_train_a(:,k) - ma)';
    end
    for k = 1:nb
        Sw = Sw + (img_train_b(:,k) - mb)*(img_train_b(:,k) - mb)';
    end
    for k = 1:nc
        Sw = Sw + (img_train_c(:,k) - mc)*(img_train_c(:,k) - mc)';
    end

    [V2, D] = eig(Sb, Sw);
    [~, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    
    % Now let's project from modes subspace to w, and find thresholds.
    lin_a = w'*img_train_a;
    lin_b = w'*img_train_b;
    lin_c = w'*img_train_c;

    % order the values based on their order of projection.
    order = [mean(lin_a) mean(lin_b) mean(lin_c); a b c]';
    order = sortrows(order, 1);
    
    switch(order(1,2))
        case a
            lin_1 = lin_a;
        case b
            lin_1 = lin_b;
        case c
            lin_1 = lin_c;
        otherwise
            disp("failed at reordering")
    end
    switch(order(2,2))
        case a
            lin_2 = lin_a;
        case b
            lin_2 = lin_b;
        case c
            lin_2 = lin_c;
        otherwise
            disp("failed at reordering")
    end
    switch(order(3,2))
        case a
            lin_3 = lin_a;
        case b
            lin_3 = lin_b;
        case c
            lin_3 = lin_c;
        otherwise
            disp("failed at reordering")
    end
    
    a = order(1,2);
    b = order(2,2);
    c = order(3,2);
    
    
    % now lin_a < lin_b < lin_c
    % time to find thresholds. 

    t1 = length(lin_a);
    t2 = 1;
    
    while (lin_a(t1) > lin_b(t2))
        t1 = t1 - 2;
        t2 = t2 + 1;
    end
    
    t3 = length(lin_b);
    t4 = 1;
    while (lin_b(t3) > lin_c(t4))
        t3 = t3 - 1;
        t4 = t4 + 2;
    end
    
    threshold = [mean([lin_a(t1) lin_b(t2)]) ...
                 mean([lin_b(t3) lin_c(t4)])];
    
    % Now let's classify each testing image.
    img_test_a = img_test(:,label_test==a);
    img_test_b = img_test(:,label_test==b);
    img_test_c = img_test(:,label_test==c);
    
    a_projections = w' * img_test_a;
    b_projections = w' * img_test_b;
    c_projections = w' * img_test_c;
    
    errors = sum(a_projections > threshold(1)) ...
        + sum(b_projections < threshold(1)) ...
        + sum(b_projections > threshold(2)) ... 
        + sum(c_projections < threshold(2));
    
    na = size(img_test_a, 2);
    nb = size(img_test_b, 2);
    nc = size(img_test_c, 2);
    performance = errors / (na+nb+nc);
    
    if verbose
        hold on; 
        histogram(a_projections, 30); 
        histogram(b_projections, 30);
        histogram(c_projections, 30);
        xline(threshold(1), 'r');
        xline(threshold(2), 'r');
        
        percent = num2str(floor(performance*10000)/100);
        title(strcat(string(a), ", ", string(b), ", and ", string(c), ": ", ...
            percent, "% training error"), "FontSize", 13);
        
        legend(strcat("Projection of ", int2str(a)), ...
            strcat("Projection of ", int2str(b)), ...
            strcat("Projection of ", int2str(c)), ...
            "first threshold", ...
            "second threshold");
        
        hold off;
    end
end


