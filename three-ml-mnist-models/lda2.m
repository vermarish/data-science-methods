function [performance, w, threshold] = lda2(a, b, img_train, label_train, img_test, label_test, verbose)
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

    % Make a < b
    if a > b
        temp = a;
        a = b;
        b = temp;
        clear temp
    end

    img_train_a = img_train(:,label_train==a);
    img_train_b = img_train(:,label_train==b);


    % Compute means of each group for variance calculation.
    ma = mean(img_train_a, 2);
    mb = mean(img_train_b, 2);

    % Sw: within-class scatter matrix. Get the variance within each group.
    % Sb: between-class scatter matrix. Get the variance between the two means.
    na = size(img_train_a, 2);
    nb = size(img_train_b, 2);

    Sb = (ma-mb)*(ma-mb)';
    Sw = zeros(size(img_train_a, 1));
    size(ma);
    size(img_train_a(:,1));

    for k = 1:na
        Sw = Sw + (img_train_a(:,k) - ma)*(img_train_a(:,k) - ma)';
    end
    for k = 1:nb
        Sw = Sw + (img_train_b(:,k) - mb)*(img_train_b(:,k) - mb)';
    end

    [V2, D] = eig(Sb, Sw);
    [~, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);

    % Now let's project from modes subspace to w, and find a threshold.
    lin_a = w'*img_train_a;
    lin_b = w'*img_train_b;

    % Let's make it so that digit a is below the threshold.
    if mean(lin_a) > mean(lin_b)
        w = -w;
        lin_a = -lin_a;
        lin_b = -lin_b;
    end
    lin_a = sort(lin_a);
    lin_b = sort(lin_b);

    t1 = length(lin_a);
    t2 = 1;
    while (lin_a(t1) > lin_b(t2))
        t1 = t1 - 1;
        t2 = t2 + 1;
    end
    threshold = mean([lin_a(t1) lin_b(t2)]);
    
    % Now let's classify each testing image.
    img_test_a = img_test(:,label_test==a);
    img_test_b = img_test(:,label_test==b);
    
    a_projections = w' * img_test_a;
    b_projections = w' * img_test_b;
    
    errors = sum(a_projections > threshold) + sum(b_projections < threshold);
    
    na = size(img_test_a, 2);
    nb = size(img_test_b, 2);
    performance = errors / (na+nb);
    
    if verbose
        hold on; 
        histogram(a_projections, 30); 
        histogram(b_projections, 30);
        xline(threshold, 'r');
        
        percent = num2str(floor(performance*10000)/100);
        title(strcat(string(a), " and ", string(b), ": ", ...
            percent, "% training error"), "FontSize", 13);
        
        legend(strcat("Projection of ", int2str(a)), ...
            strcat("Projection of ", int2str(b)), ...
            "threshold");
        hold off;
    end
end


