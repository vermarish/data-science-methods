function [error] = evaluateClassifier(classifier, data, labels)
%classifier should be usable with predict(classifier, data, labels)
    result = predict(classifier, data);
    score = sum(result==labels);
    error = score / length(labels);
end

