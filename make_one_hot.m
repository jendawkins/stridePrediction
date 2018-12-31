function [one_hot] = make_one_hot(labels, alphabet)
one_hot = [];
for label = 1:length(labels)
    len = max(alphabet);
    one_hot = [one_hot; zeros([1,len])];
    one_hot(label, labels(label)) = 1;
end
end