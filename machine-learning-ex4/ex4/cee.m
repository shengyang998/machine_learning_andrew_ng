function c = cee(h, labels)
% label must in range of [0, 1]
% when label is 0, only right part take effect and let the limit h->1 log(1-h) to very
% high
% when label is 1, only left part take effect and let the limit h->0 log(h)to very
% high
% % ====================== INCORRECT IMPLEMENTATION ======================
% c = labels' * (-log(h)) + (1-labels)' * (-log(1 - h)); % error
% ====================== CORRECTED VERSION ======================
c = -labels .* log(h) - (1-labels) .* log(1 - h);
% implementation
end