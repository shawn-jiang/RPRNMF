%-------------------------------------------------------------------------%
%   Calculate the constraint satisfied rate of a matrix factorization
%
%   Shawn Jiang
%   13/01/2017
%-------------------------------------------------------------------------%
function csr = calccsr(W, H, L1, L2, type)
right1 = 0;
for l = 1 : size(L1, 1)
    q = L1(l,1);
    r = L1(l,2);
    s = L1(l,3);
    if strcmp(type, 'mse') % mean square error
        n1 = norm(W(q,:) - W(r,:), 2);
        n2 = norm(W(q,:) - W(s,:), 2);
    elseif strcmp(type, 'md') % mean divergence
        n1 = div(W(q,:), W(r,:), true);
        n2 = div(W(q,:), W(s,:), true);
    end
    if n1 < n2
        right1 = right1 + 1;
    end
end

right2 = 0;
for l = 1 : size(L2, 1)
    q = L2(l,1);
    r = L2(l,2);
    s = L2(l,3);
    if strcmp(type, 'mse') % mean square error
        n1 = norm(H(:,q) - H(:,r), 2);
        n2 = norm(H(:,q) - H(:,s), 2);
    elseif strcmp(type, 'md') % mean divergence
        n1 = div(H(:,q), H(:,r), true);
        n2 = div(H(:,q), H(:,s), true);
    end   
    if n1 < n2
        right2 = right2 + 1;
    end
end

if size(L1, 1) ~= 0 && size(L2, 1) ~= 0
    csr = (right1 / size(L1, 1) + right2 / size(L2, 1)) / 2;
elseif size(L1, 1) ~= 0
    csr = right1 / size(L1, 1);
elseif size(L2, 1) ~= 0
    csr = right2 / size(L2, 1);
else
    csr = 0;
end
end