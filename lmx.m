%-------------------------------------------------------------------------%
%   Construct label matrix from distance constraints
%
%   Shawn Jiang
%   03/11/2016
%-------------------------------------------------------------------------%
function A = lmx(L, N)
A = zeros(N, 1);

c = 1;
for i = 1 : size(L, 1)
    q = L(i,1);
    r = L(i,2);
    % merge classes
    if A(q) > 0 && A(r) > 0
        A(A == A(r)) = A(q);
    elseif A(q) > 0
        A(r) = A(q);
    elseif A(r) > 0
        A(q) = A(r);
    else
        A(q) = c;
        A(r) = c;
        c = c + 1;
    end
end

% class with only one data
for i = 1 : N
    if A(i) == 0
        A(i) = c;
        c = c + 1;
    end
end

A = sparse((1:N)', A, ones(N,1));

% remove zero columns cased by merging
A(:, ~any(A,1)) = [];
end