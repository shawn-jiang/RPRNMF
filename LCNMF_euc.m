%-------------------------------------------------------------------------%
%   Label Constrained Non-negative Matrix Factorization(LCNMF)
%   using Euclidean Distance
%
%   Shawn Jiang
%   03/11/2016
%
%   input
%       A: label matrix for the first factorized matrix
%       B: label matrix for the second factorized matrix
%
%   We extends the original LCNMF, which only has one label
%   matrix, to this algorithm having two label matrices
%-------------------------------------------------------------------------%
function [W, H, mse, csr] = LCNMF_euc(V,K,A,B,L1,L2,Mask,X,Y)
% if no input
if nargin == 0
    N = 50;
    M = 60;
    K = 20;
    [V, L1, L2] = randmx(N, M, K, 10, 10);
    A = lmx(L1, N);
    B = lmx(L2, M)';
    Mask = V ~= 0;
end

[N, S] = size(A);
[T, M] = size(B);

% initialization
if ~exist('Mask', 'var') || numel(Mask) == 0
    Mask = ones(N,M);   
    V(V == 0) = eps;
end

if ~exist('X','var') || ~exist('Y','var')
    X = rand(S, K);
    Y = rand(K, T);
end

count = 500; % iteration times
mse = zeros(count, 1); % Mean Squared Error
csr = zeros(count, 1); % Constraints Satisfied Rate

% if the factorising matrix is huge, use sparse version
if numel(V) > 1e8
    sp = true;
else
    sp = false;
end

Temp = A' * (Mask .* V) * B';
if sp % if the matrix is huge, then use sparse version
    for c = 1 : count  
        if mod(c, round(count/10)) == 0
            fprintf('%d ', c / round(count/10));
        end
        if c == count
            fprintf('\n');
        end
        
        for k = 1 : K
            % update X
            Temp1 = Temp * Y(k,:)';
            Temp2 = A' * mulm(A * X, Y * B, Mask) * B' * Y(k,:)';  
            X(:,k) = X(:,k) .* Temp1 ./ max(Temp2, eps);

            % update Y
            Temp3 = X(:,k)' * Temp;
            Temp4 = X(:,k)' * A' * mulm(A * X, Y * B, Mask) * B';
            Y(k,:) = Y(k,:) .* Temp3 ./ max(Temp4, eps);
        end

        W = A * X;
        H = Y * B;

        % save the MSE
        mse(c,1) = calcerr(W, H, V, Mask, 'mse');

        % calculate the accuracy of generated constraints
        csr(c,1) = calccsr(W, H, L1, L2, 'mse');

        % print 
%         fprintf('MSE,CSR: %f,%f\n', mse(c,1), csr(c,1)); 
    end
else % if the matrix is not huge, use full version   
    for c = 1 : count
        if mod(c, round(count/10)) == 0
            fprintf('%d ', c / round(count/10));
        end
        if c == count
            fprintf('\n');
        end
        
        for k = 1 : K
%             tic;
            % update X
            Temp1 = Temp * Y(k,:)';
            Temp2 = A' * (Mask .* (A * X * Y * B)) * B' * Y(k,:)';        
            X(:,k) = X(:,k) .* Temp1 ./ max(Temp2, eps);

            % update Y
            Temp3 = X(:,k)' * Temp;
            Temp4 = X(:,k)' * A' * (Mask .* (A * X * Y * B)) * B';
            Y(k,:) = Y(k,:) .* Temp3 ./ max(Temp4, eps);
%             toc;
        end

        W = A * X;
        H = Y * B;

        % save the MSE
        mse(c,1) = calcerr(W, H, V, Mask, 'mse');

        % calculate the accuracy of generated constraints
        csr(c,1) = calccsr(W, H, L1, L2, 'mse');

        % print 
%         fprintf('MSE,CSR: %f,%f\n', mse(c,1), csr(c,1)); 
    end
end
    
% plot
% ecplot(mse, csr, 'mse');

% return
mse = mse(end);
csr = csr(end);

end