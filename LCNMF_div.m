%-------------------------------------------------------------------------%
%   Label Constrained Non-negative Matrix Factorization(LCNMF)
%   using Divergence
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
function [W, H, md, csr] = LCNMF_div(V,K,A,B,L1,L2,Mask,X,Y)
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
md = zeros(count, 1); % Mean Divergence
csr = zeros(count, 1); % Constraints Satisfied Rate

% if the factorising matrix is huge, use sparse version
if numel(V) > 1e8
    sp = true;
else
    sp = false;
end

Temp = A' * Mask * B';
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
            Temp2 = Mask .* V .* mdivide(mrepmat(Y(k,:) * B, N, Mask), mulm(A * X, Y * B, Mask), Mask);
            Temp2 = sum(mrepmat(sum(Temp2,2)', S, A'), 2);
            X(:,k) = X(:,k) .* Temp2 ./ max(Temp1, eps);

            % update Y
            Temp4 = X(:,k)' * Temp;  
            Temp5 = Mask .* V .* mdivide(mrepmat((A * X(:,k))', M, Mask')', mulm(A * X, Y * B, Mask), Mask);
            Temp5 = sum(mrepmat(sum(Temp5,1), T, B), 2);
            Y(k,:) = Y(k,:) .* Temp5' ./ max(Temp4, eps);
        end

        W = A * X;
        H = Y * B;

        % save the MD
        md(c,1) = calcerr(W, H, V, Mask, 'md');

        % calculate the accuracy of generated constraints
        csr(c,1) = calccsr(W, H, L1, L2, 'md');

        % print 
%         fprintf('MD,CSR: %f,%f\n', md(c,1), csr(c,1));
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
            Temp2 = Mask .* V .* (repmat(Y(k,:), N, 1) * B) ./ (A * X * Y * B);
            Temp2 = sum(repmat(sum(Temp2,2), 1, S) .* A, 1);
            X(:,k) = X(:,k) .* Temp2' ./ max(Temp1, eps);
            
            % update Y
            Temp4 = X(:,k)' * Temp;  
            Temp5 = Mask .* V .* (A * repmat(X(:,k), 1, M)) ./ (A * X * Y * B);
            Temp5 = sum(repmat(sum(Temp5,1), T, 1) .* B, 2);
            Y(k,:) = Y(k,:) .* Temp5' ./ max(Temp4, eps);
%             toc;
        end

        W = A * X;
        H = Y * B;

        % save the MD
        md(c,1) = calcerr(W, H, V, Mask, 'md');

        % calculate the accuracy of generated constraints
        csr(c,1) = calccsr(W, H, L1, L2, 'md');

        % print 
%         fprintf('MD,CSR: %f,%f\n', md(c,1), csr(c,1));
    end
end
  
% plot
% ecplot(md, csr, 'md');

% return
md = md(end);
csr = csr(end);

end