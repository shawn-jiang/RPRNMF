%-------------------------------------------------------------------------%
%   Standard Non-negative Matrix Factorization (NMF)
%   using Euclidean Distance
%
%   Shawn Jiang
%   03/11/2016
%-------------------------------------------------------------------------%
function [W, H, mse, csr] = NMF_euc(V,K,L1,L2,Mask,W,H)
% if no input
if nargin == 0
    N = 50;
    M = 60;
    K = 20;
    [V, L1, L2] = randmx(N, M, K, 10, 10);
    Mask = V ~= 0;
else
    N = size(V, 1);
    M = size(V, 2);
end

% initialization
if ~exist('Mask', 'var') || numel(Mask) == 0 
    Mask = ones(N,M);   
    V(V == 0) = eps;
end

if ~exist('W', 'var') || ~exist('H', 'var')
    W = rand(N,K);
    H = rand(K,M);
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

if sp % if the matrix is huge, then use sparse version
    for c = 1 : count
        if mod(c, round(count/10)) == 0
            fprintf('%d ', c / round(count/10));
        end
        if c == count
            fprintf('\n');
        end
        for k = 1 : K
            % update W
            W(:,k) = W(:,k) .* (Mask .* V * H(k,:)') ./ ...
                max(mulm(W, H, Mask) * H(k,:)', eps);

            % update H
            H(k,:) = H(k,:) .* (W(:,k)' * (Mask .* V)) ./ ...
                max(W(:,k)' * mulm(W, H, Mask), eps);
        end

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
            % update W
            W(:,k) = W(:,k) .* (Mask .* V * H(k,:)') ./ ...
                max(Mask .* (W * H) * H(k,:)', eps);

            % update H
            H(k,:) = H(k,:) .* (W(:,k)' * (Mask .* V)) ./ ...
                max(W(:,k)' * (Mask .* (W * H)), eps);
%             toc;
        end

        % save the MSE
        mse(c,1) = calcerr(W, H, V, Mask, 'mse');

        % calculate the accuracy of generated constraints
        csr(c,1) = calccsr(W, H, L1, L2, 'mse');

        % print 
%         fprintf('%d,MSE,CSR: %f,%f\n', c, mse(c,1), csr(c,1));
    end
end
    
% plot
% ecplot(mse, csr, 'mse');

% return
mse = mse(end);
csr = csr(end);

end