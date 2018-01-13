%-------------------------------------------------------------------------%
%   Standard Non-negative Matrix Factorization (NMF)
%   using Divergence
%
%   Shawn Jiang
%   03/11/2016
%-------------------------------------------------------------------------%
function [W, H, md, csr] = NMF_div(V,K,L1,L2,Mask,W,H)
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

if ~exist('W','var') || ~exist('H','var')
    W = rand(N,K);
    H = rand(K,M);
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
            Temp1 = sum(mdivide(V, mulm(W, H, Mask), Mask) .* mrepmat(H(k,:), N, Mask), 2);
            W(:,k) = W(:,k) .* Temp1 ./ max(Mask * H(k,:)', eps);

            % update H
            Temp2 = sum(mdivide(V, mulm(W, H, Mask), Mask) .* mrepmat(W(:,k)', M, Mask')', 1);
            H(k,:) = H(k,:) .* Temp2 ./ max(W(:,k)' * Mask, eps);
        end

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
            % update W
            Temp1 = sum(Mask .* V .* repmat(H(k,:), N, 1) ./ (W * H), 2);
            W(:,k) = W(:,k) .* Temp1 ./ max(Mask * H(k,:)', eps);

            % update H
            Temp2 = sum(Mask .* V .* repmat(W(:,k), 1, M) ./ (W * H), 1);
            H(k,:) = H(k,:) .* Temp2 ./ max(W(:,k)' * Mask, eps);
        end

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