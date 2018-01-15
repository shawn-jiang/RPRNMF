%-------------------------------------------------------------------------%
%   Construct similarity matrix from distance constraints
%
%   Shawn Jiang
%   03/11/2016
%-------------------------------------------------------------------------%
function S = smx(L, N)
global dep A
num = size(L,1);
idx = zeros(num, 3);
nodemap = containers.Map;

for i = 1 : num
    % get two nodes
    node1 = sprintf('%d,%d',min(L(i,1), L(i,2)), max(L(i,1), L(i,2)));
    node2 = sprintf('%d,%d',min(L(i,1), L(i,3)), max(L(i,1), L(i,3)));
    
    % check in the map
    if ~isKey(nodemap, node1)
        nodemap(node1) = size(nodemap, 1) + 1;
        
    end
    if ~isKey(nodemap, node2)
        nodemap(node2) = size(nodemap, 1) + 1;
    end
    
    % adjacent matrix
    idx(i,:) = [nodemap(node1), nodemap(node2), 1];
end

% refine the adjacent matrix
nn = length(nodemap);
A = sparse(idx(:,1), idx(:,2), idx(:,3), nn, nn);

% store the indices of nnz in S
S = zeros(nn,3);

% calculate depth of each node
dep = zeros(nn,1);
sps = full(~any(A,1)); % starting point
for i = 1 : length(sps)
    if sps(i) == 1
        calc_dep(i);
    end
end

% get the nodes
node_pairs = keys(nodemap);
node_ids = values(nodemap, node_pairs);

% set the similarities of each node
maxvalue = 0.9;
minvalue = 0.1;
ita = (maxvalue - minvalue) / (max(dep) - 1);
for i = 1 : nn
    indices = strsplit(node_pairs{i}, ',');
    a = str2double(indices{1});
    b = str2double(indices{2});
    S(i,:) = [a, b, minvalue + (dep(node_ids{i}) - 1) * ita];
end

% make S a symmetric sparse matrix
S = sparse(S(:,1), S(:,2), S(:,3), N, N);
S = S + S';
end

% a function that calculates the depth of a starting point and points on
% the searching path
function depth = calc_dep(sp)
global dep A
depth = 1;
ends = find(A(sp,:) ~= 0);
for j = 1 : length(ends)
    if dep(ends(j)) ~= 0
        depth = max(depth, dep(ends(j)) + 1);
    else
        depth = max(depth, calc_dep(ends(j)) + 1);
    end
end
dep(sp) = depth;
end