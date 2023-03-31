function V = repair(V, lu)

[NP, D]  = size(V);

xl = repmat(lu(1, :), NP, 1);
xu = repmat(lu(2, :), NP, 1);

% if any variable of the mutant vector violates the lower bound
pos = V < xl;
V(pos) = 2 .* xl(pos) - V(pos);
pos_ = V(pos) > xu(pos); 
V(pos(pos_)) = xu(pos(pos_));

% if any variable of the mutant vector violates the upper bound
pos = V > xu;
V(pos) = 2 .* xu(pos) - V(pos);
pos_ = V(pos) < xl(pos); 
V(pos(pos_)) = xl(pos(pos_));
end