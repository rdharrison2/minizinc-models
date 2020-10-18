% Flow free puzzle.
% minizinc -s flowfree.mzn flowfree_6x6.dzn 

include "count.mzn";

int: N; % grid size
int: C; % number of colours
set of int: Range = 1..N;

% standard colors names (red, orange, yellow, green, blue)
%  and not so standard (pink, cyan, sienna, violet, white, tan)
enum Color = {r, o, y, g, b, p, c, s, v, w, t};

% dots - defined in .dzn file
array[Range,Range] of 0..C: dots;  % 0 = empty

% dots = 
%   [| b, 0, 0, r, o,
%    | 0, 0, 0, y, 0,
%    | 0, 0, y, 0, 0,
%    | 0, r, o, 0, g,
%    | 0, b, g, 0, 0 |];

% all cells are filled with a colour
array[Range,Range] of var 1..C: puzzle;

% fix initial dots
constraint forall(i,j in Range)(if dots[i,j] > 0 then puzzle[i,j]=dots[i,j] else true endif);

% dots have exactly one of same colour north/south/east/west, lines have two
constraint forall(i,j in Range)(
  let {
    int: cnt = if dots[i,j] > 0 then 1 else 2 endif,
    array[int] of var int: cells = % get array of n/s/e/w neighbours
      [puzzle[i,m] | m in {j-1,j+1} where m >= 1 /\ m <= N ] ++
      [puzzle[m,j] | m in {i-1,i+1} where m >= 1 /\ m <= N ]
  } in count(cells, puzzle[i,j], cnt)); % /\ (dots[i,j] == 0 -> exists(i,j in index_set(cells))(i != j /\ cells[i]==cells[j])));

% solutions don't loop back (quads must not all be same color)
constraint forall(i,j in 1..N-1)(
  puzzle[i,j] != puzzle[i+1,j] \/ puzzle[i,j] != puzzle[i,j+1] \/ puzzle[i,j] != puzzle[i+1,j+1]
);

% cannot have holes
% constraint forall(i,j in 1..N where dots[i,j] == 0) (
%   let {
%     array[int] of var int: cells = 
%       [puzzle[i,m] | m in {j-1,j+1} where m >= 1 /\ m <= N ] ++
%       [puzzle[m,j] | m in {i-1,i+1} where m >= 1 /\ m <= N ]
%   } in exists(i,j in index_set(cells) where i != j)(cells[i]==cells[j]));
% %  % } in not forall(i,j in index_set(cells) where i < j)(cells[i]!=cells[j]));

%array[int] of var int: var_order = [puzzle[i,j] | i,j in Range];
%array[int] of var int: var_order = [puzzle[i,j] | i,j in reverse(Range)];
% array[1..N*N] of int: score = [
%     sum([bool2int(dots[m,n]>0) | m in {i-1,i,i+1}, n in {j-1,j,j+1}
%         where dots[i,j]==0 /\ 1 <= m /\ m <= N /\ 1 <= n /\ n <= N])
%    | i,j in Range];
% bool: ss1 = trace("score:"++show(score)++"\n", true);
% int: best_score = max(score);
% int: pos = [i | i in 1..N*N where score[i] ==  best_score][1];
% bool: ss2 = trace("pos:"++show(pos)++"\n", true);
% array[int] of int: order = [i|i in pos..N*N] ++ [i|i in 1..pos-1];
% bool: ss3 = trace("order:"++show(order)++"\n", true);

% array[int] of var int: var_order = [puzzle[((i-1) div N)+1,((i-1) mod N)+1] | i in order];

include "gecode.mzn";

%solve % :: int_search(var_order, first_fail, indomain_split, complete)
% gecode_min = basic
solve :: int_default_search(afc_size_max, indomain_min)
% gecode_split
%solve :: int_default_search(afc_max, indomain_split);
   satisfy;

output [
  show(to_enum(Color, fix(puzzle[i,j]))) ++ ", " ++
  if j == N then "\n" else "" endif | i,j in Range ];

% 5, 1, 1, 1, 2,
% 5, 1, 3, 3, 2,
% 5, 1, 3, 2, 2,
% 5, 1, 2, 2, 4,
% 5, 5, 4, 4, 4,
