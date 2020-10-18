% Suguru puzzle
%
% Fill the grid so that each square in an outlined block contains a digit.
% A block of 2 squares contains the digits 1 and 2, a block of three 1 to 3,
% and so on.
% No same digit appears in neighbouring squares, not even diagonally.
% minizinc suguru.mzn suguru.dzn

include "count.mzn";
include "alldifferent.mzn";

int: N; % grid size
int: D; % number of digits
int: B; % number of blocks
set of int: Range = 1..N;

enum Color = {a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z};

% initial digits, 0 = empty
array[Range,Range] of 0..D: digits;
% blocks
array[Range,Range] of 1..B: blocks;

% N = 6;
% D = 5;
% B = 9;
% digits = 
%   [| 0,0,0,0,2,0,
%    | 0,3,0,0,0,0,
%    | 0,4,0,2,3,0,
%    | 0,0,0,0,0,0,
%    | 0,0,0,0,0,0,
%    | 0,0,0,0,0,0 |];
% blocks = 
%   [| 1,2,3,3,3,3,
%    | 2,2,2,4,3,5,
%    | 6,2,4,4,4,5,
%    | 6,7,7,4,5,5,
%    | 6,6,7,7,5,9,
%    | 6,8,8,8,8,9 |];

% the puzzle to solve
array[Range,Range] of var 1..D: puzzle;

% initial digits
constraint forall(i,j in Range where digits[i,j] > 0)(puzzle[i,j]=digits[i,j]);

% blocks contain digits 1..len(block) and are all different
constraint forall(b in 1..B)(
   let {
     array[int] of var int: cells = [puzzle[i,j] | i,j in Range where blocks[i,j]=b]
   } in
     forall(c in cells)(c <= length(cells)) /\
     alldifferent(cells)
);

% neighbouring cells digits different
constraint forall(i,j in 1..N-1)(alldifferent([puzzle[a,b] | a in {i,i+1}, b in {j,j+1}]));

%include "gecode.mzn";
% solve :: int_default_search(afc_size_max, indomain_min)
% gecode_split
% solve :: int_default_search(afc_max, indomain_split);
solve satisfy;

output [
  show(fix(puzzle[i,j])) ++ ", " ++
  if j == N then "\n" else "" endif | i,j in Range ];
