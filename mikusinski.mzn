% Mikusinski 3x3x3 cube puzzle
% minizinc -s --all-solutions mikusinski.mzn
% minizinc -s -c mikusinski.mzn && run-fzn-gecode-gist mikusinski.fzn 

include "globals.mzn";

int: ntiles = 6;
int: height = 3;
int: width = 4;
int: depth = 4;
int: N = height*width*depth;  % 48

% the cube with extra column/row per slice so actually 4x4x3
% (x,y,z) <-> z*(width*depth) + y*width + x + 1, for x:0..2, y:0..2, z:0..2
% 0 = unoccupied edge column/row, 1..ntiles = tile colour
array[1..N] of var 0..ntiles: cube;

% cube_aux has a cube for each piece with 0/1 to make regex simple
array[1..ntiles,1..N] of var 0..1: cube_aux;

% link cube to cube_aux
% TODO: is there a channel constraint we can use?
constraint forall (i in 1..ntiles,j in 1..N) ( cube_aux[i,j] == 1 <-> cube[j] == i );

% edge is unoccupied
constraint forall (x in 1..(height*depth)) (
    cube[(x-1)*width+width] = 0);

% top is unoccupied
constraint forall (x in 1..height, w in 1..width) (
    cube[(x-1)*(width*depth)+(depth-1)*width+w] = 0);

% all other squares are occupied
constraint forall (h in 1..height, d in 1..depth-1, w in 1..width-1) (
    cube[(h-1)*(width*depth) + (d-1)*width + w] != 0);

% the pieces (4*3+5*3=12+15=27)
%     A=np.array([[0,0,0],[1,0,0],[0,1,0],[0,2,0]]),
%     B=np.array([[0,0,0],[1,0,0],[0,1,0],[0,2,0],[1,0,1]]),
%     C=np.array([[0,0,0],[1,0,0],[0,1,0],[1,0,1]]),
%     D=np.array([[0,0,0],[1,0,0],[1,1,0],[0,0,1]]),
%     E=np.array([[0,0,0],[1,0,0],[2,0,0],[1,1,0],[0,0,1]]),
%     F=np.array([[1,0,0],[1,1,0],[0,1,0],[1,0,1],[2,0,1]]),

% regex for placement of each piece
% (alternation of each unique rotation with 1 for presence - see soma.py for code to generate these)
% fix first piece to one rotation to break symmetry
array[1..ntiles] of string: regex = [
 "0* 1 1 0{3} 1 0{3} 1 0*",
 "0* 1 1 0{3} 1 0{15} 1 0{15} 1 0*|0* 1 1 1 0{3} 1 0{15} 1 0*|0* 1 0{15} 1 0{1} 1 1 1 0*|0* 1 0{15} 1 1 0{3} 1 0{3} 1 0*|0* 1 0{15} 1 0{15} 1 0{2} 1 1 0*|0* 1 1 0{7} 1 0{3} 1 0{3} 1 0*|0* 1 0{6} 1 0{3} 1 0{3} 1 1 0*|0* 1 0{3} 1 0{2} 1 1 0{14} 1 0*|0* 1 0{3} 1 0{15} 1 1 1 0*|0* 1 1 0{14} 1 0{3} 1 0{3} 1 0*|0* 1 0{15} 1 0{10} 1 0{3} 1 1 0*|0* 1 1 0{2} 1 0{3} 1 0{8} 1 0*|0* 1 0{11} 1 1 1 0{1} 1 0*|0* 1 0{3} 1 0{3} 1 0{6} 1 1 0*|0* 1 0{3} 1 0{3} 1 0{15} 1 1 0*|0* 1 0{3} 1 1 1 0{9} 1 0*|0* 1 0{3} 1 0{9} 1 1 1 0*|0* 1 0{2} 1 1 0{14} 1 0{15} 1 0*|0* 1 0{15} 1 0{15} 1 1 0{3} 1 0*|0* 1 1 1 0{13} 1 0{3} 1 0*|0* 1 1 0{2} 1 0{12} 1 0{15} 1 0*|0* 1 0{3} 1 1 0{10} 1 0{15} 1 0*|0* 1 0{15} 1 0{11} 1 1 0{2} 1 0*|0* 1 1 1 0{11} 1 0{3} 1 0*",
 "0* 1 1 0{14} 1 0{3} 1 0*|0* 1 0{10} 1 0{3} 1 1 0*|0* 1 1 0{2} 1 0{12} 1 0*|0* 1 1 0{11} 1 0{3} 1 0*|0* 1 0{3} 1 0{10} 1 1 0*|0* 1 0{2} 1 1 0{14} 1 0*|0* 1 0{3} 1 1 0{10} 1 0*|0* 1 1 0{3} 1 0{15} 1 0*|0* 1 0{11} 1 1 0{2} 1 0*|0* 1 0{15} 1 1 0{3} 1 0*|0* 1 0{3} 1 0{15} 1 1 0*|0* 1 0{15} 1 0{2} 1 1 0*",
 "0* 1 0{15} 1 0{3} 1 1 0*|0* 1 0{14} 1 1 0{2} 1 0*|0* 1 0{2} 1 1 0{11} 1 0*|0* 1 0{3} 1 0{14} 1 1 0*|0* 1 0{3} 1 1 0{15} 1 0*|0* 1 1 0{3} 1 0{10} 1 0*|0* 1 0{10} 1 1 0{3} 1 0*|0* 1 1 0{15} 1 0{3} 1 0*|0* 1 1 0{10} 1 0{3} 1 0*|0* 1 0{12} 1 0{2} 1 1 0*|0* 1 0{3} 1 0{11} 1 1 0*|0* 1 1 0{2} 1 0{15} 1 0*",
 "0* 1 0{12} 1 0{1} 1 1 1 0*|0* 1 0{3} 1 0{11} 1 1 0{14} 1 0*|0* 1 0{14} 1 1 0{15} 1 0{3} 1 0*|0* 1 0{3} 1 1 1 0{14} 1 0*|0* 1 0{2} 1 1 0{3} 1 0{7} 1 0*|0* 1 0{2} 1 1 1 0{15} 1 0*|0* 1 0{11} 1 0{3} 1 0{3} 1 1 0*|0* 1 0{7} 1 0{2} 1 1 0{3} 1 0*|0* 1 1 0{15} 1 0{3} 1 0{11} 1 0*|0* 1 1 1 0{2} 1 0{10} 1 0*|0* 1 0{12} 1 0{2} 1 1 1 0*|0* 1 1 0{10} 1 0{3} 1 0{15} 1 0*|0* 1 0{3} 1 0{14} 1 1 0{15} 1 0*|0* 1 0{13} 1 1 1 0{2} 1 0*|0* 1 0{3} 1 1 0{2} 1 0{15} 1 0*|0* 1 0{15} 1 1 0{10} 1 0{3} 1 0*|0* 1 0{10} 1 1 0{3} 1 0{3} 1 0*|0* 1 0{11} 1 0{3} 1 0{14} 1 1 0*|0* 1 0{3} 1 0{2} 1 1 0{11} 1 0*|0* 1 0{14} 1 1 1 0{1} 1 0*|0* 1 0{15} 1 0{3} 1 0{11} 1 1 0*|0* 1 1 0{2} 1 0{3} 1 0{11} 1 0*|0* 1 0{15} 1 0{3} 1 1 0{2} 1 0*|0* 1 1 1 0{3} 1 0{10} 1 0*",
 "0* 1 1 0{15} 1 0{3} 1 1 0*|0* 1 0{2} 1 1 0{11} 1 1 0*|0* 1 1 0{2} 1 0{14} 1 1 0*|0* 1 0{3} 1 0{14} 1 1 0{2} 1 0*|0* 1 0{3} 1 0{8} 1 0{2} 1 1 0*|0* 1 0{10} 1 1 0{3} 1 0{10} 1 0*|0* 1 0{12} 1 0{2} 1 1 0{11} 1 0*|0* 1 0{14} 1 1 0{2} 1 0{15} 1 0*|0* 1 0{3} 1 1 0{15} 1 0{3} 1 0*|0* 1 0{15} 1 0{3} 1 1 0{15} 1 0*|0* 1 1 0{9} 1 1 0{3} 1 0*|0* 1 1 0{3} 1 0{6} 1 0{3} 1 0*"
];

% placement of each piece is a regular expression constraint
constraint
  forall (t in 1..ntiles)(
    let {
      array[1..N] of var 0..1: x = cube_aux[t,..],
      string: r = regex[t]
    } in
      regular(x,r)
  );

% there are just two solutions, one being:

% [3, 4, 4]
% [3, 3, 4]
% [6, 2, 2]

% [3, 4, 2]
% [5, 6, 2]
% [6, 6, 2]

% [5, 1, 1]
% [5, 6, 1]
% [5, 5, 1]

solve satisfy;

%output [show(cube) ++ "\n"];
output [show(cube[(i*16+j*4+1)..(i*16+j*4+3)]) ++ "\n" ++ (if j=2 then "\n" else "" endif) | i in 0..2, j in 0..2];
