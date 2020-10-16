% Soma cube model (https://en.wikipedia.org/wiki/Soma_cube)
% minizinc -s --all-solutions soma.mzn

include "globals.mzn";

int: ntiles = 7;
int: height = 3;
int: width = 4;
int: depth = 4;
int: N = height*width*depth;  % 48

% the cube with extra column/row per slice so actually 4x4x3
% (x,y,z) <-> z*(width*depth) + y*width + x + 1, for x:0..2, y:0..2, z:0..2
% 0 = unoccupied edge column/row, 1..7 = tile colour
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

% the pieces
% V = {[0,0,0], [1,0,0], [0,0,1]};
% L = {(0,0,0), (1,0,0), (0,0,1), (0,0,2)};
% T = {(0,0,0), (0,0,1), (1,0,1), (0,0,2)};
% Z = {(0,0,0), (0,0,1), (1,0,1), (1,0,2)};
% A = {(0,0,0), (0,0,1), (0,1,0), (1,1,0)};
% B = {(0,0,0), (0,1,0), (1,1,0), (1,1,1)};
% P = {(0,0,0), (0,1,0), (1,1,0), (0,1,1)};

% regex for placement of each piece V, L, T, Z, A, B, P on cube_aux
% (alternation of each unique rotation with 1 for presence - see soma.py for code to generate these)
array[1..ntiles] of string: regex = [
 "0* 1 0{11} 1 0{3} 1 0*|0* 1 1 0{15} 1 0*|0* 1 0{15} 1 0{3} 1 0*|0* 1 1 0{14} 1 0*|0* 1 0{3} 1 0{11} 1 0*|0* 1 0{14} 1 1 0*|0* 1 1 0{2} 1 0*|0* 1 0{15} 1 1 0*|0* 1 0{2} 1 1 0*|0* 1 0{3} 1 0{15} 1 0*|0* 1 1 0{3} 1 0*|0* 1 0{3} 1 1 0*", % V
 "0* 1 1 1 0{3} 1 0*", % L - fix to one rotation to break symmetry
 "0* 1 0{3} 1 1 0{2} 1 0*|0* 1 0{2} 1 1 0{3} 1 0*|0* 1 0{14} 1 1 0{15} 1 0*|0* 1 0{3} 1 0{3} 1 0{11} 1 0*|0* 1 0{11} 1 0{3} 1 0{15} 1 0*|0* 1 1 1 0{2} 1 0*|0* 1 1 1 0{14} 1 0*|0* 1 0{11} 1 0{3} 1 0{3} 1 0*|0* 1 0{14} 1 1 1 0*|0* 1 0{15} 1 1 0{14} 1 0*|0* 1 0{2} 1 1 1 0*|0* 1 0{15} 1 0{3} 1 0{11} 1 0*", % T
 "0* 1 0{3} 1 0{15} 1 0{3} 1 0*|0* 1 0{11} 1 0{3} 1 0{11} 1 0*|0* 1 0{15} 1 0{3} 1 0{15} 1 0*|0* 1 1 0{1} 1 1 0*|0* 1 1 0{15} 1 1 0*|0* 1 0{3} 1 0{7} 1 0{3} 1 0*|0* 1 1 0{13} 1 1 0*|0* 1 0{15} 1 1 0{15} 1 0*|0* 1 1 0{3} 1 1 0*|0* 1 0{3} 1 1 0{3} 1 0*|0* 1 0{2} 1 1 0{2} 1 0*|0* 1 0{14} 1 1 0{14} 1 0*", % Z
 "0* 1 1 0{14} 1 0{3} 1 0*|0* 1 0{10} 1 0{3} 1 1 0*|0* 1 1 0{2} 1 0{12} 1 0*|0* 1 1 0{11} 1 0{3} 1 0*|0* 1 0{3} 1 0{10} 1 1 0*|0* 1 0{2} 1 1 0{14} 1 0*|0* 1 0{3} 1 1 0{10} 1 0*|0* 1 1 0{3} 1 0{15} 1 0*|0* 1 0{11} 1 1 0{2} 1 0*|0* 1 0{15} 1 1 0{3} 1 0*|0* 1 0{3} 1 0{15} 1 1 0*|0* 1 0{15} 1 0{2} 1 1 0*", % A
 "0* 1 0{15} 1 0{3} 1 1 0*|0* 1 0{14} 1 1 0{2} 1 0*|0* 1 0{2} 1 1 0{11} 1 0*|0* 1 0{3} 1 0{14} 1 1 0*|0* 1 0{3} 1 1 0{15} 1 0*|0* 1 1 0{3} 1 0{10} 1 0*|0* 1 0{10} 1 1 0{3} 1 0*|0* 1 1 0{15} 1 0{3} 1 0*|0* 1 1 0{10} 1 0{3} 1 0*|0* 1 0{12} 1 0{2} 1 1 0*|0* 1 0{3} 1 0{11} 1 1 0*|0* 1 1 0{2} 1 0{15} 1 0*", % B
 "0* 1 0{3} 1 1 0{14} 1 0*|0* 1 0{14} 1 1 0{3} 1 0*|0* 1 0{2} 1 1 0{15} 1 0*|0* 1 1 0{3} 1 0{11} 1 0*|0* 1 1 0{2} 1 0{11} 1 0*|0* 1 0{11} 1 0{2} 1 1 0*|0* 1 0{11} 1 0{3} 1 1 0*|0* 1 0{15} 1 1 0{2} 1 0*" % P
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

% search
solve satisfy;

%output [show(cube) ++ "\n"];
output [show(cube[(i*16+j*4+1)..(i*16+j*4+3)]) ++ "\n" ++ (if j=2 then "\n" else "" endif) | i in 0..2, j in 0..2];
