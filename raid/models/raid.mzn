% assignment 2
% select a set of raiding warrior
enum WARRIOR;
WARRIOR: dummy;
enum CLAN; % the clan they belong too

int: l; % minimum raid party size
int: u; % maximum raid party size
int: m; % limit of hatreds

array[WARRIOR] of int: strength;
array[WARRIOR] of CLAN: clan;

array[CLAN,CLAN] of 0..1: hates;

% decision variables
% sorted descending, only last u-l may be dummy
array[1..u] of var WARRIOR: raid;

% constrain raid cardinality and make it canonical
constraint forall(i in 1..u-1)(raid[i] + (raid[i]=dummy) > raid[i+1]);
constraint forall(i in 1..l)(raid[i] != dummy);

% note: dummy strength is 0
var int: obj = sum(w in raid)(strength[w]);

% no more than m pair of clans that hate each other
var int: hate = sum(i,j in 1..u where i<j)(hates[clan[raid[i]],clan[raid[j]]]);
constraint hate <= m;

solve maximize obj;

output ["raid \(raid) is \(obj) hate \(hate) (l=\(l), u=\(u), m=\(m))"];

% raid [W8, W7, W5, W4, W1, D] is 33 hate 2 (l=4, u=6, m=2)
% raid [W9, W8, W7, W4, W2, W1, D] is 38 hate 3 (l=6, u=7, m=3)
% raid [WARRIOR_90, WARRIOR_80, WARRIOR_70, WARRIOR_60, WARRIOR_20, WARRIOR_10] is 54 hate 0 (l=4, u=6, m=2)
