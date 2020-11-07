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
var set of WARRIOR: raid;
var int: obj = sum(w in raid)(strength[w]);

include "globals.mzn";

% cardinality of raid set
constraint l <= card(raid) /\ card(raid) <= u;
% remove dummmy from raid
constraint not (dummy in raid);

% no more than m pair of clans that hate each other
constraint sum(i,j in raid where i<j)(hates[clan[i],clan[j]]) <= m;

solve maximize obj;

output ["raid \(raid) is \(obj)"];

% raid {WARRIOR_29, WARRIOR_30, WARRIOR_40, WARRIOR_49, WARRIOR_50, WARRIOR_67} is 49
