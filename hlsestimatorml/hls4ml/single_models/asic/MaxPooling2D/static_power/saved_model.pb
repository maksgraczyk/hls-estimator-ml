??"
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
|
dense_961/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:m*!
shared_namedense_961/kernel
u
$dense_961/kernel/Read/ReadVariableOpReadVariableOpdense_961/kernel*
_output_shapes

:m*
dtype0
t
dense_961/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*
shared_namedense_961/bias
m
"dense_961/bias/Read/ReadVariableOpReadVariableOpdense_961/bias*
_output_shapes
:m*
dtype0
?
batch_normalization_865/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*.
shared_namebatch_normalization_865/gamma
?
1batch_normalization_865/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_865/gamma*
_output_shapes
:m*
dtype0
?
batch_normalization_865/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*-
shared_namebatch_normalization_865/beta
?
0batch_normalization_865/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_865/beta*
_output_shapes
:m*
dtype0
?
#batch_normalization_865/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*4
shared_name%#batch_normalization_865/moving_mean
?
7batch_normalization_865/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_865/moving_mean*
_output_shapes
:m*
dtype0
?
'batch_normalization_865/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*8
shared_name)'batch_normalization_865/moving_variance
?
;batch_normalization_865/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_865/moving_variance*
_output_shapes
:m*
dtype0
|
dense_962/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:mm*!
shared_namedense_962/kernel
u
$dense_962/kernel/Read/ReadVariableOpReadVariableOpdense_962/kernel*
_output_shapes

:mm*
dtype0
t
dense_962/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*
shared_namedense_962/bias
m
"dense_962/bias/Read/ReadVariableOpReadVariableOpdense_962/bias*
_output_shapes
:m*
dtype0
?
batch_normalization_866/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*.
shared_namebatch_normalization_866/gamma
?
1batch_normalization_866/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_866/gamma*
_output_shapes
:m*
dtype0
?
batch_normalization_866/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*-
shared_namebatch_normalization_866/beta
?
0batch_normalization_866/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_866/beta*
_output_shapes
:m*
dtype0
?
#batch_normalization_866/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*4
shared_name%#batch_normalization_866/moving_mean
?
7batch_normalization_866/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_866/moving_mean*
_output_shapes
:m*
dtype0
?
'batch_normalization_866/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*8
shared_name)'batch_normalization_866/moving_variance
?
;batch_normalization_866/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_866/moving_variance*
_output_shapes
:m*
dtype0
|
dense_963/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:mm*!
shared_namedense_963/kernel
u
$dense_963/kernel/Read/ReadVariableOpReadVariableOpdense_963/kernel*
_output_shapes

:mm*
dtype0
t
dense_963/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*
shared_namedense_963/bias
m
"dense_963/bias/Read/ReadVariableOpReadVariableOpdense_963/bias*
_output_shapes
:m*
dtype0
?
batch_normalization_867/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*.
shared_namebatch_normalization_867/gamma
?
1batch_normalization_867/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_867/gamma*
_output_shapes
:m*
dtype0
?
batch_normalization_867/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*-
shared_namebatch_normalization_867/beta
?
0batch_normalization_867/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_867/beta*
_output_shapes
:m*
dtype0
?
#batch_normalization_867/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*4
shared_name%#batch_normalization_867/moving_mean
?
7batch_normalization_867/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_867/moving_mean*
_output_shapes
:m*
dtype0
?
'batch_normalization_867/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*8
shared_name)'batch_normalization_867/moving_variance
?
;batch_normalization_867/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_867/moving_variance*
_output_shapes
:m*
dtype0
|
dense_964/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:m.*!
shared_namedense_964/kernel
u
$dense_964/kernel/Read/ReadVariableOpReadVariableOpdense_964/kernel*
_output_shapes

:m.*
dtype0
t
dense_964/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*
shared_namedense_964/bias
m
"dense_964/bias/Read/ReadVariableOpReadVariableOpdense_964/bias*
_output_shapes
:.*
dtype0
?
batch_normalization_868/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*.
shared_namebatch_normalization_868/gamma
?
1batch_normalization_868/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_868/gamma*
_output_shapes
:.*
dtype0
?
batch_normalization_868/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*-
shared_namebatch_normalization_868/beta
?
0batch_normalization_868/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_868/beta*
_output_shapes
:.*
dtype0
?
#batch_normalization_868/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*4
shared_name%#batch_normalization_868/moving_mean
?
7batch_normalization_868/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_868/moving_mean*
_output_shapes
:.*
dtype0
?
'batch_normalization_868/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*8
shared_name)'batch_normalization_868/moving_variance
?
;batch_normalization_868/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_868/moving_variance*
_output_shapes
:.*
dtype0
|
dense_965/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:..*!
shared_namedense_965/kernel
u
$dense_965/kernel/Read/ReadVariableOpReadVariableOpdense_965/kernel*
_output_shapes

:..*
dtype0
t
dense_965/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*
shared_namedense_965/bias
m
"dense_965/bias/Read/ReadVariableOpReadVariableOpdense_965/bias*
_output_shapes
:.*
dtype0
?
batch_normalization_869/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*.
shared_namebatch_normalization_869/gamma
?
1batch_normalization_869/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_869/gamma*
_output_shapes
:.*
dtype0
?
batch_normalization_869/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*-
shared_namebatch_normalization_869/beta
?
0batch_normalization_869/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_869/beta*
_output_shapes
:.*
dtype0
?
#batch_normalization_869/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*4
shared_name%#batch_normalization_869/moving_mean
?
7batch_normalization_869/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_869/moving_mean*
_output_shapes
:.*
dtype0
?
'batch_normalization_869/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*8
shared_name)'batch_normalization_869/moving_variance
?
;batch_normalization_869/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_869/moving_variance*
_output_shapes
:.*
dtype0
|
dense_966/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:.]*!
shared_namedense_966/kernel
u
$dense_966/kernel/Read/ReadVariableOpReadVariableOpdense_966/kernel*
_output_shapes

:.]*
dtype0
t
dense_966/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*
shared_namedense_966/bias
m
"dense_966/bias/Read/ReadVariableOpReadVariableOpdense_966/bias*
_output_shapes
:]*
dtype0
?
batch_normalization_870/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*.
shared_namebatch_normalization_870/gamma
?
1batch_normalization_870/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_870/gamma*
_output_shapes
:]*
dtype0
?
batch_normalization_870/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*-
shared_namebatch_normalization_870/beta
?
0batch_normalization_870/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_870/beta*
_output_shapes
:]*
dtype0
?
#batch_normalization_870/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*4
shared_name%#batch_normalization_870/moving_mean
?
7batch_normalization_870/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_870/moving_mean*
_output_shapes
:]*
dtype0
?
'batch_normalization_870/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*8
shared_name)'batch_normalization_870/moving_variance
?
;batch_normalization_870/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_870/moving_variance*
_output_shapes
:]*
dtype0
|
dense_967/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:]*!
shared_namedense_967/kernel
u
$dense_967/kernel/Read/ReadVariableOpReadVariableOpdense_967/kernel*
_output_shapes

:]*
dtype0
t
dense_967/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_967/bias
m
"dense_967/bias/Read/ReadVariableOpReadVariableOpdense_967/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_961/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:m*(
shared_nameAdam/dense_961/kernel/m
?
+Adam/dense_961/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_961/kernel/m*
_output_shapes

:m*
dtype0
?
Adam/dense_961/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*&
shared_nameAdam/dense_961/bias/m
{
)Adam/dense_961/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_961/bias/m*
_output_shapes
:m*
dtype0
?
$Adam/batch_normalization_865/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*5
shared_name&$Adam/batch_normalization_865/gamma/m
?
8Adam/batch_normalization_865/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_865/gamma/m*
_output_shapes
:m*
dtype0
?
#Adam/batch_normalization_865/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*4
shared_name%#Adam/batch_normalization_865/beta/m
?
7Adam/batch_normalization_865/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_865/beta/m*
_output_shapes
:m*
dtype0
?
Adam/dense_962/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:mm*(
shared_nameAdam/dense_962/kernel/m
?
+Adam/dense_962/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_962/kernel/m*
_output_shapes

:mm*
dtype0
?
Adam/dense_962/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*&
shared_nameAdam/dense_962/bias/m
{
)Adam/dense_962/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_962/bias/m*
_output_shapes
:m*
dtype0
?
$Adam/batch_normalization_866/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*5
shared_name&$Adam/batch_normalization_866/gamma/m
?
8Adam/batch_normalization_866/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_866/gamma/m*
_output_shapes
:m*
dtype0
?
#Adam/batch_normalization_866/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*4
shared_name%#Adam/batch_normalization_866/beta/m
?
7Adam/batch_normalization_866/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_866/beta/m*
_output_shapes
:m*
dtype0
?
Adam/dense_963/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:mm*(
shared_nameAdam/dense_963/kernel/m
?
+Adam/dense_963/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_963/kernel/m*
_output_shapes

:mm*
dtype0
?
Adam/dense_963/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*&
shared_nameAdam/dense_963/bias/m
{
)Adam/dense_963/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_963/bias/m*
_output_shapes
:m*
dtype0
?
$Adam/batch_normalization_867/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*5
shared_name&$Adam/batch_normalization_867/gamma/m
?
8Adam/batch_normalization_867/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_867/gamma/m*
_output_shapes
:m*
dtype0
?
#Adam/batch_normalization_867/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*4
shared_name%#Adam/batch_normalization_867/beta/m
?
7Adam/batch_normalization_867/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_867/beta/m*
_output_shapes
:m*
dtype0
?
Adam/dense_964/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:m.*(
shared_nameAdam/dense_964/kernel/m
?
+Adam/dense_964/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_964/kernel/m*
_output_shapes

:m.*
dtype0
?
Adam/dense_964/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*&
shared_nameAdam/dense_964/bias/m
{
)Adam/dense_964/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_964/bias/m*
_output_shapes
:.*
dtype0
?
$Adam/batch_normalization_868/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*5
shared_name&$Adam/batch_normalization_868/gamma/m
?
8Adam/batch_normalization_868/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_868/gamma/m*
_output_shapes
:.*
dtype0
?
#Adam/batch_normalization_868/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*4
shared_name%#Adam/batch_normalization_868/beta/m
?
7Adam/batch_normalization_868/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_868/beta/m*
_output_shapes
:.*
dtype0
?
Adam/dense_965/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:..*(
shared_nameAdam/dense_965/kernel/m
?
+Adam/dense_965/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_965/kernel/m*
_output_shapes

:..*
dtype0
?
Adam/dense_965/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*&
shared_nameAdam/dense_965/bias/m
{
)Adam/dense_965/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_965/bias/m*
_output_shapes
:.*
dtype0
?
$Adam/batch_normalization_869/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*5
shared_name&$Adam/batch_normalization_869/gamma/m
?
8Adam/batch_normalization_869/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_869/gamma/m*
_output_shapes
:.*
dtype0
?
#Adam/batch_normalization_869/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*4
shared_name%#Adam/batch_normalization_869/beta/m
?
7Adam/batch_normalization_869/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_869/beta/m*
_output_shapes
:.*
dtype0
?
Adam/dense_966/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:.]*(
shared_nameAdam/dense_966/kernel/m
?
+Adam/dense_966/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_966/kernel/m*
_output_shapes

:.]*
dtype0
?
Adam/dense_966/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*&
shared_nameAdam/dense_966/bias/m
{
)Adam/dense_966/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_966/bias/m*
_output_shapes
:]*
dtype0
?
$Adam/batch_normalization_870/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*5
shared_name&$Adam/batch_normalization_870/gamma/m
?
8Adam/batch_normalization_870/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_870/gamma/m*
_output_shapes
:]*
dtype0
?
#Adam/batch_normalization_870/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*4
shared_name%#Adam/batch_normalization_870/beta/m
?
7Adam/batch_normalization_870/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_870/beta/m*
_output_shapes
:]*
dtype0
?
Adam/dense_967/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:]*(
shared_nameAdam/dense_967/kernel/m
?
+Adam/dense_967/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_967/kernel/m*
_output_shapes

:]*
dtype0
?
Adam/dense_967/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_967/bias/m
{
)Adam/dense_967/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_967/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_961/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:m*(
shared_nameAdam/dense_961/kernel/v
?
+Adam/dense_961/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_961/kernel/v*
_output_shapes

:m*
dtype0
?
Adam/dense_961/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*&
shared_nameAdam/dense_961/bias/v
{
)Adam/dense_961/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_961/bias/v*
_output_shapes
:m*
dtype0
?
$Adam/batch_normalization_865/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*5
shared_name&$Adam/batch_normalization_865/gamma/v
?
8Adam/batch_normalization_865/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_865/gamma/v*
_output_shapes
:m*
dtype0
?
#Adam/batch_normalization_865/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*4
shared_name%#Adam/batch_normalization_865/beta/v
?
7Adam/batch_normalization_865/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_865/beta/v*
_output_shapes
:m*
dtype0
?
Adam/dense_962/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:mm*(
shared_nameAdam/dense_962/kernel/v
?
+Adam/dense_962/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_962/kernel/v*
_output_shapes

:mm*
dtype0
?
Adam/dense_962/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*&
shared_nameAdam/dense_962/bias/v
{
)Adam/dense_962/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_962/bias/v*
_output_shapes
:m*
dtype0
?
$Adam/batch_normalization_866/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*5
shared_name&$Adam/batch_normalization_866/gamma/v
?
8Adam/batch_normalization_866/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_866/gamma/v*
_output_shapes
:m*
dtype0
?
#Adam/batch_normalization_866/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*4
shared_name%#Adam/batch_normalization_866/beta/v
?
7Adam/batch_normalization_866/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_866/beta/v*
_output_shapes
:m*
dtype0
?
Adam/dense_963/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:mm*(
shared_nameAdam/dense_963/kernel/v
?
+Adam/dense_963/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_963/kernel/v*
_output_shapes

:mm*
dtype0
?
Adam/dense_963/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*&
shared_nameAdam/dense_963/bias/v
{
)Adam/dense_963/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_963/bias/v*
_output_shapes
:m*
dtype0
?
$Adam/batch_normalization_867/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*5
shared_name&$Adam/batch_normalization_867/gamma/v
?
8Adam/batch_normalization_867/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_867/gamma/v*
_output_shapes
:m*
dtype0
?
#Adam/batch_normalization_867/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*4
shared_name%#Adam/batch_normalization_867/beta/v
?
7Adam/batch_normalization_867/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_867/beta/v*
_output_shapes
:m*
dtype0
?
Adam/dense_964/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:m.*(
shared_nameAdam/dense_964/kernel/v
?
+Adam/dense_964/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_964/kernel/v*
_output_shapes

:m.*
dtype0
?
Adam/dense_964/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*&
shared_nameAdam/dense_964/bias/v
{
)Adam/dense_964/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_964/bias/v*
_output_shapes
:.*
dtype0
?
$Adam/batch_normalization_868/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*5
shared_name&$Adam/batch_normalization_868/gamma/v
?
8Adam/batch_normalization_868/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_868/gamma/v*
_output_shapes
:.*
dtype0
?
#Adam/batch_normalization_868/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*4
shared_name%#Adam/batch_normalization_868/beta/v
?
7Adam/batch_normalization_868/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_868/beta/v*
_output_shapes
:.*
dtype0
?
Adam/dense_965/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:..*(
shared_nameAdam/dense_965/kernel/v
?
+Adam/dense_965/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_965/kernel/v*
_output_shapes

:..*
dtype0
?
Adam/dense_965/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*&
shared_nameAdam/dense_965/bias/v
{
)Adam/dense_965/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_965/bias/v*
_output_shapes
:.*
dtype0
?
$Adam/batch_normalization_869/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*5
shared_name&$Adam/batch_normalization_869/gamma/v
?
8Adam/batch_normalization_869/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_869/gamma/v*
_output_shapes
:.*
dtype0
?
#Adam/batch_normalization_869/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*4
shared_name%#Adam/batch_normalization_869/beta/v
?
7Adam/batch_normalization_869/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_869/beta/v*
_output_shapes
:.*
dtype0
?
Adam/dense_966/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:.]*(
shared_nameAdam/dense_966/kernel/v
?
+Adam/dense_966/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_966/kernel/v*
_output_shapes

:.]*
dtype0
?
Adam/dense_966/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*&
shared_nameAdam/dense_966/bias/v
{
)Adam/dense_966/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_966/bias/v*
_output_shapes
:]*
dtype0
?
$Adam/batch_normalization_870/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*5
shared_name&$Adam/batch_normalization_870/gamma/v
?
8Adam/batch_normalization_870/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_870/gamma/v*
_output_shapes
:]*
dtype0
?
#Adam/batch_normalization_870/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*4
shared_name%#Adam/batch_normalization_870/beta/v
?
7Adam/batch_normalization_870/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_870/beta/v*
_output_shapes
:]*
dtype0
?
Adam/dense_967/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:]*(
shared_nameAdam/dense_967/kernel/v
?
+Adam/dense_967/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_967/kernel/v*
_output_shapes

:]*
dtype0
?
Adam/dense_967/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_967/bias/v
{
)Adam/dense_967/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_967/bias/v*
_output_shapes
:*
dtype0
f
ConstConst*
_output_shapes

:*
dtype0*)
value B"UU?B  A  0@  XA
h
Const_1Const*
_output_shapes

:*
dtype0*)
value B"4sE ?B  @  yB

NoOpNoOp
??
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer-18
layer_with_weights-13
layer-19
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?

_keep_axis
_reduce_axis
 _reduce_axis_mask
!_broadcast_shape
"mean
"
adapt_mean
#variance
#adapt_variance
	$count
%	keras_api
&_adapt_function*
?

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
?
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses*
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 
?

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses*
?
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses*
?
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses* 
?

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses*
?
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses*
?
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 
?

rkernel
sbias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses*
?
zaxis
	{gamma
|beta
}moving_mean
~moving_variance
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?iter
?beta_1
?beta_2

?decay'm?(m?0m?1m?@m?Am?Im?Jm?Ym?Zm?bm?cm?rm?sm?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?'v?(v?0v?1v?@v?Av?Iv?Jv?Yv?Zv?bv?cv?rv?sv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?*
?
"0
#1
$2
'3
(4
05
16
27
38
@9
A10
I11
J12
K13
L14
Y15
Z16
b17
c18
d19
e20
r21
s22
{23
|24
}25
~26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40*
?
'0
(1
02
13
@4
A5
I6
J7
Y8
Z9
b10
c11
r12
s13
{14
|15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25*
2
?0
?1
?2
?3
?4
?5* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

?serving_default* 
* 
* 
* 
* 
RL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEcount5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
`Z
VARIABLE_VALUEdense_961/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_961/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_865/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_865/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_865/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE'batch_normalization_865/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
00
11
22
33*

00
11*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_962/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_962/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

@0
A1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_866/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_866/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_866/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE'batch_normalization_866/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
I0
J1
K2
L3*

I0
J1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_963/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_963/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_867/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_867/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_867/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE'batch_normalization_867/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
b0
c1
d2
e3*

b0
c1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_964/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_964/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

r0
s1*

r0
s1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_868/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_868/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_868/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE'batch_normalization_868/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
{0
|1
}2
~3*

{0
|1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_965/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_965/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_869/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_869/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_869/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUE'batch_normalization_869/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_966/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_966/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_870/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_870/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_870/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUE'batch_normalization_870/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_967/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_967/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
v
"0
#1
$2
23
34
K5
L6
d7
e8
}9
~10
?11
?12
?13
?14*
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19*

?0*
* 
* 
* 
* 
* 
* 


?0* 
* 

20
31*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


?0* 
* 

K0
L1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


?0* 
* 

d0
e1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


?0* 
* 

}0
~1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


?0* 
* 

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


?0* 
* 

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
?}
VARIABLE_VALUEAdam/dense_961/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_961/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_865/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_865/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_962/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_962/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_866/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_866/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_963/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_963/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_867/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_867/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_964/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_964/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_868/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_868/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_965/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_965/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_869/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_869/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/dense_966/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/dense_966/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_870/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_870/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/dense_967/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/dense_967/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_961/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_961/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_865/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_865/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_962/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_962/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_866/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_866/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_963/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_963/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_867/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_867/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_964/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_964/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_868/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_868/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_965/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_965/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_869/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_869/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/dense_966/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/dense_966/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_870/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_870/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/dense_967/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/dense_967/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
&serving_default_normalization_96_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_96_inputConstConst_1dense_961/kerneldense_961/bias'batch_normalization_865/moving_variancebatch_normalization_865/gamma#batch_normalization_865/moving_meanbatch_normalization_865/betadense_962/kerneldense_962/bias'batch_normalization_866/moving_variancebatch_normalization_866/gamma#batch_normalization_866/moving_meanbatch_normalization_866/betadense_963/kerneldense_963/bias'batch_normalization_867/moving_variancebatch_normalization_867/gamma#batch_normalization_867/moving_meanbatch_normalization_867/betadense_964/kerneldense_964/bias'batch_normalization_868/moving_variancebatch_normalization_868/gamma#batch_normalization_868/moving_meanbatch_normalization_868/betadense_965/kerneldense_965/bias'batch_normalization_869/moving_variancebatch_normalization_869/gamma#batch_normalization_869/moving_meanbatch_normalization_869/betadense_966/kerneldense_966/bias'batch_normalization_870/moving_variancebatch_normalization_870/gamma#batch_normalization_870/moving_meanbatch_normalization_870/betadense_967/kerneldense_967/bias*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 !"#$%&'(*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_864481
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_961/kernel/Read/ReadVariableOp"dense_961/bias/Read/ReadVariableOp1batch_normalization_865/gamma/Read/ReadVariableOp0batch_normalization_865/beta/Read/ReadVariableOp7batch_normalization_865/moving_mean/Read/ReadVariableOp;batch_normalization_865/moving_variance/Read/ReadVariableOp$dense_962/kernel/Read/ReadVariableOp"dense_962/bias/Read/ReadVariableOp1batch_normalization_866/gamma/Read/ReadVariableOp0batch_normalization_866/beta/Read/ReadVariableOp7batch_normalization_866/moving_mean/Read/ReadVariableOp;batch_normalization_866/moving_variance/Read/ReadVariableOp$dense_963/kernel/Read/ReadVariableOp"dense_963/bias/Read/ReadVariableOp1batch_normalization_867/gamma/Read/ReadVariableOp0batch_normalization_867/beta/Read/ReadVariableOp7batch_normalization_867/moving_mean/Read/ReadVariableOp;batch_normalization_867/moving_variance/Read/ReadVariableOp$dense_964/kernel/Read/ReadVariableOp"dense_964/bias/Read/ReadVariableOp1batch_normalization_868/gamma/Read/ReadVariableOp0batch_normalization_868/beta/Read/ReadVariableOp7batch_normalization_868/moving_mean/Read/ReadVariableOp;batch_normalization_868/moving_variance/Read/ReadVariableOp$dense_965/kernel/Read/ReadVariableOp"dense_965/bias/Read/ReadVariableOp1batch_normalization_869/gamma/Read/ReadVariableOp0batch_normalization_869/beta/Read/ReadVariableOp7batch_normalization_869/moving_mean/Read/ReadVariableOp;batch_normalization_869/moving_variance/Read/ReadVariableOp$dense_966/kernel/Read/ReadVariableOp"dense_966/bias/Read/ReadVariableOp1batch_normalization_870/gamma/Read/ReadVariableOp0batch_normalization_870/beta/Read/ReadVariableOp7batch_normalization_870/moving_mean/Read/ReadVariableOp;batch_normalization_870/moving_variance/Read/ReadVariableOp$dense_967/kernel/Read/ReadVariableOp"dense_967/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_961/kernel/m/Read/ReadVariableOp)Adam/dense_961/bias/m/Read/ReadVariableOp8Adam/batch_normalization_865/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_865/beta/m/Read/ReadVariableOp+Adam/dense_962/kernel/m/Read/ReadVariableOp)Adam/dense_962/bias/m/Read/ReadVariableOp8Adam/batch_normalization_866/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_866/beta/m/Read/ReadVariableOp+Adam/dense_963/kernel/m/Read/ReadVariableOp)Adam/dense_963/bias/m/Read/ReadVariableOp8Adam/batch_normalization_867/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_867/beta/m/Read/ReadVariableOp+Adam/dense_964/kernel/m/Read/ReadVariableOp)Adam/dense_964/bias/m/Read/ReadVariableOp8Adam/batch_normalization_868/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_868/beta/m/Read/ReadVariableOp+Adam/dense_965/kernel/m/Read/ReadVariableOp)Adam/dense_965/bias/m/Read/ReadVariableOp8Adam/batch_normalization_869/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_869/beta/m/Read/ReadVariableOp+Adam/dense_966/kernel/m/Read/ReadVariableOp)Adam/dense_966/bias/m/Read/ReadVariableOp8Adam/batch_normalization_870/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_870/beta/m/Read/ReadVariableOp+Adam/dense_967/kernel/m/Read/ReadVariableOp)Adam/dense_967/bias/m/Read/ReadVariableOp+Adam/dense_961/kernel/v/Read/ReadVariableOp)Adam/dense_961/bias/v/Read/ReadVariableOp8Adam/batch_normalization_865/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_865/beta/v/Read/ReadVariableOp+Adam/dense_962/kernel/v/Read/ReadVariableOp)Adam/dense_962/bias/v/Read/ReadVariableOp8Adam/batch_normalization_866/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_866/beta/v/Read/ReadVariableOp+Adam/dense_963/kernel/v/Read/ReadVariableOp)Adam/dense_963/bias/v/Read/ReadVariableOp8Adam/batch_normalization_867/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_867/beta/v/Read/ReadVariableOp+Adam/dense_964/kernel/v/Read/ReadVariableOp)Adam/dense_964/bias/v/Read/ReadVariableOp8Adam/batch_normalization_868/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_868/beta/v/Read/ReadVariableOp+Adam/dense_965/kernel/v/Read/ReadVariableOp)Adam/dense_965/bias/v/Read/ReadVariableOp8Adam/batch_normalization_869/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_869/beta/v/Read/ReadVariableOp+Adam/dense_966/kernel/v/Read/ReadVariableOp)Adam/dense_966/bias/v/Read/ReadVariableOp8Adam/batch_normalization_870/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_870/beta/v/Read/ReadVariableOp+Adam/dense_967/kernel/v/Read/ReadVariableOp)Adam/dense_967/bias/v/Read/ReadVariableOpConst_2*p
Tini
g2e		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_865661
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_961/kerneldense_961/biasbatch_normalization_865/gammabatch_normalization_865/beta#batch_normalization_865/moving_mean'batch_normalization_865/moving_variancedense_962/kerneldense_962/biasbatch_normalization_866/gammabatch_normalization_866/beta#batch_normalization_866/moving_mean'batch_normalization_866/moving_variancedense_963/kerneldense_963/biasbatch_normalization_867/gammabatch_normalization_867/beta#batch_normalization_867/moving_mean'batch_normalization_867/moving_variancedense_964/kerneldense_964/biasbatch_normalization_868/gammabatch_normalization_868/beta#batch_normalization_868/moving_mean'batch_normalization_868/moving_variancedense_965/kerneldense_965/biasbatch_normalization_869/gammabatch_normalization_869/beta#batch_normalization_869/moving_mean'batch_normalization_869/moving_variancedense_966/kerneldense_966/biasbatch_normalization_870/gammabatch_normalization_870/beta#batch_normalization_870/moving_mean'batch_normalization_870/moving_variancedense_967/kerneldense_967/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_961/kernel/mAdam/dense_961/bias/m$Adam/batch_normalization_865/gamma/m#Adam/batch_normalization_865/beta/mAdam/dense_962/kernel/mAdam/dense_962/bias/m$Adam/batch_normalization_866/gamma/m#Adam/batch_normalization_866/beta/mAdam/dense_963/kernel/mAdam/dense_963/bias/m$Adam/batch_normalization_867/gamma/m#Adam/batch_normalization_867/beta/mAdam/dense_964/kernel/mAdam/dense_964/bias/m$Adam/batch_normalization_868/gamma/m#Adam/batch_normalization_868/beta/mAdam/dense_965/kernel/mAdam/dense_965/bias/m$Adam/batch_normalization_869/gamma/m#Adam/batch_normalization_869/beta/mAdam/dense_966/kernel/mAdam/dense_966/bias/m$Adam/batch_normalization_870/gamma/m#Adam/batch_normalization_870/beta/mAdam/dense_967/kernel/mAdam/dense_967/bias/mAdam/dense_961/kernel/vAdam/dense_961/bias/v$Adam/batch_normalization_865/gamma/v#Adam/batch_normalization_865/beta/vAdam/dense_962/kernel/vAdam/dense_962/bias/v$Adam/batch_normalization_866/gamma/v#Adam/batch_normalization_866/beta/vAdam/dense_963/kernel/vAdam/dense_963/bias/v$Adam/batch_normalization_867/gamma/v#Adam/batch_normalization_867/beta/vAdam/dense_964/kernel/vAdam/dense_964/bias/v$Adam/batch_normalization_868/gamma/v#Adam/batch_normalization_868/beta/vAdam/dense_965/kernel/vAdam/dense_965/bias/v$Adam/batch_normalization_869/gamma/v#Adam/batch_normalization_869/beta/vAdam/dense_966/kernel/vAdam/dense_966/bias/v$Adam/batch_normalization_870/gamma/v#Adam/batch_normalization_870/beta/vAdam/dense_967/kernel/vAdam/dense_967/bias/v*o
Tinh
f2d*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_865968??
?
?
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_864605

inputs/
!batchnorm_readvariableop_resource:m3
%batchnorm_mul_readvariableop_resource:m1
#batchnorm_readvariableop_1_resource:m1
#batchnorm_readvariableop_2_resource:m
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:mP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:mc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????mz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:m*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:mz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:m*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:mr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????mb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_862296

inputs5
'assignmovingavg_readvariableop_resource:m7
)assignmovingavg_1_readvariableop_resource:m3
%batchnorm_mul_readvariableop_resource:m/
!batchnorm_readvariableop_resource:m
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:m?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????ml
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:m*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:mx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:m?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:m*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:m~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:m?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:mP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:mc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????mh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:mv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:mr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????mb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
*__inference_dense_967_layer_call_fn_865263

inputs
unknown:]
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_967_layer_call_and_return_conditional_losses_862805o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????]: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_867_layer_call_and_return_conditional_losses_862679

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????m*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????m"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????m:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_864968

inputs/
!batchnorm_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:.1
#batchnorm_readvariableop_1_resource:.1
#batchnorm_readvariableop_2_resource:.
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:.P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:.~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:.*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:.z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:.*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:.r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????.?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_866_layer_call_and_return_conditional_losses_862641

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????m*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????m"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????m:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_864881

inputs5
'assignmovingavg_readvariableop_resource:m7
)assignmovingavg_1_readvariableop_resource:m3
%batchnorm_mul_readvariableop_resource:m/
!batchnorm_readvariableop_resource:m
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:m?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????ml
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:m*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:mx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:m?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:m*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:m~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:m?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:mP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:mc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????mh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:mv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:mr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????mb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
*__inference_dense_964_layer_call_fn_864906

inputs
unknown:m.
	unknown_0:.
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_964_layer_call_and_return_conditional_losses_862697o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????.`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????m: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
*__inference_dense_961_layer_call_fn_864543

inputs
unknown:m
	unknown_0:m
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_961_layer_call_and_return_conditional_losses_862583o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????m`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_865002

inputs5
'assignmovingavg_readvariableop_resource:.7
)assignmovingavg_1_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:./
!batchnorm_readvariableop_resource:.
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:.?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????.l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:.*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:.x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:.?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:.*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:.~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:.?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:.P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:.~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:.v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:.r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????.?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_869_layer_call_fn_865069

inputs
unknown:.
	unknown_0:.
	unknown_1:.
	unknown_2:.
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_862460o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????.`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
E__inference_dense_961_layer_call_and_return_conditional_losses_864559

inputs0
matmul_readvariableop_resource:m-
biasadd_readvariableop_resource:m
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_961/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
2dense_961/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#dense_961/kernel/Regularizer/SquareSquare:dense_961/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ms
"dense_961/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_961/kernel/Regularizer/SumSum'dense_961/kernel/Regularizer/Square:y:0+dense_961/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_961/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_961/kernel/Regularizer/mulMul+dense_961/kernel/Regularizer/mul/x:output:0)dense_961/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_961/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_961/kernel/Regularizer/Square/ReadVariableOp2dense_961/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_866_layer_call_fn_864765

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_866_layer_call_and_return_conditional_losses_862641`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????m"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????m:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
??
?
I__inference_sequential_96_layer_call_and_return_conditional_losses_863576
normalization_96_input
normalization_96_sub_y
normalization_96_sqrt_x"
dense_961_863444:m
dense_961_863446:m,
batch_normalization_865_863449:m,
batch_normalization_865_863451:m,
batch_normalization_865_863453:m,
batch_normalization_865_863455:m"
dense_962_863459:mm
dense_962_863461:m,
batch_normalization_866_863464:m,
batch_normalization_866_863466:m,
batch_normalization_866_863468:m,
batch_normalization_866_863470:m"
dense_963_863474:mm
dense_963_863476:m,
batch_normalization_867_863479:m,
batch_normalization_867_863481:m,
batch_normalization_867_863483:m,
batch_normalization_867_863485:m"
dense_964_863489:m.
dense_964_863491:.,
batch_normalization_868_863494:.,
batch_normalization_868_863496:.,
batch_normalization_868_863498:.,
batch_normalization_868_863500:."
dense_965_863504:..
dense_965_863506:.,
batch_normalization_869_863509:.,
batch_normalization_869_863511:.,
batch_normalization_869_863513:.,
batch_normalization_869_863515:."
dense_966_863519:.]
dense_966_863521:],
batch_normalization_870_863524:],
batch_normalization_870_863526:],
batch_normalization_870_863528:],
batch_normalization_870_863530:]"
dense_967_863534:]
dense_967_863536:
identity??/batch_normalization_865/StatefulPartitionedCall?/batch_normalization_866/StatefulPartitionedCall?/batch_normalization_867/StatefulPartitionedCall?/batch_normalization_868/StatefulPartitionedCall?/batch_normalization_869/StatefulPartitionedCall?/batch_normalization_870/StatefulPartitionedCall?!dense_961/StatefulPartitionedCall?2dense_961/kernel/Regularizer/Square/ReadVariableOp?!dense_962/StatefulPartitionedCall?2dense_962/kernel/Regularizer/Square/ReadVariableOp?!dense_963/StatefulPartitionedCall?2dense_963/kernel/Regularizer/Square/ReadVariableOp?!dense_964/StatefulPartitionedCall?2dense_964/kernel/Regularizer/Square/ReadVariableOp?!dense_965/StatefulPartitionedCall?2dense_965/kernel/Regularizer/Square/ReadVariableOp?!dense_966/StatefulPartitionedCall?2dense_966/kernel/Regularizer/Square/ReadVariableOp?!dense_967/StatefulPartitionedCall}
normalization_96/subSubnormalization_96_inputnormalization_96_sub_y*
T0*'
_output_shapes
:?????????_
normalization_96/SqrtSqrtnormalization_96_sqrt_x*
T0*
_output_shapes

:_
normalization_96/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_96/MaximumMaximumnormalization_96/Sqrt:y:0#normalization_96/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_96/truedivRealDivnormalization_96/sub:z:0normalization_96/Maximum:z:0*
T0*'
_output_shapes
:??????????
!dense_961/StatefulPartitionedCallStatefulPartitionedCallnormalization_96/truediv:z:0dense_961_863444dense_961_863446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_961_layer_call_and_return_conditional_losses_862583?
/batch_normalization_865/StatefulPartitionedCallStatefulPartitionedCall*dense_961/StatefulPartitionedCall:output:0batch_normalization_865_863449batch_normalization_865_863451batch_normalization_865_863453batch_normalization_865_863455*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_862085?
leaky_re_lu_865/PartitionedCallPartitionedCall8batch_normalization_865/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_862603?
!dense_962/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_865/PartitionedCall:output:0dense_962_863459dense_962_863461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_962_layer_call_and_return_conditional_losses_862621?
/batch_normalization_866/StatefulPartitionedCallStatefulPartitionedCall*dense_962/StatefulPartitionedCall:output:0batch_normalization_866_863464batch_normalization_866_863466batch_normalization_866_863468batch_normalization_866_863470*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_862167?
leaky_re_lu_866/PartitionedCallPartitionedCall8batch_normalization_866/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_866_layer_call_and_return_conditional_losses_862641?
!dense_963/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_866/PartitionedCall:output:0dense_963_863474dense_963_863476*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_963_layer_call_and_return_conditional_losses_862659?
/batch_normalization_867/StatefulPartitionedCallStatefulPartitionedCall*dense_963/StatefulPartitionedCall:output:0batch_normalization_867_863479batch_normalization_867_863481batch_normalization_867_863483batch_normalization_867_863485*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_862249?
leaky_re_lu_867/PartitionedCallPartitionedCall8batch_normalization_867/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_867_layer_call_and_return_conditional_losses_862679?
!dense_964/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_867/PartitionedCall:output:0dense_964_863489dense_964_863491*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_964_layer_call_and_return_conditional_losses_862697?
/batch_normalization_868/StatefulPartitionedCallStatefulPartitionedCall*dense_964/StatefulPartitionedCall:output:0batch_normalization_868_863494batch_normalization_868_863496batch_normalization_868_863498batch_normalization_868_863500*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_862331?
leaky_re_lu_868/PartitionedCallPartitionedCall8batch_normalization_868/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_868_layer_call_and_return_conditional_losses_862717?
!dense_965/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_868/PartitionedCall:output:0dense_965_863504dense_965_863506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_965_layer_call_and_return_conditional_losses_862735?
/batch_normalization_869/StatefulPartitionedCallStatefulPartitionedCall*dense_965/StatefulPartitionedCall:output:0batch_normalization_869_863509batch_normalization_869_863511batch_normalization_869_863513batch_normalization_869_863515*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_862413?
leaky_re_lu_869/PartitionedCallPartitionedCall8batch_normalization_869/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_869_layer_call_and_return_conditional_losses_862755?
!dense_966/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_869/PartitionedCall:output:0dense_966_863519dense_966_863521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_966_layer_call_and_return_conditional_losses_862773?
/batch_normalization_870/StatefulPartitionedCallStatefulPartitionedCall*dense_966/StatefulPartitionedCall:output:0batch_normalization_870_863524batch_normalization_870_863526batch_normalization_870_863528batch_normalization_870_863530*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_862495?
leaky_re_lu_870/PartitionedCallPartitionedCall8batch_normalization_870/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_870_layer_call_and_return_conditional_losses_862793?
!dense_967/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_870/PartitionedCall:output:0dense_967_863534dense_967_863536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_967_layer_call_and_return_conditional_losses_862805?
2dense_961/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_961_863444*
_output_shapes

:m*
dtype0?
#dense_961/kernel/Regularizer/SquareSquare:dense_961/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ms
"dense_961/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_961/kernel/Regularizer/SumSum'dense_961/kernel/Regularizer/Square:y:0+dense_961/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_961/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_961/kernel/Regularizer/mulMul+dense_961/kernel/Regularizer/mul/x:output:0)dense_961/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_962/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_962_863459*
_output_shapes

:mm*
dtype0?
#dense_962/kernel/Regularizer/SquareSquare:dense_962/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_962/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_962/kernel/Regularizer/SumSum'dense_962/kernel/Regularizer/Square:y:0+dense_962/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_962/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_962/kernel/Regularizer/mulMul+dense_962/kernel/Regularizer/mul/x:output:0)dense_962/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_963/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_963_863474*
_output_shapes

:mm*
dtype0?
#dense_963/kernel/Regularizer/SquareSquare:dense_963/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_963/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_963/kernel/Regularizer/SumSum'dense_963/kernel/Regularizer/Square:y:0+dense_963/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_963/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_963/kernel/Regularizer/mulMul+dense_963/kernel/Regularizer/mul/x:output:0)dense_963/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_964/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_964_863489*
_output_shapes

:m.*
dtype0?
#dense_964/kernel/Regularizer/SquareSquare:dense_964/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m.s
"dense_964/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_964/kernel/Regularizer/SumSum'dense_964/kernel/Regularizer/Square:y:0+dense_964/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_964/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_964/kernel/Regularizer/mulMul+dense_964/kernel/Regularizer/mul/x:output:0)dense_964/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_965/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_965_863504*
_output_shapes

:..*
dtype0?
#dense_965/kernel/Regularizer/SquareSquare:dense_965/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:..s
"dense_965/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_965/kernel/Regularizer/SumSum'dense_965/kernel/Regularizer/Square:y:0+dense_965/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_965/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_965/kernel/Regularizer/mulMul+dense_965/kernel/Regularizer/mul/x:output:0)dense_965/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_966/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_966_863519*
_output_shapes

:.]*
dtype0?
#dense_966/kernel/Regularizer/SquareSquare:dense_966/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:.]s
"dense_966/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_966/kernel/Regularizer/SumSum'dense_966/kernel/Regularizer/Square:y:0+dense_966/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_966/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?<?
 dense_966/kernel/Regularizer/mulMul+dense_966/kernel/Regularizer/mul/x:output:0)dense_966/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_967/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp0^batch_normalization_865/StatefulPartitionedCall0^batch_normalization_866/StatefulPartitionedCall0^batch_normalization_867/StatefulPartitionedCall0^batch_normalization_868/StatefulPartitionedCall0^batch_normalization_869/StatefulPartitionedCall0^batch_normalization_870/StatefulPartitionedCall"^dense_961/StatefulPartitionedCall3^dense_961/kernel/Regularizer/Square/ReadVariableOp"^dense_962/StatefulPartitionedCall3^dense_962/kernel/Regularizer/Square/ReadVariableOp"^dense_963/StatefulPartitionedCall3^dense_963/kernel/Regularizer/Square/ReadVariableOp"^dense_964/StatefulPartitionedCall3^dense_964/kernel/Regularizer/Square/ReadVariableOp"^dense_965/StatefulPartitionedCall3^dense_965/kernel/Regularizer/Square/ReadVariableOp"^dense_966/StatefulPartitionedCall3^dense_966/kernel/Regularizer/Square/ReadVariableOp"^dense_967/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_865/StatefulPartitionedCall/batch_normalization_865/StatefulPartitionedCall2b
/batch_normalization_866/StatefulPartitionedCall/batch_normalization_866/StatefulPartitionedCall2b
/batch_normalization_867/StatefulPartitionedCall/batch_normalization_867/StatefulPartitionedCall2b
/batch_normalization_868/StatefulPartitionedCall/batch_normalization_868/StatefulPartitionedCall2b
/batch_normalization_869/StatefulPartitionedCall/batch_normalization_869/StatefulPartitionedCall2b
/batch_normalization_870/StatefulPartitionedCall/batch_normalization_870/StatefulPartitionedCall2F
!dense_961/StatefulPartitionedCall!dense_961/StatefulPartitionedCall2h
2dense_961/kernel/Regularizer/Square/ReadVariableOp2dense_961/kernel/Regularizer/Square/ReadVariableOp2F
!dense_962/StatefulPartitionedCall!dense_962/StatefulPartitionedCall2h
2dense_962/kernel/Regularizer/Square/ReadVariableOp2dense_962/kernel/Regularizer/Square/ReadVariableOp2F
!dense_963/StatefulPartitionedCall!dense_963/StatefulPartitionedCall2h
2dense_963/kernel/Regularizer/Square/ReadVariableOp2dense_963/kernel/Regularizer/Square/ReadVariableOp2F
!dense_964/StatefulPartitionedCall!dense_964/StatefulPartitionedCall2h
2dense_964/kernel/Regularizer/Square/ReadVariableOp2dense_964/kernel/Regularizer/Square/ReadVariableOp2F
!dense_965/StatefulPartitionedCall!dense_965/StatefulPartitionedCall2h
2dense_965/kernel/Regularizer/Square/ReadVariableOp2dense_965/kernel/Regularizer/Square/ReadVariableOp2F
!dense_966/StatefulPartitionedCall!dense_966/StatefulPartitionedCall2h
2dense_966/kernel/Regularizer/Square/ReadVariableOp2dense_966/kernel/Regularizer/Square/ReadVariableOp2F
!dense_967/StatefulPartitionedCall!dense_967/StatefulPartitionedCall:_ [
'
_output_shapes
:?????????
0
_user_specified_namenormalization_96_input:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
__inference_loss_fn_2_865306M
;dense_963_kernel_regularizer_square_readvariableop_resource:mm
identity??2dense_963/kernel/Regularizer/Square/ReadVariableOp?
2dense_963/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_963_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:mm*
dtype0?
#dense_963/kernel/Regularizer/SquareSquare:dense_963/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_963/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_963/kernel/Regularizer/SumSum'dense_963/kernel/Regularizer/Square:y:0+dense_963/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_963/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_963/kernel/Regularizer/mulMul+dense_963/kernel/Regularizer/mul/x:output:0)dense_963/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_963/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_963/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_963/kernel/Regularizer/Square/ReadVariableOp2dense_963/kernel/Regularizer/Square/ReadVariableOp
?
?
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_862331

inputs/
!batchnorm_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:.1
#batchnorm_readvariableop_1_resource:.1
#batchnorm_readvariableop_2_resource:.
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:.P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:.~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:.*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:.z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:.*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:.r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????.?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
.__inference_sequential_96_layer_call_fn_863843

inputs
unknown
	unknown_0
	unknown_1:m
	unknown_2:m
	unknown_3:m
	unknown_4:m
	unknown_5:m
	unknown_6:m
	unknown_7:mm
	unknown_8:m
	unknown_9:m

unknown_10:m

unknown_11:m

unknown_12:m

unknown_13:mm

unknown_14:m

unknown_15:m

unknown_16:m

unknown_17:m

unknown_18:m

unknown_19:m.

unknown_20:.

unknown_21:.

unknown_22:.

unknown_23:.

unknown_24:.

unknown_25:..

unknown_26:.

unknown_27:.

unknown_28:.

unknown_29:.

unknown_30:.

unknown_31:.]

unknown_32:]

unknown_33:]

unknown_34:]

unknown_35:]

unknown_36:]

unknown_37:]

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 !"#$%&'(*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_96_layer_call_and_return_conditional_losses_862848o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
E__inference_dense_965_layer_call_and_return_conditional_losses_862735

inputs0
matmul_readvariableop_resource:..-
biasadd_readvariableop_resource:.
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_965/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:..*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:.*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
2dense_965/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:..*
dtype0?
#dense_965/kernel/Regularizer/SquareSquare:dense_965/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:..s
"dense_965/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_965/kernel/Regularizer/SumSum'dense_965/kernel/Regularizer/Square:y:0+dense_965/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_965/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_965/kernel/Regularizer/mulMul+dense_965/kernel/Regularizer/mul/x:output:0)dense_965/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????.?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_965/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_965/kernel/Regularizer/Square/ReadVariableOp2dense_965/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_865295M
;dense_962_kernel_regularizer_square_readvariableop_resource:mm
identity??2dense_962/kernel/Regularizer/Square/ReadVariableOp?
2dense_962/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_962_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:mm*
dtype0?
#dense_962/kernel/Regularizer/SquareSquare:dense_962/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_962/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_962/kernel/Regularizer/SumSum'dense_962/kernel/Regularizer/Square:y:0+dense_962/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_962/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_962/kernel/Regularizer/mulMul+dense_962/kernel/Regularizer/mul/x:output:0)dense_962/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_962/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_962/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_962/kernel/Regularizer/Square/ReadVariableOp2dense_962/kernel/Regularizer/Square/ReadVariableOp
?
L
0__inference_leaky_re_lu_870_layer_call_fn_865249

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_870_layer_call_and_return_conditional_losses_862793`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????]"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????]:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
E__inference_dense_962_layer_call_and_return_conditional_losses_862621

inputs0
matmul_readvariableop_resource:mm-
biasadd_readvariableop_resource:m
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_962/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:mm*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
2dense_962/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
#dense_962/kernel/Regularizer/SquareSquare:dense_962/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_962/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_962/kernel/Regularizer/SumSum'dense_962/kernel/Regularizer/Square:y:0+dense_962/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_962/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_962/kernel/Regularizer/mulMul+dense_962/kernel/Regularizer/mul/x:output:0)dense_962/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_962/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????m: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_962/kernel/Regularizer/Square/ReadVariableOp2dense_962/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_868_layer_call_and_return_conditional_losses_865012

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????.*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????.:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
*__inference_dense_963_layer_call_fn_864785

inputs
unknown:mm
	unknown_0:m
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_963_layer_call_and_return_conditional_losses_862659o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????m`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????m: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_865_layer_call_fn_864572

inputs
unknown:m
	unknown_0:m
	unknown_1:m
	unknown_2:m
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_862085o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????m`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_867_layer_call_fn_864886

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_867_layer_call_and_return_conditional_losses_862679`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????m"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????m:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_866_layer_call_fn_864706

inputs
unknown:m
	unknown_0:m
	unknown_1:m
	unknown_2:m
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_862214o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????m`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
E__inference_dense_964_layer_call_and_return_conditional_losses_864922

inputs0
matmul_readvariableop_resource:m.-
biasadd_readvariableop_resource:.
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_964/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m.*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:.*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
2dense_964/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m.*
dtype0?
#dense_964/kernel/Regularizer/SquareSquare:dense_964/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m.s
"dense_964/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_964/kernel/Regularizer/SumSum'dense_964/kernel/Regularizer/Square:y:0+dense_964/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_964/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_964/kernel/Regularizer/mulMul+dense_964/kernel/Regularizer/mul/x:output:0)dense_964/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????.?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_964/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????m: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_964/kernel/Regularizer/Square/ReadVariableOp2dense_964/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
E__inference_dense_966_layer_call_and_return_conditional_losses_865164

inputs0
matmul_readvariableop_resource:.]-
biasadd_readvariableop_resource:]
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_966/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:.]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:]*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]?
2dense_966/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:.]*
dtype0?
#dense_966/kernel/Regularizer/SquareSquare:dense_966/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:.]s
"dense_966/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_966/kernel/Regularizer/SumSum'dense_966/kernel/Regularizer/Square:y:0+dense_966/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_966/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?<?
 dense_966/kernel/Regularizer/mulMul+dense_966/kernel/Regularizer/mul/x:output:0)dense_966/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????]?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_966/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_966/kernel/Regularizer/Square/ReadVariableOp2dense_966/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_865328M
;dense_965_kernel_regularizer_square_readvariableop_resource:..
identity??2dense_965/kernel/Regularizer/Square/ReadVariableOp?
2dense_965/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_965_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:..*
dtype0?
#dense_965/kernel/Regularizer/SquareSquare:dense_965/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:..s
"dense_965/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_965/kernel/Regularizer/SumSum'dense_965/kernel/Regularizer/Square:y:0+dense_965/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_965/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_965/kernel/Regularizer/mulMul+dense_965/kernel/Regularizer/mul/x:output:0)dense_965/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_965/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_965/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_965/kernel/Regularizer/Square/ReadVariableOp2dense_965/kernel/Regularizer/Square/ReadVariableOp
??
?+
!__inference__wrapped_model_862061
normalization_96_input(
$sequential_96_normalization_96_sub_y)
%sequential_96_normalization_96_sqrt_xH
6sequential_96_dense_961_matmul_readvariableop_resource:mE
7sequential_96_dense_961_biasadd_readvariableop_resource:mU
Gsequential_96_batch_normalization_865_batchnorm_readvariableop_resource:mY
Ksequential_96_batch_normalization_865_batchnorm_mul_readvariableop_resource:mW
Isequential_96_batch_normalization_865_batchnorm_readvariableop_1_resource:mW
Isequential_96_batch_normalization_865_batchnorm_readvariableop_2_resource:mH
6sequential_96_dense_962_matmul_readvariableop_resource:mmE
7sequential_96_dense_962_biasadd_readvariableop_resource:mU
Gsequential_96_batch_normalization_866_batchnorm_readvariableop_resource:mY
Ksequential_96_batch_normalization_866_batchnorm_mul_readvariableop_resource:mW
Isequential_96_batch_normalization_866_batchnorm_readvariableop_1_resource:mW
Isequential_96_batch_normalization_866_batchnorm_readvariableop_2_resource:mH
6sequential_96_dense_963_matmul_readvariableop_resource:mmE
7sequential_96_dense_963_biasadd_readvariableop_resource:mU
Gsequential_96_batch_normalization_867_batchnorm_readvariableop_resource:mY
Ksequential_96_batch_normalization_867_batchnorm_mul_readvariableop_resource:mW
Isequential_96_batch_normalization_867_batchnorm_readvariableop_1_resource:mW
Isequential_96_batch_normalization_867_batchnorm_readvariableop_2_resource:mH
6sequential_96_dense_964_matmul_readvariableop_resource:m.E
7sequential_96_dense_964_biasadd_readvariableop_resource:.U
Gsequential_96_batch_normalization_868_batchnorm_readvariableop_resource:.Y
Ksequential_96_batch_normalization_868_batchnorm_mul_readvariableop_resource:.W
Isequential_96_batch_normalization_868_batchnorm_readvariableop_1_resource:.W
Isequential_96_batch_normalization_868_batchnorm_readvariableop_2_resource:.H
6sequential_96_dense_965_matmul_readvariableop_resource:..E
7sequential_96_dense_965_biasadd_readvariableop_resource:.U
Gsequential_96_batch_normalization_869_batchnorm_readvariableop_resource:.Y
Ksequential_96_batch_normalization_869_batchnorm_mul_readvariableop_resource:.W
Isequential_96_batch_normalization_869_batchnorm_readvariableop_1_resource:.W
Isequential_96_batch_normalization_869_batchnorm_readvariableop_2_resource:.H
6sequential_96_dense_966_matmul_readvariableop_resource:.]E
7sequential_96_dense_966_biasadd_readvariableop_resource:]U
Gsequential_96_batch_normalization_870_batchnorm_readvariableop_resource:]Y
Ksequential_96_batch_normalization_870_batchnorm_mul_readvariableop_resource:]W
Isequential_96_batch_normalization_870_batchnorm_readvariableop_1_resource:]W
Isequential_96_batch_normalization_870_batchnorm_readvariableop_2_resource:]H
6sequential_96_dense_967_matmul_readvariableop_resource:]E
7sequential_96_dense_967_biasadd_readvariableop_resource:
identity??>sequential_96/batch_normalization_865/batchnorm/ReadVariableOp?@sequential_96/batch_normalization_865/batchnorm/ReadVariableOp_1?@sequential_96/batch_normalization_865/batchnorm/ReadVariableOp_2?Bsequential_96/batch_normalization_865/batchnorm/mul/ReadVariableOp?>sequential_96/batch_normalization_866/batchnorm/ReadVariableOp?@sequential_96/batch_normalization_866/batchnorm/ReadVariableOp_1?@sequential_96/batch_normalization_866/batchnorm/ReadVariableOp_2?Bsequential_96/batch_normalization_866/batchnorm/mul/ReadVariableOp?>sequential_96/batch_normalization_867/batchnorm/ReadVariableOp?@sequential_96/batch_normalization_867/batchnorm/ReadVariableOp_1?@sequential_96/batch_normalization_867/batchnorm/ReadVariableOp_2?Bsequential_96/batch_normalization_867/batchnorm/mul/ReadVariableOp?>sequential_96/batch_normalization_868/batchnorm/ReadVariableOp?@sequential_96/batch_normalization_868/batchnorm/ReadVariableOp_1?@sequential_96/batch_normalization_868/batchnorm/ReadVariableOp_2?Bsequential_96/batch_normalization_868/batchnorm/mul/ReadVariableOp?>sequential_96/batch_normalization_869/batchnorm/ReadVariableOp?@sequential_96/batch_normalization_869/batchnorm/ReadVariableOp_1?@sequential_96/batch_normalization_869/batchnorm/ReadVariableOp_2?Bsequential_96/batch_normalization_869/batchnorm/mul/ReadVariableOp?>sequential_96/batch_normalization_870/batchnorm/ReadVariableOp?@sequential_96/batch_normalization_870/batchnorm/ReadVariableOp_1?@sequential_96/batch_normalization_870/batchnorm/ReadVariableOp_2?Bsequential_96/batch_normalization_870/batchnorm/mul/ReadVariableOp?.sequential_96/dense_961/BiasAdd/ReadVariableOp?-sequential_96/dense_961/MatMul/ReadVariableOp?.sequential_96/dense_962/BiasAdd/ReadVariableOp?-sequential_96/dense_962/MatMul/ReadVariableOp?.sequential_96/dense_963/BiasAdd/ReadVariableOp?-sequential_96/dense_963/MatMul/ReadVariableOp?.sequential_96/dense_964/BiasAdd/ReadVariableOp?-sequential_96/dense_964/MatMul/ReadVariableOp?.sequential_96/dense_965/BiasAdd/ReadVariableOp?-sequential_96/dense_965/MatMul/ReadVariableOp?.sequential_96/dense_966/BiasAdd/ReadVariableOp?-sequential_96/dense_966/MatMul/ReadVariableOp?.sequential_96/dense_967/BiasAdd/ReadVariableOp?-sequential_96/dense_967/MatMul/ReadVariableOp?
"sequential_96/normalization_96/subSubnormalization_96_input$sequential_96_normalization_96_sub_y*
T0*'
_output_shapes
:?????????{
#sequential_96/normalization_96/SqrtSqrt%sequential_96_normalization_96_sqrt_x*
T0*
_output_shapes

:m
(sequential_96/normalization_96/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
&sequential_96/normalization_96/MaximumMaximum'sequential_96/normalization_96/Sqrt:y:01sequential_96/normalization_96/Maximum/y:output:0*
T0*
_output_shapes

:?
&sequential_96/normalization_96/truedivRealDiv&sequential_96/normalization_96/sub:z:0*sequential_96/normalization_96/Maximum:z:0*
T0*'
_output_shapes
:??????????
-sequential_96/dense_961/MatMul/ReadVariableOpReadVariableOp6sequential_96_dense_961_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
sequential_96/dense_961/MatMulMatMul*sequential_96/normalization_96/truediv:z:05sequential_96/dense_961/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
.sequential_96/dense_961/BiasAdd/ReadVariableOpReadVariableOp7sequential_96_dense_961_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
sequential_96/dense_961/BiasAddBiasAdd(sequential_96/dense_961/MatMul:product:06sequential_96/dense_961/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
>sequential_96/batch_normalization_865/batchnorm/ReadVariableOpReadVariableOpGsequential_96_batch_normalization_865_batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0z
5sequential_96/batch_normalization_865/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
3sequential_96/batch_normalization_865/batchnorm/addAddV2Fsequential_96/batch_normalization_865/batchnorm/ReadVariableOp:value:0>sequential_96/batch_normalization_865/batchnorm/add/y:output:0*
T0*
_output_shapes
:m?
5sequential_96/batch_normalization_865/batchnorm/RsqrtRsqrt7sequential_96/batch_normalization_865/batchnorm/add:z:0*
T0*
_output_shapes
:m?
Bsequential_96/batch_normalization_865/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_96_batch_normalization_865_batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0?
3sequential_96/batch_normalization_865/batchnorm/mulMul9sequential_96/batch_normalization_865/batchnorm/Rsqrt:y:0Jsequential_96/batch_normalization_865/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:m?
5sequential_96/batch_normalization_865/batchnorm/mul_1Mul(sequential_96/dense_961/BiasAdd:output:07sequential_96/batch_normalization_865/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????m?
@sequential_96/batch_normalization_865/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_96_batch_normalization_865_batchnorm_readvariableop_1_resource*
_output_shapes
:m*
dtype0?
5sequential_96/batch_normalization_865/batchnorm/mul_2MulHsequential_96/batch_normalization_865/batchnorm/ReadVariableOp_1:value:07sequential_96/batch_normalization_865/batchnorm/mul:z:0*
T0*
_output_shapes
:m?
@sequential_96/batch_normalization_865/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_96_batch_normalization_865_batchnorm_readvariableop_2_resource*
_output_shapes
:m*
dtype0?
3sequential_96/batch_normalization_865/batchnorm/subSubHsequential_96/batch_normalization_865/batchnorm/ReadVariableOp_2:value:09sequential_96/batch_normalization_865/batchnorm/mul_2:z:0*
T0*
_output_shapes
:m?
5sequential_96/batch_normalization_865/batchnorm/add_1AddV29sequential_96/batch_normalization_865/batchnorm/mul_1:z:07sequential_96/batch_normalization_865/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????m?
'sequential_96/leaky_re_lu_865/LeakyRelu	LeakyRelu9sequential_96/batch_normalization_865/batchnorm/add_1:z:0*'
_output_shapes
:?????????m*
alpha%???>?
-sequential_96/dense_962/MatMul/ReadVariableOpReadVariableOp6sequential_96_dense_962_matmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
sequential_96/dense_962/MatMulMatMul5sequential_96/leaky_re_lu_865/LeakyRelu:activations:05sequential_96/dense_962/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
.sequential_96/dense_962/BiasAdd/ReadVariableOpReadVariableOp7sequential_96_dense_962_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
sequential_96/dense_962/BiasAddBiasAdd(sequential_96/dense_962/MatMul:product:06sequential_96/dense_962/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
>sequential_96/batch_normalization_866/batchnorm/ReadVariableOpReadVariableOpGsequential_96_batch_normalization_866_batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0z
5sequential_96/batch_normalization_866/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
3sequential_96/batch_normalization_866/batchnorm/addAddV2Fsequential_96/batch_normalization_866/batchnorm/ReadVariableOp:value:0>sequential_96/batch_normalization_866/batchnorm/add/y:output:0*
T0*
_output_shapes
:m?
5sequential_96/batch_normalization_866/batchnorm/RsqrtRsqrt7sequential_96/batch_normalization_866/batchnorm/add:z:0*
T0*
_output_shapes
:m?
Bsequential_96/batch_normalization_866/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_96_batch_normalization_866_batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0?
3sequential_96/batch_normalization_866/batchnorm/mulMul9sequential_96/batch_normalization_866/batchnorm/Rsqrt:y:0Jsequential_96/batch_normalization_866/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:m?
5sequential_96/batch_normalization_866/batchnorm/mul_1Mul(sequential_96/dense_962/BiasAdd:output:07sequential_96/batch_normalization_866/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????m?
@sequential_96/batch_normalization_866/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_96_batch_normalization_866_batchnorm_readvariableop_1_resource*
_output_shapes
:m*
dtype0?
5sequential_96/batch_normalization_866/batchnorm/mul_2MulHsequential_96/batch_normalization_866/batchnorm/ReadVariableOp_1:value:07sequential_96/batch_normalization_866/batchnorm/mul:z:0*
T0*
_output_shapes
:m?
@sequential_96/batch_normalization_866/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_96_batch_normalization_866_batchnorm_readvariableop_2_resource*
_output_shapes
:m*
dtype0?
3sequential_96/batch_normalization_866/batchnorm/subSubHsequential_96/batch_normalization_866/batchnorm/ReadVariableOp_2:value:09sequential_96/batch_normalization_866/batchnorm/mul_2:z:0*
T0*
_output_shapes
:m?
5sequential_96/batch_normalization_866/batchnorm/add_1AddV29sequential_96/batch_normalization_866/batchnorm/mul_1:z:07sequential_96/batch_normalization_866/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????m?
'sequential_96/leaky_re_lu_866/LeakyRelu	LeakyRelu9sequential_96/batch_normalization_866/batchnorm/add_1:z:0*'
_output_shapes
:?????????m*
alpha%???>?
-sequential_96/dense_963/MatMul/ReadVariableOpReadVariableOp6sequential_96_dense_963_matmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
sequential_96/dense_963/MatMulMatMul5sequential_96/leaky_re_lu_866/LeakyRelu:activations:05sequential_96/dense_963/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
.sequential_96/dense_963/BiasAdd/ReadVariableOpReadVariableOp7sequential_96_dense_963_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
sequential_96/dense_963/BiasAddBiasAdd(sequential_96/dense_963/MatMul:product:06sequential_96/dense_963/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
>sequential_96/batch_normalization_867/batchnorm/ReadVariableOpReadVariableOpGsequential_96_batch_normalization_867_batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0z
5sequential_96/batch_normalization_867/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
3sequential_96/batch_normalization_867/batchnorm/addAddV2Fsequential_96/batch_normalization_867/batchnorm/ReadVariableOp:value:0>sequential_96/batch_normalization_867/batchnorm/add/y:output:0*
T0*
_output_shapes
:m?
5sequential_96/batch_normalization_867/batchnorm/RsqrtRsqrt7sequential_96/batch_normalization_867/batchnorm/add:z:0*
T0*
_output_shapes
:m?
Bsequential_96/batch_normalization_867/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_96_batch_normalization_867_batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0?
3sequential_96/batch_normalization_867/batchnorm/mulMul9sequential_96/batch_normalization_867/batchnorm/Rsqrt:y:0Jsequential_96/batch_normalization_867/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:m?
5sequential_96/batch_normalization_867/batchnorm/mul_1Mul(sequential_96/dense_963/BiasAdd:output:07sequential_96/batch_normalization_867/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????m?
@sequential_96/batch_normalization_867/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_96_batch_normalization_867_batchnorm_readvariableop_1_resource*
_output_shapes
:m*
dtype0?
5sequential_96/batch_normalization_867/batchnorm/mul_2MulHsequential_96/batch_normalization_867/batchnorm/ReadVariableOp_1:value:07sequential_96/batch_normalization_867/batchnorm/mul:z:0*
T0*
_output_shapes
:m?
@sequential_96/batch_normalization_867/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_96_batch_normalization_867_batchnorm_readvariableop_2_resource*
_output_shapes
:m*
dtype0?
3sequential_96/batch_normalization_867/batchnorm/subSubHsequential_96/batch_normalization_867/batchnorm/ReadVariableOp_2:value:09sequential_96/batch_normalization_867/batchnorm/mul_2:z:0*
T0*
_output_shapes
:m?
5sequential_96/batch_normalization_867/batchnorm/add_1AddV29sequential_96/batch_normalization_867/batchnorm/mul_1:z:07sequential_96/batch_normalization_867/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????m?
'sequential_96/leaky_re_lu_867/LeakyRelu	LeakyRelu9sequential_96/batch_normalization_867/batchnorm/add_1:z:0*'
_output_shapes
:?????????m*
alpha%???>?
-sequential_96/dense_964/MatMul/ReadVariableOpReadVariableOp6sequential_96_dense_964_matmul_readvariableop_resource*
_output_shapes

:m.*
dtype0?
sequential_96/dense_964/MatMulMatMul5sequential_96/leaky_re_lu_867/LeakyRelu:activations:05sequential_96/dense_964/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
.sequential_96/dense_964/BiasAdd/ReadVariableOpReadVariableOp7sequential_96_dense_964_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype0?
sequential_96/dense_964/BiasAddBiasAdd(sequential_96/dense_964/MatMul:product:06sequential_96/dense_964/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
>sequential_96/batch_normalization_868/batchnorm/ReadVariableOpReadVariableOpGsequential_96_batch_normalization_868_batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0z
5sequential_96/batch_normalization_868/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
3sequential_96/batch_normalization_868/batchnorm/addAddV2Fsequential_96/batch_normalization_868/batchnorm/ReadVariableOp:value:0>sequential_96/batch_normalization_868/batchnorm/add/y:output:0*
T0*
_output_shapes
:.?
5sequential_96/batch_normalization_868/batchnorm/RsqrtRsqrt7sequential_96/batch_normalization_868/batchnorm/add:z:0*
T0*
_output_shapes
:.?
Bsequential_96/batch_normalization_868/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_96_batch_normalization_868_batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0?
3sequential_96/batch_normalization_868/batchnorm/mulMul9sequential_96/batch_normalization_868/batchnorm/Rsqrt:y:0Jsequential_96/batch_normalization_868/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.?
5sequential_96/batch_normalization_868/batchnorm/mul_1Mul(sequential_96/dense_964/BiasAdd:output:07sequential_96/batch_normalization_868/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.?
@sequential_96/batch_normalization_868/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_96_batch_normalization_868_batchnorm_readvariableop_1_resource*
_output_shapes
:.*
dtype0?
5sequential_96/batch_normalization_868/batchnorm/mul_2MulHsequential_96/batch_normalization_868/batchnorm/ReadVariableOp_1:value:07sequential_96/batch_normalization_868/batchnorm/mul:z:0*
T0*
_output_shapes
:.?
@sequential_96/batch_normalization_868/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_96_batch_normalization_868_batchnorm_readvariableop_2_resource*
_output_shapes
:.*
dtype0?
3sequential_96/batch_normalization_868/batchnorm/subSubHsequential_96/batch_normalization_868/batchnorm/ReadVariableOp_2:value:09sequential_96/batch_normalization_868/batchnorm/mul_2:z:0*
T0*
_output_shapes
:.?
5sequential_96/batch_normalization_868/batchnorm/add_1AddV29sequential_96/batch_normalization_868/batchnorm/mul_1:z:07sequential_96/batch_normalization_868/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.?
'sequential_96/leaky_re_lu_868/LeakyRelu	LeakyRelu9sequential_96/batch_normalization_868/batchnorm/add_1:z:0*'
_output_shapes
:?????????.*
alpha%???>?
-sequential_96/dense_965/MatMul/ReadVariableOpReadVariableOp6sequential_96_dense_965_matmul_readvariableop_resource*
_output_shapes

:..*
dtype0?
sequential_96/dense_965/MatMulMatMul5sequential_96/leaky_re_lu_868/LeakyRelu:activations:05sequential_96/dense_965/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
.sequential_96/dense_965/BiasAdd/ReadVariableOpReadVariableOp7sequential_96_dense_965_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype0?
sequential_96/dense_965/BiasAddBiasAdd(sequential_96/dense_965/MatMul:product:06sequential_96/dense_965/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
>sequential_96/batch_normalization_869/batchnorm/ReadVariableOpReadVariableOpGsequential_96_batch_normalization_869_batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0z
5sequential_96/batch_normalization_869/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
3sequential_96/batch_normalization_869/batchnorm/addAddV2Fsequential_96/batch_normalization_869/batchnorm/ReadVariableOp:value:0>sequential_96/batch_normalization_869/batchnorm/add/y:output:0*
T0*
_output_shapes
:.?
5sequential_96/batch_normalization_869/batchnorm/RsqrtRsqrt7sequential_96/batch_normalization_869/batchnorm/add:z:0*
T0*
_output_shapes
:.?
Bsequential_96/batch_normalization_869/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_96_batch_normalization_869_batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0?
3sequential_96/batch_normalization_869/batchnorm/mulMul9sequential_96/batch_normalization_869/batchnorm/Rsqrt:y:0Jsequential_96/batch_normalization_869/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.?
5sequential_96/batch_normalization_869/batchnorm/mul_1Mul(sequential_96/dense_965/BiasAdd:output:07sequential_96/batch_normalization_869/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.?
@sequential_96/batch_normalization_869/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_96_batch_normalization_869_batchnorm_readvariableop_1_resource*
_output_shapes
:.*
dtype0?
5sequential_96/batch_normalization_869/batchnorm/mul_2MulHsequential_96/batch_normalization_869/batchnorm/ReadVariableOp_1:value:07sequential_96/batch_normalization_869/batchnorm/mul:z:0*
T0*
_output_shapes
:.?
@sequential_96/batch_normalization_869/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_96_batch_normalization_869_batchnorm_readvariableop_2_resource*
_output_shapes
:.*
dtype0?
3sequential_96/batch_normalization_869/batchnorm/subSubHsequential_96/batch_normalization_869/batchnorm/ReadVariableOp_2:value:09sequential_96/batch_normalization_869/batchnorm/mul_2:z:0*
T0*
_output_shapes
:.?
5sequential_96/batch_normalization_869/batchnorm/add_1AddV29sequential_96/batch_normalization_869/batchnorm/mul_1:z:07sequential_96/batch_normalization_869/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.?
'sequential_96/leaky_re_lu_869/LeakyRelu	LeakyRelu9sequential_96/batch_normalization_869/batchnorm/add_1:z:0*'
_output_shapes
:?????????.*
alpha%???>?
-sequential_96/dense_966/MatMul/ReadVariableOpReadVariableOp6sequential_96_dense_966_matmul_readvariableop_resource*
_output_shapes

:.]*
dtype0?
sequential_96/dense_966/MatMulMatMul5sequential_96/leaky_re_lu_869/LeakyRelu:activations:05sequential_96/dense_966/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]?
.sequential_96/dense_966/BiasAdd/ReadVariableOpReadVariableOp7sequential_96_dense_966_biasadd_readvariableop_resource*
_output_shapes
:]*
dtype0?
sequential_96/dense_966/BiasAddBiasAdd(sequential_96/dense_966/MatMul:product:06sequential_96/dense_966/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]?
>sequential_96/batch_normalization_870/batchnorm/ReadVariableOpReadVariableOpGsequential_96_batch_normalization_870_batchnorm_readvariableop_resource*
_output_shapes
:]*
dtype0z
5sequential_96/batch_normalization_870/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
3sequential_96/batch_normalization_870/batchnorm/addAddV2Fsequential_96/batch_normalization_870/batchnorm/ReadVariableOp:value:0>sequential_96/batch_normalization_870/batchnorm/add/y:output:0*
T0*
_output_shapes
:]?
5sequential_96/batch_normalization_870/batchnorm/RsqrtRsqrt7sequential_96/batch_normalization_870/batchnorm/add:z:0*
T0*
_output_shapes
:]?
Bsequential_96/batch_normalization_870/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_96_batch_normalization_870_batchnorm_mul_readvariableop_resource*
_output_shapes
:]*
dtype0?
3sequential_96/batch_normalization_870/batchnorm/mulMul9sequential_96/batch_normalization_870/batchnorm/Rsqrt:y:0Jsequential_96/batch_normalization_870/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:]?
5sequential_96/batch_normalization_870/batchnorm/mul_1Mul(sequential_96/dense_966/BiasAdd:output:07sequential_96/batch_normalization_870/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????]?
@sequential_96/batch_normalization_870/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_96_batch_normalization_870_batchnorm_readvariableop_1_resource*
_output_shapes
:]*
dtype0?
5sequential_96/batch_normalization_870/batchnorm/mul_2MulHsequential_96/batch_normalization_870/batchnorm/ReadVariableOp_1:value:07sequential_96/batch_normalization_870/batchnorm/mul:z:0*
T0*
_output_shapes
:]?
@sequential_96/batch_normalization_870/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_96_batch_normalization_870_batchnorm_readvariableop_2_resource*
_output_shapes
:]*
dtype0?
3sequential_96/batch_normalization_870/batchnorm/subSubHsequential_96/batch_normalization_870/batchnorm/ReadVariableOp_2:value:09sequential_96/batch_normalization_870/batchnorm/mul_2:z:0*
T0*
_output_shapes
:]?
5sequential_96/batch_normalization_870/batchnorm/add_1AddV29sequential_96/batch_normalization_870/batchnorm/mul_1:z:07sequential_96/batch_normalization_870/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????]?
'sequential_96/leaky_re_lu_870/LeakyRelu	LeakyRelu9sequential_96/batch_normalization_870/batchnorm/add_1:z:0*'
_output_shapes
:?????????]*
alpha%???>?
-sequential_96/dense_967/MatMul/ReadVariableOpReadVariableOp6sequential_96_dense_967_matmul_readvariableop_resource*
_output_shapes

:]*
dtype0?
sequential_96/dense_967/MatMulMatMul5sequential_96/leaky_re_lu_870/LeakyRelu:activations:05sequential_96/dense_967/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.sequential_96/dense_967/BiasAdd/ReadVariableOpReadVariableOp7sequential_96_dense_967_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_96/dense_967/BiasAddBiasAdd(sequential_96/dense_967/MatMul:product:06sequential_96/dense_967/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
IdentityIdentity(sequential_96/dense_967/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp?^sequential_96/batch_normalization_865/batchnorm/ReadVariableOpA^sequential_96/batch_normalization_865/batchnorm/ReadVariableOp_1A^sequential_96/batch_normalization_865/batchnorm/ReadVariableOp_2C^sequential_96/batch_normalization_865/batchnorm/mul/ReadVariableOp?^sequential_96/batch_normalization_866/batchnorm/ReadVariableOpA^sequential_96/batch_normalization_866/batchnorm/ReadVariableOp_1A^sequential_96/batch_normalization_866/batchnorm/ReadVariableOp_2C^sequential_96/batch_normalization_866/batchnorm/mul/ReadVariableOp?^sequential_96/batch_normalization_867/batchnorm/ReadVariableOpA^sequential_96/batch_normalization_867/batchnorm/ReadVariableOp_1A^sequential_96/batch_normalization_867/batchnorm/ReadVariableOp_2C^sequential_96/batch_normalization_867/batchnorm/mul/ReadVariableOp?^sequential_96/batch_normalization_868/batchnorm/ReadVariableOpA^sequential_96/batch_normalization_868/batchnorm/ReadVariableOp_1A^sequential_96/batch_normalization_868/batchnorm/ReadVariableOp_2C^sequential_96/batch_normalization_868/batchnorm/mul/ReadVariableOp?^sequential_96/batch_normalization_869/batchnorm/ReadVariableOpA^sequential_96/batch_normalization_869/batchnorm/ReadVariableOp_1A^sequential_96/batch_normalization_869/batchnorm/ReadVariableOp_2C^sequential_96/batch_normalization_869/batchnorm/mul/ReadVariableOp?^sequential_96/batch_normalization_870/batchnorm/ReadVariableOpA^sequential_96/batch_normalization_870/batchnorm/ReadVariableOp_1A^sequential_96/batch_normalization_870/batchnorm/ReadVariableOp_2C^sequential_96/batch_normalization_870/batchnorm/mul/ReadVariableOp/^sequential_96/dense_961/BiasAdd/ReadVariableOp.^sequential_96/dense_961/MatMul/ReadVariableOp/^sequential_96/dense_962/BiasAdd/ReadVariableOp.^sequential_96/dense_962/MatMul/ReadVariableOp/^sequential_96/dense_963/BiasAdd/ReadVariableOp.^sequential_96/dense_963/MatMul/ReadVariableOp/^sequential_96/dense_964/BiasAdd/ReadVariableOp.^sequential_96/dense_964/MatMul/ReadVariableOp/^sequential_96/dense_965/BiasAdd/ReadVariableOp.^sequential_96/dense_965/MatMul/ReadVariableOp/^sequential_96/dense_966/BiasAdd/ReadVariableOp.^sequential_96/dense_966/MatMul/ReadVariableOp/^sequential_96/dense_967/BiasAdd/ReadVariableOp.^sequential_96/dense_967/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
>sequential_96/batch_normalization_865/batchnorm/ReadVariableOp>sequential_96/batch_normalization_865/batchnorm/ReadVariableOp2?
@sequential_96/batch_normalization_865/batchnorm/ReadVariableOp_1@sequential_96/batch_normalization_865/batchnorm/ReadVariableOp_12?
@sequential_96/batch_normalization_865/batchnorm/ReadVariableOp_2@sequential_96/batch_normalization_865/batchnorm/ReadVariableOp_22?
Bsequential_96/batch_normalization_865/batchnorm/mul/ReadVariableOpBsequential_96/batch_normalization_865/batchnorm/mul/ReadVariableOp2?
>sequential_96/batch_normalization_866/batchnorm/ReadVariableOp>sequential_96/batch_normalization_866/batchnorm/ReadVariableOp2?
@sequential_96/batch_normalization_866/batchnorm/ReadVariableOp_1@sequential_96/batch_normalization_866/batchnorm/ReadVariableOp_12?
@sequential_96/batch_normalization_866/batchnorm/ReadVariableOp_2@sequential_96/batch_normalization_866/batchnorm/ReadVariableOp_22?
Bsequential_96/batch_normalization_866/batchnorm/mul/ReadVariableOpBsequential_96/batch_normalization_866/batchnorm/mul/ReadVariableOp2?
>sequential_96/batch_normalization_867/batchnorm/ReadVariableOp>sequential_96/batch_normalization_867/batchnorm/ReadVariableOp2?
@sequential_96/batch_normalization_867/batchnorm/ReadVariableOp_1@sequential_96/batch_normalization_867/batchnorm/ReadVariableOp_12?
@sequential_96/batch_normalization_867/batchnorm/ReadVariableOp_2@sequential_96/batch_normalization_867/batchnorm/ReadVariableOp_22?
Bsequential_96/batch_normalization_867/batchnorm/mul/ReadVariableOpBsequential_96/batch_normalization_867/batchnorm/mul/ReadVariableOp2?
>sequential_96/batch_normalization_868/batchnorm/ReadVariableOp>sequential_96/batch_normalization_868/batchnorm/ReadVariableOp2?
@sequential_96/batch_normalization_868/batchnorm/ReadVariableOp_1@sequential_96/batch_normalization_868/batchnorm/ReadVariableOp_12?
@sequential_96/batch_normalization_868/batchnorm/ReadVariableOp_2@sequential_96/batch_normalization_868/batchnorm/ReadVariableOp_22?
Bsequential_96/batch_normalization_868/batchnorm/mul/ReadVariableOpBsequential_96/batch_normalization_868/batchnorm/mul/ReadVariableOp2?
>sequential_96/batch_normalization_869/batchnorm/ReadVariableOp>sequential_96/batch_normalization_869/batchnorm/ReadVariableOp2?
@sequential_96/batch_normalization_869/batchnorm/ReadVariableOp_1@sequential_96/batch_normalization_869/batchnorm/ReadVariableOp_12?
@sequential_96/batch_normalization_869/batchnorm/ReadVariableOp_2@sequential_96/batch_normalization_869/batchnorm/ReadVariableOp_22?
Bsequential_96/batch_normalization_869/batchnorm/mul/ReadVariableOpBsequential_96/batch_normalization_869/batchnorm/mul/ReadVariableOp2?
>sequential_96/batch_normalization_870/batchnorm/ReadVariableOp>sequential_96/batch_normalization_870/batchnorm/ReadVariableOp2?
@sequential_96/batch_normalization_870/batchnorm/ReadVariableOp_1@sequential_96/batch_normalization_870/batchnorm/ReadVariableOp_12?
@sequential_96/batch_normalization_870/batchnorm/ReadVariableOp_2@sequential_96/batch_normalization_870/batchnorm/ReadVariableOp_22?
Bsequential_96/batch_normalization_870/batchnorm/mul/ReadVariableOpBsequential_96/batch_normalization_870/batchnorm/mul/ReadVariableOp2`
.sequential_96/dense_961/BiasAdd/ReadVariableOp.sequential_96/dense_961/BiasAdd/ReadVariableOp2^
-sequential_96/dense_961/MatMul/ReadVariableOp-sequential_96/dense_961/MatMul/ReadVariableOp2`
.sequential_96/dense_962/BiasAdd/ReadVariableOp.sequential_96/dense_962/BiasAdd/ReadVariableOp2^
-sequential_96/dense_962/MatMul/ReadVariableOp-sequential_96/dense_962/MatMul/ReadVariableOp2`
.sequential_96/dense_963/BiasAdd/ReadVariableOp.sequential_96/dense_963/BiasAdd/ReadVariableOp2^
-sequential_96/dense_963/MatMul/ReadVariableOp-sequential_96/dense_963/MatMul/ReadVariableOp2`
.sequential_96/dense_964/BiasAdd/ReadVariableOp.sequential_96/dense_964/BiasAdd/ReadVariableOp2^
-sequential_96/dense_964/MatMul/ReadVariableOp-sequential_96/dense_964/MatMul/ReadVariableOp2`
.sequential_96/dense_965/BiasAdd/ReadVariableOp.sequential_96/dense_965/BiasAdd/ReadVariableOp2^
-sequential_96/dense_965/MatMul/ReadVariableOp-sequential_96/dense_965/MatMul/ReadVariableOp2`
.sequential_96/dense_966/BiasAdd/ReadVariableOp.sequential_96/dense_966/BiasAdd/ReadVariableOp2^
-sequential_96/dense_966/MatMul/ReadVariableOp-sequential_96/dense_966/MatMul/ReadVariableOp2`
.sequential_96/dense_967/BiasAdd/ReadVariableOp.sequential_96/dense_967/BiasAdd/ReadVariableOp2^
-sequential_96/dense_967/MatMul/ReadVariableOp-sequential_96/dense_967/MatMul/ReadVariableOp:_ [
'
_output_shapes
:?????????
0
_user_specified_namenormalization_96_input:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_864726

inputs/
!batchnorm_readvariableop_resource:m3
%batchnorm_mul_readvariableop_resource:m1
#batchnorm_readvariableop_1_resource:m1
#batchnorm_readvariableop_2_resource:m
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:mP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:mc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????mz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:m*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:mz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:m*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:mr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????mb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_862132

inputs5
'assignmovingavg_readvariableop_resource:m7
)assignmovingavg_1_readvariableop_resource:m3
%batchnorm_mul_readvariableop_resource:m/
!batchnorm_readvariableop_resource:m
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:m?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????ml
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:m*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:mx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:m?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:m*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:m~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:m?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:mP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:mc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????mh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:mv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:mr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????mb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_862460

inputs5
'assignmovingavg_readvariableop_resource:.7
)assignmovingavg_1_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:./
!batchnorm_readvariableop_resource:.
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:.?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????.l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:.*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:.x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:.?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:.*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:.~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:.?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:.P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:.~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:.v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:.r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????.?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_862542

inputs5
'assignmovingavg_readvariableop_resource:]7
)assignmovingavg_1_readvariableop_resource:]3
%batchnorm_mul_readvariableop_resource:]/
!batchnorm_readvariableop_resource:]
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:]*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:]?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????]l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:]*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:]*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:]*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:]*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:]x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:]?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:]*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:]~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:]?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:]P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:]~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:]*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:]c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????]h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:]v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:]*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:]r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????]b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????]?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????]: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
__inference_loss_fn_5_865339M
;dense_966_kernel_regularizer_square_readvariableop_resource:.]
identity??2dense_966/kernel/Regularizer/Square/ReadVariableOp?
2dense_966/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_966_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:.]*
dtype0?
#dense_966/kernel/Regularizer/SquareSquare:dense_966/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:.]s
"dense_966/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_966/kernel/Regularizer/SumSum'dense_966/kernel/Regularizer/Square:y:0+dense_966/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_966/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?<?
 dense_966/kernel/Regularizer/mulMul+dense_966/kernel/Regularizer/mul/x:output:0)dense_966/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_966/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_966/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_966/kernel/Regularizer/Square/ReadVariableOp2dense_966/kernel/Regularizer/Square/ReadVariableOp
?
g
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_862603

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????m*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????m"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????m:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_867_layer_call_fn_864814

inputs
unknown:m
	unknown_0:m
	unknown_1:m
	unknown_2:m
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_862249o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????m`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_864649

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????m*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????m"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????m:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_862378

inputs5
'assignmovingavg_readvariableop_resource:.7
)assignmovingavg_1_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:./
!batchnorm_readvariableop_resource:.
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:.?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????.l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:.*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:.x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:.?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:.*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:.~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:.?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:.P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:.~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:.v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:.r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????.?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
E__inference_dense_963_layer_call_and_return_conditional_losses_862659

inputs0
matmul_readvariableop_resource:mm-
biasadd_readvariableop_resource:m
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_963/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:mm*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
2dense_963/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
#dense_963/kernel/Regularizer/SquareSquare:dense_963/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_963/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_963/kernel/Regularizer/SumSum'dense_963/kernel/Regularizer/Square:y:0+dense_963/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_963/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_963/kernel/Regularizer/mulMul+dense_963/kernel/Regularizer/mul/x:output:0)dense_963/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_963/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????m: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_963/kernel/Regularizer/Square/ReadVariableOp2dense_963/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
??
?A
"__inference__traced_restore_865968
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_961_kernel:m/
!assignvariableop_4_dense_961_bias:m>
0assignvariableop_5_batch_normalization_865_gamma:m=
/assignvariableop_6_batch_normalization_865_beta:mD
6assignvariableop_7_batch_normalization_865_moving_mean:mH
:assignvariableop_8_batch_normalization_865_moving_variance:m5
#assignvariableop_9_dense_962_kernel:mm0
"assignvariableop_10_dense_962_bias:m?
1assignvariableop_11_batch_normalization_866_gamma:m>
0assignvariableop_12_batch_normalization_866_beta:mE
7assignvariableop_13_batch_normalization_866_moving_mean:mI
;assignvariableop_14_batch_normalization_866_moving_variance:m6
$assignvariableop_15_dense_963_kernel:mm0
"assignvariableop_16_dense_963_bias:m?
1assignvariableop_17_batch_normalization_867_gamma:m>
0assignvariableop_18_batch_normalization_867_beta:mE
7assignvariableop_19_batch_normalization_867_moving_mean:mI
;assignvariableop_20_batch_normalization_867_moving_variance:m6
$assignvariableop_21_dense_964_kernel:m.0
"assignvariableop_22_dense_964_bias:.?
1assignvariableop_23_batch_normalization_868_gamma:.>
0assignvariableop_24_batch_normalization_868_beta:.E
7assignvariableop_25_batch_normalization_868_moving_mean:.I
;assignvariableop_26_batch_normalization_868_moving_variance:.6
$assignvariableop_27_dense_965_kernel:..0
"assignvariableop_28_dense_965_bias:.?
1assignvariableop_29_batch_normalization_869_gamma:.>
0assignvariableop_30_batch_normalization_869_beta:.E
7assignvariableop_31_batch_normalization_869_moving_mean:.I
;assignvariableop_32_batch_normalization_869_moving_variance:.6
$assignvariableop_33_dense_966_kernel:.]0
"assignvariableop_34_dense_966_bias:]?
1assignvariableop_35_batch_normalization_870_gamma:]>
0assignvariableop_36_batch_normalization_870_beta:]E
7assignvariableop_37_batch_normalization_870_moving_mean:]I
;assignvariableop_38_batch_normalization_870_moving_variance:]6
$assignvariableop_39_dense_967_kernel:]0
"assignvariableop_40_dense_967_bias:'
assignvariableop_41_adam_iter:	 )
assignvariableop_42_adam_beta_1: )
assignvariableop_43_adam_beta_2: (
assignvariableop_44_adam_decay: #
assignvariableop_45_total: %
assignvariableop_46_count_1: =
+assignvariableop_47_adam_dense_961_kernel_m:m7
)assignvariableop_48_adam_dense_961_bias_m:mF
8assignvariableop_49_adam_batch_normalization_865_gamma_m:mE
7assignvariableop_50_adam_batch_normalization_865_beta_m:m=
+assignvariableop_51_adam_dense_962_kernel_m:mm7
)assignvariableop_52_adam_dense_962_bias_m:mF
8assignvariableop_53_adam_batch_normalization_866_gamma_m:mE
7assignvariableop_54_adam_batch_normalization_866_beta_m:m=
+assignvariableop_55_adam_dense_963_kernel_m:mm7
)assignvariableop_56_adam_dense_963_bias_m:mF
8assignvariableop_57_adam_batch_normalization_867_gamma_m:mE
7assignvariableop_58_adam_batch_normalization_867_beta_m:m=
+assignvariableop_59_adam_dense_964_kernel_m:m.7
)assignvariableop_60_adam_dense_964_bias_m:.F
8assignvariableop_61_adam_batch_normalization_868_gamma_m:.E
7assignvariableop_62_adam_batch_normalization_868_beta_m:.=
+assignvariableop_63_adam_dense_965_kernel_m:..7
)assignvariableop_64_adam_dense_965_bias_m:.F
8assignvariableop_65_adam_batch_normalization_869_gamma_m:.E
7assignvariableop_66_adam_batch_normalization_869_beta_m:.=
+assignvariableop_67_adam_dense_966_kernel_m:.]7
)assignvariableop_68_adam_dense_966_bias_m:]F
8assignvariableop_69_adam_batch_normalization_870_gamma_m:]E
7assignvariableop_70_adam_batch_normalization_870_beta_m:]=
+assignvariableop_71_adam_dense_967_kernel_m:]7
)assignvariableop_72_adam_dense_967_bias_m:=
+assignvariableop_73_adam_dense_961_kernel_v:m7
)assignvariableop_74_adam_dense_961_bias_v:mF
8assignvariableop_75_adam_batch_normalization_865_gamma_v:mE
7assignvariableop_76_adam_batch_normalization_865_beta_v:m=
+assignvariableop_77_adam_dense_962_kernel_v:mm7
)assignvariableop_78_adam_dense_962_bias_v:mF
8assignvariableop_79_adam_batch_normalization_866_gamma_v:mE
7assignvariableop_80_adam_batch_normalization_866_beta_v:m=
+assignvariableop_81_adam_dense_963_kernel_v:mm7
)assignvariableop_82_adam_dense_963_bias_v:mF
8assignvariableop_83_adam_batch_normalization_867_gamma_v:mE
7assignvariableop_84_adam_batch_normalization_867_beta_v:m=
+assignvariableop_85_adam_dense_964_kernel_v:m.7
)assignvariableop_86_adam_dense_964_bias_v:.F
8assignvariableop_87_adam_batch_normalization_868_gamma_v:.E
7assignvariableop_88_adam_batch_normalization_868_beta_v:.=
+assignvariableop_89_adam_dense_965_kernel_v:..7
)assignvariableop_90_adam_dense_965_bias_v:.F
8assignvariableop_91_adam_batch_normalization_869_gamma_v:.E
7assignvariableop_92_adam_batch_normalization_869_beta_v:.=
+assignvariableop_93_adam_dense_966_kernel_v:.]7
)assignvariableop_94_adam_dense_966_bias_v:]F
8assignvariableop_95_adam_batch_normalization_870_gamma_v:]E
7assignvariableop_96_adam_batch_normalization_870_beta_v:]=
+assignvariableop_97_adam_dense_967_kernel_v:]7
)assignvariableop_98_adam_dense_967_bias_v:
identity_100??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?7
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*?6
value?6B?6dB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*?
value?B?dB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*r
dtypesh
f2d		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_961_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_961_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batch_normalization_865_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_865_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_865_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp:assignvariableop_8_batch_normalization_865_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_962_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_962_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batch_normalization_866_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_866_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_866_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp;assignvariableop_14_batch_normalization_866_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_963_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_963_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batch_normalization_867_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_867_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_867_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp;assignvariableop_20_batch_normalization_867_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_964_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_964_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp1assignvariableop_23_batch_normalization_868_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_868_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp7assignvariableop_25_batch_normalization_868_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp;assignvariableop_26_batch_normalization_868_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_965_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_965_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_batch_normalization_869_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_869_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp7assignvariableop_31_batch_normalization_869_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp;assignvariableop_32_batch_normalization_869_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_966_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_966_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp1assignvariableop_35_batch_normalization_870_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_870_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp7assignvariableop_37_batch_normalization_870_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp;assignvariableop_38_batch_normalization_870_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp$assignvariableop_39_dense_967_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_967_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_iterIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_beta_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOpassignvariableop_43_adam_beta_2Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOpassignvariableop_44_adam_decayIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpassignvariableop_45_totalIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_dense_961_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_961_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp8assignvariableop_49_adam_batch_normalization_865_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp7assignvariableop_50_adam_batch_normalization_865_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_dense_962_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_dense_962_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp8assignvariableop_53_adam_batch_normalization_866_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp7assignvariableop_54_adam_batch_normalization_866_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_dense_963_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_dense_963_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp8assignvariableop_57_adam_batch_normalization_867_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp7assignvariableop_58_adam_batch_normalization_867_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_dense_964_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_dense_964_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp8assignvariableop_61_adam_batch_normalization_868_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp7assignvariableop_62_adam_batch_normalization_868_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_dense_965_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_dense_965_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp8assignvariableop_65_adam_batch_normalization_869_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp7assignvariableop_66_adam_batch_normalization_869_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_dense_966_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_dense_966_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp8assignvariableop_69_adam_batch_normalization_870_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp7assignvariableop_70_adam_batch_normalization_870_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_dense_967_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_dense_967_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_dense_961_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_dense_961_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp8assignvariableop_75_adam_batch_normalization_865_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp7assignvariableop_76_adam_batch_normalization_865_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_dense_962_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_dense_962_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp8assignvariableop_79_adam_batch_normalization_866_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp7assignvariableop_80_adam_batch_normalization_866_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_dense_963_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_dense_963_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp8assignvariableop_83_adam_batch_normalization_867_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp7assignvariableop_84_adam_batch_normalization_867_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_dense_964_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_dense_964_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp8assignvariableop_87_adam_batch_normalization_868_gamma_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp7assignvariableop_88_adam_batch_normalization_868_beta_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_dense_965_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_dense_965_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp8assignvariableop_91_adam_batch_normalization_869_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp7assignvariableop_92_adam_batch_normalization_869_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_dense_966_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_dense_966_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOp8assignvariableop_95_adam_batch_normalization_870_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOp7assignvariableop_96_adam_batch_normalization_870_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_dense_967_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_dense_967_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_99Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^NoOp"/device:CPU:0*
T0*
_output_shapes
: X
Identity_100IdentityIdentity_99:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98*"
_acd_function_control_output(*
_output_shapes
 "%
identity_100Identity_100:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_98:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?*
I__inference_sequential_96_layer_call_and_return_conditional_losses_864394

inputs
normalization_96_sub_y
normalization_96_sqrt_x:
(dense_961_matmul_readvariableop_resource:m7
)dense_961_biasadd_readvariableop_resource:mM
?batch_normalization_865_assignmovingavg_readvariableop_resource:mO
Abatch_normalization_865_assignmovingavg_1_readvariableop_resource:mK
=batch_normalization_865_batchnorm_mul_readvariableop_resource:mG
9batch_normalization_865_batchnorm_readvariableop_resource:m:
(dense_962_matmul_readvariableop_resource:mm7
)dense_962_biasadd_readvariableop_resource:mM
?batch_normalization_866_assignmovingavg_readvariableop_resource:mO
Abatch_normalization_866_assignmovingavg_1_readvariableop_resource:mK
=batch_normalization_866_batchnorm_mul_readvariableop_resource:mG
9batch_normalization_866_batchnorm_readvariableop_resource:m:
(dense_963_matmul_readvariableop_resource:mm7
)dense_963_biasadd_readvariableop_resource:mM
?batch_normalization_867_assignmovingavg_readvariableop_resource:mO
Abatch_normalization_867_assignmovingavg_1_readvariableop_resource:mK
=batch_normalization_867_batchnorm_mul_readvariableop_resource:mG
9batch_normalization_867_batchnorm_readvariableop_resource:m:
(dense_964_matmul_readvariableop_resource:m.7
)dense_964_biasadd_readvariableop_resource:.M
?batch_normalization_868_assignmovingavg_readvariableop_resource:.O
Abatch_normalization_868_assignmovingavg_1_readvariableop_resource:.K
=batch_normalization_868_batchnorm_mul_readvariableop_resource:.G
9batch_normalization_868_batchnorm_readvariableop_resource:.:
(dense_965_matmul_readvariableop_resource:..7
)dense_965_biasadd_readvariableop_resource:.M
?batch_normalization_869_assignmovingavg_readvariableop_resource:.O
Abatch_normalization_869_assignmovingavg_1_readvariableop_resource:.K
=batch_normalization_869_batchnorm_mul_readvariableop_resource:.G
9batch_normalization_869_batchnorm_readvariableop_resource:.:
(dense_966_matmul_readvariableop_resource:.]7
)dense_966_biasadd_readvariableop_resource:]M
?batch_normalization_870_assignmovingavg_readvariableop_resource:]O
Abatch_normalization_870_assignmovingavg_1_readvariableop_resource:]K
=batch_normalization_870_batchnorm_mul_readvariableop_resource:]G
9batch_normalization_870_batchnorm_readvariableop_resource:]:
(dense_967_matmul_readvariableop_resource:]7
)dense_967_biasadd_readvariableop_resource:
identity??'batch_normalization_865/AssignMovingAvg?6batch_normalization_865/AssignMovingAvg/ReadVariableOp?)batch_normalization_865/AssignMovingAvg_1?8batch_normalization_865/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_865/batchnorm/ReadVariableOp?4batch_normalization_865/batchnorm/mul/ReadVariableOp?'batch_normalization_866/AssignMovingAvg?6batch_normalization_866/AssignMovingAvg/ReadVariableOp?)batch_normalization_866/AssignMovingAvg_1?8batch_normalization_866/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_866/batchnorm/ReadVariableOp?4batch_normalization_866/batchnorm/mul/ReadVariableOp?'batch_normalization_867/AssignMovingAvg?6batch_normalization_867/AssignMovingAvg/ReadVariableOp?)batch_normalization_867/AssignMovingAvg_1?8batch_normalization_867/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_867/batchnorm/ReadVariableOp?4batch_normalization_867/batchnorm/mul/ReadVariableOp?'batch_normalization_868/AssignMovingAvg?6batch_normalization_868/AssignMovingAvg/ReadVariableOp?)batch_normalization_868/AssignMovingAvg_1?8batch_normalization_868/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_868/batchnorm/ReadVariableOp?4batch_normalization_868/batchnorm/mul/ReadVariableOp?'batch_normalization_869/AssignMovingAvg?6batch_normalization_869/AssignMovingAvg/ReadVariableOp?)batch_normalization_869/AssignMovingAvg_1?8batch_normalization_869/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_869/batchnorm/ReadVariableOp?4batch_normalization_869/batchnorm/mul/ReadVariableOp?'batch_normalization_870/AssignMovingAvg?6batch_normalization_870/AssignMovingAvg/ReadVariableOp?)batch_normalization_870/AssignMovingAvg_1?8batch_normalization_870/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_870/batchnorm/ReadVariableOp?4batch_normalization_870/batchnorm/mul/ReadVariableOp? dense_961/BiasAdd/ReadVariableOp?dense_961/MatMul/ReadVariableOp?2dense_961/kernel/Regularizer/Square/ReadVariableOp? dense_962/BiasAdd/ReadVariableOp?dense_962/MatMul/ReadVariableOp?2dense_962/kernel/Regularizer/Square/ReadVariableOp? dense_963/BiasAdd/ReadVariableOp?dense_963/MatMul/ReadVariableOp?2dense_963/kernel/Regularizer/Square/ReadVariableOp? dense_964/BiasAdd/ReadVariableOp?dense_964/MatMul/ReadVariableOp?2dense_964/kernel/Regularizer/Square/ReadVariableOp? dense_965/BiasAdd/ReadVariableOp?dense_965/MatMul/ReadVariableOp?2dense_965/kernel/Regularizer/Square/ReadVariableOp? dense_966/BiasAdd/ReadVariableOp?dense_966/MatMul/ReadVariableOp?2dense_966/kernel/Regularizer/Square/ReadVariableOp? dense_967/BiasAdd/ReadVariableOp?dense_967/MatMul/ReadVariableOpm
normalization_96/subSubinputsnormalization_96_sub_y*
T0*'
_output_shapes
:?????????_
normalization_96/SqrtSqrtnormalization_96_sqrt_x*
T0*
_output_shapes

:_
normalization_96/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_96/MaximumMaximumnormalization_96/Sqrt:y:0#normalization_96/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_96/truedivRealDivnormalization_96/sub:z:0normalization_96/Maximum:z:0*
T0*'
_output_shapes
:??????????
dense_961/MatMul/ReadVariableOpReadVariableOp(dense_961_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
dense_961/MatMulMatMulnormalization_96/truediv:z:0'dense_961/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
 dense_961/BiasAdd/ReadVariableOpReadVariableOp)dense_961_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
dense_961/BiasAddBiasAdddense_961/MatMul:product:0(dense_961/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
6batch_normalization_865/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization_865/moments/meanMeandense_961/BiasAdd:output:0?batch_normalization_865/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(?
,batch_normalization_865/moments/StopGradientStopGradient-batch_normalization_865/moments/mean:output:0*
T0*
_output_shapes

:m?
1batch_normalization_865/moments/SquaredDifferenceSquaredDifferencedense_961/BiasAdd:output:05batch_normalization_865/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????m?
:batch_normalization_865/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
(batch_normalization_865/moments/varianceMean5batch_normalization_865/moments/SquaredDifference:z:0Cbatch_normalization_865/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(?
'batch_normalization_865/moments/SqueezeSqueeze-batch_normalization_865/moments/mean:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 ?
)batch_normalization_865/moments/Squeeze_1Squeeze1batch_normalization_865/moments/variance:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 r
-batch_normalization_865/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_865/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_865_assignmovingavg_readvariableop_resource*
_output_shapes
:m*
dtype0?
+batch_normalization_865/AssignMovingAvg/subSub>batch_normalization_865/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_865/moments/Squeeze:output:0*
T0*
_output_shapes
:m?
+batch_normalization_865/AssignMovingAvg/mulMul/batch_normalization_865/AssignMovingAvg/sub:z:06batch_normalization_865/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:m?
'batch_normalization_865/AssignMovingAvgAssignSubVariableOp?batch_normalization_865_assignmovingavg_readvariableop_resource/batch_normalization_865/AssignMovingAvg/mul:z:07^batch_normalization_865/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_865/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8batch_normalization_865/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_865_assignmovingavg_1_readvariableop_resource*
_output_shapes
:m*
dtype0?
-batch_normalization_865/AssignMovingAvg_1/subSub@batch_normalization_865/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_865/moments/Squeeze_1:output:0*
T0*
_output_shapes
:m?
-batch_normalization_865/AssignMovingAvg_1/mulMul1batch_normalization_865/AssignMovingAvg_1/sub:z:08batch_normalization_865/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:m?
)batch_normalization_865/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_865_assignmovingavg_1_readvariableop_resource1batch_normalization_865/AssignMovingAvg_1/mul:z:09^batch_normalization_865/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_865/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_865/batchnorm/addAddV22batch_normalization_865/moments/Squeeze_1:output:00batch_normalization_865/batchnorm/add/y:output:0*
T0*
_output_shapes
:m?
'batch_normalization_865/batchnorm/RsqrtRsqrt)batch_normalization_865/batchnorm/add:z:0*
T0*
_output_shapes
:m?
4batch_normalization_865/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_865_batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0?
%batch_normalization_865/batchnorm/mulMul+batch_normalization_865/batchnorm/Rsqrt:y:0<batch_normalization_865/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:m?
'batch_normalization_865/batchnorm/mul_1Muldense_961/BiasAdd:output:0)batch_normalization_865/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????m?
'batch_normalization_865/batchnorm/mul_2Mul0batch_normalization_865/moments/Squeeze:output:0)batch_normalization_865/batchnorm/mul:z:0*
T0*
_output_shapes
:m?
0batch_normalization_865/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_865_batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0?
%batch_normalization_865/batchnorm/subSub8batch_normalization_865/batchnorm/ReadVariableOp:value:0+batch_normalization_865/batchnorm/mul_2:z:0*
T0*
_output_shapes
:m?
'batch_normalization_865/batchnorm/add_1AddV2+batch_normalization_865/batchnorm/mul_1:z:0)batch_normalization_865/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????m?
leaky_re_lu_865/LeakyRelu	LeakyRelu+batch_normalization_865/batchnorm/add_1:z:0*'
_output_shapes
:?????????m*
alpha%???>?
dense_962/MatMul/ReadVariableOpReadVariableOp(dense_962_matmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
dense_962/MatMulMatMul'leaky_re_lu_865/LeakyRelu:activations:0'dense_962/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
 dense_962/BiasAdd/ReadVariableOpReadVariableOp)dense_962_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
dense_962/BiasAddBiasAdddense_962/MatMul:product:0(dense_962/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
6batch_normalization_866/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization_866/moments/meanMeandense_962/BiasAdd:output:0?batch_normalization_866/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(?
,batch_normalization_866/moments/StopGradientStopGradient-batch_normalization_866/moments/mean:output:0*
T0*
_output_shapes

:m?
1batch_normalization_866/moments/SquaredDifferenceSquaredDifferencedense_962/BiasAdd:output:05batch_normalization_866/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????m?
:batch_normalization_866/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
(batch_normalization_866/moments/varianceMean5batch_normalization_866/moments/SquaredDifference:z:0Cbatch_normalization_866/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(?
'batch_normalization_866/moments/SqueezeSqueeze-batch_normalization_866/moments/mean:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 ?
)batch_normalization_866/moments/Squeeze_1Squeeze1batch_normalization_866/moments/variance:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 r
-batch_normalization_866/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_866/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_866_assignmovingavg_readvariableop_resource*
_output_shapes
:m*
dtype0?
+batch_normalization_866/AssignMovingAvg/subSub>batch_normalization_866/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_866/moments/Squeeze:output:0*
T0*
_output_shapes
:m?
+batch_normalization_866/AssignMovingAvg/mulMul/batch_normalization_866/AssignMovingAvg/sub:z:06batch_normalization_866/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:m?
'batch_normalization_866/AssignMovingAvgAssignSubVariableOp?batch_normalization_866_assignmovingavg_readvariableop_resource/batch_normalization_866/AssignMovingAvg/mul:z:07^batch_normalization_866/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_866/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8batch_normalization_866/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_866_assignmovingavg_1_readvariableop_resource*
_output_shapes
:m*
dtype0?
-batch_normalization_866/AssignMovingAvg_1/subSub@batch_normalization_866/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_866/moments/Squeeze_1:output:0*
T0*
_output_shapes
:m?
-batch_normalization_866/AssignMovingAvg_1/mulMul1batch_normalization_866/AssignMovingAvg_1/sub:z:08batch_normalization_866/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:m?
)batch_normalization_866/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_866_assignmovingavg_1_readvariableop_resource1batch_normalization_866/AssignMovingAvg_1/mul:z:09^batch_normalization_866/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_866/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_866/batchnorm/addAddV22batch_normalization_866/moments/Squeeze_1:output:00batch_normalization_866/batchnorm/add/y:output:0*
T0*
_output_shapes
:m?
'batch_normalization_866/batchnorm/RsqrtRsqrt)batch_normalization_866/batchnorm/add:z:0*
T0*
_output_shapes
:m?
4batch_normalization_866/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_866_batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0?
%batch_normalization_866/batchnorm/mulMul+batch_normalization_866/batchnorm/Rsqrt:y:0<batch_normalization_866/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:m?
'batch_normalization_866/batchnorm/mul_1Muldense_962/BiasAdd:output:0)batch_normalization_866/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????m?
'batch_normalization_866/batchnorm/mul_2Mul0batch_normalization_866/moments/Squeeze:output:0)batch_normalization_866/batchnorm/mul:z:0*
T0*
_output_shapes
:m?
0batch_normalization_866/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_866_batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0?
%batch_normalization_866/batchnorm/subSub8batch_normalization_866/batchnorm/ReadVariableOp:value:0+batch_normalization_866/batchnorm/mul_2:z:0*
T0*
_output_shapes
:m?
'batch_normalization_866/batchnorm/add_1AddV2+batch_normalization_866/batchnorm/mul_1:z:0)batch_normalization_866/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????m?
leaky_re_lu_866/LeakyRelu	LeakyRelu+batch_normalization_866/batchnorm/add_1:z:0*'
_output_shapes
:?????????m*
alpha%???>?
dense_963/MatMul/ReadVariableOpReadVariableOp(dense_963_matmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
dense_963/MatMulMatMul'leaky_re_lu_866/LeakyRelu:activations:0'dense_963/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
 dense_963/BiasAdd/ReadVariableOpReadVariableOp)dense_963_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
dense_963/BiasAddBiasAdddense_963/MatMul:product:0(dense_963/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
6batch_normalization_867/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization_867/moments/meanMeandense_963/BiasAdd:output:0?batch_normalization_867/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(?
,batch_normalization_867/moments/StopGradientStopGradient-batch_normalization_867/moments/mean:output:0*
T0*
_output_shapes

:m?
1batch_normalization_867/moments/SquaredDifferenceSquaredDifferencedense_963/BiasAdd:output:05batch_normalization_867/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????m?
:batch_normalization_867/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
(batch_normalization_867/moments/varianceMean5batch_normalization_867/moments/SquaredDifference:z:0Cbatch_normalization_867/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(?
'batch_normalization_867/moments/SqueezeSqueeze-batch_normalization_867/moments/mean:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 ?
)batch_normalization_867/moments/Squeeze_1Squeeze1batch_normalization_867/moments/variance:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 r
-batch_normalization_867/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_867/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_867_assignmovingavg_readvariableop_resource*
_output_shapes
:m*
dtype0?
+batch_normalization_867/AssignMovingAvg/subSub>batch_normalization_867/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_867/moments/Squeeze:output:0*
T0*
_output_shapes
:m?
+batch_normalization_867/AssignMovingAvg/mulMul/batch_normalization_867/AssignMovingAvg/sub:z:06batch_normalization_867/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:m?
'batch_normalization_867/AssignMovingAvgAssignSubVariableOp?batch_normalization_867_assignmovingavg_readvariableop_resource/batch_normalization_867/AssignMovingAvg/mul:z:07^batch_normalization_867/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_867/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8batch_normalization_867/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_867_assignmovingavg_1_readvariableop_resource*
_output_shapes
:m*
dtype0?
-batch_normalization_867/AssignMovingAvg_1/subSub@batch_normalization_867/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_867/moments/Squeeze_1:output:0*
T0*
_output_shapes
:m?
-batch_normalization_867/AssignMovingAvg_1/mulMul1batch_normalization_867/AssignMovingAvg_1/sub:z:08batch_normalization_867/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:m?
)batch_normalization_867/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_867_assignmovingavg_1_readvariableop_resource1batch_normalization_867/AssignMovingAvg_1/mul:z:09^batch_normalization_867/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_867/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_867/batchnorm/addAddV22batch_normalization_867/moments/Squeeze_1:output:00batch_normalization_867/batchnorm/add/y:output:0*
T0*
_output_shapes
:m?
'batch_normalization_867/batchnorm/RsqrtRsqrt)batch_normalization_867/batchnorm/add:z:0*
T0*
_output_shapes
:m?
4batch_normalization_867/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_867_batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0?
%batch_normalization_867/batchnorm/mulMul+batch_normalization_867/batchnorm/Rsqrt:y:0<batch_normalization_867/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:m?
'batch_normalization_867/batchnorm/mul_1Muldense_963/BiasAdd:output:0)batch_normalization_867/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????m?
'batch_normalization_867/batchnorm/mul_2Mul0batch_normalization_867/moments/Squeeze:output:0)batch_normalization_867/batchnorm/mul:z:0*
T0*
_output_shapes
:m?
0batch_normalization_867/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_867_batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0?
%batch_normalization_867/batchnorm/subSub8batch_normalization_867/batchnorm/ReadVariableOp:value:0+batch_normalization_867/batchnorm/mul_2:z:0*
T0*
_output_shapes
:m?
'batch_normalization_867/batchnorm/add_1AddV2+batch_normalization_867/batchnorm/mul_1:z:0)batch_normalization_867/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????m?
leaky_re_lu_867/LeakyRelu	LeakyRelu+batch_normalization_867/batchnorm/add_1:z:0*'
_output_shapes
:?????????m*
alpha%???>?
dense_964/MatMul/ReadVariableOpReadVariableOp(dense_964_matmul_readvariableop_resource*
_output_shapes

:m.*
dtype0?
dense_964/MatMulMatMul'leaky_re_lu_867/LeakyRelu:activations:0'dense_964/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
 dense_964/BiasAdd/ReadVariableOpReadVariableOp)dense_964_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype0?
dense_964/BiasAddBiasAdddense_964/MatMul:product:0(dense_964/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
6batch_normalization_868/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization_868/moments/meanMeandense_964/BiasAdd:output:0?batch_normalization_868/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(?
,batch_normalization_868/moments/StopGradientStopGradient-batch_normalization_868/moments/mean:output:0*
T0*
_output_shapes

:.?
1batch_normalization_868/moments/SquaredDifferenceSquaredDifferencedense_964/BiasAdd:output:05batch_normalization_868/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????.?
:batch_normalization_868/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
(batch_normalization_868/moments/varianceMean5batch_normalization_868/moments/SquaredDifference:z:0Cbatch_normalization_868/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(?
'batch_normalization_868/moments/SqueezeSqueeze-batch_normalization_868/moments/mean:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 ?
)batch_normalization_868/moments/Squeeze_1Squeeze1batch_normalization_868/moments/variance:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 r
-batch_normalization_868/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_868/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_868_assignmovingavg_readvariableop_resource*
_output_shapes
:.*
dtype0?
+batch_normalization_868/AssignMovingAvg/subSub>batch_normalization_868/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_868/moments/Squeeze:output:0*
T0*
_output_shapes
:.?
+batch_normalization_868/AssignMovingAvg/mulMul/batch_normalization_868/AssignMovingAvg/sub:z:06batch_normalization_868/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:.?
'batch_normalization_868/AssignMovingAvgAssignSubVariableOp?batch_normalization_868_assignmovingavg_readvariableop_resource/batch_normalization_868/AssignMovingAvg/mul:z:07^batch_normalization_868/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_868/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8batch_normalization_868/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_868_assignmovingavg_1_readvariableop_resource*
_output_shapes
:.*
dtype0?
-batch_normalization_868/AssignMovingAvg_1/subSub@batch_normalization_868/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_868/moments/Squeeze_1:output:0*
T0*
_output_shapes
:.?
-batch_normalization_868/AssignMovingAvg_1/mulMul1batch_normalization_868/AssignMovingAvg_1/sub:z:08batch_normalization_868/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:.?
)batch_normalization_868/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_868_assignmovingavg_1_readvariableop_resource1batch_normalization_868/AssignMovingAvg_1/mul:z:09^batch_normalization_868/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_868/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_868/batchnorm/addAddV22batch_normalization_868/moments/Squeeze_1:output:00batch_normalization_868/batchnorm/add/y:output:0*
T0*
_output_shapes
:.?
'batch_normalization_868/batchnorm/RsqrtRsqrt)batch_normalization_868/batchnorm/add:z:0*
T0*
_output_shapes
:.?
4batch_normalization_868/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_868_batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0?
%batch_normalization_868/batchnorm/mulMul+batch_normalization_868/batchnorm/Rsqrt:y:0<batch_normalization_868/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.?
'batch_normalization_868/batchnorm/mul_1Muldense_964/BiasAdd:output:0)batch_normalization_868/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.?
'batch_normalization_868/batchnorm/mul_2Mul0batch_normalization_868/moments/Squeeze:output:0)batch_normalization_868/batchnorm/mul:z:0*
T0*
_output_shapes
:.?
0batch_normalization_868/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_868_batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0?
%batch_normalization_868/batchnorm/subSub8batch_normalization_868/batchnorm/ReadVariableOp:value:0+batch_normalization_868/batchnorm/mul_2:z:0*
T0*
_output_shapes
:.?
'batch_normalization_868/batchnorm/add_1AddV2+batch_normalization_868/batchnorm/mul_1:z:0)batch_normalization_868/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.?
leaky_re_lu_868/LeakyRelu	LeakyRelu+batch_normalization_868/batchnorm/add_1:z:0*'
_output_shapes
:?????????.*
alpha%???>?
dense_965/MatMul/ReadVariableOpReadVariableOp(dense_965_matmul_readvariableop_resource*
_output_shapes

:..*
dtype0?
dense_965/MatMulMatMul'leaky_re_lu_868/LeakyRelu:activations:0'dense_965/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
 dense_965/BiasAdd/ReadVariableOpReadVariableOp)dense_965_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype0?
dense_965/BiasAddBiasAdddense_965/MatMul:product:0(dense_965/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
6batch_normalization_869/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization_869/moments/meanMeandense_965/BiasAdd:output:0?batch_normalization_869/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(?
,batch_normalization_869/moments/StopGradientStopGradient-batch_normalization_869/moments/mean:output:0*
T0*
_output_shapes

:.?
1batch_normalization_869/moments/SquaredDifferenceSquaredDifferencedense_965/BiasAdd:output:05batch_normalization_869/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????.?
:batch_normalization_869/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
(batch_normalization_869/moments/varianceMean5batch_normalization_869/moments/SquaredDifference:z:0Cbatch_normalization_869/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(?
'batch_normalization_869/moments/SqueezeSqueeze-batch_normalization_869/moments/mean:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 ?
)batch_normalization_869/moments/Squeeze_1Squeeze1batch_normalization_869/moments/variance:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 r
-batch_normalization_869/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_869/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_869_assignmovingavg_readvariableop_resource*
_output_shapes
:.*
dtype0?
+batch_normalization_869/AssignMovingAvg/subSub>batch_normalization_869/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_869/moments/Squeeze:output:0*
T0*
_output_shapes
:.?
+batch_normalization_869/AssignMovingAvg/mulMul/batch_normalization_869/AssignMovingAvg/sub:z:06batch_normalization_869/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:.?
'batch_normalization_869/AssignMovingAvgAssignSubVariableOp?batch_normalization_869_assignmovingavg_readvariableop_resource/batch_normalization_869/AssignMovingAvg/mul:z:07^batch_normalization_869/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_869/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8batch_normalization_869/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_869_assignmovingavg_1_readvariableop_resource*
_output_shapes
:.*
dtype0?
-batch_normalization_869/AssignMovingAvg_1/subSub@batch_normalization_869/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_869/moments/Squeeze_1:output:0*
T0*
_output_shapes
:.?
-batch_normalization_869/AssignMovingAvg_1/mulMul1batch_normalization_869/AssignMovingAvg_1/sub:z:08batch_normalization_869/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:.?
)batch_normalization_869/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_869_assignmovingavg_1_readvariableop_resource1batch_normalization_869/AssignMovingAvg_1/mul:z:09^batch_normalization_869/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_869/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_869/batchnorm/addAddV22batch_normalization_869/moments/Squeeze_1:output:00batch_normalization_869/batchnorm/add/y:output:0*
T0*
_output_shapes
:.?
'batch_normalization_869/batchnorm/RsqrtRsqrt)batch_normalization_869/batchnorm/add:z:0*
T0*
_output_shapes
:.?
4batch_normalization_869/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_869_batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0?
%batch_normalization_869/batchnorm/mulMul+batch_normalization_869/batchnorm/Rsqrt:y:0<batch_normalization_869/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.?
'batch_normalization_869/batchnorm/mul_1Muldense_965/BiasAdd:output:0)batch_normalization_869/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.?
'batch_normalization_869/batchnorm/mul_2Mul0batch_normalization_869/moments/Squeeze:output:0)batch_normalization_869/batchnorm/mul:z:0*
T0*
_output_shapes
:.?
0batch_normalization_869/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_869_batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0?
%batch_normalization_869/batchnorm/subSub8batch_normalization_869/batchnorm/ReadVariableOp:value:0+batch_normalization_869/batchnorm/mul_2:z:0*
T0*
_output_shapes
:.?
'batch_normalization_869/batchnorm/add_1AddV2+batch_normalization_869/batchnorm/mul_1:z:0)batch_normalization_869/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.?
leaky_re_lu_869/LeakyRelu	LeakyRelu+batch_normalization_869/batchnorm/add_1:z:0*'
_output_shapes
:?????????.*
alpha%???>?
dense_966/MatMul/ReadVariableOpReadVariableOp(dense_966_matmul_readvariableop_resource*
_output_shapes

:.]*
dtype0?
dense_966/MatMulMatMul'leaky_re_lu_869/LeakyRelu:activations:0'dense_966/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]?
 dense_966/BiasAdd/ReadVariableOpReadVariableOp)dense_966_biasadd_readvariableop_resource*
_output_shapes
:]*
dtype0?
dense_966/BiasAddBiasAdddense_966/MatMul:product:0(dense_966/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]?
6batch_normalization_870/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization_870/moments/meanMeandense_966/BiasAdd:output:0?batch_normalization_870/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:]*
	keep_dims(?
,batch_normalization_870/moments/StopGradientStopGradient-batch_normalization_870/moments/mean:output:0*
T0*
_output_shapes

:]?
1batch_normalization_870/moments/SquaredDifferenceSquaredDifferencedense_966/BiasAdd:output:05batch_normalization_870/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????]?
:batch_normalization_870/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
(batch_normalization_870/moments/varianceMean5batch_normalization_870/moments/SquaredDifference:z:0Cbatch_normalization_870/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:]*
	keep_dims(?
'batch_normalization_870/moments/SqueezeSqueeze-batch_normalization_870/moments/mean:output:0*
T0*
_output_shapes
:]*
squeeze_dims
 ?
)batch_normalization_870/moments/Squeeze_1Squeeze1batch_normalization_870/moments/variance:output:0*
T0*
_output_shapes
:]*
squeeze_dims
 r
-batch_normalization_870/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_870/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_870_assignmovingavg_readvariableop_resource*
_output_shapes
:]*
dtype0?
+batch_normalization_870/AssignMovingAvg/subSub>batch_normalization_870/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_870/moments/Squeeze:output:0*
T0*
_output_shapes
:]?
+batch_normalization_870/AssignMovingAvg/mulMul/batch_normalization_870/AssignMovingAvg/sub:z:06batch_normalization_870/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:]?
'batch_normalization_870/AssignMovingAvgAssignSubVariableOp?batch_normalization_870_assignmovingavg_readvariableop_resource/batch_normalization_870/AssignMovingAvg/mul:z:07^batch_normalization_870/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_870/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8batch_normalization_870/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_870_assignmovingavg_1_readvariableop_resource*
_output_shapes
:]*
dtype0?
-batch_normalization_870/AssignMovingAvg_1/subSub@batch_normalization_870/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_870/moments/Squeeze_1:output:0*
T0*
_output_shapes
:]?
-batch_normalization_870/AssignMovingAvg_1/mulMul1batch_normalization_870/AssignMovingAvg_1/sub:z:08batch_normalization_870/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:]?
)batch_normalization_870/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_870_assignmovingavg_1_readvariableop_resource1batch_normalization_870/AssignMovingAvg_1/mul:z:09^batch_normalization_870/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_870/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_870/batchnorm/addAddV22batch_normalization_870/moments/Squeeze_1:output:00batch_normalization_870/batchnorm/add/y:output:0*
T0*
_output_shapes
:]?
'batch_normalization_870/batchnorm/RsqrtRsqrt)batch_normalization_870/batchnorm/add:z:0*
T0*
_output_shapes
:]?
4batch_normalization_870/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_870_batchnorm_mul_readvariableop_resource*
_output_shapes
:]*
dtype0?
%batch_normalization_870/batchnorm/mulMul+batch_normalization_870/batchnorm/Rsqrt:y:0<batch_normalization_870/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:]?
'batch_normalization_870/batchnorm/mul_1Muldense_966/BiasAdd:output:0)batch_normalization_870/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????]?
'batch_normalization_870/batchnorm/mul_2Mul0batch_normalization_870/moments/Squeeze:output:0)batch_normalization_870/batchnorm/mul:z:0*
T0*
_output_shapes
:]?
0batch_normalization_870/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_870_batchnorm_readvariableop_resource*
_output_shapes
:]*
dtype0?
%batch_normalization_870/batchnorm/subSub8batch_normalization_870/batchnorm/ReadVariableOp:value:0+batch_normalization_870/batchnorm/mul_2:z:0*
T0*
_output_shapes
:]?
'batch_normalization_870/batchnorm/add_1AddV2+batch_normalization_870/batchnorm/mul_1:z:0)batch_normalization_870/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????]?
leaky_re_lu_870/LeakyRelu	LeakyRelu+batch_normalization_870/batchnorm/add_1:z:0*'
_output_shapes
:?????????]*
alpha%???>?
dense_967/MatMul/ReadVariableOpReadVariableOp(dense_967_matmul_readvariableop_resource*
_output_shapes

:]*
dtype0?
dense_967/MatMulMatMul'leaky_re_lu_870/LeakyRelu:activations:0'dense_967/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_967/BiasAdd/ReadVariableOpReadVariableOp)dense_967_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_967/BiasAddBiasAdddense_967/MatMul:product:0(dense_967/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
2dense_961/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_961_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#dense_961/kernel/Regularizer/SquareSquare:dense_961/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ms
"dense_961/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_961/kernel/Regularizer/SumSum'dense_961/kernel/Regularizer/Square:y:0+dense_961/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_961/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_961/kernel/Regularizer/mulMul+dense_961/kernel/Regularizer/mul/x:output:0)dense_961/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_962/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_962_matmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
#dense_962/kernel/Regularizer/SquareSquare:dense_962/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_962/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_962/kernel/Regularizer/SumSum'dense_962/kernel/Regularizer/Square:y:0+dense_962/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_962/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_962/kernel/Regularizer/mulMul+dense_962/kernel/Regularizer/mul/x:output:0)dense_962/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_963/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_963_matmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
#dense_963/kernel/Regularizer/SquareSquare:dense_963/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_963/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_963/kernel/Regularizer/SumSum'dense_963/kernel/Regularizer/Square:y:0+dense_963/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_963/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_963/kernel/Regularizer/mulMul+dense_963/kernel/Regularizer/mul/x:output:0)dense_963/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_964/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_964_matmul_readvariableop_resource*
_output_shapes

:m.*
dtype0?
#dense_964/kernel/Regularizer/SquareSquare:dense_964/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m.s
"dense_964/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_964/kernel/Regularizer/SumSum'dense_964/kernel/Regularizer/Square:y:0+dense_964/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_964/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_964/kernel/Regularizer/mulMul+dense_964/kernel/Regularizer/mul/x:output:0)dense_964/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_965/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_965_matmul_readvariableop_resource*
_output_shapes

:..*
dtype0?
#dense_965/kernel/Regularizer/SquareSquare:dense_965/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:..s
"dense_965/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_965/kernel/Regularizer/SumSum'dense_965/kernel/Regularizer/Square:y:0+dense_965/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_965/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_965/kernel/Regularizer/mulMul+dense_965/kernel/Regularizer/mul/x:output:0)dense_965/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_966/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_966_matmul_readvariableop_resource*
_output_shapes

:.]*
dtype0?
#dense_966/kernel/Regularizer/SquareSquare:dense_966/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:.]s
"dense_966/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_966/kernel/Regularizer/SumSum'dense_966/kernel/Regularizer/Square:y:0+dense_966/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_966/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?<?
 dense_966/kernel/Regularizer/mulMul+dense_966/kernel/Regularizer/mul/x:output:0)dense_966/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_967/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^batch_normalization_865/AssignMovingAvg7^batch_normalization_865/AssignMovingAvg/ReadVariableOp*^batch_normalization_865/AssignMovingAvg_19^batch_normalization_865/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_865/batchnorm/ReadVariableOp5^batch_normalization_865/batchnorm/mul/ReadVariableOp(^batch_normalization_866/AssignMovingAvg7^batch_normalization_866/AssignMovingAvg/ReadVariableOp*^batch_normalization_866/AssignMovingAvg_19^batch_normalization_866/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_866/batchnorm/ReadVariableOp5^batch_normalization_866/batchnorm/mul/ReadVariableOp(^batch_normalization_867/AssignMovingAvg7^batch_normalization_867/AssignMovingAvg/ReadVariableOp*^batch_normalization_867/AssignMovingAvg_19^batch_normalization_867/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_867/batchnorm/ReadVariableOp5^batch_normalization_867/batchnorm/mul/ReadVariableOp(^batch_normalization_868/AssignMovingAvg7^batch_normalization_868/AssignMovingAvg/ReadVariableOp*^batch_normalization_868/AssignMovingAvg_19^batch_normalization_868/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_868/batchnorm/ReadVariableOp5^batch_normalization_868/batchnorm/mul/ReadVariableOp(^batch_normalization_869/AssignMovingAvg7^batch_normalization_869/AssignMovingAvg/ReadVariableOp*^batch_normalization_869/AssignMovingAvg_19^batch_normalization_869/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_869/batchnorm/ReadVariableOp5^batch_normalization_869/batchnorm/mul/ReadVariableOp(^batch_normalization_870/AssignMovingAvg7^batch_normalization_870/AssignMovingAvg/ReadVariableOp*^batch_normalization_870/AssignMovingAvg_19^batch_normalization_870/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_870/batchnorm/ReadVariableOp5^batch_normalization_870/batchnorm/mul/ReadVariableOp!^dense_961/BiasAdd/ReadVariableOp ^dense_961/MatMul/ReadVariableOp3^dense_961/kernel/Regularizer/Square/ReadVariableOp!^dense_962/BiasAdd/ReadVariableOp ^dense_962/MatMul/ReadVariableOp3^dense_962/kernel/Regularizer/Square/ReadVariableOp!^dense_963/BiasAdd/ReadVariableOp ^dense_963/MatMul/ReadVariableOp3^dense_963/kernel/Regularizer/Square/ReadVariableOp!^dense_964/BiasAdd/ReadVariableOp ^dense_964/MatMul/ReadVariableOp3^dense_964/kernel/Regularizer/Square/ReadVariableOp!^dense_965/BiasAdd/ReadVariableOp ^dense_965/MatMul/ReadVariableOp3^dense_965/kernel/Regularizer/Square/ReadVariableOp!^dense_966/BiasAdd/ReadVariableOp ^dense_966/MatMul/ReadVariableOp3^dense_966/kernel/Regularizer/Square/ReadVariableOp!^dense_967/BiasAdd/ReadVariableOp ^dense_967/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_865/AssignMovingAvg'batch_normalization_865/AssignMovingAvg2p
6batch_normalization_865/AssignMovingAvg/ReadVariableOp6batch_normalization_865/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_865/AssignMovingAvg_1)batch_normalization_865/AssignMovingAvg_12t
8batch_normalization_865/AssignMovingAvg_1/ReadVariableOp8batch_normalization_865/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_865/batchnorm/ReadVariableOp0batch_normalization_865/batchnorm/ReadVariableOp2l
4batch_normalization_865/batchnorm/mul/ReadVariableOp4batch_normalization_865/batchnorm/mul/ReadVariableOp2R
'batch_normalization_866/AssignMovingAvg'batch_normalization_866/AssignMovingAvg2p
6batch_normalization_866/AssignMovingAvg/ReadVariableOp6batch_normalization_866/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_866/AssignMovingAvg_1)batch_normalization_866/AssignMovingAvg_12t
8batch_normalization_866/AssignMovingAvg_1/ReadVariableOp8batch_normalization_866/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_866/batchnorm/ReadVariableOp0batch_normalization_866/batchnorm/ReadVariableOp2l
4batch_normalization_866/batchnorm/mul/ReadVariableOp4batch_normalization_866/batchnorm/mul/ReadVariableOp2R
'batch_normalization_867/AssignMovingAvg'batch_normalization_867/AssignMovingAvg2p
6batch_normalization_867/AssignMovingAvg/ReadVariableOp6batch_normalization_867/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_867/AssignMovingAvg_1)batch_normalization_867/AssignMovingAvg_12t
8batch_normalization_867/AssignMovingAvg_1/ReadVariableOp8batch_normalization_867/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_867/batchnorm/ReadVariableOp0batch_normalization_867/batchnorm/ReadVariableOp2l
4batch_normalization_867/batchnorm/mul/ReadVariableOp4batch_normalization_867/batchnorm/mul/ReadVariableOp2R
'batch_normalization_868/AssignMovingAvg'batch_normalization_868/AssignMovingAvg2p
6batch_normalization_868/AssignMovingAvg/ReadVariableOp6batch_normalization_868/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_868/AssignMovingAvg_1)batch_normalization_868/AssignMovingAvg_12t
8batch_normalization_868/AssignMovingAvg_1/ReadVariableOp8batch_normalization_868/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_868/batchnorm/ReadVariableOp0batch_normalization_868/batchnorm/ReadVariableOp2l
4batch_normalization_868/batchnorm/mul/ReadVariableOp4batch_normalization_868/batchnorm/mul/ReadVariableOp2R
'batch_normalization_869/AssignMovingAvg'batch_normalization_869/AssignMovingAvg2p
6batch_normalization_869/AssignMovingAvg/ReadVariableOp6batch_normalization_869/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_869/AssignMovingAvg_1)batch_normalization_869/AssignMovingAvg_12t
8batch_normalization_869/AssignMovingAvg_1/ReadVariableOp8batch_normalization_869/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_869/batchnorm/ReadVariableOp0batch_normalization_869/batchnorm/ReadVariableOp2l
4batch_normalization_869/batchnorm/mul/ReadVariableOp4batch_normalization_869/batchnorm/mul/ReadVariableOp2R
'batch_normalization_870/AssignMovingAvg'batch_normalization_870/AssignMovingAvg2p
6batch_normalization_870/AssignMovingAvg/ReadVariableOp6batch_normalization_870/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_870/AssignMovingAvg_1)batch_normalization_870/AssignMovingAvg_12t
8batch_normalization_870/AssignMovingAvg_1/ReadVariableOp8batch_normalization_870/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_870/batchnorm/ReadVariableOp0batch_normalization_870/batchnorm/ReadVariableOp2l
4batch_normalization_870/batchnorm/mul/ReadVariableOp4batch_normalization_870/batchnorm/mul/ReadVariableOp2D
 dense_961/BiasAdd/ReadVariableOp dense_961/BiasAdd/ReadVariableOp2B
dense_961/MatMul/ReadVariableOpdense_961/MatMul/ReadVariableOp2h
2dense_961/kernel/Regularizer/Square/ReadVariableOp2dense_961/kernel/Regularizer/Square/ReadVariableOp2D
 dense_962/BiasAdd/ReadVariableOp dense_962/BiasAdd/ReadVariableOp2B
dense_962/MatMul/ReadVariableOpdense_962/MatMul/ReadVariableOp2h
2dense_962/kernel/Regularizer/Square/ReadVariableOp2dense_962/kernel/Regularizer/Square/ReadVariableOp2D
 dense_963/BiasAdd/ReadVariableOp dense_963/BiasAdd/ReadVariableOp2B
dense_963/MatMul/ReadVariableOpdense_963/MatMul/ReadVariableOp2h
2dense_963/kernel/Regularizer/Square/ReadVariableOp2dense_963/kernel/Regularizer/Square/ReadVariableOp2D
 dense_964/BiasAdd/ReadVariableOp dense_964/BiasAdd/ReadVariableOp2B
dense_964/MatMul/ReadVariableOpdense_964/MatMul/ReadVariableOp2h
2dense_964/kernel/Regularizer/Square/ReadVariableOp2dense_964/kernel/Regularizer/Square/ReadVariableOp2D
 dense_965/BiasAdd/ReadVariableOp dense_965/BiasAdd/ReadVariableOp2B
dense_965/MatMul/ReadVariableOpdense_965/MatMul/ReadVariableOp2h
2dense_965/kernel/Regularizer/Square/ReadVariableOp2dense_965/kernel/Regularizer/Square/ReadVariableOp2D
 dense_966/BiasAdd/ReadVariableOp dense_966/BiasAdd/ReadVariableOp2B
dense_966/MatMul/ReadVariableOpdense_966/MatMul/ReadVariableOp2h
2dense_966/kernel/Regularizer/Square/ReadVariableOp2dense_966/kernel/Regularizer/Square/ReadVariableOp2D
 dense_967/BiasAdd/ReadVariableOp dense_967/BiasAdd/ReadVariableOp2B
dense_967/MatMul/ReadVariableOpdense_967/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_865089

inputs/
!batchnorm_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:.1
#batchnorm_readvariableop_1_resource:.1
#batchnorm_readvariableop_2_resource:.
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:.P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:.~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:.*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:.z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:.*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:.r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????.?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_868_layer_call_fn_865007

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_868_layer_call_and_return_conditional_losses_862717`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????.:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_864760

inputs5
'assignmovingavg_readvariableop_resource:m7
)assignmovingavg_1_readvariableop_resource:m3
%batchnorm_mul_readvariableop_resource:m/
!batchnorm_readvariableop_resource:m
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:m?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????ml
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:m*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:mx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:m?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:m*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:m~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:m?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:mP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:mc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????mh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:mv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:mr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????mb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_865317M
;dense_964_kernel_regularizer_square_readvariableop_resource:m.
identity??2dense_964/kernel/Regularizer/Square/ReadVariableOp?
2dense_964/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_964_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:m.*
dtype0?
#dense_964/kernel/Regularizer/SquareSquare:dense_964/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m.s
"dense_964/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_964/kernel/Regularizer/SumSum'dense_964/kernel/Regularizer/Square:y:0+dense_964/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_964/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_964/kernel/Regularizer/mulMul+dense_964/kernel/Regularizer/mul/x:output:0)dense_964/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_964/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_964/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_964/kernel/Regularizer/Square/ReadVariableOp2dense_964/kernel/Regularizer/Square/ReadVariableOp
??
?
I__inference_sequential_96_layer_call_and_return_conditional_losses_862848

inputs
normalization_96_sub_y
normalization_96_sqrt_x"
dense_961_862584:m
dense_961_862586:m,
batch_normalization_865_862589:m,
batch_normalization_865_862591:m,
batch_normalization_865_862593:m,
batch_normalization_865_862595:m"
dense_962_862622:mm
dense_962_862624:m,
batch_normalization_866_862627:m,
batch_normalization_866_862629:m,
batch_normalization_866_862631:m,
batch_normalization_866_862633:m"
dense_963_862660:mm
dense_963_862662:m,
batch_normalization_867_862665:m,
batch_normalization_867_862667:m,
batch_normalization_867_862669:m,
batch_normalization_867_862671:m"
dense_964_862698:m.
dense_964_862700:.,
batch_normalization_868_862703:.,
batch_normalization_868_862705:.,
batch_normalization_868_862707:.,
batch_normalization_868_862709:."
dense_965_862736:..
dense_965_862738:.,
batch_normalization_869_862741:.,
batch_normalization_869_862743:.,
batch_normalization_869_862745:.,
batch_normalization_869_862747:."
dense_966_862774:.]
dense_966_862776:],
batch_normalization_870_862779:],
batch_normalization_870_862781:],
batch_normalization_870_862783:],
batch_normalization_870_862785:]"
dense_967_862806:]
dense_967_862808:
identity??/batch_normalization_865/StatefulPartitionedCall?/batch_normalization_866/StatefulPartitionedCall?/batch_normalization_867/StatefulPartitionedCall?/batch_normalization_868/StatefulPartitionedCall?/batch_normalization_869/StatefulPartitionedCall?/batch_normalization_870/StatefulPartitionedCall?!dense_961/StatefulPartitionedCall?2dense_961/kernel/Regularizer/Square/ReadVariableOp?!dense_962/StatefulPartitionedCall?2dense_962/kernel/Regularizer/Square/ReadVariableOp?!dense_963/StatefulPartitionedCall?2dense_963/kernel/Regularizer/Square/ReadVariableOp?!dense_964/StatefulPartitionedCall?2dense_964/kernel/Regularizer/Square/ReadVariableOp?!dense_965/StatefulPartitionedCall?2dense_965/kernel/Regularizer/Square/ReadVariableOp?!dense_966/StatefulPartitionedCall?2dense_966/kernel/Regularizer/Square/ReadVariableOp?!dense_967/StatefulPartitionedCallm
normalization_96/subSubinputsnormalization_96_sub_y*
T0*'
_output_shapes
:?????????_
normalization_96/SqrtSqrtnormalization_96_sqrt_x*
T0*
_output_shapes

:_
normalization_96/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_96/MaximumMaximumnormalization_96/Sqrt:y:0#normalization_96/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_96/truedivRealDivnormalization_96/sub:z:0normalization_96/Maximum:z:0*
T0*'
_output_shapes
:??????????
!dense_961/StatefulPartitionedCallStatefulPartitionedCallnormalization_96/truediv:z:0dense_961_862584dense_961_862586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_961_layer_call_and_return_conditional_losses_862583?
/batch_normalization_865/StatefulPartitionedCallStatefulPartitionedCall*dense_961/StatefulPartitionedCall:output:0batch_normalization_865_862589batch_normalization_865_862591batch_normalization_865_862593batch_normalization_865_862595*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_862085?
leaky_re_lu_865/PartitionedCallPartitionedCall8batch_normalization_865/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_862603?
!dense_962/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_865/PartitionedCall:output:0dense_962_862622dense_962_862624*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_962_layer_call_and_return_conditional_losses_862621?
/batch_normalization_866/StatefulPartitionedCallStatefulPartitionedCall*dense_962/StatefulPartitionedCall:output:0batch_normalization_866_862627batch_normalization_866_862629batch_normalization_866_862631batch_normalization_866_862633*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_862167?
leaky_re_lu_866/PartitionedCallPartitionedCall8batch_normalization_866/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_866_layer_call_and_return_conditional_losses_862641?
!dense_963/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_866/PartitionedCall:output:0dense_963_862660dense_963_862662*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_963_layer_call_and_return_conditional_losses_862659?
/batch_normalization_867/StatefulPartitionedCallStatefulPartitionedCall*dense_963/StatefulPartitionedCall:output:0batch_normalization_867_862665batch_normalization_867_862667batch_normalization_867_862669batch_normalization_867_862671*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_862249?
leaky_re_lu_867/PartitionedCallPartitionedCall8batch_normalization_867/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_867_layer_call_and_return_conditional_losses_862679?
!dense_964/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_867/PartitionedCall:output:0dense_964_862698dense_964_862700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_964_layer_call_and_return_conditional_losses_862697?
/batch_normalization_868/StatefulPartitionedCallStatefulPartitionedCall*dense_964/StatefulPartitionedCall:output:0batch_normalization_868_862703batch_normalization_868_862705batch_normalization_868_862707batch_normalization_868_862709*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_862331?
leaky_re_lu_868/PartitionedCallPartitionedCall8batch_normalization_868/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_868_layer_call_and_return_conditional_losses_862717?
!dense_965/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_868/PartitionedCall:output:0dense_965_862736dense_965_862738*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_965_layer_call_and_return_conditional_losses_862735?
/batch_normalization_869/StatefulPartitionedCallStatefulPartitionedCall*dense_965/StatefulPartitionedCall:output:0batch_normalization_869_862741batch_normalization_869_862743batch_normalization_869_862745batch_normalization_869_862747*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_862413?
leaky_re_lu_869/PartitionedCallPartitionedCall8batch_normalization_869/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_869_layer_call_and_return_conditional_losses_862755?
!dense_966/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_869/PartitionedCall:output:0dense_966_862774dense_966_862776*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_966_layer_call_and_return_conditional_losses_862773?
/batch_normalization_870/StatefulPartitionedCallStatefulPartitionedCall*dense_966/StatefulPartitionedCall:output:0batch_normalization_870_862779batch_normalization_870_862781batch_normalization_870_862783batch_normalization_870_862785*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_862495?
leaky_re_lu_870/PartitionedCallPartitionedCall8batch_normalization_870/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_870_layer_call_and_return_conditional_losses_862793?
!dense_967/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_870/PartitionedCall:output:0dense_967_862806dense_967_862808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_967_layer_call_and_return_conditional_losses_862805?
2dense_961/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_961_862584*
_output_shapes

:m*
dtype0?
#dense_961/kernel/Regularizer/SquareSquare:dense_961/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ms
"dense_961/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_961/kernel/Regularizer/SumSum'dense_961/kernel/Regularizer/Square:y:0+dense_961/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_961/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_961/kernel/Regularizer/mulMul+dense_961/kernel/Regularizer/mul/x:output:0)dense_961/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_962/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_962_862622*
_output_shapes

:mm*
dtype0?
#dense_962/kernel/Regularizer/SquareSquare:dense_962/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_962/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_962/kernel/Regularizer/SumSum'dense_962/kernel/Regularizer/Square:y:0+dense_962/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_962/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_962/kernel/Regularizer/mulMul+dense_962/kernel/Regularizer/mul/x:output:0)dense_962/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_963/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_963_862660*
_output_shapes

:mm*
dtype0?
#dense_963/kernel/Regularizer/SquareSquare:dense_963/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_963/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_963/kernel/Regularizer/SumSum'dense_963/kernel/Regularizer/Square:y:0+dense_963/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_963/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_963/kernel/Regularizer/mulMul+dense_963/kernel/Regularizer/mul/x:output:0)dense_963/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_964/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_964_862698*
_output_shapes

:m.*
dtype0?
#dense_964/kernel/Regularizer/SquareSquare:dense_964/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m.s
"dense_964/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_964/kernel/Regularizer/SumSum'dense_964/kernel/Regularizer/Square:y:0+dense_964/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_964/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_964/kernel/Regularizer/mulMul+dense_964/kernel/Regularizer/mul/x:output:0)dense_964/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_965/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_965_862736*
_output_shapes

:..*
dtype0?
#dense_965/kernel/Regularizer/SquareSquare:dense_965/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:..s
"dense_965/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_965/kernel/Regularizer/SumSum'dense_965/kernel/Regularizer/Square:y:0+dense_965/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_965/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_965/kernel/Regularizer/mulMul+dense_965/kernel/Regularizer/mul/x:output:0)dense_965/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_966/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_966_862774*
_output_shapes

:.]*
dtype0?
#dense_966/kernel/Regularizer/SquareSquare:dense_966/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:.]s
"dense_966/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_966/kernel/Regularizer/SumSum'dense_966/kernel/Regularizer/Square:y:0+dense_966/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_966/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?<?
 dense_966/kernel/Regularizer/mulMul+dense_966/kernel/Regularizer/mul/x:output:0)dense_966/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_967/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp0^batch_normalization_865/StatefulPartitionedCall0^batch_normalization_866/StatefulPartitionedCall0^batch_normalization_867/StatefulPartitionedCall0^batch_normalization_868/StatefulPartitionedCall0^batch_normalization_869/StatefulPartitionedCall0^batch_normalization_870/StatefulPartitionedCall"^dense_961/StatefulPartitionedCall3^dense_961/kernel/Regularizer/Square/ReadVariableOp"^dense_962/StatefulPartitionedCall3^dense_962/kernel/Regularizer/Square/ReadVariableOp"^dense_963/StatefulPartitionedCall3^dense_963/kernel/Regularizer/Square/ReadVariableOp"^dense_964/StatefulPartitionedCall3^dense_964/kernel/Regularizer/Square/ReadVariableOp"^dense_965/StatefulPartitionedCall3^dense_965/kernel/Regularizer/Square/ReadVariableOp"^dense_966/StatefulPartitionedCall3^dense_966/kernel/Regularizer/Square/ReadVariableOp"^dense_967/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_865/StatefulPartitionedCall/batch_normalization_865/StatefulPartitionedCall2b
/batch_normalization_866/StatefulPartitionedCall/batch_normalization_866/StatefulPartitionedCall2b
/batch_normalization_867/StatefulPartitionedCall/batch_normalization_867/StatefulPartitionedCall2b
/batch_normalization_868/StatefulPartitionedCall/batch_normalization_868/StatefulPartitionedCall2b
/batch_normalization_869/StatefulPartitionedCall/batch_normalization_869/StatefulPartitionedCall2b
/batch_normalization_870/StatefulPartitionedCall/batch_normalization_870/StatefulPartitionedCall2F
!dense_961/StatefulPartitionedCall!dense_961/StatefulPartitionedCall2h
2dense_961/kernel/Regularizer/Square/ReadVariableOp2dense_961/kernel/Regularizer/Square/ReadVariableOp2F
!dense_962/StatefulPartitionedCall!dense_962/StatefulPartitionedCall2h
2dense_962/kernel/Regularizer/Square/ReadVariableOp2dense_962/kernel/Regularizer/Square/ReadVariableOp2F
!dense_963/StatefulPartitionedCall!dense_963/StatefulPartitionedCall2h
2dense_963/kernel/Regularizer/Square/ReadVariableOp2dense_963/kernel/Regularizer/Square/ReadVariableOp2F
!dense_964/StatefulPartitionedCall!dense_964/StatefulPartitionedCall2h
2dense_964/kernel/Regularizer/Square/ReadVariableOp2dense_964/kernel/Regularizer/Square/ReadVariableOp2F
!dense_965/StatefulPartitionedCall!dense_965/StatefulPartitionedCall2h
2dense_965/kernel/Regularizer/Square/ReadVariableOp2dense_965/kernel/Regularizer/Square/ReadVariableOp2F
!dense_966/StatefulPartitionedCall!dense_966/StatefulPartitionedCall2h
2dense_966/kernel/Regularizer/Square/ReadVariableOp2dense_966/kernel/Regularizer/Square/ReadVariableOp2F
!dense_967/StatefulPartitionedCall!dense_967/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_862167

inputs/
!batchnorm_readvariableop_resource:m3
%batchnorm_mul_readvariableop_resource:m1
#batchnorm_readvariableop_1_resource:m1
#batchnorm_readvariableop_2_resource:m
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:mP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:mc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????mz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:m*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:mz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:m*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:mr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????mb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
??
?
I__inference_sequential_96_layer_call_and_return_conditional_losses_863718
normalization_96_input
normalization_96_sub_y
normalization_96_sqrt_x"
dense_961_863586:m
dense_961_863588:m,
batch_normalization_865_863591:m,
batch_normalization_865_863593:m,
batch_normalization_865_863595:m,
batch_normalization_865_863597:m"
dense_962_863601:mm
dense_962_863603:m,
batch_normalization_866_863606:m,
batch_normalization_866_863608:m,
batch_normalization_866_863610:m,
batch_normalization_866_863612:m"
dense_963_863616:mm
dense_963_863618:m,
batch_normalization_867_863621:m,
batch_normalization_867_863623:m,
batch_normalization_867_863625:m,
batch_normalization_867_863627:m"
dense_964_863631:m.
dense_964_863633:.,
batch_normalization_868_863636:.,
batch_normalization_868_863638:.,
batch_normalization_868_863640:.,
batch_normalization_868_863642:."
dense_965_863646:..
dense_965_863648:.,
batch_normalization_869_863651:.,
batch_normalization_869_863653:.,
batch_normalization_869_863655:.,
batch_normalization_869_863657:."
dense_966_863661:.]
dense_966_863663:],
batch_normalization_870_863666:],
batch_normalization_870_863668:],
batch_normalization_870_863670:],
batch_normalization_870_863672:]"
dense_967_863676:]
dense_967_863678:
identity??/batch_normalization_865/StatefulPartitionedCall?/batch_normalization_866/StatefulPartitionedCall?/batch_normalization_867/StatefulPartitionedCall?/batch_normalization_868/StatefulPartitionedCall?/batch_normalization_869/StatefulPartitionedCall?/batch_normalization_870/StatefulPartitionedCall?!dense_961/StatefulPartitionedCall?2dense_961/kernel/Regularizer/Square/ReadVariableOp?!dense_962/StatefulPartitionedCall?2dense_962/kernel/Regularizer/Square/ReadVariableOp?!dense_963/StatefulPartitionedCall?2dense_963/kernel/Regularizer/Square/ReadVariableOp?!dense_964/StatefulPartitionedCall?2dense_964/kernel/Regularizer/Square/ReadVariableOp?!dense_965/StatefulPartitionedCall?2dense_965/kernel/Regularizer/Square/ReadVariableOp?!dense_966/StatefulPartitionedCall?2dense_966/kernel/Regularizer/Square/ReadVariableOp?!dense_967/StatefulPartitionedCall}
normalization_96/subSubnormalization_96_inputnormalization_96_sub_y*
T0*'
_output_shapes
:?????????_
normalization_96/SqrtSqrtnormalization_96_sqrt_x*
T0*
_output_shapes

:_
normalization_96/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_96/MaximumMaximumnormalization_96/Sqrt:y:0#normalization_96/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_96/truedivRealDivnormalization_96/sub:z:0normalization_96/Maximum:z:0*
T0*'
_output_shapes
:??????????
!dense_961/StatefulPartitionedCallStatefulPartitionedCallnormalization_96/truediv:z:0dense_961_863586dense_961_863588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_961_layer_call_and_return_conditional_losses_862583?
/batch_normalization_865/StatefulPartitionedCallStatefulPartitionedCall*dense_961/StatefulPartitionedCall:output:0batch_normalization_865_863591batch_normalization_865_863593batch_normalization_865_863595batch_normalization_865_863597*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_862132?
leaky_re_lu_865/PartitionedCallPartitionedCall8batch_normalization_865/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_862603?
!dense_962/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_865/PartitionedCall:output:0dense_962_863601dense_962_863603*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_962_layer_call_and_return_conditional_losses_862621?
/batch_normalization_866/StatefulPartitionedCallStatefulPartitionedCall*dense_962/StatefulPartitionedCall:output:0batch_normalization_866_863606batch_normalization_866_863608batch_normalization_866_863610batch_normalization_866_863612*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_862214?
leaky_re_lu_866/PartitionedCallPartitionedCall8batch_normalization_866/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_866_layer_call_and_return_conditional_losses_862641?
!dense_963/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_866/PartitionedCall:output:0dense_963_863616dense_963_863618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_963_layer_call_and_return_conditional_losses_862659?
/batch_normalization_867/StatefulPartitionedCallStatefulPartitionedCall*dense_963/StatefulPartitionedCall:output:0batch_normalization_867_863621batch_normalization_867_863623batch_normalization_867_863625batch_normalization_867_863627*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_862296?
leaky_re_lu_867/PartitionedCallPartitionedCall8batch_normalization_867/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_867_layer_call_and_return_conditional_losses_862679?
!dense_964/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_867/PartitionedCall:output:0dense_964_863631dense_964_863633*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_964_layer_call_and_return_conditional_losses_862697?
/batch_normalization_868/StatefulPartitionedCallStatefulPartitionedCall*dense_964/StatefulPartitionedCall:output:0batch_normalization_868_863636batch_normalization_868_863638batch_normalization_868_863640batch_normalization_868_863642*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_862378?
leaky_re_lu_868/PartitionedCallPartitionedCall8batch_normalization_868/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_868_layer_call_and_return_conditional_losses_862717?
!dense_965/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_868/PartitionedCall:output:0dense_965_863646dense_965_863648*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_965_layer_call_and_return_conditional_losses_862735?
/batch_normalization_869/StatefulPartitionedCallStatefulPartitionedCall*dense_965/StatefulPartitionedCall:output:0batch_normalization_869_863651batch_normalization_869_863653batch_normalization_869_863655batch_normalization_869_863657*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_862460?
leaky_re_lu_869/PartitionedCallPartitionedCall8batch_normalization_869/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_869_layer_call_and_return_conditional_losses_862755?
!dense_966/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_869/PartitionedCall:output:0dense_966_863661dense_966_863663*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_966_layer_call_and_return_conditional_losses_862773?
/batch_normalization_870/StatefulPartitionedCallStatefulPartitionedCall*dense_966/StatefulPartitionedCall:output:0batch_normalization_870_863666batch_normalization_870_863668batch_normalization_870_863670batch_normalization_870_863672*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_862542?
leaky_re_lu_870/PartitionedCallPartitionedCall8batch_normalization_870/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_870_layer_call_and_return_conditional_losses_862793?
!dense_967/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_870/PartitionedCall:output:0dense_967_863676dense_967_863678*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_967_layer_call_and_return_conditional_losses_862805?
2dense_961/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_961_863586*
_output_shapes

:m*
dtype0?
#dense_961/kernel/Regularizer/SquareSquare:dense_961/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ms
"dense_961/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_961/kernel/Regularizer/SumSum'dense_961/kernel/Regularizer/Square:y:0+dense_961/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_961/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_961/kernel/Regularizer/mulMul+dense_961/kernel/Regularizer/mul/x:output:0)dense_961/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_962/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_962_863601*
_output_shapes

:mm*
dtype0?
#dense_962/kernel/Regularizer/SquareSquare:dense_962/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_962/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_962/kernel/Regularizer/SumSum'dense_962/kernel/Regularizer/Square:y:0+dense_962/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_962/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_962/kernel/Regularizer/mulMul+dense_962/kernel/Regularizer/mul/x:output:0)dense_962/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_963/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_963_863616*
_output_shapes

:mm*
dtype0?
#dense_963/kernel/Regularizer/SquareSquare:dense_963/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_963/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_963/kernel/Regularizer/SumSum'dense_963/kernel/Regularizer/Square:y:0+dense_963/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_963/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_963/kernel/Regularizer/mulMul+dense_963/kernel/Regularizer/mul/x:output:0)dense_963/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_964/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_964_863631*
_output_shapes

:m.*
dtype0?
#dense_964/kernel/Regularizer/SquareSquare:dense_964/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m.s
"dense_964/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_964/kernel/Regularizer/SumSum'dense_964/kernel/Regularizer/Square:y:0+dense_964/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_964/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_964/kernel/Regularizer/mulMul+dense_964/kernel/Regularizer/mul/x:output:0)dense_964/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_965/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_965_863646*
_output_shapes

:..*
dtype0?
#dense_965/kernel/Regularizer/SquareSquare:dense_965/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:..s
"dense_965/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_965/kernel/Regularizer/SumSum'dense_965/kernel/Regularizer/Square:y:0+dense_965/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_965/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_965/kernel/Regularizer/mulMul+dense_965/kernel/Regularizer/mul/x:output:0)dense_965/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_966/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_966_863661*
_output_shapes

:.]*
dtype0?
#dense_966/kernel/Regularizer/SquareSquare:dense_966/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:.]s
"dense_966/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_966/kernel/Regularizer/SumSum'dense_966/kernel/Regularizer/Square:y:0+dense_966/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_966/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?<?
 dense_966/kernel/Regularizer/mulMul+dense_966/kernel/Regularizer/mul/x:output:0)dense_966/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_967/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp0^batch_normalization_865/StatefulPartitionedCall0^batch_normalization_866/StatefulPartitionedCall0^batch_normalization_867/StatefulPartitionedCall0^batch_normalization_868/StatefulPartitionedCall0^batch_normalization_869/StatefulPartitionedCall0^batch_normalization_870/StatefulPartitionedCall"^dense_961/StatefulPartitionedCall3^dense_961/kernel/Regularizer/Square/ReadVariableOp"^dense_962/StatefulPartitionedCall3^dense_962/kernel/Regularizer/Square/ReadVariableOp"^dense_963/StatefulPartitionedCall3^dense_963/kernel/Regularizer/Square/ReadVariableOp"^dense_964/StatefulPartitionedCall3^dense_964/kernel/Regularizer/Square/ReadVariableOp"^dense_965/StatefulPartitionedCall3^dense_965/kernel/Regularizer/Square/ReadVariableOp"^dense_966/StatefulPartitionedCall3^dense_966/kernel/Regularizer/Square/ReadVariableOp"^dense_967/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_865/StatefulPartitionedCall/batch_normalization_865/StatefulPartitionedCall2b
/batch_normalization_866/StatefulPartitionedCall/batch_normalization_866/StatefulPartitionedCall2b
/batch_normalization_867/StatefulPartitionedCall/batch_normalization_867/StatefulPartitionedCall2b
/batch_normalization_868/StatefulPartitionedCall/batch_normalization_868/StatefulPartitionedCall2b
/batch_normalization_869/StatefulPartitionedCall/batch_normalization_869/StatefulPartitionedCall2b
/batch_normalization_870/StatefulPartitionedCall/batch_normalization_870/StatefulPartitionedCall2F
!dense_961/StatefulPartitionedCall!dense_961/StatefulPartitionedCall2h
2dense_961/kernel/Regularizer/Square/ReadVariableOp2dense_961/kernel/Regularizer/Square/ReadVariableOp2F
!dense_962/StatefulPartitionedCall!dense_962/StatefulPartitionedCall2h
2dense_962/kernel/Regularizer/Square/ReadVariableOp2dense_962/kernel/Regularizer/Square/ReadVariableOp2F
!dense_963/StatefulPartitionedCall!dense_963/StatefulPartitionedCall2h
2dense_963/kernel/Regularizer/Square/ReadVariableOp2dense_963/kernel/Regularizer/Square/ReadVariableOp2F
!dense_964/StatefulPartitionedCall!dense_964/StatefulPartitionedCall2h
2dense_964/kernel/Regularizer/Square/ReadVariableOp2dense_964/kernel/Regularizer/Square/ReadVariableOp2F
!dense_965/StatefulPartitionedCall!dense_965/StatefulPartitionedCall2h
2dense_965/kernel/Regularizer/Square/ReadVariableOp2dense_965/kernel/Regularizer/Square/ReadVariableOp2F
!dense_966/StatefulPartitionedCall!dense_966/StatefulPartitionedCall2h
2dense_966/kernel/Regularizer/Square/ReadVariableOp2dense_966/kernel/Regularizer/Square/ReadVariableOp2F
!dense_967/StatefulPartitionedCall!dense_967/StatefulPartitionedCall:_ [
'
_output_shapes
:?????????
0
_user_specified_namenormalization_96_input:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
8__inference_batch_normalization_869_layer_call_fn_865056

inputs
unknown:.
	unknown_0:.
	unknown_1:.
	unknown_2:.
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_862413o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????.`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_870_layer_call_fn_865177

inputs
unknown:]
	unknown_0:]
	unknown_1:]
	unknown_2:]
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_862495o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????]`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????]: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_864639

inputs5
'assignmovingavg_readvariableop_resource:m7
)assignmovingavg_1_readvariableop_resource:m3
%batchnorm_mul_readvariableop_resource:m/
!batchnorm_readvariableop_resource:m
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:m?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????ml
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:m*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:mx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:m?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:m*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:m~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:m?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:mP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:mc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????mh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:mv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:mr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????mb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_869_layer_call_and_return_conditional_losses_865133

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????.*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????.:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_862495

inputs/
!batchnorm_readvariableop_resource:]3
%batchnorm_mul_readvariableop_resource:]1
#batchnorm_readvariableop_1_resource:]1
#batchnorm_readvariableop_2_resource:]
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:]*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:]P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:]~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:]*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:]c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????]z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:]*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:]z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:]*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:]r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????]b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????]?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????]: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_868_layer_call_and_return_conditional_losses_862717

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????.*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????.:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_869_layer_call_fn_865128

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_869_layer_call_and_return_conditional_losses_862755`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????.:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_865_layer_call_fn_864644

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_862603`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????m"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????m:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
??
?.
__inference__traced_save_865661
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_961_kernel_read_readvariableop-
)savev2_dense_961_bias_read_readvariableop<
8savev2_batch_normalization_865_gamma_read_readvariableop;
7savev2_batch_normalization_865_beta_read_readvariableopB
>savev2_batch_normalization_865_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_865_moving_variance_read_readvariableop/
+savev2_dense_962_kernel_read_readvariableop-
)savev2_dense_962_bias_read_readvariableop<
8savev2_batch_normalization_866_gamma_read_readvariableop;
7savev2_batch_normalization_866_beta_read_readvariableopB
>savev2_batch_normalization_866_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_866_moving_variance_read_readvariableop/
+savev2_dense_963_kernel_read_readvariableop-
)savev2_dense_963_bias_read_readvariableop<
8savev2_batch_normalization_867_gamma_read_readvariableop;
7savev2_batch_normalization_867_beta_read_readvariableopB
>savev2_batch_normalization_867_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_867_moving_variance_read_readvariableop/
+savev2_dense_964_kernel_read_readvariableop-
)savev2_dense_964_bias_read_readvariableop<
8savev2_batch_normalization_868_gamma_read_readvariableop;
7savev2_batch_normalization_868_beta_read_readvariableopB
>savev2_batch_normalization_868_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_868_moving_variance_read_readvariableop/
+savev2_dense_965_kernel_read_readvariableop-
)savev2_dense_965_bias_read_readvariableop<
8savev2_batch_normalization_869_gamma_read_readvariableop;
7savev2_batch_normalization_869_beta_read_readvariableopB
>savev2_batch_normalization_869_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_869_moving_variance_read_readvariableop/
+savev2_dense_966_kernel_read_readvariableop-
)savev2_dense_966_bias_read_readvariableop<
8savev2_batch_normalization_870_gamma_read_readvariableop;
7savev2_batch_normalization_870_beta_read_readvariableopB
>savev2_batch_normalization_870_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_870_moving_variance_read_readvariableop/
+savev2_dense_967_kernel_read_readvariableop-
)savev2_dense_967_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_961_kernel_m_read_readvariableop4
0savev2_adam_dense_961_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_865_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_865_beta_m_read_readvariableop6
2savev2_adam_dense_962_kernel_m_read_readvariableop4
0savev2_adam_dense_962_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_866_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_866_beta_m_read_readvariableop6
2savev2_adam_dense_963_kernel_m_read_readvariableop4
0savev2_adam_dense_963_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_867_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_867_beta_m_read_readvariableop6
2savev2_adam_dense_964_kernel_m_read_readvariableop4
0savev2_adam_dense_964_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_868_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_868_beta_m_read_readvariableop6
2savev2_adam_dense_965_kernel_m_read_readvariableop4
0savev2_adam_dense_965_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_869_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_869_beta_m_read_readvariableop6
2savev2_adam_dense_966_kernel_m_read_readvariableop4
0savev2_adam_dense_966_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_870_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_870_beta_m_read_readvariableop6
2savev2_adam_dense_967_kernel_m_read_readvariableop4
0savev2_adam_dense_967_bias_m_read_readvariableop6
2savev2_adam_dense_961_kernel_v_read_readvariableop4
0savev2_adam_dense_961_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_865_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_865_beta_v_read_readvariableop6
2savev2_adam_dense_962_kernel_v_read_readvariableop4
0savev2_adam_dense_962_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_866_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_866_beta_v_read_readvariableop6
2savev2_adam_dense_963_kernel_v_read_readvariableop4
0savev2_adam_dense_963_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_867_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_867_beta_v_read_readvariableop6
2savev2_adam_dense_964_kernel_v_read_readvariableop4
0savev2_adam_dense_964_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_868_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_868_beta_v_read_readvariableop6
2savev2_adam_dense_965_kernel_v_read_readvariableop4
0savev2_adam_dense_965_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_869_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_869_beta_v_read_readvariableop6
2savev2_adam_dense_966_kernel_v_read_readvariableop4
0savev2_adam_dense_966_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_870_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_870_beta_v_read_readvariableop6
2savev2_adam_dense_967_kernel_v_read_readvariableop4
0savev2_adam_dense_967_bias_v_read_readvariableop
savev2_const_2

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?7
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*?6
value?6B?6dB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*?
value?B?dB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?,
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_961_kernel_read_readvariableop)savev2_dense_961_bias_read_readvariableop8savev2_batch_normalization_865_gamma_read_readvariableop7savev2_batch_normalization_865_beta_read_readvariableop>savev2_batch_normalization_865_moving_mean_read_readvariableopBsavev2_batch_normalization_865_moving_variance_read_readvariableop+savev2_dense_962_kernel_read_readvariableop)savev2_dense_962_bias_read_readvariableop8savev2_batch_normalization_866_gamma_read_readvariableop7savev2_batch_normalization_866_beta_read_readvariableop>savev2_batch_normalization_866_moving_mean_read_readvariableopBsavev2_batch_normalization_866_moving_variance_read_readvariableop+savev2_dense_963_kernel_read_readvariableop)savev2_dense_963_bias_read_readvariableop8savev2_batch_normalization_867_gamma_read_readvariableop7savev2_batch_normalization_867_beta_read_readvariableop>savev2_batch_normalization_867_moving_mean_read_readvariableopBsavev2_batch_normalization_867_moving_variance_read_readvariableop+savev2_dense_964_kernel_read_readvariableop)savev2_dense_964_bias_read_readvariableop8savev2_batch_normalization_868_gamma_read_readvariableop7savev2_batch_normalization_868_beta_read_readvariableop>savev2_batch_normalization_868_moving_mean_read_readvariableopBsavev2_batch_normalization_868_moving_variance_read_readvariableop+savev2_dense_965_kernel_read_readvariableop)savev2_dense_965_bias_read_readvariableop8savev2_batch_normalization_869_gamma_read_readvariableop7savev2_batch_normalization_869_beta_read_readvariableop>savev2_batch_normalization_869_moving_mean_read_readvariableopBsavev2_batch_normalization_869_moving_variance_read_readvariableop+savev2_dense_966_kernel_read_readvariableop)savev2_dense_966_bias_read_readvariableop8savev2_batch_normalization_870_gamma_read_readvariableop7savev2_batch_normalization_870_beta_read_readvariableop>savev2_batch_normalization_870_moving_mean_read_readvariableopBsavev2_batch_normalization_870_moving_variance_read_readvariableop+savev2_dense_967_kernel_read_readvariableop)savev2_dense_967_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_961_kernel_m_read_readvariableop0savev2_adam_dense_961_bias_m_read_readvariableop?savev2_adam_batch_normalization_865_gamma_m_read_readvariableop>savev2_adam_batch_normalization_865_beta_m_read_readvariableop2savev2_adam_dense_962_kernel_m_read_readvariableop0savev2_adam_dense_962_bias_m_read_readvariableop?savev2_adam_batch_normalization_866_gamma_m_read_readvariableop>savev2_adam_batch_normalization_866_beta_m_read_readvariableop2savev2_adam_dense_963_kernel_m_read_readvariableop0savev2_adam_dense_963_bias_m_read_readvariableop?savev2_adam_batch_normalization_867_gamma_m_read_readvariableop>savev2_adam_batch_normalization_867_beta_m_read_readvariableop2savev2_adam_dense_964_kernel_m_read_readvariableop0savev2_adam_dense_964_bias_m_read_readvariableop?savev2_adam_batch_normalization_868_gamma_m_read_readvariableop>savev2_adam_batch_normalization_868_beta_m_read_readvariableop2savev2_adam_dense_965_kernel_m_read_readvariableop0savev2_adam_dense_965_bias_m_read_readvariableop?savev2_adam_batch_normalization_869_gamma_m_read_readvariableop>savev2_adam_batch_normalization_869_beta_m_read_readvariableop2savev2_adam_dense_966_kernel_m_read_readvariableop0savev2_adam_dense_966_bias_m_read_readvariableop?savev2_adam_batch_normalization_870_gamma_m_read_readvariableop>savev2_adam_batch_normalization_870_beta_m_read_readvariableop2savev2_adam_dense_967_kernel_m_read_readvariableop0savev2_adam_dense_967_bias_m_read_readvariableop2savev2_adam_dense_961_kernel_v_read_readvariableop0savev2_adam_dense_961_bias_v_read_readvariableop?savev2_adam_batch_normalization_865_gamma_v_read_readvariableop>savev2_adam_batch_normalization_865_beta_v_read_readvariableop2savev2_adam_dense_962_kernel_v_read_readvariableop0savev2_adam_dense_962_bias_v_read_readvariableop?savev2_adam_batch_normalization_866_gamma_v_read_readvariableop>savev2_adam_batch_normalization_866_beta_v_read_readvariableop2savev2_adam_dense_963_kernel_v_read_readvariableop0savev2_adam_dense_963_bias_v_read_readvariableop?savev2_adam_batch_normalization_867_gamma_v_read_readvariableop>savev2_adam_batch_normalization_867_beta_v_read_readvariableop2savev2_adam_dense_964_kernel_v_read_readvariableop0savev2_adam_dense_964_bias_v_read_readvariableop?savev2_adam_batch_normalization_868_gamma_v_read_readvariableop>savev2_adam_batch_normalization_868_beta_v_read_readvariableop2savev2_adam_dense_965_kernel_v_read_readvariableop0savev2_adam_dense_965_bias_v_read_readvariableop?savev2_adam_batch_normalization_869_gamma_v_read_readvariableop>savev2_adam_batch_normalization_869_beta_v_read_readvariableop2savev2_adam_dense_966_kernel_v_read_readvariableop0savev2_adam_dense_966_bias_v_read_readvariableop?savev2_adam_batch_normalization_870_gamma_v_read_readvariableop>savev2_adam_batch_normalization_870_beta_v_read_readvariableop2savev2_adam_dense_967_kernel_v_read_readvariableop0savev2_adam_dense_967_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *r
dtypesh
f2d		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: :m:m:m:m:m:m:mm:m:m:m:m:m:mm:m:m:m:m:m:m.:.:.:.:.:.:..:.:.:.:.:.:.]:]:]:]:]:]:]:: : : : : : :m:m:m:m:mm:m:m:m:mm:m:m:m:m.:.:.:.:..:.:.:.:.]:]:]:]:]::m:m:m:m:mm:m:m:m:mm:m:m:m:m.:.:.:.:..:.:.:.:.]:]:]:]:]:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :$ 

_output_shapes

:m: 

_output_shapes
:m: 

_output_shapes
:m: 

_output_shapes
:m: 

_output_shapes
:m: 	

_output_shapes
:m:$
 

_output_shapes

:mm: 

_output_shapes
:m: 

_output_shapes
:m: 

_output_shapes
:m: 

_output_shapes
:m: 

_output_shapes
:m:$ 

_output_shapes

:mm: 

_output_shapes
:m: 

_output_shapes
:m: 

_output_shapes
:m: 

_output_shapes
:m: 

_output_shapes
:m:$ 

_output_shapes

:m.: 

_output_shapes
:.: 

_output_shapes
:.: 

_output_shapes
:.: 

_output_shapes
:.: 

_output_shapes
:.:$ 

_output_shapes

:..: 

_output_shapes
:.: 

_output_shapes
:.: 

_output_shapes
:.:  

_output_shapes
:.: !

_output_shapes
:.:$" 

_output_shapes

:.]: #

_output_shapes
:]: $

_output_shapes
:]: %

_output_shapes
:]: &

_output_shapes
:]: '

_output_shapes
:]:$( 

_output_shapes

:]: )

_output_shapes
::*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :$0 

_output_shapes

:m: 1

_output_shapes
:m: 2

_output_shapes
:m: 3

_output_shapes
:m:$4 

_output_shapes

:mm: 5

_output_shapes
:m: 6

_output_shapes
:m: 7

_output_shapes
:m:$8 

_output_shapes

:mm: 9

_output_shapes
:m: :

_output_shapes
:m: ;

_output_shapes
:m:$< 

_output_shapes

:m.: =

_output_shapes
:.: >

_output_shapes
:.: ?

_output_shapes
:.:$@ 

_output_shapes

:..: A

_output_shapes
:.: B

_output_shapes
:.: C

_output_shapes
:.:$D 

_output_shapes

:.]: E

_output_shapes
:]: F

_output_shapes
:]: G

_output_shapes
:]:$H 

_output_shapes

:]: I

_output_shapes
::$J 

_output_shapes

:m: K

_output_shapes
:m: L

_output_shapes
:m: M

_output_shapes
:m:$N 

_output_shapes

:mm: O

_output_shapes
:m: P

_output_shapes
:m: Q

_output_shapes
:m:$R 

_output_shapes

:mm: S

_output_shapes
:m: T

_output_shapes
:m: U

_output_shapes
:m:$V 

_output_shapes

:m.: W

_output_shapes
:.: X

_output_shapes
:.: Y

_output_shapes
:.:$Z 

_output_shapes

:..: [

_output_shapes
:.: \

_output_shapes
:.: ]

_output_shapes
:.:$^ 

_output_shapes

:.]: _

_output_shapes
:]: `

_output_shapes
:]: a

_output_shapes
:]:$b 

_output_shapes

:]: c

_output_shapes
::d

_output_shapes
: 
?
g
K__inference_leaky_re_lu_866_layer_call_and_return_conditional_losses_864770

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????m*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????m"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????m:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
*__inference_dense_965_layer_call_fn_865027

inputs
unknown:..
	unknown_0:.
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_965_layer_call_and_return_conditional_losses_862735o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????.`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????.: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_870_layer_call_and_return_conditional_losses_862793

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????]*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????]"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????]:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_864847

inputs/
!batchnorm_readvariableop_resource:m3
%batchnorm_mul_readvariableop_resource:m1
#batchnorm_readvariableop_1_resource:m1
#batchnorm_readvariableop_2_resource:m
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:mP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:mc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????mz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:m*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:mz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:m*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:mr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????mb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
E__inference_dense_965_layer_call_and_return_conditional_losses_865043

inputs0
matmul_readvariableop_resource:..-
biasadd_readvariableop_resource:.
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_965/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:..*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:.*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
2dense_965/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:..*
dtype0?
#dense_965/kernel/Regularizer/SquareSquare:dense_965/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:..s
"dense_965/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_965/kernel/Regularizer/SumSum'dense_965/kernel/Regularizer/Square:y:0+dense_965/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_965/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_965/kernel/Regularizer/mulMul+dense_965/kernel/Regularizer/mul/x:output:0)dense_965/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????.?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_965/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_965/kernel/Regularizer/Square/ReadVariableOp2dense_965/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?	
.__inference_sequential_96_layer_call_fn_862931
normalization_96_input
unknown
	unknown_0
	unknown_1:m
	unknown_2:m
	unknown_3:m
	unknown_4:m
	unknown_5:m
	unknown_6:m
	unknown_7:mm
	unknown_8:m
	unknown_9:m

unknown_10:m

unknown_11:m

unknown_12:m

unknown_13:mm

unknown_14:m

unknown_15:m

unknown_16:m

unknown_17:m

unknown_18:m

unknown_19:m.

unknown_20:.

unknown_21:.

unknown_22:.

unknown_23:.

unknown_24:.

unknown_25:..

unknown_26:.

unknown_27:.

unknown_28:.

unknown_29:.

unknown_30:.

unknown_31:.]

unknown_32:]

unknown_33:]

unknown_34:]

unknown_35:]

unknown_36:]

unknown_37:]

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallnormalization_96_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 !"#$%&'(*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_96_layer_call_and_return_conditional_losses_862848o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:?????????
0
_user_specified_namenormalization_96_input:$ 

_output_shapes

::$ 

_output_shapes

:
?%
?
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_862214

inputs5
'assignmovingavg_readvariableop_resource:m7
)assignmovingavg_1_readvariableop_resource:m3
%batchnorm_mul_readvariableop_resource:m/
!batchnorm_readvariableop_resource:m
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:m?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????ml
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:m*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:m*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:m*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:mx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:m?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:m*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:m~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:m?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:mP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:mc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????mh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:mv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:mr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????mb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?	
.__inference_sequential_96_layer_call_fn_863434
normalization_96_input
unknown
	unknown_0
	unknown_1:m
	unknown_2:m
	unknown_3:m
	unknown_4:m
	unknown_5:m
	unknown_6:m
	unknown_7:mm
	unknown_8:m
	unknown_9:m

unknown_10:m

unknown_11:m

unknown_12:m

unknown_13:mm

unknown_14:m

unknown_15:m

unknown_16:m

unknown_17:m

unknown_18:m

unknown_19:m.

unknown_20:.

unknown_21:.

unknown_22:.

unknown_23:.

unknown_24:.

unknown_25:..

unknown_26:.

unknown_27:.

unknown_28:.

unknown_29:.

unknown_30:.

unknown_31:.]

unknown_32:]

unknown_33:]

unknown_34:]

unknown_35:]

unknown_36:]

unknown_37:]

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallnormalization_96_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !"%&'(*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_96_layer_call_and_return_conditional_losses_863266o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:?????????
0
_user_specified_namenormalization_96_input:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_865210

inputs/
!batchnorm_readvariableop_resource:]3
%batchnorm_mul_readvariableop_resource:]1
#batchnorm_readvariableop_1_resource:]1
#batchnorm_readvariableop_2_resource:]
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:]*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:]P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:]~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:]*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:]c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????]z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:]*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:]z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:]*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:]r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????]b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????]?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????]: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_865_layer_call_fn_864585

inputs
unknown:m
	unknown_0:m
	unknown_1:m
	unknown_2:m
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_862132o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????m`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_866_layer_call_fn_864693

inputs
unknown:m
	unknown_0:m
	unknown_1:m
	unknown_2:m
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_862167o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????m`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_865284M
;dense_961_kernel_regularizer_square_readvariableop_resource:m
identity??2dense_961/kernel/Regularizer/Square/ReadVariableOp?
2dense_961/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_961_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:m*
dtype0?
#dense_961/kernel/Regularizer/SquareSquare:dense_961/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ms
"dense_961/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_961/kernel/Regularizer/SumSum'dense_961/kernel/Regularizer/Square:y:0+dense_961/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_961/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_961/kernel/Regularizer/mulMul+dense_961/kernel/Regularizer/mul/x:output:0)dense_961/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_961/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_961/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_961/kernel/Regularizer/Square/ReadVariableOp2dense_961/kernel/Regularizer/Square/ReadVariableOp
?
?
8__inference_batch_normalization_870_layer_call_fn_865190

inputs
unknown:]
	unknown_0:]
	unknown_1:]
	unknown_2:]
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_862542o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????]`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????]: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
.__inference_sequential_96_layer_call_fn_863928

inputs
unknown
	unknown_0
	unknown_1:m
	unknown_2:m
	unknown_3:m
	unknown_4:m
	unknown_5:m
	unknown_6:m
	unknown_7:mm
	unknown_8:m
	unknown_9:m

unknown_10:m

unknown_11:m

unknown_12:m

unknown_13:mm

unknown_14:m

unknown_15:m

unknown_16:m

unknown_17:m

unknown_18:m

unknown_19:m.

unknown_20:.

unknown_21:.

unknown_22:.

unknown_23:.

unknown_24:.

unknown_25:..

unknown_26:.

unknown_27:.

unknown_28:.

unknown_29:.

unknown_30:.

unknown_31:.]

unknown_32:]

unknown_33:]

unknown_34:]

unknown_35:]

unknown_36:]

unknown_37:]

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !"%&'(*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_96_layer_call_and_return_conditional_losses_863266o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
?
g
K__inference_leaky_re_lu_869_layer_call_and_return_conditional_losses_862755

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????.*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????.:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_862413

inputs/
!batchnorm_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:.1
#batchnorm_readvariableop_1_resource:.1
#batchnorm_readvariableop_2_resource:.
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:.P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:.~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:.*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:.z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:.*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:.r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????.?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
E__inference_dense_964_layer_call_and_return_conditional_losses_862697

inputs0
matmul_readvariableop_resource:m.-
biasadd_readvariableop_resource:.
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_964/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m.*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:.*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
2dense_964/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m.*
dtype0?
#dense_964/kernel/Regularizer/SquareSquare:dense_964/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m.s
"dense_964/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_964/kernel/Regularizer/SumSum'dense_964/kernel/Regularizer/Square:y:0+dense_964/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_964/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_964/kernel/Regularizer/mulMul+dense_964/kernel/Regularizer/mul/x:output:0)dense_964/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????.?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_964/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????m: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_964/kernel/Regularizer/Square/ReadVariableOp2dense_964/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
*__inference_dense_962_layer_call_fn_864664

inputs
unknown:mm
	unknown_0:m
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_962_layer_call_and_return_conditional_losses_862621o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????m`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????m: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_865123

inputs5
'assignmovingavg_readvariableop_resource:.7
)assignmovingavg_1_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:./
!batchnorm_readvariableop_resource:.
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:.?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????.l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:.*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:.x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:.?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:.*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:.~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:.?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:.P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:.~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:.v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:.r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????.?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
E__inference_dense_962_layer_call_and_return_conditional_losses_864680

inputs0
matmul_readvariableop_resource:mm-
biasadd_readvariableop_resource:m
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_962/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:mm*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
2dense_962/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
#dense_962/kernel/Regularizer/SquareSquare:dense_962/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_962/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_962/kernel/Regularizer/SumSum'dense_962/kernel/Regularizer/Square:y:0+dense_962/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_962/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_962/kernel/Regularizer/mulMul+dense_962/kernel/Regularizer/mul/x:output:0)dense_962/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_962/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????m: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_962/kernel/Regularizer/Square/ReadVariableOp2dense_962/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_864481
normalization_96_input
unknown
	unknown_0
	unknown_1:m
	unknown_2:m
	unknown_3:m
	unknown_4:m
	unknown_5:m
	unknown_6:m
	unknown_7:mm
	unknown_8:m
	unknown_9:m

unknown_10:m

unknown_11:m

unknown_12:m

unknown_13:mm

unknown_14:m

unknown_15:m

unknown_16:m

unknown_17:m

unknown_18:m

unknown_19:m.

unknown_20:.

unknown_21:.

unknown_22:.

unknown_23:.

unknown_24:.

unknown_25:..

unknown_26:.

unknown_27:.

unknown_28:.

unknown_29:.

unknown_30:.

unknown_31:.]

unknown_32:]

unknown_33:]

unknown_34:]

unknown_35:]

unknown_36:]

unknown_37:]

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallnormalization_96_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 !"#$%&'(*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_862061o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:?????????
0
_user_specified_namenormalization_96_input:$ 

_output_shapes

::$ 

_output_shapes

:
?	
?
E__inference_dense_967_layer_call_and_return_conditional_losses_865273

inputs0
matmul_readvariableop_resource:]-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_862085

inputs/
!batchnorm_readvariableop_resource:m3
%batchnorm_mul_readvariableop_resource:m1
#batchnorm_readvariableop_1_resource:m1
#batchnorm_readvariableop_2_resource:m
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:mP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:mc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????mz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:m*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:mz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:m*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:mr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????mb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
?
E__inference_dense_966_layer_call_and_return_conditional_losses_862773

inputs0
matmul_readvariableop_resource:.]-
biasadd_readvariableop_resource:]
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_966/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:.]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:]*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]?
2dense_966/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:.]*
dtype0?
#dense_966/kernel/Regularizer/SquareSquare:dense_966/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:.]s
"dense_966/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_966/kernel/Regularizer/SumSum'dense_966/kernel/Regularizer/Square:y:0+dense_966/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_966/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?<?
 dense_966/kernel/Regularizer/mulMul+dense_966/kernel/Regularizer/mul/x:output:0)dense_966/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????]?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_966/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_966/kernel/Regularizer/Square/ReadVariableOp2dense_966/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
E__inference_dense_961_layer_call_and_return_conditional_losses_862583

inputs0
matmul_readvariableop_resource:m-
biasadd_readvariableop_resource:m
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_961/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
2dense_961/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#dense_961/kernel/Regularizer/SquareSquare:dense_961/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ms
"dense_961/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_961/kernel/Regularizer/SumSum'dense_961/kernel/Regularizer/Square:y:0+dense_961/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_961/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_961/kernel/Regularizer/mulMul+dense_961/kernel/Regularizer/mul/x:output:0)dense_961/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_961/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_961/kernel/Regularizer/Square/ReadVariableOp2dense_961/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ܙ
?
I__inference_sequential_96_layer_call_and_return_conditional_losses_863266

inputs
normalization_96_sub_y
normalization_96_sqrt_x"
dense_961_863134:m
dense_961_863136:m,
batch_normalization_865_863139:m,
batch_normalization_865_863141:m,
batch_normalization_865_863143:m,
batch_normalization_865_863145:m"
dense_962_863149:mm
dense_962_863151:m,
batch_normalization_866_863154:m,
batch_normalization_866_863156:m,
batch_normalization_866_863158:m,
batch_normalization_866_863160:m"
dense_963_863164:mm
dense_963_863166:m,
batch_normalization_867_863169:m,
batch_normalization_867_863171:m,
batch_normalization_867_863173:m,
batch_normalization_867_863175:m"
dense_964_863179:m.
dense_964_863181:.,
batch_normalization_868_863184:.,
batch_normalization_868_863186:.,
batch_normalization_868_863188:.,
batch_normalization_868_863190:."
dense_965_863194:..
dense_965_863196:.,
batch_normalization_869_863199:.,
batch_normalization_869_863201:.,
batch_normalization_869_863203:.,
batch_normalization_869_863205:."
dense_966_863209:.]
dense_966_863211:],
batch_normalization_870_863214:],
batch_normalization_870_863216:],
batch_normalization_870_863218:],
batch_normalization_870_863220:]"
dense_967_863224:]
dense_967_863226:
identity??/batch_normalization_865/StatefulPartitionedCall?/batch_normalization_866/StatefulPartitionedCall?/batch_normalization_867/StatefulPartitionedCall?/batch_normalization_868/StatefulPartitionedCall?/batch_normalization_869/StatefulPartitionedCall?/batch_normalization_870/StatefulPartitionedCall?!dense_961/StatefulPartitionedCall?2dense_961/kernel/Regularizer/Square/ReadVariableOp?!dense_962/StatefulPartitionedCall?2dense_962/kernel/Regularizer/Square/ReadVariableOp?!dense_963/StatefulPartitionedCall?2dense_963/kernel/Regularizer/Square/ReadVariableOp?!dense_964/StatefulPartitionedCall?2dense_964/kernel/Regularizer/Square/ReadVariableOp?!dense_965/StatefulPartitionedCall?2dense_965/kernel/Regularizer/Square/ReadVariableOp?!dense_966/StatefulPartitionedCall?2dense_966/kernel/Regularizer/Square/ReadVariableOp?!dense_967/StatefulPartitionedCallm
normalization_96/subSubinputsnormalization_96_sub_y*
T0*'
_output_shapes
:?????????_
normalization_96/SqrtSqrtnormalization_96_sqrt_x*
T0*
_output_shapes

:_
normalization_96/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_96/MaximumMaximumnormalization_96/Sqrt:y:0#normalization_96/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_96/truedivRealDivnormalization_96/sub:z:0normalization_96/Maximum:z:0*
T0*'
_output_shapes
:??????????
!dense_961/StatefulPartitionedCallStatefulPartitionedCallnormalization_96/truediv:z:0dense_961_863134dense_961_863136*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_961_layer_call_and_return_conditional_losses_862583?
/batch_normalization_865/StatefulPartitionedCallStatefulPartitionedCall*dense_961/StatefulPartitionedCall:output:0batch_normalization_865_863139batch_normalization_865_863141batch_normalization_865_863143batch_normalization_865_863145*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_862132?
leaky_re_lu_865/PartitionedCallPartitionedCall8batch_normalization_865/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_862603?
!dense_962/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_865/PartitionedCall:output:0dense_962_863149dense_962_863151*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_962_layer_call_and_return_conditional_losses_862621?
/batch_normalization_866/StatefulPartitionedCallStatefulPartitionedCall*dense_962/StatefulPartitionedCall:output:0batch_normalization_866_863154batch_normalization_866_863156batch_normalization_866_863158batch_normalization_866_863160*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_862214?
leaky_re_lu_866/PartitionedCallPartitionedCall8batch_normalization_866/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_866_layer_call_and_return_conditional_losses_862641?
!dense_963/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_866/PartitionedCall:output:0dense_963_863164dense_963_863166*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_963_layer_call_and_return_conditional_losses_862659?
/batch_normalization_867/StatefulPartitionedCallStatefulPartitionedCall*dense_963/StatefulPartitionedCall:output:0batch_normalization_867_863169batch_normalization_867_863171batch_normalization_867_863173batch_normalization_867_863175*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_862296?
leaky_re_lu_867/PartitionedCallPartitionedCall8batch_normalization_867/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_867_layer_call_and_return_conditional_losses_862679?
!dense_964/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_867/PartitionedCall:output:0dense_964_863179dense_964_863181*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_964_layer_call_and_return_conditional_losses_862697?
/batch_normalization_868/StatefulPartitionedCallStatefulPartitionedCall*dense_964/StatefulPartitionedCall:output:0batch_normalization_868_863184batch_normalization_868_863186batch_normalization_868_863188batch_normalization_868_863190*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_862378?
leaky_re_lu_868/PartitionedCallPartitionedCall8batch_normalization_868/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_868_layer_call_and_return_conditional_losses_862717?
!dense_965/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_868/PartitionedCall:output:0dense_965_863194dense_965_863196*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_965_layer_call_and_return_conditional_losses_862735?
/batch_normalization_869/StatefulPartitionedCallStatefulPartitionedCall*dense_965/StatefulPartitionedCall:output:0batch_normalization_869_863199batch_normalization_869_863201batch_normalization_869_863203batch_normalization_869_863205*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_862460?
leaky_re_lu_869/PartitionedCallPartitionedCall8batch_normalization_869/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_869_layer_call_and_return_conditional_losses_862755?
!dense_966/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_869/PartitionedCall:output:0dense_966_863209dense_966_863211*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_966_layer_call_and_return_conditional_losses_862773?
/batch_normalization_870/StatefulPartitionedCallStatefulPartitionedCall*dense_966/StatefulPartitionedCall:output:0batch_normalization_870_863214batch_normalization_870_863216batch_normalization_870_863218batch_normalization_870_863220*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_862542?
leaky_re_lu_870/PartitionedCallPartitionedCall8batch_normalization_870/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_870_layer_call_and_return_conditional_losses_862793?
!dense_967/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_870/PartitionedCall:output:0dense_967_863224dense_967_863226*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_967_layer_call_and_return_conditional_losses_862805?
2dense_961/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_961_863134*
_output_shapes

:m*
dtype0?
#dense_961/kernel/Regularizer/SquareSquare:dense_961/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ms
"dense_961/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_961/kernel/Regularizer/SumSum'dense_961/kernel/Regularizer/Square:y:0+dense_961/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_961/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_961/kernel/Regularizer/mulMul+dense_961/kernel/Regularizer/mul/x:output:0)dense_961/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_962/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_962_863149*
_output_shapes

:mm*
dtype0?
#dense_962/kernel/Regularizer/SquareSquare:dense_962/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_962/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_962/kernel/Regularizer/SumSum'dense_962/kernel/Regularizer/Square:y:0+dense_962/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_962/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_962/kernel/Regularizer/mulMul+dense_962/kernel/Regularizer/mul/x:output:0)dense_962/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_963/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_963_863164*
_output_shapes

:mm*
dtype0?
#dense_963/kernel/Regularizer/SquareSquare:dense_963/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_963/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_963/kernel/Regularizer/SumSum'dense_963/kernel/Regularizer/Square:y:0+dense_963/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_963/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_963/kernel/Regularizer/mulMul+dense_963/kernel/Regularizer/mul/x:output:0)dense_963/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_964/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_964_863179*
_output_shapes

:m.*
dtype0?
#dense_964/kernel/Regularizer/SquareSquare:dense_964/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m.s
"dense_964/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_964/kernel/Regularizer/SumSum'dense_964/kernel/Regularizer/Square:y:0+dense_964/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_964/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_964/kernel/Regularizer/mulMul+dense_964/kernel/Regularizer/mul/x:output:0)dense_964/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_965/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_965_863194*
_output_shapes

:..*
dtype0?
#dense_965/kernel/Regularizer/SquareSquare:dense_965/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:..s
"dense_965/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_965/kernel/Regularizer/SumSum'dense_965/kernel/Regularizer/Square:y:0+dense_965/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_965/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_965/kernel/Regularizer/mulMul+dense_965/kernel/Regularizer/mul/x:output:0)dense_965/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_966/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_966_863209*
_output_shapes

:.]*
dtype0?
#dense_966/kernel/Regularizer/SquareSquare:dense_966/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:.]s
"dense_966/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_966/kernel/Regularizer/SumSum'dense_966/kernel/Regularizer/Square:y:0+dense_966/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_966/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?<?
 dense_966/kernel/Regularizer/mulMul+dense_966/kernel/Regularizer/mul/x:output:0)dense_966/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_967/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp0^batch_normalization_865/StatefulPartitionedCall0^batch_normalization_866/StatefulPartitionedCall0^batch_normalization_867/StatefulPartitionedCall0^batch_normalization_868/StatefulPartitionedCall0^batch_normalization_869/StatefulPartitionedCall0^batch_normalization_870/StatefulPartitionedCall"^dense_961/StatefulPartitionedCall3^dense_961/kernel/Regularizer/Square/ReadVariableOp"^dense_962/StatefulPartitionedCall3^dense_962/kernel/Regularizer/Square/ReadVariableOp"^dense_963/StatefulPartitionedCall3^dense_963/kernel/Regularizer/Square/ReadVariableOp"^dense_964/StatefulPartitionedCall3^dense_964/kernel/Regularizer/Square/ReadVariableOp"^dense_965/StatefulPartitionedCall3^dense_965/kernel/Regularizer/Square/ReadVariableOp"^dense_966/StatefulPartitionedCall3^dense_966/kernel/Regularizer/Square/ReadVariableOp"^dense_967/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_865/StatefulPartitionedCall/batch_normalization_865/StatefulPartitionedCall2b
/batch_normalization_866/StatefulPartitionedCall/batch_normalization_866/StatefulPartitionedCall2b
/batch_normalization_867/StatefulPartitionedCall/batch_normalization_867/StatefulPartitionedCall2b
/batch_normalization_868/StatefulPartitionedCall/batch_normalization_868/StatefulPartitionedCall2b
/batch_normalization_869/StatefulPartitionedCall/batch_normalization_869/StatefulPartitionedCall2b
/batch_normalization_870/StatefulPartitionedCall/batch_normalization_870/StatefulPartitionedCall2F
!dense_961/StatefulPartitionedCall!dense_961/StatefulPartitionedCall2h
2dense_961/kernel/Regularizer/Square/ReadVariableOp2dense_961/kernel/Regularizer/Square/ReadVariableOp2F
!dense_962/StatefulPartitionedCall!dense_962/StatefulPartitionedCall2h
2dense_962/kernel/Regularizer/Square/ReadVariableOp2dense_962/kernel/Regularizer/Square/ReadVariableOp2F
!dense_963/StatefulPartitionedCall!dense_963/StatefulPartitionedCall2h
2dense_963/kernel/Regularizer/Square/ReadVariableOp2dense_963/kernel/Regularizer/Square/ReadVariableOp2F
!dense_964/StatefulPartitionedCall!dense_964/StatefulPartitionedCall2h
2dense_964/kernel/Regularizer/Square/ReadVariableOp2dense_964/kernel/Regularizer/Square/ReadVariableOp2F
!dense_965/StatefulPartitionedCall!dense_965/StatefulPartitionedCall2h
2dense_965/kernel/Regularizer/Square/ReadVariableOp2dense_965/kernel/Regularizer/Square/ReadVariableOp2F
!dense_966/StatefulPartitionedCall!dense_966/StatefulPartitionedCall2h
2dense_966/kernel/Regularizer/Square/ReadVariableOp2dense_966/kernel/Regularizer/Square/ReadVariableOp2F
!dense_967/StatefulPartitionedCall!dense_967/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
?	
?
E__inference_dense_967_layer_call_and_return_conditional_losses_862805

inputs0
matmul_readvariableop_resource:]-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
E__inference_dense_963_layer_call_and_return_conditional_losses_864801

inputs0
matmul_readvariableop_resource:mm-
biasadd_readvariableop_resource:m
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_963/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:mm*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
2dense_963/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
#dense_963/kernel/Regularizer/SquareSquare:dense_963/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_963/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_963/kernel/Regularizer/SumSum'dense_963/kernel/Regularizer/Square:y:0+dense_963/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_963/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_963/kernel/Regularizer/mulMul+dense_963/kernel/Regularizer/mul/x:output:0)dense_963/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_963/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????m: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_963/kernel/Regularizer/Square/ReadVariableOp2dense_963/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_865244

inputs5
'assignmovingavg_readvariableop_resource:]7
)assignmovingavg_1_readvariableop_resource:]3
%batchnorm_mul_readvariableop_resource:]/
!batchnorm_readvariableop_resource:]
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:]*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:]?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????]l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:]*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:]*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:]*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:]*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:]x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:]?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:]*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:]~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:]?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:]P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:]~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:]*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:]c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????]h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:]v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:]*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:]r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????]b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????]?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????]: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_868_layer_call_fn_864948

inputs
unknown:.
	unknown_0:.
	unknown_1:.
	unknown_2:.
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_862378o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????.`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_867_layer_call_and_return_conditional_losses_864891

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????m*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????m"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????m:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
??
?%
I__inference_sequential_96_layer_call_and_return_conditional_losses_864119

inputs
normalization_96_sub_y
normalization_96_sqrt_x:
(dense_961_matmul_readvariableop_resource:m7
)dense_961_biasadd_readvariableop_resource:mG
9batch_normalization_865_batchnorm_readvariableop_resource:mK
=batch_normalization_865_batchnorm_mul_readvariableop_resource:mI
;batch_normalization_865_batchnorm_readvariableop_1_resource:mI
;batch_normalization_865_batchnorm_readvariableop_2_resource:m:
(dense_962_matmul_readvariableop_resource:mm7
)dense_962_biasadd_readvariableop_resource:mG
9batch_normalization_866_batchnorm_readvariableop_resource:mK
=batch_normalization_866_batchnorm_mul_readvariableop_resource:mI
;batch_normalization_866_batchnorm_readvariableop_1_resource:mI
;batch_normalization_866_batchnorm_readvariableop_2_resource:m:
(dense_963_matmul_readvariableop_resource:mm7
)dense_963_biasadd_readvariableop_resource:mG
9batch_normalization_867_batchnorm_readvariableop_resource:mK
=batch_normalization_867_batchnorm_mul_readvariableop_resource:mI
;batch_normalization_867_batchnorm_readvariableop_1_resource:mI
;batch_normalization_867_batchnorm_readvariableop_2_resource:m:
(dense_964_matmul_readvariableop_resource:m.7
)dense_964_biasadd_readvariableop_resource:.G
9batch_normalization_868_batchnorm_readvariableop_resource:.K
=batch_normalization_868_batchnorm_mul_readvariableop_resource:.I
;batch_normalization_868_batchnorm_readvariableop_1_resource:.I
;batch_normalization_868_batchnorm_readvariableop_2_resource:.:
(dense_965_matmul_readvariableop_resource:..7
)dense_965_biasadd_readvariableop_resource:.G
9batch_normalization_869_batchnorm_readvariableop_resource:.K
=batch_normalization_869_batchnorm_mul_readvariableop_resource:.I
;batch_normalization_869_batchnorm_readvariableop_1_resource:.I
;batch_normalization_869_batchnorm_readvariableop_2_resource:.:
(dense_966_matmul_readvariableop_resource:.]7
)dense_966_biasadd_readvariableop_resource:]G
9batch_normalization_870_batchnorm_readvariableop_resource:]K
=batch_normalization_870_batchnorm_mul_readvariableop_resource:]I
;batch_normalization_870_batchnorm_readvariableop_1_resource:]I
;batch_normalization_870_batchnorm_readvariableop_2_resource:]:
(dense_967_matmul_readvariableop_resource:]7
)dense_967_biasadd_readvariableop_resource:
identity??0batch_normalization_865/batchnorm/ReadVariableOp?2batch_normalization_865/batchnorm/ReadVariableOp_1?2batch_normalization_865/batchnorm/ReadVariableOp_2?4batch_normalization_865/batchnorm/mul/ReadVariableOp?0batch_normalization_866/batchnorm/ReadVariableOp?2batch_normalization_866/batchnorm/ReadVariableOp_1?2batch_normalization_866/batchnorm/ReadVariableOp_2?4batch_normalization_866/batchnorm/mul/ReadVariableOp?0batch_normalization_867/batchnorm/ReadVariableOp?2batch_normalization_867/batchnorm/ReadVariableOp_1?2batch_normalization_867/batchnorm/ReadVariableOp_2?4batch_normalization_867/batchnorm/mul/ReadVariableOp?0batch_normalization_868/batchnorm/ReadVariableOp?2batch_normalization_868/batchnorm/ReadVariableOp_1?2batch_normalization_868/batchnorm/ReadVariableOp_2?4batch_normalization_868/batchnorm/mul/ReadVariableOp?0batch_normalization_869/batchnorm/ReadVariableOp?2batch_normalization_869/batchnorm/ReadVariableOp_1?2batch_normalization_869/batchnorm/ReadVariableOp_2?4batch_normalization_869/batchnorm/mul/ReadVariableOp?0batch_normalization_870/batchnorm/ReadVariableOp?2batch_normalization_870/batchnorm/ReadVariableOp_1?2batch_normalization_870/batchnorm/ReadVariableOp_2?4batch_normalization_870/batchnorm/mul/ReadVariableOp? dense_961/BiasAdd/ReadVariableOp?dense_961/MatMul/ReadVariableOp?2dense_961/kernel/Regularizer/Square/ReadVariableOp? dense_962/BiasAdd/ReadVariableOp?dense_962/MatMul/ReadVariableOp?2dense_962/kernel/Regularizer/Square/ReadVariableOp? dense_963/BiasAdd/ReadVariableOp?dense_963/MatMul/ReadVariableOp?2dense_963/kernel/Regularizer/Square/ReadVariableOp? dense_964/BiasAdd/ReadVariableOp?dense_964/MatMul/ReadVariableOp?2dense_964/kernel/Regularizer/Square/ReadVariableOp? dense_965/BiasAdd/ReadVariableOp?dense_965/MatMul/ReadVariableOp?2dense_965/kernel/Regularizer/Square/ReadVariableOp? dense_966/BiasAdd/ReadVariableOp?dense_966/MatMul/ReadVariableOp?2dense_966/kernel/Regularizer/Square/ReadVariableOp? dense_967/BiasAdd/ReadVariableOp?dense_967/MatMul/ReadVariableOpm
normalization_96/subSubinputsnormalization_96_sub_y*
T0*'
_output_shapes
:?????????_
normalization_96/SqrtSqrtnormalization_96_sqrt_x*
T0*
_output_shapes

:_
normalization_96/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_96/MaximumMaximumnormalization_96/Sqrt:y:0#normalization_96/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_96/truedivRealDivnormalization_96/sub:z:0normalization_96/Maximum:z:0*
T0*'
_output_shapes
:??????????
dense_961/MatMul/ReadVariableOpReadVariableOp(dense_961_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
dense_961/MatMulMatMulnormalization_96/truediv:z:0'dense_961/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
 dense_961/BiasAdd/ReadVariableOpReadVariableOp)dense_961_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
dense_961/BiasAddBiasAdddense_961/MatMul:product:0(dense_961/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
0batch_normalization_865/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_865_batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0l
'batch_normalization_865/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_865/batchnorm/addAddV28batch_normalization_865/batchnorm/ReadVariableOp:value:00batch_normalization_865/batchnorm/add/y:output:0*
T0*
_output_shapes
:m?
'batch_normalization_865/batchnorm/RsqrtRsqrt)batch_normalization_865/batchnorm/add:z:0*
T0*
_output_shapes
:m?
4batch_normalization_865/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_865_batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0?
%batch_normalization_865/batchnorm/mulMul+batch_normalization_865/batchnorm/Rsqrt:y:0<batch_normalization_865/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:m?
'batch_normalization_865/batchnorm/mul_1Muldense_961/BiasAdd:output:0)batch_normalization_865/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????m?
2batch_normalization_865/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_865_batchnorm_readvariableop_1_resource*
_output_shapes
:m*
dtype0?
'batch_normalization_865/batchnorm/mul_2Mul:batch_normalization_865/batchnorm/ReadVariableOp_1:value:0)batch_normalization_865/batchnorm/mul:z:0*
T0*
_output_shapes
:m?
2batch_normalization_865/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_865_batchnorm_readvariableop_2_resource*
_output_shapes
:m*
dtype0?
%batch_normalization_865/batchnorm/subSub:batch_normalization_865/batchnorm/ReadVariableOp_2:value:0+batch_normalization_865/batchnorm/mul_2:z:0*
T0*
_output_shapes
:m?
'batch_normalization_865/batchnorm/add_1AddV2+batch_normalization_865/batchnorm/mul_1:z:0)batch_normalization_865/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????m?
leaky_re_lu_865/LeakyRelu	LeakyRelu+batch_normalization_865/batchnorm/add_1:z:0*'
_output_shapes
:?????????m*
alpha%???>?
dense_962/MatMul/ReadVariableOpReadVariableOp(dense_962_matmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
dense_962/MatMulMatMul'leaky_re_lu_865/LeakyRelu:activations:0'dense_962/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
 dense_962/BiasAdd/ReadVariableOpReadVariableOp)dense_962_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
dense_962/BiasAddBiasAdddense_962/MatMul:product:0(dense_962/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
0batch_normalization_866/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_866_batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0l
'batch_normalization_866/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_866/batchnorm/addAddV28batch_normalization_866/batchnorm/ReadVariableOp:value:00batch_normalization_866/batchnorm/add/y:output:0*
T0*
_output_shapes
:m?
'batch_normalization_866/batchnorm/RsqrtRsqrt)batch_normalization_866/batchnorm/add:z:0*
T0*
_output_shapes
:m?
4batch_normalization_866/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_866_batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0?
%batch_normalization_866/batchnorm/mulMul+batch_normalization_866/batchnorm/Rsqrt:y:0<batch_normalization_866/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:m?
'batch_normalization_866/batchnorm/mul_1Muldense_962/BiasAdd:output:0)batch_normalization_866/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????m?
2batch_normalization_866/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_866_batchnorm_readvariableop_1_resource*
_output_shapes
:m*
dtype0?
'batch_normalization_866/batchnorm/mul_2Mul:batch_normalization_866/batchnorm/ReadVariableOp_1:value:0)batch_normalization_866/batchnorm/mul:z:0*
T0*
_output_shapes
:m?
2batch_normalization_866/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_866_batchnorm_readvariableop_2_resource*
_output_shapes
:m*
dtype0?
%batch_normalization_866/batchnorm/subSub:batch_normalization_866/batchnorm/ReadVariableOp_2:value:0+batch_normalization_866/batchnorm/mul_2:z:0*
T0*
_output_shapes
:m?
'batch_normalization_866/batchnorm/add_1AddV2+batch_normalization_866/batchnorm/mul_1:z:0)batch_normalization_866/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????m?
leaky_re_lu_866/LeakyRelu	LeakyRelu+batch_normalization_866/batchnorm/add_1:z:0*'
_output_shapes
:?????????m*
alpha%???>?
dense_963/MatMul/ReadVariableOpReadVariableOp(dense_963_matmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
dense_963/MatMulMatMul'leaky_re_lu_866/LeakyRelu:activations:0'dense_963/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
 dense_963/BiasAdd/ReadVariableOpReadVariableOp)dense_963_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
dense_963/BiasAddBiasAdddense_963/MatMul:product:0(dense_963/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
0batch_normalization_867/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_867_batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0l
'batch_normalization_867/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_867/batchnorm/addAddV28batch_normalization_867/batchnorm/ReadVariableOp:value:00batch_normalization_867/batchnorm/add/y:output:0*
T0*
_output_shapes
:m?
'batch_normalization_867/batchnorm/RsqrtRsqrt)batch_normalization_867/batchnorm/add:z:0*
T0*
_output_shapes
:m?
4batch_normalization_867/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_867_batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0?
%batch_normalization_867/batchnorm/mulMul+batch_normalization_867/batchnorm/Rsqrt:y:0<batch_normalization_867/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:m?
'batch_normalization_867/batchnorm/mul_1Muldense_963/BiasAdd:output:0)batch_normalization_867/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????m?
2batch_normalization_867/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_867_batchnorm_readvariableop_1_resource*
_output_shapes
:m*
dtype0?
'batch_normalization_867/batchnorm/mul_2Mul:batch_normalization_867/batchnorm/ReadVariableOp_1:value:0)batch_normalization_867/batchnorm/mul:z:0*
T0*
_output_shapes
:m?
2batch_normalization_867/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_867_batchnorm_readvariableop_2_resource*
_output_shapes
:m*
dtype0?
%batch_normalization_867/batchnorm/subSub:batch_normalization_867/batchnorm/ReadVariableOp_2:value:0+batch_normalization_867/batchnorm/mul_2:z:0*
T0*
_output_shapes
:m?
'batch_normalization_867/batchnorm/add_1AddV2+batch_normalization_867/batchnorm/mul_1:z:0)batch_normalization_867/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????m?
leaky_re_lu_867/LeakyRelu	LeakyRelu+batch_normalization_867/batchnorm/add_1:z:0*'
_output_shapes
:?????????m*
alpha%???>?
dense_964/MatMul/ReadVariableOpReadVariableOp(dense_964_matmul_readvariableop_resource*
_output_shapes

:m.*
dtype0?
dense_964/MatMulMatMul'leaky_re_lu_867/LeakyRelu:activations:0'dense_964/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
 dense_964/BiasAdd/ReadVariableOpReadVariableOp)dense_964_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype0?
dense_964/BiasAddBiasAdddense_964/MatMul:product:0(dense_964/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
0batch_normalization_868/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_868_batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0l
'batch_normalization_868/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_868/batchnorm/addAddV28batch_normalization_868/batchnorm/ReadVariableOp:value:00batch_normalization_868/batchnorm/add/y:output:0*
T0*
_output_shapes
:.?
'batch_normalization_868/batchnorm/RsqrtRsqrt)batch_normalization_868/batchnorm/add:z:0*
T0*
_output_shapes
:.?
4batch_normalization_868/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_868_batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0?
%batch_normalization_868/batchnorm/mulMul+batch_normalization_868/batchnorm/Rsqrt:y:0<batch_normalization_868/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.?
'batch_normalization_868/batchnorm/mul_1Muldense_964/BiasAdd:output:0)batch_normalization_868/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.?
2batch_normalization_868/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_868_batchnorm_readvariableop_1_resource*
_output_shapes
:.*
dtype0?
'batch_normalization_868/batchnorm/mul_2Mul:batch_normalization_868/batchnorm/ReadVariableOp_1:value:0)batch_normalization_868/batchnorm/mul:z:0*
T0*
_output_shapes
:.?
2batch_normalization_868/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_868_batchnorm_readvariableop_2_resource*
_output_shapes
:.*
dtype0?
%batch_normalization_868/batchnorm/subSub:batch_normalization_868/batchnorm/ReadVariableOp_2:value:0+batch_normalization_868/batchnorm/mul_2:z:0*
T0*
_output_shapes
:.?
'batch_normalization_868/batchnorm/add_1AddV2+batch_normalization_868/batchnorm/mul_1:z:0)batch_normalization_868/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.?
leaky_re_lu_868/LeakyRelu	LeakyRelu+batch_normalization_868/batchnorm/add_1:z:0*'
_output_shapes
:?????????.*
alpha%???>?
dense_965/MatMul/ReadVariableOpReadVariableOp(dense_965_matmul_readvariableop_resource*
_output_shapes

:..*
dtype0?
dense_965/MatMulMatMul'leaky_re_lu_868/LeakyRelu:activations:0'dense_965/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
 dense_965/BiasAdd/ReadVariableOpReadVariableOp)dense_965_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype0?
dense_965/BiasAddBiasAdddense_965/MatMul:product:0(dense_965/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????.?
0batch_normalization_869/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_869_batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0l
'batch_normalization_869/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_869/batchnorm/addAddV28batch_normalization_869/batchnorm/ReadVariableOp:value:00batch_normalization_869/batchnorm/add/y:output:0*
T0*
_output_shapes
:.?
'batch_normalization_869/batchnorm/RsqrtRsqrt)batch_normalization_869/batchnorm/add:z:0*
T0*
_output_shapes
:.?
4batch_normalization_869/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_869_batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0?
%batch_normalization_869/batchnorm/mulMul+batch_normalization_869/batchnorm/Rsqrt:y:0<batch_normalization_869/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.?
'batch_normalization_869/batchnorm/mul_1Muldense_965/BiasAdd:output:0)batch_normalization_869/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????.?
2batch_normalization_869/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_869_batchnorm_readvariableop_1_resource*
_output_shapes
:.*
dtype0?
'batch_normalization_869/batchnorm/mul_2Mul:batch_normalization_869/batchnorm/ReadVariableOp_1:value:0)batch_normalization_869/batchnorm/mul:z:0*
T0*
_output_shapes
:.?
2batch_normalization_869/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_869_batchnorm_readvariableop_2_resource*
_output_shapes
:.*
dtype0?
%batch_normalization_869/batchnorm/subSub:batch_normalization_869/batchnorm/ReadVariableOp_2:value:0+batch_normalization_869/batchnorm/mul_2:z:0*
T0*
_output_shapes
:.?
'batch_normalization_869/batchnorm/add_1AddV2+batch_normalization_869/batchnorm/mul_1:z:0)batch_normalization_869/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????.?
leaky_re_lu_869/LeakyRelu	LeakyRelu+batch_normalization_869/batchnorm/add_1:z:0*'
_output_shapes
:?????????.*
alpha%???>?
dense_966/MatMul/ReadVariableOpReadVariableOp(dense_966_matmul_readvariableop_resource*
_output_shapes

:.]*
dtype0?
dense_966/MatMulMatMul'leaky_re_lu_869/LeakyRelu:activations:0'dense_966/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]?
 dense_966/BiasAdd/ReadVariableOpReadVariableOp)dense_966_biasadd_readvariableop_resource*
_output_shapes
:]*
dtype0?
dense_966/BiasAddBiasAdddense_966/MatMul:product:0(dense_966/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]?
0batch_normalization_870/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_870_batchnorm_readvariableop_resource*
_output_shapes
:]*
dtype0l
'batch_normalization_870/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_870/batchnorm/addAddV28batch_normalization_870/batchnorm/ReadVariableOp:value:00batch_normalization_870/batchnorm/add/y:output:0*
T0*
_output_shapes
:]?
'batch_normalization_870/batchnorm/RsqrtRsqrt)batch_normalization_870/batchnorm/add:z:0*
T0*
_output_shapes
:]?
4batch_normalization_870/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_870_batchnorm_mul_readvariableop_resource*
_output_shapes
:]*
dtype0?
%batch_normalization_870/batchnorm/mulMul+batch_normalization_870/batchnorm/Rsqrt:y:0<batch_normalization_870/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:]?
'batch_normalization_870/batchnorm/mul_1Muldense_966/BiasAdd:output:0)batch_normalization_870/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????]?
2batch_normalization_870/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_870_batchnorm_readvariableop_1_resource*
_output_shapes
:]*
dtype0?
'batch_normalization_870/batchnorm/mul_2Mul:batch_normalization_870/batchnorm/ReadVariableOp_1:value:0)batch_normalization_870/batchnorm/mul:z:0*
T0*
_output_shapes
:]?
2batch_normalization_870/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_870_batchnorm_readvariableop_2_resource*
_output_shapes
:]*
dtype0?
%batch_normalization_870/batchnorm/subSub:batch_normalization_870/batchnorm/ReadVariableOp_2:value:0+batch_normalization_870/batchnorm/mul_2:z:0*
T0*
_output_shapes
:]?
'batch_normalization_870/batchnorm/add_1AddV2+batch_normalization_870/batchnorm/mul_1:z:0)batch_normalization_870/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????]?
leaky_re_lu_870/LeakyRelu	LeakyRelu+batch_normalization_870/batchnorm/add_1:z:0*'
_output_shapes
:?????????]*
alpha%???>?
dense_967/MatMul/ReadVariableOpReadVariableOp(dense_967_matmul_readvariableop_resource*
_output_shapes

:]*
dtype0?
dense_967/MatMulMatMul'leaky_re_lu_870/LeakyRelu:activations:0'dense_967/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_967/BiasAdd/ReadVariableOpReadVariableOp)dense_967_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_967/BiasAddBiasAdddense_967/MatMul:product:0(dense_967/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
2dense_961/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_961_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#dense_961/kernel/Regularizer/SquareSquare:dense_961/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:ms
"dense_961/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_961/kernel/Regularizer/SumSum'dense_961/kernel/Regularizer/Square:y:0+dense_961/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_961/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_961/kernel/Regularizer/mulMul+dense_961/kernel/Regularizer/mul/x:output:0)dense_961/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_962/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_962_matmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
#dense_962/kernel/Regularizer/SquareSquare:dense_962/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_962/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_962/kernel/Regularizer/SumSum'dense_962/kernel/Regularizer/Square:y:0+dense_962/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_962/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_962/kernel/Regularizer/mulMul+dense_962/kernel/Regularizer/mul/x:output:0)dense_962/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_963/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_963_matmul_readvariableop_resource*
_output_shapes

:mm*
dtype0?
#dense_963/kernel/Regularizer/SquareSquare:dense_963/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mms
"dense_963/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_963/kernel/Regularizer/SumSum'dense_963/kernel/Regularizer/Square:y:0+dense_963/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_963/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
 dense_963/kernel/Regularizer/mulMul+dense_963/kernel/Regularizer/mul/x:output:0)dense_963/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_964/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_964_matmul_readvariableop_resource*
_output_shapes

:m.*
dtype0?
#dense_964/kernel/Regularizer/SquareSquare:dense_964/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m.s
"dense_964/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_964/kernel/Regularizer/SumSum'dense_964/kernel/Regularizer/Square:y:0+dense_964/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_964/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_964/kernel/Regularizer/mulMul+dense_964/kernel/Regularizer/mul/x:output:0)dense_964/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_965/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_965_matmul_readvariableop_resource*
_output_shapes

:..*
dtype0?
#dense_965/kernel/Regularizer/SquareSquare:dense_965/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:..s
"dense_965/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_965/kernel/Regularizer/SumSum'dense_965/kernel/Regularizer/Square:y:0+dense_965/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_965/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *U??<?
 dense_965/kernel/Regularizer/mulMul+dense_965/kernel/Regularizer/mul/x:output:0)dense_965/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_966/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_966_matmul_readvariableop_resource*
_output_shapes

:.]*
dtype0?
#dense_966/kernel/Regularizer/SquareSquare:dense_966/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:.]s
"dense_966/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_966/kernel/Regularizer/SumSum'dense_966/kernel/Regularizer/Square:y:0+dense_966/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_966/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?<?
 dense_966/kernel/Regularizer/mulMul+dense_966/kernel/Regularizer/mul/x:output:0)dense_966/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_967/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp1^batch_normalization_865/batchnorm/ReadVariableOp3^batch_normalization_865/batchnorm/ReadVariableOp_13^batch_normalization_865/batchnorm/ReadVariableOp_25^batch_normalization_865/batchnorm/mul/ReadVariableOp1^batch_normalization_866/batchnorm/ReadVariableOp3^batch_normalization_866/batchnorm/ReadVariableOp_13^batch_normalization_866/batchnorm/ReadVariableOp_25^batch_normalization_866/batchnorm/mul/ReadVariableOp1^batch_normalization_867/batchnorm/ReadVariableOp3^batch_normalization_867/batchnorm/ReadVariableOp_13^batch_normalization_867/batchnorm/ReadVariableOp_25^batch_normalization_867/batchnorm/mul/ReadVariableOp1^batch_normalization_868/batchnorm/ReadVariableOp3^batch_normalization_868/batchnorm/ReadVariableOp_13^batch_normalization_868/batchnorm/ReadVariableOp_25^batch_normalization_868/batchnorm/mul/ReadVariableOp1^batch_normalization_869/batchnorm/ReadVariableOp3^batch_normalization_869/batchnorm/ReadVariableOp_13^batch_normalization_869/batchnorm/ReadVariableOp_25^batch_normalization_869/batchnorm/mul/ReadVariableOp1^batch_normalization_870/batchnorm/ReadVariableOp3^batch_normalization_870/batchnorm/ReadVariableOp_13^batch_normalization_870/batchnorm/ReadVariableOp_25^batch_normalization_870/batchnorm/mul/ReadVariableOp!^dense_961/BiasAdd/ReadVariableOp ^dense_961/MatMul/ReadVariableOp3^dense_961/kernel/Regularizer/Square/ReadVariableOp!^dense_962/BiasAdd/ReadVariableOp ^dense_962/MatMul/ReadVariableOp3^dense_962/kernel/Regularizer/Square/ReadVariableOp!^dense_963/BiasAdd/ReadVariableOp ^dense_963/MatMul/ReadVariableOp3^dense_963/kernel/Regularizer/Square/ReadVariableOp!^dense_964/BiasAdd/ReadVariableOp ^dense_964/MatMul/ReadVariableOp3^dense_964/kernel/Regularizer/Square/ReadVariableOp!^dense_965/BiasAdd/ReadVariableOp ^dense_965/MatMul/ReadVariableOp3^dense_965/kernel/Regularizer/Square/ReadVariableOp!^dense_966/BiasAdd/ReadVariableOp ^dense_966/MatMul/ReadVariableOp3^dense_966/kernel/Regularizer/Square/ReadVariableOp!^dense_967/BiasAdd/ReadVariableOp ^dense_967/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_865/batchnorm/ReadVariableOp0batch_normalization_865/batchnorm/ReadVariableOp2h
2batch_normalization_865/batchnorm/ReadVariableOp_12batch_normalization_865/batchnorm/ReadVariableOp_12h
2batch_normalization_865/batchnorm/ReadVariableOp_22batch_normalization_865/batchnorm/ReadVariableOp_22l
4batch_normalization_865/batchnorm/mul/ReadVariableOp4batch_normalization_865/batchnorm/mul/ReadVariableOp2d
0batch_normalization_866/batchnorm/ReadVariableOp0batch_normalization_866/batchnorm/ReadVariableOp2h
2batch_normalization_866/batchnorm/ReadVariableOp_12batch_normalization_866/batchnorm/ReadVariableOp_12h
2batch_normalization_866/batchnorm/ReadVariableOp_22batch_normalization_866/batchnorm/ReadVariableOp_22l
4batch_normalization_866/batchnorm/mul/ReadVariableOp4batch_normalization_866/batchnorm/mul/ReadVariableOp2d
0batch_normalization_867/batchnorm/ReadVariableOp0batch_normalization_867/batchnorm/ReadVariableOp2h
2batch_normalization_867/batchnorm/ReadVariableOp_12batch_normalization_867/batchnorm/ReadVariableOp_12h
2batch_normalization_867/batchnorm/ReadVariableOp_22batch_normalization_867/batchnorm/ReadVariableOp_22l
4batch_normalization_867/batchnorm/mul/ReadVariableOp4batch_normalization_867/batchnorm/mul/ReadVariableOp2d
0batch_normalization_868/batchnorm/ReadVariableOp0batch_normalization_868/batchnorm/ReadVariableOp2h
2batch_normalization_868/batchnorm/ReadVariableOp_12batch_normalization_868/batchnorm/ReadVariableOp_12h
2batch_normalization_868/batchnorm/ReadVariableOp_22batch_normalization_868/batchnorm/ReadVariableOp_22l
4batch_normalization_868/batchnorm/mul/ReadVariableOp4batch_normalization_868/batchnorm/mul/ReadVariableOp2d
0batch_normalization_869/batchnorm/ReadVariableOp0batch_normalization_869/batchnorm/ReadVariableOp2h
2batch_normalization_869/batchnorm/ReadVariableOp_12batch_normalization_869/batchnorm/ReadVariableOp_12h
2batch_normalization_869/batchnorm/ReadVariableOp_22batch_normalization_869/batchnorm/ReadVariableOp_22l
4batch_normalization_869/batchnorm/mul/ReadVariableOp4batch_normalization_869/batchnorm/mul/ReadVariableOp2d
0batch_normalization_870/batchnorm/ReadVariableOp0batch_normalization_870/batchnorm/ReadVariableOp2h
2batch_normalization_870/batchnorm/ReadVariableOp_12batch_normalization_870/batchnorm/ReadVariableOp_12h
2batch_normalization_870/batchnorm/ReadVariableOp_22batch_normalization_870/batchnorm/ReadVariableOp_22l
4batch_normalization_870/batchnorm/mul/ReadVariableOp4batch_normalization_870/batchnorm/mul/ReadVariableOp2D
 dense_961/BiasAdd/ReadVariableOp dense_961/BiasAdd/ReadVariableOp2B
dense_961/MatMul/ReadVariableOpdense_961/MatMul/ReadVariableOp2h
2dense_961/kernel/Regularizer/Square/ReadVariableOp2dense_961/kernel/Regularizer/Square/ReadVariableOp2D
 dense_962/BiasAdd/ReadVariableOp dense_962/BiasAdd/ReadVariableOp2B
dense_962/MatMul/ReadVariableOpdense_962/MatMul/ReadVariableOp2h
2dense_962/kernel/Regularizer/Square/ReadVariableOp2dense_962/kernel/Regularizer/Square/ReadVariableOp2D
 dense_963/BiasAdd/ReadVariableOp dense_963/BiasAdd/ReadVariableOp2B
dense_963/MatMul/ReadVariableOpdense_963/MatMul/ReadVariableOp2h
2dense_963/kernel/Regularizer/Square/ReadVariableOp2dense_963/kernel/Regularizer/Square/ReadVariableOp2D
 dense_964/BiasAdd/ReadVariableOp dense_964/BiasAdd/ReadVariableOp2B
dense_964/MatMul/ReadVariableOpdense_964/MatMul/ReadVariableOp2h
2dense_964/kernel/Regularizer/Square/ReadVariableOp2dense_964/kernel/Regularizer/Square/ReadVariableOp2D
 dense_965/BiasAdd/ReadVariableOp dense_965/BiasAdd/ReadVariableOp2B
dense_965/MatMul/ReadVariableOpdense_965/MatMul/ReadVariableOp2h
2dense_965/kernel/Regularizer/Square/ReadVariableOp2dense_965/kernel/Regularizer/Square/ReadVariableOp2D
 dense_966/BiasAdd/ReadVariableOp dense_966/BiasAdd/ReadVariableOp2B
dense_966/MatMul/ReadVariableOpdense_966/MatMul/ReadVariableOp2h
2dense_966/kernel/Regularizer/Square/ReadVariableOp2dense_966/kernel/Regularizer/Square/ReadVariableOp2D
 dense_967/BiasAdd/ReadVariableOp dense_967/BiasAdd/ReadVariableOp2B
dense_967/MatMul/ReadVariableOpdense_967/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
8__inference_batch_normalization_867_layer_call_fn_864827

inputs
unknown:m
	unknown_0:m
	unknown_1:m
	unknown_2:m
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????m*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_862296o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????m`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_870_layer_call_and_return_conditional_losses_865254

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????]*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????]"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????]:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_868_layer_call_fn_864935

inputs
unknown:.
	unknown_0:.
	unknown_1:.
	unknown_2:.
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????.*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_862331o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????.`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_862249

inputs/
!batchnorm_readvariableop_resource:m3
%batchnorm_mul_readvariableop_resource:m1
#batchnorm_readvariableop_1_resource:m1
#batchnorm_readvariableop_2_resource:m
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:m*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:mP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:m*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:mc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????mz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:m*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:mz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:m*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:mr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????mb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????m: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?'
?
__inference_adapt_step_864528
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*
_output_shapes

: *
output_shapes

: *
output_types
2	b
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0	*
_output_shapes

: h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*
_output_shapes

: l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ^
ShapeConst*
_output_shapes
:*
dtype0	*%
valueB	"               Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0P
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:X
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:G
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0V
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0V
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:E
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:V
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @N
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:Z
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:I
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:I
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
?
*__inference_dense_966_layer_call_fn_865148

inputs
unknown:.]
	unknown_0:]
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_966_layer_call_and_return_conditional_losses_862773o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????]`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????.: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????.
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Y
normalization_96_input?
(serving_default_normalization_96_input:0?????????=
	dense_9670
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer-18
layer_with_weights-13
layer-19
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
?

_keep_axis
_reduce_axis
 _reduce_axis_mask
!_broadcast_shape
"mean
"
adapt_mean
#variance
#adapt_variance
	$count
%	keras_api
&_adapt_function"
_tf_keras_layer
?

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
?
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
?
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
?
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
?
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
?

rkernel
sbias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
?
zaxis
	{gamma
|beta
}moving_mean
~moving_variance
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?iter
?beta_1
?beta_2

?decay'm?(m?0m?1m?@m?Am?Im?Jm?Ym?Zm?bm?cm?rm?sm?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?'v?(v?0v?1v?@v?Av?Iv?Jv?Yv?Zv?bv?cv?rv?sv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
?
"0
#1
$2
'3
(4
05
16
27
38
@9
A10
I11
J12
K13
L14
Y15
Z16
b17
c18
d19
e20
r21
s22
{23
|24
}25
~26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40"
trackable_list_wrapper
?
'0
(1
02
13
@4
A5
I6
J7
Y8
Z9
b10
c11
r12
s13
{14
|15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25"
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_sequential_96_layer_call_fn_862931
.__inference_sequential_96_layer_call_fn_863843
.__inference_sequential_96_layer_call_fn_863928
.__inference_sequential_96_layer_call_fn_863434?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_96_layer_call_and_return_conditional_losses_864119
I__inference_sequential_96_layer_call_and_return_conditional_losses_864394
I__inference_sequential_96_layer_call_and_return_conditional_losses_863576
I__inference_sequential_96_layer_call_and_return_conditional_losses_863718?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_862061normalization_96_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
?2?
__inference_adapt_step_864528?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
": m2dense_961/kernel
:m2dense_961/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_961_layer_call_fn_864543?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_961_layer_call_and_return_conditional_losses_864559?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
+:)m2batch_normalization_865/gamma
*:(m2batch_normalization_865/beta
3:1m (2#batch_normalization_865/moving_mean
7:5m (2'batch_normalization_865/moving_variance
<
00
11
22
33"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_865_layer_call_fn_864572
8__inference_batch_normalization_865_layer_call_fn_864585?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_864605
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_864639?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_leaky_re_lu_865_layer_call_fn_864644?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_864649?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
": mm2dense_962/kernel
:m2dense_962/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_962_layer_call_fn_864664?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_962_layer_call_and_return_conditional_losses_864680?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
+:)m2batch_normalization_866/gamma
*:(m2batch_normalization_866/beta
3:1m (2#batch_normalization_866/moving_mean
7:5m (2'batch_normalization_866/moving_variance
<
I0
J1
K2
L3"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_866_layer_call_fn_864693
8__inference_batch_normalization_866_layer_call_fn_864706?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_864726
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_864760?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_leaky_re_lu_866_layer_call_fn_864765?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_866_layer_call_and_return_conditional_losses_864770?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
": mm2dense_963/kernel
:m2dense_963/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_963_layer_call_fn_864785?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_963_layer_call_and_return_conditional_losses_864801?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
+:)m2batch_normalization_867/gamma
*:(m2batch_normalization_867/beta
3:1m (2#batch_normalization_867/moving_mean
7:5m (2'batch_normalization_867/moving_variance
<
b0
c1
d2
e3"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_867_layer_call_fn_864814
8__inference_batch_normalization_867_layer_call_fn_864827?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_864847
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_864881?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_leaky_re_lu_867_layer_call_fn_864886?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_867_layer_call_and_return_conditional_losses_864891?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
": m.2dense_964/kernel
:.2dense_964/bias
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_964_layer_call_fn_864906?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_964_layer_call_and_return_conditional_losses_864922?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
+:).2batch_normalization_868/gamma
*:(.2batch_normalization_868/beta
3:1. (2#batch_normalization_868/moving_mean
7:5. (2'batch_normalization_868/moving_variance
<
{0
|1
}2
~3"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_868_layer_call_fn_864935
8__inference_batch_normalization_868_layer_call_fn_864948?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_864968
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_865002?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_leaky_re_lu_868_layer_call_fn_865007?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_868_layer_call_and_return_conditional_losses_865012?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
": ..2dense_965/kernel
:.2dense_965/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_965_layer_call_fn_865027?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_965_layer_call_and_return_conditional_losses_865043?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
+:).2batch_normalization_869/gamma
*:(.2batch_normalization_869/beta
3:1. (2#batch_normalization_869/moving_mean
7:5. (2'batch_normalization_869/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_869_layer_call_fn_865056
8__inference_batch_normalization_869_layer_call_fn_865069?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_865089
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_865123?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_leaky_re_lu_869_layer_call_fn_865128?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_869_layer_call_and_return_conditional_losses_865133?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
": .]2dense_966/kernel
:]2dense_966/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_966_layer_call_fn_865148?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_966_layer_call_and_return_conditional_losses_865164?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
+:)]2batch_normalization_870/gamma
*:(]2batch_normalization_870/beta
3:1] (2#batch_normalization_870/moving_mean
7:5] (2'batch_normalization_870/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_870_layer_call_fn_865177
8__inference_batch_normalization_870_layer_call_fn_865190?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_865210
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_865244?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_leaky_re_lu_870_layer_call_fn_865249?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_870_layer_call_and_return_conditional_losses_865254?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
": ]2dense_967/kernel
:2dense_967/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_967_layer_call_fn_865263?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_967_layer_call_and_return_conditional_losses_865273?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
?2?
__inference_loss_fn_0_865284?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_865295?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_865306?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_865317?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_865328?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_865339?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?
"0
#1
$2
23
34
K5
L6
d7
e8
}9
~10
?11
?12
?13
?14"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_864481normalization_96_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
':%m2Adam/dense_961/kernel/m
!:m2Adam/dense_961/bias/m
0:.m2$Adam/batch_normalization_865/gamma/m
/:-m2#Adam/batch_normalization_865/beta/m
':%mm2Adam/dense_962/kernel/m
!:m2Adam/dense_962/bias/m
0:.m2$Adam/batch_normalization_866/gamma/m
/:-m2#Adam/batch_normalization_866/beta/m
':%mm2Adam/dense_963/kernel/m
!:m2Adam/dense_963/bias/m
0:.m2$Adam/batch_normalization_867/gamma/m
/:-m2#Adam/batch_normalization_867/beta/m
':%m.2Adam/dense_964/kernel/m
!:.2Adam/dense_964/bias/m
0:..2$Adam/batch_normalization_868/gamma/m
/:-.2#Adam/batch_normalization_868/beta/m
':%..2Adam/dense_965/kernel/m
!:.2Adam/dense_965/bias/m
0:..2$Adam/batch_normalization_869/gamma/m
/:-.2#Adam/batch_normalization_869/beta/m
':%.]2Adam/dense_966/kernel/m
!:]2Adam/dense_966/bias/m
0:.]2$Adam/batch_normalization_870/gamma/m
/:-]2#Adam/batch_normalization_870/beta/m
':%]2Adam/dense_967/kernel/m
!:2Adam/dense_967/bias/m
':%m2Adam/dense_961/kernel/v
!:m2Adam/dense_961/bias/v
0:.m2$Adam/batch_normalization_865/gamma/v
/:-m2#Adam/batch_normalization_865/beta/v
':%mm2Adam/dense_962/kernel/v
!:m2Adam/dense_962/bias/v
0:.m2$Adam/batch_normalization_866/gamma/v
/:-m2#Adam/batch_normalization_866/beta/v
':%mm2Adam/dense_963/kernel/v
!:m2Adam/dense_963/bias/v
0:.m2$Adam/batch_normalization_867/gamma/v
/:-m2#Adam/batch_normalization_867/beta/v
':%m.2Adam/dense_964/kernel/v
!:.2Adam/dense_964/bias/v
0:..2$Adam/batch_normalization_868/gamma/v
/:-.2#Adam/batch_normalization_868/beta/v
':%..2Adam/dense_965/kernel/v
!:.2Adam/dense_965/bias/v
0:..2$Adam/batch_normalization_869/gamma/v
/:-.2#Adam/batch_normalization_869/beta/v
':%.]2Adam/dense_966/kernel/v
!:]2Adam/dense_966/bias/v
0:.]2$Adam/batch_normalization_870/gamma/v
/:-]2#Adam/batch_normalization_870/beta/v
':%]2Adam/dense_967/kernel/v
!:2Adam/dense_967/bias/v
	J
Const
J	
Const_1?
!__inference__wrapped_model_862061?8??'(3021@ALIKJYZebdcrs~{}|????????????????<
5?2
0?-
normalization_96_input?????????
? "5?2
0
	dense_967#? 
	dense_967?????????f
__inference_adapt_step_864528E$"#:?7
0?-
+?(?
? 	IteratorSpec 
? "
 ?
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_864605b30213?0
)?&
 ?
inputs?????????m
p 
? "%?"
?
0?????????m
? ?
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_864639b23013?0
)?&
 ?
inputs?????????m
p
? "%?"
?
0?????????m
? ?
8__inference_batch_normalization_865_layer_call_fn_864572U30213?0
)?&
 ?
inputs?????????m
p 
? "??????????m?
8__inference_batch_normalization_865_layer_call_fn_864585U23013?0
)?&
 ?
inputs?????????m
p
? "??????????m?
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_864726bLIKJ3?0
)?&
 ?
inputs?????????m
p 
? "%?"
?
0?????????m
? ?
S__inference_batch_normalization_866_layer_call_and_return_conditional_losses_864760bKLIJ3?0
)?&
 ?
inputs?????????m
p
? "%?"
?
0?????????m
? ?
8__inference_batch_normalization_866_layer_call_fn_864693ULIKJ3?0
)?&
 ?
inputs?????????m
p 
? "??????????m?
8__inference_batch_normalization_866_layer_call_fn_864706UKLIJ3?0
)?&
 ?
inputs?????????m
p
? "??????????m?
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_864847bebdc3?0
)?&
 ?
inputs?????????m
p 
? "%?"
?
0?????????m
? ?
S__inference_batch_normalization_867_layer_call_and_return_conditional_losses_864881bdebc3?0
)?&
 ?
inputs?????????m
p
? "%?"
?
0?????????m
? ?
8__inference_batch_normalization_867_layer_call_fn_864814Uebdc3?0
)?&
 ?
inputs?????????m
p 
? "??????????m?
8__inference_batch_normalization_867_layer_call_fn_864827Udebc3?0
)?&
 ?
inputs?????????m
p
? "??????????m?
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_864968b~{}|3?0
)?&
 ?
inputs?????????.
p 
? "%?"
?
0?????????.
? ?
S__inference_batch_normalization_868_layer_call_and_return_conditional_losses_865002b}~{|3?0
)?&
 ?
inputs?????????.
p
? "%?"
?
0?????????.
? ?
8__inference_batch_normalization_868_layer_call_fn_864935U~{}|3?0
)?&
 ?
inputs?????????.
p 
? "??????????.?
8__inference_batch_normalization_868_layer_call_fn_864948U}~{|3?0
)?&
 ?
inputs?????????.
p
? "??????????.?
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_865089f????3?0
)?&
 ?
inputs?????????.
p 
? "%?"
?
0?????????.
? ?
S__inference_batch_normalization_869_layer_call_and_return_conditional_losses_865123f????3?0
)?&
 ?
inputs?????????.
p
? "%?"
?
0?????????.
? ?
8__inference_batch_normalization_869_layer_call_fn_865056Y????3?0
)?&
 ?
inputs?????????.
p 
? "??????????.?
8__inference_batch_normalization_869_layer_call_fn_865069Y????3?0
)?&
 ?
inputs?????????.
p
? "??????????.?
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_865210f????3?0
)?&
 ?
inputs?????????]
p 
? "%?"
?
0?????????]
? ?
S__inference_batch_normalization_870_layer_call_and_return_conditional_losses_865244f????3?0
)?&
 ?
inputs?????????]
p
? "%?"
?
0?????????]
? ?
8__inference_batch_normalization_870_layer_call_fn_865177Y????3?0
)?&
 ?
inputs?????????]
p 
? "??????????]?
8__inference_batch_normalization_870_layer_call_fn_865190Y????3?0
)?&
 ?
inputs?????????]
p
? "??????????]?
E__inference_dense_961_layer_call_and_return_conditional_losses_864559\'(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????m
? }
*__inference_dense_961_layer_call_fn_864543O'(/?,
%?"
 ?
inputs?????????
? "??????????m?
E__inference_dense_962_layer_call_and_return_conditional_losses_864680\@A/?,
%?"
 ?
inputs?????????m
? "%?"
?
0?????????m
? }
*__inference_dense_962_layer_call_fn_864664O@A/?,
%?"
 ?
inputs?????????m
? "??????????m?
E__inference_dense_963_layer_call_and_return_conditional_losses_864801\YZ/?,
%?"
 ?
inputs?????????m
? "%?"
?
0?????????m
? }
*__inference_dense_963_layer_call_fn_864785OYZ/?,
%?"
 ?
inputs?????????m
? "??????????m?
E__inference_dense_964_layer_call_and_return_conditional_losses_864922\rs/?,
%?"
 ?
inputs?????????m
? "%?"
?
0?????????.
? }
*__inference_dense_964_layer_call_fn_864906Ors/?,
%?"
 ?
inputs?????????m
? "??????????.?
E__inference_dense_965_layer_call_and_return_conditional_losses_865043^??/?,
%?"
 ?
inputs?????????.
? "%?"
?
0?????????.
? 
*__inference_dense_965_layer_call_fn_865027Q??/?,
%?"
 ?
inputs?????????.
? "??????????.?
E__inference_dense_966_layer_call_and_return_conditional_losses_865164^??/?,
%?"
 ?
inputs?????????.
? "%?"
?
0?????????]
? 
*__inference_dense_966_layer_call_fn_865148Q??/?,
%?"
 ?
inputs?????????.
? "??????????]?
E__inference_dense_967_layer_call_and_return_conditional_losses_865273^??/?,
%?"
 ?
inputs?????????]
? "%?"
?
0?????????
? 
*__inference_dense_967_layer_call_fn_865263Q??/?,
%?"
 ?
inputs?????????]
? "???????????
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_864649X/?,
%?"
 ?
inputs?????????m
? "%?"
?
0?????????m
? 
0__inference_leaky_re_lu_865_layer_call_fn_864644K/?,
%?"
 ?
inputs?????????m
? "??????????m?
K__inference_leaky_re_lu_866_layer_call_and_return_conditional_losses_864770X/?,
%?"
 ?
inputs?????????m
? "%?"
?
0?????????m
? 
0__inference_leaky_re_lu_866_layer_call_fn_864765K/?,
%?"
 ?
inputs?????????m
? "??????????m?
K__inference_leaky_re_lu_867_layer_call_and_return_conditional_losses_864891X/?,
%?"
 ?
inputs?????????m
? "%?"
?
0?????????m
? 
0__inference_leaky_re_lu_867_layer_call_fn_864886K/?,
%?"
 ?
inputs?????????m
? "??????????m?
K__inference_leaky_re_lu_868_layer_call_and_return_conditional_losses_865012X/?,
%?"
 ?
inputs?????????.
? "%?"
?
0?????????.
? 
0__inference_leaky_re_lu_868_layer_call_fn_865007K/?,
%?"
 ?
inputs?????????.
? "??????????.?
K__inference_leaky_re_lu_869_layer_call_and_return_conditional_losses_865133X/?,
%?"
 ?
inputs?????????.
? "%?"
?
0?????????.
? 
0__inference_leaky_re_lu_869_layer_call_fn_865128K/?,
%?"
 ?
inputs?????????.
? "??????????.?
K__inference_leaky_re_lu_870_layer_call_and_return_conditional_losses_865254X/?,
%?"
 ?
inputs?????????]
? "%?"
?
0?????????]
? 
0__inference_leaky_re_lu_870_layer_call_fn_865249K/?,
%?"
 ?
inputs?????????]
? "??????????];
__inference_loss_fn_0_865284'?

? 
? "? ;
__inference_loss_fn_1_865295@?

? 
? "? ;
__inference_loss_fn_2_865306Y?

? 
? "? ;
__inference_loss_fn_3_865317r?

? 
? "? <
__inference_loss_fn_4_865328??

? 
? "? <
__inference_loss_fn_5_865339??

? 
? "? ?
I__inference_sequential_96_layer_call_and_return_conditional_losses_863576?8??'(3021@ALIKJYZebdcrs~{}|??????????????G?D
=?:
0?-
normalization_96_input?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_96_layer_call_and_return_conditional_losses_863718?8??'(2301@AKLIJYZdebcrs}~{|??????????????G?D
=?:
0?-
normalization_96_input?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_96_layer_call_and_return_conditional_losses_864119?8??'(3021@ALIKJYZebdcrs~{}|??????????????7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_96_layer_call_and_return_conditional_losses_864394?8??'(2301@AKLIJYZdebcrs}~{|??????????????7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
.__inference_sequential_96_layer_call_fn_862931?8??'(3021@ALIKJYZebdcrs~{}|??????????????G?D
=?:
0?-
normalization_96_input?????????
p 

 
? "???????????
.__inference_sequential_96_layer_call_fn_863434?8??'(2301@AKLIJYZdebcrs}~{|??????????????G?D
=?:
0?-
normalization_96_input?????????
p

 
? "???????????
.__inference_sequential_96_layer_call_fn_863843?8??'(3021@ALIKJYZebdcrs~{}|??????????????7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
.__inference_sequential_96_layer_call_fn_863928?8??'(2301@AKLIJYZdebcrs}~{|??????????????7?4
-?*
 ?
inputs?????????
p

 
? "???????????
$__inference_signature_wrapper_864481?8??'(3021@ALIKJYZebdcrs~{}|??????????????Y?V
? 
O?L
J
normalization_96_input0?-
normalization_96_input?????????"5?2
0
	dense_967#? 
	dense_967?????????