??
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
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
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
dense_599/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:J*!
shared_namedense_599/kernel
u
$dense_599/kernel/Read/ReadVariableOpReadVariableOpdense_599/kernel*
_output_shapes

:J*
dtype0
t
dense_599/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*
shared_namedense_599/bias
m
"dense_599/bias/Read/ReadVariableOpReadVariableOpdense_599/bias*
_output_shapes
:J*
dtype0
?
batch_normalization_542/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*.
shared_namebatch_normalization_542/gamma
?
1batch_normalization_542/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_542/gamma*
_output_shapes
:J*
dtype0
?
batch_normalization_542/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*-
shared_namebatch_normalization_542/beta
?
0batch_normalization_542/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_542/beta*
_output_shapes
:J*
dtype0
?
#batch_normalization_542/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*4
shared_name%#batch_normalization_542/moving_mean
?
7batch_normalization_542/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_542/moving_mean*
_output_shapes
:J*
dtype0
?
'batch_normalization_542/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*8
shared_name)'batch_normalization_542/moving_variance
?
;batch_normalization_542/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_542/moving_variance*
_output_shapes
:J*
dtype0
|
dense_600/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:JJ*!
shared_namedense_600/kernel
u
$dense_600/kernel/Read/ReadVariableOpReadVariableOpdense_600/kernel*
_output_shapes

:JJ*
dtype0
t
dense_600/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*
shared_namedense_600/bias
m
"dense_600/bias/Read/ReadVariableOpReadVariableOpdense_600/bias*
_output_shapes
:J*
dtype0
?
batch_normalization_543/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*.
shared_namebatch_normalization_543/gamma
?
1batch_normalization_543/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_543/gamma*
_output_shapes
:J*
dtype0
?
batch_normalization_543/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*-
shared_namebatch_normalization_543/beta
?
0batch_normalization_543/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_543/beta*
_output_shapes
:J*
dtype0
?
#batch_normalization_543/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*4
shared_name%#batch_normalization_543/moving_mean
?
7batch_normalization_543/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_543/moving_mean*
_output_shapes
:J*
dtype0
?
'batch_normalization_543/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*8
shared_name)'batch_normalization_543/moving_variance
?
;batch_normalization_543/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_543/moving_variance*
_output_shapes
:J*
dtype0
|
dense_601/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:J,*!
shared_namedense_601/kernel
u
$dense_601/kernel/Read/ReadVariableOpReadVariableOpdense_601/kernel*
_output_shapes

:J,*
dtype0
t
dense_601/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*
shared_namedense_601/bias
m
"dense_601/bias/Read/ReadVariableOpReadVariableOpdense_601/bias*
_output_shapes
:,*
dtype0
?
batch_normalization_544/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*.
shared_namebatch_normalization_544/gamma
?
1batch_normalization_544/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_544/gamma*
_output_shapes
:,*
dtype0
?
batch_normalization_544/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*-
shared_namebatch_normalization_544/beta
?
0batch_normalization_544/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_544/beta*
_output_shapes
:,*
dtype0
?
#batch_normalization_544/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*4
shared_name%#batch_normalization_544/moving_mean
?
7batch_normalization_544/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_544/moving_mean*
_output_shapes
:,*
dtype0
?
'batch_normalization_544/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*8
shared_name)'batch_normalization_544/moving_variance
?
;batch_normalization_544/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_544/moving_variance*
_output_shapes
:,*
dtype0
|
dense_602/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,,*!
shared_namedense_602/kernel
u
$dense_602/kernel/Read/ReadVariableOpReadVariableOpdense_602/kernel*
_output_shapes

:,,*
dtype0
t
dense_602/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*
shared_namedense_602/bias
m
"dense_602/bias/Read/ReadVariableOpReadVariableOpdense_602/bias*
_output_shapes
:,*
dtype0
?
batch_normalization_545/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*.
shared_namebatch_normalization_545/gamma
?
1batch_normalization_545/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_545/gamma*
_output_shapes
:,*
dtype0
?
batch_normalization_545/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*-
shared_namebatch_normalization_545/beta
?
0batch_normalization_545/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_545/beta*
_output_shapes
:,*
dtype0
?
#batch_normalization_545/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*4
shared_name%#batch_normalization_545/moving_mean
?
7batch_normalization_545/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_545/moving_mean*
_output_shapes
:,*
dtype0
?
'batch_normalization_545/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*8
shared_name)'batch_normalization_545/moving_variance
?
;batch_normalization_545/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_545/moving_variance*
_output_shapes
:,*
dtype0
|
dense_603/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,2*!
shared_namedense_603/kernel
u
$dense_603/kernel/Read/ReadVariableOpReadVariableOpdense_603/kernel*
_output_shapes

:,2*
dtype0
t
dense_603/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_603/bias
m
"dense_603/bias/Read/ReadVariableOpReadVariableOpdense_603/bias*
_output_shapes
:2*
dtype0
?
batch_normalization_546/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*.
shared_namebatch_normalization_546/gamma
?
1batch_normalization_546/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_546/gamma*
_output_shapes
:2*
dtype0
?
batch_normalization_546/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*-
shared_namebatch_normalization_546/beta
?
0batch_normalization_546/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_546/beta*
_output_shapes
:2*
dtype0
?
#batch_normalization_546/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#batch_normalization_546/moving_mean
?
7batch_normalization_546/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_546/moving_mean*
_output_shapes
:2*
dtype0
?
'batch_normalization_546/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*8
shared_name)'batch_normalization_546/moving_variance
?
;batch_normalization_546/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_546/moving_variance*
_output_shapes
:2*
dtype0
|
dense_604/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*!
shared_namedense_604/kernel
u
$dense_604/kernel/Read/ReadVariableOpReadVariableOpdense_604/kernel*
_output_shapes

:22*
dtype0
t
dense_604/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_604/bias
m
"dense_604/bias/Read/ReadVariableOpReadVariableOpdense_604/bias*
_output_shapes
:2*
dtype0
?
batch_normalization_547/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*.
shared_namebatch_normalization_547/gamma
?
1batch_normalization_547/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_547/gamma*
_output_shapes
:2*
dtype0
?
batch_normalization_547/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*-
shared_namebatch_normalization_547/beta
?
0batch_normalization_547/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_547/beta*
_output_shapes
:2*
dtype0
?
#batch_normalization_547/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#batch_normalization_547/moving_mean
?
7batch_normalization_547/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_547/moving_mean*
_output_shapes
:2*
dtype0
?
'batch_normalization_547/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*8
shared_name)'batch_normalization_547/moving_variance
?
;batch_normalization_547/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_547/moving_variance*
_output_shapes
:2*
dtype0
|
dense_605/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*!
shared_namedense_605/kernel
u
$dense_605/kernel/Read/ReadVariableOpReadVariableOpdense_605/kernel*
_output_shapes

:2*
dtype0
t
dense_605/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_605/bias
m
"dense_605/bias/Read/ReadVariableOpReadVariableOpdense_605/bias*
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
Adam/dense_599/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:J*(
shared_nameAdam/dense_599/kernel/m
?
+Adam/dense_599/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_599/kernel/m*
_output_shapes

:J*
dtype0
?
Adam/dense_599/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*&
shared_nameAdam/dense_599/bias/m
{
)Adam/dense_599/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_599/bias/m*
_output_shapes
:J*
dtype0
?
$Adam/batch_normalization_542/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*5
shared_name&$Adam/batch_normalization_542/gamma/m
?
8Adam/batch_normalization_542/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_542/gamma/m*
_output_shapes
:J*
dtype0
?
#Adam/batch_normalization_542/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*4
shared_name%#Adam/batch_normalization_542/beta/m
?
7Adam/batch_normalization_542/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_542/beta/m*
_output_shapes
:J*
dtype0
?
Adam/dense_600/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:JJ*(
shared_nameAdam/dense_600/kernel/m
?
+Adam/dense_600/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_600/kernel/m*
_output_shapes

:JJ*
dtype0
?
Adam/dense_600/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*&
shared_nameAdam/dense_600/bias/m
{
)Adam/dense_600/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_600/bias/m*
_output_shapes
:J*
dtype0
?
$Adam/batch_normalization_543/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*5
shared_name&$Adam/batch_normalization_543/gamma/m
?
8Adam/batch_normalization_543/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_543/gamma/m*
_output_shapes
:J*
dtype0
?
#Adam/batch_normalization_543/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*4
shared_name%#Adam/batch_normalization_543/beta/m
?
7Adam/batch_normalization_543/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_543/beta/m*
_output_shapes
:J*
dtype0
?
Adam/dense_601/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:J,*(
shared_nameAdam/dense_601/kernel/m
?
+Adam/dense_601/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_601/kernel/m*
_output_shapes

:J,*
dtype0
?
Adam/dense_601/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*&
shared_nameAdam/dense_601/bias/m
{
)Adam/dense_601/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_601/bias/m*
_output_shapes
:,*
dtype0
?
$Adam/batch_normalization_544/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*5
shared_name&$Adam/batch_normalization_544/gamma/m
?
8Adam/batch_normalization_544/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_544/gamma/m*
_output_shapes
:,*
dtype0
?
#Adam/batch_normalization_544/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*4
shared_name%#Adam/batch_normalization_544/beta/m
?
7Adam/batch_normalization_544/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_544/beta/m*
_output_shapes
:,*
dtype0
?
Adam/dense_602/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,,*(
shared_nameAdam/dense_602/kernel/m
?
+Adam/dense_602/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_602/kernel/m*
_output_shapes

:,,*
dtype0
?
Adam/dense_602/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*&
shared_nameAdam/dense_602/bias/m
{
)Adam/dense_602/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_602/bias/m*
_output_shapes
:,*
dtype0
?
$Adam/batch_normalization_545/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*5
shared_name&$Adam/batch_normalization_545/gamma/m
?
8Adam/batch_normalization_545/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_545/gamma/m*
_output_shapes
:,*
dtype0
?
#Adam/batch_normalization_545/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*4
shared_name%#Adam/batch_normalization_545/beta/m
?
7Adam/batch_normalization_545/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_545/beta/m*
_output_shapes
:,*
dtype0
?
Adam/dense_603/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,2*(
shared_nameAdam/dense_603/kernel/m
?
+Adam/dense_603/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_603/kernel/m*
_output_shapes

:,2*
dtype0
?
Adam/dense_603/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/dense_603/bias/m
{
)Adam/dense_603/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_603/bias/m*
_output_shapes
:2*
dtype0
?
$Adam/batch_normalization_546/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*5
shared_name&$Adam/batch_normalization_546/gamma/m
?
8Adam/batch_normalization_546/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_546/gamma/m*
_output_shapes
:2*
dtype0
?
#Adam/batch_normalization_546/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#Adam/batch_normalization_546/beta/m
?
7Adam/batch_normalization_546/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_546/beta/m*
_output_shapes
:2*
dtype0
?
Adam/dense_604/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*(
shared_nameAdam/dense_604/kernel/m
?
+Adam/dense_604/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_604/kernel/m*
_output_shapes

:22*
dtype0
?
Adam/dense_604/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/dense_604/bias/m
{
)Adam/dense_604/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_604/bias/m*
_output_shapes
:2*
dtype0
?
$Adam/batch_normalization_547/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*5
shared_name&$Adam/batch_normalization_547/gamma/m
?
8Adam/batch_normalization_547/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_547/gamma/m*
_output_shapes
:2*
dtype0
?
#Adam/batch_normalization_547/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#Adam/batch_normalization_547/beta/m
?
7Adam/batch_normalization_547/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_547/beta/m*
_output_shapes
:2*
dtype0
?
Adam/dense_605/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/dense_605/kernel/m
?
+Adam/dense_605/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_605/kernel/m*
_output_shapes

:2*
dtype0
?
Adam/dense_605/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_605/bias/m
{
)Adam/dense_605/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_605/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_599/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:J*(
shared_nameAdam/dense_599/kernel/v
?
+Adam/dense_599/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_599/kernel/v*
_output_shapes

:J*
dtype0
?
Adam/dense_599/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*&
shared_nameAdam/dense_599/bias/v
{
)Adam/dense_599/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_599/bias/v*
_output_shapes
:J*
dtype0
?
$Adam/batch_normalization_542/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*5
shared_name&$Adam/batch_normalization_542/gamma/v
?
8Adam/batch_normalization_542/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_542/gamma/v*
_output_shapes
:J*
dtype0
?
#Adam/batch_normalization_542/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*4
shared_name%#Adam/batch_normalization_542/beta/v
?
7Adam/batch_normalization_542/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_542/beta/v*
_output_shapes
:J*
dtype0
?
Adam/dense_600/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:JJ*(
shared_nameAdam/dense_600/kernel/v
?
+Adam/dense_600/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_600/kernel/v*
_output_shapes

:JJ*
dtype0
?
Adam/dense_600/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*&
shared_nameAdam/dense_600/bias/v
{
)Adam/dense_600/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_600/bias/v*
_output_shapes
:J*
dtype0
?
$Adam/batch_normalization_543/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*5
shared_name&$Adam/batch_normalization_543/gamma/v
?
8Adam/batch_normalization_543/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_543/gamma/v*
_output_shapes
:J*
dtype0
?
#Adam/batch_normalization_543/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*4
shared_name%#Adam/batch_normalization_543/beta/v
?
7Adam/batch_normalization_543/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_543/beta/v*
_output_shapes
:J*
dtype0
?
Adam/dense_601/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:J,*(
shared_nameAdam/dense_601/kernel/v
?
+Adam/dense_601/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_601/kernel/v*
_output_shapes

:J,*
dtype0
?
Adam/dense_601/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*&
shared_nameAdam/dense_601/bias/v
{
)Adam/dense_601/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_601/bias/v*
_output_shapes
:,*
dtype0
?
$Adam/batch_normalization_544/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*5
shared_name&$Adam/batch_normalization_544/gamma/v
?
8Adam/batch_normalization_544/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_544/gamma/v*
_output_shapes
:,*
dtype0
?
#Adam/batch_normalization_544/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*4
shared_name%#Adam/batch_normalization_544/beta/v
?
7Adam/batch_normalization_544/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_544/beta/v*
_output_shapes
:,*
dtype0
?
Adam/dense_602/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,,*(
shared_nameAdam/dense_602/kernel/v
?
+Adam/dense_602/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_602/kernel/v*
_output_shapes

:,,*
dtype0
?
Adam/dense_602/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*&
shared_nameAdam/dense_602/bias/v
{
)Adam/dense_602/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_602/bias/v*
_output_shapes
:,*
dtype0
?
$Adam/batch_normalization_545/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*5
shared_name&$Adam/batch_normalization_545/gamma/v
?
8Adam/batch_normalization_545/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_545/gamma/v*
_output_shapes
:,*
dtype0
?
#Adam/batch_normalization_545/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*4
shared_name%#Adam/batch_normalization_545/beta/v
?
7Adam/batch_normalization_545/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_545/beta/v*
_output_shapes
:,*
dtype0
?
Adam/dense_603/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,2*(
shared_nameAdam/dense_603/kernel/v
?
+Adam/dense_603/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_603/kernel/v*
_output_shapes

:,2*
dtype0
?
Adam/dense_603/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/dense_603/bias/v
{
)Adam/dense_603/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_603/bias/v*
_output_shapes
:2*
dtype0
?
$Adam/batch_normalization_546/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*5
shared_name&$Adam/batch_normalization_546/gamma/v
?
8Adam/batch_normalization_546/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_546/gamma/v*
_output_shapes
:2*
dtype0
?
#Adam/batch_normalization_546/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#Adam/batch_normalization_546/beta/v
?
7Adam/batch_normalization_546/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_546/beta/v*
_output_shapes
:2*
dtype0
?
Adam/dense_604/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*(
shared_nameAdam/dense_604/kernel/v
?
+Adam/dense_604/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_604/kernel/v*
_output_shapes

:22*
dtype0
?
Adam/dense_604/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/dense_604/bias/v
{
)Adam/dense_604/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_604/bias/v*
_output_shapes
:2*
dtype0
?
$Adam/batch_normalization_547/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*5
shared_name&$Adam/batch_normalization_547/gamma/v
?
8Adam/batch_normalization_547/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_547/gamma/v*
_output_shapes
:2*
dtype0
?
#Adam/batch_normalization_547/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#Adam/batch_normalization_547/beta/v
?
7Adam/batch_normalization_547/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_547/beta/v*
_output_shapes
:2*
dtype0
?
Adam/dense_605/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/dense_605/kernel/v
?
+Adam/dense_605/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_605/kernel/v*
_output_shapes

:2*
dtype0
?
Adam/dense_605/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_605/bias/v
{
)Adam/dense_605/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_605/bias/v*
_output_shapes
:*
dtype0
r
ConstConst*
_output_shapes

:*
dtype0*5
value,B*"TU?B???A @@??A??A??A???=
t
Const_1Const*
_output_shapes

:*
dtype0*5
value,B*"3sEt?B??*@??A?q?A?q?A??<

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
* 
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
VARIABLE_VALUEdense_599/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_599/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 
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
VARIABLE_VALUEbatch_normalization_542/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_542/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_542/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE'batch_normalization_542/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEdense_600/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_600/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

@0
A1*
* 
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
VARIABLE_VALUEbatch_normalization_543/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_543/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_543/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE'batch_normalization_543/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEdense_601/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_601/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*
* 
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
VARIABLE_VALUEbatch_normalization_544/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_544/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_544/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE'batch_normalization_544/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_602/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_602/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

r0
s1*

r0
s1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
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
VARIABLE_VALUEbatch_normalization_545/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_545/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_545/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE'batch_normalization_545/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEdense_603/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_603/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
mg
VARIABLE_VALUEbatch_normalization_546/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_546/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_546/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUE'batch_normalization_546/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_604/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_604/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
mg
VARIABLE_VALUEbatch_normalization_547/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_547/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_547/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUE'batch_normalization_547/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_605/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_605/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
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
* 
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
* 
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
* 
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
* 
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
VARIABLE_VALUEAdam/dense_599/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_599/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_542/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_542/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_600/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_600/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_543/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_543/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_601/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_601/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_544/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_544/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_602/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_602/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_545/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_545/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_603/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_603/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_546/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_546/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/dense_604/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/dense_604/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_547/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_547/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/dense_605/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/dense_605/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_599/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_599/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_542/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_542/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_600/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_600/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_543/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_543/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_601/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_601/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_544/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_544/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_602/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_602/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_545/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_545/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_603/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_603/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_546/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_546/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/dense_604/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/dense_604/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_547/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_547/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/dense_605/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/dense_605/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
&serving_default_normalization_57_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_57_inputConstConst_1dense_599/kerneldense_599/bias'batch_normalization_542/moving_variancebatch_normalization_542/gamma#batch_normalization_542/moving_meanbatch_normalization_542/betadense_600/kerneldense_600/bias'batch_normalization_543/moving_variancebatch_normalization_543/gamma#batch_normalization_543/moving_meanbatch_normalization_543/betadense_601/kerneldense_601/bias'batch_normalization_544/moving_variancebatch_normalization_544/gamma#batch_normalization_544/moving_meanbatch_normalization_544/betadense_602/kerneldense_602/bias'batch_normalization_545/moving_variancebatch_normalization_545/gamma#batch_normalization_545/moving_meanbatch_normalization_545/betadense_603/kerneldense_603/bias'batch_normalization_546/moving_variancebatch_normalization_546/gamma#batch_normalization_546/moving_meanbatch_normalization_546/betadense_604/kerneldense_604/bias'batch_normalization_547/moving_variancebatch_normalization_547/gamma#batch_normalization_547/moving_meanbatch_normalization_547/betadense_605/kerneldense_605/bias*4
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
$__inference_signature_wrapper_776422
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_599/kernel/Read/ReadVariableOp"dense_599/bias/Read/ReadVariableOp1batch_normalization_542/gamma/Read/ReadVariableOp0batch_normalization_542/beta/Read/ReadVariableOp7batch_normalization_542/moving_mean/Read/ReadVariableOp;batch_normalization_542/moving_variance/Read/ReadVariableOp$dense_600/kernel/Read/ReadVariableOp"dense_600/bias/Read/ReadVariableOp1batch_normalization_543/gamma/Read/ReadVariableOp0batch_normalization_543/beta/Read/ReadVariableOp7batch_normalization_543/moving_mean/Read/ReadVariableOp;batch_normalization_543/moving_variance/Read/ReadVariableOp$dense_601/kernel/Read/ReadVariableOp"dense_601/bias/Read/ReadVariableOp1batch_normalization_544/gamma/Read/ReadVariableOp0batch_normalization_544/beta/Read/ReadVariableOp7batch_normalization_544/moving_mean/Read/ReadVariableOp;batch_normalization_544/moving_variance/Read/ReadVariableOp$dense_602/kernel/Read/ReadVariableOp"dense_602/bias/Read/ReadVariableOp1batch_normalization_545/gamma/Read/ReadVariableOp0batch_normalization_545/beta/Read/ReadVariableOp7batch_normalization_545/moving_mean/Read/ReadVariableOp;batch_normalization_545/moving_variance/Read/ReadVariableOp$dense_603/kernel/Read/ReadVariableOp"dense_603/bias/Read/ReadVariableOp1batch_normalization_546/gamma/Read/ReadVariableOp0batch_normalization_546/beta/Read/ReadVariableOp7batch_normalization_546/moving_mean/Read/ReadVariableOp;batch_normalization_546/moving_variance/Read/ReadVariableOp$dense_604/kernel/Read/ReadVariableOp"dense_604/bias/Read/ReadVariableOp1batch_normalization_547/gamma/Read/ReadVariableOp0batch_normalization_547/beta/Read/ReadVariableOp7batch_normalization_547/moving_mean/Read/ReadVariableOp;batch_normalization_547/moving_variance/Read/ReadVariableOp$dense_605/kernel/Read/ReadVariableOp"dense_605/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_599/kernel/m/Read/ReadVariableOp)Adam/dense_599/bias/m/Read/ReadVariableOp8Adam/batch_normalization_542/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_542/beta/m/Read/ReadVariableOp+Adam/dense_600/kernel/m/Read/ReadVariableOp)Adam/dense_600/bias/m/Read/ReadVariableOp8Adam/batch_normalization_543/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_543/beta/m/Read/ReadVariableOp+Adam/dense_601/kernel/m/Read/ReadVariableOp)Adam/dense_601/bias/m/Read/ReadVariableOp8Adam/batch_normalization_544/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_544/beta/m/Read/ReadVariableOp+Adam/dense_602/kernel/m/Read/ReadVariableOp)Adam/dense_602/bias/m/Read/ReadVariableOp8Adam/batch_normalization_545/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_545/beta/m/Read/ReadVariableOp+Adam/dense_603/kernel/m/Read/ReadVariableOp)Adam/dense_603/bias/m/Read/ReadVariableOp8Adam/batch_normalization_546/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_546/beta/m/Read/ReadVariableOp+Adam/dense_604/kernel/m/Read/ReadVariableOp)Adam/dense_604/bias/m/Read/ReadVariableOp8Adam/batch_normalization_547/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_547/beta/m/Read/ReadVariableOp+Adam/dense_605/kernel/m/Read/ReadVariableOp)Adam/dense_605/bias/m/Read/ReadVariableOp+Adam/dense_599/kernel/v/Read/ReadVariableOp)Adam/dense_599/bias/v/Read/ReadVariableOp8Adam/batch_normalization_542/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_542/beta/v/Read/ReadVariableOp+Adam/dense_600/kernel/v/Read/ReadVariableOp)Adam/dense_600/bias/v/Read/ReadVariableOp8Adam/batch_normalization_543/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_543/beta/v/Read/ReadVariableOp+Adam/dense_601/kernel/v/Read/ReadVariableOp)Adam/dense_601/bias/v/Read/ReadVariableOp8Adam/batch_normalization_544/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_544/beta/v/Read/ReadVariableOp+Adam/dense_602/kernel/v/Read/ReadVariableOp)Adam/dense_602/bias/v/Read/ReadVariableOp8Adam/batch_normalization_545/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_545/beta/v/Read/ReadVariableOp+Adam/dense_603/kernel/v/Read/ReadVariableOp)Adam/dense_603/bias/v/Read/ReadVariableOp8Adam/batch_normalization_546/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_546/beta/v/Read/ReadVariableOp+Adam/dense_604/kernel/v/Read/ReadVariableOp)Adam/dense_604/bias/v/Read/ReadVariableOp8Adam/batch_normalization_547/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_547/beta/v/Read/ReadVariableOp+Adam/dense_605/kernel/v/Read/ReadVariableOp)Adam/dense_605/bias/v/Read/ReadVariableOpConst_2*p
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
__inference__traced_save_777464
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_599/kerneldense_599/biasbatch_normalization_542/gammabatch_normalization_542/beta#batch_normalization_542/moving_mean'batch_normalization_542/moving_variancedense_600/kerneldense_600/biasbatch_normalization_543/gammabatch_normalization_543/beta#batch_normalization_543/moving_mean'batch_normalization_543/moving_variancedense_601/kerneldense_601/biasbatch_normalization_544/gammabatch_normalization_544/beta#batch_normalization_544/moving_mean'batch_normalization_544/moving_variancedense_602/kerneldense_602/biasbatch_normalization_545/gammabatch_normalization_545/beta#batch_normalization_545/moving_mean'batch_normalization_545/moving_variancedense_603/kerneldense_603/biasbatch_normalization_546/gammabatch_normalization_546/beta#batch_normalization_546/moving_mean'batch_normalization_546/moving_variancedense_604/kerneldense_604/biasbatch_normalization_547/gammabatch_normalization_547/beta#batch_normalization_547/moving_mean'batch_normalization_547/moving_variancedense_605/kerneldense_605/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_599/kernel/mAdam/dense_599/bias/m$Adam/batch_normalization_542/gamma/m#Adam/batch_normalization_542/beta/mAdam/dense_600/kernel/mAdam/dense_600/bias/m$Adam/batch_normalization_543/gamma/m#Adam/batch_normalization_543/beta/mAdam/dense_601/kernel/mAdam/dense_601/bias/m$Adam/batch_normalization_544/gamma/m#Adam/batch_normalization_544/beta/mAdam/dense_602/kernel/mAdam/dense_602/bias/m$Adam/batch_normalization_545/gamma/m#Adam/batch_normalization_545/beta/mAdam/dense_603/kernel/mAdam/dense_603/bias/m$Adam/batch_normalization_546/gamma/m#Adam/batch_normalization_546/beta/mAdam/dense_604/kernel/mAdam/dense_604/bias/m$Adam/batch_normalization_547/gamma/m#Adam/batch_normalization_547/beta/mAdam/dense_605/kernel/mAdam/dense_605/bias/mAdam/dense_599/kernel/vAdam/dense_599/bias/v$Adam/batch_normalization_542/gamma/v#Adam/batch_normalization_542/beta/vAdam/dense_600/kernel/vAdam/dense_600/bias/v$Adam/batch_normalization_543/gamma/v#Adam/batch_normalization_543/beta/vAdam/dense_601/kernel/vAdam/dense_601/bias/v$Adam/batch_normalization_544/gamma/v#Adam/batch_normalization_544/beta/vAdam/dense_602/kernel/vAdam/dense_602/bias/v$Adam/batch_normalization_545/gamma/v#Adam/batch_normalization_545/beta/vAdam/dense_603/kernel/vAdam/dense_603/bias/v$Adam/batch_normalization_546/gamma/v#Adam/batch_normalization_546/beta/vAdam/dense_604/kernel/vAdam/dense_604/bias/v$Adam/batch_normalization_547/gamma/v#Adam/batch_normalization_547/beta/vAdam/dense_605/kernel/vAdam/dense_605/bias/v*o
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
"__inference__traced_restore_777771Ȅ
?
?	
.__inference_sequential_57_layer_call_fn_775555
normalization_57_input
unknown
	unknown_0
	unknown_1:J
	unknown_2:J
	unknown_3:J
	unknown_4:J
	unknown_5:J
	unknown_6:J
	unknown_7:JJ
	unknown_8:J
	unknown_9:J

unknown_10:J

unknown_11:J

unknown_12:J

unknown_13:J,

unknown_14:,

unknown_15:,

unknown_16:,

unknown_17:,

unknown_18:,

unknown_19:,,

unknown_20:,

unknown_21:,

unknown_22:,

unknown_23:,

unknown_24:,

unknown_25:,2

unknown_26:2

unknown_27:2

unknown_28:2

unknown_29:2

unknown_30:2

unknown_31:22

unknown_32:2

unknown_33:2

unknown_34:2

unknown_35:2

unknown_36:2

unknown_37:2

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallnormalization_57_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_57_layer_call_and_return_conditional_losses_775387o
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
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:?????????
0
_user_specified_namenormalization_57_input:$ 

_output_shapes

::$ 

_output_shapes

:
?
L
0__inference_leaky_re_lu_546_layer_call_fn_777009

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
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_546_layer_call_and_return_conditional_losses_774954`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
E__inference_dense_605_layer_call_and_return_conditional_losses_774998

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
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
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_776861

inputs/
!batchnorm_readvariableop_resource:,3
%batchnorm_mul_readvariableop_resource:,1
#batchnorm_readvariableop_1_resource:,1
#batchnorm_readvariableop_2_resource:,
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:,*
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
:,P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:,~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:,*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:,z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:,*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:,r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????,?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_545_layer_call_and_return_conditional_losses_776905

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????,*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????,"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????,:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_545_layer_call_fn_776900

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
:?????????,* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_545_layer_call_and_return_conditional_losses_774922`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????,"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????,:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_543_layer_call_and_return_conditional_losses_776687

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????J*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????J"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????J:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_546_layer_call_and_return_conditional_losses_774954

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????2*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
E__inference_dense_600_layer_call_and_return_conditional_losses_776597

inputs0
matmul_readvariableop_resource:JJ-
biasadd_readvariableop_resource:J
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:JJ*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Jr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Jw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_776752

inputs/
!batchnorm_readvariableop_resource:,3
%batchnorm_mul_readvariableop_resource:,1
#batchnorm_readvariableop_1_resource:,1
#batchnorm_readvariableop_2_resource:,
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:,*
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
:,P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:,~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:,*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:,z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:,*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:,r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????,?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_777004

inputs5
'assignmovingavg_readvariableop_resource:27
)assignmovingavg_1_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:2/
!batchnorm_readvariableop_resource:2
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:2*
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
:2*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2?
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
:2*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2?
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
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_776895

inputs5
'assignmovingavg_readvariableop_resource:,7
)assignmovingavg_1_readvariableop_resource:,3
%batchnorm_mul_readvariableop_resource:,/
!batchnorm_readvariableop_resource:,
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:,*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:,?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????,l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:,*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:,*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:,*
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
:,*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:,x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:,?
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
:,*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:,~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:,?
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
:,P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:,~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:,v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:,*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:,r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????,?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_544_layer_call_and_return_conditional_losses_774890

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????,*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????,"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????,:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_544_layer_call_fn_776719

inputs
unknown:,
	unknown_0:,
	unknown_1:,
	unknown_2:,
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_774478o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????,`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
??
?#
I__inference_sequential_57_layer_call_and_return_conditional_losses_776096

inputs
normalization_57_sub_y
normalization_57_sqrt_x:
(dense_599_matmul_readvariableop_resource:J7
)dense_599_biasadd_readvariableop_resource:JG
9batch_normalization_542_batchnorm_readvariableop_resource:JK
=batch_normalization_542_batchnorm_mul_readvariableop_resource:JI
;batch_normalization_542_batchnorm_readvariableop_1_resource:JI
;batch_normalization_542_batchnorm_readvariableop_2_resource:J:
(dense_600_matmul_readvariableop_resource:JJ7
)dense_600_biasadd_readvariableop_resource:JG
9batch_normalization_543_batchnorm_readvariableop_resource:JK
=batch_normalization_543_batchnorm_mul_readvariableop_resource:JI
;batch_normalization_543_batchnorm_readvariableop_1_resource:JI
;batch_normalization_543_batchnorm_readvariableop_2_resource:J:
(dense_601_matmul_readvariableop_resource:J,7
)dense_601_biasadd_readvariableop_resource:,G
9batch_normalization_544_batchnorm_readvariableop_resource:,K
=batch_normalization_544_batchnorm_mul_readvariableop_resource:,I
;batch_normalization_544_batchnorm_readvariableop_1_resource:,I
;batch_normalization_544_batchnorm_readvariableop_2_resource:,:
(dense_602_matmul_readvariableop_resource:,,7
)dense_602_biasadd_readvariableop_resource:,G
9batch_normalization_545_batchnorm_readvariableop_resource:,K
=batch_normalization_545_batchnorm_mul_readvariableop_resource:,I
;batch_normalization_545_batchnorm_readvariableop_1_resource:,I
;batch_normalization_545_batchnorm_readvariableop_2_resource:,:
(dense_603_matmul_readvariableop_resource:,27
)dense_603_biasadd_readvariableop_resource:2G
9batch_normalization_546_batchnorm_readvariableop_resource:2K
=batch_normalization_546_batchnorm_mul_readvariableop_resource:2I
;batch_normalization_546_batchnorm_readvariableop_1_resource:2I
;batch_normalization_546_batchnorm_readvariableop_2_resource:2:
(dense_604_matmul_readvariableop_resource:227
)dense_604_biasadd_readvariableop_resource:2G
9batch_normalization_547_batchnorm_readvariableop_resource:2K
=batch_normalization_547_batchnorm_mul_readvariableop_resource:2I
;batch_normalization_547_batchnorm_readvariableop_1_resource:2I
;batch_normalization_547_batchnorm_readvariableop_2_resource:2:
(dense_605_matmul_readvariableop_resource:27
)dense_605_biasadd_readvariableop_resource:
identity??0batch_normalization_542/batchnorm/ReadVariableOp?2batch_normalization_542/batchnorm/ReadVariableOp_1?2batch_normalization_542/batchnorm/ReadVariableOp_2?4batch_normalization_542/batchnorm/mul/ReadVariableOp?0batch_normalization_543/batchnorm/ReadVariableOp?2batch_normalization_543/batchnorm/ReadVariableOp_1?2batch_normalization_543/batchnorm/ReadVariableOp_2?4batch_normalization_543/batchnorm/mul/ReadVariableOp?0batch_normalization_544/batchnorm/ReadVariableOp?2batch_normalization_544/batchnorm/ReadVariableOp_1?2batch_normalization_544/batchnorm/ReadVariableOp_2?4batch_normalization_544/batchnorm/mul/ReadVariableOp?0batch_normalization_545/batchnorm/ReadVariableOp?2batch_normalization_545/batchnorm/ReadVariableOp_1?2batch_normalization_545/batchnorm/ReadVariableOp_2?4batch_normalization_545/batchnorm/mul/ReadVariableOp?0batch_normalization_546/batchnorm/ReadVariableOp?2batch_normalization_546/batchnorm/ReadVariableOp_1?2batch_normalization_546/batchnorm/ReadVariableOp_2?4batch_normalization_546/batchnorm/mul/ReadVariableOp?0batch_normalization_547/batchnorm/ReadVariableOp?2batch_normalization_547/batchnorm/ReadVariableOp_1?2batch_normalization_547/batchnorm/ReadVariableOp_2?4batch_normalization_547/batchnorm/mul/ReadVariableOp? dense_599/BiasAdd/ReadVariableOp?dense_599/MatMul/ReadVariableOp? dense_600/BiasAdd/ReadVariableOp?dense_600/MatMul/ReadVariableOp? dense_601/BiasAdd/ReadVariableOp?dense_601/MatMul/ReadVariableOp? dense_602/BiasAdd/ReadVariableOp?dense_602/MatMul/ReadVariableOp? dense_603/BiasAdd/ReadVariableOp?dense_603/MatMul/ReadVariableOp? dense_604/BiasAdd/ReadVariableOp?dense_604/MatMul/ReadVariableOp? dense_605/BiasAdd/ReadVariableOp?dense_605/MatMul/ReadVariableOpm
normalization_57/subSubinputsnormalization_57_sub_y*
T0*'
_output_shapes
:?????????_
normalization_57/SqrtSqrtnormalization_57_sqrt_x*
T0*
_output_shapes

:_
normalization_57/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_57/MaximumMaximumnormalization_57/Sqrt:y:0#normalization_57/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_57/truedivRealDivnormalization_57/sub:z:0normalization_57/Maximum:z:0*
T0*'
_output_shapes
:??????????
dense_599/MatMul/ReadVariableOpReadVariableOp(dense_599_matmul_readvariableop_resource*
_output_shapes

:J*
dtype0?
dense_599/MatMulMatMulnormalization_57/truediv:z:0'dense_599/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J?
 dense_599/BiasAdd/ReadVariableOpReadVariableOp)dense_599_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype0?
dense_599/BiasAddBiasAdddense_599/MatMul:product:0(dense_599/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J?
0batch_normalization_542/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_542_batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0l
'batch_normalization_542/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_542/batchnorm/addAddV28batch_normalization_542/batchnorm/ReadVariableOp:value:00batch_normalization_542/batchnorm/add/y:output:0*
T0*
_output_shapes
:J?
'batch_normalization_542/batchnorm/RsqrtRsqrt)batch_normalization_542/batchnorm/add:z:0*
T0*
_output_shapes
:J?
4batch_normalization_542/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_542_batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0?
%batch_normalization_542/batchnorm/mulMul+batch_normalization_542/batchnorm/Rsqrt:y:0<batch_normalization_542/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:J?
'batch_normalization_542/batchnorm/mul_1Muldense_599/BiasAdd:output:0)batch_normalization_542/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????J?
2batch_normalization_542/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_542_batchnorm_readvariableop_1_resource*
_output_shapes
:J*
dtype0?
'batch_normalization_542/batchnorm/mul_2Mul:batch_normalization_542/batchnorm/ReadVariableOp_1:value:0)batch_normalization_542/batchnorm/mul:z:0*
T0*
_output_shapes
:J?
2batch_normalization_542/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_542_batchnorm_readvariableop_2_resource*
_output_shapes
:J*
dtype0?
%batch_normalization_542/batchnorm/subSub:batch_normalization_542/batchnorm/ReadVariableOp_2:value:0+batch_normalization_542/batchnorm/mul_2:z:0*
T0*
_output_shapes
:J?
'batch_normalization_542/batchnorm/add_1AddV2+batch_normalization_542/batchnorm/mul_1:z:0)batch_normalization_542/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????J?
leaky_re_lu_542/LeakyRelu	LeakyRelu+batch_normalization_542/batchnorm/add_1:z:0*'
_output_shapes
:?????????J*
alpha%???>?
dense_600/MatMul/ReadVariableOpReadVariableOp(dense_600_matmul_readvariableop_resource*
_output_shapes

:JJ*
dtype0?
dense_600/MatMulMatMul'leaky_re_lu_542/LeakyRelu:activations:0'dense_600/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J?
 dense_600/BiasAdd/ReadVariableOpReadVariableOp)dense_600_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype0?
dense_600/BiasAddBiasAdddense_600/MatMul:product:0(dense_600/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J?
0batch_normalization_543/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_543_batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0l
'batch_normalization_543/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_543/batchnorm/addAddV28batch_normalization_543/batchnorm/ReadVariableOp:value:00batch_normalization_543/batchnorm/add/y:output:0*
T0*
_output_shapes
:J?
'batch_normalization_543/batchnorm/RsqrtRsqrt)batch_normalization_543/batchnorm/add:z:0*
T0*
_output_shapes
:J?
4batch_normalization_543/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_543_batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0?
%batch_normalization_543/batchnorm/mulMul+batch_normalization_543/batchnorm/Rsqrt:y:0<batch_normalization_543/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:J?
'batch_normalization_543/batchnorm/mul_1Muldense_600/BiasAdd:output:0)batch_normalization_543/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????J?
2batch_normalization_543/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_543_batchnorm_readvariableop_1_resource*
_output_shapes
:J*
dtype0?
'batch_normalization_543/batchnorm/mul_2Mul:batch_normalization_543/batchnorm/ReadVariableOp_1:value:0)batch_normalization_543/batchnorm/mul:z:0*
T0*
_output_shapes
:J?
2batch_normalization_543/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_543_batchnorm_readvariableop_2_resource*
_output_shapes
:J*
dtype0?
%batch_normalization_543/batchnorm/subSub:batch_normalization_543/batchnorm/ReadVariableOp_2:value:0+batch_normalization_543/batchnorm/mul_2:z:0*
T0*
_output_shapes
:J?
'batch_normalization_543/batchnorm/add_1AddV2+batch_normalization_543/batchnorm/mul_1:z:0)batch_normalization_543/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????J?
leaky_re_lu_543/LeakyRelu	LeakyRelu+batch_normalization_543/batchnorm/add_1:z:0*'
_output_shapes
:?????????J*
alpha%???>?
dense_601/MatMul/ReadVariableOpReadVariableOp(dense_601_matmul_readvariableop_resource*
_output_shapes

:J,*
dtype0?
dense_601/MatMulMatMul'leaky_re_lu_543/LeakyRelu:activations:0'dense_601/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,?
 dense_601/BiasAdd/ReadVariableOpReadVariableOp)dense_601_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0?
dense_601/BiasAddBiasAdddense_601/MatMul:product:0(dense_601/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,?
0batch_normalization_544/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_544_batchnorm_readvariableop_resource*
_output_shapes
:,*
dtype0l
'batch_normalization_544/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_544/batchnorm/addAddV28batch_normalization_544/batchnorm/ReadVariableOp:value:00batch_normalization_544/batchnorm/add/y:output:0*
T0*
_output_shapes
:,?
'batch_normalization_544/batchnorm/RsqrtRsqrt)batch_normalization_544/batchnorm/add:z:0*
T0*
_output_shapes
:,?
4batch_normalization_544/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_544_batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0?
%batch_normalization_544/batchnorm/mulMul+batch_normalization_544/batchnorm/Rsqrt:y:0<batch_normalization_544/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,?
'batch_normalization_544/batchnorm/mul_1Muldense_601/BiasAdd:output:0)batch_normalization_544/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,?
2batch_normalization_544/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_544_batchnorm_readvariableop_1_resource*
_output_shapes
:,*
dtype0?
'batch_normalization_544/batchnorm/mul_2Mul:batch_normalization_544/batchnorm/ReadVariableOp_1:value:0)batch_normalization_544/batchnorm/mul:z:0*
T0*
_output_shapes
:,?
2batch_normalization_544/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_544_batchnorm_readvariableop_2_resource*
_output_shapes
:,*
dtype0?
%batch_normalization_544/batchnorm/subSub:batch_normalization_544/batchnorm/ReadVariableOp_2:value:0+batch_normalization_544/batchnorm/mul_2:z:0*
T0*
_output_shapes
:,?
'batch_normalization_544/batchnorm/add_1AddV2+batch_normalization_544/batchnorm/mul_1:z:0)batch_normalization_544/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,?
leaky_re_lu_544/LeakyRelu	LeakyRelu+batch_normalization_544/batchnorm/add_1:z:0*'
_output_shapes
:?????????,*
alpha%???>?
dense_602/MatMul/ReadVariableOpReadVariableOp(dense_602_matmul_readvariableop_resource*
_output_shapes

:,,*
dtype0?
dense_602/MatMulMatMul'leaky_re_lu_544/LeakyRelu:activations:0'dense_602/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,?
 dense_602/BiasAdd/ReadVariableOpReadVariableOp)dense_602_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0?
dense_602/BiasAddBiasAdddense_602/MatMul:product:0(dense_602/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,?
0batch_normalization_545/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_545_batchnorm_readvariableop_resource*
_output_shapes
:,*
dtype0l
'batch_normalization_545/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_545/batchnorm/addAddV28batch_normalization_545/batchnorm/ReadVariableOp:value:00batch_normalization_545/batchnorm/add/y:output:0*
T0*
_output_shapes
:,?
'batch_normalization_545/batchnorm/RsqrtRsqrt)batch_normalization_545/batchnorm/add:z:0*
T0*
_output_shapes
:,?
4batch_normalization_545/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_545_batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0?
%batch_normalization_545/batchnorm/mulMul+batch_normalization_545/batchnorm/Rsqrt:y:0<batch_normalization_545/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,?
'batch_normalization_545/batchnorm/mul_1Muldense_602/BiasAdd:output:0)batch_normalization_545/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,?
2batch_normalization_545/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_545_batchnorm_readvariableop_1_resource*
_output_shapes
:,*
dtype0?
'batch_normalization_545/batchnorm/mul_2Mul:batch_normalization_545/batchnorm/ReadVariableOp_1:value:0)batch_normalization_545/batchnorm/mul:z:0*
T0*
_output_shapes
:,?
2batch_normalization_545/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_545_batchnorm_readvariableop_2_resource*
_output_shapes
:,*
dtype0?
%batch_normalization_545/batchnorm/subSub:batch_normalization_545/batchnorm/ReadVariableOp_2:value:0+batch_normalization_545/batchnorm/mul_2:z:0*
T0*
_output_shapes
:,?
'batch_normalization_545/batchnorm/add_1AddV2+batch_normalization_545/batchnorm/mul_1:z:0)batch_normalization_545/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,?
leaky_re_lu_545/LeakyRelu	LeakyRelu+batch_normalization_545/batchnorm/add_1:z:0*'
_output_shapes
:?????????,*
alpha%???>?
dense_603/MatMul/ReadVariableOpReadVariableOp(dense_603_matmul_readvariableop_resource*
_output_shapes

:,2*
dtype0?
dense_603/MatMulMatMul'leaky_re_lu_545/LeakyRelu:activations:0'dense_603/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
 dense_603/BiasAdd/ReadVariableOpReadVariableOp)dense_603_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_603/BiasAddBiasAdddense_603/MatMul:product:0(dense_603/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
0batch_normalization_546/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_546_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0l
'batch_normalization_546/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_546/batchnorm/addAddV28batch_normalization_546/batchnorm/ReadVariableOp:value:00batch_normalization_546/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
'batch_normalization_546/batchnorm/RsqrtRsqrt)batch_normalization_546/batchnorm/add:z:0*
T0*
_output_shapes
:2?
4batch_normalization_546/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_546_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0?
%batch_normalization_546/batchnorm/mulMul+batch_normalization_546/batchnorm/Rsqrt:y:0<batch_normalization_546/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
'batch_normalization_546/batchnorm/mul_1Muldense_603/BiasAdd:output:0)batch_normalization_546/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2?
2batch_normalization_546/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_546_batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0?
'batch_normalization_546/batchnorm/mul_2Mul:batch_normalization_546/batchnorm/ReadVariableOp_1:value:0)batch_normalization_546/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
2batch_normalization_546/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_546_batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0?
%batch_normalization_546/batchnorm/subSub:batch_normalization_546/batchnorm/ReadVariableOp_2:value:0+batch_normalization_546/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
'batch_normalization_546/batchnorm/add_1AddV2+batch_normalization_546/batchnorm/mul_1:z:0)batch_normalization_546/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2?
leaky_re_lu_546/LeakyRelu	LeakyRelu+batch_normalization_546/batchnorm/add_1:z:0*'
_output_shapes
:?????????2*
alpha%???>?
dense_604/MatMul/ReadVariableOpReadVariableOp(dense_604_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_604/MatMulMatMul'leaky_re_lu_546/LeakyRelu:activations:0'dense_604/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
 dense_604/BiasAdd/ReadVariableOpReadVariableOp)dense_604_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_604/BiasAddBiasAdddense_604/MatMul:product:0(dense_604/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
0batch_normalization_547/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_547_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0l
'batch_normalization_547/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_547/batchnorm/addAddV28batch_normalization_547/batchnorm/ReadVariableOp:value:00batch_normalization_547/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
'batch_normalization_547/batchnorm/RsqrtRsqrt)batch_normalization_547/batchnorm/add:z:0*
T0*
_output_shapes
:2?
4batch_normalization_547/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_547_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0?
%batch_normalization_547/batchnorm/mulMul+batch_normalization_547/batchnorm/Rsqrt:y:0<batch_normalization_547/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
'batch_normalization_547/batchnorm/mul_1Muldense_604/BiasAdd:output:0)batch_normalization_547/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2?
2batch_normalization_547/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_547_batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0?
'batch_normalization_547/batchnorm/mul_2Mul:batch_normalization_547/batchnorm/ReadVariableOp_1:value:0)batch_normalization_547/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
2batch_normalization_547/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_547_batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0?
%batch_normalization_547/batchnorm/subSub:batch_normalization_547/batchnorm/ReadVariableOp_2:value:0+batch_normalization_547/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
'batch_normalization_547/batchnorm/add_1AddV2+batch_normalization_547/batchnorm/mul_1:z:0)batch_normalization_547/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2?
leaky_re_lu_547/LeakyRelu	LeakyRelu+batch_normalization_547/batchnorm/add_1:z:0*'
_output_shapes
:?????????2*
alpha%???>?
dense_605/MatMul/ReadVariableOpReadVariableOp(dense_605_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_605/MatMulMatMul'leaky_re_lu_547/LeakyRelu:activations:0'dense_605/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_605/BiasAdd/ReadVariableOpReadVariableOp)dense_605_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_605/BiasAddBiasAdddense_605/MatMul:product:0(dense_605/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_605/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp1^batch_normalization_542/batchnorm/ReadVariableOp3^batch_normalization_542/batchnorm/ReadVariableOp_13^batch_normalization_542/batchnorm/ReadVariableOp_25^batch_normalization_542/batchnorm/mul/ReadVariableOp1^batch_normalization_543/batchnorm/ReadVariableOp3^batch_normalization_543/batchnorm/ReadVariableOp_13^batch_normalization_543/batchnorm/ReadVariableOp_25^batch_normalization_543/batchnorm/mul/ReadVariableOp1^batch_normalization_544/batchnorm/ReadVariableOp3^batch_normalization_544/batchnorm/ReadVariableOp_13^batch_normalization_544/batchnorm/ReadVariableOp_25^batch_normalization_544/batchnorm/mul/ReadVariableOp1^batch_normalization_545/batchnorm/ReadVariableOp3^batch_normalization_545/batchnorm/ReadVariableOp_13^batch_normalization_545/batchnorm/ReadVariableOp_25^batch_normalization_545/batchnorm/mul/ReadVariableOp1^batch_normalization_546/batchnorm/ReadVariableOp3^batch_normalization_546/batchnorm/ReadVariableOp_13^batch_normalization_546/batchnorm/ReadVariableOp_25^batch_normalization_546/batchnorm/mul/ReadVariableOp1^batch_normalization_547/batchnorm/ReadVariableOp3^batch_normalization_547/batchnorm/ReadVariableOp_13^batch_normalization_547/batchnorm/ReadVariableOp_25^batch_normalization_547/batchnorm/mul/ReadVariableOp!^dense_599/BiasAdd/ReadVariableOp ^dense_599/MatMul/ReadVariableOp!^dense_600/BiasAdd/ReadVariableOp ^dense_600/MatMul/ReadVariableOp!^dense_601/BiasAdd/ReadVariableOp ^dense_601/MatMul/ReadVariableOp!^dense_602/BiasAdd/ReadVariableOp ^dense_602/MatMul/ReadVariableOp!^dense_603/BiasAdd/ReadVariableOp ^dense_603/MatMul/ReadVariableOp!^dense_604/BiasAdd/ReadVariableOp ^dense_604/MatMul/ReadVariableOp!^dense_605/BiasAdd/ReadVariableOp ^dense_605/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_542/batchnorm/ReadVariableOp0batch_normalization_542/batchnorm/ReadVariableOp2h
2batch_normalization_542/batchnorm/ReadVariableOp_12batch_normalization_542/batchnorm/ReadVariableOp_12h
2batch_normalization_542/batchnorm/ReadVariableOp_22batch_normalization_542/batchnorm/ReadVariableOp_22l
4batch_normalization_542/batchnorm/mul/ReadVariableOp4batch_normalization_542/batchnorm/mul/ReadVariableOp2d
0batch_normalization_543/batchnorm/ReadVariableOp0batch_normalization_543/batchnorm/ReadVariableOp2h
2batch_normalization_543/batchnorm/ReadVariableOp_12batch_normalization_543/batchnorm/ReadVariableOp_12h
2batch_normalization_543/batchnorm/ReadVariableOp_22batch_normalization_543/batchnorm/ReadVariableOp_22l
4batch_normalization_543/batchnorm/mul/ReadVariableOp4batch_normalization_543/batchnorm/mul/ReadVariableOp2d
0batch_normalization_544/batchnorm/ReadVariableOp0batch_normalization_544/batchnorm/ReadVariableOp2h
2batch_normalization_544/batchnorm/ReadVariableOp_12batch_normalization_544/batchnorm/ReadVariableOp_12h
2batch_normalization_544/batchnorm/ReadVariableOp_22batch_normalization_544/batchnorm/ReadVariableOp_22l
4batch_normalization_544/batchnorm/mul/ReadVariableOp4batch_normalization_544/batchnorm/mul/ReadVariableOp2d
0batch_normalization_545/batchnorm/ReadVariableOp0batch_normalization_545/batchnorm/ReadVariableOp2h
2batch_normalization_545/batchnorm/ReadVariableOp_12batch_normalization_545/batchnorm/ReadVariableOp_12h
2batch_normalization_545/batchnorm/ReadVariableOp_22batch_normalization_545/batchnorm/ReadVariableOp_22l
4batch_normalization_545/batchnorm/mul/ReadVariableOp4batch_normalization_545/batchnorm/mul/ReadVariableOp2d
0batch_normalization_546/batchnorm/ReadVariableOp0batch_normalization_546/batchnorm/ReadVariableOp2h
2batch_normalization_546/batchnorm/ReadVariableOp_12batch_normalization_546/batchnorm/ReadVariableOp_12h
2batch_normalization_546/batchnorm/ReadVariableOp_22batch_normalization_546/batchnorm/ReadVariableOp_22l
4batch_normalization_546/batchnorm/mul/ReadVariableOp4batch_normalization_546/batchnorm/mul/ReadVariableOp2d
0batch_normalization_547/batchnorm/ReadVariableOp0batch_normalization_547/batchnorm/ReadVariableOp2h
2batch_normalization_547/batchnorm/ReadVariableOp_12batch_normalization_547/batchnorm/ReadVariableOp_12h
2batch_normalization_547/batchnorm/ReadVariableOp_22batch_normalization_547/batchnorm/ReadVariableOp_22l
4batch_normalization_547/batchnorm/mul/ReadVariableOp4batch_normalization_547/batchnorm/mul/ReadVariableOp2D
 dense_599/BiasAdd/ReadVariableOp dense_599/BiasAdd/ReadVariableOp2B
dense_599/MatMul/ReadVariableOpdense_599/MatMul/ReadVariableOp2D
 dense_600/BiasAdd/ReadVariableOp dense_600/BiasAdd/ReadVariableOp2B
dense_600/MatMul/ReadVariableOpdense_600/MatMul/ReadVariableOp2D
 dense_601/BiasAdd/ReadVariableOp dense_601/BiasAdd/ReadVariableOp2B
dense_601/MatMul/ReadVariableOpdense_601/MatMul/ReadVariableOp2D
 dense_602/BiasAdd/ReadVariableOp dense_602/BiasAdd/ReadVariableOp2B
dense_602/MatMul/ReadVariableOpdense_602/MatMul/ReadVariableOp2D
 dense_603/BiasAdd/ReadVariableOp dense_603/BiasAdd/ReadVariableOp2B
dense_603/MatMul/ReadVariableOpdense_603/MatMul/ReadVariableOp2D
 dense_604/BiasAdd/ReadVariableOp dense_604/BiasAdd/ReadVariableOp2B
dense_604/MatMul/ReadVariableOpdense_604/MatMul/ReadVariableOp2D
 dense_605/BiasAdd/ReadVariableOp dense_605/BiasAdd/ReadVariableOp2B
dense_605/MatMul/ReadVariableOpdense_605/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_774396

inputs/
!batchnorm_readvariableop_resource:J3
%batchnorm_mul_readvariableop_resource:J1
#batchnorm_readvariableop_1_resource:J1
#batchnorm_readvariableop_2_resource:J
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:J*
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
:JP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:J~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Jc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????Jz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:J*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Jz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:J*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Jr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????Jb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????J?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
??
?.
__inference__traced_save_777464
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_599_kernel_read_readvariableop-
)savev2_dense_599_bias_read_readvariableop<
8savev2_batch_normalization_542_gamma_read_readvariableop;
7savev2_batch_normalization_542_beta_read_readvariableopB
>savev2_batch_normalization_542_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_542_moving_variance_read_readvariableop/
+savev2_dense_600_kernel_read_readvariableop-
)savev2_dense_600_bias_read_readvariableop<
8savev2_batch_normalization_543_gamma_read_readvariableop;
7savev2_batch_normalization_543_beta_read_readvariableopB
>savev2_batch_normalization_543_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_543_moving_variance_read_readvariableop/
+savev2_dense_601_kernel_read_readvariableop-
)savev2_dense_601_bias_read_readvariableop<
8savev2_batch_normalization_544_gamma_read_readvariableop;
7savev2_batch_normalization_544_beta_read_readvariableopB
>savev2_batch_normalization_544_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_544_moving_variance_read_readvariableop/
+savev2_dense_602_kernel_read_readvariableop-
)savev2_dense_602_bias_read_readvariableop<
8savev2_batch_normalization_545_gamma_read_readvariableop;
7savev2_batch_normalization_545_beta_read_readvariableopB
>savev2_batch_normalization_545_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_545_moving_variance_read_readvariableop/
+savev2_dense_603_kernel_read_readvariableop-
)savev2_dense_603_bias_read_readvariableop<
8savev2_batch_normalization_546_gamma_read_readvariableop;
7savev2_batch_normalization_546_beta_read_readvariableopB
>savev2_batch_normalization_546_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_546_moving_variance_read_readvariableop/
+savev2_dense_604_kernel_read_readvariableop-
)savev2_dense_604_bias_read_readvariableop<
8savev2_batch_normalization_547_gamma_read_readvariableop;
7savev2_batch_normalization_547_beta_read_readvariableopB
>savev2_batch_normalization_547_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_547_moving_variance_read_readvariableop/
+savev2_dense_605_kernel_read_readvariableop-
)savev2_dense_605_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_599_kernel_m_read_readvariableop4
0savev2_adam_dense_599_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_542_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_542_beta_m_read_readvariableop6
2savev2_adam_dense_600_kernel_m_read_readvariableop4
0savev2_adam_dense_600_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_543_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_543_beta_m_read_readvariableop6
2savev2_adam_dense_601_kernel_m_read_readvariableop4
0savev2_adam_dense_601_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_544_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_544_beta_m_read_readvariableop6
2savev2_adam_dense_602_kernel_m_read_readvariableop4
0savev2_adam_dense_602_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_545_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_545_beta_m_read_readvariableop6
2savev2_adam_dense_603_kernel_m_read_readvariableop4
0savev2_adam_dense_603_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_546_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_546_beta_m_read_readvariableop6
2savev2_adam_dense_604_kernel_m_read_readvariableop4
0savev2_adam_dense_604_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_547_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_547_beta_m_read_readvariableop6
2savev2_adam_dense_605_kernel_m_read_readvariableop4
0savev2_adam_dense_605_bias_m_read_readvariableop6
2savev2_adam_dense_599_kernel_v_read_readvariableop4
0savev2_adam_dense_599_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_542_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_542_beta_v_read_readvariableop6
2savev2_adam_dense_600_kernel_v_read_readvariableop4
0savev2_adam_dense_600_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_543_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_543_beta_v_read_readvariableop6
2savev2_adam_dense_601_kernel_v_read_readvariableop4
0savev2_adam_dense_601_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_544_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_544_beta_v_read_readvariableop6
2savev2_adam_dense_602_kernel_v_read_readvariableop4
0savev2_adam_dense_602_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_545_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_545_beta_v_read_readvariableop6
2savev2_adam_dense_603_kernel_v_read_readvariableop4
0savev2_adam_dense_603_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_546_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_546_beta_v_read_readvariableop6
2savev2_adam_dense_604_kernel_v_read_readvariableop4
0savev2_adam_dense_604_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_547_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_547_beta_v_read_readvariableop6
2savev2_adam_dense_605_kernel_v_read_readvariableop4
0savev2_adam_dense_605_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_599_kernel_read_readvariableop)savev2_dense_599_bias_read_readvariableop8savev2_batch_normalization_542_gamma_read_readvariableop7savev2_batch_normalization_542_beta_read_readvariableop>savev2_batch_normalization_542_moving_mean_read_readvariableopBsavev2_batch_normalization_542_moving_variance_read_readvariableop+savev2_dense_600_kernel_read_readvariableop)savev2_dense_600_bias_read_readvariableop8savev2_batch_normalization_543_gamma_read_readvariableop7savev2_batch_normalization_543_beta_read_readvariableop>savev2_batch_normalization_543_moving_mean_read_readvariableopBsavev2_batch_normalization_543_moving_variance_read_readvariableop+savev2_dense_601_kernel_read_readvariableop)savev2_dense_601_bias_read_readvariableop8savev2_batch_normalization_544_gamma_read_readvariableop7savev2_batch_normalization_544_beta_read_readvariableop>savev2_batch_normalization_544_moving_mean_read_readvariableopBsavev2_batch_normalization_544_moving_variance_read_readvariableop+savev2_dense_602_kernel_read_readvariableop)savev2_dense_602_bias_read_readvariableop8savev2_batch_normalization_545_gamma_read_readvariableop7savev2_batch_normalization_545_beta_read_readvariableop>savev2_batch_normalization_545_moving_mean_read_readvariableopBsavev2_batch_normalization_545_moving_variance_read_readvariableop+savev2_dense_603_kernel_read_readvariableop)savev2_dense_603_bias_read_readvariableop8savev2_batch_normalization_546_gamma_read_readvariableop7savev2_batch_normalization_546_beta_read_readvariableop>savev2_batch_normalization_546_moving_mean_read_readvariableopBsavev2_batch_normalization_546_moving_variance_read_readvariableop+savev2_dense_604_kernel_read_readvariableop)savev2_dense_604_bias_read_readvariableop8savev2_batch_normalization_547_gamma_read_readvariableop7savev2_batch_normalization_547_beta_read_readvariableop>savev2_batch_normalization_547_moving_mean_read_readvariableopBsavev2_batch_normalization_547_moving_variance_read_readvariableop+savev2_dense_605_kernel_read_readvariableop)savev2_dense_605_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_599_kernel_m_read_readvariableop0savev2_adam_dense_599_bias_m_read_readvariableop?savev2_adam_batch_normalization_542_gamma_m_read_readvariableop>savev2_adam_batch_normalization_542_beta_m_read_readvariableop2savev2_adam_dense_600_kernel_m_read_readvariableop0savev2_adam_dense_600_bias_m_read_readvariableop?savev2_adam_batch_normalization_543_gamma_m_read_readvariableop>savev2_adam_batch_normalization_543_beta_m_read_readvariableop2savev2_adam_dense_601_kernel_m_read_readvariableop0savev2_adam_dense_601_bias_m_read_readvariableop?savev2_adam_batch_normalization_544_gamma_m_read_readvariableop>savev2_adam_batch_normalization_544_beta_m_read_readvariableop2savev2_adam_dense_602_kernel_m_read_readvariableop0savev2_adam_dense_602_bias_m_read_readvariableop?savev2_adam_batch_normalization_545_gamma_m_read_readvariableop>savev2_adam_batch_normalization_545_beta_m_read_readvariableop2savev2_adam_dense_603_kernel_m_read_readvariableop0savev2_adam_dense_603_bias_m_read_readvariableop?savev2_adam_batch_normalization_546_gamma_m_read_readvariableop>savev2_adam_batch_normalization_546_beta_m_read_readvariableop2savev2_adam_dense_604_kernel_m_read_readvariableop0savev2_adam_dense_604_bias_m_read_readvariableop?savev2_adam_batch_normalization_547_gamma_m_read_readvariableop>savev2_adam_batch_normalization_547_beta_m_read_readvariableop2savev2_adam_dense_605_kernel_m_read_readvariableop0savev2_adam_dense_605_bias_m_read_readvariableop2savev2_adam_dense_599_kernel_v_read_readvariableop0savev2_adam_dense_599_bias_v_read_readvariableop?savev2_adam_batch_normalization_542_gamma_v_read_readvariableop>savev2_adam_batch_normalization_542_beta_v_read_readvariableop2savev2_adam_dense_600_kernel_v_read_readvariableop0savev2_adam_dense_600_bias_v_read_readvariableop?savev2_adam_batch_normalization_543_gamma_v_read_readvariableop>savev2_adam_batch_normalization_543_beta_v_read_readvariableop2savev2_adam_dense_601_kernel_v_read_readvariableop0savev2_adam_dense_601_bias_v_read_readvariableop?savev2_adam_batch_normalization_544_gamma_v_read_readvariableop>savev2_adam_batch_normalization_544_beta_v_read_readvariableop2savev2_adam_dense_602_kernel_v_read_readvariableop0savev2_adam_dense_602_bias_v_read_readvariableop?savev2_adam_batch_normalization_545_gamma_v_read_readvariableop>savev2_adam_batch_normalization_545_beta_v_read_readvariableop2savev2_adam_dense_603_kernel_v_read_readvariableop0savev2_adam_dense_603_bias_v_read_readvariableop?savev2_adam_batch_normalization_546_gamma_v_read_readvariableop>savev2_adam_batch_normalization_546_beta_v_read_readvariableop2savev2_adam_dense_604_kernel_v_read_readvariableop0savev2_adam_dense_604_bias_v_read_readvariableop?savev2_adam_batch_normalization_547_gamma_v_read_readvariableop>savev2_adam_batch_normalization_547_beta_v_read_readvariableop2savev2_adam_dense_605_kernel_v_read_readvariableop0savev2_adam_dense_605_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
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
?: ::: :J:J:J:J:J:J:JJ:J:J:J:J:J:J,:,:,:,:,:,:,,:,:,:,:,:,:,2:2:2:2:2:2:22:2:2:2:2:2:2:: : : : : : :J:J:J:J:JJ:J:J:J:J,:,:,:,:,,:,:,:,:,2:2:2:2:22:2:2:2:2::J:J:J:J:JJ:J:J:J:J,:,:,:,:,,:,:,:,:,2:2:2:2:22:2:2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :$ 

_output_shapes

:J: 

_output_shapes
:J: 

_output_shapes
:J: 

_output_shapes
:J: 

_output_shapes
:J: 	

_output_shapes
:J:$
 

_output_shapes

:JJ: 

_output_shapes
:J: 

_output_shapes
:J: 

_output_shapes
:J: 

_output_shapes
:J: 

_output_shapes
:J:$ 

_output_shapes

:J,: 

_output_shapes
:,: 

_output_shapes
:,: 

_output_shapes
:,: 

_output_shapes
:,: 

_output_shapes
:,:$ 

_output_shapes

:,,: 

_output_shapes
:,: 

_output_shapes
:,: 

_output_shapes
:,: 

_output_shapes
:,: 

_output_shapes
:,:$ 

_output_shapes

:,2: 

_output_shapes
:2: 

_output_shapes
:2: 

_output_shapes
:2:  

_output_shapes
:2: !

_output_shapes
:2:$" 

_output_shapes

:22: #

_output_shapes
:2: $

_output_shapes
:2: %

_output_shapes
:2: &

_output_shapes
:2: '

_output_shapes
:2:$( 

_output_shapes

:2: )
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

:J: 1

_output_shapes
:J: 2

_output_shapes
:J: 3

_output_shapes
:J:$4 

_output_shapes

:JJ: 5

_output_shapes
:J: 6

_output_shapes
:J: 7

_output_shapes
:J:$8 

_output_shapes

:J,: 9

_output_shapes
:,: :

_output_shapes
:,: ;

_output_shapes
:,:$< 

_output_shapes

:,,: =

_output_shapes
:,: >

_output_shapes
:,: ?

_output_shapes
:,:$@ 

_output_shapes

:,2: A

_output_shapes
:2: B

_output_shapes
:2: C

_output_shapes
:2:$D 

_output_shapes

:22: E

_output_shapes
:2: F

_output_shapes
:2: G

_output_shapes
:2:$H 

_output_shapes

:2: I

_output_shapes
::$J 

_output_shapes

:J: K

_output_shapes
:J: L

_output_shapes
:J: M

_output_shapes
:J:$N 

_output_shapes

:JJ: O

_output_shapes
:J: P

_output_shapes
:J: Q

_output_shapes
:J:$R 

_output_shapes

:J,: S

_output_shapes
:,: T

_output_shapes
:,: U

_output_shapes
:,:$V 

_output_shapes

:,,: W

_output_shapes
:,: X

_output_shapes
:,: Y

_output_shapes
:,:$Z 

_output_shapes

:,2: [

_output_shapes
:2: \

_output_shapes
:2: ]

_output_shapes
:2:$^ 

_output_shapes

:22: _

_output_shapes
:2: `

_output_shapes
:2: a

_output_shapes
:2:$b 

_output_shapes

:2: c

_output_shapes
::d

_output_shapes
: 
?	
?
E__inference_dense_602_layer_call_and_return_conditional_losses_774902

inputs0
matmul_readvariableop_resource:,,-
biasadd_readvariableop_resource:,
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:,,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:,*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????,w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????,: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_542_layer_call_and_return_conditional_losses_774826

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????J*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????J"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????J:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_774607

inputs5
'assignmovingavg_readvariableop_resource:,7
)assignmovingavg_1_readvariableop_resource:,3
%batchnorm_mul_readvariableop_resource:,/
!batchnorm_readvariableop_resource:,
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:,*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:,?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????,l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:,*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:,*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:,*
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
:,*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:,x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:,?
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
:,*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:,~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:,?
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
:,P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:,~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:,v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:,*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:,r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????,?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
??
?(
I__inference_sequential_57_layer_call_and_return_conditional_losses_776335

inputs
normalization_57_sub_y
normalization_57_sqrt_x:
(dense_599_matmul_readvariableop_resource:J7
)dense_599_biasadd_readvariableop_resource:JM
?batch_normalization_542_assignmovingavg_readvariableop_resource:JO
Abatch_normalization_542_assignmovingavg_1_readvariableop_resource:JK
=batch_normalization_542_batchnorm_mul_readvariableop_resource:JG
9batch_normalization_542_batchnorm_readvariableop_resource:J:
(dense_600_matmul_readvariableop_resource:JJ7
)dense_600_biasadd_readvariableop_resource:JM
?batch_normalization_543_assignmovingavg_readvariableop_resource:JO
Abatch_normalization_543_assignmovingavg_1_readvariableop_resource:JK
=batch_normalization_543_batchnorm_mul_readvariableop_resource:JG
9batch_normalization_543_batchnorm_readvariableop_resource:J:
(dense_601_matmul_readvariableop_resource:J,7
)dense_601_biasadd_readvariableop_resource:,M
?batch_normalization_544_assignmovingavg_readvariableop_resource:,O
Abatch_normalization_544_assignmovingavg_1_readvariableop_resource:,K
=batch_normalization_544_batchnorm_mul_readvariableop_resource:,G
9batch_normalization_544_batchnorm_readvariableop_resource:,:
(dense_602_matmul_readvariableop_resource:,,7
)dense_602_biasadd_readvariableop_resource:,M
?batch_normalization_545_assignmovingavg_readvariableop_resource:,O
Abatch_normalization_545_assignmovingavg_1_readvariableop_resource:,K
=batch_normalization_545_batchnorm_mul_readvariableop_resource:,G
9batch_normalization_545_batchnorm_readvariableop_resource:,:
(dense_603_matmul_readvariableop_resource:,27
)dense_603_biasadd_readvariableop_resource:2M
?batch_normalization_546_assignmovingavg_readvariableop_resource:2O
Abatch_normalization_546_assignmovingavg_1_readvariableop_resource:2K
=batch_normalization_546_batchnorm_mul_readvariableop_resource:2G
9batch_normalization_546_batchnorm_readvariableop_resource:2:
(dense_604_matmul_readvariableop_resource:227
)dense_604_biasadd_readvariableop_resource:2M
?batch_normalization_547_assignmovingavg_readvariableop_resource:2O
Abatch_normalization_547_assignmovingavg_1_readvariableop_resource:2K
=batch_normalization_547_batchnorm_mul_readvariableop_resource:2G
9batch_normalization_547_batchnorm_readvariableop_resource:2:
(dense_605_matmul_readvariableop_resource:27
)dense_605_biasadd_readvariableop_resource:
identity??'batch_normalization_542/AssignMovingAvg?6batch_normalization_542/AssignMovingAvg/ReadVariableOp?)batch_normalization_542/AssignMovingAvg_1?8batch_normalization_542/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_542/batchnorm/ReadVariableOp?4batch_normalization_542/batchnorm/mul/ReadVariableOp?'batch_normalization_543/AssignMovingAvg?6batch_normalization_543/AssignMovingAvg/ReadVariableOp?)batch_normalization_543/AssignMovingAvg_1?8batch_normalization_543/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_543/batchnorm/ReadVariableOp?4batch_normalization_543/batchnorm/mul/ReadVariableOp?'batch_normalization_544/AssignMovingAvg?6batch_normalization_544/AssignMovingAvg/ReadVariableOp?)batch_normalization_544/AssignMovingAvg_1?8batch_normalization_544/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_544/batchnorm/ReadVariableOp?4batch_normalization_544/batchnorm/mul/ReadVariableOp?'batch_normalization_545/AssignMovingAvg?6batch_normalization_545/AssignMovingAvg/ReadVariableOp?)batch_normalization_545/AssignMovingAvg_1?8batch_normalization_545/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_545/batchnorm/ReadVariableOp?4batch_normalization_545/batchnorm/mul/ReadVariableOp?'batch_normalization_546/AssignMovingAvg?6batch_normalization_546/AssignMovingAvg/ReadVariableOp?)batch_normalization_546/AssignMovingAvg_1?8batch_normalization_546/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_546/batchnorm/ReadVariableOp?4batch_normalization_546/batchnorm/mul/ReadVariableOp?'batch_normalization_547/AssignMovingAvg?6batch_normalization_547/AssignMovingAvg/ReadVariableOp?)batch_normalization_547/AssignMovingAvg_1?8batch_normalization_547/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_547/batchnorm/ReadVariableOp?4batch_normalization_547/batchnorm/mul/ReadVariableOp? dense_599/BiasAdd/ReadVariableOp?dense_599/MatMul/ReadVariableOp? dense_600/BiasAdd/ReadVariableOp?dense_600/MatMul/ReadVariableOp? dense_601/BiasAdd/ReadVariableOp?dense_601/MatMul/ReadVariableOp? dense_602/BiasAdd/ReadVariableOp?dense_602/MatMul/ReadVariableOp? dense_603/BiasAdd/ReadVariableOp?dense_603/MatMul/ReadVariableOp? dense_604/BiasAdd/ReadVariableOp?dense_604/MatMul/ReadVariableOp? dense_605/BiasAdd/ReadVariableOp?dense_605/MatMul/ReadVariableOpm
normalization_57/subSubinputsnormalization_57_sub_y*
T0*'
_output_shapes
:?????????_
normalization_57/SqrtSqrtnormalization_57_sqrt_x*
T0*
_output_shapes

:_
normalization_57/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_57/MaximumMaximumnormalization_57/Sqrt:y:0#normalization_57/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_57/truedivRealDivnormalization_57/sub:z:0normalization_57/Maximum:z:0*
T0*'
_output_shapes
:??????????
dense_599/MatMul/ReadVariableOpReadVariableOp(dense_599_matmul_readvariableop_resource*
_output_shapes

:J*
dtype0?
dense_599/MatMulMatMulnormalization_57/truediv:z:0'dense_599/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J?
 dense_599/BiasAdd/ReadVariableOpReadVariableOp)dense_599_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype0?
dense_599/BiasAddBiasAdddense_599/MatMul:product:0(dense_599/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J?
6batch_normalization_542/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization_542/moments/meanMeandense_599/BiasAdd:output:0?batch_normalization_542/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:J*
	keep_dims(?
,batch_normalization_542/moments/StopGradientStopGradient-batch_normalization_542/moments/mean:output:0*
T0*
_output_shapes

:J?
1batch_normalization_542/moments/SquaredDifferenceSquaredDifferencedense_599/BiasAdd:output:05batch_normalization_542/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????J?
:batch_normalization_542/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
(batch_normalization_542/moments/varianceMean5batch_normalization_542/moments/SquaredDifference:z:0Cbatch_normalization_542/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:J*
	keep_dims(?
'batch_normalization_542/moments/SqueezeSqueeze-batch_normalization_542/moments/mean:output:0*
T0*
_output_shapes
:J*
squeeze_dims
 ?
)batch_normalization_542/moments/Squeeze_1Squeeze1batch_normalization_542/moments/variance:output:0*
T0*
_output_shapes
:J*
squeeze_dims
 r
-batch_normalization_542/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_542/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_542_assignmovingavg_readvariableop_resource*
_output_shapes
:J*
dtype0?
+batch_normalization_542/AssignMovingAvg/subSub>batch_normalization_542/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_542/moments/Squeeze:output:0*
T0*
_output_shapes
:J?
+batch_normalization_542/AssignMovingAvg/mulMul/batch_normalization_542/AssignMovingAvg/sub:z:06batch_normalization_542/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:J?
'batch_normalization_542/AssignMovingAvgAssignSubVariableOp?batch_normalization_542_assignmovingavg_readvariableop_resource/batch_normalization_542/AssignMovingAvg/mul:z:07^batch_normalization_542/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_542/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8batch_normalization_542/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_542_assignmovingavg_1_readvariableop_resource*
_output_shapes
:J*
dtype0?
-batch_normalization_542/AssignMovingAvg_1/subSub@batch_normalization_542/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_542/moments/Squeeze_1:output:0*
T0*
_output_shapes
:J?
-batch_normalization_542/AssignMovingAvg_1/mulMul1batch_normalization_542/AssignMovingAvg_1/sub:z:08batch_normalization_542/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:J?
)batch_normalization_542/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_542_assignmovingavg_1_readvariableop_resource1batch_normalization_542/AssignMovingAvg_1/mul:z:09^batch_normalization_542/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_542/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_542/batchnorm/addAddV22batch_normalization_542/moments/Squeeze_1:output:00batch_normalization_542/batchnorm/add/y:output:0*
T0*
_output_shapes
:J?
'batch_normalization_542/batchnorm/RsqrtRsqrt)batch_normalization_542/batchnorm/add:z:0*
T0*
_output_shapes
:J?
4batch_normalization_542/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_542_batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0?
%batch_normalization_542/batchnorm/mulMul+batch_normalization_542/batchnorm/Rsqrt:y:0<batch_normalization_542/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:J?
'batch_normalization_542/batchnorm/mul_1Muldense_599/BiasAdd:output:0)batch_normalization_542/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????J?
'batch_normalization_542/batchnorm/mul_2Mul0batch_normalization_542/moments/Squeeze:output:0)batch_normalization_542/batchnorm/mul:z:0*
T0*
_output_shapes
:J?
0batch_normalization_542/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_542_batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0?
%batch_normalization_542/batchnorm/subSub8batch_normalization_542/batchnorm/ReadVariableOp:value:0+batch_normalization_542/batchnorm/mul_2:z:0*
T0*
_output_shapes
:J?
'batch_normalization_542/batchnorm/add_1AddV2+batch_normalization_542/batchnorm/mul_1:z:0)batch_normalization_542/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????J?
leaky_re_lu_542/LeakyRelu	LeakyRelu+batch_normalization_542/batchnorm/add_1:z:0*'
_output_shapes
:?????????J*
alpha%???>?
dense_600/MatMul/ReadVariableOpReadVariableOp(dense_600_matmul_readvariableop_resource*
_output_shapes

:JJ*
dtype0?
dense_600/MatMulMatMul'leaky_re_lu_542/LeakyRelu:activations:0'dense_600/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J?
 dense_600/BiasAdd/ReadVariableOpReadVariableOp)dense_600_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype0?
dense_600/BiasAddBiasAdddense_600/MatMul:product:0(dense_600/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J?
6batch_normalization_543/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization_543/moments/meanMeandense_600/BiasAdd:output:0?batch_normalization_543/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:J*
	keep_dims(?
,batch_normalization_543/moments/StopGradientStopGradient-batch_normalization_543/moments/mean:output:0*
T0*
_output_shapes

:J?
1batch_normalization_543/moments/SquaredDifferenceSquaredDifferencedense_600/BiasAdd:output:05batch_normalization_543/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????J?
:batch_normalization_543/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
(batch_normalization_543/moments/varianceMean5batch_normalization_543/moments/SquaredDifference:z:0Cbatch_normalization_543/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:J*
	keep_dims(?
'batch_normalization_543/moments/SqueezeSqueeze-batch_normalization_543/moments/mean:output:0*
T0*
_output_shapes
:J*
squeeze_dims
 ?
)batch_normalization_543/moments/Squeeze_1Squeeze1batch_normalization_543/moments/variance:output:0*
T0*
_output_shapes
:J*
squeeze_dims
 r
-batch_normalization_543/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_543/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_543_assignmovingavg_readvariableop_resource*
_output_shapes
:J*
dtype0?
+batch_normalization_543/AssignMovingAvg/subSub>batch_normalization_543/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_543/moments/Squeeze:output:0*
T0*
_output_shapes
:J?
+batch_normalization_543/AssignMovingAvg/mulMul/batch_normalization_543/AssignMovingAvg/sub:z:06batch_normalization_543/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:J?
'batch_normalization_543/AssignMovingAvgAssignSubVariableOp?batch_normalization_543_assignmovingavg_readvariableop_resource/batch_normalization_543/AssignMovingAvg/mul:z:07^batch_normalization_543/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_543/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8batch_normalization_543/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_543_assignmovingavg_1_readvariableop_resource*
_output_shapes
:J*
dtype0?
-batch_normalization_543/AssignMovingAvg_1/subSub@batch_normalization_543/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_543/moments/Squeeze_1:output:0*
T0*
_output_shapes
:J?
-batch_normalization_543/AssignMovingAvg_1/mulMul1batch_normalization_543/AssignMovingAvg_1/sub:z:08batch_normalization_543/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:J?
)batch_normalization_543/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_543_assignmovingavg_1_readvariableop_resource1batch_normalization_543/AssignMovingAvg_1/mul:z:09^batch_normalization_543/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_543/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_543/batchnorm/addAddV22batch_normalization_543/moments/Squeeze_1:output:00batch_normalization_543/batchnorm/add/y:output:0*
T0*
_output_shapes
:J?
'batch_normalization_543/batchnorm/RsqrtRsqrt)batch_normalization_543/batchnorm/add:z:0*
T0*
_output_shapes
:J?
4batch_normalization_543/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_543_batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0?
%batch_normalization_543/batchnorm/mulMul+batch_normalization_543/batchnorm/Rsqrt:y:0<batch_normalization_543/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:J?
'batch_normalization_543/batchnorm/mul_1Muldense_600/BiasAdd:output:0)batch_normalization_543/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????J?
'batch_normalization_543/batchnorm/mul_2Mul0batch_normalization_543/moments/Squeeze:output:0)batch_normalization_543/batchnorm/mul:z:0*
T0*
_output_shapes
:J?
0batch_normalization_543/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_543_batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0?
%batch_normalization_543/batchnorm/subSub8batch_normalization_543/batchnorm/ReadVariableOp:value:0+batch_normalization_543/batchnorm/mul_2:z:0*
T0*
_output_shapes
:J?
'batch_normalization_543/batchnorm/add_1AddV2+batch_normalization_543/batchnorm/mul_1:z:0)batch_normalization_543/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????J?
leaky_re_lu_543/LeakyRelu	LeakyRelu+batch_normalization_543/batchnorm/add_1:z:0*'
_output_shapes
:?????????J*
alpha%???>?
dense_601/MatMul/ReadVariableOpReadVariableOp(dense_601_matmul_readvariableop_resource*
_output_shapes

:J,*
dtype0?
dense_601/MatMulMatMul'leaky_re_lu_543/LeakyRelu:activations:0'dense_601/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,?
 dense_601/BiasAdd/ReadVariableOpReadVariableOp)dense_601_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0?
dense_601/BiasAddBiasAdddense_601/MatMul:product:0(dense_601/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,?
6batch_normalization_544/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization_544/moments/meanMeandense_601/BiasAdd:output:0?batch_normalization_544/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:,*
	keep_dims(?
,batch_normalization_544/moments/StopGradientStopGradient-batch_normalization_544/moments/mean:output:0*
T0*
_output_shapes

:,?
1batch_normalization_544/moments/SquaredDifferenceSquaredDifferencedense_601/BiasAdd:output:05batch_normalization_544/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????,?
:batch_normalization_544/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
(batch_normalization_544/moments/varianceMean5batch_normalization_544/moments/SquaredDifference:z:0Cbatch_normalization_544/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:,*
	keep_dims(?
'batch_normalization_544/moments/SqueezeSqueeze-batch_normalization_544/moments/mean:output:0*
T0*
_output_shapes
:,*
squeeze_dims
 ?
)batch_normalization_544/moments/Squeeze_1Squeeze1batch_normalization_544/moments/variance:output:0*
T0*
_output_shapes
:,*
squeeze_dims
 r
-batch_normalization_544/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_544/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_544_assignmovingavg_readvariableop_resource*
_output_shapes
:,*
dtype0?
+batch_normalization_544/AssignMovingAvg/subSub>batch_normalization_544/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_544/moments/Squeeze:output:0*
T0*
_output_shapes
:,?
+batch_normalization_544/AssignMovingAvg/mulMul/batch_normalization_544/AssignMovingAvg/sub:z:06batch_normalization_544/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:,?
'batch_normalization_544/AssignMovingAvgAssignSubVariableOp?batch_normalization_544_assignmovingavg_readvariableop_resource/batch_normalization_544/AssignMovingAvg/mul:z:07^batch_normalization_544/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_544/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8batch_normalization_544/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_544_assignmovingavg_1_readvariableop_resource*
_output_shapes
:,*
dtype0?
-batch_normalization_544/AssignMovingAvg_1/subSub@batch_normalization_544/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_544/moments/Squeeze_1:output:0*
T0*
_output_shapes
:,?
-batch_normalization_544/AssignMovingAvg_1/mulMul1batch_normalization_544/AssignMovingAvg_1/sub:z:08batch_normalization_544/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:,?
)batch_normalization_544/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_544_assignmovingavg_1_readvariableop_resource1batch_normalization_544/AssignMovingAvg_1/mul:z:09^batch_normalization_544/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_544/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_544/batchnorm/addAddV22batch_normalization_544/moments/Squeeze_1:output:00batch_normalization_544/batchnorm/add/y:output:0*
T0*
_output_shapes
:,?
'batch_normalization_544/batchnorm/RsqrtRsqrt)batch_normalization_544/batchnorm/add:z:0*
T0*
_output_shapes
:,?
4batch_normalization_544/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_544_batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0?
%batch_normalization_544/batchnorm/mulMul+batch_normalization_544/batchnorm/Rsqrt:y:0<batch_normalization_544/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,?
'batch_normalization_544/batchnorm/mul_1Muldense_601/BiasAdd:output:0)batch_normalization_544/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,?
'batch_normalization_544/batchnorm/mul_2Mul0batch_normalization_544/moments/Squeeze:output:0)batch_normalization_544/batchnorm/mul:z:0*
T0*
_output_shapes
:,?
0batch_normalization_544/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_544_batchnorm_readvariableop_resource*
_output_shapes
:,*
dtype0?
%batch_normalization_544/batchnorm/subSub8batch_normalization_544/batchnorm/ReadVariableOp:value:0+batch_normalization_544/batchnorm/mul_2:z:0*
T0*
_output_shapes
:,?
'batch_normalization_544/batchnorm/add_1AddV2+batch_normalization_544/batchnorm/mul_1:z:0)batch_normalization_544/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,?
leaky_re_lu_544/LeakyRelu	LeakyRelu+batch_normalization_544/batchnorm/add_1:z:0*'
_output_shapes
:?????????,*
alpha%???>?
dense_602/MatMul/ReadVariableOpReadVariableOp(dense_602_matmul_readvariableop_resource*
_output_shapes

:,,*
dtype0?
dense_602/MatMulMatMul'leaky_re_lu_544/LeakyRelu:activations:0'dense_602/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,?
 dense_602/BiasAdd/ReadVariableOpReadVariableOp)dense_602_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0?
dense_602/BiasAddBiasAdddense_602/MatMul:product:0(dense_602/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,?
6batch_normalization_545/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization_545/moments/meanMeandense_602/BiasAdd:output:0?batch_normalization_545/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:,*
	keep_dims(?
,batch_normalization_545/moments/StopGradientStopGradient-batch_normalization_545/moments/mean:output:0*
T0*
_output_shapes

:,?
1batch_normalization_545/moments/SquaredDifferenceSquaredDifferencedense_602/BiasAdd:output:05batch_normalization_545/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????,?
:batch_normalization_545/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
(batch_normalization_545/moments/varianceMean5batch_normalization_545/moments/SquaredDifference:z:0Cbatch_normalization_545/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:,*
	keep_dims(?
'batch_normalization_545/moments/SqueezeSqueeze-batch_normalization_545/moments/mean:output:0*
T0*
_output_shapes
:,*
squeeze_dims
 ?
)batch_normalization_545/moments/Squeeze_1Squeeze1batch_normalization_545/moments/variance:output:0*
T0*
_output_shapes
:,*
squeeze_dims
 r
-batch_normalization_545/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_545/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_545_assignmovingavg_readvariableop_resource*
_output_shapes
:,*
dtype0?
+batch_normalization_545/AssignMovingAvg/subSub>batch_normalization_545/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_545/moments/Squeeze:output:0*
T0*
_output_shapes
:,?
+batch_normalization_545/AssignMovingAvg/mulMul/batch_normalization_545/AssignMovingAvg/sub:z:06batch_normalization_545/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:,?
'batch_normalization_545/AssignMovingAvgAssignSubVariableOp?batch_normalization_545_assignmovingavg_readvariableop_resource/batch_normalization_545/AssignMovingAvg/mul:z:07^batch_normalization_545/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_545/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8batch_normalization_545/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_545_assignmovingavg_1_readvariableop_resource*
_output_shapes
:,*
dtype0?
-batch_normalization_545/AssignMovingAvg_1/subSub@batch_normalization_545/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_545/moments/Squeeze_1:output:0*
T0*
_output_shapes
:,?
-batch_normalization_545/AssignMovingAvg_1/mulMul1batch_normalization_545/AssignMovingAvg_1/sub:z:08batch_normalization_545/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:,?
)batch_normalization_545/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_545_assignmovingavg_1_readvariableop_resource1batch_normalization_545/AssignMovingAvg_1/mul:z:09^batch_normalization_545/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_545/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_545/batchnorm/addAddV22batch_normalization_545/moments/Squeeze_1:output:00batch_normalization_545/batchnorm/add/y:output:0*
T0*
_output_shapes
:,?
'batch_normalization_545/batchnorm/RsqrtRsqrt)batch_normalization_545/batchnorm/add:z:0*
T0*
_output_shapes
:,?
4batch_normalization_545/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_545_batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0?
%batch_normalization_545/batchnorm/mulMul+batch_normalization_545/batchnorm/Rsqrt:y:0<batch_normalization_545/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,?
'batch_normalization_545/batchnorm/mul_1Muldense_602/BiasAdd:output:0)batch_normalization_545/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,?
'batch_normalization_545/batchnorm/mul_2Mul0batch_normalization_545/moments/Squeeze:output:0)batch_normalization_545/batchnorm/mul:z:0*
T0*
_output_shapes
:,?
0batch_normalization_545/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_545_batchnorm_readvariableop_resource*
_output_shapes
:,*
dtype0?
%batch_normalization_545/batchnorm/subSub8batch_normalization_545/batchnorm/ReadVariableOp:value:0+batch_normalization_545/batchnorm/mul_2:z:0*
T0*
_output_shapes
:,?
'batch_normalization_545/batchnorm/add_1AddV2+batch_normalization_545/batchnorm/mul_1:z:0)batch_normalization_545/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,?
leaky_re_lu_545/LeakyRelu	LeakyRelu+batch_normalization_545/batchnorm/add_1:z:0*'
_output_shapes
:?????????,*
alpha%???>?
dense_603/MatMul/ReadVariableOpReadVariableOp(dense_603_matmul_readvariableop_resource*
_output_shapes

:,2*
dtype0?
dense_603/MatMulMatMul'leaky_re_lu_545/LeakyRelu:activations:0'dense_603/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
 dense_603/BiasAdd/ReadVariableOpReadVariableOp)dense_603_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_603/BiasAddBiasAdddense_603/MatMul:product:0(dense_603/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
6batch_normalization_546/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization_546/moments/meanMeandense_603/BiasAdd:output:0?batch_normalization_546/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(?
,batch_normalization_546/moments/StopGradientStopGradient-batch_normalization_546/moments/mean:output:0*
T0*
_output_shapes

:2?
1batch_normalization_546/moments/SquaredDifferenceSquaredDifferencedense_603/BiasAdd:output:05batch_normalization_546/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2?
:batch_normalization_546/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
(batch_normalization_546/moments/varianceMean5batch_normalization_546/moments/SquaredDifference:z:0Cbatch_normalization_546/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(?
'batch_normalization_546/moments/SqueezeSqueeze-batch_normalization_546/moments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 ?
)batch_normalization_546/moments/Squeeze_1Squeeze1batch_normalization_546/moments/variance:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 r
-batch_normalization_546/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_546/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_546_assignmovingavg_readvariableop_resource*
_output_shapes
:2*
dtype0?
+batch_normalization_546/AssignMovingAvg/subSub>batch_normalization_546/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_546/moments/Squeeze:output:0*
T0*
_output_shapes
:2?
+batch_normalization_546/AssignMovingAvg/mulMul/batch_normalization_546/AssignMovingAvg/sub:z:06batch_normalization_546/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2?
'batch_normalization_546/AssignMovingAvgAssignSubVariableOp?batch_normalization_546_assignmovingavg_readvariableop_resource/batch_normalization_546/AssignMovingAvg/mul:z:07^batch_normalization_546/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_546/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8batch_normalization_546/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_546_assignmovingavg_1_readvariableop_resource*
_output_shapes
:2*
dtype0?
-batch_normalization_546/AssignMovingAvg_1/subSub@batch_normalization_546/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_546/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2?
-batch_normalization_546/AssignMovingAvg_1/mulMul1batch_normalization_546/AssignMovingAvg_1/sub:z:08batch_normalization_546/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2?
)batch_normalization_546/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_546_assignmovingavg_1_readvariableop_resource1batch_normalization_546/AssignMovingAvg_1/mul:z:09^batch_normalization_546/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_546/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_546/batchnorm/addAddV22batch_normalization_546/moments/Squeeze_1:output:00batch_normalization_546/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
'batch_normalization_546/batchnorm/RsqrtRsqrt)batch_normalization_546/batchnorm/add:z:0*
T0*
_output_shapes
:2?
4batch_normalization_546/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_546_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0?
%batch_normalization_546/batchnorm/mulMul+batch_normalization_546/batchnorm/Rsqrt:y:0<batch_normalization_546/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
'batch_normalization_546/batchnorm/mul_1Muldense_603/BiasAdd:output:0)batch_normalization_546/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2?
'batch_normalization_546/batchnorm/mul_2Mul0batch_normalization_546/moments/Squeeze:output:0)batch_normalization_546/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
0batch_normalization_546/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_546_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0?
%batch_normalization_546/batchnorm/subSub8batch_normalization_546/batchnorm/ReadVariableOp:value:0+batch_normalization_546/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
'batch_normalization_546/batchnorm/add_1AddV2+batch_normalization_546/batchnorm/mul_1:z:0)batch_normalization_546/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2?
leaky_re_lu_546/LeakyRelu	LeakyRelu+batch_normalization_546/batchnorm/add_1:z:0*'
_output_shapes
:?????????2*
alpha%???>?
dense_604/MatMul/ReadVariableOpReadVariableOp(dense_604_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_604/MatMulMatMul'leaky_re_lu_546/LeakyRelu:activations:0'dense_604/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
 dense_604/BiasAdd/ReadVariableOpReadVariableOp)dense_604_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_604/BiasAddBiasAdddense_604/MatMul:product:0(dense_604/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
6batch_normalization_547/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization_547/moments/meanMeandense_604/BiasAdd:output:0?batch_normalization_547/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(?
,batch_normalization_547/moments/StopGradientStopGradient-batch_normalization_547/moments/mean:output:0*
T0*
_output_shapes

:2?
1batch_normalization_547/moments/SquaredDifferenceSquaredDifferencedense_604/BiasAdd:output:05batch_normalization_547/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2?
:batch_normalization_547/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
(batch_normalization_547/moments/varianceMean5batch_normalization_547/moments/SquaredDifference:z:0Cbatch_normalization_547/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(?
'batch_normalization_547/moments/SqueezeSqueeze-batch_normalization_547/moments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 ?
)batch_normalization_547/moments/Squeeze_1Squeeze1batch_normalization_547/moments/variance:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 r
-batch_normalization_547/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_547/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_547_assignmovingavg_readvariableop_resource*
_output_shapes
:2*
dtype0?
+batch_normalization_547/AssignMovingAvg/subSub>batch_normalization_547/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_547/moments/Squeeze:output:0*
T0*
_output_shapes
:2?
+batch_normalization_547/AssignMovingAvg/mulMul/batch_normalization_547/AssignMovingAvg/sub:z:06batch_normalization_547/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2?
'batch_normalization_547/AssignMovingAvgAssignSubVariableOp?batch_normalization_547_assignmovingavg_readvariableop_resource/batch_normalization_547/AssignMovingAvg/mul:z:07^batch_normalization_547/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_547/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8batch_normalization_547/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_547_assignmovingavg_1_readvariableop_resource*
_output_shapes
:2*
dtype0?
-batch_normalization_547/AssignMovingAvg_1/subSub@batch_normalization_547/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_547/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2?
-batch_normalization_547/AssignMovingAvg_1/mulMul1batch_normalization_547/AssignMovingAvg_1/sub:z:08batch_normalization_547/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2?
)batch_normalization_547/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_547_assignmovingavg_1_readvariableop_resource1batch_normalization_547/AssignMovingAvg_1/mul:z:09^batch_normalization_547/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_547/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
%batch_normalization_547/batchnorm/addAddV22batch_normalization_547/moments/Squeeze_1:output:00batch_normalization_547/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
'batch_normalization_547/batchnorm/RsqrtRsqrt)batch_normalization_547/batchnorm/add:z:0*
T0*
_output_shapes
:2?
4batch_normalization_547/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_547_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0?
%batch_normalization_547/batchnorm/mulMul+batch_normalization_547/batchnorm/Rsqrt:y:0<batch_normalization_547/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
'batch_normalization_547/batchnorm/mul_1Muldense_604/BiasAdd:output:0)batch_normalization_547/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2?
'batch_normalization_547/batchnorm/mul_2Mul0batch_normalization_547/moments/Squeeze:output:0)batch_normalization_547/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
0batch_normalization_547/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_547_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0?
%batch_normalization_547/batchnorm/subSub8batch_normalization_547/batchnorm/ReadVariableOp:value:0+batch_normalization_547/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
'batch_normalization_547/batchnorm/add_1AddV2+batch_normalization_547/batchnorm/mul_1:z:0)batch_normalization_547/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2?
leaky_re_lu_547/LeakyRelu	LeakyRelu+batch_normalization_547/batchnorm/add_1:z:0*'
_output_shapes
:?????????2*
alpha%???>?
dense_605/MatMul/ReadVariableOpReadVariableOp(dense_605_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_605/MatMulMatMul'leaky_re_lu_547/LeakyRelu:activations:0'dense_605/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_605/BiasAdd/ReadVariableOpReadVariableOp)dense_605_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_605/BiasAddBiasAdddense_605/MatMul:product:0(dense_605/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_605/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^batch_normalization_542/AssignMovingAvg7^batch_normalization_542/AssignMovingAvg/ReadVariableOp*^batch_normalization_542/AssignMovingAvg_19^batch_normalization_542/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_542/batchnorm/ReadVariableOp5^batch_normalization_542/batchnorm/mul/ReadVariableOp(^batch_normalization_543/AssignMovingAvg7^batch_normalization_543/AssignMovingAvg/ReadVariableOp*^batch_normalization_543/AssignMovingAvg_19^batch_normalization_543/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_543/batchnorm/ReadVariableOp5^batch_normalization_543/batchnorm/mul/ReadVariableOp(^batch_normalization_544/AssignMovingAvg7^batch_normalization_544/AssignMovingAvg/ReadVariableOp*^batch_normalization_544/AssignMovingAvg_19^batch_normalization_544/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_544/batchnorm/ReadVariableOp5^batch_normalization_544/batchnorm/mul/ReadVariableOp(^batch_normalization_545/AssignMovingAvg7^batch_normalization_545/AssignMovingAvg/ReadVariableOp*^batch_normalization_545/AssignMovingAvg_19^batch_normalization_545/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_545/batchnorm/ReadVariableOp5^batch_normalization_545/batchnorm/mul/ReadVariableOp(^batch_normalization_546/AssignMovingAvg7^batch_normalization_546/AssignMovingAvg/ReadVariableOp*^batch_normalization_546/AssignMovingAvg_19^batch_normalization_546/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_546/batchnorm/ReadVariableOp5^batch_normalization_546/batchnorm/mul/ReadVariableOp(^batch_normalization_547/AssignMovingAvg7^batch_normalization_547/AssignMovingAvg/ReadVariableOp*^batch_normalization_547/AssignMovingAvg_19^batch_normalization_547/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_547/batchnorm/ReadVariableOp5^batch_normalization_547/batchnorm/mul/ReadVariableOp!^dense_599/BiasAdd/ReadVariableOp ^dense_599/MatMul/ReadVariableOp!^dense_600/BiasAdd/ReadVariableOp ^dense_600/MatMul/ReadVariableOp!^dense_601/BiasAdd/ReadVariableOp ^dense_601/MatMul/ReadVariableOp!^dense_602/BiasAdd/ReadVariableOp ^dense_602/MatMul/ReadVariableOp!^dense_603/BiasAdd/ReadVariableOp ^dense_603/MatMul/ReadVariableOp!^dense_604/BiasAdd/ReadVariableOp ^dense_604/MatMul/ReadVariableOp!^dense_605/BiasAdd/ReadVariableOp ^dense_605/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_542/AssignMovingAvg'batch_normalization_542/AssignMovingAvg2p
6batch_normalization_542/AssignMovingAvg/ReadVariableOp6batch_normalization_542/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_542/AssignMovingAvg_1)batch_normalization_542/AssignMovingAvg_12t
8batch_normalization_542/AssignMovingAvg_1/ReadVariableOp8batch_normalization_542/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_542/batchnorm/ReadVariableOp0batch_normalization_542/batchnorm/ReadVariableOp2l
4batch_normalization_542/batchnorm/mul/ReadVariableOp4batch_normalization_542/batchnorm/mul/ReadVariableOp2R
'batch_normalization_543/AssignMovingAvg'batch_normalization_543/AssignMovingAvg2p
6batch_normalization_543/AssignMovingAvg/ReadVariableOp6batch_normalization_543/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_543/AssignMovingAvg_1)batch_normalization_543/AssignMovingAvg_12t
8batch_normalization_543/AssignMovingAvg_1/ReadVariableOp8batch_normalization_543/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_543/batchnorm/ReadVariableOp0batch_normalization_543/batchnorm/ReadVariableOp2l
4batch_normalization_543/batchnorm/mul/ReadVariableOp4batch_normalization_543/batchnorm/mul/ReadVariableOp2R
'batch_normalization_544/AssignMovingAvg'batch_normalization_544/AssignMovingAvg2p
6batch_normalization_544/AssignMovingAvg/ReadVariableOp6batch_normalization_544/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_544/AssignMovingAvg_1)batch_normalization_544/AssignMovingAvg_12t
8batch_normalization_544/AssignMovingAvg_1/ReadVariableOp8batch_normalization_544/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_544/batchnorm/ReadVariableOp0batch_normalization_544/batchnorm/ReadVariableOp2l
4batch_normalization_544/batchnorm/mul/ReadVariableOp4batch_normalization_544/batchnorm/mul/ReadVariableOp2R
'batch_normalization_545/AssignMovingAvg'batch_normalization_545/AssignMovingAvg2p
6batch_normalization_545/AssignMovingAvg/ReadVariableOp6batch_normalization_545/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_545/AssignMovingAvg_1)batch_normalization_545/AssignMovingAvg_12t
8batch_normalization_545/AssignMovingAvg_1/ReadVariableOp8batch_normalization_545/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_545/batchnorm/ReadVariableOp0batch_normalization_545/batchnorm/ReadVariableOp2l
4batch_normalization_545/batchnorm/mul/ReadVariableOp4batch_normalization_545/batchnorm/mul/ReadVariableOp2R
'batch_normalization_546/AssignMovingAvg'batch_normalization_546/AssignMovingAvg2p
6batch_normalization_546/AssignMovingAvg/ReadVariableOp6batch_normalization_546/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_546/AssignMovingAvg_1)batch_normalization_546/AssignMovingAvg_12t
8batch_normalization_546/AssignMovingAvg_1/ReadVariableOp8batch_normalization_546/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_546/batchnorm/ReadVariableOp0batch_normalization_546/batchnorm/ReadVariableOp2l
4batch_normalization_546/batchnorm/mul/ReadVariableOp4batch_normalization_546/batchnorm/mul/ReadVariableOp2R
'batch_normalization_547/AssignMovingAvg'batch_normalization_547/AssignMovingAvg2p
6batch_normalization_547/AssignMovingAvg/ReadVariableOp6batch_normalization_547/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_547/AssignMovingAvg_1)batch_normalization_547/AssignMovingAvg_12t
8batch_normalization_547/AssignMovingAvg_1/ReadVariableOp8batch_normalization_547/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_547/batchnorm/ReadVariableOp0batch_normalization_547/batchnorm/ReadVariableOp2l
4batch_normalization_547/batchnorm/mul/ReadVariableOp4batch_normalization_547/batchnorm/mul/ReadVariableOp2D
 dense_599/BiasAdd/ReadVariableOp dense_599/BiasAdd/ReadVariableOp2B
dense_599/MatMul/ReadVariableOpdense_599/MatMul/ReadVariableOp2D
 dense_600/BiasAdd/ReadVariableOp dense_600/BiasAdd/ReadVariableOp2B
dense_600/MatMul/ReadVariableOpdense_600/MatMul/ReadVariableOp2D
 dense_601/BiasAdd/ReadVariableOp dense_601/BiasAdd/ReadVariableOp2B
dense_601/MatMul/ReadVariableOpdense_601/MatMul/ReadVariableOp2D
 dense_602/BiasAdd/ReadVariableOp dense_602/BiasAdd/ReadVariableOp2B
dense_602/MatMul/ReadVariableOpdense_602/MatMul/ReadVariableOp2D
 dense_603/BiasAdd/ReadVariableOp dense_603/BiasAdd/ReadVariableOp2B
dense_603/MatMul/ReadVariableOpdense_603/MatMul/ReadVariableOp2D
 dense_604/BiasAdd/ReadVariableOp dense_604/BiasAdd/ReadVariableOp2B
dense_604/MatMul/ReadVariableOpdense_604/MatMul/ReadVariableOp2D
 dense_605/BiasAdd/ReadVariableOp dense_605/BiasAdd/ReadVariableOp2B
dense_605/MatMul/ReadVariableOpdense_605/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
?	
?
E__inference_dense_601_layer_call_and_return_conditional_losses_774870

inputs0
matmul_readvariableop_resource:J,-
biasadd_readvariableop_resource:,
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:J,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:,*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????,w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?	
?
E__inference_dense_603_layer_call_and_return_conditional_losses_774934

inputs0
matmul_readvariableop_resource:,2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:,2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????,: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_774314

inputs/
!batchnorm_readvariableop_resource:J3
%batchnorm_mul_readvariableop_resource:J1
#batchnorm_readvariableop_1_resource:J1
#batchnorm_readvariableop_2_resource:J
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:J*
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
:JP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:J~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Jc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????Jz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:J*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Jz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:J*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Jr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????Jb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????J?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?	
?
E__inference_dense_599_layer_call_and_return_conditional_losses_776488

inputs0
matmul_readvariableop_resource:J-
biasadd_readvariableop_resource:J
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:J*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Jr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Jw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_546_layer_call_fn_776937

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_774642o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_777113

inputs5
'assignmovingavg_readvariableop_resource:27
)assignmovingavg_1_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:2/
!batchnorm_readvariableop_resource:2
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:2*
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
:2*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2?
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
:2*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2?
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
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?'
?
__inference_adapt_step_776469
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
2k
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
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
:*
dtype0P
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:X
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:G
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0V
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0V
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:E
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:V
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @N
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:Z
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:I
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:I
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:?
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
*__inference_dense_605_layer_call_fn_777132

inputs
unknown:2
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
E__inference_dense_605_layer_call_and_return_conditional_losses_774998o
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
:?????????2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_547_layer_call_and_return_conditional_losses_774986

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????2*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
.__inference_sequential_57_layer_call_fn_775856

inputs
unknown
	unknown_0
	unknown_1:J
	unknown_2:J
	unknown_3:J
	unknown_4:J
	unknown_5:J
	unknown_6:J
	unknown_7:JJ
	unknown_8:J
	unknown_9:J

unknown_10:J

unknown_11:J

unknown_12:J

unknown_13:J,

unknown_14:,

unknown_15:,

unknown_16:,

unknown_17:,

unknown_18:,

unknown_19:,,

unknown_20:,

unknown_21:,

unknown_22:,

unknown_23:,

unknown_24:,

unknown_25:,2

unknown_26:2

unknown_27:2

unknown_28:2

unknown_29:2

unknown_30:2

unknown_31:22

unknown_32:2

unknown_33:2

unknown_34:2

unknown_35:2

unknown_36:2

unknown_37:2

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
I__inference_sequential_57_layer_call_and_return_conditional_losses_775005o
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
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
8__inference_batch_normalization_542_layer_call_fn_776514

inputs
unknown:J
	unknown_0:J
	unknown_1:J
	unknown_2:J
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_774361o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????J`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?	
?
E__inference_dense_603_layer_call_and_return_conditional_losses_776924

inputs0
matmul_readvariableop_resource:,2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:,2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????,: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?i
?
I__inference_sequential_57_layer_call_and_return_conditional_losses_775767
normalization_57_input
normalization_57_sub_y
normalization_57_sqrt_x"
dense_599_775671:J
dense_599_775673:J,
batch_normalization_542_775676:J,
batch_normalization_542_775678:J,
batch_normalization_542_775680:J,
batch_normalization_542_775682:J"
dense_600_775686:JJ
dense_600_775688:J,
batch_normalization_543_775691:J,
batch_normalization_543_775693:J,
batch_normalization_543_775695:J,
batch_normalization_543_775697:J"
dense_601_775701:J,
dense_601_775703:,,
batch_normalization_544_775706:,,
batch_normalization_544_775708:,,
batch_normalization_544_775710:,,
batch_normalization_544_775712:,"
dense_602_775716:,,
dense_602_775718:,,
batch_normalization_545_775721:,,
batch_normalization_545_775723:,,
batch_normalization_545_775725:,,
batch_normalization_545_775727:,"
dense_603_775731:,2
dense_603_775733:2,
batch_normalization_546_775736:2,
batch_normalization_546_775738:2,
batch_normalization_546_775740:2,
batch_normalization_546_775742:2"
dense_604_775746:22
dense_604_775748:2,
batch_normalization_547_775751:2,
batch_normalization_547_775753:2,
batch_normalization_547_775755:2,
batch_normalization_547_775757:2"
dense_605_775761:2
dense_605_775763:
identity??/batch_normalization_542/StatefulPartitionedCall?/batch_normalization_543/StatefulPartitionedCall?/batch_normalization_544/StatefulPartitionedCall?/batch_normalization_545/StatefulPartitionedCall?/batch_normalization_546/StatefulPartitionedCall?/batch_normalization_547/StatefulPartitionedCall?!dense_599/StatefulPartitionedCall?!dense_600/StatefulPartitionedCall?!dense_601/StatefulPartitionedCall?!dense_602/StatefulPartitionedCall?!dense_603/StatefulPartitionedCall?!dense_604/StatefulPartitionedCall?!dense_605/StatefulPartitionedCall}
normalization_57/subSubnormalization_57_inputnormalization_57_sub_y*
T0*'
_output_shapes
:?????????_
normalization_57/SqrtSqrtnormalization_57_sqrt_x*
T0*
_output_shapes

:_
normalization_57/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_57/MaximumMaximumnormalization_57/Sqrt:y:0#normalization_57/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_57/truedivRealDivnormalization_57/sub:z:0normalization_57/Maximum:z:0*
T0*'
_output_shapes
:??????????
!dense_599/StatefulPartitionedCallStatefulPartitionedCallnormalization_57/truediv:z:0dense_599_775671dense_599_775673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_599_layer_call_and_return_conditional_losses_774806?
/batch_normalization_542/StatefulPartitionedCallStatefulPartitionedCall*dense_599/StatefulPartitionedCall:output:0batch_normalization_542_775676batch_normalization_542_775678batch_normalization_542_775680batch_normalization_542_775682*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_774361?
leaky_re_lu_542/PartitionedCallPartitionedCall8batch_normalization_542/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_542_layer_call_and_return_conditional_losses_774826?
!dense_600/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_542/PartitionedCall:output:0dense_600_775686dense_600_775688*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_600_layer_call_and_return_conditional_losses_774838?
/batch_normalization_543/StatefulPartitionedCallStatefulPartitionedCall*dense_600/StatefulPartitionedCall:output:0batch_normalization_543_775691batch_normalization_543_775693batch_normalization_543_775695batch_normalization_543_775697*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_774443?
leaky_re_lu_543/PartitionedCallPartitionedCall8batch_normalization_543/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_543_layer_call_and_return_conditional_losses_774858?
!dense_601/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_543/PartitionedCall:output:0dense_601_775701dense_601_775703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_601_layer_call_and_return_conditional_losses_774870?
/batch_normalization_544/StatefulPartitionedCallStatefulPartitionedCall*dense_601/StatefulPartitionedCall:output:0batch_normalization_544_775706batch_normalization_544_775708batch_normalization_544_775710batch_normalization_544_775712*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_774525?
leaky_re_lu_544/PartitionedCallPartitionedCall8batch_normalization_544/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_544_layer_call_and_return_conditional_losses_774890?
!dense_602/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_544/PartitionedCall:output:0dense_602_775716dense_602_775718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_602_layer_call_and_return_conditional_losses_774902?
/batch_normalization_545/StatefulPartitionedCallStatefulPartitionedCall*dense_602/StatefulPartitionedCall:output:0batch_normalization_545_775721batch_normalization_545_775723batch_normalization_545_775725batch_normalization_545_775727*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_774607?
leaky_re_lu_545/PartitionedCallPartitionedCall8batch_normalization_545/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_545_layer_call_and_return_conditional_losses_774922?
!dense_603/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_545/PartitionedCall:output:0dense_603_775731dense_603_775733*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_603_layer_call_and_return_conditional_losses_774934?
/batch_normalization_546/StatefulPartitionedCallStatefulPartitionedCall*dense_603/StatefulPartitionedCall:output:0batch_normalization_546_775736batch_normalization_546_775738batch_normalization_546_775740batch_normalization_546_775742*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_774689?
leaky_re_lu_546/PartitionedCallPartitionedCall8batch_normalization_546/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_546_layer_call_and_return_conditional_losses_774954?
!dense_604/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_546/PartitionedCall:output:0dense_604_775746dense_604_775748*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_604_layer_call_and_return_conditional_losses_774966?
/batch_normalization_547/StatefulPartitionedCallStatefulPartitionedCall*dense_604/StatefulPartitionedCall:output:0batch_normalization_547_775751batch_normalization_547_775753batch_normalization_547_775755batch_normalization_547_775757*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_774771?
leaky_re_lu_547/PartitionedCallPartitionedCall8batch_normalization_547/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_547_layer_call_and_return_conditional_losses_774986?
!dense_605/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_547/PartitionedCall:output:0dense_605_775761dense_605_775763*
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
E__inference_dense_605_layer_call_and_return_conditional_losses_774998y
IdentityIdentity*dense_605/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp0^batch_normalization_542/StatefulPartitionedCall0^batch_normalization_543/StatefulPartitionedCall0^batch_normalization_544/StatefulPartitionedCall0^batch_normalization_545/StatefulPartitionedCall0^batch_normalization_546/StatefulPartitionedCall0^batch_normalization_547/StatefulPartitionedCall"^dense_599/StatefulPartitionedCall"^dense_600/StatefulPartitionedCall"^dense_601/StatefulPartitionedCall"^dense_602/StatefulPartitionedCall"^dense_603/StatefulPartitionedCall"^dense_604/StatefulPartitionedCall"^dense_605/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_542/StatefulPartitionedCall/batch_normalization_542/StatefulPartitionedCall2b
/batch_normalization_543/StatefulPartitionedCall/batch_normalization_543/StatefulPartitionedCall2b
/batch_normalization_544/StatefulPartitionedCall/batch_normalization_544/StatefulPartitionedCall2b
/batch_normalization_545/StatefulPartitionedCall/batch_normalization_545/StatefulPartitionedCall2b
/batch_normalization_546/StatefulPartitionedCall/batch_normalization_546/StatefulPartitionedCall2b
/batch_normalization_547/StatefulPartitionedCall/batch_normalization_547/StatefulPartitionedCall2F
!dense_599/StatefulPartitionedCall!dense_599/StatefulPartitionedCall2F
!dense_600/StatefulPartitionedCall!dense_600/StatefulPartitionedCall2F
!dense_601/StatefulPartitionedCall!dense_601/StatefulPartitionedCall2F
!dense_602/StatefulPartitionedCall!dense_602/StatefulPartitionedCall2F
!dense_603/StatefulPartitionedCall!dense_603/StatefulPartitionedCall2F
!dense_604/StatefulPartitionedCall!dense_604/StatefulPartitionedCall2F
!dense_605/StatefulPartitionedCall!dense_605/StatefulPartitionedCall:_ [
'
_output_shapes
:?????????
0
_user_specified_namenormalization_57_input:$ 

_output_shapes

::$ 

_output_shapes

:
?i
?
I__inference_sequential_57_layer_call_and_return_conditional_losses_775661
normalization_57_input
normalization_57_sub_y
normalization_57_sqrt_x"
dense_599_775565:J
dense_599_775567:J,
batch_normalization_542_775570:J,
batch_normalization_542_775572:J,
batch_normalization_542_775574:J,
batch_normalization_542_775576:J"
dense_600_775580:JJ
dense_600_775582:J,
batch_normalization_543_775585:J,
batch_normalization_543_775587:J,
batch_normalization_543_775589:J,
batch_normalization_543_775591:J"
dense_601_775595:J,
dense_601_775597:,,
batch_normalization_544_775600:,,
batch_normalization_544_775602:,,
batch_normalization_544_775604:,,
batch_normalization_544_775606:,"
dense_602_775610:,,
dense_602_775612:,,
batch_normalization_545_775615:,,
batch_normalization_545_775617:,,
batch_normalization_545_775619:,,
batch_normalization_545_775621:,"
dense_603_775625:,2
dense_603_775627:2,
batch_normalization_546_775630:2,
batch_normalization_546_775632:2,
batch_normalization_546_775634:2,
batch_normalization_546_775636:2"
dense_604_775640:22
dense_604_775642:2,
batch_normalization_547_775645:2,
batch_normalization_547_775647:2,
batch_normalization_547_775649:2,
batch_normalization_547_775651:2"
dense_605_775655:2
dense_605_775657:
identity??/batch_normalization_542/StatefulPartitionedCall?/batch_normalization_543/StatefulPartitionedCall?/batch_normalization_544/StatefulPartitionedCall?/batch_normalization_545/StatefulPartitionedCall?/batch_normalization_546/StatefulPartitionedCall?/batch_normalization_547/StatefulPartitionedCall?!dense_599/StatefulPartitionedCall?!dense_600/StatefulPartitionedCall?!dense_601/StatefulPartitionedCall?!dense_602/StatefulPartitionedCall?!dense_603/StatefulPartitionedCall?!dense_604/StatefulPartitionedCall?!dense_605/StatefulPartitionedCall}
normalization_57/subSubnormalization_57_inputnormalization_57_sub_y*
T0*'
_output_shapes
:?????????_
normalization_57/SqrtSqrtnormalization_57_sqrt_x*
T0*
_output_shapes

:_
normalization_57/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_57/MaximumMaximumnormalization_57/Sqrt:y:0#normalization_57/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_57/truedivRealDivnormalization_57/sub:z:0normalization_57/Maximum:z:0*
T0*'
_output_shapes
:??????????
!dense_599/StatefulPartitionedCallStatefulPartitionedCallnormalization_57/truediv:z:0dense_599_775565dense_599_775567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_599_layer_call_and_return_conditional_losses_774806?
/batch_normalization_542/StatefulPartitionedCallStatefulPartitionedCall*dense_599/StatefulPartitionedCall:output:0batch_normalization_542_775570batch_normalization_542_775572batch_normalization_542_775574batch_normalization_542_775576*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_774314?
leaky_re_lu_542/PartitionedCallPartitionedCall8batch_normalization_542/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_542_layer_call_and_return_conditional_losses_774826?
!dense_600/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_542/PartitionedCall:output:0dense_600_775580dense_600_775582*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_600_layer_call_and_return_conditional_losses_774838?
/batch_normalization_543/StatefulPartitionedCallStatefulPartitionedCall*dense_600/StatefulPartitionedCall:output:0batch_normalization_543_775585batch_normalization_543_775587batch_normalization_543_775589batch_normalization_543_775591*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_774396?
leaky_re_lu_543/PartitionedCallPartitionedCall8batch_normalization_543/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_543_layer_call_and_return_conditional_losses_774858?
!dense_601/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_543/PartitionedCall:output:0dense_601_775595dense_601_775597*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_601_layer_call_and_return_conditional_losses_774870?
/batch_normalization_544/StatefulPartitionedCallStatefulPartitionedCall*dense_601/StatefulPartitionedCall:output:0batch_normalization_544_775600batch_normalization_544_775602batch_normalization_544_775604batch_normalization_544_775606*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_774478?
leaky_re_lu_544/PartitionedCallPartitionedCall8batch_normalization_544/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_544_layer_call_and_return_conditional_losses_774890?
!dense_602/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_544/PartitionedCall:output:0dense_602_775610dense_602_775612*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_602_layer_call_and_return_conditional_losses_774902?
/batch_normalization_545/StatefulPartitionedCallStatefulPartitionedCall*dense_602/StatefulPartitionedCall:output:0batch_normalization_545_775615batch_normalization_545_775617batch_normalization_545_775619batch_normalization_545_775621*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_774560?
leaky_re_lu_545/PartitionedCallPartitionedCall8batch_normalization_545/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_545_layer_call_and_return_conditional_losses_774922?
!dense_603/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_545/PartitionedCall:output:0dense_603_775625dense_603_775627*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_603_layer_call_and_return_conditional_losses_774934?
/batch_normalization_546/StatefulPartitionedCallStatefulPartitionedCall*dense_603/StatefulPartitionedCall:output:0batch_normalization_546_775630batch_normalization_546_775632batch_normalization_546_775634batch_normalization_546_775636*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_774642?
leaky_re_lu_546/PartitionedCallPartitionedCall8batch_normalization_546/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_546_layer_call_and_return_conditional_losses_774954?
!dense_604/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_546/PartitionedCall:output:0dense_604_775640dense_604_775642*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_604_layer_call_and_return_conditional_losses_774966?
/batch_normalization_547/StatefulPartitionedCallStatefulPartitionedCall*dense_604/StatefulPartitionedCall:output:0batch_normalization_547_775645batch_normalization_547_775647batch_normalization_547_775649batch_normalization_547_775651*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_774724?
leaky_re_lu_547/PartitionedCallPartitionedCall8batch_normalization_547/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_547_layer_call_and_return_conditional_losses_774986?
!dense_605/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_547/PartitionedCall:output:0dense_605_775655dense_605_775657*
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
E__inference_dense_605_layer_call_and_return_conditional_losses_774998y
IdentityIdentity*dense_605/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp0^batch_normalization_542/StatefulPartitionedCall0^batch_normalization_543/StatefulPartitionedCall0^batch_normalization_544/StatefulPartitionedCall0^batch_normalization_545/StatefulPartitionedCall0^batch_normalization_546/StatefulPartitionedCall0^batch_normalization_547/StatefulPartitionedCall"^dense_599/StatefulPartitionedCall"^dense_600/StatefulPartitionedCall"^dense_601/StatefulPartitionedCall"^dense_602/StatefulPartitionedCall"^dense_603/StatefulPartitionedCall"^dense_604/StatefulPartitionedCall"^dense_605/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_542/StatefulPartitionedCall/batch_normalization_542/StatefulPartitionedCall2b
/batch_normalization_543/StatefulPartitionedCall/batch_normalization_543/StatefulPartitionedCall2b
/batch_normalization_544/StatefulPartitionedCall/batch_normalization_544/StatefulPartitionedCall2b
/batch_normalization_545/StatefulPartitionedCall/batch_normalization_545/StatefulPartitionedCall2b
/batch_normalization_546/StatefulPartitionedCall/batch_normalization_546/StatefulPartitionedCall2b
/batch_normalization_547/StatefulPartitionedCall/batch_normalization_547/StatefulPartitionedCall2F
!dense_599/StatefulPartitionedCall!dense_599/StatefulPartitionedCall2F
!dense_600/StatefulPartitionedCall!dense_600/StatefulPartitionedCall2F
!dense_601/StatefulPartitionedCall!dense_601/StatefulPartitionedCall2F
!dense_602/StatefulPartitionedCall!dense_602/StatefulPartitionedCall2F
!dense_603/StatefulPartitionedCall!dense_603/StatefulPartitionedCall2F
!dense_604/StatefulPartitionedCall!dense_604/StatefulPartitionedCall2F
!dense_605/StatefulPartitionedCall!dense_605/StatefulPartitionedCall:_ [
'
_output_shapes
:?????????
0
_user_specified_namenormalization_57_input:$ 

_output_shapes

::$ 

_output_shapes

:
?%
?
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_776677

inputs5
'assignmovingavg_readvariableop_resource:J7
)assignmovingavg_1_readvariableop_resource:J3
%batchnorm_mul_readvariableop_resource:J/
!batchnorm_readvariableop_resource:J
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:J*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:J?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????Jl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:J*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:J*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:J*
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
:J*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Jx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:J?
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
:J*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:J~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:J?
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
:JP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:J~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Jc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????Jh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Jv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Jr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????Jb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????J?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_542_layer_call_fn_776573

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
:?????????J* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_542_layer_call_and_return_conditional_losses_774826`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????J"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????J:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_774478

inputs/
!batchnorm_readvariableop_resource:,3
%batchnorm_mul_readvariableop_resource:,1
#batchnorm_readvariableop_1_resource:,1
#batchnorm_readvariableop_2_resource:,
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:,*
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
:,P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:,~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:,*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:,z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:,*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:,r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????,?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
??
?A
"__inference__traced_restore_777771
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_599_kernel:J/
!assignvariableop_4_dense_599_bias:J>
0assignvariableop_5_batch_normalization_542_gamma:J=
/assignvariableop_6_batch_normalization_542_beta:JD
6assignvariableop_7_batch_normalization_542_moving_mean:JH
:assignvariableop_8_batch_normalization_542_moving_variance:J5
#assignvariableop_9_dense_600_kernel:JJ0
"assignvariableop_10_dense_600_bias:J?
1assignvariableop_11_batch_normalization_543_gamma:J>
0assignvariableop_12_batch_normalization_543_beta:JE
7assignvariableop_13_batch_normalization_543_moving_mean:JI
;assignvariableop_14_batch_normalization_543_moving_variance:J6
$assignvariableop_15_dense_601_kernel:J,0
"assignvariableop_16_dense_601_bias:,?
1assignvariableop_17_batch_normalization_544_gamma:,>
0assignvariableop_18_batch_normalization_544_beta:,E
7assignvariableop_19_batch_normalization_544_moving_mean:,I
;assignvariableop_20_batch_normalization_544_moving_variance:,6
$assignvariableop_21_dense_602_kernel:,,0
"assignvariableop_22_dense_602_bias:,?
1assignvariableop_23_batch_normalization_545_gamma:,>
0assignvariableop_24_batch_normalization_545_beta:,E
7assignvariableop_25_batch_normalization_545_moving_mean:,I
;assignvariableop_26_batch_normalization_545_moving_variance:,6
$assignvariableop_27_dense_603_kernel:,20
"assignvariableop_28_dense_603_bias:2?
1assignvariableop_29_batch_normalization_546_gamma:2>
0assignvariableop_30_batch_normalization_546_beta:2E
7assignvariableop_31_batch_normalization_546_moving_mean:2I
;assignvariableop_32_batch_normalization_546_moving_variance:26
$assignvariableop_33_dense_604_kernel:220
"assignvariableop_34_dense_604_bias:2?
1assignvariableop_35_batch_normalization_547_gamma:2>
0assignvariableop_36_batch_normalization_547_beta:2E
7assignvariableop_37_batch_normalization_547_moving_mean:2I
;assignvariableop_38_batch_normalization_547_moving_variance:26
$assignvariableop_39_dense_605_kernel:20
"assignvariableop_40_dense_605_bias:'
assignvariableop_41_adam_iter:	 )
assignvariableop_42_adam_beta_1: )
assignvariableop_43_adam_beta_2: (
assignvariableop_44_adam_decay: #
assignvariableop_45_total: %
assignvariableop_46_count_1: =
+assignvariableop_47_adam_dense_599_kernel_m:J7
)assignvariableop_48_adam_dense_599_bias_m:JF
8assignvariableop_49_adam_batch_normalization_542_gamma_m:JE
7assignvariableop_50_adam_batch_normalization_542_beta_m:J=
+assignvariableop_51_adam_dense_600_kernel_m:JJ7
)assignvariableop_52_adam_dense_600_bias_m:JF
8assignvariableop_53_adam_batch_normalization_543_gamma_m:JE
7assignvariableop_54_adam_batch_normalization_543_beta_m:J=
+assignvariableop_55_adam_dense_601_kernel_m:J,7
)assignvariableop_56_adam_dense_601_bias_m:,F
8assignvariableop_57_adam_batch_normalization_544_gamma_m:,E
7assignvariableop_58_adam_batch_normalization_544_beta_m:,=
+assignvariableop_59_adam_dense_602_kernel_m:,,7
)assignvariableop_60_adam_dense_602_bias_m:,F
8assignvariableop_61_adam_batch_normalization_545_gamma_m:,E
7assignvariableop_62_adam_batch_normalization_545_beta_m:,=
+assignvariableop_63_adam_dense_603_kernel_m:,27
)assignvariableop_64_adam_dense_603_bias_m:2F
8assignvariableop_65_adam_batch_normalization_546_gamma_m:2E
7assignvariableop_66_adam_batch_normalization_546_beta_m:2=
+assignvariableop_67_adam_dense_604_kernel_m:227
)assignvariableop_68_adam_dense_604_bias_m:2F
8assignvariableop_69_adam_batch_normalization_547_gamma_m:2E
7assignvariableop_70_adam_batch_normalization_547_beta_m:2=
+assignvariableop_71_adam_dense_605_kernel_m:27
)assignvariableop_72_adam_dense_605_bias_m:=
+assignvariableop_73_adam_dense_599_kernel_v:J7
)assignvariableop_74_adam_dense_599_bias_v:JF
8assignvariableop_75_adam_batch_normalization_542_gamma_v:JE
7assignvariableop_76_adam_batch_normalization_542_beta_v:J=
+assignvariableop_77_adam_dense_600_kernel_v:JJ7
)assignvariableop_78_adam_dense_600_bias_v:JF
8assignvariableop_79_adam_batch_normalization_543_gamma_v:JE
7assignvariableop_80_adam_batch_normalization_543_beta_v:J=
+assignvariableop_81_adam_dense_601_kernel_v:J,7
)assignvariableop_82_adam_dense_601_bias_v:,F
8assignvariableop_83_adam_batch_normalization_544_gamma_v:,E
7assignvariableop_84_adam_batch_normalization_544_beta_v:,=
+assignvariableop_85_adam_dense_602_kernel_v:,,7
)assignvariableop_86_adam_dense_602_bias_v:,F
8assignvariableop_87_adam_batch_normalization_545_gamma_v:,E
7assignvariableop_88_adam_batch_normalization_545_beta_v:,=
+assignvariableop_89_adam_dense_603_kernel_v:,27
)assignvariableop_90_adam_dense_603_bias_v:2F
8assignvariableop_91_adam_batch_normalization_546_gamma_v:2E
7assignvariableop_92_adam_batch_normalization_546_beta_v:2=
+assignvariableop_93_adam_dense_604_kernel_v:227
)assignvariableop_94_adam_dense_604_bias_v:2F
8assignvariableop_95_adam_batch_normalization_547_gamma_v:2E
7assignvariableop_96_adam_batch_normalization_547_beta_v:2=
+assignvariableop_97_adam_dense_605_kernel_v:27
)assignvariableop_98_adam_dense_605_bias_v:
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_599_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_599_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batch_normalization_542_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_542_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_542_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp:assignvariableop_8_batch_normalization_542_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_600_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_600_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batch_normalization_543_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_543_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_543_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp;assignvariableop_14_batch_normalization_543_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_601_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_601_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batch_normalization_544_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_544_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_544_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp;assignvariableop_20_batch_normalization_544_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_602_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_602_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp1assignvariableop_23_batch_normalization_545_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_545_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp7assignvariableop_25_batch_normalization_545_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp;assignvariableop_26_batch_normalization_545_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_603_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_603_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_batch_normalization_546_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_546_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp7assignvariableop_31_batch_normalization_546_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp;assignvariableop_32_batch_normalization_546_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_604_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_604_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp1assignvariableop_35_batch_normalization_547_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_547_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp7assignvariableop_37_batch_normalization_547_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp;assignvariableop_38_batch_normalization_547_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp$assignvariableop_39_dense_605_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_605_biasIdentity_40:output:0"/device:CPU:0*
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
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_dense_599_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_599_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp8assignvariableop_49_adam_batch_normalization_542_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp7assignvariableop_50_adam_batch_normalization_542_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_dense_600_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_dense_600_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp8assignvariableop_53_adam_batch_normalization_543_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp7assignvariableop_54_adam_batch_normalization_543_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_dense_601_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_dense_601_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp8assignvariableop_57_adam_batch_normalization_544_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp7assignvariableop_58_adam_batch_normalization_544_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_dense_602_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_dense_602_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp8assignvariableop_61_adam_batch_normalization_545_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp7assignvariableop_62_adam_batch_normalization_545_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_dense_603_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_dense_603_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp8assignvariableop_65_adam_batch_normalization_546_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp7assignvariableop_66_adam_batch_normalization_546_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_dense_604_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_dense_604_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp8assignvariableop_69_adam_batch_normalization_547_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp7assignvariableop_70_adam_batch_normalization_547_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_dense_605_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_dense_605_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_dense_599_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_dense_599_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp8assignvariableop_75_adam_batch_normalization_542_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp7assignvariableop_76_adam_batch_normalization_542_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_dense_600_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_dense_600_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp8assignvariableop_79_adam_batch_normalization_543_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp7assignvariableop_80_adam_batch_normalization_543_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_dense_601_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_dense_601_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp8assignvariableop_83_adam_batch_normalization_544_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp7assignvariableop_84_adam_batch_normalization_544_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_dense_602_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_dense_602_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp8assignvariableop_87_adam_batch_normalization_545_gamma_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp7assignvariableop_88_adam_batch_normalization_545_beta_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_dense_603_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_dense_603_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp8assignvariableop_91_adam_batch_normalization_546_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp7assignvariableop_92_adam_batch_normalization_546_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_dense_604_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_dense_604_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOp8assignvariableop_95_adam_batch_normalization_547_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOp7assignvariableop_96_adam_batch_normalization_547_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_dense_605_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_dense_605_bias_vIdentity_98:output:0"/device:CPU:0*
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
?
g
K__inference_leaky_re_lu_544_layer_call_and_return_conditional_losses_776796

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????,*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????,"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????,:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?h
?
I__inference_sequential_57_layer_call_and_return_conditional_losses_775387

inputs
normalization_57_sub_y
normalization_57_sqrt_x"
dense_599_775291:J
dense_599_775293:J,
batch_normalization_542_775296:J,
batch_normalization_542_775298:J,
batch_normalization_542_775300:J,
batch_normalization_542_775302:J"
dense_600_775306:JJ
dense_600_775308:J,
batch_normalization_543_775311:J,
batch_normalization_543_775313:J,
batch_normalization_543_775315:J,
batch_normalization_543_775317:J"
dense_601_775321:J,
dense_601_775323:,,
batch_normalization_544_775326:,,
batch_normalization_544_775328:,,
batch_normalization_544_775330:,,
batch_normalization_544_775332:,"
dense_602_775336:,,
dense_602_775338:,,
batch_normalization_545_775341:,,
batch_normalization_545_775343:,,
batch_normalization_545_775345:,,
batch_normalization_545_775347:,"
dense_603_775351:,2
dense_603_775353:2,
batch_normalization_546_775356:2,
batch_normalization_546_775358:2,
batch_normalization_546_775360:2,
batch_normalization_546_775362:2"
dense_604_775366:22
dense_604_775368:2,
batch_normalization_547_775371:2,
batch_normalization_547_775373:2,
batch_normalization_547_775375:2,
batch_normalization_547_775377:2"
dense_605_775381:2
dense_605_775383:
identity??/batch_normalization_542/StatefulPartitionedCall?/batch_normalization_543/StatefulPartitionedCall?/batch_normalization_544/StatefulPartitionedCall?/batch_normalization_545/StatefulPartitionedCall?/batch_normalization_546/StatefulPartitionedCall?/batch_normalization_547/StatefulPartitionedCall?!dense_599/StatefulPartitionedCall?!dense_600/StatefulPartitionedCall?!dense_601/StatefulPartitionedCall?!dense_602/StatefulPartitionedCall?!dense_603/StatefulPartitionedCall?!dense_604/StatefulPartitionedCall?!dense_605/StatefulPartitionedCallm
normalization_57/subSubinputsnormalization_57_sub_y*
T0*'
_output_shapes
:?????????_
normalization_57/SqrtSqrtnormalization_57_sqrt_x*
T0*
_output_shapes

:_
normalization_57/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_57/MaximumMaximumnormalization_57/Sqrt:y:0#normalization_57/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_57/truedivRealDivnormalization_57/sub:z:0normalization_57/Maximum:z:0*
T0*'
_output_shapes
:??????????
!dense_599/StatefulPartitionedCallStatefulPartitionedCallnormalization_57/truediv:z:0dense_599_775291dense_599_775293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_599_layer_call_and_return_conditional_losses_774806?
/batch_normalization_542/StatefulPartitionedCallStatefulPartitionedCall*dense_599/StatefulPartitionedCall:output:0batch_normalization_542_775296batch_normalization_542_775298batch_normalization_542_775300batch_normalization_542_775302*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_774361?
leaky_re_lu_542/PartitionedCallPartitionedCall8batch_normalization_542/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_542_layer_call_and_return_conditional_losses_774826?
!dense_600/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_542/PartitionedCall:output:0dense_600_775306dense_600_775308*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_600_layer_call_and_return_conditional_losses_774838?
/batch_normalization_543/StatefulPartitionedCallStatefulPartitionedCall*dense_600/StatefulPartitionedCall:output:0batch_normalization_543_775311batch_normalization_543_775313batch_normalization_543_775315batch_normalization_543_775317*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_774443?
leaky_re_lu_543/PartitionedCallPartitionedCall8batch_normalization_543/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_543_layer_call_and_return_conditional_losses_774858?
!dense_601/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_543/PartitionedCall:output:0dense_601_775321dense_601_775323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_601_layer_call_and_return_conditional_losses_774870?
/batch_normalization_544/StatefulPartitionedCallStatefulPartitionedCall*dense_601/StatefulPartitionedCall:output:0batch_normalization_544_775326batch_normalization_544_775328batch_normalization_544_775330batch_normalization_544_775332*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_774525?
leaky_re_lu_544/PartitionedCallPartitionedCall8batch_normalization_544/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_544_layer_call_and_return_conditional_losses_774890?
!dense_602/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_544/PartitionedCall:output:0dense_602_775336dense_602_775338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_602_layer_call_and_return_conditional_losses_774902?
/batch_normalization_545/StatefulPartitionedCallStatefulPartitionedCall*dense_602/StatefulPartitionedCall:output:0batch_normalization_545_775341batch_normalization_545_775343batch_normalization_545_775345batch_normalization_545_775347*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_774607?
leaky_re_lu_545/PartitionedCallPartitionedCall8batch_normalization_545/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_545_layer_call_and_return_conditional_losses_774922?
!dense_603/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_545/PartitionedCall:output:0dense_603_775351dense_603_775353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_603_layer_call_and_return_conditional_losses_774934?
/batch_normalization_546/StatefulPartitionedCallStatefulPartitionedCall*dense_603/StatefulPartitionedCall:output:0batch_normalization_546_775356batch_normalization_546_775358batch_normalization_546_775360batch_normalization_546_775362*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_774689?
leaky_re_lu_546/PartitionedCallPartitionedCall8batch_normalization_546/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_546_layer_call_and_return_conditional_losses_774954?
!dense_604/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_546/PartitionedCall:output:0dense_604_775366dense_604_775368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_604_layer_call_and_return_conditional_losses_774966?
/batch_normalization_547/StatefulPartitionedCallStatefulPartitionedCall*dense_604/StatefulPartitionedCall:output:0batch_normalization_547_775371batch_normalization_547_775373batch_normalization_547_775375batch_normalization_547_775377*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_774771?
leaky_re_lu_547/PartitionedCallPartitionedCall8batch_normalization_547/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_547_layer_call_and_return_conditional_losses_774986?
!dense_605/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_547/PartitionedCall:output:0dense_605_775381dense_605_775383*
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
E__inference_dense_605_layer_call_and_return_conditional_losses_774998y
IdentityIdentity*dense_605/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp0^batch_normalization_542/StatefulPartitionedCall0^batch_normalization_543/StatefulPartitionedCall0^batch_normalization_544/StatefulPartitionedCall0^batch_normalization_545/StatefulPartitionedCall0^batch_normalization_546/StatefulPartitionedCall0^batch_normalization_547/StatefulPartitionedCall"^dense_599/StatefulPartitionedCall"^dense_600/StatefulPartitionedCall"^dense_601/StatefulPartitionedCall"^dense_602/StatefulPartitionedCall"^dense_603/StatefulPartitionedCall"^dense_604/StatefulPartitionedCall"^dense_605/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_542/StatefulPartitionedCall/batch_normalization_542/StatefulPartitionedCall2b
/batch_normalization_543/StatefulPartitionedCall/batch_normalization_543/StatefulPartitionedCall2b
/batch_normalization_544/StatefulPartitionedCall/batch_normalization_544/StatefulPartitionedCall2b
/batch_normalization_545/StatefulPartitionedCall/batch_normalization_545/StatefulPartitionedCall2b
/batch_normalization_546/StatefulPartitionedCall/batch_normalization_546/StatefulPartitionedCall2b
/batch_normalization_547/StatefulPartitionedCall/batch_normalization_547/StatefulPartitionedCall2F
!dense_599/StatefulPartitionedCall!dense_599/StatefulPartitionedCall2F
!dense_600/StatefulPartitionedCall!dense_600/StatefulPartitionedCall2F
!dense_601/StatefulPartitionedCall!dense_601/StatefulPartitionedCall2F
!dense_602/StatefulPartitionedCall!dense_602/StatefulPartitionedCall2F
!dense_603/StatefulPartitionedCall!dense_603/StatefulPartitionedCall2F
!dense_604/StatefulPartitionedCall!dense_604/StatefulPartitionedCall2F
!dense_605/StatefulPartitionedCall!dense_605/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_776970

inputs/
!batchnorm_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:21
#batchnorm_readvariableop_1_resource:21
#batchnorm_readvariableop_2_resource:2
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
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
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_543_layer_call_fn_776623

inputs
unknown:J
	unknown_0:J
	unknown_1:J
	unknown_2:J
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_774443o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????J`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
??
?+
!__inference__wrapped_model_774290
normalization_57_input(
$sequential_57_normalization_57_sub_y)
%sequential_57_normalization_57_sqrt_xH
6sequential_57_dense_599_matmul_readvariableop_resource:JE
7sequential_57_dense_599_biasadd_readvariableop_resource:JU
Gsequential_57_batch_normalization_542_batchnorm_readvariableop_resource:JY
Ksequential_57_batch_normalization_542_batchnorm_mul_readvariableop_resource:JW
Isequential_57_batch_normalization_542_batchnorm_readvariableop_1_resource:JW
Isequential_57_batch_normalization_542_batchnorm_readvariableop_2_resource:JH
6sequential_57_dense_600_matmul_readvariableop_resource:JJE
7sequential_57_dense_600_biasadd_readvariableop_resource:JU
Gsequential_57_batch_normalization_543_batchnorm_readvariableop_resource:JY
Ksequential_57_batch_normalization_543_batchnorm_mul_readvariableop_resource:JW
Isequential_57_batch_normalization_543_batchnorm_readvariableop_1_resource:JW
Isequential_57_batch_normalization_543_batchnorm_readvariableop_2_resource:JH
6sequential_57_dense_601_matmul_readvariableop_resource:J,E
7sequential_57_dense_601_biasadd_readvariableop_resource:,U
Gsequential_57_batch_normalization_544_batchnorm_readvariableop_resource:,Y
Ksequential_57_batch_normalization_544_batchnorm_mul_readvariableop_resource:,W
Isequential_57_batch_normalization_544_batchnorm_readvariableop_1_resource:,W
Isequential_57_batch_normalization_544_batchnorm_readvariableop_2_resource:,H
6sequential_57_dense_602_matmul_readvariableop_resource:,,E
7sequential_57_dense_602_biasadd_readvariableop_resource:,U
Gsequential_57_batch_normalization_545_batchnorm_readvariableop_resource:,Y
Ksequential_57_batch_normalization_545_batchnorm_mul_readvariableop_resource:,W
Isequential_57_batch_normalization_545_batchnorm_readvariableop_1_resource:,W
Isequential_57_batch_normalization_545_batchnorm_readvariableop_2_resource:,H
6sequential_57_dense_603_matmul_readvariableop_resource:,2E
7sequential_57_dense_603_biasadd_readvariableop_resource:2U
Gsequential_57_batch_normalization_546_batchnorm_readvariableop_resource:2Y
Ksequential_57_batch_normalization_546_batchnorm_mul_readvariableop_resource:2W
Isequential_57_batch_normalization_546_batchnorm_readvariableop_1_resource:2W
Isequential_57_batch_normalization_546_batchnorm_readvariableop_2_resource:2H
6sequential_57_dense_604_matmul_readvariableop_resource:22E
7sequential_57_dense_604_biasadd_readvariableop_resource:2U
Gsequential_57_batch_normalization_547_batchnorm_readvariableop_resource:2Y
Ksequential_57_batch_normalization_547_batchnorm_mul_readvariableop_resource:2W
Isequential_57_batch_normalization_547_batchnorm_readvariableop_1_resource:2W
Isequential_57_batch_normalization_547_batchnorm_readvariableop_2_resource:2H
6sequential_57_dense_605_matmul_readvariableop_resource:2E
7sequential_57_dense_605_biasadd_readvariableop_resource:
identity??>sequential_57/batch_normalization_542/batchnorm/ReadVariableOp?@sequential_57/batch_normalization_542/batchnorm/ReadVariableOp_1?@sequential_57/batch_normalization_542/batchnorm/ReadVariableOp_2?Bsequential_57/batch_normalization_542/batchnorm/mul/ReadVariableOp?>sequential_57/batch_normalization_543/batchnorm/ReadVariableOp?@sequential_57/batch_normalization_543/batchnorm/ReadVariableOp_1?@sequential_57/batch_normalization_543/batchnorm/ReadVariableOp_2?Bsequential_57/batch_normalization_543/batchnorm/mul/ReadVariableOp?>sequential_57/batch_normalization_544/batchnorm/ReadVariableOp?@sequential_57/batch_normalization_544/batchnorm/ReadVariableOp_1?@sequential_57/batch_normalization_544/batchnorm/ReadVariableOp_2?Bsequential_57/batch_normalization_544/batchnorm/mul/ReadVariableOp?>sequential_57/batch_normalization_545/batchnorm/ReadVariableOp?@sequential_57/batch_normalization_545/batchnorm/ReadVariableOp_1?@sequential_57/batch_normalization_545/batchnorm/ReadVariableOp_2?Bsequential_57/batch_normalization_545/batchnorm/mul/ReadVariableOp?>sequential_57/batch_normalization_546/batchnorm/ReadVariableOp?@sequential_57/batch_normalization_546/batchnorm/ReadVariableOp_1?@sequential_57/batch_normalization_546/batchnorm/ReadVariableOp_2?Bsequential_57/batch_normalization_546/batchnorm/mul/ReadVariableOp?>sequential_57/batch_normalization_547/batchnorm/ReadVariableOp?@sequential_57/batch_normalization_547/batchnorm/ReadVariableOp_1?@sequential_57/batch_normalization_547/batchnorm/ReadVariableOp_2?Bsequential_57/batch_normalization_547/batchnorm/mul/ReadVariableOp?.sequential_57/dense_599/BiasAdd/ReadVariableOp?-sequential_57/dense_599/MatMul/ReadVariableOp?.sequential_57/dense_600/BiasAdd/ReadVariableOp?-sequential_57/dense_600/MatMul/ReadVariableOp?.sequential_57/dense_601/BiasAdd/ReadVariableOp?-sequential_57/dense_601/MatMul/ReadVariableOp?.sequential_57/dense_602/BiasAdd/ReadVariableOp?-sequential_57/dense_602/MatMul/ReadVariableOp?.sequential_57/dense_603/BiasAdd/ReadVariableOp?-sequential_57/dense_603/MatMul/ReadVariableOp?.sequential_57/dense_604/BiasAdd/ReadVariableOp?-sequential_57/dense_604/MatMul/ReadVariableOp?.sequential_57/dense_605/BiasAdd/ReadVariableOp?-sequential_57/dense_605/MatMul/ReadVariableOp?
"sequential_57/normalization_57/subSubnormalization_57_input$sequential_57_normalization_57_sub_y*
T0*'
_output_shapes
:?????????{
#sequential_57/normalization_57/SqrtSqrt%sequential_57_normalization_57_sqrt_x*
T0*
_output_shapes

:m
(sequential_57/normalization_57/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
&sequential_57/normalization_57/MaximumMaximum'sequential_57/normalization_57/Sqrt:y:01sequential_57/normalization_57/Maximum/y:output:0*
T0*
_output_shapes

:?
&sequential_57/normalization_57/truedivRealDiv&sequential_57/normalization_57/sub:z:0*sequential_57/normalization_57/Maximum:z:0*
T0*'
_output_shapes
:??????????
-sequential_57/dense_599/MatMul/ReadVariableOpReadVariableOp6sequential_57_dense_599_matmul_readvariableop_resource*
_output_shapes

:J*
dtype0?
sequential_57/dense_599/MatMulMatMul*sequential_57/normalization_57/truediv:z:05sequential_57/dense_599/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J?
.sequential_57/dense_599/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_599_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype0?
sequential_57/dense_599/BiasAddBiasAdd(sequential_57/dense_599/MatMul:product:06sequential_57/dense_599/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J?
>sequential_57/batch_normalization_542/batchnorm/ReadVariableOpReadVariableOpGsequential_57_batch_normalization_542_batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0z
5sequential_57/batch_normalization_542/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
3sequential_57/batch_normalization_542/batchnorm/addAddV2Fsequential_57/batch_normalization_542/batchnorm/ReadVariableOp:value:0>sequential_57/batch_normalization_542/batchnorm/add/y:output:0*
T0*
_output_shapes
:J?
5sequential_57/batch_normalization_542/batchnorm/RsqrtRsqrt7sequential_57/batch_normalization_542/batchnorm/add:z:0*
T0*
_output_shapes
:J?
Bsequential_57/batch_normalization_542/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_57_batch_normalization_542_batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0?
3sequential_57/batch_normalization_542/batchnorm/mulMul9sequential_57/batch_normalization_542/batchnorm/Rsqrt:y:0Jsequential_57/batch_normalization_542/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:J?
5sequential_57/batch_normalization_542/batchnorm/mul_1Mul(sequential_57/dense_599/BiasAdd:output:07sequential_57/batch_normalization_542/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????J?
@sequential_57/batch_normalization_542/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_57_batch_normalization_542_batchnorm_readvariableop_1_resource*
_output_shapes
:J*
dtype0?
5sequential_57/batch_normalization_542/batchnorm/mul_2MulHsequential_57/batch_normalization_542/batchnorm/ReadVariableOp_1:value:07sequential_57/batch_normalization_542/batchnorm/mul:z:0*
T0*
_output_shapes
:J?
@sequential_57/batch_normalization_542/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_57_batch_normalization_542_batchnorm_readvariableop_2_resource*
_output_shapes
:J*
dtype0?
3sequential_57/batch_normalization_542/batchnorm/subSubHsequential_57/batch_normalization_542/batchnorm/ReadVariableOp_2:value:09sequential_57/batch_normalization_542/batchnorm/mul_2:z:0*
T0*
_output_shapes
:J?
5sequential_57/batch_normalization_542/batchnorm/add_1AddV29sequential_57/batch_normalization_542/batchnorm/mul_1:z:07sequential_57/batch_normalization_542/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????J?
'sequential_57/leaky_re_lu_542/LeakyRelu	LeakyRelu9sequential_57/batch_normalization_542/batchnorm/add_1:z:0*'
_output_shapes
:?????????J*
alpha%???>?
-sequential_57/dense_600/MatMul/ReadVariableOpReadVariableOp6sequential_57_dense_600_matmul_readvariableop_resource*
_output_shapes

:JJ*
dtype0?
sequential_57/dense_600/MatMulMatMul5sequential_57/leaky_re_lu_542/LeakyRelu:activations:05sequential_57/dense_600/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J?
.sequential_57/dense_600/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_600_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype0?
sequential_57/dense_600/BiasAddBiasAdd(sequential_57/dense_600/MatMul:product:06sequential_57/dense_600/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J?
>sequential_57/batch_normalization_543/batchnorm/ReadVariableOpReadVariableOpGsequential_57_batch_normalization_543_batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0z
5sequential_57/batch_normalization_543/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
3sequential_57/batch_normalization_543/batchnorm/addAddV2Fsequential_57/batch_normalization_543/batchnorm/ReadVariableOp:value:0>sequential_57/batch_normalization_543/batchnorm/add/y:output:0*
T0*
_output_shapes
:J?
5sequential_57/batch_normalization_543/batchnorm/RsqrtRsqrt7sequential_57/batch_normalization_543/batchnorm/add:z:0*
T0*
_output_shapes
:J?
Bsequential_57/batch_normalization_543/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_57_batch_normalization_543_batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0?
3sequential_57/batch_normalization_543/batchnorm/mulMul9sequential_57/batch_normalization_543/batchnorm/Rsqrt:y:0Jsequential_57/batch_normalization_543/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:J?
5sequential_57/batch_normalization_543/batchnorm/mul_1Mul(sequential_57/dense_600/BiasAdd:output:07sequential_57/batch_normalization_543/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????J?
@sequential_57/batch_normalization_543/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_57_batch_normalization_543_batchnorm_readvariableop_1_resource*
_output_shapes
:J*
dtype0?
5sequential_57/batch_normalization_543/batchnorm/mul_2MulHsequential_57/batch_normalization_543/batchnorm/ReadVariableOp_1:value:07sequential_57/batch_normalization_543/batchnorm/mul:z:0*
T0*
_output_shapes
:J?
@sequential_57/batch_normalization_543/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_57_batch_normalization_543_batchnorm_readvariableop_2_resource*
_output_shapes
:J*
dtype0?
3sequential_57/batch_normalization_543/batchnorm/subSubHsequential_57/batch_normalization_543/batchnorm/ReadVariableOp_2:value:09sequential_57/batch_normalization_543/batchnorm/mul_2:z:0*
T0*
_output_shapes
:J?
5sequential_57/batch_normalization_543/batchnorm/add_1AddV29sequential_57/batch_normalization_543/batchnorm/mul_1:z:07sequential_57/batch_normalization_543/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????J?
'sequential_57/leaky_re_lu_543/LeakyRelu	LeakyRelu9sequential_57/batch_normalization_543/batchnorm/add_1:z:0*'
_output_shapes
:?????????J*
alpha%???>?
-sequential_57/dense_601/MatMul/ReadVariableOpReadVariableOp6sequential_57_dense_601_matmul_readvariableop_resource*
_output_shapes

:J,*
dtype0?
sequential_57/dense_601/MatMulMatMul5sequential_57/leaky_re_lu_543/LeakyRelu:activations:05sequential_57/dense_601/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,?
.sequential_57/dense_601/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_601_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0?
sequential_57/dense_601/BiasAddBiasAdd(sequential_57/dense_601/MatMul:product:06sequential_57/dense_601/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,?
>sequential_57/batch_normalization_544/batchnorm/ReadVariableOpReadVariableOpGsequential_57_batch_normalization_544_batchnorm_readvariableop_resource*
_output_shapes
:,*
dtype0z
5sequential_57/batch_normalization_544/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
3sequential_57/batch_normalization_544/batchnorm/addAddV2Fsequential_57/batch_normalization_544/batchnorm/ReadVariableOp:value:0>sequential_57/batch_normalization_544/batchnorm/add/y:output:0*
T0*
_output_shapes
:,?
5sequential_57/batch_normalization_544/batchnorm/RsqrtRsqrt7sequential_57/batch_normalization_544/batchnorm/add:z:0*
T0*
_output_shapes
:,?
Bsequential_57/batch_normalization_544/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_57_batch_normalization_544_batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0?
3sequential_57/batch_normalization_544/batchnorm/mulMul9sequential_57/batch_normalization_544/batchnorm/Rsqrt:y:0Jsequential_57/batch_normalization_544/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,?
5sequential_57/batch_normalization_544/batchnorm/mul_1Mul(sequential_57/dense_601/BiasAdd:output:07sequential_57/batch_normalization_544/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,?
@sequential_57/batch_normalization_544/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_57_batch_normalization_544_batchnorm_readvariableop_1_resource*
_output_shapes
:,*
dtype0?
5sequential_57/batch_normalization_544/batchnorm/mul_2MulHsequential_57/batch_normalization_544/batchnorm/ReadVariableOp_1:value:07sequential_57/batch_normalization_544/batchnorm/mul:z:0*
T0*
_output_shapes
:,?
@sequential_57/batch_normalization_544/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_57_batch_normalization_544_batchnorm_readvariableop_2_resource*
_output_shapes
:,*
dtype0?
3sequential_57/batch_normalization_544/batchnorm/subSubHsequential_57/batch_normalization_544/batchnorm/ReadVariableOp_2:value:09sequential_57/batch_normalization_544/batchnorm/mul_2:z:0*
T0*
_output_shapes
:,?
5sequential_57/batch_normalization_544/batchnorm/add_1AddV29sequential_57/batch_normalization_544/batchnorm/mul_1:z:07sequential_57/batch_normalization_544/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,?
'sequential_57/leaky_re_lu_544/LeakyRelu	LeakyRelu9sequential_57/batch_normalization_544/batchnorm/add_1:z:0*'
_output_shapes
:?????????,*
alpha%???>?
-sequential_57/dense_602/MatMul/ReadVariableOpReadVariableOp6sequential_57_dense_602_matmul_readvariableop_resource*
_output_shapes

:,,*
dtype0?
sequential_57/dense_602/MatMulMatMul5sequential_57/leaky_re_lu_544/LeakyRelu:activations:05sequential_57/dense_602/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,?
.sequential_57/dense_602/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_602_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0?
sequential_57/dense_602/BiasAddBiasAdd(sequential_57/dense_602/MatMul:product:06sequential_57/dense_602/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,?
>sequential_57/batch_normalization_545/batchnorm/ReadVariableOpReadVariableOpGsequential_57_batch_normalization_545_batchnorm_readvariableop_resource*
_output_shapes
:,*
dtype0z
5sequential_57/batch_normalization_545/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
3sequential_57/batch_normalization_545/batchnorm/addAddV2Fsequential_57/batch_normalization_545/batchnorm/ReadVariableOp:value:0>sequential_57/batch_normalization_545/batchnorm/add/y:output:0*
T0*
_output_shapes
:,?
5sequential_57/batch_normalization_545/batchnorm/RsqrtRsqrt7sequential_57/batch_normalization_545/batchnorm/add:z:0*
T0*
_output_shapes
:,?
Bsequential_57/batch_normalization_545/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_57_batch_normalization_545_batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0?
3sequential_57/batch_normalization_545/batchnorm/mulMul9sequential_57/batch_normalization_545/batchnorm/Rsqrt:y:0Jsequential_57/batch_normalization_545/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,?
5sequential_57/batch_normalization_545/batchnorm/mul_1Mul(sequential_57/dense_602/BiasAdd:output:07sequential_57/batch_normalization_545/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,?
@sequential_57/batch_normalization_545/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_57_batch_normalization_545_batchnorm_readvariableop_1_resource*
_output_shapes
:,*
dtype0?
5sequential_57/batch_normalization_545/batchnorm/mul_2MulHsequential_57/batch_normalization_545/batchnorm/ReadVariableOp_1:value:07sequential_57/batch_normalization_545/batchnorm/mul:z:0*
T0*
_output_shapes
:,?
@sequential_57/batch_normalization_545/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_57_batch_normalization_545_batchnorm_readvariableop_2_resource*
_output_shapes
:,*
dtype0?
3sequential_57/batch_normalization_545/batchnorm/subSubHsequential_57/batch_normalization_545/batchnorm/ReadVariableOp_2:value:09sequential_57/batch_normalization_545/batchnorm/mul_2:z:0*
T0*
_output_shapes
:,?
5sequential_57/batch_normalization_545/batchnorm/add_1AddV29sequential_57/batch_normalization_545/batchnorm/mul_1:z:07sequential_57/batch_normalization_545/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,?
'sequential_57/leaky_re_lu_545/LeakyRelu	LeakyRelu9sequential_57/batch_normalization_545/batchnorm/add_1:z:0*'
_output_shapes
:?????????,*
alpha%???>?
-sequential_57/dense_603/MatMul/ReadVariableOpReadVariableOp6sequential_57_dense_603_matmul_readvariableop_resource*
_output_shapes

:,2*
dtype0?
sequential_57/dense_603/MatMulMatMul5sequential_57/leaky_re_lu_545/LeakyRelu:activations:05sequential_57/dense_603/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
.sequential_57/dense_603/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_603_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
sequential_57/dense_603/BiasAddBiasAdd(sequential_57/dense_603/MatMul:product:06sequential_57/dense_603/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
>sequential_57/batch_normalization_546/batchnorm/ReadVariableOpReadVariableOpGsequential_57_batch_normalization_546_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0z
5sequential_57/batch_normalization_546/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
3sequential_57/batch_normalization_546/batchnorm/addAddV2Fsequential_57/batch_normalization_546/batchnorm/ReadVariableOp:value:0>sequential_57/batch_normalization_546/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
5sequential_57/batch_normalization_546/batchnorm/RsqrtRsqrt7sequential_57/batch_normalization_546/batchnorm/add:z:0*
T0*
_output_shapes
:2?
Bsequential_57/batch_normalization_546/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_57_batch_normalization_546_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0?
3sequential_57/batch_normalization_546/batchnorm/mulMul9sequential_57/batch_normalization_546/batchnorm/Rsqrt:y:0Jsequential_57/batch_normalization_546/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
5sequential_57/batch_normalization_546/batchnorm/mul_1Mul(sequential_57/dense_603/BiasAdd:output:07sequential_57/batch_normalization_546/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2?
@sequential_57/batch_normalization_546/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_57_batch_normalization_546_batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0?
5sequential_57/batch_normalization_546/batchnorm/mul_2MulHsequential_57/batch_normalization_546/batchnorm/ReadVariableOp_1:value:07sequential_57/batch_normalization_546/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
@sequential_57/batch_normalization_546/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_57_batch_normalization_546_batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0?
3sequential_57/batch_normalization_546/batchnorm/subSubHsequential_57/batch_normalization_546/batchnorm/ReadVariableOp_2:value:09sequential_57/batch_normalization_546/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
5sequential_57/batch_normalization_546/batchnorm/add_1AddV29sequential_57/batch_normalization_546/batchnorm/mul_1:z:07sequential_57/batch_normalization_546/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2?
'sequential_57/leaky_re_lu_546/LeakyRelu	LeakyRelu9sequential_57/batch_normalization_546/batchnorm/add_1:z:0*'
_output_shapes
:?????????2*
alpha%???>?
-sequential_57/dense_604/MatMul/ReadVariableOpReadVariableOp6sequential_57_dense_604_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
sequential_57/dense_604/MatMulMatMul5sequential_57/leaky_re_lu_546/LeakyRelu:activations:05sequential_57/dense_604/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
.sequential_57/dense_604/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_604_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
sequential_57/dense_604/BiasAddBiasAdd(sequential_57/dense_604/MatMul:product:06sequential_57/dense_604/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
>sequential_57/batch_normalization_547/batchnorm/ReadVariableOpReadVariableOpGsequential_57_batch_normalization_547_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0z
5sequential_57/batch_normalization_547/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
3sequential_57/batch_normalization_547/batchnorm/addAddV2Fsequential_57/batch_normalization_547/batchnorm/ReadVariableOp:value:0>sequential_57/batch_normalization_547/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
5sequential_57/batch_normalization_547/batchnorm/RsqrtRsqrt7sequential_57/batch_normalization_547/batchnorm/add:z:0*
T0*
_output_shapes
:2?
Bsequential_57/batch_normalization_547/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_57_batch_normalization_547_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0?
3sequential_57/batch_normalization_547/batchnorm/mulMul9sequential_57/batch_normalization_547/batchnorm/Rsqrt:y:0Jsequential_57/batch_normalization_547/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
5sequential_57/batch_normalization_547/batchnorm/mul_1Mul(sequential_57/dense_604/BiasAdd:output:07sequential_57/batch_normalization_547/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2?
@sequential_57/batch_normalization_547/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_57_batch_normalization_547_batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0?
5sequential_57/batch_normalization_547/batchnorm/mul_2MulHsequential_57/batch_normalization_547/batchnorm/ReadVariableOp_1:value:07sequential_57/batch_normalization_547/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
@sequential_57/batch_normalization_547/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_57_batch_normalization_547_batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0?
3sequential_57/batch_normalization_547/batchnorm/subSubHsequential_57/batch_normalization_547/batchnorm/ReadVariableOp_2:value:09sequential_57/batch_normalization_547/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
5sequential_57/batch_normalization_547/batchnorm/add_1AddV29sequential_57/batch_normalization_547/batchnorm/mul_1:z:07sequential_57/batch_normalization_547/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2?
'sequential_57/leaky_re_lu_547/LeakyRelu	LeakyRelu9sequential_57/batch_normalization_547/batchnorm/add_1:z:0*'
_output_shapes
:?????????2*
alpha%???>?
-sequential_57/dense_605/MatMul/ReadVariableOpReadVariableOp6sequential_57_dense_605_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
sequential_57/dense_605/MatMulMatMul5sequential_57/leaky_re_lu_547/LeakyRelu:activations:05sequential_57/dense_605/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.sequential_57/dense_605/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_605_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_57/dense_605/BiasAddBiasAdd(sequential_57/dense_605/MatMul:product:06sequential_57/dense_605/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
IdentityIdentity(sequential_57/dense_605/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp?^sequential_57/batch_normalization_542/batchnorm/ReadVariableOpA^sequential_57/batch_normalization_542/batchnorm/ReadVariableOp_1A^sequential_57/batch_normalization_542/batchnorm/ReadVariableOp_2C^sequential_57/batch_normalization_542/batchnorm/mul/ReadVariableOp?^sequential_57/batch_normalization_543/batchnorm/ReadVariableOpA^sequential_57/batch_normalization_543/batchnorm/ReadVariableOp_1A^sequential_57/batch_normalization_543/batchnorm/ReadVariableOp_2C^sequential_57/batch_normalization_543/batchnorm/mul/ReadVariableOp?^sequential_57/batch_normalization_544/batchnorm/ReadVariableOpA^sequential_57/batch_normalization_544/batchnorm/ReadVariableOp_1A^sequential_57/batch_normalization_544/batchnorm/ReadVariableOp_2C^sequential_57/batch_normalization_544/batchnorm/mul/ReadVariableOp?^sequential_57/batch_normalization_545/batchnorm/ReadVariableOpA^sequential_57/batch_normalization_545/batchnorm/ReadVariableOp_1A^sequential_57/batch_normalization_545/batchnorm/ReadVariableOp_2C^sequential_57/batch_normalization_545/batchnorm/mul/ReadVariableOp?^sequential_57/batch_normalization_546/batchnorm/ReadVariableOpA^sequential_57/batch_normalization_546/batchnorm/ReadVariableOp_1A^sequential_57/batch_normalization_546/batchnorm/ReadVariableOp_2C^sequential_57/batch_normalization_546/batchnorm/mul/ReadVariableOp?^sequential_57/batch_normalization_547/batchnorm/ReadVariableOpA^sequential_57/batch_normalization_547/batchnorm/ReadVariableOp_1A^sequential_57/batch_normalization_547/batchnorm/ReadVariableOp_2C^sequential_57/batch_normalization_547/batchnorm/mul/ReadVariableOp/^sequential_57/dense_599/BiasAdd/ReadVariableOp.^sequential_57/dense_599/MatMul/ReadVariableOp/^sequential_57/dense_600/BiasAdd/ReadVariableOp.^sequential_57/dense_600/MatMul/ReadVariableOp/^sequential_57/dense_601/BiasAdd/ReadVariableOp.^sequential_57/dense_601/MatMul/ReadVariableOp/^sequential_57/dense_602/BiasAdd/ReadVariableOp.^sequential_57/dense_602/MatMul/ReadVariableOp/^sequential_57/dense_603/BiasAdd/ReadVariableOp.^sequential_57/dense_603/MatMul/ReadVariableOp/^sequential_57/dense_604/BiasAdd/ReadVariableOp.^sequential_57/dense_604/MatMul/ReadVariableOp/^sequential_57/dense_605/BiasAdd/ReadVariableOp.^sequential_57/dense_605/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
>sequential_57/batch_normalization_542/batchnorm/ReadVariableOp>sequential_57/batch_normalization_542/batchnorm/ReadVariableOp2?
@sequential_57/batch_normalization_542/batchnorm/ReadVariableOp_1@sequential_57/batch_normalization_542/batchnorm/ReadVariableOp_12?
@sequential_57/batch_normalization_542/batchnorm/ReadVariableOp_2@sequential_57/batch_normalization_542/batchnorm/ReadVariableOp_22?
Bsequential_57/batch_normalization_542/batchnorm/mul/ReadVariableOpBsequential_57/batch_normalization_542/batchnorm/mul/ReadVariableOp2?
>sequential_57/batch_normalization_543/batchnorm/ReadVariableOp>sequential_57/batch_normalization_543/batchnorm/ReadVariableOp2?
@sequential_57/batch_normalization_543/batchnorm/ReadVariableOp_1@sequential_57/batch_normalization_543/batchnorm/ReadVariableOp_12?
@sequential_57/batch_normalization_543/batchnorm/ReadVariableOp_2@sequential_57/batch_normalization_543/batchnorm/ReadVariableOp_22?
Bsequential_57/batch_normalization_543/batchnorm/mul/ReadVariableOpBsequential_57/batch_normalization_543/batchnorm/mul/ReadVariableOp2?
>sequential_57/batch_normalization_544/batchnorm/ReadVariableOp>sequential_57/batch_normalization_544/batchnorm/ReadVariableOp2?
@sequential_57/batch_normalization_544/batchnorm/ReadVariableOp_1@sequential_57/batch_normalization_544/batchnorm/ReadVariableOp_12?
@sequential_57/batch_normalization_544/batchnorm/ReadVariableOp_2@sequential_57/batch_normalization_544/batchnorm/ReadVariableOp_22?
Bsequential_57/batch_normalization_544/batchnorm/mul/ReadVariableOpBsequential_57/batch_normalization_544/batchnorm/mul/ReadVariableOp2?
>sequential_57/batch_normalization_545/batchnorm/ReadVariableOp>sequential_57/batch_normalization_545/batchnorm/ReadVariableOp2?
@sequential_57/batch_normalization_545/batchnorm/ReadVariableOp_1@sequential_57/batch_normalization_545/batchnorm/ReadVariableOp_12?
@sequential_57/batch_normalization_545/batchnorm/ReadVariableOp_2@sequential_57/batch_normalization_545/batchnorm/ReadVariableOp_22?
Bsequential_57/batch_normalization_545/batchnorm/mul/ReadVariableOpBsequential_57/batch_normalization_545/batchnorm/mul/ReadVariableOp2?
>sequential_57/batch_normalization_546/batchnorm/ReadVariableOp>sequential_57/batch_normalization_546/batchnorm/ReadVariableOp2?
@sequential_57/batch_normalization_546/batchnorm/ReadVariableOp_1@sequential_57/batch_normalization_546/batchnorm/ReadVariableOp_12?
@sequential_57/batch_normalization_546/batchnorm/ReadVariableOp_2@sequential_57/batch_normalization_546/batchnorm/ReadVariableOp_22?
Bsequential_57/batch_normalization_546/batchnorm/mul/ReadVariableOpBsequential_57/batch_normalization_546/batchnorm/mul/ReadVariableOp2?
>sequential_57/batch_normalization_547/batchnorm/ReadVariableOp>sequential_57/batch_normalization_547/batchnorm/ReadVariableOp2?
@sequential_57/batch_normalization_547/batchnorm/ReadVariableOp_1@sequential_57/batch_normalization_547/batchnorm/ReadVariableOp_12?
@sequential_57/batch_normalization_547/batchnorm/ReadVariableOp_2@sequential_57/batch_normalization_547/batchnorm/ReadVariableOp_22?
Bsequential_57/batch_normalization_547/batchnorm/mul/ReadVariableOpBsequential_57/batch_normalization_547/batchnorm/mul/ReadVariableOp2`
.sequential_57/dense_599/BiasAdd/ReadVariableOp.sequential_57/dense_599/BiasAdd/ReadVariableOp2^
-sequential_57/dense_599/MatMul/ReadVariableOp-sequential_57/dense_599/MatMul/ReadVariableOp2`
.sequential_57/dense_600/BiasAdd/ReadVariableOp.sequential_57/dense_600/BiasAdd/ReadVariableOp2^
-sequential_57/dense_600/MatMul/ReadVariableOp-sequential_57/dense_600/MatMul/ReadVariableOp2`
.sequential_57/dense_601/BiasAdd/ReadVariableOp.sequential_57/dense_601/BiasAdd/ReadVariableOp2^
-sequential_57/dense_601/MatMul/ReadVariableOp-sequential_57/dense_601/MatMul/ReadVariableOp2`
.sequential_57/dense_602/BiasAdd/ReadVariableOp.sequential_57/dense_602/BiasAdd/ReadVariableOp2^
-sequential_57/dense_602/MatMul/ReadVariableOp-sequential_57/dense_602/MatMul/ReadVariableOp2`
.sequential_57/dense_603/BiasAdd/ReadVariableOp.sequential_57/dense_603/BiasAdd/ReadVariableOp2^
-sequential_57/dense_603/MatMul/ReadVariableOp-sequential_57/dense_603/MatMul/ReadVariableOp2`
.sequential_57/dense_604/BiasAdd/ReadVariableOp.sequential_57/dense_604/BiasAdd/ReadVariableOp2^
-sequential_57/dense_604/MatMul/ReadVariableOp-sequential_57/dense_604/MatMul/ReadVariableOp2`
.sequential_57/dense_605/BiasAdd/ReadVariableOp.sequential_57/dense_605/BiasAdd/ReadVariableOp2^
-sequential_57/dense_605/MatMul/ReadVariableOp-sequential_57/dense_605/MatMul/ReadVariableOp:_ [
'
_output_shapes
:?????????
0
_user_specified_namenormalization_57_input:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
*__inference_dense_599_layer_call_fn_776478

inputs
unknown:J
	unknown_0:J
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_599_layer_call_and_return_conditional_losses_774806o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????J`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_777079

inputs/
!batchnorm_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:21
#batchnorm_readvariableop_1_resource:21
#batchnorm_readvariableop_2_resource:2
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
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
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
E__inference_dense_602_layer_call_and_return_conditional_losses_776815

inputs0
matmul_readvariableop_resource:,,-
biasadd_readvariableop_resource:,
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:,,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:,*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????,w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????,: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_542_layer_call_fn_776501

inputs
unknown:J
	unknown_0:J
	unknown_1:J
	unknown_2:J
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_774314o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????J`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_544_layer_call_fn_776732

inputs
unknown:,
	unknown_0:,
	unknown_1:,
	unknown_2:,
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_774525o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????,`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_547_layer_call_fn_777118

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
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_547_layer_call_and_return_conditional_losses_774986`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
*__inference_dense_604_layer_call_fn_777023

inputs
unknown:22
	unknown_0:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_604_layer_call_and_return_conditional_losses_774966o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_774642

inputs/
!batchnorm_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:21
#batchnorm_readvariableop_1_resource:21
#batchnorm_readvariableop_2_resource:2
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
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
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_547_layer_call_fn_777059

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_774771o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_542_layer_call_and_return_conditional_losses_776578

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????J*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????J"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????J:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
?
*__inference_dense_600_layer_call_fn_776587

inputs
unknown:JJ
	unknown_0:J
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_600_layer_call_and_return_conditional_losses_774838o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????J`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?	
?
E__inference_dense_600_layer_call_and_return_conditional_losses_774838

inputs0
matmul_readvariableop_resource:JJ-
biasadd_readvariableop_resource:J
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:JJ*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Jr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Jw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_776534

inputs/
!batchnorm_readvariableop_resource:J3
%batchnorm_mul_readvariableop_resource:J1
#batchnorm_readvariableop_1_resource:J1
#batchnorm_readvariableop_2_resource:J
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:J*
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
:JP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:J~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Jc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????Jz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:J*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Jz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:J*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Jr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????Jb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????J?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
?
.__inference_sequential_57_layer_call_fn_775941

inputs
unknown
	unknown_0
	unknown_1:J
	unknown_2:J
	unknown_3:J
	unknown_4:J
	unknown_5:J
	unknown_6:J
	unknown_7:JJ
	unknown_8:J
	unknown_9:J

unknown_10:J

unknown_11:J

unknown_12:J

unknown_13:J,

unknown_14:,

unknown_15:,

unknown_16:,

unknown_17:,

unknown_18:,

unknown_19:,,

unknown_20:,

unknown_21:,

unknown_22:,

unknown_23:,

unknown_24:,

unknown_25:,2

unknown_26:2

unknown_27:2

unknown_28:2

unknown_29:2

unknown_30:2

unknown_31:22

unknown_32:2

unknown_33:2

unknown_34:2

unknown_35:2

unknown_36:2

unknown_37:2

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
I__inference_sequential_57_layer_call_and_return_conditional_losses_775387o
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
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
*__inference_dense_603_layer_call_fn_776914

inputs
unknown:,2
	unknown_0:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_603_layer_call_and_return_conditional_losses_774934o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????,: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_776786

inputs5
'assignmovingavg_readvariableop_resource:,7
)assignmovingavg_1_readvariableop_resource:,3
%batchnorm_mul_readvariableop_resource:,/
!batchnorm_readvariableop_resource:,
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:,*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:,?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????,l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:,*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:,*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:,*
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
:,*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:,x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:,?
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
:,*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:,~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:,?
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
:,P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:,~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:,v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:,*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:,r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????,?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_545_layer_call_fn_776841

inputs
unknown:,
	unknown_0:,
	unknown_1:,
	unknown_2:,
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_774607o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????,`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_547_layer_call_fn_777046

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_774724o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_774724

inputs/
!batchnorm_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:21
#batchnorm_readvariableop_1_resource:21
#batchnorm_readvariableop_2_resource:2
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
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
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
E__inference_dense_605_layer_call_and_return_conditional_losses_777142

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
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
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?	
.__inference_sequential_57_layer_call_fn_775088
normalization_57_input
unknown
	unknown_0
	unknown_1:J
	unknown_2:J
	unknown_3:J
	unknown_4:J
	unknown_5:J
	unknown_6:J
	unknown_7:JJ
	unknown_8:J
	unknown_9:J

unknown_10:J

unknown_11:J

unknown_12:J

unknown_13:J,

unknown_14:,

unknown_15:,

unknown_16:,

unknown_17:,

unknown_18:,

unknown_19:,,

unknown_20:,

unknown_21:,

unknown_22:,

unknown_23:,

unknown_24:,

unknown_25:,2

unknown_26:2

unknown_27:2

unknown_28:2

unknown_29:2

unknown_30:2

unknown_31:22

unknown_32:2

unknown_33:2

unknown_34:2

unknown_35:2

unknown_36:2

unknown_37:2

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallnormalization_57_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_57_layer_call_and_return_conditional_losses_775005o
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
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:?????????
0
_user_specified_namenormalization_57_input:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
*__inference_dense_601_layer_call_fn_776696

inputs
unknown:J,
	unknown_0:,
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_601_layer_call_and_return_conditional_losses_774870o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????,`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_543_layer_call_and_return_conditional_losses_774858

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????J*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????J"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????J:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_774689

inputs5
'assignmovingavg_readvariableop_resource:27
)assignmovingavg_1_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:2/
!batchnorm_readvariableop_resource:2
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:2*
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
:2*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2?
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
:2*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2?
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
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?i
?
I__inference_sequential_57_layer_call_and_return_conditional_losses_775005

inputs
normalization_57_sub_y
normalization_57_sqrt_x"
dense_599_774807:J
dense_599_774809:J,
batch_normalization_542_774812:J,
batch_normalization_542_774814:J,
batch_normalization_542_774816:J,
batch_normalization_542_774818:J"
dense_600_774839:JJ
dense_600_774841:J,
batch_normalization_543_774844:J,
batch_normalization_543_774846:J,
batch_normalization_543_774848:J,
batch_normalization_543_774850:J"
dense_601_774871:J,
dense_601_774873:,,
batch_normalization_544_774876:,,
batch_normalization_544_774878:,,
batch_normalization_544_774880:,,
batch_normalization_544_774882:,"
dense_602_774903:,,
dense_602_774905:,,
batch_normalization_545_774908:,,
batch_normalization_545_774910:,,
batch_normalization_545_774912:,,
batch_normalization_545_774914:,"
dense_603_774935:,2
dense_603_774937:2,
batch_normalization_546_774940:2,
batch_normalization_546_774942:2,
batch_normalization_546_774944:2,
batch_normalization_546_774946:2"
dense_604_774967:22
dense_604_774969:2,
batch_normalization_547_774972:2,
batch_normalization_547_774974:2,
batch_normalization_547_774976:2,
batch_normalization_547_774978:2"
dense_605_774999:2
dense_605_775001:
identity??/batch_normalization_542/StatefulPartitionedCall?/batch_normalization_543/StatefulPartitionedCall?/batch_normalization_544/StatefulPartitionedCall?/batch_normalization_545/StatefulPartitionedCall?/batch_normalization_546/StatefulPartitionedCall?/batch_normalization_547/StatefulPartitionedCall?!dense_599/StatefulPartitionedCall?!dense_600/StatefulPartitionedCall?!dense_601/StatefulPartitionedCall?!dense_602/StatefulPartitionedCall?!dense_603/StatefulPartitionedCall?!dense_604/StatefulPartitionedCall?!dense_605/StatefulPartitionedCallm
normalization_57/subSubinputsnormalization_57_sub_y*
T0*'
_output_shapes
:?????????_
normalization_57/SqrtSqrtnormalization_57_sqrt_x*
T0*
_output_shapes

:_
normalization_57/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_57/MaximumMaximumnormalization_57/Sqrt:y:0#normalization_57/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_57/truedivRealDivnormalization_57/sub:z:0normalization_57/Maximum:z:0*
T0*'
_output_shapes
:??????????
!dense_599/StatefulPartitionedCallStatefulPartitionedCallnormalization_57/truediv:z:0dense_599_774807dense_599_774809*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_599_layer_call_and_return_conditional_losses_774806?
/batch_normalization_542/StatefulPartitionedCallStatefulPartitionedCall*dense_599/StatefulPartitionedCall:output:0batch_normalization_542_774812batch_normalization_542_774814batch_normalization_542_774816batch_normalization_542_774818*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_774314?
leaky_re_lu_542/PartitionedCallPartitionedCall8batch_normalization_542/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_542_layer_call_and_return_conditional_losses_774826?
!dense_600/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_542/PartitionedCall:output:0dense_600_774839dense_600_774841*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_600_layer_call_and_return_conditional_losses_774838?
/batch_normalization_543/StatefulPartitionedCallStatefulPartitionedCall*dense_600/StatefulPartitionedCall:output:0batch_normalization_543_774844batch_normalization_543_774846batch_normalization_543_774848batch_normalization_543_774850*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_774396?
leaky_re_lu_543/PartitionedCallPartitionedCall8batch_normalization_543/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_543_layer_call_and_return_conditional_losses_774858?
!dense_601/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_543/PartitionedCall:output:0dense_601_774871dense_601_774873*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_601_layer_call_and_return_conditional_losses_774870?
/batch_normalization_544/StatefulPartitionedCallStatefulPartitionedCall*dense_601/StatefulPartitionedCall:output:0batch_normalization_544_774876batch_normalization_544_774878batch_normalization_544_774880batch_normalization_544_774882*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_774478?
leaky_re_lu_544/PartitionedCallPartitionedCall8batch_normalization_544/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_544_layer_call_and_return_conditional_losses_774890?
!dense_602/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_544/PartitionedCall:output:0dense_602_774903dense_602_774905*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_602_layer_call_and_return_conditional_losses_774902?
/batch_normalization_545/StatefulPartitionedCallStatefulPartitionedCall*dense_602/StatefulPartitionedCall:output:0batch_normalization_545_774908batch_normalization_545_774910batch_normalization_545_774912batch_normalization_545_774914*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_774560?
leaky_re_lu_545/PartitionedCallPartitionedCall8batch_normalization_545/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_545_layer_call_and_return_conditional_losses_774922?
!dense_603/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_545/PartitionedCall:output:0dense_603_774935dense_603_774937*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_603_layer_call_and_return_conditional_losses_774934?
/batch_normalization_546/StatefulPartitionedCallStatefulPartitionedCall*dense_603/StatefulPartitionedCall:output:0batch_normalization_546_774940batch_normalization_546_774942batch_normalization_546_774944batch_normalization_546_774946*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_774642?
leaky_re_lu_546/PartitionedCallPartitionedCall8batch_normalization_546/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_546_layer_call_and_return_conditional_losses_774954?
!dense_604/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_546/PartitionedCall:output:0dense_604_774967dense_604_774969*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_604_layer_call_and_return_conditional_losses_774966?
/batch_normalization_547/StatefulPartitionedCallStatefulPartitionedCall*dense_604/StatefulPartitionedCall:output:0batch_normalization_547_774972batch_normalization_547_774974batch_normalization_547_774976batch_normalization_547_774978*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_774724?
leaky_re_lu_547/PartitionedCallPartitionedCall8batch_normalization_547/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_547_layer_call_and_return_conditional_losses_774986?
!dense_605/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_547/PartitionedCall:output:0dense_605_774999dense_605_775001*
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
E__inference_dense_605_layer_call_and_return_conditional_losses_774998y
IdentityIdentity*dense_605/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp0^batch_normalization_542/StatefulPartitionedCall0^batch_normalization_543/StatefulPartitionedCall0^batch_normalization_544/StatefulPartitionedCall0^batch_normalization_545/StatefulPartitionedCall0^batch_normalization_546/StatefulPartitionedCall0^batch_normalization_547/StatefulPartitionedCall"^dense_599/StatefulPartitionedCall"^dense_600/StatefulPartitionedCall"^dense_601/StatefulPartitionedCall"^dense_602/StatefulPartitionedCall"^dense_603/StatefulPartitionedCall"^dense_604/StatefulPartitionedCall"^dense_605/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_542/StatefulPartitionedCall/batch_normalization_542/StatefulPartitionedCall2b
/batch_normalization_543/StatefulPartitionedCall/batch_normalization_543/StatefulPartitionedCall2b
/batch_normalization_544/StatefulPartitionedCall/batch_normalization_544/StatefulPartitionedCall2b
/batch_normalization_545/StatefulPartitionedCall/batch_normalization_545/StatefulPartitionedCall2b
/batch_normalization_546/StatefulPartitionedCall/batch_normalization_546/StatefulPartitionedCall2b
/batch_normalization_547/StatefulPartitionedCall/batch_normalization_547/StatefulPartitionedCall2F
!dense_599/StatefulPartitionedCall!dense_599/StatefulPartitionedCall2F
!dense_600/StatefulPartitionedCall!dense_600/StatefulPartitionedCall2F
!dense_601/StatefulPartitionedCall!dense_601/StatefulPartitionedCall2F
!dense_602/StatefulPartitionedCall!dense_602/StatefulPartitionedCall2F
!dense_603/StatefulPartitionedCall!dense_603/StatefulPartitionedCall2F
!dense_604/StatefulPartitionedCall!dense_604/StatefulPartitionedCall2F
!dense_605/StatefulPartitionedCall!dense_605/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_776643

inputs/
!batchnorm_readvariableop_resource:J3
%batchnorm_mul_readvariableop_resource:J1
#batchnorm_readvariableop_1_resource:J1
#batchnorm_readvariableop_2_resource:J
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:J*
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
:JP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:J~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Jc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????Jz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:J*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Jz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:J*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Jr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????Jb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????J?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_774525

inputs5
'assignmovingavg_readvariableop_resource:,7
)assignmovingavg_1_readvariableop_resource:,3
%batchnorm_mul_readvariableop_resource:,/
!batchnorm_readvariableop_resource:,
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:,*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:,?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????,l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:,*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:,*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:,*
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
:,*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:,x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:,?
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
:,*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:,~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:,?
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
:,P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:,~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:,v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:,*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:,r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????,?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_776568

inputs5
'assignmovingavg_readvariableop_resource:J7
)assignmovingavg_1_readvariableop_resource:J3
%batchnorm_mul_readvariableop_resource:J/
!batchnorm_readvariableop_resource:J
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:J*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:J?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????Jl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:J*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:J*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:J*
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
:J*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Jx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:J?
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
:J*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:J~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:J?
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
:JP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:J~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Jc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????Jh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Jv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Jr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????Jb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????J?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
?
*__inference_dense_602_layer_call_fn_776805

inputs
unknown:,,
	unknown_0:,
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_602_layer_call_and_return_conditional_losses_774902o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????,`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????,: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_547_layer_call_and_return_conditional_losses_777123

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????2*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
E__inference_dense_604_layer_call_and_return_conditional_losses_774966

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_774560

inputs/
!batchnorm_readvariableop_resource:,3
%batchnorm_mul_readvariableop_resource:,1
#batchnorm_readvariableop_1_resource:,1
#batchnorm_readvariableop_2_resource:,
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:,*
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
:,P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:,~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:,*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:,c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????,z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:,*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:,z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:,*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:,r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????,b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????,?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?	
?
E__inference_dense_601_layer_call_and_return_conditional_losses_776706

inputs0
matmul_readvariableop_resource:J,-
biasadd_readvariableop_resource:,
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:J,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:,*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????,_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????,w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_774771

inputs5
'assignmovingavg_readvariableop_resource:27
)assignmovingavg_1_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:2/
!batchnorm_readvariableop_resource:2
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:2*
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
:2*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2?
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
:2*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2?
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
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_546_layer_call_fn_776950

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_774689o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_544_layer_call_fn_776791

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
:?????????,* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_544_layer_call_and_return_conditional_losses_774890`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????,"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????,:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_776422
normalization_57_input
unknown
	unknown_0
	unknown_1:J
	unknown_2:J
	unknown_3:J
	unknown_4:J
	unknown_5:J
	unknown_6:J
	unknown_7:JJ
	unknown_8:J
	unknown_9:J

unknown_10:J

unknown_11:J

unknown_12:J

unknown_13:J,

unknown_14:,

unknown_15:,

unknown_16:,

unknown_17:,

unknown_18:,

unknown_19:,,

unknown_20:,

unknown_21:,

unknown_22:,

unknown_23:,

unknown_24:,

unknown_25:,2

unknown_26:2

unknown_27:2

unknown_28:2

unknown_29:2

unknown_30:2

unknown_31:22

unknown_32:2

unknown_33:2

unknown_34:2

unknown_35:2

unknown_36:2

unknown_37:2

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallnormalization_57_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_774290o
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
s:?????????::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:?????????
0
_user_specified_namenormalization_57_input:$ 

_output_shapes

::$ 

_output_shapes

:
?
?
8__inference_batch_normalization_545_layer_call_fn_776828

inputs
unknown:,
	unknown_0:,
	unknown_1:,
	unknown_2:,
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????,*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_774560o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????,`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_774443

inputs5
'assignmovingavg_readvariableop_resource:J7
)assignmovingavg_1_readvariableop_resource:J3
%batchnorm_mul_readvariableop_resource:J/
!batchnorm_readvariableop_resource:J
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:J*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:J?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????Jl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:J*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:J*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:J*
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
:J*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Jx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:J?
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
:J*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:J~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:J?
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
:JP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:J~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Jc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????Jh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Jv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Jr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????Jb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????J?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_546_layer_call_and_return_conditional_losses_777014

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????2*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
E__inference_dense_604_layer_call_and_return_conditional_losses_777033

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_543_layer_call_fn_776610

inputs
unknown:J
	unknown_0:J
	unknown_1:J
	unknown_2:J
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????J*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_774396o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????J`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_545_layer_call_and_return_conditional_losses_774922

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????,*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????,"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????,:O K
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_543_layer_call_fn_776682

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
:?????????J* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_543_layer_call_and_return_conditional_losses_774858`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????J"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????J:O K
'
_output_shapes
:?????????J
 
_user_specified_nameinputs
?	
?
E__inference_dense_599_layer_call_and_return_conditional_losses_774806

inputs0
matmul_readvariableop_resource:J-
biasadd_readvariableop_resource:J
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:J*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Jr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????J_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Jw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_774361

inputs5
'assignmovingavg_readvariableop_resource:J7
)assignmovingavg_1_readvariableop_resource:J3
%batchnorm_mul_readvariableop_resource:J/
!batchnorm_readvariableop_resource:J
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:J*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:J?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????Jl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:J*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:J*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:J*
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
:J*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Jx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:J?
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
:J*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:J~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:J?
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
:JP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:J~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:J*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Jc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????Jh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Jv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Jr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????Jb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????J?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????J
 
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
normalization_57_input?
(serving_default_normalization_57_input:0?????????=
	dense_6050
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
 "
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
.__inference_sequential_57_layer_call_fn_775088
.__inference_sequential_57_layer_call_fn_775856
.__inference_sequential_57_layer_call_fn_775941
.__inference_sequential_57_layer_call_fn_775555?
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
I__inference_sequential_57_layer_call_and_return_conditional_losses_776096
I__inference_sequential_57_layer_call_and_return_conditional_losses_776335
I__inference_sequential_57_layer_call_and_return_conditional_losses_775661
I__inference_sequential_57_layer_call_and_return_conditional_losses_775767?
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
!__inference__wrapped_model_774290normalization_57_input"?
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
:2mean
:2variance
:	 2count
"
_generic_user_object
?2?
__inference_adapt_step_776469?
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
": J2dense_599/kernel
:J2dense_599/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
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
*__inference_dense_599_layer_call_fn_776478?
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
E__inference_dense_599_layer_call_and_return_conditional_losses_776488?
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
+:)J2batch_normalization_542/gamma
*:(J2batch_normalization_542/beta
3:1J (2#batch_normalization_542/moving_mean
7:5J (2'batch_normalization_542/moving_variance
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
8__inference_batch_normalization_542_layer_call_fn_776501
8__inference_batch_normalization_542_layer_call_fn_776514?
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
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_776534
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_776568?
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
0__inference_leaky_re_lu_542_layer_call_fn_776573?
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
K__inference_leaky_re_lu_542_layer_call_and_return_conditional_losses_776578?
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
": JJ2dense_600/kernel
:J2dense_600/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
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
*__inference_dense_600_layer_call_fn_776587?
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
E__inference_dense_600_layer_call_and_return_conditional_losses_776597?
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
+:)J2batch_normalization_543/gamma
*:(J2batch_normalization_543/beta
3:1J (2#batch_normalization_543/moving_mean
7:5J (2'batch_normalization_543/moving_variance
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
8__inference_batch_normalization_543_layer_call_fn_776610
8__inference_batch_normalization_543_layer_call_fn_776623?
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
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_776643
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_776677?
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
0__inference_leaky_re_lu_543_layer_call_fn_776682?
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
K__inference_leaky_re_lu_543_layer_call_and_return_conditional_losses_776687?
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
": J,2dense_601/kernel
:,2dense_601/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
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
*__inference_dense_601_layer_call_fn_776696?
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
E__inference_dense_601_layer_call_and_return_conditional_losses_776706?
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
+:),2batch_normalization_544/gamma
*:(,2batch_normalization_544/beta
3:1, (2#batch_normalization_544/moving_mean
7:5, (2'batch_normalization_544/moving_variance
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
8__inference_batch_normalization_544_layer_call_fn_776719
8__inference_batch_normalization_544_layer_call_fn_776732?
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
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_776752
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_776786?
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
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_leaky_re_lu_544_layer_call_fn_776791?
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
K__inference_leaky_re_lu_544_layer_call_and_return_conditional_losses_776796?
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
": ,,2dense_602/kernel
:,2dense_602/bias
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_602_layer_call_fn_776805?
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
E__inference_dense_602_layer_call_and_return_conditional_losses_776815?
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
+:),2batch_normalization_545/gamma
*:(,2batch_normalization_545/beta
3:1, (2#batch_normalization_545/moving_mean
7:5, (2'batch_normalization_545/moving_variance
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
8__inference_batch_normalization_545_layer_call_fn_776828
8__inference_batch_normalization_545_layer_call_fn_776841?
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
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_776861
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_776895?
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
0__inference_leaky_re_lu_545_layer_call_fn_776900?
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
K__inference_leaky_re_lu_545_layer_call_and_return_conditional_losses_776905?
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
": ,22dense_603/kernel
:22dense_603/bias
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
*__inference_dense_603_layer_call_fn_776914?
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
E__inference_dense_603_layer_call_and_return_conditional_losses_776924?
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
+:)22batch_normalization_546/gamma
*:(22batch_normalization_546/beta
3:12 (2#batch_normalization_546/moving_mean
7:52 (2'batch_normalization_546/moving_variance
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
8__inference_batch_normalization_546_layer_call_fn_776937
8__inference_batch_normalization_546_layer_call_fn_776950?
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
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_776970
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_777004?
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
0__inference_leaky_re_lu_546_layer_call_fn_777009?
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
K__inference_leaky_re_lu_546_layer_call_and_return_conditional_losses_777014?
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
": 222dense_604/kernel
:22dense_604/bias
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
*__inference_dense_604_layer_call_fn_777023?
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
E__inference_dense_604_layer_call_and_return_conditional_losses_777033?
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
+:)22batch_normalization_547/gamma
*:(22batch_normalization_547/beta
3:12 (2#batch_normalization_547/moving_mean
7:52 (2'batch_normalization_547/moving_variance
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
8__inference_batch_normalization_547_layer_call_fn_777046
8__inference_batch_normalization_547_layer_call_fn_777059?
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
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_777079
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_777113?
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
0__inference_leaky_re_lu_547_layer_call_fn_777118?
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
K__inference_leaky_re_lu_547_layer_call_and_return_conditional_losses_777123?
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
": 22dense_605/kernel
:2dense_605/bias
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
*__inference_dense_605_layer_call_fn_777132?
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
E__inference_dense_605_layer_call_and_return_conditional_losses_777142?
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
$__inference_signature_wrapper_776422normalization_57_input"?
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
 "
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
 "
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
 "
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
 "
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
 "
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
':%J2Adam/dense_599/kernel/m
!:J2Adam/dense_599/bias/m
0:.J2$Adam/batch_normalization_542/gamma/m
/:-J2#Adam/batch_normalization_542/beta/m
':%JJ2Adam/dense_600/kernel/m
!:J2Adam/dense_600/bias/m
0:.J2$Adam/batch_normalization_543/gamma/m
/:-J2#Adam/batch_normalization_543/beta/m
':%J,2Adam/dense_601/kernel/m
!:,2Adam/dense_601/bias/m
0:.,2$Adam/batch_normalization_544/gamma/m
/:-,2#Adam/batch_normalization_544/beta/m
':%,,2Adam/dense_602/kernel/m
!:,2Adam/dense_602/bias/m
0:.,2$Adam/batch_normalization_545/gamma/m
/:-,2#Adam/batch_normalization_545/beta/m
':%,22Adam/dense_603/kernel/m
!:22Adam/dense_603/bias/m
0:.22$Adam/batch_normalization_546/gamma/m
/:-22#Adam/batch_normalization_546/beta/m
':%222Adam/dense_604/kernel/m
!:22Adam/dense_604/bias/m
0:.22$Adam/batch_normalization_547/gamma/m
/:-22#Adam/batch_normalization_547/beta/m
':%22Adam/dense_605/kernel/m
!:2Adam/dense_605/bias/m
':%J2Adam/dense_599/kernel/v
!:J2Adam/dense_599/bias/v
0:.J2$Adam/batch_normalization_542/gamma/v
/:-J2#Adam/batch_normalization_542/beta/v
':%JJ2Adam/dense_600/kernel/v
!:J2Adam/dense_600/bias/v
0:.J2$Adam/batch_normalization_543/gamma/v
/:-J2#Adam/batch_normalization_543/beta/v
':%J,2Adam/dense_601/kernel/v
!:,2Adam/dense_601/bias/v
0:.,2$Adam/batch_normalization_544/gamma/v
/:-,2#Adam/batch_normalization_544/beta/v
':%,,2Adam/dense_602/kernel/v
!:,2Adam/dense_602/bias/v
0:.,2$Adam/batch_normalization_545/gamma/v
/:-,2#Adam/batch_normalization_545/beta/v
':%,22Adam/dense_603/kernel/v
!:22Adam/dense_603/bias/v
0:.22$Adam/batch_normalization_546/gamma/v
/:-22#Adam/batch_normalization_546/beta/v
':%222Adam/dense_604/kernel/v
!:22Adam/dense_604/bias/v
0:.22$Adam/batch_normalization_547/gamma/v
/:-22#Adam/batch_normalization_547/beta/v
':%22Adam/dense_605/kernel/v
!:2Adam/dense_605/bias/v
	J
Const
J	
Const_1?
!__inference__wrapped_model_774290?8??'(3021@ALIKJYZebdcrs~{}|????????????????<
5?2
0?-
normalization_57_input?????????
? "5?2
0
	dense_605#? 
	dense_605?????????o
__inference_adapt_step_776469N$"#C?@
9?6
4?1?
??????????IteratorSpec 
? "
 ?
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_776534b30213?0
)?&
 ?
inputs?????????J
p 
? "%?"
?
0?????????J
? ?
S__inference_batch_normalization_542_layer_call_and_return_conditional_losses_776568b23013?0
)?&
 ?
inputs?????????J
p
? "%?"
?
0?????????J
? ?
8__inference_batch_normalization_542_layer_call_fn_776501U30213?0
)?&
 ?
inputs?????????J
p 
? "??????????J?
8__inference_batch_normalization_542_layer_call_fn_776514U23013?0
)?&
 ?
inputs?????????J
p
? "??????????J?
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_776643bLIKJ3?0
)?&
 ?
inputs?????????J
p 
? "%?"
?
0?????????J
? ?
S__inference_batch_normalization_543_layer_call_and_return_conditional_losses_776677bKLIJ3?0
)?&
 ?
inputs?????????J
p
? "%?"
?
0?????????J
? ?
8__inference_batch_normalization_543_layer_call_fn_776610ULIKJ3?0
)?&
 ?
inputs?????????J
p 
? "??????????J?
8__inference_batch_normalization_543_layer_call_fn_776623UKLIJ3?0
)?&
 ?
inputs?????????J
p
? "??????????J?
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_776752bebdc3?0
)?&
 ?
inputs?????????,
p 
? "%?"
?
0?????????,
? ?
S__inference_batch_normalization_544_layer_call_and_return_conditional_losses_776786bdebc3?0
)?&
 ?
inputs?????????,
p
? "%?"
?
0?????????,
? ?
8__inference_batch_normalization_544_layer_call_fn_776719Uebdc3?0
)?&
 ?
inputs?????????,
p 
? "??????????,?
8__inference_batch_normalization_544_layer_call_fn_776732Udebc3?0
)?&
 ?
inputs?????????,
p
? "??????????,?
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_776861b~{}|3?0
)?&
 ?
inputs?????????,
p 
? "%?"
?
0?????????,
? ?
S__inference_batch_normalization_545_layer_call_and_return_conditional_losses_776895b}~{|3?0
)?&
 ?
inputs?????????,
p
? "%?"
?
0?????????,
? ?
8__inference_batch_normalization_545_layer_call_fn_776828U~{}|3?0
)?&
 ?
inputs?????????,
p 
? "??????????,?
8__inference_batch_normalization_545_layer_call_fn_776841U}~{|3?0
)?&
 ?
inputs?????????,
p
? "??????????,?
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_776970f????3?0
)?&
 ?
inputs?????????2
p 
? "%?"
?
0?????????2
? ?
S__inference_batch_normalization_546_layer_call_and_return_conditional_losses_777004f????3?0
)?&
 ?
inputs?????????2
p
? "%?"
?
0?????????2
? ?
8__inference_batch_normalization_546_layer_call_fn_776937Y????3?0
)?&
 ?
inputs?????????2
p 
? "??????????2?
8__inference_batch_normalization_546_layer_call_fn_776950Y????3?0
)?&
 ?
inputs?????????2
p
? "??????????2?
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_777079f????3?0
)?&
 ?
inputs?????????2
p 
? "%?"
?
0?????????2
? ?
S__inference_batch_normalization_547_layer_call_and_return_conditional_losses_777113f????3?0
)?&
 ?
inputs?????????2
p
? "%?"
?
0?????????2
? ?
8__inference_batch_normalization_547_layer_call_fn_777046Y????3?0
)?&
 ?
inputs?????????2
p 
? "??????????2?
8__inference_batch_normalization_547_layer_call_fn_777059Y????3?0
)?&
 ?
inputs?????????2
p
? "??????????2?
E__inference_dense_599_layer_call_and_return_conditional_losses_776488\'(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????J
? }
*__inference_dense_599_layer_call_fn_776478O'(/?,
%?"
 ?
inputs?????????
? "??????????J?
E__inference_dense_600_layer_call_and_return_conditional_losses_776597\@A/?,
%?"
 ?
inputs?????????J
? "%?"
?
0?????????J
? }
*__inference_dense_600_layer_call_fn_776587O@A/?,
%?"
 ?
inputs?????????J
? "??????????J?
E__inference_dense_601_layer_call_and_return_conditional_losses_776706\YZ/?,
%?"
 ?
inputs?????????J
? "%?"
?
0?????????,
? }
*__inference_dense_601_layer_call_fn_776696OYZ/?,
%?"
 ?
inputs?????????J
? "??????????,?
E__inference_dense_602_layer_call_and_return_conditional_losses_776815\rs/?,
%?"
 ?
inputs?????????,
? "%?"
?
0?????????,
? }
*__inference_dense_602_layer_call_fn_776805Ors/?,
%?"
 ?
inputs?????????,
? "??????????,?
E__inference_dense_603_layer_call_and_return_conditional_losses_776924^??/?,
%?"
 ?
inputs?????????,
? "%?"
?
0?????????2
? 
*__inference_dense_603_layer_call_fn_776914Q??/?,
%?"
 ?
inputs?????????,
? "??????????2?
E__inference_dense_604_layer_call_and_return_conditional_losses_777033^??/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? 
*__inference_dense_604_layer_call_fn_777023Q??/?,
%?"
 ?
inputs?????????2
? "??????????2?
E__inference_dense_605_layer_call_and_return_conditional_losses_777142^??/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? 
*__inference_dense_605_layer_call_fn_777132Q??/?,
%?"
 ?
inputs?????????2
? "???????????
K__inference_leaky_re_lu_542_layer_call_and_return_conditional_losses_776578X/?,
%?"
 ?
inputs?????????J
? "%?"
?
0?????????J
? 
0__inference_leaky_re_lu_542_layer_call_fn_776573K/?,
%?"
 ?
inputs?????????J
? "??????????J?
K__inference_leaky_re_lu_543_layer_call_and_return_conditional_losses_776687X/?,
%?"
 ?
inputs?????????J
? "%?"
?
0?????????J
? 
0__inference_leaky_re_lu_543_layer_call_fn_776682K/?,
%?"
 ?
inputs?????????J
? "??????????J?
K__inference_leaky_re_lu_544_layer_call_and_return_conditional_losses_776796X/?,
%?"
 ?
inputs?????????,
? "%?"
?
0?????????,
? 
0__inference_leaky_re_lu_544_layer_call_fn_776791K/?,
%?"
 ?
inputs?????????,
? "??????????,?
K__inference_leaky_re_lu_545_layer_call_and_return_conditional_losses_776905X/?,
%?"
 ?
inputs?????????,
? "%?"
?
0?????????,
? 
0__inference_leaky_re_lu_545_layer_call_fn_776900K/?,
%?"
 ?
inputs?????????,
? "??????????,?
K__inference_leaky_re_lu_546_layer_call_and_return_conditional_losses_777014X/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? 
0__inference_leaky_re_lu_546_layer_call_fn_777009K/?,
%?"
 ?
inputs?????????2
? "??????????2?
K__inference_leaky_re_lu_547_layer_call_and_return_conditional_losses_777123X/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? 
0__inference_leaky_re_lu_547_layer_call_fn_777118K/?,
%?"
 ?
inputs?????????2
? "??????????2?
I__inference_sequential_57_layer_call_and_return_conditional_losses_775661?8??'(3021@ALIKJYZebdcrs~{}|??????????????G?D
=?:
0?-
normalization_57_input?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_57_layer_call_and_return_conditional_losses_775767?8??'(2301@AKLIJYZdebcrs}~{|??????????????G?D
=?:
0?-
normalization_57_input?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_57_layer_call_and_return_conditional_losses_776096?8??'(3021@ALIKJYZebdcrs~{}|??????????????7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_57_layer_call_and_return_conditional_losses_776335?8??'(2301@AKLIJYZdebcrs}~{|??????????????7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
.__inference_sequential_57_layer_call_fn_775088?8??'(3021@ALIKJYZebdcrs~{}|??????????????G?D
=?:
0?-
normalization_57_input?????????
p 

 
? "???????????
.__inference_sequential_57_layer_call_fn_775555?8??'(2301@AKLIJYZdebcrs}~{|??????????????G?D
=?:
0?-
normalization_57_input?????????
p

 
? "???????????
.__inference_sequential_57_layer_call_fn_775856?8??'(3021@ALIKJYZebdcrs~{}|??????????????7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
.__inference_sequential_57_layer_call_fn_775941?8??'(2301@AKLIJYZdebcrs}~{|??????????????7?4
-?*
 ?
inputs?????????
p

 
? "???????????
$__inference_signature_wrapper_776422?8??'(3021@ALIKJYZebdcrs~{}|??????????????Y?V
? 
O?L
J
normalization_57_input0?-
normalization_57_input?????????"5?2
0
	dense_605#? 
	dense_605?????????