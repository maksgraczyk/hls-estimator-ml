ò'
ß
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
alphafloat%ÍÌL>"
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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
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
list(type)(0
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
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68$
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
dense_588/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Y*!
shared_namedense_588/kernel
u
$dense_588/kernel/Read/ReadVariableOpReadVariableOpdense_588/kernel*
_output_shapes

:Y*
dtype0
t
dense_588/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*
shared_namedense_588/bias
m
"dense_588/bias/Read/ReadVariableOpReadVariableOpdense_588/bias*
_output_shapes
:Y*
dtype0

batch_normalization_533/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*.
shared_namebatch_normalization_533/gamma

1batch_normalization_533/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_533/gamma*
_output_shapes
:Y*
dtype0

batch_normalization_533/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*-
shared_namebatch_normalization_533/beta

0batch_normalization_533/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_533/beta*
_output_shapes
:Y*
dtype0

#batch_normalization_533/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*4
shared_name%#batch_normalization_533/moving_mean

7batch_normalization_533/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_533/moving_mean*
_output_shapes
:Y*
dtype0
¦
'batch_normalization_533/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*8
shared_name)'batch_normalization_533/moving_variance

;batch_normalization_533/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_533/moving_variance*
_output_shapes
:Y*
dtype0
|
dense_589/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:YY*!
shared_namedense_589/kernel
u
$dense_589/kernel/Read/ReadVariableOpReadVariableOpdense_589/kernel*
_output_shapes

:YY*
dtype0
t
dense_589/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*
shared_namedense_589/bias
m
"dense_589/bias/Read/ReadVariableOpReadVariableOpdense_589/bias*
_output_shapes
:Y*
dtype0

batch_normalization_534/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*.
shared_namebatch_normalization_534/gamma

1batch_normalization_534/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_534/gamma*
_output_shapes
:Y*
dtype0

batch_normalization_534/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*-
shared_namebatch_normalization_534/beta

0batch_normalization_534/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_534/beta*
_output_shapes
:Y*
dtype0

#batch_normalization_534/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*4
shared_name%#batch_normalization_534/moving_mean

7batch_normalization_534/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_534/moving_mean*
_output_shapes
:Y*
dtype0
¦
'batch_normalization_534/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*8
shared_name)'batch_normalization_534/moving_variance

;batch_normalization_534/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_534/moving_variance*
_output_shapes
:Y*
dtype0
|
dense_590/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Y*!
shared_namedense_590/kernel
u
$dense_590/kernel/Read/ReadVariableOpReadVariableOpdense_590/kernel*
_output_shapes

:Y*
dtype0
t
dense_590/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_590/bias
m
"dense_590/bias/Read/ReadVariableOpReadVariableOpdense_590/bias*
_output_shapes
:*
dtype0

batch_normalization_535/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_535/gamma

1batch_normalization_535/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_535/gamma*
_output_shapes
:*
dtype0

batch_normalization_535/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_535/beta

0batch_normalization_535/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_535/beta*
_output_shapes
:*
dtype0

#batch_normalization_535/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_535/moving_mean

7batch_normalization_535/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_535/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_535/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_535/moving_variance

;batch_normalization_535/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_535/moving_variance*
_output_shapes
:*
dtype0
|
dense_591/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_591/kernel
u
$dense_591/kernel/Read/ReadVariableOpReadVariableOpdense_591/kernel*
_output_shapes

:*
dtype0
t
dense_591/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_591/bias
m
"dense_591/bias/Read/ReadVariableOpReadVariableOpdense_591/bias*
_output_shapes
:*
dtype0

batch_normalization_536/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_536/gamma

1batch_normalization_536/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_536/gamma*
_output_shapes
:*
dtype0

batch_normalization_536/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_536/beta

0batch_normalization_536/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_536/beta*
_output_shapes
:*
dtype0

#batch_normalization_536/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_536/moving_mean

7batch_normalization_536/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_536/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_536/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_536/moving_variance

;batch_normalization_536/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_536/moving_variance*
_output_shapes
:*
dtype0
|
dense_592/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_592/kernel
u
$dense_592/kernel/Read/ReadVariableOpReadVariableOpdense_592/kernel*
_output_shapes

:*
dtype0
t
dense_592/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_592/bias
m
"dense_592/bias/Read/ReadVariableOpReadVariableOpdense_592/bias*
_output_shapes
:*
dtype0

batch_normalization_537/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_537/gamma

1batch_normalization_537/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_537/gamma*
_output_shapes
:*
dtype0

batch_normalization_537/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_537/beta

0batch_normalization_537/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_537/beta*
_output_shapes
:*
dtype0

#batch_normalization_537/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_537/moving_mean

7batch_normalization_537/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_537/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_537/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_537/moving_variance

;batch_normalization_537/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_537/moving_variance*
_output_shapes
:*
dtype0
|
dense_593/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_593/kernel
u
$dense_593/kernel/Read/ReadVariableOpReadVariableOpdense_593/kernel*
_output_shapes

:*
dtype0
t
dense_593/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_593/bias
m
"dense_593/bias/Read/ReadVariableOpReadVariableOpdense_593/bias*
_output_shapes
:*
dtype0

batch_normalization_538/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_538/gamma

1batch_normalization_538/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_538/gamma*
_output_shapes
:*
dtype0

batch_normalization_538/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_538/beta

0batch_normalization_538/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_538/beta*
_output_shapes
:*
dtype0

#batch_normalization_538/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_538/moving_mean

7batch_normalization_538/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_538/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_538/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_538/moving_variance

;batch_normalization_538/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_538/moving_variance*
_output_shapes
:*
dtype0
|
dense_594/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:E*!
shared_namedense_594/kernel
u
$dense_594/kernel/Read/ReadVariableOpReadVariableOpdense_594/kernel*
_output_shapes

:E*
dtype0
t
dense_594/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*
shared_namedense_594/bias
m
"dense_594/bias/Read/ReadVariableOpReadVariableOpdense_594/bias*
_output_shapes
:E*
dtype0

batch_normalization_539/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*.
shared_namebatch_normalization_539/gamma

1batch_normalization_539/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_539/gamma*
_output_shapes
:E*
dtype0

batch_normalization_539/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*-
shared_namebatch_normalization_539/beta

0batch_normalization_539/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_539/beta*
_output_shapes
:E*
dtype0

#batch_normalization_539/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*4
shared_name%#batch_normalization_539/moving_mean

7batch_normalization_539/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_539/moving_mean*
_output_shapes
:E*
dtype0
¦
'batch_normalization_539/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*8
shared_name)'batch_normalization_539/moving_variance

;batch_normalization_539/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_539/moving_variance*
_output_shapes
:E*
dtype0
|
dense_595/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:EE*!
shared_namedense_595/kernel
u
$dense_595/kernel/Read/ReadVariableOpReadVariableOpdense_595/kernel*
_output_shapes

:EE*
dtype0
t
dense_595/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*
shared_namedense_595/bias
m
"dense_595/bias/Read/ReadVariableOpReadVariableOpdense_595/bias*
_output_shapes
:E*
dtype0

batch_normalization_540/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*.
shared_namebatch_normalization_540/gamma

1batch_normalization_540/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_540/gamma*
_output_shapes
:E*
dtype0

batch_normalization_540/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*-
shared_namebatch_normalization_540/beta

0batch_normalization_540/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_540/beta*
_output_shapes
:E*
dtype0

#batch_normalization_540/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*4
shared_name%#batch_normalization_540/moving_mean

7batch_normalization_540/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_540/moving_mean*
_output_shapes
:E*
dtype0
¦
'batch_normalization_540/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*8
shared_name)'batch_normalization_540/moving_variance

;batch_normalization_540/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_540/moving_variance*
_output_shapes
:E*
dtype0
|
dense_596/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:E*!
shared_namedense_596/kernel
u
$dense_596/kernel/Read/ReadVariableOpReadVariableOpdense_596/kernel*
_output_shapes

:E*
dtype0
t
dense_596/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_596/bias
m
"dense_596/bias/Read/ReadVariableOpReadVariableOpdense_596/bias*
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

Adam/dense_588/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Y*(
shared_nameAdam/dense_588/kernel/m

+Adam/dense_588/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_588/kernel/m*
_output_shapes

:Y*
dtype0

Adam/dense_588/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*&
shared_nameAdam/dense_588/bias/m
{
)Adam/dense_588/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_588/bias/m*
_output_shapes
:Y*
dtype0
 
$Adam/batch_normalization_533/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*5
shared_name&$Adam/batch_normalization_533/gamma/m

8Adam/batch_normalization_533/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_533/gamma/m*
_output_shapes
:Y*
dtype0

#Adam/batch_normalization_533/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*4
shared_name%#Adam/batch_normalization_533/beta/m

7Adam/batch_normalization_533/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_533/beta/m*
_output_shapes
:Y*
dtype0

Adam/dense_589/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:YY*(
shared_nameAdam/dense_589/kernel/m

+Adam/dense_589/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_589/kernel/m*
_output_shapes

:YY*
dtype0

Adam/dense_589/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*&
shared_nameAdam/dense_589/bias/m
{
)Adam/dense_589/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_589/bias/m*
_output_shapes
:Y*
dtype0
 
$Adam/batch_normalization_534/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*5
shared_name&$Adam/batch_normalization_534/gamma/m

8Adam/batch_normalization_534/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_534/gamma/m*
_output_shapes
:Y*
dtype0

#Adam/batch_normalization_534/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*4
shared_name%#Adam/batch_normalization_534/beta/m

7Adam/batch_normalization_534/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_534/beta/m*
_output_shapes
:Y*
dtype0

Adam/dense_590/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Y*(
shared_nameAdam/dense_590/kernel/m

+Adam/dense_590/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_590/kernel/m*
_output_shapes

:Y*
dtype0

Adam/dense_590/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_590/bias/m
{
)Adam/dense_590/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_590/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_535/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_535/gamma/m

8Adam/batch_normalization_535/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_535/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_535/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_535/beta/m

7Adam/batch_normalization_535/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_535/beta/m*
_output_shapes
:*
dtype0

Adam/dense_591/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_591/kernel/m

+Adam/dense_591/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_591/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_591/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_591/bias/m
{
)Adam/dense_591/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_591/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_536/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_536/gamma/m

8Adam/batch_normalization_536/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_536/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_536/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_536/beta/m

7Adam/batch_normalization_536/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_536/beta/m*
_output_shapes
:*
dtype0

Adam/dense_592/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_592/kernel/m

+Adam/dense_592/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_592/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_592/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_592/bias/m
{
)Adam/dense_592/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_592/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_537/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_537/gamma/m

8Adam/batch_normalization_537/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_537/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_537/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_537/beta/m

7Adam/batch_normalization_537/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_537/beta/m*
_output_shapes
:*
dtype0

Adam/dense_593/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_593/kernel/m

+Adam/dense_593/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_593/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_593/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_593/bias/m
{
)Adam/dense_593/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_593/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_538/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_538/gamma/m

8Adam/batch_normalization_538/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_538/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_538/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_538/beta/m

7Adam/batch_normalization_538/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_538/beta/m*
_output_shapes
:*
dtype0

Adam/dense_594/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:E*(
shared_nameAdam/dense_594/kernel/m

+Adam/dense_594/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_594/kernel/m*
_output_shapes

:E*
dtype0

Adam/dense_594/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*&
shared_nameAdam/dense_594/bias/m
{
)Adam/dense_594/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_594/bias/m*
_output_shapes
:E*
dtype0
 
$Adam/batch_normalization_539/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*5
shared_name&$Adam/batch_normalization_539/gamma/m

8Adam/batch_normalization_539/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_539/gamma/m*
_output_shapes
:E*
dtype0

#Adam/batch_normalization_539/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*4
shared_name%#Adam/batch_normalization_539/beta/m

7Adam/batch_normalization_539/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_539/beta/m*
_output_shapes
:E*
dtype0

Adam/dense_595/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:EE*(
shared_nameAdam/dense_595/kernel/m

+Adam/dense_595/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_595/kernel/m*
_output_shapes

:EE*
dtype0

Adam/dense_595/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*&
shared_nameAdam/dense_595/bias/m
{
)Adam/dense_595/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_595/bias/m*
_output_shapes
:E*
dtype0
 
$Adam/batch_normalization_540/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*5
shared_name&$Adam/batch_normalization_540/gamma/m

8Adam/batch_normalization_540/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_540/gamma/m*
_output_shapes
:E*
dtype0

#Adam/batch_normalization_540/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*4
shared_name%#Adam/batch_normalization_540/beta/m

7Adam/batch_normalization_540/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_540/beta/m*
_output_shapes
:E*
dtype0

Adam/dense_596/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:E*(
shared_nameAdam/dense_596/kernel/m

+Adam/dense_596/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_596/kernel/m*
_output_shapes

:E*
dtype0

Adam/dense_596/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_596/bias/m
{
)Adam/dense_596/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_596/bias/m*
_output_shapes
:*
dtype0

Adam/dense_588/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Y*(
shared_nameAdam/dense_588/kernel/v

+Adam/dense_588/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_588/kernel/v*
_output_shapes

:Y*
dtype0

Adam/dense_588/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*&
shared_nameAdam/dense_588/bias/v
{
)Adam/dense_588/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_588/bias/v*
_output_shapes
:Y*
dtype0
 
$Adam/batch_normalization_533/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*5
shared_name&$Adam/batch_normalization_533/gamma/v

8Adam/batch_normalization_533/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_533/gamma/v*
_output_shapes
:Y*
dtype0

#Adam/batch_normalization_533/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*4
shared_name%#Adam/batch_normalization_533/beta/v

7Adam/batch_normalization_533/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_533/beta/v*
_output_shapes
:Y*
dtype0

Adam/dense_589/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:YY*(
shared_nameAdam/dense_589/kernel/v

+Adam/dense_589/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_589/kernel/v*
_output_shapes

:YY*
dtype0

Adam/dense_589/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*&
shared_nameAdam/dense_589/bias/v
{
)Adam/dense_589/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_589/bias/v*
_output_shapes
:Y*
dtype0
 
$Adam/batch_normalization_534/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*5
shared_name&$Adam/batch_normalization_534/gamma/v

8Adam/batch_normalization_534/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_534/gamma/v*
_output_shapes
:Y*
dtype0

#Adam/batch_normalization_534/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Y*4
shared_name%#Adam/batch_normalization_534/beta/v

7Adam/batch_normalization_534/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_534/beta/v*
_output_shapes
:Y*
dtype0

Adam/dense_590/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Y*(
shared_nameAdam/dense_590/kernel/v

+Adam/dense_590/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_590/kernel/v*
_output_shapes

:Y*
dtype0

Adam/dense_590/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_590/bias/v
{
)Adam/dense_590/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_590/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_535/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_535/gamma/v

8Adam/batch_normalization_535/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_535/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_535/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_535/beta/v

7Adam/batch_normalization_535/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_535/beta/v*
_output_shapes
:*
dtype0

Adam/dense_591/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_591/kernel/v

+Adam/dense_591/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_591/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_591/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_591/bias/v
{
)Adam/dense_591/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_591/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_536/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_536/gamma/v

8Adam/batch_normalization_536/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_536/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_536/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_536/beta/v

7Adam/batch_normalization_536/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_536/beta/v*
_output_shapes
:*
dtype0

Adam/dense_592/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_592/kernel/v

+Adam/dense_592/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_592/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_592/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_592/bias/v
{
)Adam/dense_592/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_592/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_537/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_537/gamma/v

8Adam/batch_normalization_537/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_537/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_537/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_537/beta/v

7Adam/batch_normalization_537/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_537/beta/v*
_output_shapes
:*
dtype0

Adam/dense_593/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_593/kernel/v

+Adam/dense_593/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_593/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_593/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_593/bias/v
{
)Adam/dense_593/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_593/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_538/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_538/gamma/v

8Adam/batch_normalization_538/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_538/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_538/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_538/beta/v

7Adam/batch_normalization_538/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_538/beta/v*
_output_shapes
:*
dtype0

Adam/dense_594/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:E*(
shared_nameAdam/dense_594/kernel/v

+Adam/dense_594/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_594/kernel/v*
_output_shapes

:E*
dtype0

Adam/dense_594/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*&
shared_nameAdam/dense_594/bias/v
{
)Adam/dense_594/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_594/bias/v*
_output_shapes
:E*
dtype0
 
$Adam/batch_normalization_539/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*5
shared_name&$Adam/batch_normalization_539/gamma/v

8Adam/batch_normalization_539/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_539/gamma/v*
_output_shapes
:E*
dtype0

#Adam/batch_normalization_539/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*4
shared_name%#Adam/batch_normalization_539/beta/v

7Adam/batch_normalization_539/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_539/beta/v*
_output_shapes
:E*
dtype0

Adam/dense_595/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:EE*(
shared_nameAdam/dense_595/kernel/v

+Adam/dense_595/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_595/kernel/v*
_output_shapes

:EE*
dtype0

Adam/dense_595/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*&
shared_nameAdam/dense_595/bias/v
{
)Adam/dense_595/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_595/bias/v*
_output_shapes
:E*
dtype0
 
$Adam/batch_normalization_540/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*5
shared_name&$Adam/batch_normalization_540/gamma/v

8Adam/batch_normalization_540/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_540/gamma/v*
_output_shapes
:E*
dtype0

#Adam/batch_normalization_540/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:E*4
shared_name%#Adam/batch_normalization_540/beta/v

7Adam/batch_normalization_540/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_540/beta/v*
_output_shapes
:E*
dtype0

Adam/dense_596/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:E*(
shared_nameAdam/dense_596/kernel/v

+Adam/dense_596/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_596/kernel/v*
_output_shapes

:E*
dtype0

Adam/dense_596/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_596/bias/v
{
)Adam/dense_596/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_596/bias/v*
_output_shapes
:*
dtype0
f
ConstConst*
_output_shapes

:*
dtype0*)
value B"UUéB  A  0@  XA
h
Const_1Const*
_output_shapes

:*
dtype0*)
value B"4sE ·B  @  yB

NoOpNoOp
ä
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B

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
layer_with_weights-14
layer-20
layer-21
layer_with_weights-15
layer-22
layer_with_weights-16
layer-23
layer-24
layer_with_weights-17
layer-25
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_default_save_signature
#
signatures*
¾
$
_keep_axis
%_reduce_axis
&_reduce_axis_mask
'_broadcast_shape
(mean
(
adapt_mean
)variance
)adapt_variance
	*count
+	keras_api
,_adapt_function*
¦

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses*
Õ
5axis
	6gamma
7beta
8moving_mean
9moving_variance
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*

@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses* 
¦

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses*
Õ
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses*

Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses* 
¦

_kernel
`bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses*
Õ
gaxis
	hgamma
ibeta
jmoving_mean
kmoving_variance
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses*

r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses* 
¦

xkernel
ybias
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
 regularization_losses
¡	keras_api
¢__call__
+£&call_and_return_all_conditional_losses*

¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses* 
®
ªkernel
	«bias
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses*
à
	²axis

³gamma
	´beta
µmoving_mean
¶moving_variance
·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
»__call__
+¼&call_and_return_all_conditional_losses*

½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses* 
®
Ãkernel
	Äbias
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses*
à
	Ëaxis

Ìgamma
	Íbeta
Îmoving_mean
Ïmoving_variance
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ó	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses*

Ö	variables
×trainable_variables
Øregularization_losses
Ù	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses* 
®
Ükernel
	Ýbias
Þ	variables
ßtrainable_variables
àregularization_losses
á	keras_api
â__call__
+ã&call_and_return_all_conditional_losses*
à
	äaxis

ågamma
	æbeta
çmoving_mean
èmoving_variance
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
í__call__
+î&call_and_return_all_conditional_losses*

ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses* 
®
õkernel
	öbias
÷	variables
øtrainable_variables
ùregularization_losses
ú	keras_api
û__call__
+ü&call_and_return_all_conditional_losses*

	ýiter
þbeta_1
ÿbeta_2

decay-m.m6m7mFmGmOmPm_m`mhmimxmym	m	m	m	m	m	m	ªm	«m	³m	´m 	Ãm¡	Äm¢	Ìm£	Ím¤	Üm¥	Ým¦	åm§	æm¨	õm©	ömª-v«.v¬6v­7v®Fv¯Gv°Ov±Pv²_v³`v´hvµiv¶xv·yv¸	v¹	vº	v»	v¼	v½	v¾	ªv¿	«vÀ	³vÁ	´vÂ	ÃvÃ	ÄvÄ	ÌvÅ	ÍvÆ	ÜvÇ	ÝvÈ	åvÉ	ævÊ	õvË	övÌ*
À
(0
)1
*2
-3
.4
65
76
87
98
F9
G10
O11
P12
Q13
R14
_15
`16
h17
i18
j19
k20
x21
y22
23
24
25
26
27
28
29
30
31
32
ª33
«34
³35
´36
µ37
¶38
Ã39
Ä40
Ì41
Í42
Î43
Ï44
Ü45
Ý46
å47
æ48
ç49
è50
õ51
ö52*

-0
.1
62
73
F4
G5
O6
P7
_8
`9
h10
i11
x12
y13
14
15
16
17
18
19
ª20
«21
³22
´23
Ã24
Ä25
Ì26
Í27
Ü28
Ý29
å30
æ31
õ32
ö33*
* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
"_default_save_signature
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
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
VARIABLE_VALUEdense_588/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_588/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_533/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_533/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_533/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_533/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
60
71
82
93*

60
71*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_589/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_589/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

F0
G1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_534/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_534/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_534/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_534/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
O0
P1
Q2
R3*

O0
P1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_590/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_590/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

_0
`1*

_0
`1*
* 

¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_535/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_535/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_535/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_535/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
h0
i1
j2
k3*

h0
i1*
* 

ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_591/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_591/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

x0
y1*

x0
y1*
* 

´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_536/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_536/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_536/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_536/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_592/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_592/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_537/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_537/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_537/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_537/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
	variables
trainable_variables
 regularization_losses
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_593/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_593/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

ª0
«1*

ª0
«1*
* 

Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_538/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_538/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_538/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_538/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
³0
´1
µ2
¶3*

³0
´1*
* 

×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
·	variables
¸trainable_variables
¹regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_594/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_594/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ã0
Ä1*

Ã0
Ä1*
* 

ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
Å	variables
Ætrainable_variables
Çregularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_539/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_539/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_539/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_539/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
Ì0
Í1
Î2
Ï3*

Ì0
Í1*
* 

ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
Ö	variables
×trainable_variables
Øregularization_losses
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_595/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_595/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ü0
Ý1*

Ü0
Ý1*
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
Þ	variables
ßtrainable_variables
àregularization_losses
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_540/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_540/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_540/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_540/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
å0
æ1
ç2
è3*

å0
æ1*
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
é	variables
êtrainable_variables
ëregularization_losses
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_596/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_596/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*

õ0
ö1*

õ0
ö1*
* 

ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
÷	variables
øtrainable_variables
ùregularization_losses
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses*
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

(0
)1
*2
83
94
Q5
R6
j7
k8
9
10
11
12
µ13
¶14
Î15
Ï16
ç17
è18*
Ê
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
19
20
21
22
23
24
25*

0*
* 
* 
* 
* 
* 
* 
* 
* 

80
91*
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
Q0
R1*
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
j0
k1*
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
0
1*
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
0
1*
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
µ0
¶1*
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
Î0
Ï1*
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
ç0
è1*
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

total

count
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
}
VARIABLE_VALUEAdam/dense_588/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_588/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_533/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_533/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_589/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_589/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_534/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_534/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_590/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_590/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_535/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_535/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_591/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_591/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_536/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_536/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_592/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_592/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_537/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_537/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_593/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_593/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_538/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_538/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_594/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_594/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_539/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_539/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_595/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_595/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_540/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_540/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_596/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_596/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_588/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_588/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_533/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_533/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_589/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_589/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_534/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_534/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_590/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_590/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_535/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_535/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_591/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_591/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_536/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_536/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_592/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_592/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_537/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_537/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_593/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_593/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_538/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_538/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_594/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_594/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_539/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_539/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_595/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_595/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_540/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_540/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_596/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_596/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

&serving_default_normalization_55_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Á
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_55_inputConstConst_1dense_588/kerneldense_588/bias'batch_normalization_533/moving_variancebatch_normalization_533/gamma#batch_normalization_533/moving_meanbatch_normalization_533/betadense_589/kerneldense_589/bias'batch_normalization_534/moving_variancebatch_normalization_534/gamma#batch_normalization_534/moving_meanbatch_normalization_534/betadense_590/kerneldense_590/bias'batch_normalization_535/moving_variancebatch_normalization_535/gamma#batch_normalization_535/moving_meanbatch_normalization_535/betadense_591/kerneldense_591/bias'batch_normalization_536/moving_variancebatch_normalization_536/gamma#batch_normalization_536/moving_meanbatch_normalization_536/betadense_592/kerneldense_592/bias'batch_normalization_537/moving_variancebatch_normalization_537/gamma#batch_normalization_537/moving_meanbatch_normalization_537/betadense_593/kerneldense_593/bias'batch_normalization_538/moving_variancebatch_normalization_538/gamma#batch_normalization_538/moving_meanbatch_normalization_538/betadense_594/kerneldense_594/bias'batch_normalization_539/moving_variancebatch_normalization_539/gamma#batch_normalization_539/moving_meanbatch_normalization_539/betadense_595/kerneldense_595/bias'batch_normalization_540/moving_variancebatch_normalization_540/gamma#batch_normalization_540/moving_meanbatch_normalization_540/betadense_596/kerneldense_596/bias*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_666469
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
þ2
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_588/kernel/Read/ReadVariableOp"dense_588/bias/Read/ReadVariableOp1batch_normalization_533/gamma/Read/ReadVariableOp0batch_normalization_533/beta/Read/ReadVariableOp7batch_normalization_533/moving_mean/Read/ReadVariableOp;batch_normalization_533/moving_variance/Read/ReadVariableOp$dense_589/kernel/Read/ReadVariableOp"dense_589/bias/Read/ReadVariableOp1batch_normalization_534/gamma/Read/ReadVariableOp0batch_normalization_534/beta/Read/ReadVariableOp7batch_normalization_534/moving_mean/Read/ReadVariableOp;batch_normalization_534/moving_variance/Read/ReadVariableOp$dense_590/kernel/Read/ReadVariableOp"dense_590/bias/Read/ReadVariableOp1batch_normalization_535/gamma/Read/ReadVariableOp0batch_normalization_535/beta/Read/ReadVariableOp7batch_normalization_535/moving_mean/Read/ReadVariableOp;batch_normalization_535/moving_variance/Read/ReadVariableOp$dense_591/kernel/Read/ReadVariableOp"dense_591/bias/Read/ReadVariableOp1batch_normalization_536/gamma/Read/ReadVariableOp0batch_normalization_536/beta/Read/ReadVariableOp7batch_normalization_536/moving_mean/Read/ReadVariableOp;batch_normalization_536/moving_variance/Read/ReadVariableOp$dense_592/kernel/Read/ReadVariableOp"dense_592/bias/Read/ReadVariableOp1batch_normalization_537/gamma/Read/ReadVariableOp0batch_normalization_537/beta/Read/ReadVariableOp7batch_normalization_537/moving_mean/Read/ReadVariableOp;batch_normalization_537/moving_variance/Read/ReadVariableOp$dense_593/kernel/Read/ReadVariableOp"dense_593/bias/Read/ReadVariableOp1batch_normalization_538/gamma/Read/ReadVariableOp0batch_normalization_538/beta/Read/ReadVariableOp7batch_normalization_538/moving_mean/Read/ReadVariableOp;batch_normalization_538/moving_variance/Read/ReadVariableOp$dense_594/kernel/Read/ReadVariableOp"dense_594/bias/Read/ReadVariableOp1batch_normalization_539/gamma/Read/ReadVariableOp0batch_normalization_539/beta/Read/ReadVariableOp7batch_normalization_539/moving_mean/Read/ReadVariableOp;batch_normalization_539/moving_variance/Read/ReadVariableOp$dense_595/kernel/Read/ReadVariableOp"dense_595/bias/Read/ReadVariableOp1batch_normalization_540/gamma/Read/ReadVariableOp0batch_normalization_540/beta/Read/ReadVariableOp7batch_normalization_540/moving_mean/Read/ReadVariableOp;batch_normalization_540/moving_variance/Read/ReadVariableOp$dense_596/kernel/Read/ReadVariableOp"dense_596/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_588/kernel/m/Read/ReadVariableOp)Adam/dense_588/bias/m/Read/ReadVariableOp8Adam/batch_normalization_533/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_533/beta/m/Read/ReadVariableOp+Adam/dense_589/kernel/m/Read/ReadVariableOp)Adam/dense_589/bias/m/Read/ReadVariableOp8Adam/batch_normalization_534/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_534/beta/m/Read/ReadVariableOp+Adam/dense_590/kernel/m/Read/ReadVariableOp)Adam/dense_590/bias/m/Read/ReadVariableOp8Adam/batch_normalization_535/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_535/beta/m/Read/ReadVariableOp+Adam/dense_591/kernel/m/Read/ReadVariableOp)Adam/dense_591/bias/m/Read/ReadVariableOp8Adam/batch_normalization_536/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_536/beta/m/Read/ReadVariableOp+Adam/dense_592/kernel/m/Read/ReadVariableOp)Adam/dense_592/bias/m/Read/ReadVariableOp8Adam/batch_normalization_537/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_537/beta/m/Read/ReadVariableOp+Adam/dense_593/kernel/m/Read/ReadVariableOp)Adam/dense_593/bias/m/Read/ReadVariableOp8Adam/batch_normalization_538/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_538/beta/m/Read/ReadVariableOp+Adam/dense_594/kernel/m/Read/ReadVariableOp)Adam/dense_594/bias/m/Read/ReadVariableOp8Adam/batch_normalization_539/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_539/beta/m/Read/ReadVariableOp+Adam/dense_595/kernel/m/Read/ReadVariableOp)Adam/dense_595/bias/m/Read/ReadVariableOp8Adam/batch_normalization_540/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_540/beta/m/Read/ReadVariableOp+Adam/dense_596/kernel/m/Read/ReadVariableOp)Adam/dense_596/bias/m/Read/ReadVariableOp+Adam/dense_588/kernel/v/Read/ReadVariableOp)Adam/dense_588/bias/v/Read/ReadVariableOp8Adam/batch_normalization_533/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_533/beta/v/Read/ReadVariableOp+Adam/dense_589/kernel/v/Read/ReadVariableOp)Adam/dense_589/bias/v/Read/ReadVariableOp8Adam/batch_normalization_534/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_534/beta/v/Read/ReadVariableOp+Adam/dense_590/kernel/v/Read/ReadVariableOp)Adam/dense_590/bias/v/Read/ReadVariableOp8Adam/batch_normalization_535/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_535/beta/v/Read/ReadVariableOp+Adam/dense_591/kernel/v/Read/ReadVariableOp)Adam/dense_591/bias/v/Read/ReadVariableOp8Adam/batch_normalization_536/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_536/beta/v/Read/ReadVariableOp+Adam/dense_592/kernel/v/Read/ReadVariableOp)Adam/dense_592/bias/v/Read/ReadVariableOp8Adam/batch_normalization_537/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_537/beta/v/Read/ReadVariableOp+Adam/dense_593/kernel/v/Read/ReadVariableOp)Adam/dense_593/bias/v/Read/ReadVariableOp8Adam/batch_normalization_538/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_538/beta/v/Read/ReadVariableOp+Adam/dense_594/kernel/v/Read/ReadVariableOp)Adam/dense_594/bias/v/Read/ReadVariableOp8Adam/batch_normalization_539/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_539/beta/v/Read/ReadVariableOp+Adam/dense_595/kernel/v/Read/ReadVariableOp)Adam/dense_595/bias/v/Read/ReadVariableOp8Adam/batch_normalization_540/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_540/beta/v/Read/ReadVariableOp+Adam/dense_596/kernel/v/Read/ReadVariableOp)Adam/dense_596/bias/v/Read/ReadVariableOpConst_2*
Tin
2		*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_667813

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_588/kerneldense_588/biasbatch_normalization_533/gammabatch_normalization_533/beta#batch_normalization_533/moving_mean'batch_normalization_533/moving_variancedense_589/kerneldense_589/biasbatch_normalization_534/gammabatch_normalization_534/beta#batch_normalization_534/moving_mean'batch_normalization_534/moving_variancedense_590/kerneldense_590/biasbatch_normalization_535/gammabatch_normalization_535/beta#batch_normalization_535/moving_mean'batch_normalization_535/moving_variancedense_591/kerneldense_591/biasbatch_normalization_536/gammabatch_normalization_536/beta#batch_normalization_536/moving_mean'batch_normalization_536/moving_variancedense_592/kerneldense_592/biasbatch_normalization_537/gammabatch_normalization_537/beta#batch_normalization_537/moving_mean'batch_normalization_537/moving_variancedense_593/kerneldense_593/biasbatch_normalization_538/gammabatch_normalization_538/beta#batch_normalization_538/moving_mean'batch_normalization_538/moving_variancedense_594/kerneldense_594/biasbatch_normalization_539/gammabatch_normalization_539/beta#batch_normalization_539/moving_mean'batch_normalization_539/moving_variancedense_595/kerneldense_595/biasbatch_normalization_540/gammabatch_normalization_540/beta#batch_normalization_540/moving_mean'batch_normalization_540/moving_variancedense_596/kerneldense_596/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_588/kernel/mAdam/dense_588/bias/m$Adam/batch_normalization_533/gamma/m#Adam/batch_normalization_533/beta/mAdam/dense_589/kernel/mAdam/dense_589/bias/m$Adam/batch_normalization_534/gamma/m#Adam/batch_normalization_534/beta/mAdam/dense_590/kernel/mAdam/dense_590/bias/m$Adam/batch_normalization_535/gamma/m#Adam/batch_normalization_535/beta/mAdam/dense_591/kernel/mAdam/dense_591/bias/m$Adam/batch_normalization_536/gamma/m#Adam/batch_normalization_536/beta/mAdam/dense_592/kernel/mAdam/dense_592/bias/m$Adam/batch_normalization_537/gamma/m#Adam/batch_normalization_537/beta/mAdam/dense_593/kernel/mAdam/dense_593/bias/m$Adam/batch_normalization_538/gamma/m#Adam/batch_normalization_538/beta/mAdam/dense_594/kernel/mAdam/dense_594/bias/m$Adam/batch_normalization_539/gamma/m#Adam/batch_normalization_539/beta/mAdam/dense_595/kernel/mAdam/dense_595/bias/m$Adam/batch_normalization_540/gamma/m#Adam/batch_normalization_540/beta/mAdam/dense_596/kernel/mAdam/dense_596/bias/mAdam/dense_588/kernel/vAdam/dense_588/bias/v$Adam/batch_normalization_533/gamma/v#Adam/batch_normalization_533/beta/vAdam/dense_589/kernel/vAdam/dense_589/bias/v$Adam/batch_normalization_534/gamma/v#Adam/batch_normalization_534/beta/vAdam/dense_590/kernel/vAdam/dense_590/bias/v$Adam/batch_normalization_535/gamma/v#Adam/batch_normalization_535/beta/vAdam/dense_591/kernel/vAdam/dense_591/bias/v$Adam/batch_normalization_536/gamma/v#Adam/batch_normalization_536/beta/vAdam/dense_592/kernel/vAdam/dense_592/bias/v$Adam/batch_normalization_537/gamma/v#Adam/batch_normalization_537/beta/vAdam/dense_593/kernel/vAdam/dense_593/bias/v$Adam/batch_normalization_538/gamma/v#Adam/batch_normalization_538/beta/vAdam/dense_594/kernel/vAdam/dense_594/bias/v$Adam/batch_normalization_539/gamma/v#Adam/batch_normalization_539/beta/vAdam/dense_595/kernel/vAdam/dense_595/bias/v$Adam/batch_normalization_540/gamma/v#Adam/batch_normalization_540/beta/vAdam/dense_596/kernel/vAdam/dense_596/bias/v*
Tin
2*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_668204
Ð
²
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_666908

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Èâ
§9
!__inference__wrapped_model_663699
normalization_55_input(
$sequential_55_normalization_55_sub_y)
%sequential_55_normalization_55_sqrt_xH
6sequential_55_dense_588_matmul_readvariableop_resource:YE
7sequential_55_dense_588_biasadd_readvariableop_resource:YU
Gsequential_55_batch_normalization_533_batchnorm_readvariableop_resource:YY
Ksequential_55_batch_normalization_533_batchnorm_mul_readvariableop_resource:YW
Isequential_55_batch_normalization_533_batchnorm_readvariableop_1_resource:YW
Isequential_55_batch_normalization_533_batchnorm_readvariableop_2_resource:YH
6sequential_55_dense_589_matmul_readvariableop_resource:YYE
7sequential_55_dense_589_biasadd_readvariableop_resource:YU
Gsequential_55_batch_normalization_534_batchnorm_readvariableop_resource:YY
Ksequential_55_batch_normalization_534_batchnorm_mul_readvariableop_resource:YW
Isequential_55_batch_normalization_534_batchnorm_readvariableop_1_resource:YW
Isequential_55_batch_normalization_534_batchnorm_readvariableop_2_resource:YH
6sequential_55_dense_590_matmul_readvariableop_resource:YE
7sequential_55_dense_590_biasadd_readvariableop_resource:U
Gsequential_55_batch_normalization_535_batchnorm_readvariableop_resource:Y
Ksequential_55_batch_normalization_535_batchnorm_mul_readvariableop_resource:W
Isequential_55_batch_normalization_535_batchnorm_readvariableop_1_resource:W
Isequential_55_batch_normalization_535_batchnorm_readvariableop_2_resource:H
6sequential_55_dense_591_matmul_readvariableop_resource:E
7sequential_55_dense_591_biasadd_readvariableop_resource:U
Gsequential_55_batch_normalization_536_batchnorm_readvariableop_resource:Y
Ksequential_55_batch_normalization_536_batchnorm_mul_readvariableop_resource:W
Isequential_55_batch_normalization_536_batchnorm_readvariableop_1_resource:W
Isequential_55_batch_normalization_536_batchnorm_readvariableop_2_resource:H
6sequential_55_dense_592_matmul_readvariableop_resource:E
7sequential_55_dense_592_biasadd_readvariableop_resource:U
Gsequential_55_batch_normalization_537_batchnorm_readvariableop_resource:Y
Ksequential_55_batch_normalization_537_batchnorm_mul_readvariableop_resource:W
Isequential_55_batch_normalization_537_batchnorm_readvariableop_1_resource:W
Isequential_55_batch_normalization_537_batchnorm_readvariableop_2_resource:H
6sequential_55_dense_593_matmul_readvariableop_resource:E
7sequential_55_dense_593_biasadd_readvariableop_resource:U
Gsequential_55_batch_normalization_538_batchnorm_readvariableop_resource:Y
Ksequential_55_batch_normalization_538_batchnorm_mul_readvariableop_resource:W
Isequential_55_batch_normalization_538_batchnorm_readvariableop_1_resource:W
Isequential_55_batch_normalization_538_batchnorm_readvariableop_2_resource:H
6sequential_55_dense_594_matmul_readvariableop_resource:EE
7sequential_55_dense_594_biasadd_readvariableop_resource:EU
Gsequential_55_batch_normalization_539_batchnorm_readvariableop_resource:EY
Ksequential_55_batch_normalization_539_batchnorm_mul_readvariableop_resource:EW
Isequential_55_batch_normalization_539_batchnorm_readvariableop_1_resource:EW
Isequential_55_batch_normalization_539_batchnorm_readvariableop_2_resource:EH
6sequential_55_dense_595_matmul_readvariableop_resource:EEE
7sequential_55_dense_595_biasadd_readvariableop_resource:EU
Gsequential_55_batch_normalization_540_batchnorm_readvariableop_resource:EY
Ksequential_55_batch_normalization_540_batchnorm_mul_readvariableop_resource:EW
Isequential_55_batch_normalization_540_batchnorm_readvariableop_1_resource:EW
Isequential_55_batch_normalization_540_batchnorm_readvariableop_2_resource:EH
6sequential_55_dense_596_matmul_readvariableop_resource:EE
7sequential_55_dense_596_biasadd_readvariableop_resource:
identity¢>sequential_55/batch_normalization_533/batchnorm/ReadVariableOp¢@sequential_55/batch_normalization_533/batchnorm/ReadVariableOp_1¢@sequential_55/batch_normalization_533/batchnorm/ReadVariableOp_2¢Bsequential_55/batch_normalization_533/batchnorm/mul/ReadVariableOp¢>sequential_55/batch_normalization_534/batchnorm/ReadVariableOp¢@sequential_55/batch_normalization_534/batchnorm/ReadVariableOp_1¢@sequential_55/batch_normalization_534/batchnorm/ReadVariableOp_2¢Bsequential_55/batch_normalization_534/batchnorm/mul/ReadVariableOp¢>sequential_55/batch_normalization_535/batchnorm/ReadVariableOp¢@sequential_55/batch_normalization_535/batchnorm/ReadVariableOp_1¢@sequential_55/batch_normalization_535/batchnorm/ReadVariableOp_2¢Bsequential_55/batch_normalization_535/batchnorm/mul/ReadVariableOp¢>sequential_55/batch_normalization_536/batchnorm/ReadVariableOp¢@sequential_55/batch_normalization_536/batchnorm/ReadVariableOp_1¢@sequential_55/batch_normalization_536/batchnorm/ReadVariableOp_2¢Bsequential_55/batch_normalization_536/batchnorm/mul/ReadVariableOp¢>sequential_55/batch_normalization_537/batchnorm/ReadVariableOp¢@sequential_55/batch_normalization_537/batchnorm/ReadVariableOp_1¢@sequential_55/batch_normalization_537/batchnorm/ReadVariableOp_2¢Bsequential_55/batch_normalization_537/batchnorm/mul/ReadVariableOp¢>sequential_55/batch_normalization_538/batchnorm/ReadVariableOp¢@sequential_55/batch_normalization_538/batchnorm/ReadVariableOp_1¢@sequential_55/batch_normalization_538/batchnorm/ReadVariableOp_2¢Bsequential_55/batch_normalization_538/batchnorm/mul/ReadVariableOp¢>sequential_55/batch_normalization_539/batchnorm/ReadVariableOp¢@sequential_55/batch_normalization_539/batchnorm/ReadVariableOp_1¢@sequential_55/batch_normalization_539/batchnorm/ReadVariableOp_2¢Bsequential_55/batch_normalization_539/batchnorm/mul/ReadVariableOp¢>sequential_55/batch_normalization_540/batchnorm/ReadVariableOp¢@sequential_55/batch_normalization_540/batchnorm/ReadVariableOp_1¢@sequential_55/batch_normalization_540/batchnorm/ReadVariableOp_2¢Bsequential_55/batch_normalization_540/batchnorm/mul/ReadVariableOp¢.sequential_55/dense_588/BiasAdd/ReadVariableOp¢-sequential_55/dense_588/MatMul/ReadVariableOp¢.sequential_55/dense_589/BiasAdd/ReadVariableOp¢-sequential_55/dense_589/MatMul/ReadVariableOp¢.sequential_55/dense_590/BiasAdd/ReadVariableOp¢-sequential_55/dense_590/MatMul/ReadVariableOp¢.sequential_55/dense_591/BiasAdd/ReadVariableOp¢-sequential_55/dense_591/MatMul/ReadVariableOp¢.sequential_55/dense_592/BiasAdd/ReadVariableOp¢-sequential_55/dense_592/MatMul/ReadVariableOp¢.sequential_55/dense_593/BiasAdd/ReadVariableOp¢-sequential_55/dense_593/MatMul/ReadVariableOp¢.sequential_55/dense_594/BiasAdd/ReadVariableOp¢-sequential_55/dense_594/MatMul/ReadVariableOp¢.sequential_55/dense_595/BiasAdd/ReadVariableOp¢-sequential_55/dense_595/MatMul/ReadVariableOp¢.sequential_55/dense_596/BiasAdd/ReadVariableOp¢-sequential_55/dense_596/MatMul/ReadVariableOp
"sequential_55/normalization_55/subSubnormalization_55_input$sequential_55_normalization_55_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#sequential_55/normalization_55/SqrtSqrt%sequential_55_normalization_55_sqrt_x*
T0*
_output_shapes

:m
(sequential_55/normalization_55/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¶
&sequential_55/normalization_55/MaximumMaximum'sequential_55/normalization_55/Sqrt:y:01sequential_55/normalization_55/Maximum/y:output:0*
T0*
_output_shapes

:·
&sequential_55/normalization_55/truedivRealDiv&sequential_55/normalization_55/sub:z:0*sequential_55/normalization_55/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_55/dense_588/MatMul/ReadVariableOpReadVariableOp6sequential_55_dense_588_matmul_readvariableop_resource*
_output_shapes

:Y*
dtype0½
sequential_55/dense_588/MatMulMatMul*sequential_55/normalization_55/truediv:z:05sequential_55/dense_588/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY¢
.sequential_55/dense_588/BiasAdd/ReadVariableOpReadVariableOp7sequential_55_dense_588_biasadd_readvariableop_resource*
_output_shapes
:Y*
dtype0¾
sequential_55/dense_588/BiasAddBiasAdd(sequential_55/dense_588/MatMul:product:06sequential_55/dense_588/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYÂ
>sequential_55/batch_normalization_533/batchnorm/ReadVariableOpReadVariableOpGsequential_55_batch_normalization_533_batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0z
5sequential_55/batch_normalization_533/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_55/batch_normalization_533/batchnorm/addAddV2Fsequential_55/batch_normalization_533/batchnorm/ReadVariableOp:value:0>sequential_55/batch_normalization_533/batchnorm/add/y:output:0*
T0*
_output_shapes
:Y
5sequential_55/batch_normalization_533/batchnorm/RsqrtRsqrt7sequential_55/batch_normalization_533/batchnorm/add:z:0*
T0*
_output_shapes
:YÊ
Bsequential_55/batch_normalization_533/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_55_batch_normalization_533_batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0æ
3sequential_55/batch_normalization_533/batchnorm/mulMul9sequential_55/batch_normalization_533/batchnorm/Rsqrt:y:0Jsequential_55/batch_normalization_533/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:YÑ
5sequential_55/batch_normalization_533/batchnorm/mul_1Mul(sequential_55/dense_588/BiasAdd:output:07sequential_55/batch_normalization_533/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYÆ
@sequential_55/batch_normalization_533/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_55_batch_normalization_533_batchnorm_readvariableop_1_resource*
_output_shapes
:Y*
dtype0ä
5sequential_55/batch_normalization_533/batchnorm/mul_2MulHsequential_55/batch_normalization_533/batchnorm/ReadVariableOp_1:value:07sequential_55/batch_normalization_533/batchnorm/mul:z:0*
T0*
_output_shapes
:YÆ
@sequential_55/batch_normalization_533/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_55_batch_normalization_533_batchnorm_readvariableop_2_resource*
_output_shapes
:Y*
dtype0ä
3sequential_55/batch_normalization_533/batchnorm/subSubHsequential_55/batch_normalization_533/batchnorm/ReadVariableOp_2:value:09sequential_55/batch_normalization_533/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yä
5sequential_55/batch_normalization_533/batchnorm/add_1AddV29sequential_55/batch_normalization_533/batchnorm/mul_1:z:07sequential_55/batch_normalization_533/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY¨
'sequential_55/leaky_re_lu_533/LeakyRelu	LeakyRelu9sequential_55/batch_normalization_533/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*
alpha%>¤
-sequential_55/dense_589/MatMul/ReadVariableOpReadVariableOp6sequential_55_dense_589_matmul_readvariableop_resource*
_output_shapes

:YY*
dtype0È
sequential_55/dense_589/MatMulMatMul5sequential_55/leaky_re_lu_533/LeakyRelu:activations:05sequential_55/dense_589/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY¢
.sequential_55/dense_589/BiasAdd/ReadVariableOpReadVariableOp7sequential_55_dense_589_biasadd_readvariableop_resource*
_output_shapes
:Y*
dtype0¾
sequential_55/dense_589/BiasAddBiasAdd(sequential_55/dense_589/MatMul:product:06sequential_55/dense_589/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYÂ
>sequential_55/batch_normalization_534/batchnorm/ReadVariableOpReadVariableOpGsequential_55_batch_normalization_534_batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0z
5sequential_55/batch_normalization_534/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_55/batch_normalization_534/batchnorm/addAddV2Fsequential_55/batch_normalization_534/batchnorm/ReadVariableOp:value:0>sequential_55/batch_normalization_534/batchnorm/add/y:output:0*
T0*
_output_shapes
:Y
5sequential_55/batch_normalization_534/batchnorm/RsqrtRsqrt7sequential_55/batch_normalization_534/batchnorm/add:z:0*
T0*
_output_shapes
:YÊ
Bsequential_55/batch_normalization_534/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_55_batch_normalization_534_batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0æ
3sequential_55/batch_normalization_534/batchnorm/mulMul9sequential_55/batch_normalization_534/batchnorm/Rsqrt:y:0Jsequential_55/batch_normalization_534/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:YÑ
5sequential_55/batch_normalization_534/batchnorm/mul_1Mul(sequential_55/dense_589/BiasAdd:output:07sequential_55/batch_normalization_534/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYÆ
@sequential_55/batch_normalization_534/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_55_batch_normalization_534_batchnorm_readvariableop_1_resource*
_output_shapes
:Y*
dtype0ä
5sequential_55/batch_normalization_534/batchnorm/mul_2MulHsequential_55/batch_normalization_534/batchnorm/ReadVariableOp_1:value:07sequential_55/batch_normalization_534/batchnorm/mul:z:0*
T0*
_output_shapes
:YÆ
@sequential_55/batch_normalization_534/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_55_batch_normalization_534_batchnorm_readvariableop_2_resource*
_output_shapes
:Y*
dtype0ä
3sequential_55/batch_normalization_534/batchnorm/subSubHsequential_55/batch_normalization_534/batchnorm/ReadVariableOp_2:value:09sequential_55/batch_normalization_534/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yä
5sequential_55/batch_normalization_534/batchnorm/add_1AddV29sequential_55/batch_normalization_534/batchnorm/mul_1:z:07sequential_55/batch_normalization_534/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY¨
'sequential_55/leaky_re_lu_534/LeakyRelu	LeakyRelu9sequential_55/batch_normalization_534/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*
alpha%>¤
-sequential_55/dense_590/MatMul/ReadVariableOpReadVariableOp6sequential_55_dense_590_matmul_readvariableop_resource*
_output_shapes

:Y*
dtype0È
sequential_55/dense_590/MatMulMatMul5sequential_55/leaky_re_lu_534/LeakyRelu:activations:05sequential_55/dense_590/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_55/dense_590/BiasAdd/ReadVariableOpReadVariableOp7sequential_55_dense_590_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_55/dense_590/BiasAddBiasAdd(sequential_55/dense_590/MatMul:product:06sequential_55/dense_590/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_55/batch_normalization_535/batchnorm/ReadVariableOpReadVariableOpGsequential_55_batch_normalization_535_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_55/batch_normalization_535/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_55/batch_normalization_535/batchnorm/addAddV2Fsequential_55/batch_normalization_535/batchnorm/ReadVariableOp:value:0>sequential_55/batch_normalization_535/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_55/batch_normalization_535/batchnorm/RsqrtRsqrt7sequential_55/batch_normalization_535/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_55/batch_normalization_535/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_55_batch_normalization_535_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_55/batch_normalization_535/batchnorm/mulMul9sequential_55/batch_normalization_535/batchnorm/Rsqrt:y:0Jsequential_55/batch_normalization_535/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ñ
5sequential_55/batch_normalization_535/batchnorm/mul_1Mul(sequential_55/dense_590/BiasAdd:output:07sequential_55/batch_normalization_535/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_55/batch_normalization_535/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_55_batch_normalization_535_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_55/batch_normalization_535/batchnorm/mul_2MulHsequential_55/batch_normalization_535/batchnorm/ReadVariableOp_1:value:07sequential_55/batch_normalization_535/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_55/batch_normalization_535/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_55_batch_normalization_535_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_55/batch_normalization_535/batchnorm/subSubHsequential_55/batch_normalization_535/batchnorm/ReadVariableOp_2:value:09sequential_55/batch_normalization_535/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_55/batch_normalization_535/batchnorm/add_1AddV29sequential_55/batch_normalization_535/batchnorm/mul_1:z:07sequential_55/batch_normalization_535/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_55/leaky_re_lu_535/LeakyRelu	LeakyRelu9sequential_55/batch_normalization_535/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¤
-sequential_55/dense_591/MatMul/ReadVariableOpReadVariableOp6sequential_55_dense_591_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
sequential_55/dense_591/MatMulMatMul5sequential_55/leaky_re_lu_535/LeakyRelu:activations:05sequential_55/dense_591/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_55/dense_591/BiasAdd/ReadVariableOpReadVariableOp7sequential_55_dense_591_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_55/dense_591/BiasAddBiasAdd(sequential_55/dense_591/MatMul:product:06sequential_55/dense_591/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_55/batch_normalization_536/batchnorm/ReadVariableOpReadVariableOpGsequential_55_batch_normalization_536_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_55/batch_normalization_536/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_55/batch_normalization_536/batchnorm/addAddV2Fsequential_55/batch_normalization_536/batchnorm/ReadVariableOp:value:0>sequential_55/batch_normalization_536/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_55/batch_normalization_536/batchnorm/RsqrtRsqrt7sequential_55/batch_normalization_536/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_55/batch_normalization_536/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_55_batch_normalization_536_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_55/batch_normalization_536/batchnorm/mulMul9sequential_55/batch_normalization_536/batchnorm/Rsqrt:y:0Jsequential_55/batch_normalization_536/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ñ
5sequential_55/batch_normalization_536/batchnorm/mul_1Mul(sequential_55/dense_591/BiasAdd:output:07sequential_55/batch_normalization_536/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_55/batch_normalization_536/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_55_batch_normalization_536_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_55/batch_normalization_536/batchnorm/mul_2MulHsequential_55/batch_normalization_536/batchnorm/ReadVariableOp_1:value:07sequential_55/batch_normalization_536/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_55/batch_normalization_536/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_55_batch_normalization_536_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_55/batch_normalization_536/batchnorm/subSubHsequential_55/batch_normalization_536/batchnorm/ReadVariableOp_2:value:09sequential_55/batch_normalization_536/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_55/batch_normalization_536/batchnorm/add_1AddV29sequential_55/batch_normalization_536/batchnorm/mul_1:z:07sequential_55/batch_normalization_536/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_55/leaky_re_lu_536/LeakyRelu	LeakyRelu9sequential_55/batch_normalization_536/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¤
-sequential_55/dense_592/MatMul/ReadVariableOpReadVariableOp6sequential_55_dense_592_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
sequential_55/dense_592/MatMulMatMul5sequential_55/leaky_re_lu_536/LeakyRelu:activations:05sequential_55/dense_592/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_55/dense_592/BiasAdd/ReadVariableOpReadVariableOp7sequential_55_dense_592_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_55/dense_592/BiasAddBiasAdd(sequential_55/dense_592/MatMul:product:06sequential_55/dense_592/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_55/batch_normalization_537/batchnorm/ReadVariableOpReadVariableOpGsequential_55_batch_normalization_537_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_55/batch_normalization_537/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_55/batch_normalization_537/batchnorm/addAddV2Fsequential_55/batch_normalization_537/batchnorm/ReadVariableOp:value:0>sequential_55/batch_normalization_537/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_55/batch_normalization_537/batchnorm/RsqrtRsqrt7sequential_55/batch_normalization_537/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_55/batch_normalization_537/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_55_batch_normalization_537_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_55/batch_normalization_537/batchnorm/mulMul9sequential_55/batch_normalization_537/batchnorm/Rsqrt:y:0Jsequential_55/batch_normalization_537/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ñ
5sequential_55/batch_normalization_537/batchnorm/mul_1Mul(sequential_55/dense_592/BiasAdd:output:07sequential_55/batch_normalization_537/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_55/batch_normalization_537/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_55_batch_normalization_537_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_55/batch_normalization_537/batchnorm/mul_2MulHsequential_55/batch_normalization_537/batchnorm/ReadVariableOp_1:value:07sequential_55/batch_normalization_537/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_55/batch_normalization_537/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_55_batch_normalization_537_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_55/batch_normalization_537/batchnorm/subSubHsequential_55/batch_normalization_537/batchnorm/ReadVariableOp_2:value:09sequential_55/batch_normalization_537/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_55/batch_normalization_537/batchnorm/add_1AddV29sequential_55/batch_normalization_537/batchnorm/mul_1:z:07sequential_55/batch_normalization_537/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_55/leaky_re_lu_537/LeakyRelu	LeakyRelu9sequential_55/batch_normalization_537/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¤
-sequential_55/dense_593/MatMul/ReadVariableOpReadVariableOp6sequential_55_dense_593_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
sequential_55/dense_593/MatMulMatMul5sequential_55/leaky_re_lu_537/LeakyRelu:activations:05sequential_55/dense_593/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_55/dense_593/BiasAdd/ReadVariableOpReadVariableOp7sequential_55_dense_593_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_55/dense_593/BiasAddBiasAdd(sequential_55/dense_593/MatMul:product:06sequential_55/dense_593/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_55/batch_normalization_538/batchnorm/ReadVariableOpReadVariableOpGsequential_55_batch_normalization_538_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_55/batch_normalization_538/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_55/batch_normalization_538/batchnorm/addAddV2Fsequential_55/batch_normalization_538/batchnorm/ReadVariableOp:value:0>sequential_55/batch_normalization_538/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_55/batch_normalization_538/batchnorm/RsqrtRsqrt7sequential_55/batch_normalization_538/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_55/batch_normalization_538/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_55_batch_normalization_538_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_55/batch_normalization_538/batchnorm/mulMul9sequential_55/batch_normalization_538/batchnorm/Rsqrt:y:0Jsequential_55/batch_normalization_538/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ñ
5sequential_55/batch_normalization_538/batchnorm/mul_1Mul(sequential_55/dense_593/BiasAdd:output:07sequential_55/batch_normalization_538/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_55/batch_normalization_538/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_55_batch_normalization_538_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_55/batch_normalization_538/batchnorm/mul_2MulHsequential_55/batch_normalization_538/batchnorm/ReadVariableOp_1:value:07sequential_55/batch_normalization_538/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_55/batch_normalization_538/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_55_batch_normalization_538_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_55/batch_normalization_538/batchnorm/subSubHsequential_55/batch_normalization_538/batchnorm/ReadVariableOp_2:value:09sequential_55/batch_normalization_538/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_55/batch_normalization_538/batchnorm/add_1AddV29sequential_55/batch_normalization_538/batchnorm/mul_1:z:07sequential_55/batch_normalization_538/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_55/leaky_re_lu_538/LeakyRelu	LeakyRelu9sequential_55/batch_normalization_538/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¤
-sequential_55/dense_594/MatMul/ReadVariableOpReadVariableOp6sequential_55_dense_594_matmul_readvariableop_resource*
_output_shapes

:E*
dtype0È
sequential_55/dense_594/MatMulMatMul5sequential_55/leaky_re_lu_538/LeakyRelu:activations:05sequential_55/dense_594/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE¢
.sequential_55/dense_594/BiasAdd/ReadVariableOpReadVariableOp7sequential_55_dense_594_biasadd_readvariableop_resource*
_output_shapes
:E*
dtype0¾
sequential_55/dense_594/BiasAddBiasAdd(sequential_55/dense_594/MatMul:product:06sequential_55/dense_594/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEÂ
>sequential_55/batch_normalization_539/batchnorm/ReadVariableOpReadVariableOpGsequential_55_batch_normalization_539_batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0z
5sequential_55/batch_normalization_539/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_55/batch_normalization_539/batchnorm/addAddV2Fsequential_55/batch_normalization_539/batchnorm/ReadVariableOp:value:0>sequential_55/batch_normalization_539/batchnorm/add/y:output:0*
T0*
_output_shapes
:E
5sequential_55/batch_normalization_539/batchnorm/RsqrtRsqrt7sequential_55/batch_normalization_539/batchnorm/add:z:0*
T0*
_output_shapes
:EÊ
Bsequential_55/batch_normalization_539/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_55_batch_normalization_539_batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0æ
3sequential_55/batch_normalization_539/batchnorm/mulMul9sequential_55/batch_normalization_539/batchnorm/Rsqrt:y:0Jsequential_55/batch_normalization_539/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:EÑ
5sequential_55/batch_normalization_539/batchnorm/mul_1Mul(sequential_55/dense_594/BiasAdd:output:07sequential_55/batch_normalization_539/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEÆ
@sequential_55/batch_normalization_539/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_55_batch_normalization_539_batchnorm_readvariableop_1_resource*
_output_shapes
:E*
dtype0ä
5sequential_55/batch_normalization_539/batchnorm/mul_2MulHsequential_55/batch_normalization_539/batchnorm/ReadVariableOp_1:value:07sequential_55/batch_normalization_539/batchnorm/mul:z:0*
T0*
_output_shapes
:EÆ
@sequential_55/batch_normalization_539/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_55_batch_normalization_539_batchnorm_readvariableop_2_resource*
_output_shapes
:E*
dtype0ä
3sequential_55/batch_normalization_539/batchnorm/subSubHsequential_55/batch_normalization_539/batchnorm/ReadVariableOp_2:value:09sequential_55/batch_normalization_539/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Eä
5sequential_55/batch_normalization_539/batchnorm/add_1AddV29sequential_55/batch_normalization_539/batchnorm/mul_1:z:07sequential_55/batch_normalization_539/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE¨
'sequential_55/leaky_re_lu_539/LeakyRelu	LeakyRelu9sequential_55/batch_normalization_539/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*
alpha%>¤
-sequential_55/dense_595/MatMul/ReadVariableOpReadVariableOp6sequential_55_dense_595_matmul_readvariableop_resource*
_output_shapes

:EE*
dtype0È
sequential_55/dense_595/MatMulMatMul5sequential_55/leaky_re_lu_539/LeakyRelu:activations:05sequential_55/dense_595/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE¢
.sequential_55/dense_595/BiasAdd/ReadVariableOpReadVariableOp7sequential_55_dense_595_biasadd_readvariableop_resource*
_output_shapes
:E*
dtype0¾
sequential_55/dense_595/BiasAddBiasAdd(sequential_55/dense_595/MatMul:product:06sequential_55/dense_595/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEÂ
>sequential_55/batch_normalization_540/batchnorm/ReadVariableOpReadVariableOpGsequential_55_batch_normalization_540_batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0z
5sequential_55/batch_normalization_540/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_55/batch_normalization_540/batchnorm/addAddV2Fsequential_55/batch_normalization_540/batchnorm/ReadVariableOp:value:0>sequential_55/batch_normalization_540/batchnorm/add/y:output:0*
T0*
_output_shapes
:E
5sequential_55/batch_normalization_540/batchnorm/RsqrtRsqrt7sequential_55/batch_normalization_540/batchnorm/add:z:0*
T0*
_output_shapes
:EÊ
Bsequential_55/batch_normalization_540/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_55_batch_normalization_540_batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0æ
3sequential_55/batch_normalization_540/batchnorm/mulMul9sequential_55/batch_normalization_540/batchnorm/Rsqrt:y:0Jsequential_55/batch_normalization_540/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:EÑ
5sequential_55/batch_normalization_540/batchnorm/mul_1Mul(sequential_55/dense_595/BiasAdd:output:07sequential_55/batch_normalization_540/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEÆ
@sequential_55/batch_normalization_540/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_55_batch_normalization_540_batchnorm_readvariableop_1_resource*
_output_shapes
:E*
dtype0ä
5sequential_55/batch_normalization_540/batchnorm/mul_2MulHsequential_55/batch_normalization_540/batchnorm/ReadVariableOp_1:value:07sequential_55/batch_normalization_540/batchnorm/mul:z:0*
T0*
_output_shapes
:EÆ
@sequential_55/batch_normalization_540/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_55_batch_normalization_540_batchnorm_readvariableop_2_resource*
_output_shapes
:E*
dtype0ä
3sequential_55/batch_normalization_540/batchnorm/subSubHsequential_55/batch_normalization_540/batchnorm/ReadVariableOp_2:value:09sequential_55/batch_normalization_540/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Eä
5sequential_55/batch_normalization_540/batchnorm/add_1AddV29sequential_55/batch_normalization_540/batchnorm/mul_1:z:07sequential_55/batch_normalization_540/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE¨
'sequential_55/leaky_re_lu_540/LeakyRelu	LeakyRelu9sequential_55/batch_normalization_540/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*
alpha%>¤
-sequential_55/dense_596/MatMul/ReadVariableOpReadVariableOp6sequential_55_dense_596_matmul_readvariableop_resource*
_output_shapes

:E*
dtype0È
sequential_55/dense_596/MatMulMatMul5sequential_55/leaky_re_lu_540/LeakyRelu:activations:05sequential_55/dense_596/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_55/dense_596/BiasAdd/ReadVariableOpReadVariableOp7sequential_55_dense_596_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_55/dense_596/BiasAddBiasAdd(sequential_55/dense_596/MatMul:product:06sequential_55/dense_596/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_55/dense_596/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp?^sequential_55/batch_normalization_533/batchnorm/ReadVariableOpA^sequential_55/batch_normalization_533/batchnorm/ReadVariableOp_1A^sequential_55/batch_normalization_533/batchnorm/ReadVariableOp_2C^sequential_55/batch_normalization_533/batchnorm/mul/ReadVariableOp?^sequential_55/batch_normalization_534/batchnorm/ReadVariableOpA^sequential_55/batch_normalization_534/batchnorm/ReadVariableOp_1A^sequential_55/batch_normalization_534/batchnorm/ReadVariableOp_2C^sequential_55/batch_normalization_534/batchnorm/mul/ReadVariableOp?^sequential_55/batch_normalization_535/batchnorm/ReadVariableOpA^sequential_55/batch_normalization_535/batchnorm/ReadVariableOp_1A^sequential_55/batch_normalization_535/batchnorm/ReadVariableOp_2C^sequential_55/batch_normalization_535/batchnorm/mul/ReadVariableOp?^sequential_55/batch_normalization_536/batchnorm/ReadVariableOpA^sequential_55/batch_normalization_536/batchnorm/ReadVariableOp_1A^sequential_55/batch_normalization_536/batchnorm/ReadVariableOp_2C^sequential_55/batch_normalization_536/batchnorm/mul/ReadVariableOp?^sequential_55/batch_normalization_537/batchnorm/ReadVariableOpA^sequential_55/batch_normalization_537/batchnorm/ReadVariableOp_1A^sequential_55/batch_normalization_537/batchnorm/ReadVariableOp_2C^sequential_55/batch_normalization_537/batchnorm/mul/ReadVariableOp?^sequential_55/batch_normalization_538/batchnorm/ReadVariableOpA^sequential_55/batch_normalization_538/batchnorm/ReadVariableOp_1A^sequential_55/batch_normalization_538/batchnorm/ReadVariableOp_2C^sequential_55/batch_normalization_538/batchnorm/mul/ReadVariableOp?^sequential_55/batch_normalization_539/batchnorm/ReadVariableOpA^sequential_55/batch_normalization_539/batchnorm/ReadVariableOp_1A^sequential_55/batch_normalization_539/batchnorm/ReadVariableOp_2C^sequential_55/batch_normalization_539/batchnorm/mul/ReadVariableOp?^sequential_55/batch_normalization_540/batchnorm/ReadVariableOpA^sequential_55/batch_normalization_540/batchnorm/ReadVariableOp_1A^sequential_55/batch_normalization_540/batchnorm/ReadVariableOp_2C^sequential_55/batch_normalization_540/batchnorm/mul/ReadVariableOp/^sequential_55/dense_588/BiasAdd/ReadVariableOp.^sequential_55/dense_588/MatMul/ReadVariableOp/^sequential_55/dense_589/BiasAdd/ReadVariableOp.^sequential_55/dense_589/MatMul/ReadVariableOp/^sequential_55/dense_590/BiasAdd/ReadVariableOp.^sequential_55/dense_590/MatMul/ReadVariableOp/^sequential_55/dense_591/BiasAdd/ReadVariableOp.^sequential_55/dense_591/MatMul/ReadVariableOp/^sequential_55/dense_592/BiasAdd/ReadVariableOp.^sequential_55/dense_592/MatMul/ReadVariableOp/^sequential_55/dense_593/BiasAdd/ReadVariableOp.^sequential_55/dense_593/MatMul/ReadVariableOp/^sequential_55/dense_594/BiasAdd/ReadVariableOp.^sequential_55/dense_594/MatMul/ReadVariableOp/^sequential_55/dense_595/BiasAdd/ReadVariableOp.^sequential_55/dense_595/MatMul/ReadVariableOp/^sequential_55/dense_596/BiasAdd/ReadVariableOp.^sequential_55/dense_596/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>sequential_55/batch_normalization_533/batchnorm/ReadVariableOp>sequential_55/batch_normalization_533/batchnorm/ReadVariableOp2
@sequential_55/batch_normalization_533/batchnorm/ReadVariableOp_1@sequential_55/batch_normalization_533/batchnorm/ReadVariableOp_12
@sequential_55/batch_normalization_533/batchnorm/ReadVariableOp_2@sequential_55/batch_normalization_533/batchnorm/ReadVariableOp_22
Bsequential_55/batch_normalization_533/batchnorm/mul/ReadVariableOpBsequential_55/batch_normalization_533/batchnorm/mul/ReadVariableOp2
>sequential_55/batch_normalization_534/batchnorm/ReadVariableOp>sequential_55/batch_normalization_534/batchnorm/ReadVariableOp2
@sequential_55/batch_normalization_534/batchnorm/ReadVariableOp_1@sequential_55/batch_normalization_534/batchnorm/ReadVariableOp_12
@sequential_55/batch_normalization_534/batchnorm/ReadVariableOp_2@sequential_55/batch_normalization_534/batchnorm/ReadVariableOp_22
Bsequential_55/batch_normalization_534/batchnorm/mul/ReadVariableOpBsequential_55/batch_normalization_534/batchnorm/mul/ReadVariableOp2
>sequential_55/batch_normalization_535/batchnorm/ReadVariableOp>sequential_55/batch_normalization_535/batchnorm/ReadVariableOp2
@sequential_55/batch_normalization_535/batchnorm/ReadVariableOp_1@sequential_55/batch_normalization_535/batchnorm/ReadVariableOp_12
@sequential_55/batch_normalization_535/batchnorm/ReadVariableOp_2@sequential_55/batch_normalization_535/batchnorm/ReadVariableOp_22
Bsequential_55/batch_normalization_535/batchnorm/mul/ReadVariableOpBsequential_55/batch_normalization_535/batchnorm/mul/ReadVariableOp2
>sequential_55/batch_normalization_536/batchnorm/ReadVariableOp>sequential_55/batch_normalization_536/batchnorm/ReadVariableOp2
@sequential_55/batch_normalization_536/batchnorm/ReadVariableOp_1@sequential_55/batch_normalization_536/batchnorm/ReadVariableOp_12
@sequential_55/batch_normalization_536/batchnorm/ReadVariableOp_2@sequential_55/batch_normalization_536/batchnorm/ReadVariableOp_22
Bsequential_55/batch_normalization_536/batchnorm/mul/ReadVariableOpBsequential_55/batch_normalization_536/batchnorm/mul/ReadVariableOp2
>sequential_55/batch_normalization_537/batchnorm/ReadVariableOp>sequential_55/batch_normalization_537/batchnorm/ReadVariableOp2
@sequential_55/batch_normalization_537/batchnorm/ReadVariableOp_1@sequential_55/batch_normalization_537/batchnorm/ReadVariableOp_12
@sequential_55/batch_normalization_537/batchnorm/ReadVariableOp_2@sequential_55/batch_normalization_537/batchnorm/ReadVariableOp_22
Bsequential_55/batch_normalization_537/batchnorm/mul/ReadVariableOpBsequential_55/batch_normalization_537/batchnorm/mul/ReadVariableOp2
>sequential_55/batch_normalization_538/batchnorm/ReadVariableOp>sequential_55/batch_normalization_538/batchnorm/ReadVariableOp2
@sequential_55/batch_normalization_538/batchnorm/ReadVariableOp_1@sequential_55/batch_normalization_538/batchnorm/ReadVariableOp_12
@sequential_55/batch_normalization_538/batchnorm/ReadVariableOp_2@sequential_55/batch_normalization_538/batchnorm/ReadVariableOp_22
Bsequential_55/batch_normalization_538/batchnorm/mul/ReadVariableOpBsequential_55/batch_normalization_538/batchnorm/mul/ReadVariableOp2
>sequential_55/batch_normalization_539/batchnorm/ReadVariableOp>sequential_55/batch_normalization_539/batchnorm/ReadVariableOp2
@sequential_55/batch_normalization_539/batchnorm/ReadVariableOp_1@sequential_55/batch_normalization_539/batchnorm/ReadVariableOp_12
@sequential_55/batch_normalization_539/batchnorm/ReadVariableOp_2@sequential_55/batch_normalization_539/batchnorm/ReadVariableOp_22
Bsequential_55/batch_normalization_539/batchnorm/mul/ReadVariableOpBsequential_55/batch_normalization_539/batchnorm/mul/ReadVariableOp2
>sequential_55/batch_normalization_540/batchnorm/ReadVariableOp>sequential_55/batch_normalization_540/batchnorm/ReadVariableOp2
@sequential_55/batch_normalization_540/batchnorm/ReadVariableOp_1@sequential_55/batch_normalization_540/batchnorm/ReadVariableOp_12
@sequential_55/batch_normalization_540/batchnorm/ReadVariableOp_2@sequential_55/batch_normalization_540/batchnorm/ReadVariableOp_22
Bsequential_55/batch_normalization_540/batchnorm/mul/ReadVariableOpBsequential_55/batch_normalization_540/batchnorm/mul/ReadVariableOp2`
.sequential_55/dense_588/BiasAdd/ReadVariableOp.sequential_55/dense_588/BiasAdd/ReadVariableOp2^
-sequential_55/dense_588/MatMul/ReadVariableOp-sequential_55/dense_588/MatMul/ReadVariableOp2`
.sequential_55/dense_589/BiasAdd/ReadVariableOp.sequential_55/dense_589/BiasAdd/ReadVariableOp2^
-sequential_55/dense_589/MatMul/ReadVariableOp-sequential_55/dense_589/MatMul/ReadVariableOp2`
.sequential_55/dense_590/BiasAdd/ReadVariableOp.sequential_55/dense_590/BiasAdd/ReadVariableOp2^
-sequential_55/dense_590/MatMul/ReadVariableOp-sequential_55/dense_590/MatMul/ReadVariableOp2`
.sequential_55/dense_591/BiasAdd/ReadVariableOp.sequential_55/dense_591/BiasAdd/ReadVariableOp2^
-sequential_55/dense_591/MatMul/ReadVariableOp-sequential_55/dense_591/MatMul/ReadVariableOp2`
.sequential_55/dense_592/BiasAdd/ReadVariableOp.sequential_55/dense_592/BiasAdd/ReadVariableOp2^
-sequential_55/dense_592/MatMul/ReadVariableOp-sequential_55/dense_592/MatMul/ReadVariableOp2`
.sequential_55/dense_593/BiasAdd/ReadVariableOp.sequential_55/dense_593/BiasAdd/ReadVariableOp2^
-sequential_55/dense_593/MatMul/ReadVariableOp-sequential_55/dense_593/MatMul/ReadVariableOp2`
.sequential_55/dense_594/BiasAdd/ReadVariableOp.sequential_55/dense_594/BiasAdd/ReadVariableOp2^
-sequential_55/dense_594/MatMul/ReadVariableOp-sequential_55/dense_594/MatMul/ReadVariableOp2`
.sequential_55/dense_595/BiasAdd/ReadVariableOp.sequential_55/dense_595/BiasAdd/ReadVariableOp2^
-sequential_55/dense_595/MatMul/ReadVariableOp-sequential_55/dense_595/MatMul/ReadVariableOp2`
.sequential_55/dense_596/BiasAdd/ReadVariableOp.sequential_55/dense_596/BiasAdd/ReadVariableOp2^
-sequential_55/dense_596/MatMul/ReadVariableOp-sequential_55/dense_596/MatMul/ReadVariableOp:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_55_input:$ 

_output_shapes

::$ 

_output_shapes

:
óô
Á;
__inference__traced_save_667813
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_588_kernel_read_readvariableop-
)savev2_dense_588_bias_read_readvariableop<
8savev2_batch_normalization_533_gamma_read_readvariableop;
7savev2_batch_normalization_533_beta_read_readvariableopB
>savev2_batch_normalization_533_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_533_moving_variance_read_readvariableop/
+savev2_dense_589_kernel_read_readvariableop-
)savev2_dense_589_bias_read_readvariableop<
8savev2_batch_normalization_534_gamma_read_readvariableop;
7savev2_batch_normalization_534_beta_read_readvariableopB
>savev2_batch_normalization_534_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_534_moving_variance_read_readvariableop/
+savev2_dense_590_kernel_read_readvariableop-
)savev2_dense_590_bias_read_readvariableop<
8savev2_batch_normalization_535_gamma_read_readvariableop;
7savev2_batch_normalization_535_beta_read_readvariableopB
>savev2_batch_normalization_535_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_535_moving_variance_read_readvariableop/
+savev2_dense_591_kernel_read_readvariableop-
)savev2_dense_591_bias_read_readvariableop<
8savev2_batch_normalization_536_gamma_read_readvariableop;
7savev2_batch_normalization_536_beta_read_readvariableopB
>savev2_batch_normalization_536_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_536_moving_variance_read_readvariableop/
+savev2_dense_592_kernel_read_readvariableop-
)savev2_dense_592_bias_read_readvariableop<
8savev2_batch_normalization_537_gamma_read_readvariableop;
7savev2_batch_normalization_537_beta_read_readvariableopB
>savev2_batch_normalization_537_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_537_moving_variance_read_readvariableop/
+savev2_dense_593_kernel_read_readvariableop-
)savev2_dense_593_bias_read_readvariableop<
8savev2_batch_normalization_538_gamma_read_readvariableop;
7savev2_batch_normalization_538_beta_read_readvariableopB
>savev2_batch_normalization_538_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_538_moving_variance_read_readvariableop/
+savev2_dense_594_kernel_read_readvariableop-
)savev2_dense_594_bias_read_readvariableop<
8savev2_batch_normalization_539_gamma_read_readvariableop;
7savev2_batch_normalization_539_beta_read_readvariableopB
>savev2_batch_normalization_539_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_539_moving_variance_read_readvariableop/
+savev2_dense_595_kernel_read_readvariableop-
)savev2_dense_595_bias_read_readvariableop<
8savev2_batch_normalization_540_gamma_read_readvariableop;
7savev2_batch_normalization_540_beta_read_readvariableopB
>savev2_batch_normalization_540_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_540_moving_variance_read_readvariableop/
+savev2_dense_596_kernel_read_readvariableop-
)savev2_dense_596_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_588_kernel_m_read_readvariableop4
0savev2_adam_dense_588_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_533_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_533_beta_m_read_readvariableop6
2savev2_adam_dense_589_kernel_m_read_readvariableop4
0savev2_adam_dense_589_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_534_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_534_beta_m_read_readvariableop6
2savev2_adam_dense_590_kernel_m_read_readvariableop4
0savev2_adam_dense_590_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_535_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_535_beta_m_read_readvariableop6
2savev2_adam_dense_591_kernel_m_read_readvariableop4
0savev2_adam_dense_591_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_536_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_536_beta_m_read_readvariableop6
2savev2_adam_dense_592_kernel_m_read_readvariableop4
0savev2_adam_dense_592_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_537_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_537_beta_m_read_readvariableop6
2savev2_adam_dense_593_kernel_m_read_readvariableop4
0savev2_adam_dense_593_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_538_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_538_beta_m_read_readvariableop6
2savev2_adam_dense_594_kernel_m_read_readvariableop4
0savev2_adam_dense_594_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_539_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_539_beta_m_read_readvariableop6
2savev2_adam_dense_595_kernel_m_read_readvariableop4
0savev2_adam_dense_595_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_540_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_540_beta_m_read_readvariableop6
2savev2_adam_dense_596_kernel_m_read_readvariableop4
0savev2_adam_dense_596_bias_m_read_readvariableop6
2savev2_adam_dense_588_kernel_v_read_readvariableop4
0savev2_adam_dense_588_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_533_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_533_beta_v_read_readvariableop6
2savev2_adam_dense_589_kernel_v_read_readvariableop4
0savev2_adam_dense_589_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_534_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_534_beta_v_read_readvariableop6
2savev2_adam_dense_590_kernel_v_read_readvariableop4
0savev2_adam_dense_590_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_535_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_535_beta_v_read_readvariableop6
2savev2_adam_dense_591_kernel_v_read_readvariableop4
0savev2_adam_dense_591_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_536_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_536_beta_v_read_readvariableop6
2savev2_adam_dense_592_kernel_v_read_readvariableop4
0savev2_adam_dense_592_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_537_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_537_beta_v_read_readvariableop6
2savev2_adam_dense_593_kernel_v_read_readvariableop4
0savev2_adam_dense_593_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_538_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_538_beta_v_read_readvariableop6
2savev2_adam_dense_594_kernel_v_read_readvariableop4
0savev2_adam_dense_594_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_539_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_539_beta_v_read_readvariableop6
2savev2_adam_dense_595_kernel_v_read_readvariableop4
0savev2_adam_dense_595_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_540_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_540_beta_v_read_readvariableop6
2savev2_adam_dense_596_kernel_v_read_readvariableop4
0savev2_adam_dense_596_bias_v_read_readvariableop
savev2_const_2

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ºG
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*âF
valueØFBÕFB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHò
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*
valueBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 9
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_588_kernel_read_readvariableop)savev2_dense_588_bias_read_readvariableop8savev2_batch_normalization_533_gamma_read_readvariableop7savev2_batch_normalization_533_beta_read_readvariableop>savev2_batch_normalization_533_moving_mean_read_readvariableopBsavev2_batch_normalization_533_moving_variance_read_readvariableop+savev2_dense_589_kernel_read_readvariableop)savev2_dense_589_bias_read_readvariableop8savev2_batch_normalization_534_gamma_read_readvariableop7savev2_batch_normalization_534_beta_read_readvariableop>savev2_batch_normalization_534_moving_mean_read_readvariableopBsavev2_batch_normalization_534_moving_variance_read_readvariableop+savev2_dense_590_kernel_read_readvariableop)savev2_dense_590_bias_read_readvariableop8savev2_batch_normalization_535_gamma_read_readvariableop7savev2_batch_normalization_535_beta_read_readvariableop>savev2_batch_normalization_535_moving_mean_read_readvariableopBsavev2_batch_normalization_535_moving_variance_read_readvariableop+savev2_dense_591_kernel_read_readvariableop)savev2_dense_591_bias_read_readvariableop8savev2_batch_normalization_536_gamma_read_readvariableop7savev2_batch_normalization_536_beta_read_readvariableop>savev2_batch_normalization_536_moving_mean_read_readvariableopBsavev2_batch_normalization_536_moving_variance_read_readvariableop+savev2_dense_592_kernel_read_readvariableop)savev2_dense_592_bias_read_readvariableop8savev2_batch_normalization_537_gamma_read_readvariableop7savev2_batch_normalization_537_beta_read_readvariableop>savev2_batch_normalization_537_moving_mean_read_readvariableopBsavev2_batch_normalization_537_moving_variance_read_readvariableop+savev2_dense_593_kernel_read_readvariableop)savev2_dense_593_bias_read_readvariableop8savev2_batch_normalization_538_gamma_read_readvariableop7savev2_batch_normalization_538_beta_read_readvariableop>savev2_batch_normalization_538_moving_mean_read_readvariableopBsavev2_batch_normalization_538_moving_variance_read_readvariableop+savev2_dense_594_kernel_read_readvariableop)savev2_dense_594_bias_read_readvariableop8savev2_batch_normalization_539_gamma_read_readvariableop7savev2_batch_normalization_539_beta_read_readvariableop>savev2_batch_normalization_539_moving_mean_read_readvariableopBsavev2_batch_normalization_539_moving_variance_read_readvariableop+savev2_dense_595_kernel_read_readvariableop)savev2_dense_595_bias_read_readvariableop8savev2_batch_normalization_540_gamma_read_readvariableop7savev2_batch_normalization_540_beta_read_readvariableop>savev2_batch_normalization_540_moving_mean_read_readvariableopBsavev2_batch_normalization_540_moving_variance_read_readvariableop+savev2_dense_596_kernel_read_readvariableop)savev2_dense_596_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_588_kernel_m_read_readvariableop0savev2_adam_dense_588_bias_m_read_readvariableop?savev2_adam_batch_normalization_533_gamma_m_read_readvariableop>savev2_adam_batch_normalization_533_beta_m_read_readvariableop2savev2_adam_dense_589_kernel_m_read_readvariableop0savev2_adam_dense_589_bias_m_read_readvariableop?savev2_adam_batch_normalization_534_gamma_m_read_readvariableop>savev2_adam_batch_normalization_534_beta_m_read_readvariableop2savev2_adam_dense_590_kernel_m_read_readvariableop0savev2_adam_dense_590_bias_m_read_readvariableop?savev2_adam_batch_normalization_535_gamma_m_read_readvariableop>savev2_adam_batch_normalization_535_beta_m_read_readvariableop2savev2_adam_dense_591_kernel_m_read_readvariableop0savev2_adam_dense_591_bias_m_read_readvariableop?savev2_adam_batch_normalization_536_gamma_m_read_readvariableop>savev2_adam_batch_normalization_536_beta_m_read_readvariableop2savev2_adam_dense_592_kernel_m_read_readvariableop0savev2_adam_dense_592_bias_m_read_readvariableop?savev2_adam_batch_normalization_537_gamma_m_read_readvariableop>savev2_adam_batch_normalization_537_beta_m_read_readvariableop2savev2_adam_dense_593_kernel_m_read_readvariableop0savev2_adam_dense_593_bias_m_read_readvariableop?savev2_adam_batch_normalization_538_gamma_m_read_readvariableop>savev2_adam_batch_normalization_538_beta_m_read_readvariableop2savev2_adam_dense_594_kernel_m_read_readvariableop0savev2_adam_dense_594_bias_m_read_readvariableop?savev2_adam_batch_normalization_539_gamma_m_read_readvariableop>savev2_adam_batch_normalization_539_beta_m_read_readvariableop2savev2_adam_dense_595_kernel_m_read_readvariableop0savev2_adam_dense_595_bias_m_read_readvariableop?savev2_adam_batch_normalization_540_gamma_m_read_readvariableop>savev2_adam_batch_normalization_540_beta_m_read_readvariableop2savev2_adam_dense_596_kernel_m_read_readvariableop0savev2_adam_dense_596_bias_m_read_readvariableop2savev2_adam_dense_588_kernel_v_read_readvariableop0savev2_adam_dense_588_bias_v_read_readvariableop?savev2_adam_batch_normalization_533_gamma_v_read_readvariableop>savev2_adam_batch_normalization_533_beta_v_read_readvariableop2savev2_adam_dense_589_kernel_v_read_readvariableop0savev2_adam_dense_589_bias_v_read_readvariableop?savev2_adam_batch_normalization_534_gamma_v_read_readvariableop>savev2_adam_batch_normalization_534_beta_v_read_readvariableop2savev2_adam_dense_590_kernel_v_read_readvariableop0savev2_adam_dense_590_bias_v_read_readvariableop?savev2_adam_batch_normalization_535_gamma_v_read_readvariableop>savev2_adam_batch_normalization_535_beta_v_read_readvariableop2savev2_adam_dense_591_kernel_v_read_readvariableop0savev2_adam_dense_591_bias_v_read_readvariableop?savev2_adam_batch_normalization_536_gamma_v_read_readvariableop>savev2_adam_batch_normalization_536_beta_v_read_readvariableop2savev2_adam_dense_592_kernel_v_read_readvariableop0savev2_adam_dense_592_bias_v_read_readvariableop?savev2_adam_batch_normalization_537_gamma_v_read_readvariableop>savev2_adam_batch_normalization_537_beta_v_read_readvariableop2savev2_adam_dense_593_kernel_v_read_readvariableop0savev2_adam_dense_593_bias_v_read_readvariableop?savev2_adam_batch_normalization_538_gamma_v_read_readvariableop>savev2_adam_batch_normalization_538_beta_v_read_readvariableop2savev2_adam_dense_594_kernel_v_read_readvariableop0savev2_adam_dense_594_bias_v_read_readvariableop?savev2_adam_batch_normalization_539_gamma_v_read_readvariableop>savev2_adam_batch_normalization_539_beta_v_read_readvariableop2savev2_adam_dense_595_kernel_v_read_readvariableop0savev2_adam_dense_595_bias_v_read_readvariableop?savev2_adam_batch_normalization_540_gamma_v_read_readvariableop>savev2_adam_batch_normalization_540_beta_v_read_readvariableop2savev2_adam_dense_596_kernel_v_read_readvariableop0savev2_adam_dense_596_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *
dtypes
2		
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*ã
_input_shapesÑ
Î: ::: :Y:Y:Y:Y:Y:Y:YY:Y:Y:Y:Y:Y:Y::::::::::::::::::::::::E:E:E:E:E:E:EE:E:E:E:E:E:E:: : : : : : :Y:Y:Y:Y:YY:Y:Y:Y:Y::::::::::::::::E:E:E:E:EE:E:E:E:E::Y:Y:Y:Y:YY:Y:Y:Y:Y::::::::::::::::E:E:E:E:EE:E:E:E:E:: 2(
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

:Y: 

_output_shapes
:Y: 

_output_shapes
:Y: 

_output_shapes
:Y: 

_output_shapes
:Y: 	

_output_shapes
:Y:$
 

_output_shapes

:YY: 

_output_shapes
:Y: 

_output_shapes
:Y: 

_output_shapes
:Y: 

_output_shapes
:Y: 

_output_shapes
:Y:$ 

_output_shapes

:Y: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
::$( 

_output_shapes

:E: )

_output_shapes
:E: *

_output_shapes
:E: +

_output_shapes
:E: ,

_output_shapes
:E: -

_output_shapes
:E:$. 

_output_shapes

:EE: /

_output_shapes
:E: 0

_output_shapes
:E: 1

_output_shapes
:E: 2

_output_shapes
:E: 3

_output_shapes
:E:$4 

_output_shapes

:E: 5

_output_shapes
::6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :$< 

_output_shapes

:Y: =

_output_shapes
:Y: >

_output_shapes
:Y: ?

_output_shapes
:Y:$@ 

_output_shapes

:YY: A

_output_shapes
:Y: B

_output_shapes
:Y: C

_output_shapes
:Y:$D 

_output_shapes

:Y: E

_output_shapes
:: F

_output_shapes
:: G

_output_shapes
::$H 

_output_shapes

:: I

_output_shapes
:: J

_output_shapes
:: K

_output_shapes
::$L 

_output_shapes

:: M

_output_shapes
:: N

_output_shapes
:: O

_output_shapes
::$P 

_output_shapes

:: Q

_output_shapes
:: R

_output_shapes
:: S

_output_shapes
::$T 

_output_shapes

:E: U

_output_shapes
:E: V

_output_shapes
:E: W

_output_shapes
:E:$X 

_output_shapes

:EE: Y

_output_shapes
:E: Z

_output_shapes
:E: [

_output_shapes
:E:$\ 

_output_shapes

:E: ]

_output_shapes
::$^ 

_output_shapes

:Y: _

_output_shapes
:Y: `

_output_shapes
:Y: a

_output_shapes
:Y:$b 

_output_shapes

:YY: c

_output_shapes
:Y: d

_output_shapes
:Y: e

_output_shapes
:Y:$f 

_output_shapes

:Y: g

_output_shapes
:: h

_output_shapes
:: i

_output_shapes
::$j 

_output_shapes

:: k

_output_shapes
:: l

_output_shapes
:: m

_output_shapes
::$n 

_output_shapes

:: o

_output_shapes
:: p

_output_shapes
:: q

_output_shapes
::$r 

_output_shapes

:: s

_output_shapes
:: t

_output_shapes
:: u

_output_shapes
::$v 

_output_shapes

:E: w

_output_shapes
:E: x

_output_shapes
:E: y

_output_shapes
:E:$z 

_output_shapes

:EE: {

_output_shapes
:E: |

_output_shapes
:E: }

_output_shapes
:E:$~ 

_output_shapes

:E: 

_output_shapes
::

_output_shapes
: 
Ä

*__inference_dense_589_layer_call_fn_666634

inputs
unknown:YY
	unknown_0:Y
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_589_layer_call_and_return_conditional_losses_664411o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
Ä

*__inference_dense_592_layer_call_fn_666961

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_592_layer_call_and_return_conditional_losses_664507o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_534_layer_call_fn_666657

inputs
unknown:Y
	unknown_0:Y
	unknown_1:Y
	unknown_2:Y
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_663805o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_536_layer_call_fn_666888

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_664016o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_sequential_55_layer_call_and_return_conditional_losses_664642

inputs
normalization_55_sub_y
normalization_55_sqrt_x"
dense_588_664380:Y
dense_588_664382:Y,
batch_normalization_533_664385:Y,
batch_normalization_533_664387:Y,
batch_normalization_533_664389:Y,
batch_normalization_533_664391:Y"
dense_589_664412:YY
dense_589_664414:Y,
batch_normalization_534_664417:Y,
batch_normalization_534_664419:Y,
batch_normalization_534_664421:Y,
batch_normalization_534_664423:Y"
dense_590_664444:Y
dense_590_664446:,
batch_normalization_535_664449:,
batch_normalization_535_664451:,
batch_normalization_535_664453:,
batch_normalization_535_664455:"
dense_591_664476:
dense_591_664478:,
batch_normalization_536_664481:,
batch_normalization_536_664483:,
batch_normalization_536_664485:,
batch_normalization_536_664487:"
dense_592_664508:
dense_592_664510:,
batch_normalization_537_664513:,
batch_normalization_537_664515:,
batch_normalization_537_664517:,
batch_normalization_537_664519:"
dense_593_664540:
dense_593_664542:,
batch_normalization_538_664545:,
batch_normalization_538_664547:,
batch_normalization_538_664549:,
batch_normalization_538_664551:"
dense_594_664572:E
dense_594_664574:E,
batch_normalization_539_664577:E,
batch_normalization_539_664579:E,
batch_normalization_539_664581:E,
batch_normalization_539_664583:E"
dense_595_664604:EE
dense_595_664606:E,
batch_normalization_540_664609:E,
batch_normalization_540_664611:E,
batch_normalization_540_664613:E,
batch_normalization_540_664615:E"
dense_596_664636:E
dense_596_664638:
identity¢/batch_normalization_533/StatefulPartitionedCall¢/batch_normalization_534/StatefulPartitionedCall¢/batch_normalization_535/StatefulPartitionedCall¢/batch_normalization_536/StatefulPartitionedCall¢/batch_normalization_537/StatefulPartitionedCall¢/batch_normalization_538/StatefulPartitionedCall¢/batch_normalization_539/StatefulPartitionedCall¢/batch_normalization_540/StatefulPartitionedCall¢!dense_588/StatefulPartitionedCall¢!dense_589/StatefulPartitionedCall¢!dense_590/StatefulPartitionedCall¢!dense_591/StatefulPartitionedCall¢!dense_592/StatefulPartitionedCall¢!dense_593/StatefulPartitionedCall¢!dense_594/StatefulPartitionedCall¢!dense_595/StatefulPartitionedCall¢!dense_596/StatefulPartitionedCallm
normalization_55/subSubinputsnormalization_55_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_55/SqrtSqrtnormalization_55_sqrt_x*
T0*
_output_shapes

:_
normalization_55/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_55/MaximumMaximumnormalization_55/Sqrt:y:0#normalization_55/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_55/truedivRealDivnormalization_55/sub:z:0normalization_55/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_588/StatefulPartitionedCallStatefulPartitionedCallnormalization_55/truediv:z:0dense_588_664380dense_588_664382*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_588_layer_call_and_return_conditional_losses_664379
/batch_normalization_533/StatefulPartitionedCallStatefulPartitionedCall*dense_588/StatefulPartitionedCall:output:0batch_normalization_533_664385batch_normalization_533_664387batch_normalization_533_664389batch_normalization_533_664391*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_663723ø
leaky_re_lu_533/PartitionedCallPartitionedCall8batch_normalization_533/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_533_layer_call_and_return_conditional_losses_664399
!dense_589/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_533/PartitionedCall:output:0dense_589_664412dense_589_664414*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_589_layer_call_and_return_conditional_losses_664411
/batch_normalization_534/StatefulPartitionedCallStatefulPartitionedCall*dense_589/StatefulPartitionedCall:output:0batch_normalization_534_664417batch_normalization_534_664419batch_normalization_534_664421batch_normalization_534_664423*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_663805ø
leaky_re_lu_534/PartitionedCallPartitionedCall8batch_normalization_534/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_534_layer_call_and_return_conditional_losses_664431
!dense_590/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_534/PartitionedCall:output:0dense_590_664444dense_590_664446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_590_layer_call_and_return_conditional_losses_664443
/batch_normalization_535/StatefulPartitionedCallStatefulPartitionedCall*dense_590/StatefulPartitionedCall:output:0batch_normalization_535_664449batch_normalization_535_664451batch_normalization_535_664453batch_normalization_535_664455*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_663887ø
leaky_re_lu_535/PartitionedCallPartitionedCall8batch_normalization_535/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_535_layer_call_and_return_conditional_losses_664463
!dense_591/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_535/PartitionedCall:output:0dense_591_664476dense_591_664478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_591_layer_call_and_return_conditional_losses_664475
/batch_normalization_536/StatefulPartitionedCallStatefulPartitionedCall*dense_591/StatefulPartitionedCall:output:0batch_normalization_536_664481batch_normalization_536_664483batch_normalization_536_664485batch_normalization_536_664487*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_663969ø
leaky_re_lu_536/PartitionedCallPartitionedCall8batch_normalization_536/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_536_layer_call_and_return_conditional_losses_664495
!dense_592/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_536/PartitionedCall:output:0dense_592_664508dense_592_664510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_592_layer_call_and_return_conditional_losses_664507
/batch_normalization_537/StatefulPartitionedCallStatefulPartitionedCall*dense_592/StatefulPartitionedCall:output:0batch_normalization_537_664513batch_normalization_537_664515batch_normalization_537_664517batch_normalization_537_664519*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_664051ø
leaky_re_lu_537/PartitionedCallPartitionedCall8batch_normalization_537/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_537_layer_call_and_return_conditional_losses_664527
!dense_593/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_537/PartitionedCall:output:0dense_593_664540dense_593_664542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_593_layer_call_and_return_conditional_losses_664539
/batch_normalization_538/StatefulPartitionedCallStatefulPartitionedCall*dense_593/StatefulPartitionedCall:output:0batch_normalization_538_664545batch_normalization_538_664547batch_normalization_538_664549batch_normalization_538_664551*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_664133ø
leaky_re_lu_538/PartitionedCallPartitionedCall8batch_normalization_538/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_538_layer_call_and_return_conditional_losses_664559
!dense_594/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_538/PartitionedCall:output:0dense_594_664572dense_594_664574*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_594_layer_call_and_return_conditional_losses_664571
/batch_normalization_539/StatefulPartitionedCallStatefulPartitionedCall*dense_594/StatefulPartitionedCall:output:0batch_normalization_539_664577batch_normalization_539_664579batch_normalization_539_664581batch_normalization_539_664583*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_664215ø
leaky_re_lu_539/PartitionedCallPartitionedCall8batch_normalization_539/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_539_layer_call_and_return_conditional_losses_664591
!dense_595/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_539/PartitionedCall:output:0dense_595_664604dense_595_664606*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_595_layer_call_and_return_conditional_losses_664603
/batch_normalization_540/StatefulPartitionedCallStatefulPartitionedCall*dense_595/StatefulPartitionedCall:output:0batch_normalization_540_664609batch_normalization_540_664611batch_normalization_540_664613batch_normalization_540_664615*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_664297ø
leaky_re_lu_540/PartitionedCallPartitionedCall8batch_normalization_540/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_540_layer_call_and_return_conditional_losses_664623
!dense_596/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_540/PartitionedCall:output:0dense_596_664636dense_596_664638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_596_layer_call_and_return_conditional_losses_664635y
IdentityIdentity*dense_596/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_533/StatefulPartitionedCall0^batch_normalization_534/StatefulPartitionedCall0^batch_normalization_535/StatefulPartitionedCall0^batch_normalization_536/StatefulPartitionedCall0^batch_normalization_537/StatefulPartitionedCall0^batch_normalization_538/StatefulPartitionedCall0^batch_normalization_539/StatefulPartitionedCall0^batch_normalization_540/StatefulPartitionedCall"^dense_588/StatefulPartitionedCall"^dense_589/StatefulPartitionedCall"^dense_590/StatefulPartitionedCall"^dense_591/StatefulPartitionedCall"^dense_592/StatefulPartitionedCall"^dense_593/StatefulPartitionedCall"^dense_594/StatefulPartitionedCall"^dense_595/StatefulPartitionedCall"^dense_596/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_533/StatefulPartitionedCall/batch_normalization_533/StatefulPartitionedCall2b
/batch_normalization_534/StatefulPartitionedCall/batch_normalization_534/StatefulPartitionedCall2b
/batch_normalization_535/StatefulPartitionedCall/batch_normalization_535/StatefulPartitionedCall2b
/batch_normalization_536/StatefulPartitionedCall/batch_normalization_536/StatefulPartitionedCall2b
/batch_normalization_537/StatefulPartitionedCall/batch_normalization_537/StatefulPartitionedCall2b
/batch_normalization_538/StatefulPartitionedCall/batch_normalization_538/StatefulPartitionedCall2b
/batch_normalization_539/StatefulPartitionedCall/batch_normalization_539/StatefulPartitionedCall2b
/batch_normalization_540/StatefulPartitionedCall/batch_normalization_540/StatefulPartitionedCall2F
!dense_588/StatefulPartitionedCall!dense_588/StatefulPartitionedCall2F
!dense_589/StatefulPartitionedCall!dense_589/StatefulPartitionedCall2F
!dense_590/StatefulPartitionedCall!dense_590/StatefulPartitionedCall2F
!dense_591/StatefulPartitionedCall!dense_591/StatefulPartitionedCall2F
!dense_592/StatefulPartitionedCall!dense_592/StatefulPartitionedCall2F
!dense_593/StatefulPartitionedCall!dense_593/StatefulPartitionedCall2F
!dense_594/StatefulPartitionedCall!dense_594/StatefulPartitionedCall2F
!dense_595/StatefulPartitionedCall!dense_595/StatefulPartitionedCall2F
!dense_596/StatefulPartitionedCall!dense_596/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ä

*__inference_dense_588_layer_call_fn_666525

inputs
unknown:Y
	unknown_0:Y
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_588_layer_call_and_return_conditional_losses_664379o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_534_layer_call_and_return_conditional_losses_666734

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿY:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
È	
ö
E__inference_dense_588_layer_call_and_return_conditional_losses_666535

inputs0
matmul_readvariableop_resource:Y-
biasadd_readvariableop_resource:Y
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Y*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Y*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_594_layer_call_and_return_conditional_losses_667189

inputs0
matmul_readvariableop_resource:E-
biasadd_readvariableop_resource:E
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:E*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:E*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_536_layer_call_fn_666875

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_663969o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_592_layer_call_and_return_conditional_losses_666971

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_535_layer_call_and_return_conditional_losses_666843

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_540_layer_call_and_return_conditional_losses_667388

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿE:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_537_layer_call_fn_666997

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_664098o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_663805

inputs/
!batchnorm_readvariableop_resource:Y3
%batchnorm_mul_readvariableop_resource:Y1
#batchnorm_readvariableop_1_resource:Y1
#batchnorm_readvariableop_2_resource:Y
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:YP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Y~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Yc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:Y*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Yz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:Y*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
È	
ö
E__inference_dense_591_layer_call_and_return_conditional_losses_664475

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
û4
I__inference_sequential_55_layer_call_and_return_conditional_losses_666358

inputs
normalization_55_sub_y
normalization_55_sqrt_x:
(dense_588_matmul_readvariableop_resource:Y7
)dense_588_biasadd_readvariableop_resource:YM
?batch_normalization_533_assignmovingavg_readvariableop_resource:YO
Abatch_normalization_533_assignmovingavg_1_readvariableop_resource:YK
=batch_normalization_533_batchnorm_mul_readvariableop_resource:YG
9batch_normalization_533_batchnorm_readvariableop_resource:Y:
(dense_589_matmul_readvariableop_resource:YY7
)dense_589_biasadd_readvariableop_resource:YM
?batch_normalization_534_assignmovingavg_readvariableop_resource:YO
Abatch_normalization_534_assignmovingavg_1_readvariableop_resource:YK
=batch_normalization_534_batchnorm_mul_readvariableop_resource:YG
9batch_normalization_534_batchnorm_readvariableop_resource:Y:
(dense_590_matmul_readvariableop_resource:Y7
)dense_590_biasadd_readvariableop_resource:M
?batch_normalization_535_assignmovingavg_readvariableop_resource:O
Abatch_normalization_535_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_535_batchnorm_mul_readvariableop_resource:G
9batch_normalization_535_batchnorm_readvariableop_resource::
(dense_591_matmul_readvariableop_resource:7
)dense_591_biasadd_readvariableop_resource:M
?batch_normalization_536_assignmovingavg_readvariableop_resource:O
Abatch_normalization_536_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_536_batchnorm_mul_readvariableop_resource:G
9batch_normalization_536_batchnorm_readvariableop_resource::
(dense_592_matmul_readvariableop_resource:7
)dense_592_biasadd_readvariableop_resource:M
?batch_normalization_537_assignmovingavg_readvariableop_resource:O
Abatch_normalization_537_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_537_batchnorm_mul_readvariableop_resource:G
9batch_normalization_537_batchnorm_readvariableop_resource::
(dense_593_matmul_readvariableop_resource:7
)dense_593_biasadd_readvariableop_resource:M
?batch_normalization_538_assignmovingavg_readvariableop_resource:O
Abatch_normalization_538_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_538_batchnorm_mul_readvariableop_resource:G
9batch_normalization_538_batchnorm_readvariableop_resource::
(dense_594_matmul_readvariableop_resource:E7
)dense_594_biasadd_readvariableop_resource:EM
?batch_normalization_539_assignmovingavg_readvariableop_resource:EO
Abatch_normalization_539_assignmovingavg_1_readvariableop_resource:EK
=batch_normalization_539_batchnorm_mul_readvariableop_resource:EG
9batch_normalization_539_batchnorm_readvariableop_resource:E:
(dense_595_matmul_readvariableop_resource:EE7
)dense_595_biasadd_readvariableop_resource:EM
?batch_normalization_540_assignmovingavg_readvariableop_resource:EO
Abatch_normalization_540_assignmovingavg_1_readvariableop_resource:EK
=batch_normalization_540_batchnorm_mul_readvariableop_resource:EG
9batch_normalization_540_batchnorm_readvariableop_resource:E:
(dense_596_matmul_readvariableop_resource:E7
)dense_596_biasadd_readvariableop_resource:
identity¢'batch_normalization_533/AssignMovingAvg¢6batch_normalization_533/AssignMovingAvg/ReadVariableOp¢)batch_normalization_533/AssignMovingAvg_1¢8batch_normalization_533/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_533/batchnorm/ReadVariableOp¢4batch_normalization_533/batchnorm/mul/ReadVariableOp¢'batch_normalization_534/AssignMovingAvg¢6batch_normalization_534/AssignMovingAvg/ReadVariableOp¢)batch_normalization_534/AssignMovingAvg_1¢8batch_normalization_534/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_534/batchnorm/ReadVariableOp¢4batch_normalization_534/batchnorm/mul/ReadVariableOp¢'batch_normalization_535/AssignMovingAvg¢6batch_normalization_535/AssignMovingAvg/ReadVariableOp¢)batch_normalization_535/AssignMovingAvg_1¢8batch_normalization_535/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_535/batchnorm/ReadVariableOp¢4batch_normalization_535/batchnorm/mul/ReadVariableOp¢'batch_normalization_536/AssignMovingAvg¢6batch_normalization_536/AssignMovingAvg/ReadVariableOp¢)batch_normalization_536/AssignMovingAvg_1¢8batch_normalization_536/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_536/batchnorm/ReadVariableOp¢4batch_normalization_536/batchnorm/mul/ReadVariableOp¢'batch_normalization_537/AssignMovingAvg¢6batch_normalization_537/AssignMovingAvg/ReadVariableOp¢)batch_normalization_537/AssignMovingAvg_1¢8batch_normalization_537/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_537/batchnorm/ReadVariableOp¢4batch_normalization_537/batchnorm/mul/ReadVariableOp¢'batch_normalization_538/AssignMovingAvg¢6batch_normalization_538/AssignMovingAvg/ReadVariableOp¢)batch_normalization_538/AssignMovingAvg_1¢8batch_normalization_538/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_538/batchnorm/ReadVariableOp¢4batch_normalization_538/batchnorm/mul/ReadVariableOp¢'batch_normalization_539/AssignMovingAvg¢6batch_normalization_539/AssignMovingAvg/ReadVariableOp¢)batch_normalization_539/AssignMovingAvg_1¢8batch_normalization_539/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_539/batchnorm/ReadVariableOp¢4batch_normalization_539/batchnorm/mul/ReadVariableOp¢'batch_normalization_540/AssignMovingAvg¢6batch_normalization_540/AssignMovingAvg/ReadVariableOp¢)batch_normalization_540/AssignMovingAvg_1¢8batch_normalization_540/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_540/batchnorm/ReadVariableOp¢4batch_normalization_540/batchnorm/mul/ReadVariableOp¢ dense_588/BiasAdd/ReadVariableOp¢dense_588/MatMul/ReadVariableOp¢ dense_589/BiasAdd/ReadVariableOp¢dense_589/MatMul/ReadVariableOp¢ dense_590/BiasAdd/ReadVariableOp¢dense_590/MatMul/ReadVariableOp¢ dense_591/BiasAdd/ReadVariableOp¢dense_591/MatMul/ReadVariableOp¢ dense_592/BiasAdd/ReadVariableOp¢dense_592/MatMul/ReadVariableOp¢ dense_593/BiasAdd/ReadVariableOp¢dense_593/MatMul/ReadVariableOp¢ dense_594/BiasAdd/ReadVariableOp¢dense_594/MatMul/ReadVariableOp¢ dense_595/BiasAdd/ReadVariableOp¢dense_595/MatMul/ReadVariableOp¢ dense_596/BiasAdd/ReadVariableOp¢dense_596/MatMul/ReadVariableOpm
normalization_55/subSubinputsnormalization_55_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_55/SqrtSqrtnormalization_55_sqrt_x*
T0*
_output_shapes

:_
normalization_55/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_55/MaximumMaximumnormalization_55/Sqrt:y:0#normalization_55/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_55/truedivRealDivnormalization_55/sub:z:0normalization_55/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_588/MatMul/ReadVariableOpReadVariableOp(dense_588_matmul_readvariableop_resource*
_output_shapes

:Y*
dtype0
dense_588/MatMulMatMulnormalization_55/truediv:z:0'dense_588/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 dense_588/BiasAdd/ReadVariableOpReadVariableOp)dense_588_biasadd_readvariableop_resource*
_output_shapes
:Y*
dtype0
dense_588/BiasAddBiasAdddense_588/MatMul:product:0(dense_588/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
6batch_normalization_533/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_533/moments/meanMeandense_588/BiasAdd:output:0?batch_normalization_533/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Y*
	keep_dims(
,batch_normalization_533/moments/StopGradientStopGradient-batch_normalization_533/moments/mean:output:0*
T0*
_output_shapes

:YË
1batch_normalization_533/moments/SquaredDifferenceSquaredDifferencedense_588/BiasAdd:output:05batch_normalization_533/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
:batch_normalization_533/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_533/moments/varianceMean5batch_normalization_533/moments/SquaredDifference:z:0Cbatch_normalization_533/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Y*
	keep_dims(
'batch_normalization_533/moments/SqueezeSqueeze-batch_normalization_533/moments/mean:output:0*
T0*
_output_shapes
:Y*
squeeze_dims
 £
)batch_normalization_533/moments/Squeeze_1Squeeze1batch_normalization_533/moments/variance:output:0*
T0*
_output_shapes
:Y*
squeeze_dims
 r
-batch_normalization_533/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_533/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_533_assignmovingavg_readvariableop_resource*
_output_shapes
:Y*
dtype0É
+batch_normalization_533/AssignMovingAvg/subSub>batch_normalization_533/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_533/moments/Squeeze:output:0*
T0*
_output_shapes
:YÀ
+batch_normalization_533/AssignMovingAvg/mulMul/batch_normalization_533/AssignMovingAvg/sub:z:06batch_normalization_533/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Y
'batch_normalization_533/AssignMovingAvgAssignSubVariableOp?batch_normalization_533_assignmovingavg_readvariableop_resource/batch_normalization_533/AssignMovingAvg/mul:z:07^batch_normalization_533/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_533/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_533/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_533_assignmovingavg_1_readvariableop_resource*
_output_shapes
:Y*
dtype0Ï
-batch_normalization_533/AssignMovingAvg_1/subSub@batch_normalization_533/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_533/moments/Squeeze_1:output:0*
T0*
_output_shapes
:YÆ
-batch_normalization_533/AssignMovingAvg_1/mulMul1batch_normalization_533/AssignMovingAvg_1/sub:z:08batch_normalization_533/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Y
)batch_normalization_533/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_533_assignmovingavg_1_readvariableop_resource1batch_normalization_533/AssignMovingAvg_1/mul:z:09^batch_normalization_533/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_533/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_533/batchnorm/addAddV22batch_normalization_533/moments/Squeeze_1:output:00batch_normalization_533/batchnorm/add/y:output:0*
T0*
_output_shapes
:Y
'batch_normalization_533/batchnorm/RsqrtRsqrt)batch_normalization_533/batchnorm/add:z:0*
T0*
_output_shapes
:Y®
4batch_normalization_533/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_533_batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0¼
%batch_normalization_533/batchnorm/mulMul+batch_normalization_533/batchnorm/Rsqrt:y:0<batch_normalization_533/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Y§
'batch_normalization_533/batchnorm/mul_1Muldense_588/BiasAdd:output:0)batch_normalization_533/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY°
'batch_normalization_533/batchnorm/mul_2Mul0batch_normalization_533/moments/Squeeze:output:0)batch_normalization_533/batchnorm/mul:z:0*
T0*
_output_shapes
:Y¦
0batch_normalization_533/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_533_batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0¸
%batch_normalization_533/batchnorm/subSub8batch_normalization_533/batchnorm/ReadVariableOp:value:0+batch_normalization_533/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yº
'batch_normalization_533/batchnorm/add_1AddV2+batch_normalization_533/batchnorm/mul_1:z:0)batch_normalization_533/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
leaky_re_lu_533/LeakyRelu	LeakyRelu+batch_normalization_533/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*
alpha%>
dense_589/MatMul/ReadVariableOpReadVariableOp(dense_589_matmul_readvariableop_resource*
_output_shapes

:YY*
dtype0
dense_589/MatMulMatMul'leaky_re_lu_533/LeakyRelu:activations:0'dense_589/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 dense_589/BiasAdd/ReadVariableOpReadVariableOp)dense_589_biasadd_readvariableop_resource*
_output_shapes
:Y*
dtype0
dense_589/BiasAddBiasAdddense_589/MatMul:product:0(dense_589/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
6batch_normalization_534/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_534/moments/meanMeandense_589/BiasAdd:output:0?batch_normalization_534/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Y*
	keep_dims(
,batch_normalization_534/moments/StopGradientStopGradient-batch_normalization_534/moments/mean:output:0*
T0*
_output_shapes

:YË
1batch_normalization_534/moments/SquaredDifferenceSquaredDifferencedense_589/BiasAdd:output:05batch_normalization_534/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
:batch_normalization_534/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_534/moments/varianceMean5batch_normalization_534/moments/SquaredDifference:z:0Cbatch_normalization_534/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Y*
	keep_dims(
'batch_normalization_534/moments/SqueezeSqueeze-batch_normalization_534/moments/mean:output:0*
T0*
_output_shapes
:Y*
squeeze_dims
 £
)batch_normalization_534/moments/Squeeze_1Squeeze1batch_normalization_534/moments/variance:output:0*
T0*
_output_shapes
:Y*
squeeze_dims
 r
-batch_normalization_534/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_534/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_534_assignmovingavg_readvariableop_resource*
_output_shapes
:Y*
dtype0É
+batch_normalization_534/AssignMovingAvg/subSub>batch_normalization_534/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_534/moments/Squeeze:output:0*
T0*
_output_shapes
:YÀ
+batch_normalization_534/AssignMovingAvg/mulMul/batch_normalization_534/AssignMovingAvg/sub:z:06batch_normalization_534/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Y
'batch_normalization_534/AssignMovingAvgAssignSubVariableOp?batch_normalization_534_assignmovingavg_readvariableop_resource/batch_normalization_534/AssignMovingAvg/mul:z:07^batch_normalization_534/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_534/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_534/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_534_assignmovingavg_1_readvariableop_resource*
_output_shapes
:Y*
dtype0Ï
-batch_normalization_534/AssignMovingAvg_1/subSub@batch_normalization_534/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_534/moments/Squeeze_1:output:0*
T0*
_output_shapes
:YÆ
-batch_normalization_534/AssignMovingAvg_1/mulMul1batch_normalization_534/AssignMovingAvg_1/sub:z:08batch_normalization_534/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Y
)batch_normalization_534/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_534_assignmovingavg_1_readvariableop_resource1batch_normalization_534/AssignMovingAvg_1/mul:z:09^batch_normalization_534/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_534/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_534/batchnorm/addAddV22batch_normalization_534/moments/Squeeze_1:output:00batch_normalization_534/batchnorm/add/y:output:0*
T0*
_output_shapes
:Y
'batch_normalization_534/batchnorm/RsqrtRsqrt)batch_normalization_534/batchnorm/add:z:0*
T0*
_output_shapes
:Y®
4batch_normalization_534/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_534_batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0¼
%batch_normalization_534/batchnorm/mulMul+batch_normalization_534/batchnorm/Rsqrt:y:0<batch_normalization_534/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Y§
'batch_normalization_534/batchnorm/mul_1Muldense_589/BiasAdd:output:0)batch_normalization_534/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY°
'batch_normalization_534/batchnorm/mul_2Mul0batch_normalization_534/moments/Squeeze:output:0)batch_normalization_534/batchnorm/mul:z:0*
T0*
_output_shapes
:Y¦
0batch_normalization_534/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_534_batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0¸
%batch_normalization_534/batchnorm/subSub8batch_normalization_534/batchnorm/ReadVariableOp:value:0+batch_normalization_534/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yº
'batch_normalization_534/batchnorm/add_1AddV2+batch_normalization_534/batchnorm/mul_1:z:0)batch_normalization_534/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
leaky_re_lu_534/LeakyRelu	LeakyRelu+batch_normalization_534/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*
alpha%>
dense_590/MatMul/ReadVariableOpReadVariableOp(dense_590_matmul_readvariableop_resource*
_output_shapes

:Y*
dtype0
dense_590/MatMulMatMul'leaky_re_lu_534/LeakyRelu:activations:0'dense_590/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_590/BiasAdd/ReadVariableOpReadVariableOp)dense_590_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_590/BiasAddBiasAdddense_590/MatMul:product:0(dense_590/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_535/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_535/moments/meanMeandense_590/BiasAdd:output:0?batch_normalization_535/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_535/moments/StopGradientStopGradient-batch_normalization_535/moments/mean:output:0*
T0*
_output_shapes

:Ë
1batch_normalization_535/moments/SquaredDifferenceSquaredDifferencedense_590/BiasAdd:output:05batch_normalization_535/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_535/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_535/moments/varianceMean5batch_normalization_535/moments/SquaredDifference:z:0Cbatch_normalization_535/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_535/moments/SqueezeSqueeze-batch_normalization_535/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_535/moments/Squeeze_1Squeeze1batch_normalization_535/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_535/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_535/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_535_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_535/AssignMovingAvg/subSub>batch_normalization_535/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_535/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_535/AssignMovingAvg/mulMul/batch_normalization_535/AssignMovingAvg/sub:z:06batch_normalization_535/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_535/AssignMovingAvgAssignSubVariableOp?batch_normalization_535_assignmovingavg_readvariableop_resource/batch_normalization_535/AssignMovingAvg/mul:z:07^batch_normalization_535/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_535/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_535/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_535_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_535/AssignMovingAvg_1/subSub@batch_normalization_535/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_535/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_535/AssignMovingAvg_1/mulMul1batch_normalization_535/AssignMovingAvg_1/sub:z:08batch_normalization_535/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_535/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_535_assignmovingavg_1_readvariableop_resource1batch_normalization_535/AssignMovingAvg_1/mul:z:09^batch_normalization_535/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_535/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_535/batchnorm/addAddV22batch_normalization_535/moments/Squeeze_1:output:00batch_normalization_535/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_535/batchnorm/RsqrtRsqrt)batch_normalization_535/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_535/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_535_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_535/batchnorm/mulMul+batch_normalization_535/batchnorm/Rsqrt:y:0<batch_normalization_535/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_535/batchnorm/mul_1Muldense_590/BiasAdd:output:0)batch_normalization_535/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_535/batchnorm/mul_2Mul0batch_normalization_535/moments/Squeeze:output:0)batch_normalization_535/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_535/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_535_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_535/batchnorm/subSub8batch_normalization_535/batchnorm/ReadVariableOp:value:0+batch_normalization_535/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_535/batchnorm/add_1AddV2+batch_normalization_535/batchnorm/mul_1:z:0)batch_normalization_535/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_535/LeakyRelu	LeakyRelu+batch_normalization_535/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_591/MatMul/ReadVariableOpReadVariableOp(dense_591_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_591/MatMulMatMul'leaky_re_lu_535/LeakyRelu:activations:0'dense_591/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_591/BiasAdd/ReadVariableOpReadVariableOp)dense_591_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_591/BiasAddBiasAdddense_591/MatMul:product:0(dense_591/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_536/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_536/moments/meanMeandense_591/BiasAdd:output:0?batch_normalization_536/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_536/moments/StopGradientStopGradient-batch_normalization_536/moments/mean:output:0*
T0*
_output_shapes

:Ë
1batch_normalization_536/moments/SquaredDifferenceSquaredDifferencedense_591/BiasAdd:output:05batch_normalization_536/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_536/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_536/moments/varianceMean5batch_normalization_536/moments/SquaredDifference:z:0Cbatch_normalization_536/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_536/moments/SqueezeSqueeze-batch_normalization_536/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_536/moments/Squeeze_1Squeeze1batch_normalization_536/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_536/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_536/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_536_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_536/AssignMovingAvg/subSub>batch_normalization_536/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_536/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_536/AssignMovingAvg/mulMul/batch_normalization_536/AssignMovingAvg/sub:z:06batch_normalization_536/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_536/AssignMovingAvgAssignSubVariableOp?batch_normalization_536_assignmovingavg_readvariableop_resource/batch_normalization_536/AssignMovingAvg/mul:z:07^batch_normalization_536/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_536/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_536/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_536_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_536/AssignMovingAvg_1/subSub@batch_normalization_536/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_536/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_536/AssignMovingAvg_1/mulMul1batch_normalization_536/AssignMovingAvg_1/sub:z:08batch_normalization_536/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_536/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_536_assignmovingavg_1_readvariableop_resource1batch_normalization_536/AssignMovingAvg_1/mul:z:09^batch_normalization_536/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_536/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_536/batchnorm/addAddV22batch_normalization_536/moments/Squeeze_1:output:00batch_normalization_536/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_536/batchnorm/RsqrtRsqrt)batch_normalization_536/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_536/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_536_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_536/batchnorm/mulMul+batch_normalization_536/batchnorm/Rsqrt:y:0<batch_normalization_536/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_536/batchnorm/mul_1Muldense_591/BiasAdd:output:0)batch_normalization_536/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_536/batchnorm/mul_2Mul0batch_normalization_536/moments/Squeeze:output:0)batch_normalization_536/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_536/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_536_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_536/batchnorm/subSub8batch_normalization_536/batchnorm/ReadVariableOp:value:0+batch_normalization_536/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_536/batchnorm/add_1AddV2+batch_normalization_536/batchnorm/mul_1:z:0)batch_normalization_536/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_536/LeakyRelu	LeakyRelu+batch_normalization_536/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_592/MatMul/ReadVariableOpReadVariableOp(dense_592_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_592/MatMulMatMul'leaky_re_lu_536/LeakyRelu:activations:0'dense_592/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_592/BiasAdd/ReadVariableOpReadVariableOp)dense_592_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_592/BiasAddBiasAdddense_592/MatMul:product:0(dense_592/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_537/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_537/moments/meanMeandense_592/BiasAdd:output:0?batch_normalization_537/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_537/moments/StopGradientStopGradient-batch_normalization_537/moments/mean:output:0*
T0*
_output_shapes

:Ë
1batch_normalization_537/moments/SquaredDifferenceSquaredDifferencedense_592/BiasAdd:output:05batch_normalization_537/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_537/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_537/moments/varianceMean5batch_normalization_537/moments/SquaredDifference:z:0Cbatch_normalization_537/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_537/moments/SqueezeSqueeze-batch_normalization_537/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_537/moments/Squeeze_1Squeeze1batch_normalization_537/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_537/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_537/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_537_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_537/AssignMovingAvg/subSub>batch_normalization_537/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_537/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_537/AssignMovingAvg/mulMul/batch_normalization_537/AssignMovingAvg/sub:z:06batch_normalization_537/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_537/AssignMovingAvgAssignSubVariableOp?batch_normalization_537_assignmovingavg_readvariableop_resource/batch_normalization_537/AssignMovingAvg/mul:z:07^batch_normalization_537/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_537/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_537/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_537_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_537/AssignMovingAvg_1/subSub@batch_normalization_537/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_537/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_537/AssignMovingAvg_1/mulMul1batch_normalization_537/AssignMovingAvg_1/sub:z:08batch_normalization_537/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_537/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_537_assignmovingavg_1_readvariableop_resource1batch_normalization_537/AssignMovingAvg_1/mul:z:09^batch_normalization_537/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_537/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_537/batchnorm/addAddV22batch_normalization_537/moments/Squeeze_1:output:00batch_normalization_537/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_537/batchnorm/RsqrtRsqrt)batch_normalization_537/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_537/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_537_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_537/batchnorm/mulMul+batch_normalization_537/batchnorm/Rsqrt:y:0<batch_normalization_537/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_537/batchnorm/mul_1Muldense_592/BiasAdd:output:0)batch_normalization_537/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_537/batchnorm/mul_2Mul0batch_normalization_537/moments/Squeeze:output:0)batch_normalization_537/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_537/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_537_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_537/batchnorm/subSub8batch_normalization_537/batchnorm/ReadVariableOp:value:0+batch_normalization_537/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_537/batchnorm/add_1AddV2+batch_normalization_537/batchnorm/mul_1:z:0)batch_normalization_537/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_537/LeakyRelu	LeakyRelu+batch_normalization_537/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_593/MatMul/ReadVariableOpReadVariableOp(dense_593_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_593/MatMulMatMul'leaky_re_lu_537/LeakyRelu:activations:0'dense_593/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_593/BiasAdd/ReadVariableOpReadVariableOp)dense_593_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_593/BiasAddBiasAdddense_593/MatMul:product:0(dense_593/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_538/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_538/moments/meanMeandense_593/BiasAdd:output:0?batch_normalization_538/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_538/moments/StopGradientStopGradient-batch_normalization_538/moments/mean:output:0*
T0*
_output_shapes

:Ë
1batch_normalization_538/moments/SquaredDifferenceSquaredDifferencedense_593/BiasAdd:output:05batch_normalization_538/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_538/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_538/moments/varianceMean5batch_normalization_538/moments/SquaredDifference:z:0Cbatch_normalization_538/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_538/moments/SqueezeSqueeze-batch_normalization_538/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_538/moments/Squeeze_1Squeeze1batch_normalization_538/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_538/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_538/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_538_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_538/AssignMovingAvg/subSub>batch_normalization_538/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_538/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_538/AssignMovingAvg/mulMul/batch_normalization_538/AssignMovingAvg/sub:z:06batch_normalization_538/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_538/AssignMovingAvgAssignSubVariableOp?batch_normalization_538_assignmovingavg_readvariableop_resource/batch_normalization_538/AssignMovingAvg/mul:z:07^batch_normalization_538/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_538/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_538/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_538_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_538/AssignMovingAvg_1/subSub@batch_normalization_538/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_538/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_538/AssignMovingAvg_1/mulMul1batch_normalization_538/AssignMovingAvg_1/sub:z:08batch_normalization_538/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_538/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_538_assignmovingavg_1_readvariableop_resource1batch_normalization_538/AssignMovingAvg_1/mul:z:09^batch_normalization_538/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_538/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_538/batchnorm/addAddV22batch_normalization_538/moments/Squeeze_1:output:00batch_normalization_538/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_538/batchnorm/RsqrtRsqrt)batch_normalization_538/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_538/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_538_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_538/batchnorm/mulMul+batch_normalization_538/batchnorm/Rsqrt:y:0<batch_normalization_538/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_538/batchnorm/mul_1Muldense_593/BiasAdd:output:0)batch_normalization_538/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_538/batchnorm/mul_2Mul0batch_normalization_538/moments/Squeeze:output:0)batch_normalization_538/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_538/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_538_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_538/batchnorm/subSub8batch_normalization_538/batchnorm/ReadVariableOp:value:0+batch_normalization_538/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_538/batchnorm/add_1AddV2+batch_normalization_538/batchnorm/mul_1:z:0)batch_normalization_538/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_538/LeakyRelu	LeakyRelu+batch_normalization_538/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_594/MatMul/ReadVariableOpReadVariableOp(dense_594_matmul_readvariableop_resource*
_output_shapes

:E*
dtype0
dense_594/MatMulMatMul'leaky_re_lu_538/LeakyRelu:activations:0'dense_594/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 dense_594/BiasAdd/ReadVariableOpReadVariableOp)dense_594_biasadd_readvariableop_resource*
_output_shapes
:E*
dtype0
dense_594/BiasAddBiasAdddense_594/MatMul:product:0(dense_594/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
6batch_normalization_539/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_539/moments/meanMeandense_594/BiasAdd:output:0?batch_normalization_539/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:E*
	keep_dims(
,batch_normalization_539/moments/StopGradientStopGradient-batch_normalization_539/moments/mean:output:0*
T0*
_output_shapes

:EË
1batch_normalization_539/moments/SquaredDifferenceSquaredDifferencedense_594/BiasAdd:output:05batch_normalization_539/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
:batch_normalization_539/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_539/moments/varianceMean5batch_normalization_539/moments/SquaredDifference:z:0Cbatch_normalization_539/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:E*
	keep_dims(
'batch_normalization_539/moments/SqueezeSqueeze-batch_normalization_539/moments/mean:output:0*
T0*
_output_shapes
:E*
squeeze_dims
 £
)batch_normalization_539/moments/Squeeze_1Squeeze1batch_normalization_539/moments/variance:output:0*
T0*
_output_shapes
:E*
squeeze_dims
 r
-batch_normalization_539/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_539/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_539_assignmovingavg_readvariableop_resource*
_output_shapes
:E*
dtype0É
+batch_normalization_539/AssignMovingAvg/subSub>batch_normalization_539/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_539/moments/Squeeze:output:0*
T0*
_output_shapes
:EÀ
+batch_normalization_539/AssignMovingAvg/mulMul/batch_normalization_539/AssignMovingAvg/sub:z:06batch_normalization_539/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:E
'batch_normalization_539/AssignMovingAvgAssignSubVariableOp?batch_normalization_539_assignmovingavg_readvariableop_resource/batch_normalization_539/AssignMovingAvg/mul:z:07^batch_normalization_539/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_539/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_539/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_539_assignmovingavg_1_readvariableop_resource*
_output_shapes
:E*
dtype0Ï
-batch_normalization_539/AssignMovingAvg_1/subSub@batch_normalization_539/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_539/moments/Squeeze_1:output:0*
T0*
_output_shapes
:EÆ
-batch_normalization_539/AssignMovingAvg_1/mulMul1batch_normalization_539/AssignMovingAvg_1/sub:z:08batch_normalization_539/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:E
)batch_normalization_539/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_539_assignmovingavg_1_readvariableop_resource1batch_normalization_539/AssignMovingAvg_1/mul:z:09^batch_normalization_539/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_539/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_539/batchnorm/addAddV22batch_normalization_539/moments/Squeeze_1:output:00batch_normalization_539/batchnorm/add/y:output:0*
T0*
_output_shapes
:E
'batch_normalization_539/batchnorm/RsqrtRsqrt)batch_normalization_539/batchnorm/add:z:0*
T0*
_output_shapes
:E®
4batch_normalization_539/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_539_batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0¼
%batch_normalization_539/batchnorm/mulMul+batch_normalization_539/batchnorm/Rsqrt:y:0<batch_normalization_539/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:E§
'batch_normalization_539/batchnorm/mul_1Muldense_594/BiasAdd:output:0)batch_normalization_539/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE°
'batch_normalization_539/batchnorm/mul_2Mul0batch_normalization_539/moments/Squeeze:output:0)batch_normalization_539/batchnorm/mul:z:0*
T0*
_output_shapes
:E¦
0batch_normalization_539/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_539_batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0¸
%batch_normalization_539/batchnorm/subSub8batch_normalization_539/batchnorm/ReadVariableOp:value:0+batch_normalization_539/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Eº
'batch_normalization_539/batchnorm/add_1AddV2+batch_normalization_539/batchnorm/mul_1:z:0)batch_normalization_539/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
leaky_re_lu_539/LeakyRelu	LeakyRelu+batch_normalization_539/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*
alpha%>
dense_595/MatMul/ReadVariableOpReadVariableOp(dense_595_matmul_readvariableop_resource*
_output_shapes

:EE*
dtype0
dense_595/MatMulMatMul'leaky_re_lu_539/LeakyRelu:activations:0'dense_595/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 dense_595/BiasAdd/ReadVariableOpReadVariableOp)dense_595_biasadd_readvariableop_resource*
_output_shapes
:E*
dtype0
dense_595/BiasAddBiasAdddense_595/MatMul:product:0(dense_595/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
6batch_normalization_540/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_540/moments/meanMeandense_595/BiasAdd:output:0?batch_normalization_540/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:E*
	keep_dims(
,batch_normalization_540/moments/StopGradientStopGradient-batch_normalization_540/moments/mean:output:0*
T0*
_output_shapes

:EË
1batch_normalization_540/moments/SquaredDifferenceSquaredDifferencedense_595/BiasAdd:output:05batch_normalization_540/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
:batch_normalization_540/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_540/moments/varianceMean5batch_normalization_540/moments/SquaredDifference:z:0Cbatch_normalization_540/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:E*
	keep_dims(
'batch_normalization_540/moments/SqueezeSqueeze-batch_normalization_540/moments/mean:output:0*
T0*
_output_shapes
:E*
squeeze_dims
 £
)batch_normalization_540/moments/Squeeze_1Squeeze1batch_normalization_540/moments/variance:output:0*
T0*
_output_shapes
:E*
squeeze_dims
 r
-batch_normalization_540/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_540/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_540_assignmovingavg_readvariableop_resource*
_output_shapes
:E*
dtype0É
+batch_normalization_540/AssignMovingAvg/subSub>batch_normalization_540/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_540/moments/Squeeze:output:0*
T0*
_output_shapes
:EÀ
+batch_normalization_540/AssignMovingAvg/mulMul/batch_normalization_540/AssignMovingAvg/sub:z:06batch_normalization_540/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:E
'batch_normalization_540/AssignMovingAvgAssignSubVariableOp?batch_normalization_540_assignmovingavg_readvariableop_resource/batch_normalization_540/AssignMovingAvg/mul:z:07^batch_normalization_540/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_540/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_540/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_540_assignmovingavg_1_readvariableop_resource*
_output_shapes
:E*
dtype0Ï
-batch_normalization_540/AssignMovingAvg_1/subSub@batch_normalization_540/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_540/moments/Squeeze_1:output:0*
T0*
_output_shapes
:EÆ
-batch_normalization_540/AssignMovingAvg_1/mulMul1batch_normalization_540/AssignMovingAvg_1/sub:z:08batch_normalization_540/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:E
)batch_normalization_540/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_540_assignmovingavg_1_readvariableop_resource1batch_normalization_540/AssignMovingAvg_1/mul:z:09^batch_normalization_540/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_540/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_540/batchnorm/addAddV22batch_normalization_540/moments/Squeeze_1:output:00batch_normalization_540/batchnorm/add/y:output:0*
T0*
_output_shapes
:E
'batch_normalization_540/batchnorm/RsqrtRsqrt)batch_normalization_540/batchnorm/add:z:0*
T0*
_output_shapes
:E®
4batch_normalization_540/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_540_batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0¼
%batch_normalization_540/batchnorm/mulMul+batch_normalization_540/batchnorm/Rsqrt:y:0<batch_normalization_540/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:E§
'batch_normalization_540/batchnorm/mul_1Muldense_595/BiasAdd:output:0)batch_normalization_540/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE°
'batch_normalization_540/batchnorm/mul_2Mul0batch_normalization_540/moments/Squeeze:output:0)batch_normalization_540/batchnorm/mul:z:0*
T0*
_output_shapes
:E¦
0batch_normalization_540/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_540_batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0¸
%batch_normalization_540/batchnorm/subSub8batch_normalization_540/batchnorm/ReadVariableOp:value:0+batch_normalization_540/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Eº
'batch_normalization_540/batchnorm/add_1AddV2+batch_normalization_540/batchnorm/mul_1:z:0)batch_normalization_540/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
leaky_re_lu_540/LeakyRelu	LeakyRelu+batch_normalization_540/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*
alpha%>
dense_596/MatMul/ReadVariableOpReadVariableOp(dense_596_matmul_readvariableop_resource*
_output_shapes

:E*
dtype0
dense_596/MatMulMatMul'leaky_re_lu_540/LeakyRelu:activations:0'dense_596/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_596/BiasAdd/ReadVariableOpReadVariableOp)dense_596_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_596/BiasAddBiasAdddense_596/MatMul:product:0(dense_596/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_596/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp(^batch_normalization_533/AssignMovingAvg7^batch_normalization_533/AssignMovingAvg/ReadVariableOp*^batch_normalization_533/AssignMovingAvg_19^batch_normalization_533/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_533/batchnorm/ReadVariableOp5^batch_normalization_533/batchnorm/mul/ReadVariableOp(^batch_normalization_534/AssignMovingAvg7^batch_normalization_534/AssignMovingAvg/ReadVariableOp*^batch_normalization_534/AssignMovingAvg_19^batch_normalization_534/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_534/batchnorm/ReadVariableOp5^batch_normalization_534/batchnorm/mul/ReadVariableOp(^batch_normalization_535/AssignMovingAvg7^batch_normalization_535/AssignMovingAvg/ReadVariableOp*^batch_normalization_535/AssignMovingAvg_19^batch_normalization_535/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_535/batchnorm/ReadVariableOp5^batch_normalization_535/batchnorm/mul/ReadVariableOp(^batch_normalization_536/AssignMovingAvg7^batch_normalization_536/AssignMovingAvg/ReadVariableOp*^batch_normalization_536/AssignMovingAvg_19^batch_normalization_536/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_536/batchnorm/ReadVariableOp5^batch_normalization_536/batchnorm/mul/ReadVariableOp(^batch_normalization_537/AssignMovingAvg7^batch_normalization_537/AssignMovingAvg/ReadVariableOp*^batch_normalization_537/AssignMovingAvg_19^batch_normalization_537/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_537/batchnorm/ReadVariableOp5^batch_normalization_537/batchnorm/mul/ReadVariableOp(^batch_normalization_538/AssignMovingAvg7^batch_normalization_538/AssignMovingAvg/ReadVariableOp*^batch_normalization_538/AssignMovingAvg_19^batch_normalization_538/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_538/batchnorm/ReadVariableOp5^batch_normalization_538/batchnorm/mul/ReadVariableOp(^batch_normalization_539/AssignMovingAvg7^batch_normalization_539/AssignMovingAvg/ReadVariableOp*^batch_normalization_539/AssignMovingAvg_19^batch_normalization_539/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_539/batchnorm/ReadVariableOp5^batch_normalization_539/batchnorm/mul/ReadVariableOp(^batch_normalization_540/AssignMovingAvg7^batch_normalization_540/AssignMovingAvg/ReadVariableOp*^batch_normalization_540/AssignMovingAvg_19^batch_normalization_540/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_540/batchnorm/ReadVariableOp5^batch_normalization_540/batchnorm/mul/ReadVariableOp!^dense_588/BiasAdd/ReadVariableOp ^dense_588/MatMul/ReadVariableOp!^dense_589/BiasAdd/ReadVariableOp ^dense_589/MatMul/ReadVariableOp!^dense_590/BiasAdd/ReadVariableOp ^dense_590/MatMul/ReadVariableOp!^dense_591/BiasAdd/ReadVariableOp ^dense_591/MatMul/ReadVariableOp!^dense_592/BiasAdd/ReadVariableOp ^dense_592/MatMul/ReadVariableOp!^dense_593/BiasAdd/ReadVariableOp ^dense_593/MatMul/ReadVariableOp!^dense_594/BiasAdd/ReadVariableOp ^dense_594/MatMul/ReadVariableOp!^dense_595/BiasAdd/ReadVariableOp ^dense_595/MatMul/ReadVariableOp!^dense_596/BiasAdd/ReadVariableOp ^dense_596/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_533/AssignMovingAvg'batch_normalization_533/AssignMovingAvg2p
6batch_normalization_533/AssignMovingAvg/ReadVariableOp6batch_normalization_533/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_533/AssignMovingAvg_1)batch_normalization_533/AssignMovingAvg_12t
8batch_normalization_533/AssignMovingAvg_1/ReadVariableOp8batch_normalization_533/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_533/batchnorm/ReadVariableOp0batch_normalization_533/batchnorm/ReadVariableOp2l
4batch_normalization_533/batchnorm/mul/ReadVariableOp4batch_normalization_533/batchnorm/mul/ReadVariableOp2R
'batch_normalization_534/AssignMovingAvg'batch_normalization_534/AssignMovingAvg2p
6batch_normalization_534/AssignMovingAvg/ReadVariableOp6batch_normalization_534/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_534/AssignMovingAvg_1)batch_normalization_534/AssignMovingAvg_12t
8batch_normalization_534/AssignMovingAvg_1/ReadVariableOp8batch_normalization_534/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_534/batchnorm/ReadVariableOp0batch_normalization_534/batchnorm/ReadVariableOp2l
4batch_normalization_534/batchnorm/mul/ReadVariableOp4batch_normalization_534/batchnorm/mul/ReadVariableOp2R
'batch_normalization_535/AssignMovingAvg'batch_normalization_535/AssignMovingAvg2p
6batch_normalization_535/AssignMovingAvg/ReadVariableOp6batch_normalization_535/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_535/AssignMovingAvg_1)batch_normalization_535/AssignMovingAvg_12t
8batch_normalization_535/AssignMovingAvg_1/ReadVariableOp8batch_normalization_535/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_535/batchnorm/ReadVariableOp0batch_normalization_535/batchnorm/ReadVariableOp2l
4batch_normalization_535/batchnorm/mul/ReadVariableOp4batch_normalization_535/batchnorm/mul/ReadVariableOp2R
'batch_normalization_536/AssignMovingAvg'batch_normalization_536/AssignMovingAvg2p
6batch_normalization_536/AssignMovingAvg/ReadVariableOp6batch_normalization_536/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_536/AssignMovingAvg_1)batch_normalization_536/AssignMovingAvg_12t
8batch_normalization_536/AssignMovingAvg_1/ReadVariableOp8batch_normalization_536/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_536/batchnorm/ReadVariableOp0batch_normalization_536/batchnorm/ReadVariableOp2l
4batch_normalization_536/batchnorm/mul/ReadVariableOp4batch_normalization_536/batchnorm/mul/ReadVariableOp2R
'batch_normalization_537/AssignMovingAvg'batch_normalization_537/AssignMovingAvg2p
6batch_normalization_537/AssignMovingAvg/ReadVariableOp6batch_normalization_537/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_537/AssignMovingAvg_1)batch_normalization_537/AssignMovingAvg_12t
8batch_normalization_537/AssignMovingAvg_1/ReadVariableOp8batch_normalization_537/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_537/batchnorm/ReadVariableOp0batch_normalization_537/batchnorm/ReadVariableOp2l
4batch_normalization_537/batchnorm/mul/ReadVariableOp4batch_normalization_537/batchnorm/mul/ReadVariableOp2R
'batch_normalization_538/AssignMovingAvg'batch_normalization_538/AssignMovingAvg2p
6batch_normalization_538/AssignMovingAvg/ReadVariableOp6batch_normalization_538/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_538/AssignMovingAvg_1)batch_normalization_538/AssignMovingAvg_12t
8batch_normalization_538/AssignMovingAvg_1/ReadVariableOp8batch_normalization_538/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_538/batchnorm/ReadVariableOp0batch_normalization_538/batchnorm/ReadVariableOp2l
4batch_normalization_538/batchnorm/mul/ReadVariableOp4batch_normalization_538/batchnorm/mul/ReadVariableOp2R
'batch_normalization_539/AssignMovingAvg'batch_normalization_539/AssignMovingAvg2p
6batch_normalization_539/AssignMovingAvg/ReadVariableOp6batch_normalization_539/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_539/AssignMovingAvg_1)batch_normalization_539/AssignMovingAvg_12t
8batch_normalization_539/AssignMovingAvg_1/ReadVariableOp8batch_normalization_539/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_539/batchnorm/ReadVariableOp0batch_normalization_539/batchnorm/ReadVariableOp2l
4batch_normalization_539/batchnorm/mul/ReadVariableOp4batch_normalization_539/batchnorm/mul/ReadVariableOp2R
'batch_normalization_540/AssignMovingAvg'batch_normalization_540/AssignMovingAvg2p
6batch_normalization_540/AssignMovingAvg/ReadVariableOp6batch_normalization_540/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_540/AssignMovingAvg_1)batch_normalization_540/AssignMovingAvg_12t
8batch_normalization_540/AssignMovingAvg_1/ReadVariableOp8batch_normalization_540/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_540/batchnorm/ReadVariableOp0batch_normalization_540/batchnorm/ReadVariableOp2l
4batch_normalization_540/batchnorm/mul/ReadVariableOp4batch_normalization_540/batchnorm/mul/ReadVariableOp2D
 dense_588/BiasAdd/ReadVariableOp dense_588/BiasAdd/ReadVariableOp2B
dense_588/MatMul/ReadVariableOpdense_588/MatMul/ReadVariableOp2D
 dense_589/BiasAdd/ReadVariableOp dense_589/BiasAdd/ReadVariableOp2B
dense_589/MatMul/ReadVariableOpdense_589/MatMul/ReadVariableOp2D
 dense_590/BiasAdd/ReadVariableOp dense_590/BiasAdd/ReadVariableOp2B
dense_590/MatMul/ReadVariableOpdense_590/MatMul/ReadVariableOp2D
 dense_591/BiasAdd/ReadVariableOp dense_591/BiasAdd/ReadVariableOp2B
dense_591/MatMul/ReadVariableOpdense_591/MatMul/ReadVariableOp2D
 dense_592/BiasAdd/ReadVariableOp dense_592/BiasAdd/ReadVariableOp2B
dense_592/MatMul/ReadVariableOpdense_592/MatMul/ReadVariableOp2D
 dense_593/BiasAdd/ReadVariableOp dense_593/BiasAdd/ReadVariableOp2B
dense_593/MatMul/ReadVariableOpdense_593/MatMul/ReadVariableOp2D
 dense_594/BiasAdd/ReadVariableOp dense_594/BiasAdd/ReadVariableOp2B
dense_594/MatMul/ReadVariableOpdense_594/MatMul/ReadVariableOp2D
 dense_595/BiasAdd/ReadVariableOp dense_595/BiasAdd/ReadVariableOp2B
dense_595/MatMul/ReadVariableOpdense_595/MatMul/ReadVariableOp2D
 dense_596/BiasAdd/ReadVariableOp dense_596/BiasAdd/ReadVariableOp2B
dense_596/MatMul/ReadVariableOpdense_596/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ð
²
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_664215

inputs/
!batchnorm_readvariableop_resource:E3
%batchnorm_mul_readvariableop_resource:E1
#batchnorm_readvariableop_1_resource:E1
#batchnorm_readvariableop_2_resource:E
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:EP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:E~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ec
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:E*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Ez
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:E*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Er
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
È	
ö
E__inference_dense_594_layer_call_and_return_conditional_losses_664571

inputs0
matmul_readvariableop_resource:E-
biasadd_readvariableop_resource:E
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:E*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:E*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_595_layer_call_fn_667288

inputs
unknown:EE
	unknown_0:E
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_595_layer_call_and_return_conditional_losses_664603o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
³ 
«.
I__inference_sequential_55_layer_call_and_return_conditional_losses_666045

inputs
normalization_55_sub_y
normalization_55_sqrt_x:
(dense_588_matmul_readvariableop_resource:Y7
)dense_588_biasadd_readvariableop_resource:YG
9batch_normalization_533_batchnorm_readvariableop_resource:YK
=batch_normalization_533_batchnorm_mul_readvariableop_resource:YI
;batch_normalization_533_batchnorm_readvariableop_1_resource:YI
;batch_normalization_533_batchnorm_readvariableop_2_resource:Y:
(dense_589_matmul_readvariableop_resource:YY7
)dense_589_biasadd_readvariableop_resource:YG
9batch_normalization_534_batchnorm_readvariableop_resource:YK
=batch_normalization_534_batchnorm_mul_readvariableop_resource:YI
;batch_normalization_534_batchnorm_readvariableop_1_resource:YI
;batch_normalization_534_batchnorm_readvariableop_2_resource:Y:
(dense_590_matmul_readvariableop_resource:Y7
)dense_590_biasadd_readvariableop_resource:G
9batch_normalization_535_batchnorm_readvariableop_resource:K
=batch_normalization_535_batchnorm_mul_readvariableop_resource:I
;batch_normalization_535_batchnorm_readvariableop_1_resource:I
;batch_normalization_535_batchnorm_readvariableop_2_resource::
(dense_591_matmul_readvariableop_resource:7
)dense_591_biasadd_readvariableop_resource:G
9batch_normalization_536_batchnorm_readvariableop_resource:K
=batch_normalization_536_batchnorm_mul_readvariableop_resource:I
;batch_normalization_536_batchnorm_readvariableop_1_resource:I
;batch_normalization_536_batchnorm_readvariableop_2_resource::
(dense_592_matmul_readvariableop_resource:7
)dense_592_biasadd_readvariableop_resource:G
9batch_normalization_537_batchnorm_readvariableop_resource:K
=batch_normalization_537_batchnorm_mul_readvariableop_resource:I
;batch_normalization_537_batchnorm_readvariableop_1_resource:I
;batch_normalization_537_batchnorm_readvariableop_2_resource::
(dense_593_matmul_readvariableop_resource:7
)dense_593_biasadd_readvariableop_resource:G
9batch_normalization_538_batchnorm_readvariableop_resource:K
=batch_normalization_538_batchnorm_mul_readvariableop_resource:I
;batch_normalization_538_batchnorm_readvariableop_1_resource:I
;batch_normalization_538_batchnorm_readvariableop_2_resource::
(dense_594_matmul_readvariableop_resource:E7
)dense_594_biasadd_readvariableop_resource:EG
9batch_normalization_539_batchnorm_readvariableop_resource:EK
=batch_normalization_539_batchnorm_mul_readvariableop_resource:EI
;batch_normalization_539_batchnorm_readvariableop_1_resource:EI
;batch_normalization_539_batchnorm_readvariableop_2_resource:E:
(dense_595_matmul_readvariableop_resource:EE7
)dense_595_biasadd_readvariableop_resource:EG
9batch_normalization_540_batchnorm_readvariableop_resource:EK
=batch_normalization_540_batchnorm_mul_readvariableop_resource:EI
;batch_normalization_540_batchnorm_readvariableop_1_resource:EI
;batch_normalization_540_batchnorm_readvariableop_2_resource:E:
(dense_596_matmul_readvariableop_resource:E7
)dense_596_biasadd_readvariableop_resource:
identity¢0batch_normalization_533/batchnorm/ReadVariableOp¢2batch_normalization_533/batchnorm/ReadVariableOp_1¢2batch_normalization_533/batchnorm/ReadVariableOp_2¢4batch_normalization_533/batchnorm/mul/ReadVariableOp¢0batch_normalization_534/batchnorm/ReadVariableOp¢2batch_normalization_534/batchnorm/ReadVariableOp_1¢2batch_normalization_534/batchnorm/ReadVariableOp_2¢4batch_normalization_534/batchnorm/mul/ReadVariableOp¢0batch_normalization_535/batchnorm/ReadVariableOp¢2batch_normalization_535/batchnorm/ReadVariableOp_1¢2batch_normalization_535/batchnorm/ReadVariableOp_2¢4batch_normalization_535/batchnorm/mul/ReadVariableOp¢0batch_normalization_536/batchnorm/ReadVariableOp¢2batch_normalization_536/batchnorm/ReadVariableOp_1¢2batch_normalization_536/batchnorm/ReadVariableOp_2¢4batch_normalization_536/batchnorm/mul/ReadVariableOp¢0batch_normalization_537/batchnorm/ReadVariableOp¢2batch_normalization_537/batchnorm/ReadVariableOp_1¢2batch_normalization_537/batchnorm/ReadVariableOp_2¢4batch_normalization_537/batchnorm/mul/ReadVariableOp¢0batch_normalization_538/batchnorm/ReadVariableOp¢2batch_normalization_538/batchnorm/ReadVariableOp_1¢2batch_normalization_538/batchnorm/ReadVariableOp_2¢4batch_normalization_538/batchnorm/mul/ReadVariableOp¢0batch_normalization_539/batchnorm/ReadVariableOp¢2batch_normalization_539/batchnorm/ReadVariableOp_1¢2batch_normalization_539/batchnorm/ReadVariableOp_2¢4batch_normalization_539/batchnorm/mul/ReadVariableOp¢0batch_normalization_540/batchnorm/ReadVariableOp¢2batch_normalization_540/batchnorm/ReadVariableOp_1¢2batch_normalization_540/batchnorm/ReadVariableOp_2¢4batch_normalization_540/batchnorm/mul/ReadVariableOp¢ dense_588/BiasAdd/ReadVariableOp¢dense_588/MatMul/ReadVariableOp¢ dense_589/BiasAdd/ReadVariableOp¢dense_589/MatMul/ReadVariableOp¢ dense_590/BiasAdd/ReadVariableOp¢dense_590/MatMul/ReadVariableOp¢ dense_591/BiasAdd/ReadVariableOp¢dense_591/MatMul/ReadVariableOp¢ dense_592/BiasAdd/ReadVariableOp¢dense_592/MatMul/ReadVariableOp¢ dense_593/BiasAdd/ReadVariableOp¢dense_593/MatMul/ReadVariableOp¢ dense_594/BiasAdd/ReadVariableOp¢dense_594/MatMul/ReadVariableOp¢ dense_595/BiasAdd/ReadVariableOp¢dense_595/MatMul/ReadVariableOp¢ dense_596/BiasAdd/ReadVariableOp¢dense_596/MatMul/ReadVariableOpm
normalization_55/subSubinputsnormalization_55_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_55/SqrtSqrtnormalization_55_sqrt_x*
T0*
_output_shapes

:_
normalization_55/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_55/MaximumMaximumnormalization_55/Sqrt:y:0#normalization_55/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_55/truedivRealDivnormalization_55/sub:z:0normalization_55/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_588/MatMul/ReadVariableOpReadVariableOp(dense_588_matmul_readvariableop_resource*
_output_shapes

:Y*
dtype0
dense_588/MatMulMatMulnormalization_55/truediv:z:0'dense_588/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 dense_588/BiasAdd/ReadVariableOpReadVariableOp)dense_588_biasadd_readvariableop_resource*
_output_shapes
:Y*
dtype0
dense_588/BiasAddBiasAdddense_588/MatMul:product:0(dense_588/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY¦
0batch_normalization_533/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_533_batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0l
'batch_normalization_533/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_533/batchnorm/addAddV28batch_normalization_533/batchnorm/ReadVariableOp:value:00batch_normalization_533/batchnorm/add/y:output:0*
T0*
_output_shapes
:Y
'batch_normalization_533/batchnorm/RsqrtRsqrt)batch_normalization_533/batchnorm/add:z:0*
T0*
_output_shapes
:Y®
4batch_normalization_533/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_533_batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0¼
%batch_normalization_533/batchnorm/mulMul+batch_normalization_533/batchnorm/Rsqrt:y:0<batch_normalization_533/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Y§
'batch_normalization_533/batchnorm/mul_1Muldense_588/BiasAdd:output:0)batch_normalization_533/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYª
2batch_normalization_533/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_533_batchnorm_readvariableop_1_resource*
_output_shapes
:Y*
dtype0º
'batch_normalization_533/batchnorm/mul_2Mul:batch_normalization_533/batchnorm/ReadVariableOp_1:value:0)batch_normalization_533/batchnorm/mul:z:0*
T0*
_output_shapes
:Yª
2batch_normalization_533/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_533_batchnorm_readvariableop_2_resource*
_output_shapes
:Y*
dtype0º
%batch_normalization_533/batchnorm/subSub:batch_normalization_533/batchnorm/ReadVariableOp_2:value:0+batch_normalization_533/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yº
'batch_normalization_533/batchnorm/add_1AddV2+batch_normalization_533/batchnorm/mul_1:z:0)batch_normalization_533/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
leaky_re_lu_533/LeakyRelu	LeakyRelu+batch_normalization_533/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*
alpha%>
dense_589/MatMul/ReadVariableOpReadVariableOp(dense_589_matmul_readvariableop_resource*
_output_shapes

:YY*
dtype0
dense_589/MatMulMatMul'leaky_re_lu_533/LeakyRelu:activations:0'dense_589/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 dense_589/BiasAdd/ReadVariableOpReadVariableOp)dense_589_biasadd_readvariableop_resource*
_output_shapes
:Y*
dtype0
dense_589/BiasAddBiasAdddense_589/MatMul:product:0(dense_589/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY¦
0batch_normalization_534/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_534_batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0l
'batch_normalization_534/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_534/batchnorm/addAddV28batch_normalization_534/batchnorm/ReadVariableOp:value:00batch_normalization_534/batchnorm/add/y:output:0*
T0*
_output_shapes
:Y
'batch_normalization_534/batchnorm/RsqrtRsqrt)batch_normalization_534/batchnorm/add:z:0*
T0*
_output_shapes
:Y®
4batch_normalization_534/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_534_batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0¼
%batch_normalization_534/batchnorm/mulMul+batch_normalization_534/batchnorm/Rsqrt:y:0<batch_normalization_534/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Y§
'batch_normalization_534/batchnorm/mul_1Muldense_589/BiasAdd:output:0)batch_normalization_534/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYª
2batch_normalization_534/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_534_batchnorm_readvariableop_1_resource*
_output_shapes
:Y*
dtype0º
'batch_normalization_534/batchnorm/mul_2Mul:batch_normalization_534/batchnorm/ReadVariableOp_1:value:0)batch_normalization_534/batchnorm/mul:z:0*
T0*
_output_shapes
:Yª
2batch_normalization_534/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_534_batchnorm_readvariableop_2_resource*
_output_shapes
:Y*
dtype0º
%batch_normalization_534/batchnorm/subSub:batch_normalization_534/batchnorm/ReadVariableOp_2:value:0+batch_normalization_534/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yº
'batch_normalization_534/batchnorm/add_1AddV2+batch_normalization_534/batchnorm/mul_1:z:0)batch_normalization_534/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
leaky_re_lu_534/LeakyRelu	LeakyRelu+batch_normalization_534/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*
alpha%>
dense_590/MatMul/ReadVariableOpReadVariableOp(dense_590_matmul_readvariableop_resource*
_output_shapes

:Y*
dtype0
dense_590/MatMulMatMul'leaky_re_lu_534/LeakyRelu:activations:0'dense_590/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_590/BiasAdd/ReadVariableOpReadVariableOp)dense_590_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_590/BiasAddBiasAdddense_590/MatMul:product:0(dense_590/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_535/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_535_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_535/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_535/batchnorm/addAddV28batch_normalization_535/batchnorm/ReadVariableOp:value:00batch_normalization_535/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_535/batchnorm/RsqrtRsqrt)batch_normalization_535/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_535/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_535_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_535/batchnorm/mulMul+batch_normalization_535/batchnorm/Rsqrt:y:0<batch_normalization_535/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_535/batchnorm/mul_1Muldense_590/BiasAdd:output:0)batch_normalization_535/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_535/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_535_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_535/batchnorm/mul_2Mul:batch_normalization_535/batchnorm/ReadVariableOp_1:value:0)batch_normalization_535/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_535/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_535_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_535/batchnorm/subSub:batch_normalization_535/batchnorm/ReadVariableOp_2:value:0+batch_normalization_535/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_535/batchnorm/add_1AddV2+batch_normalization_535/batchnorm/mul_1:z:0)batch_normalization_535/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_535/LeakyRelu	LeakyRelu+batch_normalization_535/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_591/MatMul/ReadVariableOpReadVariableOp(dense_591_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_591/MatMulMatMul'leaky_re_lu_535/LeakyRelu:activations:0'dense_591/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_591/BiasAdd/ReadVariableOpReadVariableOp)dense_591_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_591/BiasAddBiasAdddense_591/MatMul:product:0(dense_591/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_536/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_536_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_536/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_536/batchnorm/addAddV28batch_normalization_536/batchnorm/ReadVariableOp:value:00batch_normalization_536/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_536/batchnorm/RsqrtRsqrt)batch_normalization_536/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_536/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_536_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_536/batchnorm/mulMul+batch_normalization_536/batchnorm/Rsqrt:y:0<batch_normalization_536/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_536/batchnorm/mul_1Muldense_591/BiasAdd:output:0)batch_normalization_536/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_536/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_536_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_536/batchnorm/mul_2Mul:batch_normalization_536/batchnorm/ReadVariableOp_1:value:0)batch_normalization_536/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_536/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_536_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_536/batchnorm/subSub:batch_normalization_536/batchnorm/ReadVariableOp_2:value:0+batch_normalization_536/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_536/batchnorm/add_1AddV2+batch_normalization_536/batchnorm/mul_1:z:0)batch_normalization_536/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_536/LeakyRelu	LeakyRelu+batch_normalization_536/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_592/MatMul/ReadVariableOpReadVariableOp(dense_592_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_592/MatMulMatMul'leaky_re_lu_536/LeakyRelu:activations:0'dense_592/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_592/BiasAdd/ReadVariableOpReadVariableOp)dense_592_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_592/BiasAddBiasAdddense_592/MatMul:product:0(dense_592/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_537/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_537_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_537/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_537/batchnorm/addAddV28batch_normalization_537/batchnorm/ReadVariableOp:value:00batch_normalization_537/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_537/batchnorm/RsqrtRsqrt)batch_normalization_537/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_537/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_537_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_537/batchnorm/mulMul+batch_normalization_537/batchnorm/Rsqrt:y:0<batch_normalization_537/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_537/batchnorm/mul_1Muldense_592/BiasAdd:output:0)batch_normalization_537/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_537/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_537_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_537/batchnorm/mul_2Mul:batch_normalization_537/batchnorm/ReadVariableOp_1:value:0)batch_normalization_537/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_537/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_537_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_537/batchnorm/subSub:batch_normalization_537/batchnorm/ReadVariableOp_2:value:0+batch_normalization_537/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_537/batchnorm/add_1AddV2+batch_normalization_537/batchnorm/mul_1:z:0)batch_normalization_537/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_537/LeakyRelu	LeakyRelu+batch_normalization_537/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_593/MatMul/ReadVariableOpReadVariableOp(dense_593_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_593/MatMulMatMul'leaky_re_lu_537/LeakyRelu:activations:0'dense_593/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_593/BiasAdd/ReadVariableOpReadVariableOp)dense_593_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_593/BiasAddBiasAdddense_593/MatMul:product:0(dense_593/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_538/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_538_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_538/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_538/batchnorm/addAddV28batch_normalization_538/batchnorm/ReadVariableOp:value:00batch_normalization_538/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_538/batchnorm/RsqrtRsqrt)batch_normalization_538/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_538/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_538_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_538/batchnorm/mulMul+batch_normalization_538/batchnorm/Rsqrt:y:0<batch_normalization_538/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_538/batchnorm/mul_1Muldense_593/BiasAdd:output:0)batch_normalization_538/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_538/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_538_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_538/batchnorm/mul_2Mul:batch_normalization_538/batchnorm/ReadVariableOp_1:value:0)batch_normalization_538/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_538/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_538_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_538/batchnorm/subSub:batch_normalization_538/batchnorm/ReadVariableOp_2:value:0+batch_normalization_538/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_538/batchnorm/add_1AddV2+batch_normalization_538/batchnorm/mul_1:z:0)batch_normalization_538/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_538/LeakyRelu	LeakyRelu+batch_normalization_538/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_594/MatMul/ReadVariableOpReadVariableOp(dense_594_matmul_readvariableop_resource*
_output_shapes

:E*
dtype0
dense_594/MatMulMatMul'leaky_re_lu_538/LeakyRelu:activations:0'dense_594/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 dense_594/BiasAdd/ReadVariableOpReadVariableOp)dense_594_biasadd_readvariableop_resource*
_output_shapes
:E*
dtype0
dense_594/BiasAddBiasAdddense_594/MatMul:product:0(dense_594/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE¦
0batch_normalization_539/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_539_batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0l
'batch_normalization_539/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_539/batchnorm/addAddV28batch_normalization_539/batchnorm/ReadVariableOp:value:00batch_normalization_539/batchnorm/add/y:output:0*
T0*
_output_shapes
:E
'batch_normalization_539/batchnorm/RsqrtRsqrt)batch_normalization_539/batchnorm/add:z:0*
T0*
_output_shapes
:E®
4batch_normalization_539/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_539_batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0¼
%batch_normalization_539/batchnorm/mulMul+batch_normalization_539/batchnorm/Rsqrt:y:0<batch_normalization_539/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:E§
'batch_normalization_539/batchnorm/mul_1Muldense_594/BiasAdd:output:0)batch_normalization_539/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEª
2batch_normalization_539/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_539_batchnorm_readvariableop_1_resource*
_output_shapes
:E*
dtype0º
'batch_normalization_539/batchnorm/mul_2Mul:batch_normalization_539/batchnorm/ReadVariableOp_1:value:0)batch_normalization_539/batchnorm/mul:z:0*
T0*
_output_shapes
:Eª
2batch_normalization_539/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_539_batchnorm_readvariableop_2_resource*
_output_shapes
:E*
dtype0º
%batch_normalization_539/batchnorm/subSub:batch_normalization_539/batchnorm/ReadVariableOp_2:value:0+batch_normalization_539/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Eº
'batch_normalization_539/batchnorm/add_1AddV2+batch_normalization_539/batchnorm/mul_1:z:0)batch_normalization_539/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
leaky_re_lu_539/LeakyRelu	LeakyRelu+batch_normalization_539/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*
alpha%>
dense_595/MatMul/ReadVariableOpReadVariableOp(dense_595_matmul_readvariableop_resource*
_output_shapes

:EE*
dtype0
dense_595/MatMulMatMul'leaky_re_lu_539/LeakyRelu:activations:0'dense_595/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 dense_595/BiasAdd/ReadVariableOpReadVariableOp)dense_595_biasadd_readvariableop_resource*
_output_shapes
:E*
dtype0
dense_595/BiasAddBiasAdddense_595/MatMul:product:0(dense_595/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE¦
0batch_normalization_540/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_540_batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0l
'batch_normalization_540/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_540/batchnorm/addAddV28batch_normalization_540/batchnorm/ReadVariableOp:value:00batch_normalization_540/batchnorm/add/y:output:0*
T0*
_output_shapes
:E
'batch_normalization_540/batchnorm/RsqrtRsqrt)batch_normalization_540/batchnorm/add:z:0*
T0*
_output_shapes
:E®
4batch_normalization_540/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_540_batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0¼
%batch_normalization_540/batchnorm/mulMul+batch_normalization_540/batchnorm/Rsqrt:y:0<batch_normalization_540/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:E§
'batch_normalization_540/batchnorm/mul_1Muldense_595/BiasAdd:output:0)batch_normalization_540/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEª
2batch_normalization_540/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_540_batchnorm_readvariableop_1_resource*
_output_shapes
:E*
dtype0º
'batch_normalization_540/batchnorm/mul_2Mul:batch_normalization_540/batchnorm/ReadVariableOp_1:value:0)batch_normalization_540/batchnorm/mul:z:0*
T0*
_output_shapes
:Eª
2batch_normalization_540/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_540_batchnorm_readvariableop_2_resource*
_output_shapes
:E*
dtype0º
%batch_normalization_540/batchnorm/subSub:batch_normalization_540/batchnorm/ReadVariableOp_2:value:0+batch_normalization_540/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Eº
'batch_normalization_540/batchnorm/add_1AddV2+batch_normalization_540/batchnorm/mul_1:z:0)batch_normalization_540/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
leaky_re_lu_540/LeakyRelu	LeakyRelu+batch_normalization_540/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*
alpha%>
dense_596/MatMul/ReadVariableOpReadVariableOp(dense_596_matmul_readvariableop_resource*
_output_shapes

:E*
dtype0
dense_596/MatMulMatMul'leaky_re_lu_540/LeakyRelu:activations:0'dense_596/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_596/BiasAdd/ReadVariableOpReadVariableOp)dense_596_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_596/BiasAddBiasAdddense_596/MatMul:product:0(dense_596/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_596/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp1^batch_normalization_533/batchnorm/ReadVariableOp3^batch_normalization_533/batchnorm/ReadVariableOp_13^batch_normalization_533/batchnorm/ReadVariableOp_25^batch_normalization_533/batchnorm/mul/ReadVariableOp1^batch_normalization_534/batchnorm/ReadVariableOp3^batch_normalization_534/batchnorm/ReadVariableOp_13^batch_normalization_534/batchnorm/ReadVariableOp_25^batch_normalization_534/batchnorm/mul/ReadVariableOp1^batch_normalization_535/batchnorm/ReadVariableOp3^batch_normalization_535/batchnorm/ReadVariableOp_13^batch_normalization_535/batchnorm/ReadVariableOp_25^batch_normalization_535/batchnorm/mul/ReadVariableOp1^batch_normalization_536/batchnorm/ReadVariableOp3^batch_normalization_536/batchnorm/ReadVariableOp_13^batch_normalization_536/batchnorm/ReadVariableOp_25^batch_normalization_536/batchnorm/mul/ReadVariableOp1^batch_normalization_537/batchnorm/ReadVariableOp3^batch_normalization_537/batchnorm/ReadVariableOp_13^batch_normalization_537/batchnorm/ReadVariableOp_25^batch_normalization_537/batchnorm/mul/ReadVariableOp1^batch_normalization_538/batchnorm/ReadVariableOp3^batch_normalization_538/batchnorm/ReadVariableOp_13^batch_normalization_538/batchnorm/ReadVariableOp_25^batch_normalization_538/batchnorm/mul/ReadVariableOp1^batch_normalization_539/batchnorm/ReadVariableOp3^batch_normalization_539/batchnorm/ReadVariableOp_13^batch_normalization_539/batchnorm/ReadVariableOp_25^batch_normalization_539/batchnorm/mul/ReadVariableOp1^batch_normalization_540/batchnorm/ReadVariableOp3^batch_normalization_540/batchnorm/ReadVariableOp_13^batch_normalization_540/batchnorm/ReadVariableOp_25^batch_normalization_540/batchnorm/mul/ReadVariableOp!^dense_588/BiasAdd/ReadVariableOp ^dense_588/MatMul/ReadVariableOp!^dense_589/BiasAdd/ReadVariableOp ^dense_589/MatMul/ReadVariableOp!^dense_590/BiasAdd/ReadVariableOp ^dense_590/MatMul/ReadVariableOp!^dense_591/BiasAdd/ReadVariableOp ^dense_591/MatMul/ReadVariableOp!^dense_592/BiasAdd/ReadVariableOp ^dense_592/MatMul/ReadVariableOp!^dense_593/BiasAdd/ReadVariableOp ^dense_593/MatMul/ReadVariableOp!^dense_594/BiasAdd/ReadVariableOp ^dense_594/MatMul/ReadVariableOp!^dense_595/BiasAdd/ReadVariableOp ^dense_595/MatMul/ReadVariableOp!^dense_596/BiasAdd/ReadVariableOp ^dense_596/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_533/batchnorm/ReadVariableOp0batch_normalization_533/batchnorm/ReadVariableOp2h
2batch_normalization_533/batchnorm/ReadVariableOp_12batch_normalization_533/batchnorm/ReadVariableOp_12h
2batch_normalization_533/batchnorm/ReadVariableOp_22batch_normalization_533/batchnorm/ReadVariableOp_22l
4batch_normalization_533/batchnorm/mul/ReadVariableOp4batch_normalization_533/batchnorm/mul/ReadVariableOp2d
0batch_normalization_534/batchnorm/ReadVariableOp0batch_normalization_534/batchnorm/ReadVariableOp2h
2batch_normalization_534/batchnorm/ReadVariableOp_12batch_normalization_534/batchnorm/ReadVariableOp_12h
2batch_normalization_534/batchnorm/ReadVariableOp_22batch_normalization_534/batchnorm/ReadVariableOp_22l
4batch_normalization_534/batchnorm/mul/ReadVariableOp4batch_normalization_534/batchnorm/mul/ReadVariableOp2d
0batch_normalization_535/batchnorm/ReadVariableOp0batch_normalization_535/batchnorm/ReadVariableOp2h
2batch_normalization_535/batchnorm/ReadVariableOp_12batch_normalization_535/batchnorm/ReadVariableOp_12h
2batch_normalization_535/batchnorm/ReadVariableOp_22batch_normalization_535/batchnorm/ReadVariableOp_22l
4batch_normalization_535/batchnorm/mul/ReadVariableOp4batch_normalization_535/batchnorm/mul/ReadVariableOp2d
0batch_normalization_536/batchnorm/ReadVariableOp0batch_normalization_536/batchnorm/ReadVariableOp2h
2batch_normalization_536/batchnorm/ReadVariableOp_12batch_normalization_536/batchnorm/ReadVariableOp_12h
2batch_normalization_536/batchnorm/ReadVariableOp_22batch_normalization_536/batchnorm/ReadVariableOp_22l
4batch_normalization_536/batchnorm/mul/ReadVariableOp4batch_normalization_536/batchnorm/mul/ReadVariableOp2d
0batch_normalization_537/batchnorm/ReadVariableOp0batch_normalization_537/batchnorm/ReadVariableOp2h
2batch_normalization_537/batchnorm/ReadVariableOp_12batch_normalization_537/batchnorm/ReadVariableOp_12h
2batch_normalization_537/batchnorm/ReadVariableOp_22batch_normalization_537/batchnorm/ReadVariableOp_22l
4batch_normalization_537/batchnorm/mul/ReadVariableOp4batch_normalization_537/batchnorm/mul/ReadVariableOp2d
0batch_normalization_538/batchnorm/ReadVariableOp0batch_normalization_538/batchnorm/ReadVariableOp2h
2batch_normalization_538/batchnorm/ReadVariableOp_12batch_normalization_538/batchnorm/ReadVariableOp_12h
2batch_normalization_538/batchnorm/ReadVariableOp_22batch_normalization_538/batchnorm/ReadVariableOp_22l
4batch_normalization_538/batchnorm/mul/ReadVariableOp4batch_normalization_538/batchnorm/mul/ReadVariableOp2d
0batch_normalization_539/batchnorm/ReadVariableOp0batch_normalization_539/batchnorm/ReadVariableOp2h
2batch_normalization_539/batchnorm/ReadVariableOp_12batch_normalization_539/batchnorm/ReadVariableOp_12h
2batch_normalization_539/batchnorm/ReadVariableOp_22batch_normalization_539/batchnorm/ReadVariableOp_22l
4batch_normalization_539/batchnorm/mul/ReadVariableOp4batch_normalization_539/batchnorm/mul/ReadVariableOp2d
0batch_normalization_540/batchnorm/ReadVariableOp0batch_normalization_540/batchnorm/ReadVariableOp2h
2batch_normalization_540/batchnorm/ReadVariableOp_12batch_normalization_540/batchnorm/ReadVariableOp_12h
2batch_normalization_540/batchnorm/ReadVariableOp_22batch_normalization_540/batchnorm/ReadVariableOp_22l
4batch_normalization_540/batchnorm/mul/ReadVariableOp4batch_normalization_540/batchnorm/mul/ReadVariableOp2D
 dense_588/BiasAdd/ReadVariableOp dense_588/BiasAdd/ReadVariableOp2B
dense_588/MatMul/ReadVariableOpdense_588/MatMul/ReadVariableOp2D
 dense_589/BiasAdd/ReadVariableOp dense_589/BiasAdd/ReadVariableOp2B
dense_589/MatMul/ReadVariableOpdense_589/MatMul/ReadVariableOp2D
 dense_590/BiasAdd/ReadVariableOp dense_590/BiasAdd/ReadVariableOp2B
dense_590/MatMul/ReadVariableOpdense_590/MatMul/ReadVariableOp2D
 dense_591/BiasAdd/ReadVariableOp dense_591/BiasAdd/ReadVariableOp2B
dense_591/MatMul/ReadVariableOpdense_591/MatMul/ReadVariableOp2D
 dense_592/BiasAdd/ReadVariableOp dense_592/BiasAdd/ReadVariableOp2B
dense_592/MatMul/ReadVariableOpdense_592/MatMul/ReadVariableOp2D
 dense_593/BiasAdd/ReadVariableOp dense_593/BiasAdd/ReadVariableOp2B
dense_593/MatMul/ReadVariableOpdense_593/MatMul/ReadVariableOp2D
 dense_594/BiasAdd/ReadVariableOp dense_594/BiasAdd/ReadVariableOp2B
dense_594/MatMul/ReadVariableOpdense_594/MatMul/ReadVariableOp2D
 dense_595/BiasAdd/ReadVariableOp dense_595/BiasAdd/ReadVariableOp2B
dense_595/MatMul/ReadVariableOpdense_595/MatMul/ReadVariableOp2D
 dense_596/BiasAdd/ReadVariableOp dense_596/BiasAdd/ReadVariableOp2B
dense_596/MatMul/ReadVariableOpdense_596/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ð
²
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_666581

inputs/
!batchnorm_readvariableop_resource:Y3
%batchnorm_mul_readvariableop_resource:Y1
#batchnorm_readvariableop_1_resource:Y1
#batchnorm_readvariableop_2_resource:Y
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:YP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Y~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Yc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:Y*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Yz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:Y*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_663887

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û'
Ò
__inference_adapt_step_666516
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢IteratorGetNext¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢add/ReadVariableOp
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
valueB: 
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*
_output_shapes

: l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
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
value	B : 
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
 *  ?H
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
:
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0
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
Ä

*__inference_dense_594_layer_call_fn_667179

inputs
unknown:E
	unknown_0:E
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_594_layer_call_and_return_conditional_losses_664571o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_533_layer_call_fn_666561

inputs
unknown:Y
	unknown_0:Y
	unknown_1:Y
	unknown_2:Y
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_663770o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_539_layer_call_fn_667274

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_539_layer_call_and_return_conditional_losses_664591`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿE:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
Ä

*__inference_dense_596_layer_call_fn_667397

inputs
unknown:E
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_596_layer_call_and_return_conditional_losses_664635o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
º
©
I__inference_sequential_55_layer_call_and_return_conditional_losses_665622
normalization_55_input
normalization_55_sub_y
normalization_55_sqrt_x"
dense_588_665496:Y
dense_588_665498:Y,
batch_normalization_533_665501:Y,
batch_normalization_533_665503:Y,
batch_normalization_533_665505:Y,
batch_normalization_533_665507:Y"
dense_589_665511:YY
dense_589_665513:Y,
batch_normalization_534_665516:Y,
batch_normalization_534_665518:Y,
batch_normalization_534_665520:Y,
batch_normalization_534_665522:Y"
dense_590_665526:Y
dense_590_665528:,
batch_normalization_535_665531:,
batch_normalization_535_665533:,
batch_normalization_535_665535:,
batch_normalization_535_665537:"
dense_591_665541:
dense_591_665543:,
batch_normalization_536_665546:,
batch_normalization_536_665548:,
batch_normalization_536_665550:,
batch_normalization_536_665552:"
dense_592_665556:
dense_592_665558:,
batch_normalization_537_665561:,
batch_normalization_537_665563:,
batch_normalization_537_665565:,
batch_normalization_537_665567:"
dense_593_665571:
dense_593_665573:,
batch_normalization_538_665576:,
batch_normalization_538_665578:,
batch_normalization_538_665580:,
batch_normalization_538_665582:"
dense_594_665586:E
dense_594_665588:E,
batch_normalization_539_665591:E,
batch_normalization_539_665593:E,
batch_normalization_539_665595:E,
batch_normalization_539_665597:E"
dense_595_665601:EE
dense_595_665603:E,
batch_normalization_540_665606:E,
batch_normalization_540_665608:E,
batch_normalization_540_665610:E,
batch_normalization_540_665612:E"
dense_596_665616:E
dense_596_665618:
identity¢/batch_normalization_533/StatefulPartitionedCall¢/batch_normalization_534/StatefulPartitionedCall¢/batch_normalization_535/StatefulPartitionedCall¢/batch_normalization_536/StatefulPartitionedCall¢/batch_normalization_537/StatefulPartitionedCall¢/batch_normalization_538/StatefulPartitionedCall¢/batch_normalization_539/StatefulPartitionedCall¢/batch_normalization_540/StatefulPartitionedCall¢!dense_588/StatefulPartitionedCall¢!dense_589/StatefulPartitionedCall¢!dense_590/StatefulPartitionedCall¢!dense_591/StatefulPartitionedCall¢!dense_592/StatefulPartitionedCall¢!dense_593/StatefulPartitionedCall¢!dense_594/StatefulPartitionedCall¢!dense_595/StatefulPartitionedCall¢!dense_596/StatefulPartitionedCall}
normalization_55/subSubnormalization_55_inputnormalization_55_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_55/SqrtSqrtnormalization_55_sqrt_x*
T0*
_output_shapes

:_
normalization_55/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_55/MaximumMaximumnormalization_55/Sqrt:y:0#normalization_55/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_55/truedivRealDivnormalization_55/sub:z:0normalization_55/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_588/StatefulPartitionedCallStatefulPartitionedCallnormalization_55/truediv:z:0dense_588_665496dense_588_665498*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_588_layer_call_and_return_conditional_losses_664379
/batch_normalization_533/StatefulPartitionedCallStatefulPartitionedCall*dense_588/StatefulPartitionedCall:output:0batch_normalization_533_665501batch_normalization_533_665503batch_normalization_533_665505batch_normalization_533_665507*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_663770ø
leaky_re_lu_533/PartitionedCallPartitionedCall8batch_normalization_533/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_533_layer_call_and_return_conditional_losses_664399
!dense_589/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_533/PartitionedCall:output:0dense_589_665511dense_589_665513*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_589_layer_call_and_return_conditional_losses_664411
/batch_normalization_534/StatefulPartitionedCallStatefulPartitionedCall*dense_589/StatefulPartitionedCall:output:0batch_normalization_534_665516batch_normalization_534_665518batch_normalization_534_665520batch_normalization_534_665522*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_663852ø
leaky_re_lu_534/PartitionedCallPartitionedCall8batch_normalization_534/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_534_layer_call_and_return_conditional_losses_664431
!dense_590/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_534/PartitionedCall:output:0dense_590_665526dense_590_665528*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_590_layer_call_and_return_conditional_losses_664443
/batch_normalization_535/StatefulPartitionedCallStatefulPartitionedCall*dense_590/StatefulPartitionedCall:output:0batch_normalization_535_665531batch_normalization_535_665533batch_normalization_535_665535batch_normalization_535_665537*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_663934ø
leaky_re_lu_535/PartitionedCallPartitionedCall8batch_normalization_535/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_535_layer_call_and_return_conditional_losses_664463
!dense_591/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_535/PartitionedCall:output:0dense_591_665541dense_591_665543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_591_layer_call_and_return_conditional_losses_664475
/batch_normalization_536/StatefulPartitionedCallStatefulPartitionedCall*dense_591/StatefulPartitionedCall:output:0batch_normalization_536_665546batch_normalization_536_665548batch_normalization_536_665550batch_normalization_536_665552*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_664016ø
leaky_re_lu_536/PartitionedCallPartitionedCall8batch_normalization_536/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_536_layer_call_and_return_conditional_losses_664495
!dense_592/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_536/PartitionedCall:output:0dense_592_665556dense_592_665558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_592_layer_call_and_return_conditional_losses_664507
/batch_normalization_537/StatefulPartitionedCallStatefulPartitionedCall*dense_592/StatefulPartitionedCall:output:0batch_normalization_537_665561batch_normalization_537_665563batch_normalization_537_665565batch_normalization_537_665567*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_664098ø
leaky_re_lu_537/PartitionedCallPartitionedCall8batch_normalization_537/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_537_layer_call_and_return_conditional_losses_664527
!dense_593/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_537/PartitionedCall:output:0dense_593_665571dense_593_665573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_593_layer_call_and_return_conditional_losses_664539
/batch_normalization_538/StatefulPartitionedCallStatefulPartitionedCall*dense_593/StatefulPartitionedCall:output:0batch_normalization_538_665576batch_normalization_538_665578batch_normalization_538_665580batch_normalization_538_665582*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_664180ø
leaky_re_lu_538/PartitionedCallPartitionedCall8batch_normalization_538/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_538_layer_call_and_return_conditional_losses_664559
!dense_594/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_538/PartitionedCall:output:0dense_594_665586dense_594_665588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_594_layer_call_and_return_conditional_losses_664571
/batch_normalization_539/StatefulPartitionedCallStatefulPartitionedCall*dense_594/StatefulPartitionedCall:output:0batch_normalization_539_665591batch_normalization_539_665593batch_normalization_539_665595batch_normalization_539_665597*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_664262ø
leaky_re_lu_539/PartitionedCallPartitionedCall8batch_normalization_539/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_539_layer_call_and_return_conditional_losses_664591
!dense_595/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_539/PartitionedCall:output:0dense_595_665601dense_595_665603*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_595_layer_call_and_return_conditional_losses_664603
/batch_normalization_540/StatefulPartitionedCallStatefulPartitionedCall*dense_595/StatefulPartitionedCall:output:0batch_normalization_540_665606batch_normalization_540_665608batch_normalization_540_665610batch_normalization_540_665612*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_664344ø
leaky_re_lu_540/PartitionedCallPartitionedCall8batch_normalization_540/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_540_layer_call_and_return_conditional_losses_664623
!dense_596/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_540/PartitionedCall:output:0dense_596_665616dense_596_665618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_596_layer_call_and_return_conditional_losses_664635y
IdentityIdentity*dense_596/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_533/StatefulPartitionedCall0^batch_normalization_534/StatefulPartitionedCall0^batch_normalization_535/StatefulPartitionedCall0^batch_normalization_536/StatefulPartitionedCall0^batch_normalization_537/StatefulPartitionedCall0^batch_normalization_538/StatefulPartitionedCall0^batch_normalization_539/StatefulPartitionedCall0^batch_normalization_540/StatefulPartitionedCall"^dense_588/StatefulPartitionedCall"^dense_589/StatefulPartitionedCall"^dense_590/StatefulPartitionedCall"^dense_591/StatefulPartitionedCall"^dense_592/StatefulPartitionedCall"^dense_593/StatefulPartitionedCall"^dense_594/StatefulPartitionedCall"^dense_595/StatefulPartitionedCall"^dense_596/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_533/StatefulPartitionedCall/batch_normalization_533/StatefulPartitionedCall2b
/batch_normalization_534/StatefulPartitionedCall/batch_normalization_534/StatefulPartitionedCall2b
/batch_normalization_535/StatefulPartitionedCall/batch_normalization_535/StatefulPartitionedCall2b
/batch_normalization_536/StatefulPartitionedCall/batch_normalization_536/StatefulPartitionedCall2b
/batch_normalization_537/StatefulPartitionedCall/batch_normalization_537/StatefulPartitionedCall2b
/batch_normalization_538/StatefulPartitionedCall/batch_normalization_538/StatefulPartitionedCall2b
/batch_normalization_539/StatefulPartitionedCall/batch_normalization_539/StatefulPartitionedCall2b
/batch_normalization_540/StatefulPartitionedCall/batch_normalization_540/StatefulPartitionedCall2F
!dense_588/StatefulPartitionedCall!dense_588/StatefulPartitionedCall2F
!dense_589/StatefulPartitionedCall!dense_589/StatefulPartitionedCall2F
!dense_590/StatefulPartitionedCall!dense_590/StatefulPartitionedCall2F
!dense_591/StatefulPartitionedCall!dense_591/StatefulPartitionedCall2F
!dense_592/StatefulPartitionedCall!dense_592/StatefulPartitionedCall2F
!dense_593/StatefulPartitionedCall!dense_593/StatefulPartitionedCall2F
!dense_594/StatefulPartitionedCall!dense_594/StatefulPartitionedCall2F
!dense_595/StatefulPartitionedCall!dense_595/StatefulPartitionedCall2F
!dense_596/StatefulPartitionedCall!dense_596/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_55_input:$ 

_output_shapes

::$ 

_output_shapes

:
ª
Ó
8__inference_batch_normalization_538_layer_call_fn_667106

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_664180o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_sequential_55_layer_call_and_return_conditional_losses_665134

inputs
normalization_55_sub_y
normalization_55_sqrt_x"
dense_588_665008:Y
dense_588_665010:Y,
batch_normalization_533_665013:Y,
batch_normalization_533_665015:Y,
batch_normalization_533_665017:Y,
batch_normalization_533_665019:Y"
dense_589_665023:YY
dense_589_665025:Y,
batch_normalization_534_665028:Y,
batch_normalization_534_665030:Y,
batch_normalization_534_665032:Y,
batch_normalization_534_665034:Y"
dense_590_665038:Y
dense_590_665040:,
batch_normalization_535_665043:,
batch_normalization_535_665045:,
batch_normalization_535_665047:,
batch_normalization_535_665049:"
dense_591_665053:
dense_591_665055:,
batch_normalization_536_665058:,
batch_normalization_536_665060:,
batch_normalization_536_665062:,
batch_normalization_536_665064:"
dense_592_665068:
dense_592_665070:,
batch_normalization_537_665073:,
batch_normalization_537_665075:,
batch_normalization_537_665077:,
batch_normalization_537_665079:"
dense_593_665083:
dense_593_665085:,
batch_normalization_538_665088:,
batch_normalization_538_665090:,
batch_normalization_538_665092:,
batch_normalization_538_665094:"
dense_594_665098:E
dense_594_665100:E,
batch_normalization_539_665103:E,
batch_normalization_539_665105:E,
batch_normalization_539_665107:E,
batch_normalization_539_665109:E"
dense_595_665113:EE
dense_595_665115:E,
batch_normalization_540_665118:E,
batch_normalization_540_665120:E,
batch_normalization_540_665122:E,
batch_normalization_540_665124:E"
dense_596_665128:E
dense_596_665130:
identity¢/batch_normalization_533/StatefulPartitionedCall¢/batch_normalization_534/StatefulPartitionedCall¢/batch_normalization_535/StatefulPartitionedCall¢/batch_normalization_536/StatefulPartitionedCall¢/batch_normalization_537/StatefulPartitionedCall¢/batch_normalization_538/StatefulPartitionedCall¢/batch_normalization_539/StatefulPartitionedCall¢/batch_normalization_540/StatefulPartitionedCall¢!dense_588/StatefulPartitionedCall¢!dense_589/StatefulPartitionedCall¢!dense_590/StatefulPartitionedCall¢!dense_591/StatefulPartitionedCall¢!dense_592/StatefulPartitionedCall¢!dense_593/StatefulPartitionedCall¢!dense_594/StatefulPartitionedCall¢!dense_595/StatefulPartitionedCall¢!dense_596/StatefulPartitionedCallm
normalization_55/subSubinputsnormalization_55_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_55/SqrtSqrtnormalization_55_sqrt_x*
T0*
_output_shapes

:_
normalization_55/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_55/MaximumMaximumnormalization_55/Sqrt:y:0#normalization_55/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_55/truedivRealDivnormalization_55/sub:z:0normalization_55/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_588/StatefulPartitionedCallStatefulPartitionedCallnormalization_55/truediv:z:0dense_588_665008dense_588_665010*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_588_layer_call_and_return_conditional_losses_664379
/batch_normalization_533/StatefulPartitionedCallStatefulPartitionedCall*dense_588/StatefulPartitionedCall:output:0batch_normalization_533_665013batch_normalization_533_665015batch_normalization_533_665017batch_normalization_533_665019*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_663770ø
leaky_re_lu_533/PartitionedCallPartitionedCall8batch_normalization_533/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_533_layer_call_and_return_conditional_losses_664399
!dense_589/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_533/PartitionedCall:output:0dense_589_665023dense_589_665025*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_589_layer_call_and_return_conditional_losses_664411
/batch_normalization_534/StatefulPartitionedCallStatefulPartitionedCall*dense_589/StatefulPartitionedCall:output:0batch_normalization_534_665028batch_normalization_534_665030batch_normalization_534_665032batch_normalization_534_665034*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_663852ø
leaky_re_lu_534/PartitionedCallPartitionedCall8batch_normalization_534/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_534_layer_call_and_return_conditional_losses_664431
!dense_590/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_534/PartitionedCall:output:0dense_590_665038dense_590_665040*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_590_layer_call_and_return_conditional_losses_664443
/batch_normalization_535/StatefulPartitionedCallStatefulPartitionedCall*dense_590/StatefulPartitionedCall:output:0batch_normalization_535_665043batch_normalization_535_665045batch_normalization_535_665047batch_normalization_535_665049*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_663934ø
leaky_re_lu_535/PartitionedCallPartitionedCall8batch_normalization_535/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_535_layer_call_and_return_conditional_losses_664463
!dense_591/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_535/PartitionedCall:output:0dense_591_665053dense_591_665055*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_591_layer_call_and_return_conditional_losses_664475
/batch_normalization_536/StatefulPartitionedCallStatefulPartitionedCall*dense_591/StatefulPartitionedCall:output:0batch_normalization_536_665058batch_normalization_536_665060batch_normalization_536_665062batch_normalization_536_665064*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_664016ø
leaky_re_lu_536/PartitionedCallPartitionedCall8batch_normalization_536/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_536_layer_call_and_return_conditional_losses_664495
!dense_592/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_536/PartitionedCall:output:0dense_592_665068dense_592_665070*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_592_layer_call_and_return_conditional_losses_664507
/batch_normalization_537/StatefulPartitionedCallStatefulPartitionedCall*dense_592/StatefulPartitionedCall:output:0batch_normalization_537_665073batch_normalization_537_665075batch_normalization_537_665077batch_normalization_537_665079*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_664098ø
leaky_re_lu_537/PartitionedCallPartitionedCall8batch_normalization_537/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_537_layer_call_and_return_conditional_losses_664527
!dense_593/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_537/PartitionedCall:output:0dense_593_665083dense_593_665085*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_593_layer_call_and_return_conditional_losses_664539
/batch_normalization_538/StatefulPartitionedCallStatefulPartitionedCall*dense_593/StatefulPartitionedCall:output:0batch_normalization_538_665088batch_normalization_538_665090batch_normalization_538_665092batch_normalization_538_665094*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_664180ø
leaky_re_lu_538/PartitionedCallPartitionedCall8batch_normalization_538/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_538_layer_call_and_return_conditional_losses_664559
!dense_594/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_538/PartitionedCall:output:0dense_594_665098dense_594_665100*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_594_layer_call_and_return_conditional_losses_664571
/batch_normalization_539/StatefulPartitionedCallStatefulPartitionedCall*dense_594/StatefulPartitionedCall:output:0batch_normalization_539_665103batch_normalization_539_665105batch_normalization_539_665107batch_normalization_539_665109*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_664262ø
leaky_re_lu_539/PartitionedCallPartitionedCall8batch_normalization_539/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_539_layer_call_and_return_conditional_losses_664591
!dense_595/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_539/PartitionedCall:output:0dense_595_665113dense_595_665115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_595_layer_call_and_return_conditional_losses_664603
/batch_normalization_540/StatefulPartitionedCallStatefulPartitionedCall*dense_595/StatefulPartitionedCall:output:0batch_normalization_540_665118batch_normalization_540_665120batch_normalization_540_665122batch_normalization_540_665124*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_664344ø
leaky_re_lu_540/PartitionedCallPartitionedCall8batch_normalization_540/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_540_layer_call_and_return_conditional_losses_664623
!dense_596/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_540/PartitionedCall:output:0dense_596_665128dense_596_665130*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_596_layer_call_and_return_conditional_losses_664635y
IdentityIdentity*dense_596/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_533/StatefulPartitionedCall0^batch_normalization_534/StatefulPartitionedCall0^batch_normalization_535/StatefulPartitionedCall0^batch_normalization_536/StatefulPartitionedCall0^batch_normalization_537/StatefulPartitionedCall0^batch_normalization_538/StatefulPartitionedCall0^batch_normalization_539/StatefulPartitionedCall0^batch_normalization_540/StatefulPartitionedCall"^dense_588/StatefulPartitionedCall"^dense_589/StatefulPartitionedCall"^dense_590/StatefulPartitionedCall"^dense_591/StatefulPartitionedCall"^dense_592/StatefulPartitionedCall"^dense_593/StatefulPartitionedCall"^dense_594/StatefulPartitionedCall"^dense_595/StatefulPartitionedCall"^dense_596/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_533/StatefulPartitionedCall/batch_normalization_533/StatefulPartitionedCall2b
/batch_normalization_534/StatefulPartitionedCall/batch_normalization_534/StatefulPartitionedCall2b
/batch_normalization_535/StatefulPartitionedCall/batch_normalization_535/StatefulPartitionedCall2b
/batch_normalization_536/StatefulPartitionedCall/batch_normalization_536/StatefulPartitionedCall2b
/batch_normalization_537/StatefulPartitionedCall/batch_normalization_537/StatefulPartitionedCall2b
/batch_normalization_538/StatefulPartitionedCall/batch_normalization_538/StatefulPartitionedCall2b
/batch_normalization_539/StatefulPartitionedCall/batch_normalization_539/StatefulPartitionedCall2b
/batch_normalization_540/StatefulPartitionedCall/batch_normalization_540/StatefulPartitionedCall2F
!dense_588/StatefulPartitionedCall!dense_588/StatefulPartitionedCall2F
!dense_589/StatefulPartitionedCall!dense_589/StatefulPartitionedCall2F
!dense_590/StatefulPartitionedCall!dense_590/StatefulPartitionedCall2F
!dense_591/StatefulPartitionedCall!dense_591/StatefulPartitionedCall2F
!dense_592/StatefulPartitionedCall!dense_592/StatefulPartitionedCall2F
!dense_593/StatefulPartitionedCall!dense_593/StatefulPartitionedCall2F
!dense_594/StatefulPartitionedCall!dense_594/StatefulPartitionedCall2F
!dense_595/StatefulPartitionedCall!dense_595/StatefulPartitionedCall2F
!dense_596/StatefulPartitionedCall!dense_596/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
È	
ö
E__inference_dense_590_layer_call_and_return_conditional_losses_664443

inputs0
matmul_readvariableop_resource:Y-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Y*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_663770

inputs5
'assignmovingavg_readvariableop_resource:Y7
)assignmovingavg_1_readvariableop_resource:Y3
%batchnorm_mul_readvariableop_resource:Y/
!batchnorm_readvariableop_resource:Y
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Y*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:Y
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Y*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:Y*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:Y*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:Y*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Yx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Y¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:Y*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Y~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Y´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:YP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Y~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Yc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Yv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_534_layer_call_fn_666670

inputs
unknown:Y
	unknown_0:Y
	unknown_1:Y
	unknown_2:Y
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_663852o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_540_layer_call_fn_667324

inputs
unknown:E
	unknown_0:E
	unknown_1:E
	unknown_2:E
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_664344o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
È	
ö
E__inference_dense_590_layer_call_and_return_conditional_losses_666753

inputs0
matmul_readvariableop_resource:Y-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Y*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_535_layer_call_fn_666779

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_663934o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¸
$__inference_signature_wrapper_666469
normalization_55_input
unknown
	unknown_0
	unknown_1:Y
	unknown_2:Y
	unknown_3:Y
	unknown_4:Y
	unknown_5:Y
	unknown_6:Y
	unknown_7:YY
	unknown_8:Y
	unknown_9:Y

unknown_10:Y

unknown_11:Y

unknown_12:Y

unknown_13:Y

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:E

unknown_38:E

unknown_39:E

unknown_40:E

unknown_41:E

unknown_42:E

unknown_43:EE

unknown_44:E

unknown_45:E

unknown_46:E

unknown_47:E

unknown_48:E

unknown_49:E

unknown_50:
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallnormalization_55_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_663699o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_55_input:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_533_layer_call_and_return_conditional_losses_664399

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿY:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
È	
ö
E__inference_dense_589_layer_call_and_return_conditional_losses_664411

inputs0
matmul_readvariableop_resource:YY-
biasadd_readvariableop_resource:Y
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:YY*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Y*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_663969

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_664016

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_592_layer_call_and_return_conditional_losses_664507

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_537_layer_call_and_return_conditional_losses_664527

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_666799

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_536_layer_call_and_return_conditional_losses_666952

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_664180

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_535_layer_call_and_return_conditional_losses_664463

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_595_layer_call_and_return_conditional_losses_664603

inputs0
matmul_readvariableop_resource:EE-
biasadd_readvariableop_resource:E
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:EE*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:E*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
Ä

*__inference_dense_590_layer_call_fn_666743

inputs
unknown:Y
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_590_layer_call_and_return_conditional_losses_664443o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_667017

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_666942

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
Â
.__inference_sequential_55_layer_call_fn_665350
normalization_55_input
unknown
	unknown_0
	unknown_1:Y
	unknown_2:Y
	unknown_3:Y
	unknown_4:Y
	unknown_5:Y
	unknown_6:Y
	unknown_7:YY
	unknown_8:Y
	unknown_9:Y

unknown_10:Y

unknown_11:Y

unknown_12:Y

unknown_13:Y

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:E

unknown_38:E

unknown_39:E

unknown_40:E

unknown_41:E

unknown_42:E

unknown_43:EE

unknown_44:E

unknown_45:E

unknown_46:E

unknown_47:E

unknown_48:E

unknown_49:E

unknown_50:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallnormalization_55_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.1234*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_55_layer_call_and_return_conditional_losses_665134o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_55_input:$ 

_output_shapes

::$ 

_output_shapes

:
%
ì
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_666724

inputs5
'assignmovingavg_readvariableop_resource:Y7
)assignmovingavg_1_readvariableop_resource:Y3
%batchnorm_mul_readvariableop_resource:Y/
!batchnorm_readvariableop_resource:Y
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Y*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:Y
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Y*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:Y*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:Y*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:Y*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Yx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Y¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:Y*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Y~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Y´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:YP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Y~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Yc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Yv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_664262

inputs5
'assignmovingavg_readvariableop_resource:E7
)assignmovingavg_1_readvariableop_resource:E3
%batchnorm_mul_readvariableop_resource:E/
!batchnorm_readvariableop_resource:E
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:E*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:E
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:E*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:E*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:E*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:E*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Ex
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:E¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:E*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:E~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:E´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:EP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:E~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ec
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Ev
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Er
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_537_layer_call_and_return_conditional_losses_667061

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_539_layer_call_fn_667215

inputs
unknown:E
	unknown_0:E
	unknown_1:E
	unknown_2:E
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_664262o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_533_layer_call_fn_666620

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_533_layer_call_and_return_conditional_losses_664399`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿY:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_664051

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_538_layer_call_and_return_conditional_losses_664559

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_533_layer_call_fn_666548

inputs
unknown:Y
	unknown_0:Y
	unknown_1:Y
	unknown_2:Y
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_663723o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
Ê
©
I__inference_sequential_55_layer_call_and_return_conditional_losses_665486
normalization_55_input
normalization_55_sub_y
normalization_55_sqrt_x"
dense_588_665360:Y
dense_588_665362:Y,
batch_normalization_533_665365:Y,
batch_normalization_533_665367:Y,
batch_normalization_533_665369:Y,
batch_normalization_533_665371:Y"
dense_589_665375:YY
dense_589_665377:Y,
batch_normalization_534_665380:Y,
batch_normalization_534_665382:Y,
batch_normalization_534_665384:Y,
batch_normalization_534_665386:Y"
dense_590_665390:Y
dense_590_665392:,
batch_normalization_535_665395:,
batch_normalization_535_665397:,
batch_normalization_535_665399:,
batch_normalization_535_665401:"
dense_591_665405:
dense_591_665407:,
batch_normalization_536_665410:,
batch_normalization_536_665412:,
batch_normalization_536_665414:,
batch_normalization_536_665416:"
dense_592_665420:
dense_592_665422:,
batch_normalization_537_665425:,
batch_normalization_537_665427:,
batch_normalization_537_665429:,
batch_normalization_537_665431:"
dense_593_665435:
dense_593_665437:,
batch_normalization_538_665440:,
batch_normalization_538_665442:,
batch_normalization_538_665444:,
batch_normalization_538_665446:"
dense_594_665450:E
dense_594_665452:E,
batch_normalization_539_665455:E,
batch_normalization_539_665457:E,
batch_normalization_539_665459:E,
batch_normalization_539_665461:E"
dense_595_665465:EE
dense_595_665467:E,
batch_normalization_540_665470:E,
batch_normalization_540_665472:E,
batch_normalization_540_665474:E,
batch_normalization_540_665476:E"
dense_596_665480:E
dense_596_665482:
identity¢/batch_normalization_533/StatefulPartitionedCall¢/batch_normalization_534/StatefulPartitionedCall¢/batch_normalization_535/StatefulPartitionedCall¢/batch_normalization_536/StatefulPartitionedCall¢/batch_normalization_537/StatefulPartitionedCall¢/batch_normalization_538/StatefulPartitionedCall¢/batch_normalization_539/StatefulPartitionedCall¢/batch_normalization_540/StatefulPartitionedCall¢!dense_588/StatefulPartitionedCall¢!dense_589/StatefulPartitionedCall¢!dense_590/StatefulPartitionedCall¢!dense_591/StatefulPartitionedCall¢!dense_592/StatefulPartitionedCall¢!dense_593/StatefulPartitionedCall¢!dense_594/StatefulPartitionedCall¢!dense_595/StatefulPartitionedCall¢!dense_596/StatefulPartitionedCall}
normalization_55/subSubnormalization_55_inputnormalization_55_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_55/SqrtSqrtnormalization_55_sqrt_x*
T0*
_output_shapes

:_
normalization_55/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_55/MaximumMaximumnormalization_55/Sqrt:y:0#normalization_55/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_55/truedivRealDivnormalization_55/sub:z:0normalization_55/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_588/StatefulPartitionedCallStatefulPartitionedCallnormalization_55/truediv:z:0dense_588_665360dense_588_665362*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_588_layer_call_and_return_conditional_losses_664379
/batch_normalization_533/StatefulPartitionedCallStatefulPartitionedCall*dense_588/StatefulPartitionedCall:output:0batch_normalization_533_665365batch_normalization_533_665367batch_normalization_533_665369batch_normalization_533_665371*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_663723ø
leaky_re_lu_533/PartitionedCallPartitionedCall8batch_normalization_533/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_533_layer_call_and_return_conditional_losses_664399
!dense_589/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_533/PartitionedCall:output:0dense_589_665375dense_589_665377*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_589_layer_call_and_return_conditional_losses_664411
/batch_normalization_534/StatefulPartitionedCallStatefulPartitionedCall*dense_589/StatefulPartitionedCall:output:0batch_normalization_534_665380batch_normalization_534_665382batch_normalization_534_665384batch_normalization_534_665386*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_663805ø
leaky_re_lu_534/PartitionedCallPartitionedCall8batch_normalization_534/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_534_layer_call_and_return_conditional_losses_664431
!dense_590/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_534/PartitionedCall:output:0dense_590_665390dense_590_665392*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_590_layer_call_and_return_conditional_losses_664443
/batch_normalization_535/StatefulPartitionedCallStatefulPartitionedCall*dense_590/StatefulPartitionedCall:output:0batch_normalization_535_665395batch_normalization_535_665397batch_normalization_535_665399batch_normalization_535_665401*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_663887ø
leaky_re_lu_535/PartitionedCallPartitionedCall8batch_normalization_535/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_535_layer_call_and_return_conditional_losses_664463
!dense_591/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_535/PartitionedCall:output:0dense_591_665405dense_591_665407*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_591_layer_call_and_return_conditional_losses_664475
/batch_normalization_536/StatefulPartitionedCallStatefulPartitionedCall*dense_591/StatefulPartitionedCall:output:0batch_normalization_536_665410batch_normalization_536_665412batch_normalization_536_665414batch_normalization_536_665416*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_663969ø
leaky_re_lu_536/PartitionedCallPartitionedCall8batch_normalization_536/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_536_layer_call_and_return_conditional_losses_664495
!dense_592/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_536/PartitionedCall:output:0dense_592_665420dense_592_665422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_592_layer_call_and_return_conditional_losses_664507
/batch_normalization_537/StatefulPartitionedCallStatefulPartitionedCall*dense_592/StatefulPartitionedCall:output:0batch_normalization_537_665425batch_normalization_537_665427batch_normalization_537_665429batch_normalization_537_665431*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_664051ø
leaky_re_lu_537/PartitionedCallPartitionedCall8batch_normalization_537/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_537_layer_call_and_return_conditional_losses_664527
!dense_593/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_537/PartitionedCall:output:0dense_593_665435dense_593_665437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_593_layer_call_and_return_conditional_losses_664539
/batch_normalization_538/StatefulPartitionedCallStatefulPartitionedCall*dense_593/StatefulPartitionedCall:output:0batch_normalization_538_665440batch_normalization_538_665442batch_normalization_538_665444batch_normalization_538_665446*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_664133ø
leaky_re_lu_538/PartitionedCallPartitionedCall8batch_normalization_538/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_538_layer_call_and_return_conditional_losses_664559
!dense_594/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_538/PartitionedCall:output:0dense_594_665450dense_594_665452*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_594_layer_call_and_return_conditional_losses_664571
/batch_normalization_539/StatefulPartitionedCallStatefulPartitionedCall*dense_594/StatefulPartitionedCall:output:0batch_normalization_539_665455batch_normalization_539_665457batch_normalization_539_665459batch_normalization_539_665461*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_664215ø
leaky_re_lu_539/PartitionedCallPartitionedCall8batch_normalization_539/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_539_layer_call_and_return_conditional_losses_664591
!dense_595/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_539/PartitionedCall:output:0dense_595_665465dense_595_665467*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_595_layer_call_and_return_conditional_losses_664603
/batch_normalization_540/StatefulPartitionedCallStatefulPartitionedCall*dense_595/StatefulPartitionedCall:output:0batch_normalization_540_665470batch_normalization_540_665472batch_normalization_540_665474batch_normalization_540_665476*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_664297ø
leaky_re_lu_540/PartitionedCallPartitionedCall8batch_normalization_540/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_540_layer_call_and_return_conditional_losses_664623
!dense_596/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_540/PartitionedCall:output:0dense_596_665480dense_596_665482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_596_layer_call_and_return_conditional_losses_664635y
IdentityIdentity*dense_596/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_533/StatefulPartitionedCall0^batch_normalization_534/StatefulPartitionedCall0^batch_normalization_535/StatefulPartitionedCall0^batch_normalization_536/StatefulPartitionedCall0^batch_normalization_537/StatefulPartitionedCall0^batch_normalization_538/StatefulPartitionedCall0^batch_normalization_539/StatefulPartitionedCall0^batch_normalization_540/StatefulPartitionedCall"^dense_588/StatefulPartitionedCall"^dense_589/StatefulPartitionedCall"^dense_590/StatefulPartitionedCall"^dense_591/StatefulPartitionedCall"^dense_592/StatefulPartitionedCall"^dense_593/StatefulPartitionedCall"^dense_594/StatefulPartitionedCall"^dense_595/StatefulPartitionedCall"^dense_596/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_533/StatefulPartitionedCall/batch_normalization_533/StatefulPartitionedCall2b
/batch_normalization_534/StatefulPartitionedCall/batch_normalization_534/StatefulPartitionedCall2b
/batch_normalization_535/StatefulPartitionedCall/batch_normalization_535/StatefulPartitionedCall2b
/batch_normalization_536/StatefulPartitionedCall/batch_normalization_536/StatefulPartitionedCall2b
/batch_normalization_537/StatefulPartitionedCall/batch_normalization_537/StatefulPartitionedCall2b
/batch_normalization_538/StatefulPartitionedCall/batch_normalization_538/StatefulPartitionedCall2b
/batch_normalization_539/StatefulPartitionedCall/batch_normalization_539/StatefulPartitionedCall2b
/batch_normalization_540/StatefulPartitionedCall/batch_normalization_540/StatefulPartitionedCall2F
!dense_588/StatefulPartitionedCall!dense_588/StatefulPartitionedCall2F
!dense_589/StatefulPartitionedCall!dense_589/StatefulPartitionedCall2F
!dense_590/StatefulPartitionedCall!dense_590/StatefulPartitionedCall2F
!dense_591/StatefulPartitionedCall!dense_591/StatefulPartitionedCall2F
!dense_592/StatefulPartitionedCall!dense_592/StatefulPartitionedCall2F
!dense_593/StatefulPartitionedCall!dense_593/StatefulPartitionedCall2F
!dense_594/StatefulPartitionedCall!dense_594/StatefulPartitionedCall2F
!dense_595/StatefulPartitionedCall!dense_595/StatefulPartitionedCall2F
!dense_596/StatefulPartitionedCall!dense_596/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_55_input:$ 

_output_shapes

::$ 

_output_shapes

:
È	
ö
E__inference_dense_588_layer_call_and_return_conditional_losses_664379

inputs0
matmul_readvariableop_resource:Y-
biasadd_readvariableop_resource:Y
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Y*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Y*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_664297

inputs/
!batchnorm_readvariableop_resource:E3
%batchnorm_mul_readvariableop_resource:E1
#batchnorm_readvariableop_1_resource:E1
#batchnorm_readvariableop_2_resource:E
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:EP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:E~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ec
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:E*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Ez
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:E*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Er
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_666615

inputs5
'assignmovingavg_readvariableop_resource:Y7
)assignmovingavg_1_readvariableop_resource:Y3
%batchnorm_mul_readvariableop_resource:Y/
!batchnorm_readvariableop_resource:Y
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Y*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:Y
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Y*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:Y*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:Y*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:Y*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Yx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Y¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:Y*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Y~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Y´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:YP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Y~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Yc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Yv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
È	
ö
E__inference_dense_591_layer_call_and_return_conditional_losses_666862

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_663934

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_536_layer_call_and_return_conditional_losses_664495

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_537_layer_call_fn_666984

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_664051o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_539_layer_call_and_return_conditional_losses_664591

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿE:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_537_layer_call_fn_667056

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_537_layer_call_and_return_conditional_losses_664527`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_593_layer_call_and_return_conditional_losses_667080

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
²
.__inference_sequential_55_layer_call_fn_665844

inputs
unknown
	unknown_0
	unknown_1:Y
	unknown_2:Y
	unknown_3:Y
	unknown_4:Y
	unknown_5:Y
	unknown_6:Y
	unknown_7:YY
	unknown_8:Y
	unknown_9:Y

unknown_10:Y

unknown_11:Y

unknown_12:Y

unknown_13:Y

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:E

unknown_38:E

unknown_39:E

unknown_40:E

unknown_41:E

unknown_42:E

unknown_43:EE

unknown_44:E

unknown_45:E

unknown_46:E

unknown_47:E

unknown_48:E

unknown_49:E

unknown_50:
identity¢StatefulPartitionedCallÿ
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
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.1234*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_55_layer_call_and_return_conditional_losses_665134o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
%
ì
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_663852

inputs5
'assignmovingavg_readvariableop_resource:Y7
)assignmovingavg_1_readvariableop_resource:Y3
%batchnorm_mul_readvariableop_resource:Y/
!batchnorm_readvariableop_resource:Y
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Y*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:Y
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Y*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:Y*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:Y*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:Y*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Yx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Y¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:Y*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Y~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Y´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:YP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Y~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Yc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Yv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_667051

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_667126

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_534_layer_call_fn_666729

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_534_layer_call_and_return_conditional_losses_664431`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿY:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
È	
ö
E__inference_dense_596_layer_call_and_return_conditional_losses_667407

inputs0
matmul_readvariableop_resource:E-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:E*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_666833

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_595_layer_call_and_return_conditional_losses_667298

inputs0
matmul_readvariableop_resource:EE-
biasadd_readvariableop_resource:E
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:EE*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:E*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_667378

inputs5
'assignmovingavg_readvariableop_resource:E7
)assignmovingavg_1_readvariableop_resource:E3
%batchnorm_mul_readvariableop_resource:E/
!batchnorm_readvariableop_resource:E
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:E*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:E
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:E*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:E*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:E*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:E*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Ex
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:E¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:E*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:E~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:E´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:EP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:E~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ec
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Ev
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Er
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_539_layer_call_and_return_conditional_losses_667279

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿE:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
È	
ö
E__inference_dense_596_layer_call_and_return_conditional_losses_664635

inputs0
matmul_readvariableop_resource:E-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:E*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_667344

inputs/
!batchnorm_readvariableop_resource:E3
%batchnorm_mul_readvariableop_resource:E1
#batchnorm_readvariableop_1_resource:E1
#batchnorm_readvariableop_2_resource:E
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:EP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:E~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ec
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:E*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Ez
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:E*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Er
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_533_layer_call_and_return_conditional_losses_666625

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿY:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_536_layer_call_fn_666947

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_536_layer_call_and_return_conditional_losses_664495`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_534_layer_call_and_return_conditional_losses_664431

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿY:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs

Â
.__inference_sequential_55_layer_call_fn_664749
normalization_55_input
unknown
	unknown_0
	unknown_1:Y
	unknown_2:Y
	unknown_3:Y
	unknown_4:Y
	unknown_5:Y
	unknown_6:Y
	unknown_7:YY
	unknown_8:Y
	unknown_9:Y

unknown_10:Y

unknown_11:Y

unknown_12:Y

unknown_13:Y

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:E

unknown_38:E

unknown_39:E

unknown_40:E

unknown_41:E

unknown_42:E

unknown_43:EE

unknown_44:E

unknown_45:E

unknown_46:E

unknown_47:E

unknown_48:E

unknown_49:E

unknown_50:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallnormalization_55_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_55_layer_call_and_return_conditional_losses_664642o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_55_input:$ 

_output_shapes

::$ 

_output_shapes

:
%
ì
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_664098

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_535_layer_call_fn_666766

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_663887o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_591_layer_call_fn_666852

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_591_layer_call_and_return_conditional_losses_664475o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_667235

inputs/
!batchnorm_readvariableop_resource:E3
%batchnorm_mul_readvariableop_resource:E1
#batchnorm_readvariableop_1_resource:E1
#batchnorm_readvariableop_2_resource:E
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:EP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:E~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ec
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:E*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Ez
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:E*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Er
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_540_layer_call_fn_667311

inputs
unknown:E
	unknown_0:E
	unknown_1:E
	unknown_2:E
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_664297o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_538_layer_call_and_return_conditional_losses_667170

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_593_layer_call_and_return_conditional_losses_664539

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
²
.__inference_sequential_55_layer_call_fn_665735

inputs
unknown
	unknown_0
	unknown_1:Y
	unknown_2:Y
	unknown_3:Y
	unknown_4:Y
	unknown_5:Y
	unknown_6:Y
	unknown_7:YY
	unknown_8:Y
	unknown_9:Y

unknown_10:Y

unknown_11:Y

unknown_12:Y

unknown_13:Y

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:E

unknown_38:E

unknown_39:E

unknown_40:E

unknown_41:E

unknown_42:E

unknown_43:EE

unknown_44:E

unknown_45:E

unknown_46:E

unknown_47:E

unknown_48:E

unknown_49:E

unknown_50:
identity¢StatefulPartitionedCall
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
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_55_layer_call_and_return_conditional_losses_664642o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Í
òT
"__inference__traced_restore_668204
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_588_kernel:Y/
!assignvariableop_4_dense_588_bias:Y>
0assignvariableop_5_batch_normalization_533_gamma:Y=
/assignvariableop_6_batch_normalization_533_beta:YD
6assignvariableop_7_batch_normalization_533_moving_mean:YH
:assignvariableop_8_batch_normalization_533_moving_variance:Y5
#assignvariableop_9_dense_589_kernel:YY0
"assignvariableop_10_dense_589_bias:Y?
1assignvariableop_11_batch_normalization_534_gamma:Y>
0assignvariableop_12_batch_normalization_534_beta:YE
7assignvariableop_13_batch_normalization_534_moving_mean:YI
;assignvariableop_14_batch_normalization_534_moving_variance:Y6
$assignvariableop_15_dense_590_kernel:Y0
"assignvariableop_16_dense_590_bias:?
1assignvariableop_17_batch_normalization_535_gamma:>
0assignvariableop_18_batch_normalization_535_beta:E
7assignvariableop_19_batch_normalization_535_moving_mean:I
;assignvariableop_20_batch_normalization_535_moving_variance:6
$assignvariableop_21_dense_591_kernel:0
"assignvariableop_22_dense_591_bias:?
1assignvariableop_23_batch_normalization_536_gamma:>
0assignvariableop_24_batch_normalization_536_beta:E
7assignvariableop_25_batch_normalization_536_moving_mean:I
;assignvariableop_26_batch_normalization_536_moving_variance:6
$assignvariableop_27_dense_592_kernel:0
"assignvariableop_28_dense_592_bias:?
1assignvariableop_29_batch_normalization_537_gamma:>
0assignvariableop_30_batch_normalization_537_beta:E
7assignvariableop_31_batch_normalization_537_moving_mean:I
;assignvariableop_32_batch_normalization_537_moving_variance:6
$assignvariableop_33_dense_593_kernel:0
"assignvariableop_34_dense_593_bias:?
1assignvariableop_35_batch_normalization_538_gamma:>
0assignvariableop_36_batch_normalization_538_beta:E
7assignvariableop_37_batch_normalization_538_moving_mean:I
;assignvariableop_38_batch_normalization_538_moving_variance:6
$assignvariableop_39_dense_594_kernel:E0
"assignvariableop_40_dense_594_bias:E?
1assignvariableop_41_batch_normalization_539_gamma:E>
0assignvariableop_42_batch_normalization_539_beta:EE
7assignvariableop_43_batch_normalization_539_moving_mean:EI
;assignvariableop_44_batch_normalization_539_moving_variance:E6
$assignvariableop_45_dense_595_kernel:EE0
"assignvariableop_46_dense_595_bias:E?
1assignvariableop_47_batch_normalization_540_gamma:E>
0assignvariableop_48_batch_normalization_540_beta:EE
7assignvariableop_49_batch_normalization_540_moving_mean:EI
;assignvariableop_50_batch_normalization_540_moving_variance:E6
$assignvariableop_51_dense_596_kernel:E0
"assignvariableop_52_dense_596_bias:'
assignvariableop_53_adam_iter:	 )
assignvariableop_54_adam_beta_1: )
assignvariableop_55_adam_beta_2: (
assignvariableop_56_adam_decay: #
assignvariableop_57_total: %
assignvariableop_58_count_1: =
+assignvariableop_59_adam_dense_588_kernel_m:Y7
)assignvariableop_60_adam_dense_588_bias_m:YF
8assignvariableop_61_adam_batch_normalization_533_gamma_m:YE
7assignvariableop_62_adam_batch_normalization_533_beta_m:Y=
+assignvariableop_63_adam_dense_589_kernel_m:YY7
)assignvariableop_64_adam_dense_589_bias_m:YF
8assignvariableop_65_adam_batch_normalization_534_gamma_m:YE
7assignvariableop_66_adam_batch_normalization_534_beta_m:Y=
+assignvariableop_67_adam_dense_590_kernel_m:Y7
)assignvariableop_68_adam_dense_590_bias_m:F
8assignvariableop_69_adam_batch_normalization_535_gamma_m:E
7assignvariableop_70_adam_batch_normalization_535_beta_m:=
+assignvariableop_71_adam_dense_591_kernel_m:7
)assignvariableop_72_adam_dense_591_bias_m:F
8assignvariableop_73_adam_batch_normalization_536_gamma_m:E
7assignvariableop_74_adam_batch_normalization_536_beta_m:=
+assignvariableop_75_adam_dense_592_kernel_m:7
)assignvariableop_76_adam_dense_592_bias_m:F
8assignvariableop_77_adam_batch_normalization_537_gamma_m:E
7assignvariableop_78_adam_batch_normalization_537_beta_m:=
+assignvariableop_79_adam_dense_593_kernel_m:7
)assignvariableop_80_adam_dense_593_bias_m:F
8assignvariableop_81_adam_batch_normalization_538_gamma_m:E
7assignvariableop_82_adam_batch_normalization_538_beta_m:=
+assignvariableop_83_adam_dense_594_kernel_m:E7
)assignvariableop_84_adam_dense_594_bias_m:EF
8assignvariableop_85_adam_batch_normalization_539_gamma_m:EE
7assignvariableop_86_adam_batch_normalization_539_beta_m:E=
+assignvariableop_87_adam_dense_595_kernel_m:EE7
)assignvariableop_88_adam_dense_595_bias_m:EF
8assignvariableop_89_adam_batch_normalization_540_gamma_m:EE
7assignvariableop_90_adam_batch_normalization_540_beta_m:E=
+assignvariableop_91_adam_dense_596_kernel_m:E7
)assignvariableop_92_adam_dense_596_bias_m:=
+assignvariableop_93_adam_dense_588_kernel_v:Y7
)assignvariableop_94_adam_dense_588_bias_v:YF
8assignvariableop_95_adam_batch_normalization_533_gamma_v:YE
7assignvariableop_96_adam_batch_normalization_533_beta_v:Y=
+assignvariableop_97_adam_dense_589_kernel_v:YY7
)assignvariableop_98_adam_dense_589_bias_v:YF
8assignvariableop_99_adam_batch_normalization_534_gamma_v:YF
8assignvariableop_100_adam_batch_normalization_534_beta_v:Y>
,assignvariableop_101_adam_dense_590_kernel_v:Y8
*assignvariableop_102_adam_dense_590_bias_v:G
9assignvariableop_103_adam_batch_normalization_535_gamma_v:F
8assignvariableop_104_adam_batch_normalization_535_beta_v:>
,assignvariableop_105_adam_dense_591_kernel_v:8
*assignvariableop_106_adam_dense_591_bias_v:G
9assignvariableop_107_adam_batch_normalization_536_gamma_v:F
8assignvariableop_108_adam_batch_normalization_536_beta_v:>
,assignvariableop_109_adam_dense_592_kernel_v:8
*assignvariableop_110_adam_dense_592_bias_v:G
9assignvariableop_111_adam_batch_normalization_537_gamma_v:F
8assignvariableop_112_adam_batch_normalization_537_beta_v:>
,assignvariableop_113_adam_dense_593_kernel_v:8
*assignvariableop_114_adam_dense_593_bias_v:G
9assignvariableop_115_adam_batch_normalization_538_gamma_v:F
8assignvariableop_116_adam_batch_normalization_538_beta_v:>
,assignvariableop_117_adam_dense_594_kernel_v:E8
*assignvariableop_118_adam_dense_594_bias_v:EG
9assignvariableop_119_adam_batch_normalization_539_gamma_v:EF
8assignvariableop_120_adam_batch_normalization_539_beta_v:E>
,assignvariableop_121_adam_dense_595_kernel_v:EE8
*assignvariableop_122_adam_dense_595_bias_v:EG
9assignvariableop_123_adam_batch_normalization_540_gamma_v:EF
8assignvariableop_124_adam_batch_normalization_540_beta_v:E>
,assignvariableop_125_adam_dense_596_kernel_v:E8
*assignvariableop_126_adam_dense_596_bias_v:
identity_128¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_111¢AssignVariableOp_112¢AssignVariableOp_113¢AssignVariableOp_114¢AssignVariableOp_115¢AssignVariableOp_116¢AssignVariableOp_117¢AssignVariableOp_118¢AssignVariableOp_119¢AssignVariableOp_12¢AssignVariableOp_120¢AssignVariableOp_121¢AssignVariableOp_122¢AssignVariableOp_123¢AssignVariableOp_124¢AssignVariableOp_125¢AssignVariableOp_126¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99½G
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*âF
valueØFBÕFB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHõ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*
valueBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¥
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_588_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_588_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batch_normalization_533_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_533_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_533_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_8AssignVariableOp:assignvariableop_8_batch_normalization_533_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_589_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_589_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batch_normalization_534_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_534_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_534_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_14AssignVariableOp;assignvariableop_14_batch_normalization_534_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_590_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_590_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batch_normalization_535_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_535_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_535_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_20AssignVariableOp;assignvariableop_20_batch_normalization_535_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_591_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_591_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_23AssignVariableOp1assignvariableop_23_batch_normalization_536_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_536_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_25AssignVariableOp7assignvariableop_25_batch_normalization_536_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_26AssignVariableOp;assignvariableop_26_batch_normalization_536_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_592_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_592_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_29AssignVariableOp1assignvariableop_29_batch_normalization_537_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_537_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_31AssignVariableOp7assignvariableop_31_batch_normalization_537_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_32AssignVariableOp;assignvariableop_32_batch_normalization_537_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_593_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_593_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_35AssignVariableOp1assignvariableop_35_batch_normalization_538_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_538_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_37AssignVariableOp7assignvariableop_37_batch_normalization_538_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_38AssignVariableOp;assignvariableop_38_batch_normalization_538_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp$assignvariableop_39_dense_594_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_594_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_41AssignVariableOp1assignvariableop_41_batch_normalization_539_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_batch_normalization_539_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_43AssignVariableOp7assignvariableop_43_batch_normalization_539_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_44AssignVariableOp;assignvariableop_44_batch_normalization_539_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp$assignvariableop_45_dense_595_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_595_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_47AssignVariableOp1assignvariableop_47_batch_normalization_540_gammaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_48AssignVariableOp0assignvariableop_48_batch_normalization_540_betaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_49AssignVariableOp7assignvariableop_49_batch_normalization_540_moving_meanIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_50AssignVariableOp;assignvariableop_50_batch_normalization_540_moving_varianceIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp$assignvariableop_51_dense_596_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp"assignvariableop_52_dense_596_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_53AssignVariableOpassignvariableop_53_adam_iterIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_beta_1Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_beta_2Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_decayIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOpassignvariableop_57_totalIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOpassignvariableop_58_count_1Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_dense_588_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_dense_588_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_61AssignVariableOp8assignvariableop_61_adam_batch_normalization_533_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_62AssignVariableOp7assignvariableop_62_adam_batch_normalization_533_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_dense_589_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_dense_589_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_65AssignVariableOp8assignvariableop_65_adam_batch_normalization_534_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_66AssignVariableOp7assignvariableop_66_adam_batch_normalization_534_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_dense_590_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_dense_590_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_69AssignVariableOp8assignvariableop_69_adam_batch_normalization_535_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_70AssignVariableOp7assignvariableop_70_adam_batch_normalization_535_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_dense_591_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_dense_591_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_73AssignVariableOp8assignvariableop_73_adam_batch_normalization_536_gamma_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_74AssignVariableOp7assignvariableop_74_adam_batch_normalization_536_beta_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_dense_592_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_dense_592_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_77AssignVariableOp8assignvariableop_77_adam_batch_normalization_537_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_78AssignVariableOp7assignvariableop_78_adam_batch_normalization_537_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_dense_593_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_dense_593_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_81AssignVariableOp8assignvariableop_81_adam_batch_normalization_538_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_82AssignVariableOp7assignvariableop_82_adam_batch_normalization_538_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adam_dense_594_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adam_dense_594_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_85AssignVariableOp8assignvariableop_85_adam_batch_normalization_539_gamma_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_86AssignVariableOp7assignvariableop_86_adam_batch_normalization_539_beta_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_87AssignVariableOp+assignvariableop_87_adam_dense_595_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_88AssignVariableOp)assignvariableop_88_adam_dense_595_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_89AssignVariableOp8assignvariableop_89_adam_batch_normalization_540_gamma_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_90AssignVariableOp7assignvariableop_90_adam_batch_normalization_540_beta_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_91AssignVariableOp+assignvariableop_91_adam_dense_596_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_92AssignVariableOp)assignvariableop_92_adam_dense_596_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_dense_588_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_dense_588_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_95AssignVariableOp8assignvariableop_95_adam_batch_normalization_533_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_96AssignVariableOp7assignvariableop_96_adam_batch_normalization_533_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_dense_589_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_dense_589_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_99AssignVariableOp8assignvariableop_99_adam_batch_normalization_534_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_100AssignVariableOp8assignvariableop_100_adam_batch_normalization_534_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_dense_590_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_dense_590_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_103AssignVariableOp9assignvariableop_103_adam_batch_normalization_535_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_104AssignVariableOp8assignvariableop_104_adam_batch_normalization_535_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_dense_591_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_dense_591_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_107AssignVariableOp9assignvariableop_107_adam_batch_normalization_536_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_108AssignVariableOp8assignvariableop_108_adam_batch_normalization_536_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_dense_592_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_dense_592_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_111AssignVariableOp9assignvariableop_111_adam_batch_normalization_537_gamma_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_112AssignVariableOp8assignvariableop_112_adam_batch_normalization_537_beta_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_dense_593_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_dense_593_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_115AssignVariableOp9assignvariableop_115_adam_batch_normalization_538_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_116AssignVariableOp8assignvariableop_116_adam_batch_normalization_538_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_dense_594_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_dense_594_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_119AssignVariableOp9assignvariableop_119_adam_batch_normalization_539_gamma_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_120AssignVariableOp8assignvariableop_120_adam_batch_normalization_539_beta_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_121AssignVariableOp,assignvariableop_121_adam_dense_595_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_122AssignVariableOp*assignvariableop_122_adam_dense_595_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_123AssignVariableOp9assignvariableop_123_adam_batch_normalization_540_gamma_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_124AssignVariableOp8assignvariableop_124_adam_batch_normalization_540_beta_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_125AssignVariableOp,assignvariableop_125_adam_dense_596_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_126AssignVariableOp*assignvariableop_126_adam_dense_596_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Õ
Identity_127Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_128IdentityIdentity_127:output:0^NoOp_1*
T0*
_output_shapes
: Á
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_128Identity_128:output:0*
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262*
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
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ð
²
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_666690

inputs/
!batchnorm_readvariableop_resource:Y3
%batchnorm_mul_readvariableop_resource:Y1
#batchnorm_readvariableop_1_resource:Y1
#batchnorm_readvariableop_2_resource:Y
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:YP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Y~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Yc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:Y*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Yz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:Y*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_538_layer_call_fn_667093

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_664133o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_667160

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_540_layer_call_and_return_conditional_losses_664623

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿE:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_539_layer_call_fn_667202

inputs
unknown:E
	unknown_0:E
	unknown_1:E
	unknown_2:E
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_664215o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_663723

inputs/
!batchnorm_readvariableop_resource:Y3
%batchnorm_mul_readvariableop_resource:Y1
#batchnorm_readvariableop_1_resource:Y1
#batchnorm_readvariableop_2_resource:Y
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Y*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:YP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Y~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Y*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Yc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:Y*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Yz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:Y*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Yr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_535_layer_call_fn_666838

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_535_layer_call_and_return_conditional_losses_664463`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_589_layer_call_and_return_conditional_losses_666644

inputs0
matmul_readvariableop_resource:YY-
biasadd_readvariableop_resource:Y
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:YY*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Y*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿYw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿY: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_664344

inputs5
'assignmovingavg_readvariableop_resource:E7
)assignmovingavg_1_readvariableop_resource:E3
%batchnorm_mul_readvariableop_resource:E/
!batchnorm_readvariableop_resource:E
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:E*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:E
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:E*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:E*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:E*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:E*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Ex
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:E¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:E*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:E~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:E´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:EP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:E~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ec
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Ev
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Er
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_538_layer_call_fn_667165

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_538_layer_call_and_return_conditional_losses_664559`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_667269

inputs5
'assignmovingavg_readvariableop_resource:E7
)assignmovingavg_1_readvariableop_resource:E3
%batchnorm_mul_readvariableop_resource:E/
!batchnorm_readvariableop_resource:E
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:E*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:E
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:E*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:E*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:E*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:E*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Ex
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:E¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:E*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:E~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:E´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:EP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:E~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:E*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ec
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Ev
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:E*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Er
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿEê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_540_layer_call_fn_667383

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_540_layer_call_and_return_conditional_losses_664623`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿE:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_664133

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_593_layer_call_fn_667070

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_593_layer_call_and_return_conditional_losses_664539o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ê
serving_default¶
Y
normalization_55_input?
(serving_default_normalization_55_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_5960
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ïÅ
¤
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
layer_with_weights-14
layer-20
layer-21
layer_with_weights-15
layer-22
layer_with_weights-16
layer-23
layer-24
layer_with_weights-17
layer-25
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_default_save_signature
#
signatures"
_tf_keras_sequential
Ó
$
_keep_axis
%_reduce_axis
&_reduce_axis_mask
'_broadcast_shape
(mean
(
adapt_mean
)variance
)adapt_variance
	*count
+	keras_api
,_adapt_function"
_tf_keras_layer
»

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
5axis
	6gamma
7beta
8moving_mean
9moving_variance
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
»

_kernel
`bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
gaxis
	hgamma
ibeta
jmoving_mean
kmoving_variance
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
»

xkernel
ybias
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
 regularization_losses
¡	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"
_tf_keras_layer
«
¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
ªkernel
	«bias
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	²axis

³gamma
	´beta
µmoving_mean
¶moving_variance
·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
»__call__
+¼&call_and_return_all_conditional_losses"
_tf_keras_layer
«
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ãkernel
	Äbias
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	Ëaxis

Ìgamma
	Íbeta
Îmoving_mean
Ïmoving_variance
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ó	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ö	variables
×trainable_variables
Øregularization_losses
Ù	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ükernel
	Ýbias
Þ	variables
ßtrainable_variables
àregularization_losses
á	keras_api
â__call__
+ã&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	äaxis

ågamma
	æbeta
çmoving_mean
èmoving_variance
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
í__call__
+î&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
õkernel
	öbias
÷	variables
øtrainable_variables
ùregularization_losses
ú	keras_api
û__call__
+ü&call_and_return_all_conditional_losses"
_tf_keras_layer

	ýiter
þbeta_1
ÿbeta_2

decay-m.m6m7mFmGmOmPm_m`mhmimxmym	m	m	m	m	m	m	ªm	«m	³m	´m 	Ãm¡	Äm¢	Ìm£	Ím¤	Üm¥	Ým¦	åm§	æm¨	õm©	ömª-v«.v¬6v­7v®Fv¯Gv°Ov±Pv²_v³`v´hvµiv¶xv·yv¸	v¹	vº	v»	v¼	v½	v¾	ªv¿	«vÀ	³vÁ	´vÂ	ÃvÃ	ÄvÄ	ÌvÅ	ÍvÆ	ÜvÇ	ÝvÈ	åvÉ	ævÊ	õvË	övÌ"
	optimizer
Ü
(0
)1
*2
-3
.4
65
76
87
98
F9
G10
O11
P12
Q13
R14
_15
`16
h17
i18
j19
k20
x21
y22
23
24
25
26
27
28
29
30
31
32
ª33
«34
³35
´36
µ37
¶38
Ã39
Ä40
Ì41
Í42
Î43
Ï44
Ü45
Ý46
å47
æ48
ç49
è50
õ51
ö52"
trackable_list_wrapper
º
-0
.1
62
73
F4
G5
O6
P7
_8
`9
h10
i11
x12
y13
14
15
16
17
18
19
ª20
«21
³22
´23
Ã24
Ä25
Ì26
Í27
Ü28
Ý29
å30
æ31
õ32
ö33"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
"_default_save_signature
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_sequential_55_layer_call_fn_664749
.__inference_sequential_55_layer_call_fn_665735
.__inference_sequential_55_layer_call_fn_665844
.__inference_sequential_55_layer_call_fn_665350À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
I__inference_sequential_55_layer_call_and_return_conditional_losses_666045
I__inference_sequential_55_layer_call_and_return_conditional_losses_666358
I__inference_sequential_55_layer_call_and_return_conditional_losses_665486
I__inference_sequential_55_layer_call_and_return_conditional_losses_665622À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÛBØ
!__inference__wrapped_model_663699normalization_55_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-
serving_default"
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
¿2¼
__inference_adapt_step_666516
²
FullArgSpec
args

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": Y2dense_588/kernel
:Y2dense_588/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_588_layer_call_fn_666525¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_588_layer_call_and_return_conditional_losses_666535¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)Y2batch_normalization_533/gamma
*:(Y2batch_normalization_533/beta
3:1Y (2#batch_normalization_533/moving_mean
7:5Y (2'batch_normalization_533/moving_variance
<
60
71
82
93"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_533_layer_call_fn_666548
8__inference_batch_normalization_533_layer_call_fn_666561´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_666581
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_666615´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_533_layer_call_fn_666620¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_leaky_re_lu_533_layer_call_and_return_conditional_losses_666625¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": YY2dense_589/kernel
:Y2dense_589/bias
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_589_layer_call_fn_666634¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_589_layer_call_and_return_conditional_losses_666644¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)Y2batch_normalization_534/gamma
*:(Y2batch_normalization_534/beta
3:1Y (2#batch_normalization_534/moving_mean
7:5Y (2'batch_normalization_534/moving_variance
<
O0
P1
Q2
R3"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_534_layer_call_fn_666657
8__inference_batch_normalization_534_layer_call_fn_666670´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_666690
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_666724´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_534_layer_call_fn_666729¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_leaky_re_lu_534_layer_call_and_return_conditional_losses_666734¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": Y2dense_590/kernel
:2dense_590/bias
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_590_layer_call_fn_666743¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_590_layer_call_and_return_conditional_losses_666753¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)2batch_normalization_535/gamma
*:(2batch_normalization_535/beta
3:1 (2#batch_normalization_535/moving_mean
7:5 (2'batch_normalization_535/moving_variance
<
h0
i1
j2
k3"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_535_layer_call_fn_666766
8__inference_batch_normalization_535_layer_call_fn_666779´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_666799
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_666833´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_535_layer_call_fn_666838¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_leaky_re_lu_535_layer_call_and_return_conditional_losses_666843¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 2dense_591/kernel
:2dense_591/bias
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_591_layer_call_fn_666852¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_591_layer_call_and_return_conditional_losses_666862¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)2batch_normalization_536/gamma
*:(2batch_normalization_536/beta
3:1 (2#batch_normalization_536/moving_mean
7:5 (2'batch_normalization_536/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_536_layer_call_fn_666875
8__inference_batch_normalization_536_layer_call_fn_666888´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_666908
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_666942´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_536_layer_call_fn_666947¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_leaky_re_lu_536_layer_call_and_return_conditional_losses_666952¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 2dense_592/kernel
:2dense_592/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_592_layer_call_fn_666961¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_592_layer_call_and_return_conditional_losses_666971¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)2batch_normalization_537/gamma
*:(2batch_normalization_537/beta
3:1 (2#batch_normalization_537/moving_mean
7:5 (2'batch_normalization_537/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
	variables
trainable_variables
 regularization_losses
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_537_layer_call_fn_666984
8__inference_batch_normalization_537_layer_call_fn_666997´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_667017
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_667051´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_537_layer_call_fn_667056¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_leaky_re_lu_537_layer_call_and_return_conditional_losses_667061¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 2dense_593/kernel
:2dense_593/bias
0
ª0
«1"
trackable_list_wrapper
0
ª0
«1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_593_layer_call_fn_667070¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_593_layer_call_and_return_conditional_losses_667080¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)2batch_normalization_538/gamma
*:(2batch_normalization_538/beta
3:1 (2#batch_normalization_538/moving_mean
7:5 (2'batch_normalization_538/moving_variance
@
³0
´1
µ2
¶3"
trackable_list_wrapper
0
³0
´1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
·	variables
¸trainable_variables
¹regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_538_layer_call_fn_667093
8__inference_batch_normalization_538_layer_call_fn_667106´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_667126
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_667160´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_538_layer_call_fn_667165¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_leaky_re_lu_538_layer_call_and_return_conditional_losses_667170¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": E2dense_594/kernel
:E2dense_594/bias
0
Ã0
Ä1"
trackable_list_wrapper
0
Ã0
Ä1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
Å	variables
Ætrainable_variables
Çregularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_594_layer_call_fn_667179¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_594_layer_call_and_return_conditional_losses_667189¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)E2batch_normalization_539/gamma
*:(E2batch_normalization_539/beta
3:1E (2#batch_normalization_539/moving_mean
7:5E (2'batch_normalization_539/moving_variance
@
Ì0
Í1
Î2
Ï3"
trackable_list_wrapper
0
Ì0
Í1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_539_layer_call_fn_667202
8__inference_batch_normalization_539_layer_call_fn_667215´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_667235
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_667269´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
Ö	variables
×trainable_variables
Øregularization_losses
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_539_layer_call_fn_667274¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_leaky_re_lu_539_layer_call_and_return_conditional_losses_667279¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": EE2dense_595/kernel
:E2dense_595/bias
0
Ü0
Ý1"
trackable_list_wrapper
0
Ü0
Ý1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
Þ	variables
ßtrainable_variables
àregularization_losses
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_595_layer_call_fn_667288¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_595_layer_call_and_return_conditional_losses_667298¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)E2batch_normalization_540/gamma
*:(E2batch_normalization_540/beta
3:1E (2#batch_normalization_540/moving_mean
7:5E (2'batch_normalization_540/moving_variance
@
å0
æ1
ç2
è3"
trackable_list_wrapper
0
å0
æ1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
é	variables
êtrainable_variables
ëregularization_losses
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_540_layer_call_fn_667311
8__inference_batch_normalization_540_layer_call_fn_667324´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_667344
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_667378´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_540_layer_call_fn_667383¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_leaky_re_lu_540_layer_call_and_return_conditional_losses_667388¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": E2dense_596/kernel
:2dense_596/bias
0
õ0
ö1"
trackable_list_wrapper
0
õ0
ö1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
÷	variables
øtrainable_variables
ùregularization_losses
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_596_layer_call_fn_667397¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_596_layer_call_and_return_conditional_losses_667407¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
¸
(0
)1
*2
83
94
Q5
R6
j7
k8
9
10
11
12
µ13
¶14
Î15
Ï16
ç17
è18"
trackable_list_wrapper
æ
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
19
20
21
22
23
24
25"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
$__inference_signature_wrapper_666469normalization_55_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
80
91"
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
Q0
R1"
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
j0
k1"
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
0
1"
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
0
1"
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
µ0
¶1"
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
Î0
Ï1"
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
ç0
è1"
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

total

count
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
':%Y2Adam/dense_588/kernel/m
!:Y2Adam/dense_588/bias/m
0:.Y2$Adam/batch_normalization_533/gamma/m
/:-Y2#Adam/batch_normalization_533/beta/m
':%YY2Adam/dense_589/kernel/m
!:Y2Adam/dense_589/bias/m
0:.Y2$Adam/batch_normalization_534/gamma/m
/:-Y2#Adam/batch_normalization_534/beta/m
':%Y2Adam/dense_590/kernel/m
!:2Adam/dense_590/bias/m
0:.2$Adam/batch_normalization_535/gamma/m
/:-2#Adam/batch_normalization_535/beta/m
':%2Adam/dense_591/kernel/m
!:2Adam/dense_591/bias/m
0:.2$Adam/batch_normalization_536/gamma/m
/:-2#Adam/batch_normalization_536/beta/m
':%2Adam/dense_592/kernel/m
!:2Adam/dense_592/bias/m
0:.2$Adam/batch_normalization_537/gamma/m
/:-2#Adam/batch_normalization_537/beta/m
':%2Adam/dense_593/kernel/m
!:2Adam/dense_593/bias/m
0:.2$Adam/batch_normalization_538/gamma/m
/:-2#Adam/batch_normalization_538/beta/m
':%E2Adam/dense_594/kernel/m
!:E2Adam/dense_594/bias/m
0:.E2$Adam/batch_normalization_539/gamma/m
/:-E2#Adam/batch_normalization_539/beta/m
':%EE2Adam/dense_595/kernel/m
!:E2Adam/dense_595/bias/m
0:.E2$Adam/batch_normalization_540/gamma/m
/:-E2#Adam/batch_normalization_540/beta/m
':%E2Adam/dense_596/kernel/m
!:2Adam/dense_596/bias/m
':%Y2Adam/dense_588/kernel/v
!:Y2Adam/dense_588/bias/v
0:.Y2$Adam/batch_normalization_533/gamma/v
/:-Y2#Adam/batch_normalization_533/beta/v
':%YY2Adam/dense_589/kernel/v
!:Y2Adam/dense_589/bias/v
0:.Y2$Adam/batch_normalization_534/gamma/v
/:-Y2#Adam/batch_normalization_534/beta/v
':%Y2Adam/dense_590/kernel/v
!:2Adam/dense_590/bias/v
0:.2$Adam/batch_normalization_535/gamma/v
/:-2#Adam/batch_normalization_535/beta/v
':%2Adam/dense_591/kernel/v
!:2Adam/dense_591/bias/v
0:.2$Adam/batch_normalization_536/gamma/v
/:-2#Adam/batch_normalization_536/beta/v
':%2Adam/dense_592/kernel/v
!:2Adam/dense_592/bias/v
0:.2$Adam/batch_normalization_537/gamma/v
/:-2#Adam/batch_normalization_537/beta/v
':%2Adam/dense_593/kernel/v
!:2Adam/dense_593/bias/v
0:.2$Adam/batch_normalization_538/gamma/v
/:-2#Adam/batch_normalization_538/beta/v
':%E2Adam/dense_594/kernel/v
!:E2Adam/dense_594/bias/v
0:.E2$Adam/batch_normalization_539/gamma/v
/:-E2#Adam/batch_normalization_539/beta/v
':%EE2Adam/dense_595/kernel/v
!:E2Adam/dense_595/bias/v
0:.E2$Adam/batch_normalization_540/gamma/v
/:-E2#Adam/batch_normalization_540/beta/v
':%E2Adam/dense_596/kernel/v
!:2Adam/dense_596/bias/v
	J
Const
J	
Const_1ô
!__inference__wrapped_model_663699ÎTÍÎ-.9687FGROQP_`khjixyª«¶³µ´ÃÄÏÌÎÍÜÝèåçæõö?¢<
5¢2
0-
normalization_55_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_596# 
	dense_596ÿÿÿÿÿÿÿÿÿf
__inference_adapt_step_666516E*():¢7
0¢-
+(¢
 	IteratorSpec 
ª "
 ¹
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_666581b96873¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿY
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿY
 ¹
S__inference_batch_normalization_533_layer_call_and_return_conditional_losses_666615b89673¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿY
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿY
 
8__inference_batch_normalization_533_layer_call_fn_666548U96873¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿY
p 
ª "ÿÿÿÿÿÿÿÿÿY
8__inference_batch_normalization_533_layer_call_fn_666561U89673¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿY
p
ª "ÿÿÿÿÿÿÿÿÿY¹
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_666690bROQP3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿY
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿY
 ¹
S__inference_batch_normalization_534_layer_call_and_return_conditional_losses_666724bQROP3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿY
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿY
 
8__inference_batch_normalization_534_layer_call_fn_666657UROQP3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿY
p 
ª "ÿÿÿÿÿÿÿÿÿY
8__inference_batch_normalization_534_layer_call_fn_666670UQROP3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿY
p
ª "ÿÿÿÿÿÿÿÿÿY¹
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_666799bkhji3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¹
S__inference_batch_normalization_535_layer_call_and_return_conditional_losses_666833bjkhi3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_535_layer_call_fn_666766Ukhji3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_535_layer_call_fn_666779Ujkhi3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ½
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_666908f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
S__inference_batch_normalization_536_layer_call_and_return_conditional_losses_666942f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_536_layer_call_fn_666875Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_536_layer_call_fn_666888Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ½
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_667017f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
S__inference_batch_normalization_537_layer_call_and_return_conditional_losses_667051f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_537_layer_call_fn_666984Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_537_layer_call_fn_666997Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ½
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_667126f¶³µ´3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
S__inference_batch_normalization_538_layer_call_and_return_conditional_losses_667160fµ¶³´3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_538_layer_call_fn_667093Y¶³µ´3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_538_layer_call_fn_667106Yµ¶³´3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ½
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_667235fÏÌÎÍ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿE
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿE
 ½
S__inference_batch_normalization_539_layer_call_and_return_conditional_losses_667269fÎÏÌÍ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿE
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿE
 
8__inference_batch_normalization_539_layer_call_fn_667202YÏÌÎÍ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿE
p 
ª "ÿÿÿÿÿÿÿÿÿE
8__inference_batch_normalization_539_layer_call_fn_667215YÎÏÌÍ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿE
p
ª "ÿÿÿÿÿÿÿÿÿE½
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_667344fèåçæ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿE
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿE
 ½
S__inference_batch_normalization_540_layer_call_and_return_conditional_losses_667378fçèåæ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿE
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿE
 
8__inference_batch_normalization_540_layer_call_fn_667311Yèåçæ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿE
p 
ª "ÿÿÿÿÿÿÿÿÿE
8__inference_batch_normalization_540_layer_call_fn_667324Yçèåæ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿE
p
ª "ÿÿÿÿÿÿÿÿÿE¥
E__inference_dense_588_layer_call_and_return_conditional_losses_666535\-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿY
 }
*__inference_dense_588_layer_call_fn_666525O-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿY¥
E__inference_dense_589_layer_call_and_return_conditional_losses_666644\FG/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿY
ª "%¢"

0ÿÿÿÿÿÿÿÿÿY
 }
*__inference_dense_589_layer_call_fn_666634OFG/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿY
ª "ÿÿÿÿÿÿÿÿÿY¥
E__inference_dense_590_layer_call_and_return_conditional_losses_666753\_`/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿY
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_590_layer_call_fn_666743O_`/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿY
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_591_layer_call_and_return_conditional_losses_666862\xy/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_591_layer_call_fn_666852Oxy/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_592_layer_call_and_return_conditional_losses_666971^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_592_layer_call_fn_666961Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_593_layer_call_and_return_conditional_losses_667080^ª«/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_593_layer_call_fn_667070Qª«/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_594_layer_call_and_return_conditional_losses_667189^ÃÄ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿE
 
*__inference_dense_594_layer_call_fn_667179QÃÄ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿE§
E__inference_dense_595_layer_call_and_return_conditional_losses_667298^ÜÝ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿE
ª "%¢"

0ÿÿÿÿÿÿÿÿÿE
 
*__inference_dense_595_layer_call_fn_667288QÜÝ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿE
ª "ÿÿÿÿÿÿÿÿÿE§
E__inference_dense_596_layer_call_and_return_conditional_losses_667407^õö/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿE
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_596_layer_call_fn_667397Qõö/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿE
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_533_layer_call_and_return_conditional_losses_666625X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿY
ª "%¢"

0ÿÿÿÿÿÿÿÿÿY
 
0__inference_leaky_re_lu_533_layer_call_fn_666620K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿY
ª "ÿÿÿÿÿÿÿÿÿY§
K__inference_leaky_re_lu_534_layer_call_and_return_conditional_losses_666734X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿY
ª "%¢"

0ÿÿÿÿÿÿÿÿÿY
 
0__inference_leaky_re_lu_534_layer_call_fn_666729K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿY
ª "ÿÿÿÿÿÿÿÿÿY§
K__inference_leaky_re_lu_535_layer_call_and_return_conditional_losses_666843X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_leaky_re_lu_535_layer_call_fn_666838K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_536_layer_call_and_return_conditional_losses_666952X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_leaky_re_lu_536_layer_call_fn_666947K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_537_layer_call_and_return_conditional_losses_667061X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_leaky_re_lu_537_layer_call_fn_667056K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_538_layer_call_and_return_conditional_losses_667170X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_leaky_re_lu_538_layer_call_fn_667165K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_539_layer_call_and_return_conditional_losses_667279X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿE
ª "%¢"

0ÿÿÿÿÿÿÿÿÿE
 
0__inference_leaky_re_lu_539_layer_call_fn_667274K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿE
ª "ÿÿÿÿÿÿÿÿÿE§
K__inference_leaky_re_lu_540_layer_call_and_return_conditional_losses_667388X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿE
ª "%¢"

0ÿÿÿÿÿÿÿÿÿE
 
0__inference_leaky_re_lu_540_layer_call_fn_667383K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿE
ª "ÿÿÿÿÿÿÿÿÿE
I__inference_sequential_55_layer_call_and_return_conditional_losses_665486ÆTÍÎ-.9687FGROQP_`khjixyª«¶³µ´ÃÄÏÌÎÍÜÝèåçæõöG¢D
=¢:
0-
normalization_55_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
I__inference_sequential_55_layer_call_and_return_conditional_losses_665622ÆTÍÎ-.8967FGQROP_`jkhixyª«µ¶³´ÃÄÎÏÌÍÜÝçèåæõöG¢D
=¢:
0-
normalization_55_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
I__inference_sequential_55_layer_call_and_return_conditional_losses_666045¶TÍÎ-.9687FGROQP_`khjixyª«¶³µ´ÃÄÏÌÎÍÜÝèåçæõö7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
I__inference_sequential_55_layer_call_and_return_conditional_losses_666358¶TÍÎ-.8967FGQROP_`jkhixyª«µ¶³´ÃÄÎÏÌÍÜÝçèåæõö7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ì
.__inference_sequential_55_layer_call_fn_664749¹TÍÎ-.9687FGROQP_`khjixyª«¶³µ´ÃÄÏÌÎÍÜÝèåçæõöG¢D
=¢:
0-
normalization_55_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿì
.__inference_sequential_55_layer_call_fn_665350¹TÍÎ-.8967FGQROP_`jkhixyª«µ¶³´ÃÄÎÏÌÍÜÝçèåæõöG¢D
=¢:
0-
normalization_55_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÜ
.__inference_sequential_55_layer_call_fn_665735©TÍÎ-.9687FGROQP_`khjixyª«¶³µ´ÃÄÏÌÎÍÜÝèåçæõö7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÜ
.__inference_sequential_55_layer_call_fn_665844©TÍÎ-.8967FGQROP_`jkhixyª«µ¶³´ÃÄÎÏÌÍÜÝçèåæõö7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_signature_wrapper_666469èTÍÎ-.9687FGROQP_`khjixyª«¶³µ´ÃÄÏÌÎÍÜÝèåçæõöY¢V
¢ 
OªL
J
normalization_55_input0-
normalization_55_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_596# 
	dense_596ÿÿÿÿÿÿÿÿÿ