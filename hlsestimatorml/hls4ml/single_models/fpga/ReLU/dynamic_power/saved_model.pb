ñ·,
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68¸«(
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
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
dense_552/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*!
shared_namedense_552/kernel
u
$dense_552/kernel/Read/ReadVariableOpReadVariableOpdense_552/kernel*
_output_shapes

:P*
dtype0
t
dense_552/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense_552/bias
m
"dense_552/bias/Read/ReadVariableOpReadVariableOpdense_552/bias*
_output_shapes
:P*
dtype0

batch_normalization_499/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*.
shared_namebatch_normalization_499/gamma

1batch_normalization_499/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_499/gamma*
_output_shapes
:P*
dtype0

batch_normalization_499/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*-
shared_namebatch_normalization_499/beta

0batch_normalization_499/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_499/beta*
_output_shapes
:P*
dtype0

#batch_normalization_499/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*4
shared_name%#batch_normalization_499/moving_mean

7batch_normalization_499/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_499/moving_mean*
_output_shapes
:P*
dtype0
¦
'batch_normalization_499/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*8
shared_name)'batch_normalization_499/moving_variance

;batch_normalization_499/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_499/moving_variance*
_output_shapes
:P*
dtype0
|
dense_553/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*!
shared_namedense_553/kernel
u
$dense_553/kernel/Read/ReadVariableOpReadVariableOpdense_553/kernel*
_output_shapes

:PP*
dtype0
t
dense_553/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense_553/bias
m
"dense_553/bias/Read/ReadVariableOpReadVariableOpdense_553/bias*
_output_shapes
:P*
dtype0

batch_normalization_500/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*.
shared_namebatch_normalization_500/gamma

1batch_normalization_500/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_500/gamma*
_output_shapes
:P*
dtype0

batch_normalization_500/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*-
shared_namebatch_normalization_500/beta

0batch_normalization_500/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_500/beta*
_output_shapes
:P*
dtype0

#batch_normalization_500/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*4
shared_name%#batch_normalization_500/moving_mean

7batch_normalization_500/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_500/moving_mean*
_output_shapes
:P*
dtype0
¦
'batch_normalization_500/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*8
shared_name)'batch_normalization_500/moving_variance

;batch_normalization_500/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_500/moving_variance*
_output_shapes
:P*
dtype0
|
dense_554/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pa*!
shared_namedense_554/kernel
u
$dense_554/kernel/Read/ReadVariableOpReadVariableOpdense_554/kernel*
_output_shapes

:Pa*
dtype0
t
dense_554/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*
shared_namedense_554/bias
m
"dense_554/bias/Read/ReadVariableOpReadVariableOpdense_554/bias*
_output_shapes
:a*
dtype0

batch_normalization_501/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*.
shared_namebatch_normalization_501/gamma

1batch_normalization_501/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_501/gamma*
_output_shapes
:a*
dtype0

batch_normalization_501/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*-
shared_namebatch_normalization_501/beta

0batch_normalization_501/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_501/beta*
_output_shapes
:a*
dtype0

#batch_normalization_501/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#batch_normalization_501/moving_mean

7batch_normalization_501/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_501/moving_mean*
_output_shapes
:a*
dtype0
¦
'batch_normalization_501/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*8
shared_name)'batch_normalization_501/moving_variance

;batch_normalization_501/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_501/moving_variance*
_output_shapes
:a*
dtype0
|
dense_555/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aa*!
shared_namedense_555/kernel
u
$dense_555/kernel/Read/ReadVariableOpReadVariableOpdense_555/kernel*
_output_shapes

:aa*
dtype0
t
dense_555/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*
shared_namedense_555/bias
m
"dense_555/bias/Read/ReadVariableOpReadVariableOpdense_555/bias*
_output_shapes
:a*
dtype0

batch_normalization_502/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*.
shared_namebatch_normalization_502/gamma

1batch_normalization_502/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_502/gamma*
_output_shapes
:a*
dtype0

batch_normalization_502/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*-
shared_namebatch_normalization_502/beta

0batch_normalization_502/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_502/beta*
_output_shapes
:a*
dtype0

#batch_normalization_502/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#batch_normalization_502/moving_mean

7batch_normalization_502/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_502/moving_mean*
_output_shapes
:a*
dtype0
¦
'batch_normalization_502/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*8
shared_name)'batch_normalization_502/moving_variance

;batch_normalization_502/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_502/moving_variance*
_output_shapes
:a*
dtype0
|
dense_556/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aa*!
shared_namedense_556/kernel
u
$dense_556/kernel/Read/ReadVariableOpReadVariableOpdense_556/kernel*
_output_shapes

:aa*
dtype0
t
dense_556/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*
shared_namedense_556/bias
m
"dense_556/bias/Read/ReadVariableOpReadVariableOpdense_556/bias*
_output_shapes
:a*
dtype0

batch_normalization_503/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*.
shared_namebatch_normalization_503/gamma

1batch_normalization_503/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_503/gamma*
_output_shapes
:a*
dtype0

batch_normalization_503/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*-
shared_namebatch_normalization_503/beta

0batch_normalization_503/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_503/beta*
_output_shapes
:a*
dtype0

#batch_normalization_503/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#batch_normalization_503/moving_mean

7batch_normalization_503/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_503/moving_mean*
_output_shapes
:a*
dtype0
¦
'batch_normalization_503/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*8
shared_name)'batch_normalization_503/moving_variance

;batch_normalization_503/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_503/moving_variance*
_output_shapes
:a*
dtype0
|
dense_557/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aa*!
shared_namedense_557/kernel
u
$dense_557/kernel/Read/ReadVariableOpReadVariableOpdense_557/kernel*
_output_shapes

:aa*
dtype0
t
dense_557/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*
shared_namedense_557/bias
m
"dense_557/bias/Read/ReadVariableOpReadVariableOpdense_557/bias*
_output_shapes
:a*
dtype0

batch_normalization_504/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*.
shared_namebatch_normalization_504/gamma

1batch_normalization_504/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_504/gamma*
_output_shapes
:a*
dtype0

batch_normalization_504/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*-
shared_namebatch_normalization_504/beta

0batch_normalization_504/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_504/beta*
_output_shapes
:a*
dtype0

#batch_normalization_504/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#batch_normalization_504/moving_mean

7batch_normalization_504/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_504/moving_mean*
_output_shapes
:a*
dtype0
¦
'batch_normalization_504/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*8
shared_name)'batch_normalization_504/moving_variance

;batch_normalization_504/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_504/moving_variance*
_output_shapes
:a*
dtype0
|
dense_558/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:a=*!
shared_namedense_558/kernel
u
$dense_558/kernel/Read/ReadVariableOpReadVariableOpdense_558/kernel*
_output_shapes

:a=*
dtype0
t
dense_558/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*
shared_namedense_558/bias
m
"dense_558/bias/Read/ReadVariableOpReadVariableOpdense_558/bias*
_output_shapes
:=*
dtype0

batch_normalization_505/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*.
shared_namebatch_normalization_505/gamma

1batch_normalization_505/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_505/gamma*
_output_shapes
:=*
dtype0

batch_normalization_505/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*-
shared_namebatch_normalization_505/beta

0batch_normalization_505/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_505/beta*
_output_shapes
:=*
dtype0

#batch_normalization_505/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#batch_normalization_505/moving_mean

7batch_normalization_505/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_505/moving_mean*
_output_shapes
:=*
dtype0
¦
'batch_normalization_505/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*8
shared_name)'batch_normalization_505/moving_variance

;batch_normalization_505/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_505/moving_variance*
_output_shapes
:=*
dtype0
|
dense_559/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:==*!
shared_namedense_559/kernel
u
$dense_559/kernel/Read/ReadVariableOpReadVariableOpdense_559/kernel*
_output_shapes

:==*
dtype0
t
dense_559/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*
shared_namedense_559/bias
m
"dense_559/bias/Read/ReadVariableOpReadVariableOpdense_559/bias*
_output_shapes
:=*
dtype0

batch_normalization_506/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*.
shared_namebatch_normalization_506/gamma

1batch_normalization_506/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_506/gamma*
_output_shapes
:=*
dtype0

batch_normalization_506/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*-
shared_namebatch_normalization_506/beta

0batch_normalization_506/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_506/beta*
_output_shapes
:=*
dtype0

#batch_normalization_506/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#batch_normalization_506/moving_mean

7batch_normalization_506/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_506/moving_mean*
_output_shapes
:=*
dtype0
¦
'batch_normalization_506/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*8
shared_name)'batch_normalization_506/moving_variance

;batch_normalization_506/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_506/moving_variance*
_output_shapes
:=*
dtype0
|
dense_560/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:==*!
shared_namedense_560/kernel
u
$dense_560/kernel/Read/ReadVariableOpReadVariableOpdense_560/kernel*
_output_shapes

:==*
dtype0
t
dense_560/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*
shared_namedense_560/bias
m
"dense_560/bias/Read/ReadVariableOpReadVariableOpdense_560/bias*
_output_shapes
:=*
dtype0

batch_normalization_507/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*.
shared_namebatch_normalization_507/gamma

1batch_normalization_507/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_507/gamma*
_output_shapes
:=*
dtype0

batch_normalization_507/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*-
shared_namebatch_normalization_507/beta

0batch_normalization_507/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_507/beta*
_output_shapes
:=*
dtype0

#batch_normalization_507/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#batch_normalization_507/moving_mean

7batch_normalization_507/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_507/moving_mean*
_output_shapes
:=*
dtype0
¦
'batch_normalization_507/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*8
shared_name)'batch_normalization_507/moving_variance

;batch_normalization_507/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_507/moving_variance*
_output_shapes
:=*
dtype0
|
dense_561/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:=*!
shared_namedense_561/kernel
u
$dense_561/kernel/Read/ReadVariableOpReadVariableOpdense_561/kernel*
_output_shapes

:=*
dtype0
t
dense_561/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_561/bias
m
"dense_561/bias/Read/ReadVariableOpReadVariableOpdense_561/bias*
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
Adam/dense_552/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*(
shared_nameAdam/dense_552/kernel/m

+Adam/dense_552/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_552/kernel/m*
_output_shapes

:P*
dtype0

Adam/dense_552/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/dense_552/bias/m
{
)Adam/dense_552/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_552/bias/m*
_output_shapes
:P*
dtype0
 
$Adam/batch_normalization_499/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*5
shared_name&$Adam/batch_normalization_499/gamma/m

8Adam/batch_normalization_499/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_499/gamma/m*
_output_shapes
:P*
dtype0

#Adam/batch_normalization_499/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*4
shared_name%#Adam/batch_normalization_499/beta/m

7Adam/batch_normalization_499/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_499/beta/m*
_output_shapes
:P*
dtype0

Adam/dense_553/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*(
shared_nameAdam/dense_553/kernel/m

+Adam/dense_553/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_553/kernel/m*
_output_shapes

:PP*
dtype0

Adam/dense_553/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/dense_553/bias/m
{
)Adam/dense_553/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_553/bias/m*
_output_shapes
:P*
dtype0
 
$Adam/batch_normalization_500/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*5
shared_name&$Adam/batch_normalization_500/gamma/m

8Adam/batch_normalization_500/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_500/gamma/m*
_output_shapes
:P*
dtype0

#Adam/batch_normalization_500/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*4
shared_name%#Adam/batch_normalization_500/beta/m

7Adam/batch_normalization_500/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_500/beta/m*
_output_shapes
:P*
dtype0

Adam/dense_554/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pa*(
shared_nameAdam/dense_554/kernel/m

+Adam/dense_554/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_554/kernel/m*
_output_shapes

:Pa*
dtype0

Adam/dense_554/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*&
shared_nameAdam/dense_554/bias/m
{
)Adam/dense_554/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_554/bias/m*
_output_shapes
:a*
dtype0
 
$Adam/batch_normalization_501/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*5
shared_name&$Adam/batch_normalization_501/gamma/m

8Adam/batch_normalization_501/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_501/gamma/m*
_output_shapes
:a*
dtype0

#Adam/batch_normalization_501/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#Adam/batch_normalization_501/beta/m

7Adam/batch_normalization_501/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_501/beta/m*
_output_shapes
:a*
dtype0

Adam/dense_555/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aa*(
shared_nameAdam/dense_555/kernel/m

+Adam/dense_555/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_555/kernel/m*
_output_shapes

:aa*
dtype0

Adam/dense_555/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*&
shared_nameAdam/dense_555/bias/m
{
)Adam/dense_555/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_555/bias/m*
_output_shapes
:a*
dtype0
 
$Adam/batch_normalization_502/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*5
shared_name&$Adam/batch_normalization_502/gamma/m

8Adam/batch_normalization_502/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_502/gamma/m*
_output_shapes
:a*
dtype0

#Adam/batch_normalization_502/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#Adam/batch_normalization_502/beta/m

7Adam/batch_normalization_502/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_502/beta/m*
_output_shapes
:a*
dtype0

Adam/dense_556/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aa*(
shared_nameAdam/dense_556/kernel/m

+Adam/dense_556/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_556/kernel/m*
_output_shapes

:aa*
dtype0

Adam/dense_556/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*&
shared_nameAdam/dense_556/bias/m
{
)Adam/dense_556/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_556/bias/m*
_output_shapes
:a*
dtype0
 
$Adam/batch_normalization_503/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*5
shared_name&$Adam/batch_normalization_503/gamma/m

8Adam/batch_normalization_503/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_503/gamma/m*
_output_shapes
:a*
dtype0

#Adam/batch_normalization_503/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#Adam/batch_normalization_503/beta/m

7Adam/batch_normalization_503/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_503/beta/m*
_output_shapes
:a*
dtype0

Adam/dense_557/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aa*(
shared_nameAdam/dense_557/kernel/m

+Adam/dense_557/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_557/kernel/m*
_output_shapes

:aa*
dtype0

Adam/dense_557/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*&
shared_nameAdam/dense_557/bias/m
{
)Adam/dense_557/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_557/bias/m*
_output_shapes
:a*
dtype0
 
$Adam/batch_normalization_504/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*5
shared_name&$Adam/batch_normalization_504/gamma/m

8Adam/batch_normalization_504/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_504/gamma/m*
_output_shapes
:a*
dtype0

#Adam/batch_normalization_504/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#Adam/batch_normalization_504/beta/m

7Adam/batch_normalization_504/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_504/beta/m*
_output_shapes
:a*
dtype0

Adam/dense_558/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:a=*(
shared_nameAdam/dense_558/kernel/m

+Adam/dense_558/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_558/kernel/m*
_output_shapes

:a=*
dtype0

Adam/dense_558/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*&
shared_nameAdam/dense_558/bias/m
{
)Adam/dense_558/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_558/bias/m*
_output_shapes
:=*
dtype0
 
$Adam/batch_normalization_505/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*5
shared_name&$Adam/batch_normalization_505/gamma/m

8Adam/batch_normalization_505/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_505/gamma/m*
_output_shapes
:=*
dtype0

#Adam/batch_normalization_505/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#Adam/batch_normalization_505/beta/m

7Adam/batch_normalization_505/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_505/beta/m*
_output_shapes
:=*
dtype0

Adam/dense_559/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:==*(
shared_nameAdam/dense_559/kernel/m

+Adam/dense_559/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_559/kernel/m*
_output_shapes

:==*
dtype0

Adam/dense_559/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*&
shared_nameAdam/dense_559/bias/m
{
)Adam/dense_559/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_559/bias/m*
_output_shapes
:=*
dtype0
 
$Adam/batch_normalization_506/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*5
shared_name&$Adam/batch_normalization_506/gamma/m

8Adam/batch_normalization_506/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_506/gamma/m*
_output_shapes
:=*
dtype0

#Adam/batch_normalization_506/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#Adam/batch_normalization_506/beta/m

7Adam/batch_normalization_506/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_506/beta/m*
_output_shapes
:=*
dtype0

Adam/dense_560/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:==*(
shared_nameAdam/dense_560/kernel/m

+Adam/dense_560/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_560/kernel/m*
_output_shapes

:==*
dtype0

Adam/dense_560/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*&
shared_nameAdam/dense_560/bias/m
{
)Adam/dense_560/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_560/bias/m*
_output_shapes
:=*
dtype0
 
$Adam/batch_normalization_507/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*5
shared_name&$Adam/batch_normalization_507/gamma/m

8Adam/batch_normalization_507/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_507/gamma/m*
_output_shapes
:=*
dtype0

#Adam/batch_normalization_507/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#Adam/batch_normalization_507/beta/m

7Adam/batch_normalization_507/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_507/beta/m*
_output_shapes
:=*
dtype0

Adam/dense_561/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:=*(
shared_nameAdam/dense_561/kernel/m

+Adam/dense_561/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_561/kernel/m*
_output_shapes

:=*
dtype0

Adam/dense_561/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_561/bias/m
{
)Adam/dense_561/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_561/bias/m*
_output_shapes
:*
dtype0

Adam/dense_552/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*(
shared_nameAdam/dense_552/kernel/v

+Adam/dense_552/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_552/kernel/v*
_output_shapes

:P*
dtype0

Adam/dense_552/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/dense_552/bias/v
{
)Adam/dense_552/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_552/bias/v*
_output_shapes
:P*
dtype0
 
$Adam/batch_normalization_499/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*5
shared_name&$Adam/batch_normalization_499/gamma/v

8Adam/batch_normalization_499/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_499/gamma/v*
_output_shapes
:P*
dtype0

#Adam/batch_normalization_499/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*4
shared_name%#Adam/batch_normalization_499/beta/v

7Adam/batch_normalization_499/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_499/beta/v*
_output_shapes
:P*
dtype0

Adam/dense_553/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*(
shared_nameAdam/dense_553/kernel/v

+Adam/dense_553/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_553/kernel/v*
_output_shapes

:PP*
dtype0

Adam/dense_553/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/dense_553/bias/v
{
)Adam/dense_553/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_553/bias/v*
_output_shapes
:P*
dtype0
 
$Adam/batch_normalization_500/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*5
shared_name&$Adam/batch_normalization_500/gamma/v

8Adam/batch_normalization_500/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_500/gamma/v*
_output_shapes
:P*
dtype0

#Adam/batch_normalization_500/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*4
shared_name%#Adam/batch_normalization_500/beta/v

7Adam/batch_normalization_500/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_500/beta/v*
_output_shapes
:P*
dtype0

Adam/dense_554/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pa*(
shared_nameAdam/dense_554/kernel/v

+Adam/dense_554/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_554/kernel/v*
_output_shapes

:Pa*
dtype0

Adam/dense_554/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*&
shared_nameAdam/dense_554/bias/v
{
)Adam/dense_554/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_554/bias/v*
_output_shapes
:a*
dtype0
 
$Adam/batch_normalization_501/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*5
shared_name&$Adam/batch_normalization_501/gamma/v

8Adam/batch_normalization_501/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_501/gamma/v*
_output_shapes
:a*
dtype0

#Adam/batch_normalization_501/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#Adam/batch_normalization_501/beta/v

7Adam/batch_normalization_501/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_501/beta/v*
_output_shapes
:a*
dtype0

Adam/dense_555/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aa*(
shared_nameAdam/dense_555/kernel/v

+Adam/dense_555/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_555/kernel/v*
_output_shapes

:aa*
dtype0

Adam/dense_555/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*&
shared_nameAdam/dense_555/bias/v
{
)Adam/dense_555/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_555/bias/v*
_output_shapes
:a*
dtype0
 
$Adam/batch_normalization_502/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*5
shared_name&$Adam/batch_normalization_502/gamma/v

8Adam/batch_normalization_502/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_502/gamma/v*
_output_shapes
:a*
dtype0

#Adam/batch_normalization_502/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#Adam/batch_normalization_502/beta/v

7Adam/batch_normalization_502/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_502/beta/v*
_output_shapes
:a*
dtype0

Adam/dense_556/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aa*(
shared_nameAdam/dense_556/kernel/v

+Adam/dense_556/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_556/kernel/v*
_output_shapes

:aa*
dtype0

Adam/dense_556/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*&
shared_nameAdam/dense_556/bias/v
{
)Adam/dense_556/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_556/bias/v*
_output_shapes
:a*
dtype0
 
$Adam/batch_normalization_503/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*5
shared_name&$Adam/batch_normalization_503/gamma/v

8Adam/batch_normalization_503/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_503/gamma/v*
_output_shapes
:a*
dtype0

#Adam/batch_normalization_503/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#Adam/batch_normalization_503/beta/v

7Adam/batch_normalization_503/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_503/beta/v*
_output_shapes
:a*
dtype0

Adam/dense_557/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aa*(
shared_nameAdam/dense_557/kernel/v

+Adam/dense_557/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_557/kernel/v*
_output_shapes

:aa*
dtype0

Adam/dense_557/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*&
shared_nameAdam/dense_557/bias/v
{
)Adam/dense_557/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_557/bias/v*
_output_shapes
:a*
dtype0
 
$Adam/batch_normalization_504/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*5
shared_name&$Adam/batch_normalization_504/gamma/v

8Adam/batch_normalization_504/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_504/gamma/v*
_output_shapes
:a*
dtype0

#Adam/batch_normalization_504/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#Adam/batch_normalization_504/beta/v

7Adam/batch_normalization_504/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_504/beta/v*
_output_shapes
:a*
dtype0

Adam/dense_558/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:a=*(
shared_nameAdam/dense_558/kernel/v

+Adam/dense_558/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_558/kernel/v*
_output_shapes

:a=*
dtype0

Adam/dense_558/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*&
shared_nameAdam/dense_558/bias/v
{
)Adam/dense_558/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_558/bias/v*
_output_shapes
:=*
dtype0
 
$Adam/batch_normalization_505/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*5
shared_name&$Adam/batch_normalization_505/gamma/v

8Adam/batch_normalization_505/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_505/gamma/v*
_output_shapes
:=*
dtype0

#Adam/batch_normalization_505/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#Adam/batch_normalization_505/beta/v

7Adam/batch_normalization_505/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_505/beta/v*
_output_shapes
:=*
dtype0

Adam/dense_559/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:==*(
shared_nameAdam/dense_559/kernel/v

+Adam/dense_559/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_559/kernel/v*
_output_shapes

:==*
dtype0

Adam/dense_559/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*&
shared_nameAdam/dense_559/bias/v
{
)Adam/dense_559/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_559/bias/v*
_output_shapes
:=*
dtype0
 
$Adam/batch_normalization_506/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*5
shared_name&$Adam/batch_normalization_506/gamma/v

8Adam/batch_normalization_506/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_506/gamma/v*
_output_shapes
:=*
dtype0

#Adam/batch_normalization_506/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#Adam/batch_normalization_506/beta/v

7Adam/batch_normalization_506/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_506/beta/v*
_output_shapes
:=*
dtype0

Adam/dense_560/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:==*(
shared_nameAdam/dense_560/kernel/v

+Adam/dense_560/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_560/kernel/v*
_output_shapes

:==*
dtype0

Adam/dense_560/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*&
shared_nameAdam/dense_560/bias/v
{
)Adam/dense_560/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_560/bias/v*
_output_shapes
:=*
dtype0
 
$Adam/batch_normalization_507/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*5
shared_name&$Adam/batch_normalization_507/gamma/v

8Adam/batch_normalization_507/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_507/gamma/v*
_output_shapes
:=*
dtype0

#Adam/batch_normalization_507/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#Adam/batch_normalization_507/beta/v

7Adam/batch_normalization_507/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_507/beta/v*
_output_shapes
:=*
dtype0

Adam/dense_561/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:=*(
shared_nameAdam/dense_561/kernel/v

+Adam/dense_561/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_561/kernel/v*
_output_shapes

:=*
dtype0

Adam/dense_561/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_561/bias/v
{
)Adam/dense_561/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_561/bias/v*
_output_shapes
:*
dtype0
b
ConstConst*
_output_shapes

:*
dtype0*%
valueB"VUéBc'B  DA
d
Const_1Const*
_output_shapes

:*
dtype0*%
valueB"5sEsÍvE ÀB

NoOpNoOp

Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*Í
valueÂB¾ B¶
ê
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
layer_with_weights-18
layer-26
layer-27
layer_with_weights-19
layer-28
	optimizer
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_default_save_signature
&
signatures*
¾
'
_keep_axis
(_reduce_axis
)_reduce_axis_mask
*_broadcast_shape
+mean
+
adapt_mean
,variance
,adapt_variance
	-count
.	keras_api
/_adapt_function*
¦

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses*
Õ
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*

C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 
¦

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses*
Õ
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses*

\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses* 
¦

bkernel
cbias
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses*
Õ
jaxis
	kgamma
lbeta
mmoving_mean
nmoving_variance
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses*

u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses* 
©

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
 moving_variance
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses*

§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses* 
®
­kernel
	®bias
¯	variables
°trainable_variables
±regularization_losses
²	keras_api
³__call__
+´&call_and_return_all_conditional_losses*
à
	µaxis

¶gamma
	·beta
¸moving_mean
¹moving_variance
º	variables
»trainable_variables
¼regularization_losses
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses*

À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses* 
®
Ækernel
	Çbias
È	variables
Étrainable_variables
Êregularization_losses
Ë	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses*
à
	Îaxis

Ïgamma
	Ðbeta
Ñmoving_mean
Òmoving_variance
Ó	variables
Ôtrainable_variables
Õregularization_losses
Ö	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses*

Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses* 
®
ßkernel
	àbias
á	variables
âtrainable_variables
ãregularization_losses
ä	keras_api
å__call__
+æ&call_and_return_all_conditional_losses*
à
	çaxis

ègamma
	ébeta
êmoving_mean
ëmoving_variance
ì	variables
ítrainable_variables
îregularization_losses
ï	keras_api
ð__call__
+ñ&call_and_return_all_conditional_losses*

ò	variables
ótrainable_variables
ôregularization_losses
õ	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses* 
®
økernel
	ùbias
ú	variables
ûtrainable_variables
üregularization_losses
ý	keras_api
þ__call__
+ÿ&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
Ý
	iter
beta_1
beta_2

decay0m´1mµ9m¶:m·Im¸Jm¹RmºSm»bm¼cm½km¾lm¿{mÀ|mÁ	mÂ	mÃ	mÄ	mÅ	mÆ	mÇ	­mÈ	®mÉ	¶mÊ	·mË	ÆmÌ	ÇmÍ	ÏmÎ	ÐmÏ	ßmÐ	àmÑ	èmÒ	émÓ	ømÔ	ùmÕ	mÖ	m×	mØ	mÙ0vÚ1vÛ9vÜ:vÝIvÞJvßRvàSvábvâcvãkvälvå{væ|vç	vè	vé	vê	vë	vì	ví	­vî	®vï	¶vð	·vñ	Ævò	Çvó	Ïvô	Ðvõ	ßvö	àv÷	èvø	évù	øvú	ùvû	vü	vý	vþ	vÿ*
ö
+0
,1
-2
03
14
95
:6
;7
<8
I9
J10
R11
S12
T13
U14
b15
c16
k17
l18
m19
n20
{21
|22
23
24
25
26
27
28
29
30
31
 32
­33
®34
¶35
·36
¸37
¹38
Æ39
Ç40
Ï41
Ð42
Ñ43
Ò44
ß45
à46
è47
é48
ê49
ë50
ø51
ù52
53
54
55
56
57
58*
Â
00
11
92
:3
I4
J5
R6
S7
b8
c9
k10
l11
{12
|13
14
15
16
17
18
19
­20
®21
¶22
·23
Æ24
Ç25
Ï26
Ð27
ß28
à29
è30
é31
ø32
ù33
34
35
36
37*
* 
µ
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
%_default_save_signature
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
* 

¢serving_default* 
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
VARIABLE_VALUEdense_552/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_552/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

00
11*

00
11*
* 

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_499/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_499/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_499/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_499/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
90
:1
;2
<3*

90
:1*
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_553/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_553/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

I0
J1*

I0
J1*
* 

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_500/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_500/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_500/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_500/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
R0
S1
T2
U3*

R0
S1*
* 

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_554/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_554/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

b0
c1*

b0
c1*
* 

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_501/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_501/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_501/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_501/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
k0
l1
m2
n3*

k0
l1*
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_555/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_555/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

{0
|1*

{0
|1*
* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_502/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_502/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_502/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_502/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_556/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_556/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_503/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_503/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_503/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_503/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
 3*

0
1*
* 

änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_557/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_557/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

­0
®1*

­0
®1*
* 

înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
¯	variables
°trainable_variables
±regularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_504/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_504/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_504/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_504/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
¶0
·1
¸2
¹3*

¶0
·1*
* 

ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ønon_trainable_variables
ùlayers
úmetrics
 ûlayer_regularization_losses
ülayer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_558/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_558/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

Æ0
Ç1*

Æ0
Ç1*
* 

ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
È	variables
Étrainable_variables
Êregularization_losses
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_505/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_505/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_505/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_505/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
Ï0
Ð1
Ñ2
Ò3*

Ï0
Ð1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ó	variables
Ôtrainable_variables
Õregularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ù	variables
Útrainable_variables
Ûregularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_559/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_559/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

ß0
à1*

ß0
à1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
á	variables
âtrainable_variables
ãregularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_506/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_506/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_506/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_506/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
è0
é1
ê2
ë3*

è0
é1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ì	variables
ítrainable_variables
îregularization_losses
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ò	variables
ótrainable_variables
ôregularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_560/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_560/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*

ø0
ù1*

ø0
ù1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ú	variables
ûtrainable_variables
üregularization_losses
þ__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_507/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_507/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_507/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_507/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_561/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_561/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
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
®
+0
,1
-2
;3
<4
T5
U6
m7
n8
9
10
11
 12
¸13
¹14
Ñ15
Ò16
ê17
ë18
19
20*
â
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
25
26
27
28*

¯0*
* 
* 
* 
* 
* 
* 
* 
* 

;0
<1*
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
T0
U1*
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
m0
n1*
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
0
1*
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
0
 1*
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
¸0
¹1*
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
Ñ0
Ò1*
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
ê0
ë1*
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
0
1*
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

°total

±count
²	variables
³	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

°0
±1*

²	variables*
}
VARIABLE_VALUEAdam/dense_552/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_552/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_499/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_499/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_553/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_553/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_500/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_500/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_554/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_554/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_501/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_501/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_555/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_555/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_502/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_502/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_556/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_556/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_503/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_503/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_557/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_557/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_504/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_504/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_558/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_558/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_505/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_505/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_559/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_559/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_506/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_506/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_560/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_560/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_507/gamma/mRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_507/beta/mQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_561/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_561/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_552/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_552/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_499/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_499/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_553/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_553/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_500/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_500/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_554/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_554/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_501/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_501/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_555/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_555/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_502/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_502/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_556/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_556/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_503/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_503/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_557/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_557/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_504/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_504/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_558/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_558/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_505/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_505/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_559/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_559/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_506/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_506/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_560/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_560/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_507/gamma/vRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_507/beta/vQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_561/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_561/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

&serving_default_normalization_53_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ú
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_53_inputConstConst_1dense_552/kerneldense_552/bias'batch_normalization_499/moving_variancebatch_normalization_499/gamma#batch_normalization_499/moving_meanbatch_normalization_499/betadense_553/kerneldense_553/bias'batch_normalization_500/moving_variancebatch_normalization_500/gamma#batch_normalization_500/moving_meanbatch_normalization_500/betadense_554/kerneldense_554/bias'batch_normalization_501/moving_variancebatch_normalization_501/gamma#batch_normalization_501/moving_meanbatch_normalization_501/betadense_555/kerneldense_555/bias'batch_normalization_502/moving_variancebatch_normalization_502/gamma#batch_normalization_502/moving_meanbatch_normalization_502/betadense_556/kerneldense_556/bias'batch_normalization_503/moving_variancebatch_normalization_503/gamma#batch_normalization_503/moving_meanbatch_normalization_503/betadense_557/kerneldense_557/bias'batch_normalization_504/moving_variancebatch_normalization_504/gamma#batch_normalization_504/moving_meanbatch_normalization_504/betadense_558/kerneldense_558/bias'batch_normalization_505/moving_variancebatch_normalization_505/gamma#batch_normalization_505/moving_meanbatch_normalization_505/betadense_559/kerneldense_559/bias'batch_normalization_506/moving_variancebatch_normalization_506/gamma#batch_normalization_506/moving_meanbatch_normalization_506/betadense_560/kerneldense_560/bias'batch_normalization_507/moving_variancebatch_normalization_507/gamma#batch_normalization_507/moving_meanbatch_normalization_507/betadense_561/kerneldense_561/bias*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./0123456789:*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_760232
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ç8
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_552/kernel/Read/ReadVariableOp"dense_552/bias/Read/ReadVariableOp1batch_normalization_499/gamma/Read/ReadVariableOp0batch_normalization_499/beta/Read/ReadVariableOp7batch_normalization_499/moving_mean/Read/ReadVariableOp;batch_normalization_499/moving_variance/Read/ReadVariableOp$dense_553/kernel/Read/ReadVariableOp"dense_553/bias/Read/ReadVariableOp1batch_normalization_500/gamma/Read/ReadVariableOp0batch_normalization_500/beta/Read/ReadVariableOp7batch_normalization_500/moving_mean/Read/ReadVariableOp;batch_normalization_500/moving_variance/Read/ReadVariableOp$dense_554/kernel/Read/ReadVariableOp"dense_554/bias/Read/ReadVariableOp1batch_normalization_501/gamma/Read/ReadVariableOp0batch_normalization_501/beta/Read/ReadVariableOp7batch_normalization_501/moving_mean/Read/ReadVariableOp;batch_normalization_501/moving_variance/Read/ReadVariableOp$dense_555/kernel/Read/ReadVariableOp"dense_555/bias/Read/ReadVariableOp1batch_normalization_502/gamma/Read/ReadVariableOp0batch_normalization_502/beta/Read/ReadVariableOp7batch_normalization_502/moving_mean/Read/ReadVariableOp;batch_normalization_502/moving_variance/Read/ReadVariableOp$dense_556/kernel/Read/ReadVariableOp"dense_556/bias/Read/ReadVariableOp1batch_normalization_503/gamma/Read/ReadVariableOp0batch_normalization_503/beta/Read/ReadVariableOp7batch_normalization_503/moving_mean/Read/ReadVariableOp;batch_normalization_503/moving_variance/Read/ReadVariableOp$dense_557/kernel/Read/ReadVariableOp"dense_557/bias/Read/ReadVariableOp1batch_normalization_504/gamma/Read/ReadVariableOp0batch_normalization_504/beta/Read/ReadVariableOp7batch_normalization_504/moving_mean/Read/ReadVariableOp;batch_normalization_504/moving_variance/Read/ReadVariableOp$dense_558/kernel/Read/ReadVariableOp"dense_558/bias/Read/ReadVariableOp1batch_normalization_505/gamma/Read/ReadVariableOp0batch_normalization_505/beta/Read/ReadVariableOp7batch_normalization_505/moving_mean/Read/ReadVariableOp;batch_normalization_505/moving_variance/Read/ReadVariableOp$dense_559/kernel/Read/ReadVariableOp"dense_559/bias/Read/ReadVariableOp1batch_normalization_506/gamma/Read/ReadVariableOp0batch_normalization_506/beta/Read/ReadVariableOp7batch_normalization_506/moving_mean/Read/ReadVariableOp;batch_normalization_506/moving_variance/Read/ReadVariableOp$dense_560/kernel/Read/ReadVariableOp"dense_560/bias/Read/ReadVariableOp1batch_normalization_507/gamma/Read/ReadVariableOp0batch_normalization_507/beta/Read/ReadVariableOp7batch_normalization_507/moving_mean/Read/ReadVariableOp;batch_normalization_507/moving_variance/Read/ReadVariableOp$dense_561/kernel/Read/ReadVariableOp"dense_561/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_552/kernel/m/Read/ReadVariableOp)Adam/dense_552/bias/m/Read/ReadVariableOp8Adam/batch_normalization_499/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_499/beta/m/Read/ReadVariableOp+Adam/dense_553/kernel/m/Read/ReadVariableOp)Adam/dense_553/bias/m/Read/ReadVariableOp8Adam/batch_normalization_500/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_500/beta/m/Read/ReadVariableOp+Adam/dense_554/kernel/m/Read/ReadVariableOp)Adam/dense_554/bias/m/Read/ReadVariableOp8Adam/batch_normalization_501/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_501/beta/m/Read/ReadVariableOp+Adam/dense_555/kernel/m/Read/ReadVariableOp)Adam/dense_555/bias/m/Read/ReadVariableOp8Adam/batch_normalization_502/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_502/beta/m/Read/ReadVariableOp+Adam/dense_556/kernel/m/Read/ReadVariableOp)Adam/dense_556/bias/m/Read/ReadVariableOp8Adam/batch_normalization_503/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_503/beta/m/Read/ReadVariableOp+Adam/dense_557/kernel/m/Read/ReadVariableOp)Adam/dense_557/bias/m/Read/ReadVariableOp8Adam/batch_normalization_504/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_504/beta/m/Read/ReadVariableOp+Adam/dense_558/kernel/m/Read/ReadVariableOp)Adam/dense_558/bias/m/Read/ReadVariableOp8Adam/batch_normalization_505/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_505/beta/m/Read/ReadVariableOp+Adam/dense_559/kernel/m/Read/ReadVariableOp)Adam/dense_559/bias/m/Read/ReadVariableOp8Adam/batch_normalization_506/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_506/beta/m/Read/ReadVariableOp+Adam/dense_560/kernel/m/Read/ReadVariableOp)Adam/dense_560/bias/m/Read/ReadVariableOp8Adam/batch_normalization_507/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_507/beta/m/Read/ReadVariableOp+Adam/dense_561/kernel/m/Read/ReadVariableOp)Adam/dense_561/bias/m/Read/ReadVariableOp+Adam/dense_552/kernel/v/Read/ReadVariableOp)Adam/dense_552/bias/v/Read/ReadVariableOp8Adam/batch_normalization_499/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_499/beta/v/Read/ReadVariableOp+Adam/dense_553/kernel/v/Read/ReadVariableOp)Adam/dense_553/bias/v/Read/ReadVariableOp8Adam/batch_normalization_500/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_500/beta/v/Read/ReadVariableOp+Adam/dense_554/kernel/v/Read/ReadVariableOp)Adam/dense_554/bias/v/Read/ReadVariableOp8Adam/batch_normalization_501/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_501/beta/v/Read/ReadVariableOp+Adam/dense_555/kernel/v/Read/ReadVariableOp)Adam/dense_555/bias/v/Read/ReadVariableOp8Adam/batch_normalization_502/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_502/beta/v/Read/ReadVariableOp+Adam/dense_556/kernel/v/Read/ReadVariableOp)Adam/dense_556/bias/v/Read/ReadVariableOp8Adam/batch_normalization_503/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_503/beta/v/Read/ReadVariableOp+Adam/dense_557/kernel/v/Read/ReadVariableOp)Adam/dense_557/bias/v/Read/ReadVariableOp8Adam/batch_normalization_504/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_504/beta/v/Read/ReadVariableOp+Adam/dense_558/kernel/v/Read/ReadVariableOp)Adam/dense_558/bias/v/Read/ReadVariableOp8Adam/batch_normalization_505/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_505/beta/v/Read/ReadVariableOp+Adam/dense_559/kernel/v/Read/ReadVariableOp)Adam/dense_559/bias/v/Read/ReadVariableOp8Adam/batch_normalization_506/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_506/beta/v/Read/ReadVariableOp+Adam/dense_560/kernel/v/Read/ReadVariableOp)Adam/dense_560/bias/v/Read/ReadVariableOp8Adam/batch_normalization_507/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_507/beta/v/Read/ReadVariableOp+Adam/dense_561/kernel/v/Read/ReadVariableOp)Adam/dense_561/bias/v/Read/ReadVariableOpConst_2*
Tin
2		*
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
__inference__traced_save_761727
¼"
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_552/kerneldense_552/biasbatch_normalization_499/gammabatch_normalization_499/beta#batch_normalization_499/moving_mean'batch_normalization_499/moving_variancedense_553/kerneldense_553/biasbatch_normalization_500/gammabatch_normalization_500/beta#batch_normalization_500/moving_mean'batch_normalization_500/moving_variancedense_554/kerneldense_554/biasbatch_normalization_501/gammabatch_normalization_501/beta#batch_normalization_501/moving_mean'batch_normalization_501/moving_variancedense_555/kerneldense_555/biasbatch_normalization_502/gammabatch_normalization_502/beta#batch_normalization_502/moving_mean'batch_normalization_502/moving_variancedense_556/kerneldense_556/biasbatch_normalization_503/gammabatch_normalization_503/beta#batch_normalization_503/moving_mean'batch_normalization_503/moving_variancedense_557/kerneldense_557/biasbatch_normalization_504/gammabatch_normalization_504/beta#batch_normalization_504/moving_mean'batch_normalization_504/moving_variancedense_558/kerneldense_558/biasbatch_normalization_505/gammabatch_normalization_505/beta#batch_normalization_505/moving_mean'batch_normalization_505/moving_variancedense_559/kerneldense_559/biasbatch_normalization_506/gammabatch_normalization_506/beta#batch_normalization_506/moving_mean'batch_normalization_506/moving_variancedense_560/kerneldense_560/biasbatch_normalization_507/gammabatch_normalization_507/beta#batch_normalization_507/moving_mean'batch_normalization_507/moving_variancedense_561/kerneldense_561/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_552/kernel/mAdam/dense_552/bias/m$Adam/batch_normalization_499/gamma/m#Adam/batch_normalization_499/beta/mAdam/dense_553/kernel/mAdam/dense_553/bias/m$Adam/batch_normalization_500/gamma/m#Adam/batch_normalization_500/beta/mAdam/dense_554/kernel/mAdam/dense_554/bias/m$Adam/batch_normalization_501/gamma/m#Adam/batch_normalization_501/beta/mAdam/dense_555/kernel/mAdam/dense_555/bias/m$Adam/batch_normalization_502/gamma/m#Adam/batch_normalization_502/beta/mAdam/dense_556/kernel/mAdam/dense_556/bias/m$Adam/batch_normalization_503/gamma/m#Adam/batch_normalization_503/beta/mAdam/dense_557/kernel/mAdam/dense_557/bias/m$Adam/batch_normalization_504/gamma/m#Adam/batch_normalization_504/beta/mAdam/dense_558/kernel/mAdam/dense_558/bias/m$Adam/batch_normalization_505/gamma/m#Adam/batch_normalization_505/beta/mAdam/dense_559/kernel/mAdam/dense_559/bias/m$Adam/batch_normalization_506/gamma/m#Adam/batch_normalization_506/beta/mAdam/dense_560/kernel/mAdam/dense_560/bias/m$Adam/batch_normalization_507/gamma/m#Adam/batch_normalization_507/beta/mAdam/dense_561/kernel/mAdam/dense_561/bias/mAdam/dense_552/kernel/vAdam/dense_552/bias/v$Adam/batch_normalization_499/gamma/v#Adam/batch_normalization_499/beta/vAdam/dense_553/kernel/vAdam/dense_553/bias/v$Adam/batch_normalization_500/gamma/v#Adam/batch_normalization_500/beta/vAdam/dense_554/kernel/vAdam/dense_554/bias/v$Adam/batch_normalization_501/gamma/v#Adam/batch_normalization_501/beta/vAdam/dense_555/kernel/vAdam/dense_555/bias/v$Adam/batch_normalization_502/gamma/v#Adam/batch_normalization_502/beta/vAdam/dense_556/kernel/vAdam/dense_556/bias/v$Adam/batch_normalization_503/gamma/v#Adam/batch_normalization_503/beta/vAdam/dense_557/kernel/vAdam/dense_557/bias/v$Adam/batch_normalization_504/gamma/v#Adam/batch_normalization_504/beta/vAdam/dense_558/kernel/vAdam/dense_558/bias/v$Adam/batch_normalization_505/gamma/v#Adam/batch_normalization_505/beta/vAdam/dense_559/kernel/vAdam/dense_559/bias/v$Adam/batch_normalization_506/gamma/v#Adam/batch_normalization_506/beta/vAdam/dense_560/kernel/vAdam/dense_560/bias/v$Adam/batch_normalization_507/gamma/v#Adam/batch_normalization_507/beta/vAdam/dense_561/kernel/vAdam/dense_561/bias/v*
Tin
2*
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
"__inference__traced_restore_762160 è"
Ð
²
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_761107

inputs/
!batchnorm_readvariableop_resource:=3
%batchnorm_mul_readvariableop_resource:=1
#batchnorm_readvariableop_1_resource:=1
#batchnorm_readvariableop_2_resource:=
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:=*
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
:=P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:=~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:=z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:=r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_760487

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
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
:P*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P¬
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
:P*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P´
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_757214

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
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
:P*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P¬
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
:P*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P´
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_504_layer_call_fn_760928

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
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_758085`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿa:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_761260

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ="
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
È	
ö
E__inference_dense_556_layer_call_and_return_conditional_losses_758033

inputs0
matmul_readvariableop_resource:aa-
biasadd_readvariableop_resource:a
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:aa*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿar
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:a*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
È	
ö
E__inference_dense_552_layer_call_and_return_conditional_losses_760298

inputs0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:P
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_501_layer_call_fn_760529

inputs
unknown:a
	unknown_0:a
	unknown_1:a
	unknown_2:a
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_757331o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_757249

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_758149

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ="
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_757542

inputs5
'assignmovingavg_readvariableop_resource:a7
)assignmovingavg_1_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a/
!batchnorm_readvariableop_resource:a
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:a
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿal
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:a*
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
:a*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a¬
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
:a*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:a~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a´
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿah
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_504_layer_call_fn_760856

inputs
unknown:a
	unknown_0:a
	unknown_1:a
	unknown_2:a
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_757577o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
Ä

*__inference_dense_558_layer_call_fn_760942

inputs
unknown:a=
	unknown_0:=
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_558_layer_call_and_return_conditional_losses_758097o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_500_layer_call_and_return_conditional_losses_757957

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿP:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
Ä

*__inference_dense_559_layer_call_fn_761051

inputs
unknown:==
	unknown_0:=
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_559_layer_call_and_return_conditional_losses_758129o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_503_layer_call_fn_760747

inputs
unknown:a
	unknown_0:a
	unknown_1:a
	unknown_2:a
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_757495o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_506_layer_call_fn_761087

inputs
unknown:=
	unknown_0:=
	unknown_1:=
	unknown_2:=
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_757788o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
¥
ë
I__inference_sequential_53_layer_call_and_return_conditional_losses_758200

inputs
normalization_53_sub_y
normalization_53_sqrt_x"
dense_552_757906:P
dense_552_757908:P,
batch_normalization_499_757911:P,
batch_normalization_499_757913:P,
batch_normalization_499_757915:P,
batch_normalization_499_757917:P"
dense_553_757938:PP
dense_553_757940:P,
batch_normalization_500_757943:P,
batch_normalization_500_757945:P,
batch_normalization_500_757947:P,
batch_normalization_500_757949:P"
dense_554_757970:Pa
dense_554_757972:a,
batch_normalization_501_757975:a,
batch_normalization_501_757977:a,
batch_normalization_501_757979:a,
batch_normalization_501_757981:a"
dense_555_758002:aa
dense_555_758004:a,
batch_normalization_502_758007:a,
batch_normalization_502_758009:a,
batch_normalization_502_758011:a,
batch_normalization_502_758013:a"
dense_556_758034:aa
dense_556_758036:a,
batch_normalization_503_758039:a,
batch_normalization_503_758041:a,
batch_normalization_503_758043:a,
batch_normalization_503_758045:a"
dense_557_758066:aa
dense_557_758068:a,
batch_normalization_504_758071:a,
batch_normalization_504_758073:a,
batch_normalization_504_758075:a,
batch_normalization_504_758077:a"
dense_558_758098:a=
dense_558_758100:=,
batch_normalization_505_758103:=,
batch_normalization_505_758105:=,
batch_normalization_505_758107:=,
batch_normalization_505_758109:="
dense_559_758130:==
dense_559_758132:=,
batch_normalization_506_758135:=,
batch_normalization_506_758137:=,
batch_normalization_506_758139:=,
batch_normalization_506_758141:="
dense_560_758162:==
dense_560_758164:=,
batch_normalization_507_758167:=,
batch_normalization_507_758169:=,
batch_normalization_507_758171:=,
batch_normalization_507_758173:="
dense_561_758194:=
dense_561_758196:
identity¢/batch_normalization_499/StatefulPartitionedCall¢/batch_normalization_500/StatefulPartitionedCall¢/batch_normalization_501/StatefulPartitionedCall¢/batch_normalization_502/StatefulPartitionedCall¢/batch_normalization_503/StatefulPartitionedCall¢/batch_normalization_504/StatefulPartitionedCall¢/batch_normalization_505/StatefulPartitionedCall¢/batch_normalization_506/StatefulPartitionedCall¢/batch_normalization_507/StatefulPartitionedCall¢!dense_552/StatefulPartitionedCall¢!dense_553/StatefulPartitionedCall¢!dense_554/StatefulPartitionedCall¢!dense_555/StatefulPartitionedCall¢!dense_556/StatefulPartitionedCall¢!dense_557/StatefulPartitionedCall¢!dense_558/StatefulPartitionedCall¢!dense_559/StatefulPartitionedCall¢!dense_560/StatefulPartitionedCall¢!dense_561/StatefulPartitionedCallm
normalization_53/subSubinputsnormalization_53_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_53/SqrtSqrtnormalization_53_sqrt_x*
T0*
_output_shapes

:_
normalization_53/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_53/MaximumMaximumnormalization_53/Sqrt:y:0#normalization_53/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_53/truedivRealDivnormalization_53/sub:z:0normalization_53/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_552/StatefulPartitionedCallStatefulPartitionedCallnormalization_53/truediv:z:0dense_552_757906dense_552_757908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_552_layer_call_and_return_conditional_losses_757905
/batch_normalization_499/StatefulPartitionedCallStatefulPartitionedCall*dense_552/StatefulPartitionedCall:output:0batch_normalization_499_757911batch_normalization_499_757913batch_normalization_499_757915batch_normalization_499_757917*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_757167ø
leaky_re_lu_499/PartitionedCallPartitionedCall8batch_normalization_499/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_499_layer_call_and_return_conditional_losses_757925
!dense_553/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_499/PartitionedCall:output:0dense_553_757938dense_553_757940*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_553_layer_call_and_return_conditional_losses_757937
/batch_normalization_500/StatefulPartitionedCallStatefulPartitionedCall*dense_553/StatefulPartitionedCall:output:0batch_normalization_500_757943batch_normalization_500_757945batch_normalization_500_757947batch_normalization_500_757949*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_757249ø
leaky_re_lu_500/PartitionedCallPartitionedCall8batch_normalization_500/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_500_layer_call_and_return_conditional_losses_757957
!dense_554/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_500/PartitionedCall:output:0dense_554_757970dense_554_757972*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_554_layer_call_and_return_conditional_losses_757969
/batch_normalization_501/StatefulPartitionedCallStatefulPartitionedCall*dense_554/StatefulPartitionedCall:output:0batch_normalization_501_757975batch_normalization_501_757977batch_normalization_501_757979batch_normalization_501_757981*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_757331ø
leaky_re_lu_501/PartitionedCallPartitionedCall8batch_normalization_501/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_501_layer_call_and_return_conditional_losses_757989
!dense_555/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_501/PartitionedCall:output:0dense_555_758002dense_555_758004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_555_layer_call_and_return_conditional_losses_758001
/batch_normalization_502/StatefulPartitionedCallStatefulPartitionedCall*dense_555/StatefulPartitionedCall:output:0batch_normalization_502_758007batch_normalization_502_758009batch_normalization_502_758011batch_normalization_502_758013*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_757413ø
leaky_re_lu_502/PartitionedCallPartitionedCall8batch_normalization_502/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_502_layer_call_and_return_conditional_losses_758021
!dense_556/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_502/PartitionedCall:output:0dense_556_758034dense_556_758036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_556_layer_call_and_return_conditional_losses_758033
/batch_normalization_503/StatefulPartitionedCallStatefulPartitionedCall*dense_556/StatefulPartitionedCall:output:0batch_normalization_503_758039batch_normalization_503_758041batch_normalization_503_758043batch_normalization_503_758045*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_757495ø
leaky_re_lu_503/PartitionedCallPartitionedCall8batch_normalization_503/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_503_layer_call_and_return_conditional_losses_758053
!dense_557/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_503/PartitionedCall:output:0dense_557_758066dense_557_758068*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_557_layer_call_and_return_conditional_losses_758065
/batch_normalization_504/StatefulPartitionedCallStatefulPartitionedCall*dense_557/StatefulPartitionedCall:output:0batch_normalization_504_758071batch_normalization_504_758073batch_normalization_504_758075batch_normalization_504_758077*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_757577ø
leaky_re_lu_504/PartitionedCallPartitionedCall8batch_normalization_504/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_758085
!dense_558/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_504/PartitionedCall:output:0dense_558_758098dense_558_758100*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_558_layer_call_and_return_conditional_losses_758097
/batch_normalization_505/StatefulPartitionedCallStatefulPartitionedCall*dense_558/StatefulPartitionedCall:output:0batch_normalization_505_758103batch_normalization_505_758105batch_normalization_505_758107batch_normalization_505_758109*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_757659ø
leaky_re_lu_505/PartitionedCallPartitionedCall8batch_normalization_505/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_758117
!dense_559/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_505/PartitionedCall:output:0dense_559_758130dense_559_758132*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_559_layer_call_and_return_conditional_losses_758129
/batch_normalization_506/StatefulPartitionedCallStatefulPartitionedCall*dense_559/StatefulPartitionedCall:output:0batch_normalization_506_758135batch_normalization_506_758137batch_normalization_506_758139batch_normalization_506_758141*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_757741ø
leaky_re_lu_506/PartitionedCallPartitionedCall8batch_normalization_506/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_758149
!dense_560/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_506/PartitionedCall:output:0dense_560_758162dense_560_758164*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_560_layer_call_and_return_conditional_losses_758161
/batch_normalization_507/StatefulPartitionedCallStatefulPartitionedCall*dense_560/StatefulPartitionedCall:output:0batch_normalization_507_758167batch_normalization_507_758169batch_normalization_507_758171batch_normalization_507_758173*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_757823ø
leaky_re_lu_507/PartitionedCallPartitionedCall8batch_normalization_507/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_758181
!dense_561/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_507/PartitionedCall:output:0dense_561_758194dense_561_758196*
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
E__inference_dense_561_layer_call_and_return_conditional_losses_758193y
IdentityIdentity*dense_561/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp0^batch_normalization_499/StatefulPartitionedCall0^batch_normalization_500/StatefulPartitionedCall0^batch_normalization_501/StatefulPartitionedCall0^batch_normalization_502/StatefulPartitionedCall0^batch_normalization_503/StatefulPartitionedCall0^batch_normalization_504/StatefulPartitionedCall0^batch_normalization_505/StatefulPartitionedCall0^batch_normalization_506/StatefulPartitionedCall0^batch_normalization_507/StatefulPartitionedCall"^dense_552/StatefulPartitionedCall"^dense_553/StatefulPartitionedCall"^dense_554/StatefulPartitionedCall"^dense_555/StatefulPartitionedCall"^dense_556/StatefulPartitionedCall"^dense_557/StatefulPartitionedCall"^dense_558/StatefulPartitionedCall"^dense_559/StatefulPartitionedCall"^dense_560/StatefulPartitionedCall"^dense_561/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_499/StatefulPartitionedCall/batch_normalization_499/StatefulPartitionedCall2b
/batch_normalization_500/StatefulPartitionedCall/batch_normalization_500/StatefulPartitionedCall2b
/batch_normalization_501/StatefulPartitionedCall/batch_normalization_501/StatefulPartitionedCall2b
/batch_normalization_502/StatefulPartitionedCall/batch_normalization_502/StatefulPartitionedCall2b
/batch_normalization_503/StatefulPartitionedCall/batch_normalization_503/StatefulPartitionedCall2b
/batch_normalization_504/StatefulPartitionedCall/batch_normalization_504/StatefulPartitionedCall2b
/batch_normalization_505/StatefulPartitionedCall/batch_normalization_505/StatefulPartitionedCall2b
/batch_normalization_506/StatefulPartitionedCall/batch_normalization_506/StatefulPartitionedCall2b
/batch_normalization_507/StatefulPartitionedCall/batch_normalization_507/StatefulPartitionedCall2F
!dense_552/StatefulPartitionedCall!dense_552/StatefulPartitionedCall2F
!dense_553/StatefulPartitionedCall!dense_553/StatefulPartitionedCall2F
!dense_554/StatefulPartitionedCall!dense_554/StatefulPartitionedCall2F
!dense_555/StatefulPartitionedCall!dense_555/StatefulPartitionedCall2F
!dense_556/StatefulPartitionedCall!dense_556/StatefulPartitionedCall2F
!dense_557/StatefulPartitionedCall!dense_557/StatefulPartitionedCall2F
!dense_558/StatefulPartitionedCall!dense_558/StatefulPartitionedCall2F
!dense_559/StatefulPartitionedCall!dense_559/StatefulPartitionedCall2F
!dense_560/StatefulPartitionedCall!dense_560/StatefulPartitionedCall2F
!dense_561/StatefulPartitionedCall!dense_561/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
%
ì
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_757378

inputs5
'assignmovingavg_readvariableop_resource:a7
)assignmovingavg_1_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a/
!batchnorm_readvariableop_resource:a
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:a
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿal
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:a*
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
:a*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a¬
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
:a*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:a~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a´
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿah
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_501_layer_call_fn_760542

inputs
unknown:a
	unknown_0:a
	unknown_1:a
	unknown_2:a
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_757378o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_760378

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
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
:P*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P¬
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
:P*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P´
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_499_layer_call_fn_760311

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_757167o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_503_layer_call_fn_760760

inputs
unknown:a
	unknown_0:a
	unknown_1:a
	unknown_2:a
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_757542o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
È	
ö
E__inference_dense_556_layer_call_and_return_conditional_losses_760734

inputs0
matmul_readvariableop_resource:aa-
biasadd_readvariableop_resource:a
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:aa*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿar
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:a*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_757706

inputs5
'assignmovingavg_readvariableop_resource:=7
)assignmovingavg_1_readvariableop_resource:=3
%batchnorm_mul_readvariableop_resource:=/
!batchnorm_readvariableop_resource:=
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:=
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:=*
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
:=*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:=x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:=¬
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
:=*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:=~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:=´
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
:=P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:=~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:=v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:=r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_505_layer_call_fn_760965

inputs
unknown:=
	unknown_0:=
	unknown_1:=
	unknown_2:=
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_757659o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
¡
@
!__inference__wrapped_model_757143
normalization_53_input(
$sequential_53_normalization_53_sub_y)
%sequential_53_normalization_53_sqrt_xH
6sequential_53_dense_552_matmul_readvariableop_resource:PE
7sequential_53_dense_552_biasadd_readvariableop_resource:PU
Gsequential_53_batch_normalization_499_batchnorm_readvariableop_resource:PY
Ksequential_53_batch_normalization_499_batchnorm_mul_readvariableop_resource:PW
Isequential_53_batch_normalization_499_batchnorm_readvariableop_1_resource:PW
Isequential_53_batch_normalization_499_batchnorm_readvariableop_2_resource:PH
6sequential_53_dense_553_matmul_readvariableop_resource:PPE
7sequential_53_dense_553_biasadd_readvariableop_resource:PU
Gsequential_53_batch_normalization_500_batchnorm_readvariableop_resource:PY
Ksequential_53_batch_normalization_500_batchnorm_mul_readvariableop_resource:PW
Isequential_53_batch_normalization_500_batchnorm_readvariableop_1_resource:PW
Isequential_53_batch_normalization_500_batchnorm_readvariableop_2_resource:PH
6sequential_53_dense_554_matmul_readvariableop_resource:PaE
7sequential_53_dense_554_biasadd_readvariableop_resource:aU
Gsequential_53_batch_normalization_501_batchnorm_readvariableop_resource:aY
Ksequential_53_batch_normalization_501_batchnorm_mul_readvariableop_resource:aW
Isequential_53_batch_normalization_501_batchnorm_readvariableop_1_resource:aW
Isequential_53_batch_normalization_501_batchnorm_readvariableop_2_resource:aH
6sequential_53_dense_555_matmul_readvariableop_resource:aaE
7sequential_53_dense_555_biasadd_readvariableop_resource:aU
Gsequential_53_batch_normalization_502_batchnorm_readvariableop_resource:aY
Ksequential_53_batch_normalization_502_batchnorm_mul_readvariableop_resource:aW
Isequential_53_batch_normalization_502_batchnorm_readvariableop_1_resource:aW
Isequential_53_batch_normalization_502_batchnorm_readvariableop_2_resource:aH
6sequential_53_dense_556_matmul_readvariableop_resource:aaE
7sequential_53_dense_556_biasadd_readvariableop_resource:aU
Gsequential_53_batch_normalization_503_batchnorm_readvariableop_resource:aY
Ksequential_53_batch_normalization_503_batchnorm_mul_readvariableop_resource:aW
Isequential_53_batch_normalization_503_batchnorm_readvariableop_1_resource:aW
Isequential_53_batch_normalization_503_batchnorm_readvariableop_2_resource:aH
6sequential_53_dense_557_matmul_readvariableop_resource:aaE
7sequential_53_dense_557_biasadd_readvariableop_resource:aU
Gsequential_53_batch_normalization_504_batchnorm_readvariableop_resource:aY
Ksequential_53_batch_normalization_504_batchnorm_mul_readvariableop_resource:aW
Isequential_53_batch_normalization_504_batchnorm_readvariableop_1_resource:aW
Isequential_53_batch_normalization_504_batchnorm_readvariableop_2_resource:aH
6sequential_53_dense_558_matmul_readvariableop_resource:a=E
7sequential_53_dense_558_biasadd_readvariableop_resource:=U
Gsequential_53_batch_normalization_505_batchnorm_readvariableop_resource:=Y
Ksequential_53_batch_normalization_505_batchnorm_mul_readvariableop_resource:=W
Isequential_53_batch_normalization_505_batchnorm_readvariableop_1_resource:=W
Isequential_53_batch_normalization_505_batchnorm_readvariableop_2_resource:=H
6sequential_53_dense_559_matmul_readvariableop_resource:==E
7sequential_53_dense_559_biasadd_readvariableop_resource:=U
Gsequential_53_batch_normalization_506_batchnorm_readvariableop_resource:=Y
Ksequential_53_batch_normalization_506_batchnorm_mul_readvariableop_resource:=W
Isequential_53_batch_normalization_506_batchnorm_readvariableop_1_resource:=W
Isequential_53_batch_normalization_506_batchnorm_readvariableop_2_resource:=H
6sequential_53_dense_560_matmul_readvariableop_resource:==E
7sequential_53_dense_560_biasadd_readvariableop_resource:=U
Gsequential_53_batch_normalization_507_batchnorm_readvariableop_resource:=Y
Ksequential_53_batch_normalization_507_batchnorm_mul_readvariableop_resource:=W
Isequential_53_batch_normalization_507_batchnorm_readvariableop_1_resource:=W
Isequential_53_batch_normalization_507_batchnorm_readvariableop_2_resource:=H
6sequential_53_dense_561_matmul_readvariableop_resource:=E
7sequential_53_dense_561_biasadd_readvariableop_resource:
identity¢>sequential_53/batch_normalization_499/batchnorm/ReadVariableOp¢@sequential_53/batch_normalization_499/batchnorm/ReadVariableOp_1¢@sequential_53/batch_normalization_499/batchnorm/ReadVariableOp_2¢Bsequential_53/batch_normalization_499/batchnorm/mul/ReadVariableOp¢>sequential_53/batch_normalization_500/batchnorm/ReadVariableOp¢@sequential_53/batch_normalization_500/batchnorm/ReadVariableOp_1¢@sequential_53/batch_normalization_500/batchnorm/ReadVariableOp_2¢Bsequential_53/batch_normalization_500/batchnorm/mul/ReadVariableOp¢>sequential_53/batch_normalization_501/batchnorm/ReadVariableOp¢@sequential_53/batch_normalization_501/batchnorm/ReadVariableOp_1¢@sequential_53/batch_normalization_501/batchnorm/ReadVariableOp_2¢Bsequential_53/batch_normalization_501/batchnorm/mul/ReadVariableOp¢>sequential_53/batch_normalization_502/batchnorm/ReadVariableOp¢@sequential_53/batch_normalization_502/batchnorm/ReadVariableOp_1¢@sequential_53/batch_normalization_502/batchnorm/ReadVariableOp_2¢Bsequential_53/batch_normalization_502/batchnorm/mul/ReadVariableOp¢>sequential_53/batch_normalization_503/batchnorm/ReadVariableOp¢@sequential_53/batch_normalization_503/batchnorm/ReadVariableOp_1¢@sequential_53/batch_normalization_503/batchnorm/ReadVariableOp_2¢Bsequential_53/batch_normalization_503/batchnorm/mul/ReadVariableOp¢>sequential_53/batch_normalization_504/batchnorm/ReadVariableOp¢@sequential_53/batch_normalization_504/batchnorm/ReadVariableOp_1¢@sequential_53/batch_normalization_504/batchnorm/ReadVariableOp_2¢Bsequential_53/batch_normalization_504/batchnorm/mul/ReadVariableOp¢>sequential_53/batch_normalization_505/batchnorm/ReadVariableOp¢@sequential_53/batch_normalization_505/batchnorm/ReadVariableOp_1¢@sequential_53/batch_normalization_505/batchnorm/ReadVariableOp_2¢Bsequential_53/batch_normalization_505/batchnorm/mul/ReadVariableOp¢>sequential_53/batch_normalization_506/batchnorm/ReadVariableOp¢@sequential_53/batch_normalization_506/batchnorm/ReadVariableOp_1¢@sequential_53/batch_normalization_506/batchnorm/ReadVariableOp_2¢Bsequential_53/batch_normalization_506/batchnorm/mul/ReadVariableOp¢>sequential_53/batch_normalization_507/batchnorm/ReadVariableOp¢@sequential_53/batch_normalization_507/batchnorm/ReadVariableOp_1¢@sequential_53/batch_normalization_507/batchnorm/ReadVariableOp_2¢Bsequential_53/batch_normalization_507/batchnorm/mul/ReadVariableOp¢.sequential_53/dense_552/BiasAdd/ReadVariableOp¢-sequential_53/dense_552/MatMul/ReadVariableOp¢.sequential_53/dense_553/BiasAdd/ReadVariableOp¢-sequential_53/dense_553/MatMul/ReadVariableOp¢.sequential_53/dense_554/BiasAdd/ReadVariableOp¢-sequential_53/dense_554/MatMul/ReadVariableOp¢.sequential_53/dense_555/BiasAdd/ReadVariableOp¢-sequential_53/dense_555/MatMul/ReadVariableOp¢.sequential_53/dense_556/BiasAdd/ReadVariableOp¢-sequential_53/dense_556/MatMul/ReadVariableOp¢.sequential_53/dense_557/BiasAdd/ReadVariableOp¢-sequential_53/dense_557/MatMul/ReadVariableOp¢.sequential_53/dense_558/BiasAdd/ReadVariableOp¢-sequential_53/dense_558/MatMul/ReadVariableOp¢.sequential_53/dense_559/BiasAdd/ReadVariableOp¢-sequential_53/dense_559/MatMul/ReadVariableOp¢.sequential_53/dense_560/BiasAdd/ReadVariableOp¢-sequential_53/dense_560/MatMul/ReadVariableOp¢.sequential_53/dense_561/BiasAdd/ReadVariableOp¢-sequential_53/dense_561/MatMul/ReadVariableOp
"sequential_53/normalization_53/subSubnormalization_53_input$sequential_53_normalization_53_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#sequential_53/normalization_53/SqrtSqrt%sequential_53_normalization_53_sqrt_x*
T0*
_output_shapes

:m
(sequential_53/normalization_53/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¶
&sequential_53/normalization_53/MaximumMaximum'sequential_53/normalization_53/Sqrt:y:01sequential_53/normalization_53/Maximum/y:output:0*
T0*
_output_shapes

:·
&sequential_53/normalization_53/truedivRealDiv&sequential_53/normalization_53/sub:z:0*sequential_53/normalization_53/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_53/dense_552/MatMul/ReadVariableOpReadVariableOp6sequential_53_dense_552_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0½
sequential_53/dense_552/MatMulMatMul*sequential_53/normalization_53/truediv:z:05sequential_53/dense_552/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP¢
.sequential_53/dense_552/BiasAdd/ReadVariableOpReadVariableOp7sequential_53_dense_552_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0¾
sequential_53/dense_552/BiasAddBiasAdd(sequential_53/dense_552/MatMul:product:06sequential_53/dense_552/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPÂ
>sequential_53/batch_normalization_499/batchnorm/ReadVariableOpReadVariableOpGsequential_53_batch_normalization_499_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0z
5sequential_53/batch_normalization_499/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_53/batch_normalization_499/batchnorm/addAddV2Fsequential_53/batch_normalization_499/batchnorm/ReadVariableOp:value:0>sequential_53/batch_normalization_499/batchnorm/add/y:output:0*
T0*
_output_shapes
:P
5sequential_53/batch_normalization_499/batchnorm/RsqrtRsqrt7sequential_53/batch_normalization_499/batchnorm/add:z:0*
T0*
_output_shapes
:PÊ
Bsequential_53/batch_normalization_499/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_53_batch_normalization_499_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0æ
3sequential_53/batch_normalization_499/batchnorm/mulMul9sequential_53/batch_normalization_499/batchnorm/Rsqrt:y:0Jsequential_53/batch_normalization_499/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:PÑ
5sequential_53/batch_normalization_499/batchnorm/mul_1Mul(sequential_53/dense_552/BiasAdd:output:07sequential_53/batch_normalization_499/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPÆ
@sequential_53/batch_normalization_499/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_53_batch_normalization_499_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0ä
5sequential_53/batch_normalization_499/batchnorm/mul_2MulHsequential_53/batch_normalization_499/batchnorm/ReadVariableOp_1:value:07sequential_53/batch_normalization_499/batchnorm/mul:z:0*
T0*
_output_shapes
:PÆ
@sequential_53/batch_normalization_499/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_53_batch_normalization_499_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0ä
3sequential_53/batch_normalization_499/batchnorm/subSubHsequential_53/batch_normalization_499/batchnorm/ReadVariableOp_2:value:09sequential_53/batch_normalization_499/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pä
5sequential_53/batch_normalization_499/batchnorm/add_1AddV29sequential_53/batch_normalization_499/batchnorm/mul_1:z:07sequential_53/batch_normalization_499/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP¨
'sequential_53/leaky_re_lu_499/LeakyRelu	LeakyRelu9sequential_53/batch_normalization_499/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*
alpha%>¤
-sequential_53/dense_553/MatMul/ReadVariableOpReadVariableOp6sequential_53_dense_553_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0È
sequential_53/dense_553/MatMulMatMul5sequential_53/leaky_re_lu_499/LeakyRelu:activations:05sequential_53/dense_553/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP¢
.sequential_53/dense_553/BiasAdd/ReadVariableOpReadVariableOp7sequential_53_dense_553_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0¾
sequential_53/dense_553/BiasAddBiasAdd(sequential_53/dense_553/MatMul:product:06sequential_53/dense_553/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPÂ
>sequential_53/batch_normalization_500/batchnorm/ReadVariableOpReadVariableOpGsequential_53_batch_normalization_500_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0z
5sequential_53/batch_normalization_500/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_53/batch_normalization_500/batchnorm/addAddV2Fsequential_53/batch_normalization_500/batchnorm/ReadVariableOp:value:0>sequential_53/batch_normalization_500/batchnorm/add/y:output:0*
T0*
_output_shapes
:P
5sequential_53/batch_normalization_500/batchnorm/RsqrtRsqrt7sequential_53/batch_normalization_500/batchnorm/add:z:0*
T0*
_output_shapes
:PÊ
Bsequential_53/batch_normalization_500/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_53_batch_normalization_500_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0æ
3sequential_53/batch_normalization_500/batchnorm/mulMul9sequential_53/batch_normalization_500/batchnorm/Rsqrt:y:0Jsequential_53/batch_normalization_500/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:PÑ
5sequential_53/batch_normalization_500/batchnorm/mul_1Mul(sequential_53/dense_553/BiasAdd:output:07sequential_53/batch_normalization_500/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPÆ
@sequential_53/batch_normalization_500/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_53_batch_normalization_500_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0ä
5sequential_53/batch_normalization_500/batchnorm/mul_2MulHsequential_53/batch_normalization_500/batchnorm/ReadVariableOp_1:value:07sequential_53/batch_normalization_500/batchnorm/mul:z:0*
T0*
_output_shapes
:PÆ
@sequential_53/batch_normalization_500/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_53_batch_normalization_500_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0ä
3sequential_53/batch_normalization_500/batchnorm/subSubHsequential_53/batch_normalization_500/batchnorm/ReadVariableOp_2:value:09sequential_53/batch_normalization_500/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pä
5sequential_53/batch_normalization_500/batchnorm/add_1AddV29sequential_53/batch_normalization_500/batchnorm/mul_1:z:07sequential_53/batch_normalization_500/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP¨
'sequential_53/leaky_re_lu_500/LeakyRelu	LeakyRelu9sequential_53/batch_normalization_500/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*
alpha%>¤
-sequential_53/dense_554/MatMul/ReadVariableOpReadVariableOp6sequential_53_dense_554_matmul_readvariableop_resource*
_output_shapes

:Pa*
dtype0È
sequential_53/dense_554/MatMulMatMul5sequential_53/leaky_re_lu_500/LeakyRelu:activations:05sequential_53/dense_554/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¢
.sequential_53/dense_554/BiasAdd/ReadVariableOpReadVariableOp7sequential_53_dense_554_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0¾
sequential_53/dense_554/BiasAddBiasAdd(sequential_53/dense_554/MatMul:product:06sequential_53/dense_554/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaÂ
>sequential_53/batch_normalization_501/batchnorm/ReadVariableOpReadVariableOpGsequential_53_batch_normalization_501_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0z
5sequential_53/batch_normalization_501/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_53/batch_normalization_501/batchnorm/addAddV2Fsequential_53/batch_normalization_501/batchnorm/ReadVariableOp:value:0>sequential_53/batch_normalization_501/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
5sequential_53/batch_normalization_501/batchnorm/RsqrtRsqrt7sequential_53/batch_normalization_501/batchnorm/add:z:0*
T0*
_output_shapes
:aÊ
Bsequential_53/batch_normalization_501/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_53_batch_normalization_501_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0æ
3sequential_53/batch_normalization_501/batchnorm/mulMul9sequential_53/batch_normalization_501/batchnorm/Rsqrt:y:0Jsequential_53/batch_normalization_501/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:aÑ
5sequential_53/batch_normalization_501/batchnorm/mul_1Mul(sequential_53/dense_554/BiasAdd:output:07sequential_53/batch_normalization_501/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaÆ
@sequential_53/batch_normalization_501/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_53_batch_normalization_501_batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0ä
5sequential_53/batch_normalization_501/batchnorm/mul_2MulHsequential_53/batch_normalization_501/batchnorm/ReadVariableOp_1:value:07sequential_53/batch_normalization_501/batchnorm/mul:z:0*
T0*
_output_shapes
:aÆ
@sequential_53/batch_normalization_501/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_53_batch_normalization_501_batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0ä
3sequential_53/batch_normalization_501/batchnorm/subSubHsequential_53/batch_normalization_501/batchnorm/ReadVariableOp_2:value:09sequential_53/batch_normalization_501/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aä
5sequential_53/batch_normalization_501/batchnorm/add_1AddV29sequential_53/batch_normalization_501/batchnorm/mul_1:z:07sequential_53/batch_normalization_501/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¨
'sequential_53/leaky_re_lu_501/LeakyRelu	LeakyRelu9sequential_53/batch_normalization_501/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>¤
-sequential_53/dense_555/MatMul/ReadVariableOpReadVariableOp6sequential_53_dense_555_matmul_readvariableop_resource*
_output_shapes

:aa*
dtype0È
sequential_53/dense_555/MatMulMatMul5sequential_53/leaky_re_lu_501/LeakyRelu:activations:05sequential_53/dense_555/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¢
.sequential_53/dense_555/BiasAdd/ReadVariableOpReadVariableOp7sequential_53_dense_555_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0¾
sequential_53/dense_555/BiasAddBiasAdd(sequential_53/dense_555/MatMul:product:06sequential_53/dense_555/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaÂ
>sequential_53/batch_normalization_502/batchnorm/ReadVariableOpReadVariableOpGsequential_53_batch_normalization_502_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0z
5sequential_53/batch_normalization_502/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_53/batch_normalization_502/batchnorm/addAddV2Fsequential_53/batch_normalization_502/batchnorm/ReadVariableOp:value:0>sequential_53/batch_normalization_502/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
5sequential_53/batch_normalization_502/batchnorm/RsqrtRsqrt7sequential_53/batch_normalization_502/batchnorm/add:z:0*
T0*
_output_shapes
:aÊ
Bsequential_53/batch_normalization_502/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_53_batch_normalization_502_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0æ
3sequential_53/batch_normalization_502/batchnorm/mulMul9sequential_53/batch_normalization_502/batchnorm/Rsqrt:y:0Jsequential_53/batch_normalization_502/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:aÑ
5sequential_53/batch_normalization_502/batchnorm/mul_1Mul(sequential_53/dense_555/BiasAdd:output:07sequential_53/batch_normalization_502/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaÆ
@sequential_53/batch_normalization_502/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_53_batch_normalization_502_batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0ä
5sequential_53/batch_normalization_502/batchnorm/mul_2MulHsequential_53/batch_normalization_502/batchnorm/ReadVariableOp_1:value:07sequential_53/batch_normalization_502/batchnorm/mul:z:0*
T0*
_output_shapes
:aÆ
@sequential_53/batch_normalization_502/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_53_batch_normalization_502_batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0ä
3sequential_53/batch_normalization_502/batchnorm/subSubHsequential_53/batch_normalization_502/batchnorm/ReadVariableOp_2:value:09sequential_53/batch_normalization_502/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aä
5sequential_53/batch_normalization_502/batchnorm/add_1AddV29sequential_53/batch_normalization_502/batchnorm/mul_1:z:07sequential_53/batch_normalization_502/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¨
'sequential_53/leaky_re_lu_502/LeakyRelu	LeakyRelu9sequential_53/batch_normalization_502/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>¤
-sequential_53/dense_556/MatMul/ReadVariableOpReadVariableOp6sequential_53_dense_556_matmul_readvariableop_resource*
_output_shapes

:aa*
dtype0È
sequential_53/dense_556/MatMulMatMul5sequential_53/leaky_re_lu_502/LeakyRelu:activations:05sequential_53/dense_556/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¢
.sequential_53/dense_556/BiasAdd/ReadVariableOpReadVariableOp7sequential_53_dense_556_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0¾
sequential_53/dense_556/BiasAddBiasAdd(sequential_53/dense_556/MatMul:product:06sequential_53/dense_556/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaÂ
>sequential_53/batch_normalization_503/batchnorm/ReadVariableOpReadVariableOpGsequential_53_batch_normalization_503_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0z
5sequential_53/batch_normalization_503/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_53/batch_normalization_503/batchnorm/addAddV2Fsequential_53/batch_normalization_503/batchnorm/ReadVariableOp:value:0>sequential_53/batch_normalization_503/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
5sequential_53/batch_normalization_503/batchnorm/RsqrtRsqrt7sequential_53/batch_normalization_503/batchnorm/add:z:0*
T0*
_output_shapes
:aÊ
Bsequential_53/batch_normalization_503/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_53_batch_normalization_503_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0æ
3sequential_53/batch_normalization_503/batchnorm/mulMul9sequential_53/batch_normalization_503/batchnorm/Rsqrt:y:0Jsequential_53/batch_normalization_503/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:aÑ
5sequential_53/batch_normalization_503/batchnorm/mul_1Mul(sequential_53/dense_556/BiasAdd:output:07sequential_53/batch_normalization_503/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaÆ
@sequential_53/batch_normalization_503/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_53_batch_normalization_503_batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0ä
5sequential_53/batch_normalization_503/batchnorm/mul_2MulHsequential_53/batch_normalization_503/batchnorm/ReadVariableOp_1:value:07sequential_53/batch_normalization_503/batchnorm/mul:z:0*
T0*
_output_shapes
:aÆ
@sequential_53/batch_normalization_503/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_53_batch_normalization_503_batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0ä
3sequential_53/batch_normalization_503/batchnorm/subSubHsequential_53/batch_normalization_503/batchnorm/ReadVariableOp_2:value:09sequential_53/batch_normalization_503/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aä
5sequential_53/batch_normalization_503/batchnorm/add_1AddV29sequential_53/batch_normalization_503/batchnorm/mul_1:z:07sequential_53/batch_normalization_503/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¨
'sequential_53/leaky_re_lu_503/LeakyRelu	LeakyRelu9sequential_53/batch_normalization_503/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>¤
-sequential_53/dense_557/MatMul/ReadVariableOpReadVariableOp6sequential_53_dense_557_matmul_readvariableop_resource*
_output_shapes

:aa*
dtype0È
sequential_53/dense_557/MatMulMatMul5sequential_53/leaky_re_lu_503/LeakyRelu:activations:05sequential_53/dense_557/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¢
.sequential_53/dense_557/BiasAdd/ReadVariableOpReadVariableOp7sequential_53_dense_557_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0¾
sequential_53/dense_557/BiasAddBiasAdd(sequential_53/dense_557/MatMul:product:06sequential_53/dense_557/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaÂ
>sequential_53/batch_normalization_504/batchnorm/ReadVariableOpReadVariableOpGsequential_53_batch_normalization_504_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0z
5sequential_53/batch_normalization_504/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_53/batch_normalization_504/batchnorm/addAddV2Fsequential_53/batch_normalization_504/batchnorm/ReadVariableOp:value:0>sequential_53/batch_normalization_504/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
5sequential_53/batch_normalization_504/batchnorm/RsqrtRsqrt7sequential_53/batch_normalization_504/batchnorm/add:z:0*
T0*
_output_shapes
:aÊ
Bsequential_53/batch_normalization_504/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_53_batch_normalization_504_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0æ
3sequential_53/batch_normalization_504/batchnorm/mulMul9sequential_53/batch_normalization_504/batchnorm/Rsqrt:y:0Jsequential_53/batch_normalization_504/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:aÑ
5sequential_53/batch_normalization_504/batchnorm/mul_1Mul(sequential_53/dense_557/BiasAdd:output:07sequential_53/batch_normalization_504/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaÆ
@sequential_53/batch_normalization_504/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_53_batch_normalization_504_batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0ä
5sequential_53/batch_normalization_504/batchnorm/mul_2MulHsequential_53/batch_normalization_504/batchnorm/ReadVariableOp_1:value:07sequential_53/batch_normalization_504/batchnorm/mul:z:0*
T0*
_output_shapes
:aÆ
@sequential_53/batch_normalization_504/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_53_batch_normalization_504_batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0ä
3sequential_53/batch_normalization_504/batchnorm/subSubHsequential_53/batch_normalization_504/batchnorm/ReadVariableOp_2:value:09sequential_53/batch_normalization_504/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aä
5sequential_53/batch_normalization_504/batchnorm/add_1AddV29sequential_53/batch_normalization_504/batchnorm/mul_1:z:07sequential_53/batch_normalization_504/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¨
'sequential_53/leaky_re_lu_504/LeakyRelu	LeakyRelu9sequential_53/batch_normalization_504/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>¤
-sequential_53/dense_558/MatMul/ReadVariableOpReadVariableOp6sequential_53_dense_558_matmul_readvariableop_resource*
_output_shapes

:a=*
dtype0È
sequential_53/dense_558/MatMulMatMul5sequential_53/leaky_re_lu_504/LeakyRelu:activations:05sequential_53/dense_558/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¢
.sequential_53/dense_558/BiasAdd/ReadVariableOpReadVariableOp7sequential_53_dense_558_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0¾
sequential_53/dense_558/BiasAddBiasAdd(sequential_53/dense_558/MatMul:product:06sequential_53/dense_558/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Â
>sequential_53/batch_normalization_505/batchnorm/ReadVariableOpReadVariableOpGsequential_53_batch_normalization_505_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0z
5sequential_53/batch_normalization_505/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_53/batch_normalization_505/batchnorm/addAddV2Fsequential_53/batch_normalization_505/batchnorm/ReadVariableOp:value:0>sequential_53/batch_normalization_505/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
5sequential_53/batch_normalization_505/batchnorm/RsqrtRsqrt7sequential_53/batch_normalization_505/batchnorm/add:z:0*
T0*
_output_shapes
:=Ê
Bsequential_53/batch_normalization_505/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_53_batch_normalization_505_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0æ
3sequential_53/batch_normalization_505/batchnorm/mulMul9sequential_53/batch_normalization_505/batchnorm/Rsqrt:y:0Jsequential_53/batch_normalization_505/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=Ñ
5sequential_53/batch_normalization_505/batchnorm/mul_1Mul(sequential_53/dense_558/BiasAdd:output:07sequential_53/batch_normalization_505/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Æ
@sequential_53/batch_normalization_505/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_53_batch_normalization_505_batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0ä
5sequential_53/batch_normalization_505/batchnorm/mul_2MulHsequential_53/batch_normalization_505/batchnorm/ReadVariableOp_1:value:07sequential_53/batch_normalization_505/batchnorm/mul:z:0*
T0*
_output_shapes
:=Æ
@sequential_53/batch_normalization_505/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_53_batch_normalization_505_batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0ä
3sequential_53/batch_normalization_505/batchnorm/subSubHsequential_53/batch_normalization_505/batchnorm/ReadVariableOp_2:value:09sequential_53/batch_normalization_505/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=ä
5sequential_53/batch_normalization_505/batchnorm/add_1AddV29sequential_53/batch_normalization_505/batchnorm/mul_1:z:07sequential_53/batch_normalization_505/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¨
'sequential_53/leaky_re_lu_505/LeakyRelu	LeakyRelu9sequential_53/batch_normalization_505/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>¤
-sequential_53/dense_559/MatMul/ReadVariableOpReadVariableOp6sequential_53_dense_559_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0È
sequential_53/dense_559/MatMulMatMul5sequential_53/leaky_re_lu_505/LeakyRelu:activations:05sequential_53/dense_559/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¢
.sequential_53/dense_559/BiasAdd/ReadVariableOpReadVariableOp7sequential_53_dense_559_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0¾
sequential_53/dense_559/BiasAddBiasAdd(sequential_53/dense_559/MatMul:product:06sequential_53/dense_559/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Â
>sequential_53/batch_normalization_506/batchnorm/ReadVariableOpReadVariableOpGsequential_53_batch_normalization_506_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0z
5sequential_53/batch_normalization_506/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_53/batch_normalization_506/batchnorm/addAddV2Fsequential_53/batch_normalization_506/batchnorm/ReadVariableOp:value:0>sequential_53/batch_normalization_506/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
5sequential_53/batch_normalization_506/batchnorm/RsqrtRsqrt7sequential_53/batch_normalization_506/batchnorm/add:z:0*
T0*
_output_shapes
:=Ê
Bsequential_53/batch_normalization_506/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_53_batch_normalization_506_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0æ
3sequential_53/batch_normalization_506/batchnorm/mulMul9sequential_53/batch_normalization_506/batchnorm/Rsqrt:y:0Jsequential_53/batch_normalization_506/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=Ñ
5sequential_53/batch_normalization_506/batchnorm/mul_1Mul(sequential_53/dense_559/BiasAdd:output:07sequential_53/batch_normalization_506/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Æ
@sequential_53/batch_normalization_506/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_53_batch_normalization_506_batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0ä
5sequential_53/batch_normalization_506/batchnorm/mul_2MulHsequential_53/batch_normalization_506/batchnorm/ReadVariableOp_1:value:07sequential_53/batch_normalization_506/batchnorm/mul:z:0*
T0*
_output_shapes
:=Æ
@sequential_53/batch_normalization_506/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_53_batch_normalization_506_batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0ä
3sequential_53/batch_normalization_506/batchnorm/subSubHsequential_53/batch_normalization_506/batchnorm/ReadVariableOp_2:value:09sequential_53/batch_normalization_506/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=ä
5sequential_53/batch_normalization_506/batchnorm/add_1AddV29sequential_53/batch_normalization_506/batchnorm/mul_1:z:07sequential_53/batch_normalization_506/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¨
'sequential_53/leaky_re_lu_506/LeakyRelu	LeakyRelu9sequential_53/batch_normalization_506/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>¤
-sequential_53/dense_560/MatMul/ReadVariableOpReadVariableOp6sequential_53_dense_560_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0È
sequential_53/dense_560/MatMulMatMul5sequential_53/leaky_re_lu_506/LeakyRelu:activations:05sequential_53/dense_560/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¢
.sequential_53/dense_560/BiasAdd/ReadVariableOpReadVariableOp7sequential_53_dense_560_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0¾
sequential_53/dense_560/BiasAddBiasAdd(sequential_53/dense_560/MatMul:product:06sequential_53/dense_560/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Â
>sequential_53/batch_normalization_507/batchnorm/ReadVariableOpReadVariableOpGsequential_53_batch_normalization_507_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0z
5sequential_53/batch_normalization_507/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_53/batch_normalization_507/batchnorm/addAddV2Fsequential_53/batch_normalization_507/batchnorm/ReadVariableOp:value:0>sequential_53/batch_normalization_507/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
5sequential_53/batch_normalization_507/batchnorm/RsqrtRsqrt7sequential_53/batch_normalization_507/batchnorm/add:z:0*
T0*
_output_shapes
:=Ê
Bsequential_53/batch_normalization_507/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_53_batch_normalization_507_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0æ
3sequential_53/batch_normalization_507/batchnorm/mulMul9sequential_53/batch_normalization_507/batchnorm/Rsqrt:y:0Jsequential_53/batch_normalization_507/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=Ñ
5sequential_53/batch_normalization_507/batchnorm/mul_1Mul(sequential_53/dense_560/BiasAdd:output:07sequential_53/batch_normalization_507/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Æ
@sequential_53/batch_normalization_507/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_53_batch_normalization_507_batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0ä
5sequential_53/batch_normalization_507/batchnorm/mul_2MulHsequential_53/batch_normalization_507/batchnorm/ReadVariableOp_1:value:07sequential_53/batch_normalization_507/batchnorm/mul:z:0*
T0*
_output_shapes
:=Æ
@sequential_53/batch_normalization_507/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_53_batch_normalization_507_batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0ä
3sequential_53/batch_normalization_507/batchnorm/subSubHsequential_53/batch_normalization_507/batchnorm/ReadVariableOp_2:value:09sequential_53/batch_normalization_507/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=ä
5sequential_53/batch_normalization_507/batchnorm/add_1AddV29sequential_53/batch_normalization_507/batchnorm/mul_1:z:07sequential_53/batch_normalization_507/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¨
'sequential_53/leaky_re_lu_507/LeakyRelu	LeakyRelu9sequential_53/batch_normalization_507/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>¤
-sequential_53/dense_561/MatMul/ReadVariableOpReadVariableOp6sequential_53_dense_561_matmul_readvariableop_resource*
_output_shapes

:=*
dtype0È
sequential_53/dense_561/MatMulMatMul5sequential_53/leaky_re_lu_507/LeakyRelu:activations:05sequential_53/dense_561/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_53/dense_561/BiasAdd/ReadVariableOpReadVariableOp7sequential_53_dense_561_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_53/dense_561/BiasAddBiasAdd(sequential_53/dense_561/MatMul:product:06sequential_53/dense_561/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_53/dense_561/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
NoOpNoOp?^sequential_53/batch_normalization_499/batchnorm/ReadVariableOpA^sequential_53/batch_normalization_499/batchnorm/ReadVariableOp_1A^sequential_53/batch_normalization_499/batchnorm/ReadVariableOp_2C^sequential_53/batch_normalization_499/batchnorm/mul/ReadVariableOp?^sequential_53/batch_normalization_500/batchnorm/ReadVariableOpA^sequential_53/batch_normalization_500/batchnorm/ReadVariableOp_1A^sequential_53/batch_normalization_500/batchnorm/ReadVariableOp_2C^sequential_53/batch_normalization_500/batchnorm/mul/ReadVariableOp?^sequential_53/batch_normalization_501/batchnorm/ReadVariableOpA^sequential_53/batch_normalization_501/batchnorm/ReadVariableOp_1A^sequential_53/batch_normalization_501/batchnorm/ReadVariableOp_2C^sequential_53/batch_normalization_501/batchnorm/mul/ReadVariableOp?^sequential_53/batch_normalization_502/batchnorm/ReadVariableOpA^sequential_53/batch_normalization_502/batchnorm/ReadVariableOp_1A^sequential_53/batch_normalization_502/batchnorm/ReadVariableOp_2C^sequential_53/batch_normalization_502/batchnorm/mul/ReadVariableOp?^sequential_53/batch_normalization_503/batchnorm/ReadVariableOpA^sequential_53/batch_normalization_503/batchnorm/ReadVariableOp_1A^sequential_53/batch_normalization_503/batchnorm/ReadVariableOp_2C^sequential_53/batch_normalization_503/batchnorm/mul/ReadVariableOp?^sequential_53/batch_normalization_504/batchnorm/ReadVariableOpA^sequential_53/batch_normalization_504/batchnorm/ReadVariableOp_1A^sequential_53/batch_normalization_504/batchnorm/ReadVariableOp_2C^sequential_53/batch_normalization_504/batchnorm/mul/ReadVariableOp?^sequential_53/batch_normalization_505/batchnorm/ReadVariableOpA^sequential_53/batch_normalization_505/batchnorm/ReadVariableOp_1A^sequential_53/batch_normalization_505/batchnorm/ReadVariableOp_2C^sequential_53/batch_normalization_505/batchnorm/mul/ReadVariableOp?^sequential_53/batch_normalization_506/batchnorm/ReadVariableOpA^sequential_53/batch_normalization_506/batchnorm/ReadVariableOp_1A^sequential_53/batch_normalization_506/batchnorm/ReadVariableOp_2C^sequential_53/batch_normalization_506/batchnorm/mul/ReadVariableOp?^sequential_53/batch_normalization_507/batchnorm/ReadVariableOpA^sequential_53/batch_normalization_507/batchnorm/ReadVariableOp_1A^sequential_53/batch_normalization_507/batchnorm/ReadVariableOp_2C^sequential_53/batch_normalization_507/batchnorm/mul/ReadVariableOp/^sequential_53/dense_552/BiasAdd/ReadVariableOp.^sequential_53/dense_552/MatMul/ReadVariableOp/^sequential_53/dense_553/BiasAdd/ReadVariableOp.^sequential_53/dense_553/MatMul/ReadVariableOp/^sequential_53/dense_554/BiasAdd/ReadVariableOp.^sequential_53/dense_554/MatMul/ReadVariableOp/^sequential_53/dense_555/BiasAdd/ReadVariableOp.^sequential_53/dense_555/MatMul/ReadVariableOp/^sequential_53/dense_556/BiasAdd/ReadVariableOp.^sequential_53/dense_556/MatMul/ReadVariableOp/^sequential_53/dense_557/BiasAdd/ReadVariableOp.^sequential_53/dense_557/MatMul/ReadVariableOp/^sequential_53/dense_558/BiasAdd/ReadVariableOp.^sequential_53/dense_558/MatMul/ReadVariableOp/^sequential_53/dense_559/BiasAdd/ReadVariableOp.^sequential_53/dense_559/MatMul/ReadVariableOp/^sequential_53/dense_560/BiasAdd/ReadVariableOp.^sequential_53/dense_560/MatMul/ReadVariableOp/^sequential_53/dense_561/BiasAdd/ReadVariableOp.^sequential_53/dense_561/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>sequential_53/batch_normalization_499/batchnorm/ReadVariableOp>sequential_53/batch_normalization_499/batchnorm/ReadVariableOp2
@sequential_53/batch_normalization_499/batchnorm/ReadVariableOp_1@sequential_53/batch_normalization_499/batchnorm/ReadVariableOp_12
@sequential_53/batch_normalization_499/batchnorm/ReadVariableOp_2@sequential_53/batch_normalization_499/batchnorm/ReadVariableOp_22
Bsequential_53/batch_normalization_499/batchnorm/mul/ReadVariableOpBsequential_53/batch_normalization_499/batchnorm/mul/ReadVariableOp2
>sequential_53/batch_normalization_500/batchnorm/ReadVariableOp>sequential_53/batch_normalization_500/batchnorm/ReadVariableOp2
@sequential_53/batch_normalization_500/batchnorm/ReadVariableOp_1@sequential_53/batch_normalization_500/batchnorm/ReadVariableOp_12
@sequential_53/batch_normalization_500/batchnorm/ReadVariableOp_2@sequential_53/batch_normalization_500/batchnorm/ReadVariableOp_22
Bsequential_53/batch_normalization_500/batchnorm/mul/ReadVariableOpBsequential_53/batch_normalization_500/batchnorm/mul/ReadVariableOp2
>sequential_53/batch_normalization_501/batchnorm/ReadVariableOp>sequential_53/batch_normalization_501/batchnorm/ReadVariableOp2
@sequential_53/batch_normalization_501/batchnorm/ReadVariableOp_1@sequential_53/batch_normalization_501/batchnorm/ReadVariableOp_12
@sequential_53/batch_normalization_501/batchnorm/ReadVariableOp_2@sequential_53/batch_normalization_501/batchnorm/ReadVariableOp_22
Bsequential_53/batch_normalization_501/batchnorm/mul/ReadVariableOpBsequential_53/batch_normalization_501/batchnorm/mul/ReadVariableOp2
>sequential_53/batch_normalization_502/batchnorm/ReadVariableOp>sequential_53/batch_normalization_502/batchnorm/ReadVariableOp2
@sequential_53/batch_normalization_502/batchnorm/ReadVariableOp_1@sequential_53/batch_normalization_502/batchnorm/ReadVariableOp_12
@sequential_53/batch_normalization_502/batchnorm/ReadVariableOp_2@sequential_53/batch_normalization_502/batchnorm/ReadVariableOp_22
Bsequential_53/batch_normalization_502/batchnorm/mul/ReadVariableOpBsequential_53/batch_normalization_502/batchnorm/mul/ReadVariableOp2
>sequential_53/batch_normalization_503/batchnorm/ReadVariableOp>sequential_53/batch_normalization_503/batchnorm/ReadVariableOp2
@sequential_53/batch_normalization_503/batchnorm/ReadVariableOp_1@sequential_53/batch_normalization_503/batchnorm/ReadVariableOp_12
@sequential_53/batch_normalization_503/batchnorm/ReadVariableOp_2@sequential_53/batch_normalization_503/batchnorm/ReadVariableOp_22
Bsequential_53/batch_normalization_503/batchnorm/mul/ReadVariableOpBsequential_53/batch_normalization_503/batchnorm/mul/ReadVariableOp2
>sequential_53/batch_normalization_504/batchnorm/ReadVariableOp>sequential_53/batch_normalization_504/batchnorm/ReadVariableOp2
@sequential_53/batch_normalization_504/batchnorm/ReadVariableOp_1@sequential_53/batch_normalization_504/batchnorm/ReadVariableOp_12
@sequential_53/batch_normalization_504/batchnorm/ReadVariableOp_2@sequential_53/batch_normalization_504/batchnorm/ReadVariableOp_22
Bsequential_53/batch_normalization_504/batchnorm/mul/ReadVariableOpBsequential_53/batch_normalization_504/batchnorm/mul/ReadVariableOp2
>sequential_53/batch_normalization_505/batchnorm/ReadVariableOp>sequential_53/batch_normalization_505/batchnorm/ReadVariableOp2
@sequential_53/batch_normalization_505/batchnorm/ReadVariableOp_1@sequential_53/batch_normalization_505/batchnorm/ReadVariableOp_12
@sequential_53/batch_normalization_505/batchnorm/ReadVariableOp_2@sequential_53/batch_normalization_505/batchnorm/ReadVariableOp_22
Bsequential_53/batch_normalization_505/batchnorm/mul/ReadVariableOpBsequential_53/batch_normalization_505/batchnorm/mul/ReadVariableOp2
>sequential_53/batch_normalization_506/batchnorm/ReadVariableOp>sequential_53/batch_normalization_506/batchnorm/ReadVariableOp2
@sequential_53/batch_normalization_506/batchnorm/ReadVariableOp_1@sequential_53/batch_normalization_506/batchnorm/ReadVariableOp_12
@sequential_53/batch_normalization_506/batchnorm/ReadVariableOp_2@sequential_53/batch_normalization_506/batchnorm/ReadVariableOp_22
Bsequential_53/batch_normalization_506/batchnorm/mul/ReadVariableOpBsequential_53/batch_normalization_506/batchnorm/mul/ReadVariableOp2
>sequential_53/batch_normalization_507/batchnorm/ReadVariableOp>sequential_53/batch_normalization_507/batchnorm/ReadVariableOp2
@sequential_53/batch_normalization_507/batchnorm/ReadVariableOp_1@sequential_53/batch_normalization_507/batchnorm/ReadVariableOp_12
@sequential_53/batch_normalization_507/batchnorm/ReadVariableOp_2@sequential_53/batch_normalization_507/batchnorm/ReadVariableOp_22
Bsequential_53/batch_normalization_507/batchnorm/mul/ReadVariableOpBsequential_53/batch_normalization_507/batchnorm/mul/ReadVariableOp2`
.sequential_53/dense_552/BiasAdd/ReadVariableOp.sequential_53/dense_552/BiasAdd/ReadVariableOp2^
-sequential_53/dense_552/MatMul/ReadVariableOp-sequential_53/dense_552/MatMul/ReadVariableOp2`
.sequential_53/dense_553/BiasAdd/ReadVariableOp.sequential_53/dense_553/BiasAdd/ReadVariableOp2^
-sequential_53/dense_553/MatMul/ReadVariableOp-sequential_53/dense_553/MatMul/ReadVariableOp2`
.sequential_53/dense_554/BiasAdd/ReadVariableOp.sequential_53/dense_554/BiasAdd/ReadVariableOp2^
-sequential_53/dense_554/MatMul/ReadVariableOp-sequential_53/dense_554/MatMul/ReadVariableOp2`
.sequential_53/dense_555/BiasAdd/ReadVariableOp.sequential_53/dense_555/BiasAdd/ReadVariableOp2^
-sequential_53/dense_555/MatMul/ReadVariableOp-sequential_53/dense_555/MatMul/ReadVariableOp2`
.sequential_53/dense_556/BiasAdd/ReadVariableOp.sequential_53/dense_556/BiasAdd/ReadVariableOp2^
-sequential_53/dense_556/MatMul/ReadVariableOp-sequential_53/dense_556/MatMul/ReadVariableOp2`
.sequential_53/dense_557/BiasAdd/ReadVariableOp.sequential_53/dense_557/BiasAdd/ReadVariableOp2^
-sequential_53/dense_557/MatMul/ReadVariableOp-sequential_53/dense_557/MatMul/ReadVariableOp2`
.sequential_53/dense_558/BiasAdd/ReadVariableOp.sequential_53/dense_558/BiasAdd/ReadVariableOp2^
-sequential_53/dense_558/MatMul/ReadVariableOp-sequential_53/dense_558/MatMul/ReadVariableOp2`
.sequential_53/dense_559/BiasAdd/ReadVariableOp.sequential_53/dense_559/BiasAdd/ReadVariableOp2^
-sequential_53/dense_559/MatMul/ReadVariableOp-sequential_53/dense_559/MatMul/ReadVariableOp2`
.sequential_53/dense_560/BiasAdd/ReadVariableOp.sequential_53/dense_560/BiasAdd/ReadVariableOp2^
-sequential_53/dense_560/MatMul/ReadVariableOp-sequential_53/dense_560/MatMul/ReadVariableOp2`
.sequential_53/dense_561/BiasAdd/ReadVariableOp.sequential_53/dense_561/BiasAdd/ReadVariableOp2^
-sequential_53/dense_561/MatMul/ReadVariableOp-sequential_53/dense_561/MatMul/ReadVariableOp:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_53_input:$ 

_output_shapes

::$ 

_output_shapes

:
î'
Ò
__inference_adapt_step_760279
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢IteratorGetNext¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢add/ReadVariableOp±
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
output_shapes
:ÿÿÿÿÿÿÿÿÿ*
output_types
2	k
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0P
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:X
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:G
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0V
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0V
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:E
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:V
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @N
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:Z
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:I
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:I
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:
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
%
ì
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_760923

inputs5
'assignmovingavg_readvariableop_resource:a7
)assignmovingavg_1_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a/
!batchnorm_readvariableop_resource:a
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:a
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿal
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:a*
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
:a*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a¬
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
:a*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:a~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a´
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿah
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
 ¯
;
I__inference_sequential_53_layer_call_and_return_conditional_losses_760109

inputs
normalization_53_sub_y
normalization_53_sqrt_x:
(dense_552_matmul_readvariableop_resource:P7
)dense_552_biasadd_readvariableop_resource:PM
?batch_normalization_499_assignmovingavg_readvariableop_resource:PO
Abatch_normalization_499_assignmovingavg_1_readvariableop_resource:PK
=batch_normalization_499_batchnorm_mul_readvariableop_resource:PG
9batch_normalization_499_batchnorm_readvariableop_resource:P:
(dense_553_matmul_readvariableop_resource:PP7
)dense_553_biasadd_readvariableop_resource:PM
?batch_normalization_500_assignmovingavg_readvariableop_resource:PO
Abatch_normalization_500_assignmovingavg_1_readvariableop_resource:PK
=batch_normalization_500_batchnorm_mul_readvariableop_resource:PG
9batch_normalization_500_batchnorm_readvariableop_resource:P:
(dense_554_matmul_readvariableop_resource:Pa7
)dense_554_biasadd_readvariableop_resource:aM
?batch_normalization_501_assignmovingavg_readvariableop_resource:aO
Abatch_normalization_501_assignmovingavg_1_readvariableop_resource:aK
=batch_normalization_501_batchnorm_mul_readvariableop_resource:aG
9batch_normalization_501_batchnorm_readvariableop_resource:a:
(dense_555_matmul_readvariableop_resource:aa7
)dense_555_biasadd_readvariableop_resource:aM
?batch_normalization_502_assignmovingavg_readvariableop_resource:aO
Abatch_normalization_502_assignmovingavg_1_readvariableop_resource:aK
=batch_normalization_502_batchnorm_mul_readvariableop_resource:aG
9batch_normalization_502_batchnorm_readvariableop_resource:a:
(dense_556_matmul_readvariableop_resource:aa7
)dense_556_biasadd_readvariableop_resource:aM
?batch_normalization_503_assignmovingavg_readvariableop_resource:aO
Abatch_normalization_503_assignmovingavg_1_readvariableop_resource:aK
=batch_normalization_503_batchnorm_mul_readvariableop_resource:aG
9batch_normalization_503_batchnorm_readvariableop_resource:a:
(dense_557_matmul_readvariableop_resource:aa7
)dense_557_biasadd_readvariableop_resource:aM
?batch_normalization_504_assignmovingavg_readvariableop_resource:aO
Abatch_normalization_504_assignmovingavg_1_readvariableop_resource:aK
=batch_normalization_504_batchnorm_mul_readvariableop_resource:aG
9batch_normalization_504_batchnorm_readvariableop_resource:a:
(dense_558_matmul_readvariableop_resource:a=7
)dense_558_biasadd_readvariableop_resource:=M
?batch_normalization_505_assignmovingavg_readvariableop_resource:=O
Abatch_normalization_505_assignmovingavg_1_readvariableop_resource:=K
=batch_normalization_505_batchnorm_mul_readvariableop_resource:=G
9batch_normalization_505_batchnorm_readvariableop_resource:=:
(dense_559_matmul_readvariableop_resource:==7
)dense_559_biasadd_readvariableop_resource:=M
?batch_normalization_506_assignmovingavg_readvariableop_resource:=O
Abatch_normalization_506_assignmovingavg_1_readvariableop_resource:=K
=batch_normalization_506_batchnorm_mul_readvariableop_resource:=G
9batch_normalization_506_batchnorm_readvariableop_resource:=:
(dense_560_matmul_readvariableop_resource:==7
)dense_560_biasadd_readvariableop_resource:=M
?batch_normalization_507_assignmovingavg_readvariableop_resource:=O
Abatch_normalization_507_assignmovingavg_1_readvariableop_resource:=K
=batch_normalization_507_batchnorm_mul_readvariableop_resource:=G
9batch_normalization_507_batchnorm_readvariableop_resource:=:
(dense_561_matmul_readvariableop_resource:=7
)dense_561_biasadd_readvariableop_resource:
identity¢'batch_normalization_499/AssignMovingAvg¢6batch_normalization_499/AssignMovingAvg/ReadVariableOp¢)batch_normalization_499/AssignMovingAvg_1¢8batch_normalization_499/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_499/batchnorm/ReadVariableOp¢4batch_normalization_499/batchnorm/mul/ReadVariableOp¢'batch_normalization_500/AssignMovingAvg¢6batch_normalization_500/AssignMovingAvg/ReadVariableOp¢)batch_normalization_500/AssignMovingAvg_1¢8batch_normalization_500/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_500/batchnorm/ReadVariableOp¢4batch_normalization_500/batchnorm/mul/ReadVariableOp¢'batch_normalization_501/AssignMovingAvg¢6batch_normalization_501/AssignMovingAvg/ReadVariableOp¢)batch_normalization_501/AssignMovingAvg_1¢8batch_normalization_501/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_501/batchnorm/ReadVariableOp¢4batch_normalization_501/batchnorm/mul/ReadVariableOp¢'batch_normalization_502/AssignMovingAvg¢6batch_normalization_502/AssignMovingAvg/ReadVariableOp¢)batch_normalization_502/AssignMovingAvg_1¢8batch_normalization_502/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_502/batchnorm/ReadVariableOp¢4batch_normalization_502/batchnorm/mul/ReadVariableOp¢'batch_normalization_503/AssignMovingAvg¢6batch_normalization_503/AssignMovingAvg/ReadVariableOp¢)batch_normalization_503/AssignMovingAvg_1¢8batch_normalization_503/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_503/batchnorm/ReadVariableOp¢4batch_normalization_503/batchnorm/mul/ReadVariableOp¢'batch_normalization_504/AssignMovingAvg¢6batch_normalization_504/AssignMovingAvg/ReadVariableOp¢)batch_normalization_504/AssignMovingAvg_1¢8batch_normalization_504/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_504/batchnorm/ReadVariableOp¢4batch_normalization_504/batchnorm/mul/ReadVariableOp¢'batch_normalization_505/AssignMovingAvg¢6batch_normalization_505/AssignMovingAvg/ReadVariableOp¢)batch_normalization_505/AssignMovingAvg_1¢8batch_normalization_505/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_505/batchnorm/ReadVariableOp¢4batch_normalization_505/batchnorm/mul/ReadVariableOp¢'batch_normalization_506/AssignMovingAvg¢6batch_normalization_506/AssignMovingAvg/ReadVariableOp¢)batch_normalization_506/AssignMovingAvg_1¢8batch_normalization_506/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_506/batchnorm/ReadVariableOp¢4batch_normalization_506/batchnorm/mul/ReadVariableOp¢'batch_normalization_507/AssignMovingAvg¢6batch_normalization_507/AssignMovingAvg/ReadVariableOp¢)batch_normalization_507/AssignMovingAvg_1¢8batch_normalization_507/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_507/batchnorm/ReadVariableOp¢4batch_normalization_507/batchnorm/mul/ReadVariableOp¢ dense_552/BiasAdd/ReadVariableOp¢dense_552/MatMul/ReadVariableOp¢ dense_553/BiasAdd/ReadVariableOp¢dense_553/MatMul/ReadVariableOp¢ dense_554/BiasAdd/ReadVariableOp¢dense_554/MatMul/ReadVariableOp¢ dense_555/BiasAdd/ReadVariableOp¢dense_555/MatMul/ReadVariableOp¢ dense_556/BiasAdd/ReadVariableOp¢dense_556/MatMul/ReadVariableOp¢ dense_557/BiasAdd/ReadVariableOp¢dense_557/MatMul/ReadVariableOp¢ dense_558/BiasAdd/ReadVariableOp¢dense_558/MatMul/ReadVariableOp¢ dense_559/BiasAdd/ReadVariableOp¢dense_559/MatMul/ReadVariableOp¢ dense_560/BiasAdd/ReadVariableOp¢dense_560/MatMul/ReadVariableOp¢ dense_561/BiasAdd/ReadVariableOp¢dense_561/MatMul/ReadVariableOpm
normalization_53/subSubinputsnormalization_53_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_53/SqrtSqrtnormalization_53_sqrt_x*
T0*
_output_shapes

:_
normalization_53/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_53/MaximumMaximumnormalization_53/Sqrt:y:0#normalization_53/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_53/truedivRealDivnormalization_53/sub:z:0normalization_53/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_552/MatMul/ReadVariableOpReadVariableOp(dense_552_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0
dense_552/MatMulMatMulnormalization_53/truediv:z:0'dense_552/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 dense_552/BiasAdd/ReadVariableOpReadVariableOp)dense_552_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
dense_552/BiasAddBiasAdddense_552/MatMul:product:0(dense_552/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
6batch_normalization_499/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_499/moments/meanMeandense_552/BiasAdd:output:0?batch_normalization_499/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(
,batch_normalization_499/moments/StopGradientStopGradient-batch_normalization_499/moments/mean:output:0*
T0*
_output_shapes

:PË
1batch_normalization_499/moments/SquaredDifferenceSquaredDifferencedense_552/BiasAdd:output:05batch_normalization_499/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
:batch_normalization_499/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_499/moments/varianceMean5batch_normalization_499/moments/SquaredDifference:z:0Cbatch_normalization_499/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(
'batch_normalization_499/moments/SqueezeSqueeze-batch_normalization_499/moments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 £
)batch_normalization_499/moments/Squeeze_1Squeeze1batch_normalization_499/moments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 r
-batch_normalization_499/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_499/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_499_assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0É
+batch_normalization_499/AssignMovingAvg/subSub>batch_normalization_499/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_499/moments/Squeeze:output:0*
T0*
_output_shapes
:PÀ
+batch_normalization_499/AssignMovingAvg/mulMul/batch_normalization_499/AssignMovingAvg/sub:z:06batch_normalization_499/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P
'batch_normalization_499/AssignMovingAvgAssignSubVariableOp?batch_normalization_499_assignmovingavg_readvariableop_resource/batch_normalization_499/AssignMovingAvg/mul:z:07^batch_normalization_499/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_499/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_499/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_499_assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0Ï
-batch_normalization_499/AssignMovingAvg_1/subSub@batch_normalization_499/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_499/moments/Squeeze_1:output:0*
T0*
_output_shapes
:PÆ
-batch_normalization_499/AssignMovingAvg_1/mulMul1batch_normalization_499/AssignMovingAvg_1/sub:z:08batch_normalization_499/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P
)batch_normalization_499/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_499_assignmovingavg_1_readvariableop_resource1batch_normalization_499/AssignMovingAvg_1/mul:z:09^batch_normalization_499/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_499/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_499/batchnorm/addAddV22batch_normalization_499/moments/Squeeze_1:output:00batch_normalization_499/batchnorm/add/y:output:0*
T0*
_output_shapes
:P
'batch_normalization_499/batchnorm/RsqrtRsqrt)batch_normalization_499/batchnorm/add:z:0*
T0*
_output_shapes
:P®
4batch_normalization_499/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_499_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0¼
%batch_normalization_499/batchnorm/mulMul+batch_normalization_499/batchnorm/Rsqrt:y:0<batch_normalization_499/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P§
'batch_normalization_499/batchnorm/mul_1Muldense_552/BiasAdd:output:0)batch_normalization_499/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP°
'batch_normalization_499/batchnorm/mul_2Mul0batch_normalization_499/moments/Squeeze:output:0)batch_normalization_499/batchnorm/mul:z:0*
T0*
_output_shapes
:P¦
0batch_normalization_499/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_499_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0¸
%batch_normalization_499/batchnorm/subSub8batch_normalization_499/batchnorm/ReadVariableOp:value:0+batch_normalization_499/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pº
'batch_normalization_499/batchnorm/add_1AddV2+batch_normalization_499/batchnorm/mul_1:z:0)batch_normalization_499/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
leaky_re_lu_499/LeakyRelu	LeakyRelu+batch_normalization_499/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*
alpha%>
dense_553/MatMul/ReadVariableOpReadVariableOp(dense_553_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0
dense_553/MatMulMatMul'leaky_re_lu_499/LeakyRelu:activations:0'dense_553/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 dense_553/BiasAdd/ReadVariableOpReadVariableOp)dense_553_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
dense_553/BiasAddBiasAdddense_553/MatMul:product:0(dense_553/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
6batch_normalization_500/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_500/moments/meanMeandense_553/BiasAdd:output:0?batch_normalization_500/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(
,batch_normalization_500/moments/StopGradientStopGradient-batch_normalization_500/moments/mean:output:0*
T0*
_output_shapes

:PË
1batch_normalization_500/moments/SquaredDifferenceSquaredDifferencedense_553/BiasAdd:output:05batch_normalization_500/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
:batch_normalization_500/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_500/moments/varianceMean5batch_normalization_500/moments/SquaredDifference:z:0Cbatch_normalization_500/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(
'batch_normalization_500/moments/SqueezeSqueeze-batch_normalization_500/moments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 £
)batch_normalization_500/moments/Squeeze_1Squeeze1batch_normalization_500/moments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 r
-batch_normalization_500/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_500/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_500_assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0É
+batch_normalization_500/AssignMovingAvg/subSub>batch_normalization_500/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_500/moments/Squeeze:output:0*
T0*
_output_shapes
:PÀ
+batch_normalization_500/AssignMovingAvg/mulMul/batch_normalization_500/AssignMovingAvg/sub:z:06batch_normalization_500/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P
'batch_normalization_500/AssignMovingAvgAssignSubVariableOp?batch_normalization_500_assignmovingavg_readvariableop_resource/batch_normalization_500/AssignMovingAvg/mul:z:07^batch_normalization_500/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_500/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_500/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_500_assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0Ï
-batch_normalization_500/AssignMovingAvg_1/subSub@batch_normalization_500/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_500/moments/Squeeze_1:output:0*
T0*
_output_shapes
:PÆ
-batch_normalization_500/AssignMovingAvg_1/mulMul1batch_normalization_500/AssignMovingAvg_1/sub:z:08batch_normalization_500/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P
)batch_normalization_500/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_500_assignmovingavg_1_readvariableop_resource1batch_normalization_500/AssignMovingAvg_1/mul:z:09^batch_normalization_500/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_500/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_500/batchnorm/addAddV22batch_normalization_500/moments/Squeeze_1:output:00batch_normalization_500/batchnorm/add/y:output:0*
T0*
_output_shapes
:P
'batch_normalization_500/batchnorm/RsqrtRsqrt)batch_normalization_500/batchnorm/add:z:0*
T0*
_output_shapes
:P®
4batch_normalization_500/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_500_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0¼
%batch_normalization_500/batchnorm/mulMul+batch_normalization_500/batchnorm/Rsqrt:y:0<batch_normalization_500/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P§
'batch_normalization_500/batchnorm/mul_1Muldense_553/BiasAdd:output:0)batch_normalization_500/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP°
'batch_normalization_500/batchnorm/mul_2Mul0batch_normalization_500/moments/Squeeze:output:0)batch_normalization_500/batchnorm/mul:z:0*
T0*
_output_shapes
:P¦
0batch_normalization_500/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_500_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0¸
%batch_normalization_500/batchnorm/subSub8batch_normalization_500/batchnorm/ReadVariableOp:value:0+batch_normalization_500/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pº
'batch_normalization_500/batchnorm/add_1AddV2+batch_normalization_500/batchnorm/mul_1:z:0)batch_normalization_500/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
leaky_re_lu_500/LeakyRelu	LeakyRelu+batch_normalization_500/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*
alpha%>
dense_554/MatMul/ReadVariableOpReadVariableOp(dense_554_matmul_readvariableop_resource*
_output_shapes

:Pa*
dtype0
dense_554/MatMulMatMul'leaky_re_lu_500/LeakyRelu:activations:0'dense_554/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 dense_554/BiasAdd/ReadVariableOpReadVariableOp)dense_554_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
dense_554/BiasAddBiasAdddense_554/MatMul:product:0(dense_554/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
6batch_normalization_501/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_501/moments/meanMeandense_554/BiasAdd:output:0?batch_normalization_501/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(
,batch_normalization_501/moments/StopGradientStopGradient-batch_normalization_501/moments/mean:output:0*
T0*
_output_shapes

:aË
1batch_normalization_501/moments/SquaredDifferenceSquaredDifferencedense_554/BiasAdd:output:05batch_normalization_501/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
:batch_normalization_501/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_501/moments/varianceMean5batch_normalization_501/moments/SquaredDifference:z:0Cbatch_normalization_501/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(
'batch_normalization_501/moments/SqueezeSqueeze-batch_normalization_501/moments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 £
)batch_normalization_501/moments/Squeeze_1Squeeze1batch_normalization_501/moments/variance:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 r
-batch_normalization_501/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_501/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_501_assignmovingavg_readvariableop_resource*
_output_shapes
:a*
dtype0É
+batch_normalization_501/AssignMovingAvg/subSub>batch_normalization_501/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_501/moments/Squeeze:output:0*
T0*
_output_shapes
:aÀ
+batch_normalization_501/AssignMovingAvg/mulMul/batch_normalization_501/AssignMovingAvg/sub:z:06batch_normalization_501/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a
'batch_normalization_501/AssignMovingAvgAssignSubVariableOp?batch_normalization_501_assignmovingavg_readvariableop_resource/batch_normalization_501/AssignMovingAvg/mul:z:07^batch_normalization_501/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_501/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_501/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_501_assignmovingavg_1_readvariableop_resource*
_output_shapes
:a*
dtype0Ï
-batch_normalization_501/AssignMovingAvg_1/subSub@batch_normalization_501/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_501/moments/Squeeze_1:output:0*
T0*
_output_shapes
:aÆ
-batch_normalization_501/AssignMovingAvg_1/mulMul1batch_normalization_501/AssignMovingAvg_1/sub:z:08batch_normalization_501/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a
)batch_normalization_501/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_501_assignmovingavg_1_readvariableop_resource1batch_normalization_501/AssignMovingAvg_1/mul:z:09^batch_normalization_501/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_501/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_501/batchnorm/addAddV22batch_normalization_501/moments/Squeeze_1:output:00batch_normalization_501/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
'batch_normalization_501/batchnorm/RsqrtRsqrt)batch_normalization_501/batchnorm/add:z:0*
T0*
_output_shapes
:a®
4batch_normalization_501/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_501_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0¼
%batch_normalization_501/batchnorm/mulMul+batch_normalization_501/batchnorm/Rsqrt:y:0<batch_normalization_501/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:a§
'batch_normalization_501/batchnorm/mul_1Muldense_554/BiasAdd:output:0)batch_normalization_501/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa°
'batch_normalization_501/batchnorm/mul_2Mul0batch_normalization_501/moments/Squeeze:output:0)batch_normalization_501/batchnorm/mul:z:0*
T0*
_output_shapes
:a¦
0batch_normalization_501/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_501_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0¸
%batch_normalization_501/batchnorm/subSub8batch_normalization_501/batchnorm/ReadVariableOp:value:0+batch_normalization_501/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aº
'batch_normalization_501/batchnorm/add_1AddV2+batch_normalization_501/batchnorm/mul_1:z:0)batch_normalization_501/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
leaky_re_lu_501/LeakyRelu	LeakyRelu+batch_normalization_501/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>
dense_555/MatMul/ReadVariableOpReadVariableOp(dense_555_matmul_readvariableop_resource*
_output_shapes

:aa*
dtype0
dense_555/MatMulMatMul'leaky_re_lu_501/LeakyRelu:activations:0'dense_555/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 dense_555/BiasAdd/ReadVariableOpReadVariableOp)dense_555_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
dense_555/BiasAddBiasAdddense_555/MatMul:product:0(dense_555/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
6batch_normalization_502/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_502/moments/meanMeandense_555/BiasAdd:output:0?batch_normalization_502/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(
,batch_normalization_502/moments/StopGradientStopGradient-batch_normalization_502/moments/mean:output:0*
T0*
_output_shapes

:aË
1batch_normalization_502/moments/SquaredDifferenceSquaredDifferencedense_555/BiasAdd:output:05batch_normalization_502/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
:batch_normalization_502/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_502/moments/varianceMean5batch_normalization_502/moments/SquaredDifference:z:0Cbatch_normalization_502/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(
'batch_normalization_502/moments/SqueezeSqueeze-batch_normalization_502/moments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 £
)batch_normalization_502/moments/Squeeze_1Squeeze1batch_normalization_502/moments/variance:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 r
-batch_normalization_502/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_502/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_502_assignmovingavg_readvariableop_resource*
_output_shapes
:a*
dtype0É
+batch_normalization_502/AssignMovingAvg/subSub>batch_normalization_502/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_502/moments/Squeeze:output:0*
T0*
_output_shapes
:aÀ
+batch_normalization_502/AssignMovingAvg/mulMul/batch_normalization_502/AssignMovingAvg/sub:z:06batch_normalization_502/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a
'batch_normalization_502/AssignMovingAvgAssignSubVariableOp?batch_normalization_502_assignmovingavg_readvariableop_resource/batch_normalization_502/AssignMovingAvg/mul:z:07^batch_normalization_502/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_502/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_502/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_502_assignmovingavg_1_readvariableop_resource*
_output_shapes
:a*
dtype0Ï
-batch_normalization_502/AssignMovingAvg_1/subSub@batch_normalization_502/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_502/moments/Squeeze_1:output:0*
T0*
_output_shapes
:aÆ
-batch_normalization_502/AssignMovingAvg_1/mulMul1batch_normalization_502/AssignMovingAvg_1/sub:z:08batch_normalization_502/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a
)batch_normalization_502/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_502_assignmovingavg_1_readvariableop_resource1batch_normalization_502/AssignMovingAvg_1/mul:z:09^batch_normalization_502/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_502/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_502/batchnorm/addAddV22batch_normalization_502/moments/Squeeze_1:output:00batch_normalization_502/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
'batch_normalization_502/batchnorm/RsqrtRsqrt)batch_normalization_502/batchnorm/add:z:0*
T0*
_output_shapes
:a®
4batch_normalization_502/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_502_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0¼
%batch_normalization_502/batchnorm/mulMul+batch_normalization_502/batchnorm/Rsqrt:y:0<batch_normalization_502/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:a§
'batch_normalization_502/batchnorm/mul_1Muldense_555/BiasAdd:output:0)batch_normalization_502/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa°
'batch_normalization_502/batchnorm/mul_2Mul0batch_normalization_502/moments/Squeeze:output:0)batch_normalization_502/batchnorm/mul:z:0*
T0*
_output_shapes
:a¦
0batch_normalization_502/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_502_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0¸
%batch_normalization_502/batchnorm/subSub8batch_normalization_502/batchnorm/ReadVariableOp:value:0+batch_normalization_502/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aº
'batch_normalization_502/batchnorm/add_1AddV2+batch_normalization_502/batchnorm/mul_1:z:0)batch_normalization_502/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
leaky_re_lu_502/LeakyRelu	LeakyRelu+batch_normalization_502/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>
dense_556/MatMul/ReadVariableOpReadVariableOp(dense_556_matmul_readvariableop_resource*
_output_shapes

:aa*
dtype0
dense_556/MatMulMatMul'leaky_re_lu_502/LeakyRelu:activations:0'dense_556/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 dense_556/BiasAdd/ReadVariableOpReadVariableOp)dense_556_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
dense_556/BiasAddBiasAdddense_556/MatMul:product:0(dense_556/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
6batch_normalization_503/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_503/moments/meanMeandense_556/BiasAdd:output:0?batch_normalization_503/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(
,batch_normalization_503/moments/StopGradientStopGradient-batch_normalization_503/moments/mean:output:0*
T0*
_output_shapes

:aË
1batch_normalization_503/moments/SquaredDifferenceSquaredDifferencedense_556/BiasAdd:output:05batch_normalization_503/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
:batch_normalization_503/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_503/moments/varianceMean5batch_normalization_503/moments/SquaredDifference:z:0Cbatch_normalization_503/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(
'batch_normalization_503/moments/SqueezeSqueeze-batch_normalization_503/moments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 £
)batch_normalization_503/moments/Squeeze_1Squeeze1batch_normalization_503/moments/variance:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 r
-batch_normalization_503/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_503/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_503_assignmovingavg_readvariableop_resource*
_output_shapes
:a*
dtype0É
+batch_normalization_503/AssignMovingAvg/subSub>batch_normalization_503/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_503/moments/Squeeze:output:0*
T0*
_output_shapes
:aÀ
+batch_normalization_503/AssignMovingAvg/mulMul/batch_normalization_503/AssignMovingAvg/sub:z:06batch_normalization_503/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a
'batch_normalization_503/AssignMovingAvgAssignSubVariableOp?batch_normalization_503_assignmovingavg_readvariableop_resource/batch_normalization_503/AssignMovingAvg/mul:z:07^batch_normalization_503/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_503/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_503/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_503_assignmovingavg_1_readvariableop_resource*
_output_shapes
:a*
dtype0Ï
-batch_normalization_503/AssignMovingAvg_1/subSub@batch_normalization_503/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_503/moments/Squeeze_1:output:0*
T0*
_output_shapes
:aÆ
-batch_normalization_503/AssignMovingAvg_1/mulMul1batch_normalization_503/AssignMovingAvg_1/sub:z:08batch_normalization_503/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a
)batch_normalization_503/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_503_assignmovingavg_1_readvariableop_resource1batch_normalization_503/AssignMovingAvg_1/mul:z:09^batch_normalization_503/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_503/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_503/batchnorm/addAddV22batch_normalization_503/moments/Squeeze_1:output:00batch_normalization_503/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
'batch_normalization_503/batchnorm/RsqrtRsqrt)batch_normalization_503/batchnorm/add:z:0*
T0*
_output_shapes
:a®
4batch_normalization_503/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_503_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0¼
%batch_normalization_503/batchnorm/mulMul+batch_normalization_503/batchnorm/Rsqrt:y:0<batch_normalization_503/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:a§
'batch_normalization_503/batchnorm/mul_1Muldense_556/BiasAdd:output:0)batch_normalization_503/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa°
'batch_normalization_503/batchnorm/mul_2Mul0batch_normalization_503/moments/Squeeze:output:0)batch_normalization_503/batchnorm/mul:z:0*
T0*
_output_shapes
:a¦
0batch_normalization_503/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_503_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0¸
%batch_normalization_503/batchnorm/subSub8batch_normalization_503/batchnorm/ReadVariableOp:value:0+batch_normalization_503/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aº
'batch_normalization_503/batchnorm/add_1AddV2+batch_normalization_503/batchnorm/mul_1:z:0)batch_normalization_503/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
leaky_re_lu_503/LeakyRelu	LeakyRelu+batch_normalization_503/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>
dense_557/MatMul/ReadVariableOpReadVariableOp(dense_557_matmul_readvariableop_resource*
_output_shapes

:aa*
dtype0
dense_557/MatMulMatMul'leaky_re_lu_503/LeakyRelu:activations:0'dense_557/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 dense_557/BiasAdd/ReadVariableOpReadVariableOp)dense_557_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
dense_557/BiasAddBiasAdddense_557/MatMul:product:0(dense_557/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
6batch_normalization_504/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_504/moments/meanMeandense_557/BiasAdd:output:0?batch_normalization_504/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(
,batch_normalization_504/moments/StopGradientStopGradient-batch_normalization_504/moments/mean:output:0*
T0*
_output_shapes

:aË
1batch_normalization_504/moments/SquaredDifferenceSquaredDifferencedense_557/BiasAdd:output:05batch_normalization_504/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
:batch_normalization_504/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_504/moments/varianceMean5batch_normalization_504/moments/SquaredDifference:z:0Cbatch_normalization_504/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(
'batch_normalization_504/moments/SqueezeSqueeze-batch_normalization_504/moments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 £
)batch_normalization_504/moments/Squeeze_1Squeeze1batch_normalization_504/moments/variance:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 r
-batch_normalization_504/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_504/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_504_assignmovingavg_readvariableop_resource*
_output_shapes
:a*
dtype0É
+batch_normalization_504/AssignMovingAvg/subSub>batch_normalization_504/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_504/moments/Squeeze:output:0*
T0*
_output_shapes
:aÀ
+batch_normalization_504/AssignMovingAvg/mulMul/batch_normalization_504/AssignMovingAvg/sub:z:06batch_normalization_504/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a
'batch_normalization_504/AssignMovingAvgAssignSubVariableOp?batch_normalization_504_assignmovingavg_readvariableop_resource/batch_normalization_504/AssignMovingAvg/mul:z:07^batch_normalization_504/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_504/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_504/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_504_assignmovingavg_1_readvariableop_resource*
_output_shapes
:a*
dtype0Ï
-batch_normalization_504/AssignMovingAvg_1/subSub@batch_normalization_504/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_504/moments/Squeeze_1:output:0*
T0*
_output_shapes
:aÆ
-batch_normalization_504/AssignMovingAvg_1/mulMul1batch_normalization_504/AssignMovingAvg_1/sub:z:08batch_normalization_504/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a
)batch_normalization_504/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_504_assignmovingavg_1_readvariableop_resource1batch_normalization_504/AssignMovingAvg_1/mul:z:09^batch_normalization_504/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_504/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_504/batchnorm/addAddV22batch_normalization_504/moments/Squeeze_1:output:00batch_normalization_504/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
'batch_normalization_504/batchnorm/RsqrtRsqrt)batch_normalization_504/batchnorm/add:z:0*
T0*
_output_shapes
:a®
4batch_normalization_504/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_504_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0¼
%batch_normalization_504/batchnorm/mulMul+batch_normalization_504/batchnorm/Rsqrt:y:0<batch_normalization_504/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:a§
'batch_normalization_504/batchnorm/mul_1Muldense_557/BiasAdd:output:0)batch_normalization_504/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa°
'batch_normalization_504/batchnorm/mul_2Mul0batch_normalization_504/moments/Squeeze:output:0)batch_normalization_504/batchnorm/mul:z:0*
T0*
_output_shapes
:a¦
0batch_normalization_504/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_504_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0¸
%batch_normalization_504/batchnorm/subSub8batch_normalization_504/batchnorm/ReadVariableOp:value:0+batch_normalization_504/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aº
'batch_normalization_504/batchnorm/add_1AddV2+batch_normalization_504/batchnorm/mul_1:z:0)batch_normalization_504/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
leaky_re_lu_504/LeakyRelu	LeakyRelu+batch_normalization_504/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>
dense_558/MatMul/ReadVariableOpReadVariableOp(dense_558_matmul_readvariableop_resource*
_output_shapes

:a=*
dtype0
dense_558/MatMulMatMul'leaky_re_lu_504/LeakyRelu:activations:0'dense_558/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 dense_558/BiasAdd/ReadVariableOpReadVariableOp)dense_558_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0
dense_558/BiasAddBiasAdddense_558/MatMul:product:0(dense_558/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
6batch_normalization_505/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_505/moments/meanMeandense_558/BiasAdd:output:0?batch_normalization_505/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(
,batch_normalization_505/moments/StopGradientStopGradient-batch_normalization_505/moments/mean:output:0*
T0*
_output_shapes

:=Ë
1batch_normalization_505/moments/SquaredDifferenceSquaredDifferencedense_558/BiasAdd:output:05batch_normalization_505/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
:batch_normalization_505/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_505/moments/varianceMean5batch_normalization_505/moments/SquaredDifference:z:0Cbatch_normalization_505/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(
'batch_normalization_505/moments/SqueezeSqueeze-batch_normalization_505/moments/mean:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 £
)batch_normalization_505/moments/Squeeze_1Squeeze1batch_normalization_505/moments/variance:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 r
-batch_normalization_505/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_505/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_505_assignmovingavg_readvariableop_resource*
_output_shapes
:=*
dtype0É
+batch_normalization_505/AssignMovingAvg/subSub>batch_normalization_505/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_505/moments/Squeeze:output:0*
T0*
_output_shapes
:=À
+batch_normalization_505/AssignMovingAvg/mulMul/batch_normalization_505/AssignMovingAvg/sub:z:06batch_normalization_505/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:=
'batch_normalization_505/AssignMovingAvgAssignSubVariableOp?batch_normalization_505_assignmovingavg_readvariableop_resource/batch_normalization_505/AssignMovingAvg/mul:z:07^batch_normalization_505/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_505/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_505/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_505_assignmovingavg_1_readvariableop_resource*
_output_shapes
:=*
dtype0Ï
-batch_normalization_505/AssignMovingAvg_1/subSub@batch_normalization_505/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_505/moments/Squeeze_1:output:0*
T0*
_output_shapes
:=Æ
-batch_normalization_505/AssignMovingAvg_1/mulMul1batch_normalization_505/AssignMovingAvg_1/sub:z:08batch_normalization_505/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:=
)batch_normalization_505/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_505_assignmovingavg_1_readvariableop_resource1batch_normalization_505/AssignMovingAvg_1/mul:z:09^batch_normalization_505/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_505/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_505/batchnorm/addAddV22batch_normalization_505/moments/Squeeze_1:output:00batch_normalization_505/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
'batch_normalization_505/batchnorm/RsqrtRsqrt)batch_normalization_505/batchnorm/add:z:0*
T0*
_output_shapes
:=®
4batch_normalization_505/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_505_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0¼
%batch_normalization_505/batchnorm/mulMul+batch_normalization_505/batchnorm/Rsqrt:y:0<batch_normalization_505/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=§
'batch_normalization_505/batchnorm/mul_1Muldense_558/BiasAdd:output:0)batch_normalization_505/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=°
'batch_normalization_505/batchnorm/mul_2Mul0batch_normalization_505/moments/Squeeze:output:0)batch_normalization_505/batchnorm/mul:z:0*
T0*
_output_shapes
:=¦
0batch_normalization_505/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_505_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0¸
%batch_normalization_505/batchnorm/subSub8batch_normalization_505/batchnorm/ReadVariableOp:value:0+batch_normalization_505/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=º
'batch_normalization_505/batchnorm/add_1AddV2+batch_normalization_505/batchnorm/mul_1:z:0)batch_normalization_505/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
leaky_re_lu_505/LeakyRelu	LeakyRelu+batch_normalization_505/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>
dense_559/MatMul/ReadVariableOpReadVariableOp(dense_559_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0
dense_559/MatMulMatMul'leaky_re_lu_505/LeakyRelu:activations:0'dense_559/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 dense_559/BiasAdd/ReadVariableOpReadVariableOp)dense_559_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0
dense_559/BiasAddBiasAdddense_559/MatMul:product:0(dense_559/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
6batch_normalization_506/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_506/moments/meanMeandense_559/BiasAdd:output:0?batch_normalization_506/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(
,batch_normalization_506/moments/StopGradientStopGradient-batch_normalization_506/moments/mean:output:0*
T0*
_output_shapes

:=Ë
1batch_normalization_506/moments/SquaredDifferenceSquaredDifferencedense_559/BiasAdd:output:05batch_normalization_506/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
:batch_normalization_506/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_506/moments/varianceMean5batch_normalization_506/moments/SquaredDifference:z:0Cbatch_normalization_506/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(
'batch_normalization_506/moments/SqueezeSqueeze-batch_normalization_506/moments/mean:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 £
)batch_normalization_506/moments/Squeeze_1Squeeze1batch_normalization_506/moments/variance:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 r
-batch_normalization_506/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_506/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_506_assignmovingavg_readvariableop_resource*
_output_shapes
:=*
dtype0É
+batch_normalization_506/AssignMovingAvg/subSub>batch_normalization_506/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_506/moments/Squeeze:output:0*
T0*
_output_shapes
:=À
+batch_normalization_506/AssignMovingAvg/mulMul/batch_normalization_506/AssignMovingAvg/sub:z:06batch_normalization_506/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:=
'batch_normalization_506/AssignMovingAvgAssignSubVariableOp?batch_normalization_506_assignmovingavg_readvariableop_resource/batch_normalization_506/AssignMovingAvg/mul:z:07^batch_normalization_506/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_506/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_506/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_506_assignmovingavg_1_readvariableop_resource*
_output_shapes
:=*
dtype0Ï
-batch_normalization_506/AssignMovingAvg_1/subSub@batch_normalization_506/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_506/moments/Squeeze_1:output:0*
T0*
_output_shapes
:=Æ
-batch_normalization_506/AssignMovingAvg_1/mulMul1batch_normalization_506/AssignMovingAvg_1/sub:z:08batch_normalization_506/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:=
)batch_normalization_506/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_506_assignmovingavg_1_readvariableop_resource1batch_normalization_506/AssignMovingAvg_1/mul:z:09^batch_normalization_506/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_506/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_506/batchnorm/addAddV22batch_normalization_506/moments/Squeeze_1:output:00batch_normalization_506/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
'batch_normalization_506/batchnorm/RsqrtRsqrt)batch_normalization_506/batchnorm/add:z:0*
T0*
_output_shapes
:=®
4batch_normalization_506/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_506_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0¼
%batch_normalization_506/batchnorm/mulMul+batch_normalization_506/batchnorm/Rsqrt:y:0<batch_normalization_506/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=§
'batch_normalization_506/batchnorm/mul_1Muldense_559/BiasAdd:output:0)batch_normalization_506/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=°
'batch_normalization_506/batchnorm/mul_2Mul0batch_normalization_506/moments/Squeeze:output:0)batch_normalization_506/batchnorm/mul:z:0*
T0*
_output_shapes
:=¦
0batch_normalization_506/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_506_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0¸
%batch_normalization_506/batchnorm/subSub8batch_normalization_506/batchnorm/ReadVariableOp:value:0+batch_normalization_506/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=º
'batch_normalization_506/batchnorm/add_1AddV2+batch_normalization_506/batchnorm/mul_1:z:0)batch_normalization_506/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
leaky_re_lu_506/LeakyRelu	LeakyRelu+batch_normalization_506/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>
dense_560/MatMul/ReadVariableOpReadVariableOp(dense_560_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0
dense_560/MatMulMatMul'leaky_re_lu_506/LeakyRelu:activations:0'dense_560/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 dense_560/BiasAdd/ReadVariableOpReadVariableOp)dense_560_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0
dense_560/BiasAddBiasAdddense_560/MatMul:product:0(dense_560/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
6batch_normalization_507/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_507/moments/meanMeandense_560/BiasAdd:output:0?batch_normalization_507/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(
,batch_normalization_507/moments/StopGradientStopGradient-batch_normalization_507/moments/mean:output:0*
T0*
_output_shapes

:=Ë
1batch_normalization_507/moments/SquaredDifferenceSquaredDifferencedense_560/BiasAdd:output:05batch_normalization_507/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
:batch_normalization_507/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_507/moments/varianceMean5batch_normalization_507/moments/SquaredDifference:z:0Cbatch_normalization_507/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(
'batch_normalization_507/moments/SqueezeSqueeze-batch_normalization_507/moments/mean:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 £
)batch_normalization_507/moments/Squeeze_1Squeeze1batch_normalization_507/moments/variance:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 r
-batch_normalization_507/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_507/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_507_assignmovingavg_readvariableop_resource*
_output_shapes
:=*
dtype0É
+batch_normalization_507/AssignMovingAvg/subSub>batch_normalization_507/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_507/moments/Squeeze:output:0*
T0*
_output_shapes
:=À
+batch_normalization_507/AssignMovingAvg/mulMul/batch_normalization_507/AssignMovingAvg/sub:z:06batch_normalization_507/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:=
'batch_normalization_507/AssignMovingAvgAssignSubVariableOp?batch_normalization_507_assignmovingavg_readvariableop_resource/batch_normalization_507/AssignMovingAvg/mul:z:07^batch_normalization_507/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_507/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_507/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_507_assignmovingavg_1_readvariableop_resource*
_output_shapes
:=*
dtype0Ï
-batch_normalization_507/AssignMovingAvg_1/subSub@batch_normalization_507/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_507/moments/Squeeze_1:output:0*
T0*
_output_shapes
:=Æ
-batch_normalization_507/AssignMovingAvg_1/mulMul1batch_normalization_507/AssignMovingAvg_1/sub:z:08batch_normalization_507/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:=
)batch_normalization_507/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_507_assignmovingavg_1_readvariableop_resource1batch_normalization_507/AssignMovingAvg_1/mul:z:09^batch_normalization_507/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_507/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_507/batchnorm/addAddV22batch_normalization_507/moments/Squeeze_1:output:00batch_normalization_507/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
'batch_normalization_507/batchnorm/RsqrtRsqrt)batch_normalization_507/batchnorm/add:z:0*
T0*
_output_shapes
:=®
4batch_normalization_507/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_507_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0¼
%batch_normalization_507/batchnorm/mulMul+batch_normalization_507/batchnorm/Rsqrt:y:0<batch_normalization_507/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=§
'batch_normalization_507/batchnorm/mul_1Muldense_560/BiasAdd:output:0)batch_normalization_507/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=°
'batch_normalization_507/batchnorm/mul_2Mul0batch_normalization_507/moments/Squeeze:output:0)batch_normalization_507/batchnorm/mul:z:0*
T0*
_output_shapes
:=¦
0batch_normalization_507/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_507_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0¸
%batch_normalization_507/batchnorm/subSub8batch_normalization_507/batchnorm/ReadVariableOp:value:0+batch_normalization_507/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=º
'batch_normalization_507/batchnorm/add_1AddV2+batch_normalization_507/batchnorm/mul_1:z:0)batch_normalization_507/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
leaky_re_lu_507/LeakyRelu	LeakyRelu+batch_normalization_507/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>
dense_561/MatMul/ReadVariableOpReadVariableOp(dense_561_matmul_readvariableop_resource*
_output_shapes

:=*
dtype0
dense_561/MatMulMatMul'leaky_re_lu_507/LeakyRelu:activations:0'dense_561/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_561/BiasAdd/ReadVariableOpReadVariableOp)dense_561_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_561/BiasAddBiasAdddense_561/MatMul:product:0(dense_561/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_561/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
NoOpNoOp(^batch_normalization_499/AssignMovingAvg7^batch_normalization_499/AssignMovingAvg/ReadVariableOp*^batch_normalization_499/AssignMovingAvg_19^batch_normalization_499/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_499/batchnorm/ReadVariableOp5^batch_normalization_499/batchnorm/mul/ReadVariableOp(^batch_normalization_500/AssignMovingAvg7^batch_normalization_500/AssignMovingAvg/ReadVariableOp*^batch_normalization_500/AssignMovingAvg_19^batch_normalization_500/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_500/batchnorm/ReadVariableOp5^batch_normalization_500/batchnorm/mul/ReadVariableOp(^batch_normalization_501/AssignMovingAvg7^batch_normalization_501/AssignMovingAvg/ReadVariableOp*^batch_normalization_501/AssignMovingAvg_19^batch_normalization_501/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_501/batchnorm/ReadVariableOp5^batch_normalization_501/batchnorm/mul/ReadVariableOp(^batch_normalization_502/AssignMovingAvg7^batch_normalization_502/AssignMovingAvg/ReadVariableOp*^batch_normalization_502/AssignMovingAvg_19^batch_normalization_502/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_502/batchnorm/ReadVariableOp5^batch_normalization_502/batchnorm/mul/ReadVariableOp(^batch_normalization_503/AssignMovingAvg7^batch_normalization_503/AssignMovingAvg/ReadVariableOp*^batch_normalization_503/AssignMovingAvg_19^batch_normalization_503/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_503/batchnorm/ReadVariableOp5^batch_normalization_503/batchnorm/mul/ReadVariableOp(^batch_normalization_504/AssignMovingAvg7^batch_normalization_504/AssignMovingAvg/ReadVariableOp*^batch_normalization_504/AssignMovingAvg_19^batch_normalization_504/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_504/batchnorm/ReadVariableOp5^batch_normalization_504/batchnorm/mul/ReadVariableOp(^batch_normalization_505/AssignMovingAvg7^batch_normalization_505/AssignMovingAvg/ReadVariableOp*^batch_normalization_505/AssignMovingAvg_19^batch_normalization_505/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_505/batchnorm/ReadVariableOp5^batch_normalization_505/batchnorm/mul/ReadVariableOp(^batch_normalization_506/AssignMovingAvg7^batch_normalization_506/AssignMovingAvg/ReadVariableOp*^batch_normalization_506/AssignMovingAvg_19^batch_normalization_506/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_506/batchnorm/ReadVariableOp5^batch_normalization_506/batchnorm/mul/ReadVariableOp(^batch_normalization_507/AssignMovingAvg7^batch_normalization_507/AssignMovingAvg/ReadVariableOp*^batch_normalization_507/AssignMovingAvg_19^batch_normalization_507/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_507/batchnorm/ReadVariableOp5^batch_normalization_507/batchnorm/mul/ReadVariableOp!^dense_552/BiasAdd/ReadVariableOp ^dense_552/MatMul/ReadVariableOp!^dense_553/BiasAdd/ReadVariableOp ^dense_553/MatMul/ReadVariableOp!^dense_554/BiasAdd/ReadVariableOp ^dense_554/MatMul/ReadVariableOp!^dense_555/BiasAdd/ReadVariableOp ^dense_555/MatMul/ReadVariableOp!^dense_556/BiasAdd/ReadVariableOp ^dense_556/MatMul/ReadVariableOp!^dense_557/BiasAdd/ReadVariableOp ^dense_557/MatMul/ReadVariableOp!^dense_558/BiasAdd/ReadVariableOp ^dense_558/MatMul/ReadVariableOp!^dense_559/BiasAdd/ReadVariableOp ^dense_559/MatMul/ReadVariableOp!^dense_560/BiasAdd/ReadVariableOp ^dense_560/MatMul/ReadVariableOp!^dense_561/BiasAdd/ReadVariableOp ^dense_561/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_499/AssignMovingAvg'batch_normalization_499/AssignMovingAvg2p
6batch_normalization_499/AssignMovingAvg/ReadVariableOp6batch_normalization_499/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_499/AssignMovingAvg_1)batch_normalization_499/AssignMovingAvg_12t
8batch_normalization_499/AssignMovingAvg_1/ReadVariableOp8batch_normalization_499/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_499/batchnorm/ReadVariableOp0batch_normalization_499/batchnorm/ReadVariableOp2l
4batch_normalization_499/batchnorm/mul/ReadVariableOp4batch_normalization_499/batchnorm/mul/ReadVariableOp2R
'batch_normalization_500/AssignMovingAvg'batch_normalization_500/AssignMovingAvg2p
6batch_normalization_500/AssignMovingAvg/ReadVariableOp6batch_normalization_500/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_500/AssignMovingAvg_1)batch_normalization_500/AssignMovingAvg_12t
8batch_normalization_500/AssignMovingAvg_1/ReadVariableOp8batch_normalization_500/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_500/batchnorm/ReadVariableOp0batch_normalization_500/batchnorm/ReadVariableOp2l
4batch_normalization_500/batchnorm/mul/ReadVariableOp4batch_normalization_500/batchnorm/mul/ReadVariableOp2R
'batch_normalization_501/AssignMovingAvg'batch_normalization_501/AssignMovingAvg2p
6batch_normalization_501/AssignMovingAvg/ReadVariableOp6batch_normalization_501/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_501/AssignMovingAvg_1)batch_normalization_501/AssignMovingAvg_12t
8batch_normalization_501/AssignMovingAvg_1/ReadVariableOp8batch_normalization_501/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_501/batchnorm/ReadVariableOp0batch_normalization_501/batchnorm/ReadVariableOp2l
4batch_normalization_501/batchnorm/mul/ReadVariableOp4batch_normalization_501/batchnorm/mul/ReadVariableOp2R
'batch_normalization_502/AssignMovingAvg'batch_normalization_502/AssignMovingAvg2p
6batch_normalization_502/AssignMovingAvg/ReadVariableOp6batch_normalization_502/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_502/AssignMovingAvg_1)batch_normalization_502/AssignMovingAvg_12t
8batch_normalization_502/AssignMovingAvg_1/ReadVariableOp8batch_normalization_502/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_502/batchnorm/ReadVariableOp0batch_normalization_502/batchnorm/ReadVariableOp2l
4batch_normalization_502/batchnorm/mul/ReadVariableOp4batch_normalization_502/batchnorm/mul/ReadVariableOp2R
'batch_normalization_503/AssignMovingAvg'batch_normalization_503/AssignMovingAvg2p
6batch_normalization_503/AssignMovingAvg/ReadVariableOp6batch_normalization_503/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_503/AssignMovingAvg_1)batch_normalization_503/AssignMovingAvg_12t
8batch_normalization_503/AssignMovingAvg_1/ReadVariableOp8batch_normalization_503/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_503/batchnorm/ReadVariableOp0batch_normalization_503/batchnorm/ReadVariableOp2l
4batch_normalization_503/batchnorm/mul/ReadVariableOp4batch_normalization_503/batchnorm/mul/ReadVariableOp2R
'batch_normalization_504/AssignMovingAvg'batch_normalization_504/AssignMovingAvg2p
6batch_normalization_504/AssignMovingAvg/ReadVariableOp6batch_normalization_504/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_504/AssignMovingAvg_1)batch_normalization_504/AssignMovingAvg_12t
8batch_normalization_504/AssignMovingAvg_1/ReadVariableOp8batch_normalization_504/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_504/batchnorm/ReadVariableOp0batch_normalization_504/batchnorm/ReadVariableOp2l
4batch_normalization_504/batchnorm/mul/ReadVariableOp4batch_normalization_504/batchnorm/mul/ReadVariableOp2R
'batch_normalization_505/AssignMovingAvg'batch_normalization_505/AssignMovingAvg2p
6batch_normalization_505/AssignMovingAvg/ReadVariableOp6batch_normalization_505/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_505/AssignMovingAvg_1)batch_normalization_505/AssignMovingAvg_12t
8batch_normalization_505/AssignMovingAvg_1/ReadVariableOp8batch_normalization_505/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_505/batchnorm/ReadVariableOp0batch_normalization_505/batchnorm/ReadVariableOp2l
4batch_normalization_505/batchnorm/mul/ReadVariableOp4batch_normalization_505/batchnorm/mul/ReadVariableOp2R
'batch_normalization_506/AssignMovingAvg'batch_normalization_506/AssignMovingAvg2p
6batch_normalization_506/AssignMovingAvg/ReadVariableOp6batch_normalization_506/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_506/AssignMovingAvg_1)batch_normalization_506/AssignMovingAvg_12t
8batch_normalization_506/AssignMovingAvg_1/ReadVariableOp8batch_normalization_506/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_506/batchnorm/ReadVariableOp0batch_normalization_506/batchnorm/ReadVariableOp2l
4batch_normalization_506/batchnorm/mul/ReadVariableOp4batch_normalization_506/batchnorm/mul/ReadVariableOp2R
'batch_normalization_507/AssignMovingAvg'batch_normalization_507/AssignMovingAvg2p
6batch_normalization_507/AssignMovingAvg/ReadVariableOp6batch_normalization_507/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_507/AssignMovingAvg_1)batch_normalization_507/AssignMovingAvg_12t
8batch_normalization_507/AssignMovingAvg_1/ReadVariableOp8batch_normalization_507/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_507/batchnorm/ReadVariableOp0batch_normalization_507/batchnorm/ReadVariableOp2l
4batch_normalization_507/batchnorm/mul/ReadVariableOp4batch_normalization_507/batchnorm/mul/ReadVariableOp2D
 dense_552/BiasAdd/ReadVariableOp dense_552/BiasAdd/ReadVariableOp2B
dense_552/MatMul/ReadVariableOpdense_552/MatMul/ReadVariableOp2D
 dense_553/BiasAdd/ReadVariableOp dense_553/BiasAdd/ReadVariableOp2B
dense_553/MatMul/ReadVariableOpdense_553/MatMul/ReadVariableOp2D
 dense_554/BiasAdd/ReadVariableOp dense_554/BiasAdd/ReadVariableOp2B
dense_554/MatMul/ReadVariableOpdense_554/MatMul/ReadVariableOp2D
 dense_555/BiasAdd/ReadVariableOp dense_555/BiasAdd/ReadVariableOp2B
dense_555/MatMul/ReadVariableOpdense_555/MatMul/ReadVariableOp2D
 dense_556/BiasAdd/ReadVariableOp dense_556/BiasAdd/ReadVariableOp2B
dense_556/MatMul/ReadVariableOpdense_556/MatMul/ReadVariableOp2D
 dense_557/BiasAdd/ReadVariableOp dense_557/BiasAdd/ReadVariableOp2B
dense_557/MatMul/ReadVariableOpdense_557/MatMul/ReadVariableOp2D
 dense_558/BiasAdd/ReadVariableOp dense_558/BiasAdd/ReadVariableOp2B
dense_558/MatMul/ReadVariableOpdense_558/MatMul/ReadVariableOp2D
 dense_559/BiasAdd/ReadVariableOp dense_559/BiasAdd/ReadVariableOp2B
dense_559/MatMul/ReadVariableOpdense_559/MatMul/ReadVariableOp2D
 dense_560/BiasAdd/ReadVariableOp dense_560/BiasAdd/ReadVariableOp2B
dense_560/MatMul/ReadVariableOpdense_560/MatMul/ReadVariableOp2D
 dense_561/BiasAdd/ReadVariableOp dense_561/BiasAdd/ReadVariableOp2B
dense_561/MatMul/ReadVariableOpdense_561/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
%
ì
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_757296

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
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
:P*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P¬
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
:P*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P´
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_504_layer_call_fn_760869

inputs
unknown:a
	unknown_0:a
	unknown_1:a
	unknown_2:a
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_757624o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
È	
ö
E__inference_dense_553_layer_call_and_return_conditional_losses_757937

inputs0
matmul_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
Ã
û
I__inference_sequential_53_layer_call_and_return_conditional_losses_759289
normalization_53_input
normalization_53_sub_y
normalization_53_sqrt_x"
dense_552_759148:P
dense_552_759150:P,
batch_normalization_499_759153:P,
batch_normalization_499_759155:P,
batch_normalization_499_759157:P,
batch_normalization_499_759159:P"
dense_553_759163:PP
dense_553_759165:P,
batch_normalization_500_759168:P,
batch_normalization_500_759170:P,
batch_normalization_500_759172:P,
batch_normalization_500_759174:P"
dense_554_759178:Pa
dense_554_759180:a,
batch_normalization_501_759183:a,
batch_normalization_501_759185:a,
batch_normalization_501_759187:a,
batch_normalization_501_759189:a"
dense_555_759193:aa
dense_555_759195:a,
batch_normalization_502_759198:a,
batch_normalization_502_759200:a,
batch_normalization_502_759202:a,
batch_normalization_502_759204:a"
dense_556_759208:aa
dense_556_759210:a,
batch_normalization_503_759213:a,
batch_normalization_503_759215:a,
batch_normalization_503_759217:a,
batch_normalization_503_759219:a"
dense_557_759223:aa
dense_557_759225:a,
batch_normalization_504_759228:a,
batch_normalization_504_759230:a,
batch_normalization_504_759232:a,
batch_normalization_504_759234:a"
dense_558_759238:a=
dense_558_759240:=,
batch_normalization_505_759243:=,
batch_normalization_505_759245:=,
batch_normalization_505_759247:=,
batch_normalization_505_759249:="
dense_559_759253:==
dense_559_759255:=,
batch_normalization_506_759258:=,
batch_normalization_506_759260:=,
batch_normalization_506_759262:=,
batch_normalization_506_759264:="
dense_560_759268:==
dense_560_759270:=,
batch_normalization_507_759273:=,
batch_normalization_507_759275:=,
batch_normalization_507_759277:=,
batch_normalization_507_759279:="
dense_561_759283:=
dense_561_759285:
identity¢/batch_normalization_499/StatefulPartitionedCall¢/batch_normalization_500/StatefulPartitionedCall¢/batch_normalization_501/StatefulPartitionedCall¢/batch_normalization_502/StatefulPartitionedCall¢/batch_normalization_503/StatefulPartitionedCall¢/batch_normalization_504/StatefulPartitionedCall¢/batch_normalization_505/StatefulPartitionedCall¢/batch_normalization_506/StatefulPartitionedCall¢/batch_normalization_507/StatefulPartitionedCall¢!dense_552/StatefulPartitionedCall¢!dense_553/StatefulPartitionedCall¢!dense_554/StatefulPartitionedCall¢!dense_555/StatefulPartitionedCall¢!dense_556/StatefulPartitionedCall¢!dense_557/StatefulPartitionedCall¢!dense_558/StatefulPartitionedCall¢!dense_559/StatefulPartitionedCall¢!dense_560/StatefulPartitionedCall¢!dense_561/StatefulPartitionedCall}
normalization_53/subSubnormalization_53_inputnormalization_53_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_53/SqrtSqrtnormalization_53_sqrt_x*
T0*
_output_shapes

:_
normalization_53/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_53/MaximumMaximumnormalization_53/Sqrt:y:0#normalization_53/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_53/truedivRealDivnormalization_53/sub:z:0normalization_53/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_552/StatefulPartitionedCallStatefulPartitionedCallnormalization_53/truediv:z:0dense_552_759148dense_552_759150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_552_layer_call_and_return_conditional_losses_757905
/batch_normalization_499/StatefulPartitionedCallStatefulPartitionedCall*dense_552/StatefulPartitionedCall:output:0batch_normalization_499_759153batch_normalization_499_759155batch_normalization_499_759157batch_normalization_499_759159*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_757214ø
leaky_re_lu_499/PartitionedCallPartitionedCall8batch_normalization_499/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_499_layer_call_and_return_conditional_losses_757925
!dense_553/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_499/PartitionedCall:output:0dense_553_759163dense_553_759165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_553_layer_call_and_return_conditional_losses_757937
/batch_normalization_500/StatefulPartitionedCallStatefulPartitionedCall*dense_553/StatefulPartitionedCall:output:0batch_normalization_500_759168batch_normalization_500_759170batch_normalization_500_759172batch_normalization_500_759174*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_757296ø
leaky_re_lu_500/PartitionedCallPartitionedCall8batch_normalization_500/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_500_layer_call_and_return_conditional_losses_757957
!dense_554/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_500/PartitionedCall:output:0dense_554_759178dense_554_759180*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_554_layer_call_and_return_conditional_losses_757969
/batch_normalization_501/StatefulPartitionedCallStatefulPartitionedCall*dense_554/StatefulPartitionedCall:output:0batch_normalization_501_759183batch_normalization_501_759185batch_normalization_501_759187batch_normalization_501_759189*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_757378ø
leaky_re_lu_501/PartitionedCallPartitionedCall8batch_normalization_501/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_501_layer_call_and_return_conditional_losses_757989
!dense_555/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_501/PartitionedCall:output:0dense_555_759193dense_555_759195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_555_layer_call_and_return_conditional_losses_758001
/batch_normalization_502/StatefulPartitionedCallStatefulPartitionedCall*dense_555/StatefulPartitionedCall:output:0batch_normalization_502_759198batch_normalization_502_759200batch_normalization_502_759202batch_normalization_502_759204*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_757460ø
leaky_re_lu_502/PartitionedCallPartitionedCall8batch_normalization_502/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_502_layer_call_and_return_conditional_losses_758021
!dense_556/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_502/PartitionedCall:output:0dense_556_759208dense_556_759210*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_556_layer_call_and_return_conditional_losses_758033
/batch_normalization_503/StatefulPartitionedCallStatefulPartitionedCall*dense_556/StatefulPartitionedCall:output:0batch_normalization_503_759213batch_normalization_503_759215batch_normalization_503_759217batch_normalization_503_759219*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_757542ø
leaky_re_lu_503/PartitionedCallPartitionedCall8batch_normalization_503/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_503_layer_call_and_return_conditional_losses_758053
!dense_557/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_503/PartitionedCall:output:0dense_557_759223dense_557_759225*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_557_layer_call_and_return_conditional_losses_758065
/batch_normalization_504/StatefulPartitionedCallStatefulPartitionedCall*dense_557/StatefulPartitionedCall:output:0batch_normalization_504_759228batch_normalization_504_759230batch_normalization_504_759232batch_normalization_504_759234*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_757624ø
leaky_re_lu_504/PartitionedCallPartitionedCall8batch_normalization_504/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_758085
!dense_558/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_504/PartitionedCall:output:0dense_558_759238dense_558_759240*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_558_layer_call_and_return_conditional_losses_758097
/batch_normalization_505/StatefulPartitionedCallStatefulPartitionedCall*dense_558/StatefulPartitionedCall:output:0batch_normalization_505_759243batch_normalization_505_759245batch_normalization_505_759247batch_normalization_505_759249*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_757706ø
leaky_re_lu_505/PartitionedCallPartitionedCall8batch_normalization_505/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_758117
!dense_559/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_505/PartitionedCall:output:0dense_559_759253dense_559_759255*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_559_layer_call_and_return_conditional_losses_758129
/batch_normalization_506/StatefulPartitionedCallStatefulPartitionedCall*dense_559/StatefulPartitionedCall:output:0batch_normalization_506_759258batch_normalization_506_759260batch_normalization_506_759262batch_normalization_506_759264*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_757788ø
leaky_re_lu_506/PartitionedCallPartitionedCall8batch_normalization_506/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_758149
!dense_560/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_506/PartitionedCall:output:0dense_560_759268dense_560_759270*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_560_layer_call_and_return_conditional_losses_758161
/batch_normalization_507/StatefulPartitionedCallStatefulPartitionedCall*dense_560/StatefulPartitionedCall:output:0batch_normalization_507_759273batch_normalization_507_759275batch_normalization_507_759277batch_normalization_507_759279*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_757870ø
leaky_re_lu_507/PartitionedCallPartitionedCall8batch_normalization_507/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_758181
!dense_561/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_507/PartitionedCall:output:0dense_561_759283dense_561_759285*
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
E__inference_dense_561_layer_call_and_return_conditional_losses_758193y
IdentityIdentity*dense_561/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp0^batch_normalization_499/StatefulPartitionedCall0^batch_normalization_500/StatefulPartitionedCall0^batch_normalization_501/StatefulPartitionedCall0^batch_normalization_502/StatefulPartitionedCall0^batch_normalization_503/StatefulPartitionedCall0^batch_normalization_504/StatefulPartitionedCall0^batch_normalization_505/StatefulPartitionedCall0^batch_normalization_506/StatefulPartitionedCall0^batch_normalization_507/StatefulPartitionedCall"^dense_552/StatefulPartitionedCall"^dense_553/StatefulPartitionedCall"^dense_554/StatefulPartitionedCall"^dense_555/StatefulPartitionedCall"^dense_556/StatefulPartitionedCall"^dense_557/StatefulPartitionedCall"^dense_558/StatefulPartitionedCall"^dense_559/StatefulPartitionedCall"^dense_560/StatefulPartitionedCall"^dense_561/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_499/StatefulPartitionedCall/batch_normalization_499/StatefulPartitionedCall2b
/batch_normalization_500/StatefulPartitionedCall/batch_normalization_500/StatefulPartitionedCall2b
/batch_normalization_501/StatefulPartitionedCall/batch_normalization_501/StatefulPartitionedCall2b
/batch_normalization_502/StatefulPartitionedCall/batch_normalization_502/StatefulPartitionedCall2b
/batch_normalization_503/StatefulPartitionedCall/batch_normalization_503/StatefulPartitionedCall2b
/batch_normalization_504/StatefulPartitionedCall/batch_normalization_504/StatefulPartitionedCall2b
/batch_normalization_505/StatefulPartitionedCall/batch_normalization_505/StatefulPartitionedCall2b
/batch_normalization_506/StatefulPartitionedCall/batch_normalization_506/StatefulPartitionedCall2b
/batch_normalization_507/StatefulPartitionedCall/batch_normalization_507/StatefulPartitionedCall2F
!dense_552/StatefulPartitionedCall!dense_552/StatefulPartitionedCall2F
!dense_553/StatefulPartitionedCall!dense_553/StatefulPartitionedCall2F
!dense_554/StatefulPartitionedCall!dense_554/StatefulPartitionedCall2F
!dense_555/StatefulPartitionedCall!dense_555/StatefulPartitionedCall2F
!dense_556/StatefulPartitionedCall!dense_556/StatefulPartitionedCall2F
!dense_557/StatefulPartitionedCall!dense_557/StatefulPartitionedCall2F
!dense_558/StatefulPartitionedCall!dense_558/StatefulPartitionedCall2F
!dense_559/StatefulPartitionedCall!dense_559/StatefulPartitionedCall2F
!dense_560/StatefulPartitionedCall!dense_560/StatefulPartitionedCall2F
!dense_561/StatefulPartitionedCall!dense_561/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_53_input:$ 

_output_shapes

::$ 

_output_shapes

:
¬
Ó
8__inference_batch_normalization_507_layer_call_fn_761183

inputs
unknown:=
	unknown_0:=
	unknown_1:=
	unknown_2:=
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_757823o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_757331

inputs/
!batchnorm_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a1
#batchnorm_readvariableop_1_resource:a1
#batchnorm_readvariableop_2_resource:a
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_760814

inputs5
'assignmovingavg_readvariableop_resource:a7
)assignmovingavg_1_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a/
!batchnorm_readvariableop_resource:a
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:a
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿal
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:a*
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
:a*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a¬
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
:a*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:a~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a´
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿah
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_503_layer_call_and_return_conditional_losses_760824

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿa:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
È	
ö
E__inference_dense_560_layer_call_and_return_conditional_losses_758161

inputs0
matmul_readvariableop_resource:==-
biasadd_readvariableop_resource:=
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:==*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:=*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_760453

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_760998

inputs/
!batchnorm_readvariableop_resource:=3
%batchnorm_mul_readvariableop_resource:=1
#batchnorm_readvariableop_1_resource:=1
#batchnorm_readvariableop_2_resource:=
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:=*
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
:=P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:=~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:=z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:=r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
È	
ö
E__inference_dense_554_layer_call_and_return_conditional_losses_760516

inputs0
matmul_readvariableop_resource:Pa-
biasadd_readvariableop_resource:a
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Pa*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿar
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:a*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
Õ
Ø
$__inference_signature_wrapper_760232
normalization_53_input
unknown
	unknown_0
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P

unknown_10:P

unknown_11:P

unknown_12:P

unknown_13:Pa

unknown_14:a

unknown_15:a

unknown_16:a

unknown_17:a

unknown_18:a

unknown_19:aa

unknown_20:a

unknown_21:a

unknown_22:a

unknown_23:a

unknown_24:a

unknown_25:aa

unknown_26:a

unknown_27:a

unknown_28:a

unknown_29:a

unknown_30:a

unknown_31:aa

unknown_32:a

unknown_33:a

unknown_34:a

unknown_35:a

unknown_36:a

unknown_37:a=

unknown_38:=

unknown_39:=

unknown_40:=

unknown_41:=

unknown_42:=

unknown_43:==

unknown_44:=

unknown_45:=

unknown_46:=

unknown_47:=

unknown_48:=

unknown_49:==

unknown_50:=

unknown_51:=

unknown_52:=

unknown_53:=

unknown_54:=

unknown_55:=

unknown_56:
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallnormalization_53_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./0123456789:*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_757143o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_53_input:$ 

_output_shapes

::$ 

_output_shapes

:
ª
Ó
8__inference_batch_normalization_505_layer_call_fn_760978

inputs
unknown:=
	unknown_0:=
	unknown_1:=
	unknown_2:=
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_757706o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_760344

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_500_layer_call_and_return_conditional_losses_760497

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿP:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_757413

inputs/
!batchnorm_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a1
#batchnorm_readvariableop_1_resource:a1
#batchnorm_readvariableop_2_resource:a
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_757788

inputs5
'assignmovingavg_readvariableop_resource:=7
)assignmovingavg_1_readvariableop_resource:=3
%batchnorm_mul_readvariableop_resource:=/
!batchnorm_readvariableop_resource:=
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:=
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:=*
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
:=*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:=x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:=¬
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
:=*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:=~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:=´
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
:=P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:=~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:=v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:=r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_499_layer_call_and_return_conditional_losses_760388

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿP:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
Ä

*__inference_dense_554_layer_call_fn_760506

inputs
unknown:Pa
	unknown_0:a
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_554_layer_call_and_return_conditional_losses_757969o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
¦Â
å3
I__inference_sequential_53_layer_call_and_return_conditional_losses_759759

inputs
normalization_53_sub_y
normalization_53_sqrt_x:
(dense_552_matmul_readvariableop_resource:P7
)dense_552_biasadd_readvariableop_resource:PG
9batch_normalization_499_batchnorm_readvariableop_resource:PK
=batch_normalization_499_batchnorm_mul_readvariableop_resource:PI
;batch_normalization_499_batchnorm_readvariableop_1_resource:PI
;batch_normalization_499_batchnorm_readvariableop_2_resource:P:
(dense_553_matmul_readvariableop_resource:PP7
)dense_553_biasadd_readvariableop_resource:PG
9batch_normalization_500_batchnorm_readvariableop_resource:PK
=batch_normalization_500_batchnorm_mul_readvariableop_resource:PI
;batch_normalization_500_batchnorm_readvariableop_1_resource:PI
;batch_normalization_500_batchnorm_readvariableop_2_resource:P:
(dense_554_matmul_readvariableop_resource:Pa7
)dense_554_biasadd_readvariableop_resource:aG
9batch_normalization_501_batchnorm_readvariableop_resource:aK
=batch_normalization_501_batchnorm_mul_readvariableop_resource:aI
;batch_normalization_501_batchnorm_readvariableop_1_resource:aI
;batch_normalization_501_batchnorm_readvariableop_2_resource:a:
(dense_555_matmul_readvariableop_resource:aa7
)dense_555_biasadd_readvariableop_resource:aG
9batch_normalization_502_batchnorm_readvariableop_resource:aK
=batch_normalization_502_batchnorm_mul_readvariableop_resource:aI
;batch_normalization_502_batchnorm_readvariableop_1_resource:aI
;batch_normalization_502_batchnorm_readvariableop_2_resource:a:
(dense_556_matmul_readvariableop_resource:aa7
)dense_556_biasadd_readvariableop_resource:aG
9batch_normalization_503_batchnorm_readvariableop_resource:aK
=batch_normalization_503_batchnorm_mul_readvariableop_resource:aI
;batch_normalization_503_batchnorm_readvariableop_1_resource:aI
;batch_normalization_503_batchnorm_readvariableop_2_resource:a:
(dense_557_matmul_readvariableop_resource:aa7
)dense_557_biasadd_readvariableop_resource:aG
9batch_normalization_504_batchnorm_readvariableop_resource:aK
=batch_normalization_504_batchnorm_mul_readvariableop_resource:aI
;batch_normalization_504_batchnorm_readvariableop_1_resource:aI
;batch_normalization_504_batchnorm_readvariableop_2_resource:a:
(dense_558_matmul_readvariableop_resource:a=7
)dense_558_biasadd_readvariableop_resource:=G
9batch_normalization_505_batchnorm_readvariableop_resource:=K
=batch_normalization_505_batchnorm_mul_readvariableop_resource:=I
;batch_normalization_505_batchnorm_readvariableop_1_resource:=I
;batch_normalization_505_batchnorm_readvariableop_2_resource:=:
(dense_559_matmul_readvariableop_resource:==7
)dense_559_biasadd_readvariableop_resource:=G
9batch_normalization_506_batchnorm_readvariableop_resource:=K
=batch_normalization_506_batchnorm_mul_readvariableop_resource:=I
;batch_normalization_506_batchnorm_readvariableop_1_resource:=I
;batch_normalization_506_batchnorm_readvariableop_2_resource:=:
(dense_560_matmul_readvariableop_resource:==7
)dense_560_biasadd_readvariableop_resource:=G
9batch_normalization_507_batchnorm_readvariableop_resource:=K
=batch_normalization_507_batchnorm_mul_readvariableop_resource:=I
;batch_normalization_507_batchnorm_readvariableop_1_resource:=I
;batch_normalization_507_batchnorm_readvariableop_2_resource:=:
(dense_561_matmul_readvariableop_resource:=7
)dense_561_biasadd_readvariableop_resource:
identity¢0batch_normalization_499/batchnorm/ReadVariableOp¢2batch_normalization_499/batchnorm/ReadVariableOp_1¢2batch_normalization_499/batchnorm/ReadVariableOp_2¢4batch_normalization_499/batchnorm/mul/ReadVariableOp¢0batch_normalization_500/batchnorm/ReadVariableOp¢2batch_normalization_500/batchnorm/ReadVariableOp_1¢2batch_normalization_500/batchnorm/ReadVariableOp_2¢4batch_normalization_500/batchnorm/mul/ReadVariableOp¢0batch_normalization_501/batchnorm/ReadVariableOp¢2batch_normalization_501/batchnorm/ReadVariableOp_1¢2batch_normalization_501/batchnorm/ReadVariableOp_2¢4batch_normalization_501/batchnorm/mul/ReadVariableOp¢0batch_normalization_502/batchnorm/ReadVariableOp¢2batch_normalization_502/batchnorm/ReadVariableOp_1¢2batch_normalization_502/batchnorm/ReadVariableOp_2¢4batch_normalization_502/batchnorm/mul/ReadVariableOp¢0batch_normalization_503/batchnorm/ReadVariableOp¢2batch_normalization_503/batchnorm/ReadVariableOp_1¢2batch_normalization_503/batchnorm/ReadVariableOp_2¢4batch_normalization_503/batchnorm/mul/ReadVariableOp¢0batch_normalization_504/batchnorm/ReadVariableOp¢2batch_normalization_504/batchnorm/ReadVariableOp_1¢2batch_normalization_504/batchnorm/ReadVariableOp_2¢4batch_normalization_504/batchnorm/mul/ReadVariableOp¢0batch_normalization_505/batchnorm/ReadVariableOp¢2batch_normalization_505/batchnorm/ReadVariableOp_1¢2batch_normalization_505/batchnorm/ReadVariableOp_2¢4batch_normalization_505/batchnorm/mul/ReadVariableOp¢0batch_normalization_506/batchnorm/ReadVariableOp¢2batch_normalization_506/batchnorm/ReadVariableOp_1¢2batch_normalization_506/batchnorm/ReadVariableOp_2¢4batch_normalization_506/batchnorm/mul/ReadVariableOp¢0batch_normalization_507/batchnorm/ReadVariableOp¢2batch_normalization_507/batchnorm/ReadVariableOp_1¢2batch_normalization_507/batchnorm/ReadVariableOp_2¢4batch_normalization_507/batchnorm/mul/ReadVariableOp¢ dense_552/BiasAdd/ReadVariableOp¢dense_552/MatMul/ReadVariableOp¢ dense_553/BiasAdd/ReadVariableOp¢dense_553/MatMul/ReadVariableOp¢ dense_554/BiasAdd/ReadVariableOp¢dense_554/MatMul/ReadVariableOp¢ dense_555/BiasAdd/ReadVariableOp¢dense_555/MatMul/ReadVariableOp¢ dense_556/BiasAdd/ReadVariableOp¢dense_556/MatMul/ReadVariableOp¢ dense_557/BiasAdd/ReadVariableOp¢dense_557/MatMul/ReadVariableOp¢ dense_558/BiasAdd/ReadVariableOp¢dense_558/MatMul/ReadVariableOp¢ dense_559/BiasAdd/ReadVariableOp¢dense_559/MatMul/ReadVariableOp¢ dense_560/BiasAdd/ReadVariableOp¢dense_560/MatMul/ReadVariableOp¢ dense_561/BiasAdd/ReadVariableOp¢dense_561/MatMul/ReadVariableOpm
normalization_53/subSubinputsnormalization_53_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_53/SqrtSqrtnormalization_53_sqrt_x*
T0*
_output_shapes

:_
normalization_53/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_53/MaximumMaximumnormalization_53/Sqrt:y:0#normalization_53/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_53/truedivRealDivnormalization_53/sub:z:0normalization_53/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_552/MatMul/ReadVariableOpReadVariableOp(dense_552_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0
dense_552/MatMulMatMulnormalization_53/truediv:z:0'dense_552/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 dense_552/BiasAdd/ReadVariableOpReadVariableOp)dense_552_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
dense_552/BiasAddBiasAdddense_552/MatMul:product:0(dense_552/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP¦
0batch_normalization_499/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_499_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0l
'batch_normalization_499/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_499/batchnorm/addAddV28batch_normalization_499/batchnorm/ReadVariableOp:value:00batch_normalization_499/batchnorm/add/y:output:0*
T0*
_output_shapes
:P
'batch_normalization_499/batchnorm/RsqrtRsqrt)batch_normalization_499/batchnorm/add:z:0*
T0*
_output_shapes
:P®
4batch_normalization_499/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_499_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0¼
%batch_normalization_499/batchnorm/mulMul+batch_normalization_499/batchnorm/Rsqrt:y:0<batch_normalization_499/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P§
'batch_normalization_499/batchnorm/mul_1Muldense_552/BiasAdd:output:0)batch_normalization_499/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPª
2batch_normalization_499/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_499_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0º
'batch_normalization_499/batchnorm/mul_2Mul:batch_normalization_499/batchnorm/ReadVariableOp_1:value:0)batch_normalization_499/batchnorm/mul:z:0*
T0*
_output_shapes
:Pª
2batch_normalization_499/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_499_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0º
%batch_normalization_499/batchnorm/subSub:batch_normalization_499/batchnorm/ReadVariableOp_2:value:0+batch_normalization_499/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pº
'batch_normalization_499/batchnorm/add_1AddV2+batch_normalization_499/batchnorm/mul_1:z:0)batch_normalization_499/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
leaky_re_lu_499/LeakyRelu	LeakyRelu+batch_normalization_499/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*
alpha%>
dense_553/MatMul/ReadVariableOpReadVariableOp(dense_553_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0
dense_553/MatMulMatMul'leaky_re_lu_499/LeakyRelu:activations:0'dense_553/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 dense_553/BiasAdd/ReadVariableOpReadVariableOp)dense_553_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
dense_553/BiasAddBiasAdddense_553/MatMul:product:0(dense_553/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP¦
0batch_normalization_500/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_500_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0l
'batch_normalization_500/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_500/batchnorm/addAddV28batch_normalization_500/batchnorm/ReadVariableOp:value:00batch_normalization_500/batchnorm/add/y:output:0*
T0*
_output_shapes
:P
'batch_normalization_500/batchnorm/RsqrtRsqrt)batch_normalization_500/batchnorm/add:z:0*
T0*
_output_shapes
:P®
4batch_normalization_500/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_500_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0¼
%batch_normalization_500/batchnorm/mulMul+batch_normalization_500/batchnorm/Rsqrt:y:0<batch_normalization_500/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P§
'batch_normalization_500/batchnorm/mul_1Muldense_553/BiasAdd:output:0)batch_normalization_500/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPª
2batch_normalization_500/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_500_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0º
'batch_normalization_500/batchnorm/mul_2Mul:batch_normalization_500/batchnorm/ReadVariableOp_1:value:0)batch_normalization_500/batchnorm/mul:z:0*
T0*
_output_shapes
:Pª
2batch_normalization_500/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_500_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0º
%batch_normalization_500/batchnorm/subSub:batch_normalization_500/batchnorm/ReadVariableOp_2:value:0+batch_normalization_500/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pº
'batch_normalization_500/batchnorm/add_1AddV2+batch_normalization_500/batchnorm/mul_1:z:0)batch_normalization_500/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
leaky_re_lu_500/LeakyRelu	LeakyRelu+batch_normalization_500/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*
alpha%>
dense_554/MatMul/ReadVariableOpReadVariableOp(dense_554_matmul_readvariableop_resource*
_output_shapes

:Pa*
dtype0
dense_554/MatMulMatMul'leaky_re_lu_500/LeakyRelu:activations:0'dense_554/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 dense_554/BiasAdd/ReadVariableOpReadVariableOp)dense_554_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
dense_554/BiasAddBiasAdddense_554/MatMul:product:0(dense_554/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¦
0batch_normalization_501/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_501_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0l
'batch_normalization_501/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_501/batchnorm/addAddV28batch_normalization_501/batchnorm/ReadVariableOp:value:00batch_normalization_501/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
'batch_normalization_501/batchnorm/RsqrtRsqrt)batch_normalization_501/batchnorm/add:z:0*
T0*
_output_shapes
:a®
4batch_normalization_501/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_501_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0¼
%batch_normalization_501/batchnorm/mulMul+batch_normalization_501/batchnorm/Rsqrt:y:0<batch_normalization_501/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:a§
'batch_normalization_501/batchnorm/mul_1Muldense_554/BiasAdd:output:0)batch_normalization_501/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaª
2batch_normalization_501/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_501_batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0º
'batch_normalization_501/batchnorm/mul_2Mul:batch_normalization_501/batchnorm/ReadVariableOp_1:value:0)batch_normalization_501/batchnorm/mul:z:0*
T0*
_output_shapes
:aª
2batch_normalization_501/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_501_batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0º
%batch_normalization_501/batchnorm/subSub:batch_normalization_501/batchnorm/ReadVariableOp_2:value:0+batch_normalization_501/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aº
'batch_normalization_501/batchnorm/add_1AddV2+batch_normalization_501/batchnorm/mul_1:z:0)batch_normalization_501/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
leaky_re_lu_501/LeakyRelu	LeakyRelu+batch_normalization_501/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>
dense_555/MatMul/ReadVariableOpReadVariableOp(dense_555_matmul_readvariableop_resource*
_output_shapes

:aa*
dtype0
dense_555/MatMulMatMul'leaky_re_lu_501/LeakyRelu:activations:0'dense_555/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 dense_555/BiasAdd/ReadVariableOpReadVariableOp)dense_555_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
dense_555/BiasAddBiasAdddense_555/MatMul:product:0(dense_555/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¦
0batch_normalization_502/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_502_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0l
'batch_normalization_502/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_502/batchnorm/addAddV28batch_normalization_502/batchnorm/ReadVariableOp:value:00batch_normalization_502/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
'batch_normalization_502/batchnorm/RsqrtRsqrt)batch_normalization_502/batchnorm/add:z:0*
T0*
_output_shapes
:a®
4batch_normalization_502/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_502_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0¼
%batch_normalization_502/batchnorm/mulMul+batch_normalization_502/batchnorm/Rsqrt:y:0<batch_normalization_502/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:a§
'batch_normalization_502/batchnorm/mul_1Muldense_555/BiasAdd:output:0)batch_normalization_502/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaª
2batch_normalization_502/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_502_batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0º
'batch_normalization_502/batchnorm/mul_2Mul:batch_normalization_502/batchnorm/ReadVariableOp_1:value:0)batch_normalization_502/batchnorm/mul:z:0*
T0*
_output_shapes
:aª
2batch_normalization_502/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_502_batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0º
%batch_normalization_502/batchnorm/subSub:batch_normalization_502/batchnorm/ReadVariableOp_2:value:0+batch_normalization_502/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aº
'batch_normalization_502/batchnorm/add_1AddV2+batch_normalization_502/batchnorm/mul_1:z:0)batch_normalization_502/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
leaky_re_lu_502/LeakyRelu	LeakyRelu+batch_normalization_502/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>
dense_556/MatMul/ReadVariableOpReadVariableOp(dense_556_matmul_readvariableop_resource*
_output_shapes

:aa*
dtype0
dense_556/MatMulMatMul'leaky_re_lu_502/LeakyRelu:activations:0'dense_556/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 dense_556/BiasAdd/ReadVariableOpReadVariableOp)dense_556_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
dense_556/BiasAddBiasAdddense_556/MatMul:product:0(dense_556/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¦
0batch_normalization_503/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_503_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0l
'batch_normalization_503/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_503/batchnorm/addAddV28batch_normalization_503/batchnorm/ReadVariableOp:value:00batch_normalization_503/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
'batch_normalization_503/batchnorm/RsqrtRsqrt)batch_normalization_503/batchnorm/add:z:0*
T0*
_output_shapes
:a®
4batch_normalization_503/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_503_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0¼
%batch_normalization_503/batchnorm/mulMul+batch_normalization_503/batchnorm/Rsqrt:y:0<batch_normalization_503/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:a§
'batch_normalization_503/batchnorm/mul_1Muldense_556/BiasAdd:output:0)batch_normalization_503/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaª
2batch_normalization_503/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_503_batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0º
'batch_normalization_503/batchnorm/mul_2Mul:batch_normalization_503/batchnorm/ReadVariableOp_1:value:0)batch_normalization_503/batchnorm/mul:z:0*
T0*
_output_shapes
:aª
2batch_normalization_503/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_503_batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0º
%batch_normalization_503/batchnorm/subSub:batch_normalization_503/batchnorm/ReadVariableOp_2:value:0+batch_normalization_503/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aº
'batch_normalization_503/batchnorm/add_1AddV2+batch_normalization_503/batchnorm/mul_1:z:0)batch_normalization_503/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
leaky_re_lu_503/LeakyRelu	LeakyRelu+batch_normalization_503/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>
dense_557/MatMul/ReadVariableOpReadVariableOp(dense_557_matmul_readvariableop_resource*
_output_shapes

:aa*
dtype0
dense_557/MatMulMatMul'leaky_re_lu_503/LeakyRelu:activations:0'dense_557/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 dense_557/BiasAdd/ReadVariableOpReadVariableOp)dense_557_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
dense_557/BiasAddBiasAdddense_557/MatMul:product:0(dense_557/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¦
0batch_normalization_504/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_504_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0l
'batch_normalization_504/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_504/batchnorm/addAddV28batch_normalization_504/batchnorm/ReadVariableOp:value:00batch_normalization_504/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
'batch_normalization_504/batchnorm/RsqrtRsqrt)batch_normalization_504/batchnorm/add:z:0*
T0*
_output_shapes
:a®
4batch_normalization_504/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_504_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0¼
%batch_normalization_504/batchnorm/mulMul+batch_normalization_504/batchnorm/Rsqrt:y:0<batch_normalization_504/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:a§
'batch_normalization_504/batchnorm/mul_1Muldense_557/BiasAdd:output:0)batch_normalization_504/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaª
2batch_normalization_504/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_504_batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0º
'batch_normalization_504/batchnorm/mul_2Mul:batch_normalization_504/batchnorm/ReadVariableOp_1:value:0)batch_normalization_504/batchnorm/mul:z:0*
T0*
_output_shapes
:aª
2batch_normalization_504/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_504_batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0º
%batch_normalization_504/batchnorm/subSub:batch_normalization_504/batchnorm/ReadVariableOp_2:value:0+batch_normalization_504/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aº
'batch_normalization_504/batchnorm/add_1AddV2+batch_normalization_504/batchnorm/mul_1:z:0)batch_normalization_504/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
leaky_re_lu_504/LeakyRelu	LeakyRelu+batch_normalization_504/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>
dense_558/MatMul/ReadVariableOpReadVariableOp(dense_558_matmul_readvariableop_resource*
_output_shapes

:a=*
dtype0
dense_558/MatMulMatMul'leaky_re_lu_504/LeakyRelu:activations:0'dense_558/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 dense_558/BiasAdd/ReadVariableOpReadVariableOp)dense_558_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0
dense_558/BiasAddBiasAdddense_558/MatMul:product:0(dense_558/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¦
0batch_normalization_505/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_505_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0l
'batch_normalization_505/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_505/batchnorm/addAddV28batch_normalization_505/batchnorm/ReadVariableOp:value:00batch_normalization_505/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
'batch_normalization_505/batchnorm/RsqrtRsqrt)batch_normalization_505/batchnorm/add:z:0*
T0*
_output_shapes
:=®
4batch_normalization_505/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_505_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0¼
%batch_normalization_505/batchnorm/mulMul+batch_normalization_505/batchnorm/Rsqrt:y:0<batch_normalization_505/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=§
'batch_normalization_505/batchnorm/mul_1Muldense_558/BiasAdd:output:0)batch_normalization_505/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=ª
2batch_normalization_505/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_505_batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0º
'batch_normalization_505/batchnorm/mul_2Mul:batch_normalization_505/batchnorm/ReadVariableOp_1:value:0)batch_normalization_505/batchnorm/mul:z:0*
T0*
_output_shapes
:=ª
2batch_normalization_505/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_505_batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0º
%batch_normalization_505/batchnorm/subSub:batch_normalization_505/batchnorm/ReadVariableOp_2:value:0+batch_normalization_505/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=º
'batch_normalization_505/batchnorm/add_1AddV2+batch_normalization_505/batchnorm/mul_1:z:0)batch_normalization_505/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
leaky_re_lu_505/LeakyRelu	LeakyRelu+batch_normalization_505/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>
dense_559/MatMul/ReadVariableOpReadVariableOp(dense_559_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0
dense_559/MatMulMatMul'leaky_re_lu_505/LeakyRelu:activations:0'dense_559/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 dense_559/BiasAdd/ReadVariableOpReadVariableOp)dense_559_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0
dense_559/BiasAddBiasAdddense_559/MatMul:product:0(dense_559/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¦
0batch_normalization_506/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_506_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0l
'batch_normalization_506/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_506/batchnorm/addAddV28batch_normalization_506/batchnorm/ReadVariableOp:value:00batch_normalization_506/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
'batch_normalization_506/batchnorm/RsqrtRsqrt)batch_normalization_506/batchnorm/add:z:0*
T0*
_output_shapes
:=®
4batch_normalization_506/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_506_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0¼
%batch_normalization_506/batchnorm/mulMul+batch_normalization_506/batchnorm/Rsqrt:y:0<batch_normalization_506/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=§
'batch_normalization_506/batchnorm/mul_1Muldense_559/BiasAdd:output:0)batch_normalization_506/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=ª
2batch_normalization_506/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_506_batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0º
'batch_normalization_506/batchnorm/mul_2Mul:batch_normalization_506/batchnorm/ReadVariableOp_1:value:0)batch_normalization_506/batchnorm/mul:z:0*
T0*
_output_shapes
:=ª
2batch_normalization_506/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_506_batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0º
%batch_normalization_506/batchnorm/subSub:batch_normalization_506/batchnorm/ReadVariableOp_2:value:0+batch_normalization_506/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=º
'batch_normalization_506/batchnorm/add_1AddV2+batch_normalization_506/batchnorm/mul_1:z:0)batch_normalization_506/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
leaky_re_lu_506/LeakyRelu	LeakyRelu+batch_normalization_506/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>
dense_560/MatMul/ReadVariableOpReadVariableOp(dense_560_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0
dense_560/MatMulMatMul'leaky_re_lu_506/LeakyRelu:activations:0'dense_560/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 dense_560/BiasAdd/ReadVariableOpReadVariableOp)dense_560_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0
dense_560/BiasAddBiasAdddense_560/MatMul:product:0(dense_560/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¦
0batch_normalization_507/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_507_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0l
'batch_normalization_507/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_507/batchnorm/addAddV28batch_normalization_507/batchnorm/ReadVariableOp:value:00batch_normalization_507/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
'batch_normalization_507/batchnorm/RsqrtRsqrt)batch_normalization_507/batchnorm/add:z:0*
T0*
_output_shapes
:=®
4batch_normalization_507/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_507_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0¼
%batch_normalization_507/batchnorm/mulMul+batch_normalization_507/batchnorm/Rsqrt:y:0<batch_normalization_507/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=§
'batch_normalization_507/batchnorm/mul_1Muldense_560/BiasAdd:output:0)batch_normalization_507/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=ª
2batch_normalization_507/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_507_batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0º
'batch_normalization_507/batchnorm/mul_2Mul:batch_normalization_507/batchnorm/ReadVariableOp_1:value:0)batch_normalization_507/batchnorm/mul:z:0*
T0*
_output_shapes
:=ª
2batch_normalization_507/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_507_batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0º
%batch_normalization_507/batchnorm/subSub:batch_normalization_507/batchnorm/ReadVariableOp_2:value:0+batch_normalization_507/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=º
'batch_normalization_507/batchnorm/add_1AddV2+batch_normalization_507/batchnorm/mul_1:z:0)batch_normalization_507/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
leaky_re_lu_507/LeakyRelu	LeakyRelu+batch_normalization_507/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>
dense_561/MatMul/ReadVariableOpReadVariableOp(dense_561_matmul_readvariableop_resource*
_output_shapes

:=*
dtype0
dense_561/MatMulMatMul'leaky_re_lu_507/LeakyRelu:activations:0'dense_561/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_561/BiasAdd/ReadVariableOpReadVariableOp)dense_561_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_561/BiasAddBiasAdddense_561/MatMul:product:0(dense_561/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_561/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
NoOpNoOp1^batch_normalization_499/batchnorm/ReadVariableOp3^batch_normalization_499/batchnorm/ReadVariableOp_13^batch_normalization_499/batchnorm/ReadVariableOp_25^batch_normalization_499/batchnorm/mul/ReadVariableOp1^batch_normalization_500/batchnorm/ReadVariableOp3^batch_normalization_500/batchnorm/ReadVariableOp_13^batch_normalization_500/batchnorm/ReadVariableOp_25^batch_normalization_500/batchnorm/mul/ReadVariableOp1^batch_normalization_501/batchnorm/ReadVariableOp3^batch_normalization_501/batchnorm/ReadVariableOp_13^batch_normalization_501/batchnorm/ReadVariableOp_25^batch_normalization_501/batchnorm/mul/ReadVariableOp1^batch_normalization_502/batchnorm/ReadVariableOp3^batch_normalization_502/batchnorm/ReadVariableOp_13^batch_normalization_502/batchnorm/ReadVariableOp_25^batch_normalization_502/batchnorm/mul/ReadVariableOp1^batch_normalization_503/batchnorm/ReadVariableOp3^batch_normalization_503/batchnorm/ReadVariableOp_13^batch_normalization_503/batchnorm/ReadVariableOp_25^batch_normalization_503/batchnorm/mul/ReadVariableOp1^batch_normalization_504/batchnorm/ReadVariableOp3^batch_normalization_504/batchnorm/ReadVariableOp_13^batch_normalization_504/batchnorm/ReadVariableOp_25^batch_normalization_504/batchnorm/mul/ReadVariableOp1^batch_normalization_505/batchnorm/ReadVariableOp3^batch_normalization_505/batchnorm/ReadVariableOp_13^batch_normalization_505/batchnorm/ReadVariableOp_25^batch_normalization_505/batchnorm/mul/ReadVariableOp1^batch_normalization_506/batchnorm/ReadVariableOp3^batch_normalization_506/batchnorm/ReadVariableOp_13^batch_normalization_506/batchnorm/ReadVariableOp_25^batch_normalization_506/batchnorm/mul/ReadVariableOp1^batch_normalization_507/batchnorm/ReadVariableOp3^batch_normalization_507/batchnorm/ReadVariableOp_13^batch_normalization_507/batchnorm/ReadVariableOp_25^batch_normalization_507/batchnorm/mul/ReadVariableOp!^dense_552/BiasAdd/ReadVariableOp ^dense_552/MatMul/ReadVariableOp!^dense_553/BiasAdd/ReadVariableOp ^dense_553/MatMul/ReadVariableOp!^dense_554/BiasAdd/ReadVariableOp ^dense_554/MatMul/ReadVariableOp!^dense_555/BiasAdd/ReadVariableOp ^dense_555/MatMul/ReadVariableOp!^dense_556/BiasAdd/ReadVariableOp ^dense_556/MatMul/ReadVariableOp!^dense_557/BiasAdd/ReadVariableOp ^dense_557/MatMul/ReadVariableOp!^dense_558/BiasAdd/ReadVariableOp ^dense_558/MatMul/ReadVariableOp!^dense_559/BiasAdd/ReadVariableOp ^dense_559/MatMul/ReadVariableOp!^dense_560/BiasAdd/ReadVariableOp ^dense_560/MatMul/ReadVariableOp!^dense_561/BiasAdd/ReadVariableOp ^dense_561/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_499/batchnorm/ReadVariableOp0batch_normalization_499/batchnorm/ReadVariableOp2h
2batch_normalization_499/batchnorm/ReadVariableOp_12batch_normalization_499/batchnorm/ReadVariableOp_12h
2batch_normalization_499/batchnorm/ReadVariableOp_22batch_normalization_499/batchnorm/ReadVariableOp_22l
4batch_normalization_499/batchnorm/mul/ReadVariableOp4batch_normalization_499/batchnorm/mul/ReadVariableOp2d
0batch_normalization_500/batchnorm/ReadVariableOp0batch_normalization_500/batchnorm/ReadVariableOp2h
2batch_normalization_500/batchnorm/ReadVariableOp_12batch_normalization_500/batchnorm/ReadVariableOp_12h
2batch_normalization_500/batchnorm/ReadVariableOp_22batch_normalization_500/batchnorm/ReadVariableOp_22l
4batch_normalization_500/batchnorm/mul/ReadVariableOp4batch_normalization_500/batchnorm/mul/ReadVariableOp2d
0batch_normalization_501/batchnorm/ReadVariableOp0batch_normalization_501/batchnorm/ReadVariableOp2h
2batch_normalization_501/batchnorm/ReadVariableOp_12batch_normalization_501/batchnorm/ReadVariableOp_12h
2batch_normalization_501/batchnorm/ReadVariableOp_22batch_normalization_501/batchnorm/ReadVariableOp_22l
4batch_normalization_501/batchnorm/mul/ReadVariableOp4batch_normalization_501/batchnorm/mul/ReadVariableOp2d
0batch_normalization_502/batchnorm/ReadVariableOp0batch_normalization_502/batchnorm/ReadVariableOp2h
2batch_normalization_502/batchnorm/ReadVariableOp_12batch_normalization_502/batchnorm/ReadVariableOp_12h
2batch_normalization_502/batchnorm/ReadVariableOp_22batch_normalization_502/batchnorm/ReadVariableOp_22l
4batch_normalization_502/batchnorm/mul/ReadVariableOp4batch_normalization_502/batchnorm/mul/ReadVariableOp2d
0batch_normalization_503/batchnorm/ReadVariableOp0batch_normalization_503/batchnorm/ReadVariableOp2h
2batch_normalization_503/batchnorm/ReadVariableOp_12batch_normalization_503/batchnorm/ReadVariableOp_12h
2batch_normalization_503/batchnorm/ReadVariableOp_22batch_normalization_503/batchnorm/ReadVariableOp_22l
4batch_normalization_503/batchnorm/mul/ReadVariableOp4batch_normalization_503/batchnorm/mul/ReadVariableOp2d
0batch_normalization_504/batchnorm/ReadVariableOp0batch_normalization_504/batchnorm/ReadVariableOp2h
2batch_normalization_504/batchnorm/ReadVariableOp_12batch_normalization_504/batchnorm/ReadVariableOp_12h
2batch_normalization_504/batchnorm/ReadVariableOp_22batch_normalization_504/batchnorm/ReadVariableOp_22l
4batch_normalization_504/batchnorm/mul/ReadVariableOp4batch_normalization_504/batchnorm/mul/ReadVariableOp2d
0batch_normalization_505/batchnorm/ReadVariableOp0batch_normalization_505/batchnorm/ReadVariableOp2h
2batch_normalization_505/batchnorm/ReadVariableOp_12batch_normalization_505/batchnorm/ReadVariableOp_12h
2batch_normalization_505/batchnorm/ReadVariableOp_22batch_normalization_505/batchnorm/ReadVariableOp_22l
4batch_normalization_505/batchnorm/mul/ReadVariableOp4batch_normalization_505/batchnorm/mul/ReadVariableOp2d
0batch_normalization_506/batchnorm/ReadVariableOp0batch_normalization_506/batchnorm/ReadVariableOp2h
2batch_normalization_506/batchnorm/ReadVariableOp_12batch_normalization_506/batchnorm/ReadVariableOp_12h
2batch_normalization_506/batchnorm/ReadVariableOp_22batch_normalization_506/batchnorm/ReadVariableOp_22l
4batch_normalization_506/batchnorm/mul/ReadVariableOp4batch_normalization_506/batchnorm/mul/ReadVariableOp2d
0batch_normalization_507/batchnorm/ReadVariableOp0batch_normalization_507/batchnorm/ReadVariableOp2h
2batch_normalization_507/batchnorm/ReadVariableOp_12batch_normalization_507/batchnorm/ReadVariableOp_12h
2batch_normalization_507/batchnorm/ReadVariableOp_22batch_normalization_507/batchnorm/ReadVariableOp_22l
4batch_normalization_507/batchnorm/mul/ReadVariableOp4batch_normalization_507/batchnorm/mul/ReadVariableOp2D
 dense_552/BiasAdd/ReadVariableOp dense_552/BiasAdd/ReadVariableOp2B
dense_552/MatMul/ReadVariableOpdense_552/MatMul/ReadVariableOp2D
 dense_553/BiasAdd/ReadVariableOp dense_553/BiasAdd/ReadVariableOp2B
dense_553/MatMul/ReadVariableOpdense_553/MatMul/ReadVariableOp2D
 dense_554/BiasAdd/ReadVariableOp dense_554/BiasAdd/ReadVariableOp2B
dense_554/MatMul/ReadVariableOpdense_554/MatMul/ReadVariableOp2D
 dense_555/BiasAdd/ReadVariableOp dense_555/BiasAdd/ReadVariableOp2B
dense_555/MatMul/ReadVariableOpdense_555/MatMul/ReadVariableOp2D
 dense_556/BiasAdd/ReadVariableOp dense_556/BiasAdd/ReadVariableOp2B
dense_556/MatMul/ReadVariableOpdense_556/MatMul/ReadVariableOp2D
 dense_557/BiasAdd/ReadVariableOp dense_557/BiasAdd/ReadVariableOp2B
dense_557/MatMul/ReadVariableOpdense_557/MatMul/ReadVariableOp2D
 dense_558/BiasAdd/ReadVariableOp dense_558/BiasAdd/ReadVariableOp2B
dense_558/MatMul/ReadVariableOpdense_558/MatMul/ReadVariableOp2D
 dense_559/BiasAdd/ReadVariableOp dense_559/BiasAdd/ReadVariableOp2B
dense_559/MatMul/ReadVariableOpdense_559/MatMul/ReadVariableOp2D
 dense_560/BiasAdd/ReadVariableOp dense_560/BiasAdd/ReadVariableOp2B
dense_560/MatMul/ReadVariableOpdense_560/MatMul/ReadVariableOp2D
 dense_561/BiasAdd/ReadVariableOp dense_561/BiasAdd/ReadVariableOp2B
dense_561/MatMul/ReadVariableOpdense_561/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
«
L
0__inference_leaky_re_lu_506_layer_call_fn_761146

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
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_758149`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ="
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_507_layer_call_fn_761255

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
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_758181`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ="
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Ä

*__inference_dense_560_layer_call_fn_761160

inputs
unknown:==
	unknown_0:=
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_560_layer_call_and_return_conditional_losses_758161o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_506_layer_call_fn_761074

inputs
unknown:=
	unknown_0:=
	unknown_1:=
	unknown_2:=
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_757741o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
È	
ö
E__inference_dense_554_layer_call_and_return_conditional_losses_757969

inputs0
matmul_readvariableop_resource:Pa-
biasadd_readvariableop_resource:a
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Pa*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿar
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:a*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
È	
ö
E__inference_dense_561_layer_call_and_return_conditional_losses_758193

inputs0
matmul_readvariableop_resource:=-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:=*
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
:ÿÿÿÿÿÿÿÿÿ=: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
È	
ö
E__inference_dense_559_layer_call_and_return_conditional_losses_758129

inputs0
matmul_readvariableop_resource:==-
biasadd_readvariableop_resource:=
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:==*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:=*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
È	
ö
E__inference_dense_560_layer_call_and_return_conditional_losses_761170

inputs0
matmul_readvariableop_resource:==-
biasadd_readvariableop_resource:=
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:==*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:=*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Ä

*__inference_dense_561_layer_call_fn_761269

inputs
unknown:=
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
E__inference_dense_561_layer_call_and_return_conditional_losses_758193o
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
:ÿÿÿÿÿÿÿÿÿ=: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_501_layer_call_and_return_conditional_losses_760606

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿa:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
È	
ö
E__inference_dense_553_layer_call_and_return_conditional_losses_760407

inputs0
matmul_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_757577

inputs/
!batchnorm_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a1
#batchnorm_readvariableop_1_resource:a1
#batchnorm_readvariableop_2_resource:a
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_500_layer_call_fn_760492

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
:ÿÿÿÿÿÿÿÿÿP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_500_layer_call_and_return_conditional_losses_757957`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿP:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_761250

inputs5
'assignmovingavg_readvariableop_resource:=7
)assignmovingavg_1_readvariableop_resource:=3
%batchnorm_mul_readvariableop_resource:=/
!batchnorm_readvariableop_resource:=
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:=
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:=*
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
:=*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:=x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:=¬
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
:=*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:=~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:=´
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
:=P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:=~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:=v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:=r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_499_layer_call_fn_760383

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
:ÿÿÿÿÿÿÿÿÿP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_499_layer_call_and_return_conditional_losses_757925`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿP:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_505_layer_call_fn_761037

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
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_758117`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ="
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_757167

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
È	
ö
E__inference_dense_555_layer_call_and_return_conditional_losses_758001

inputs0
matmul_readvariableop_resource:aa-
biasadd_readvariableop_resource:a
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:aa*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿar
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:a*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_503_layer_call_fn_760819

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
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_503_layer_call_and_return_conditional_losses_758053`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿa:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
È	
ö
E__inference_dense_558_layer_call_and_return_conditional_losses_760952

inputs0
matmul_readvariableop_resource:a=-
biasadd_readvariableop_resource:=
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:a=*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:=*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_503_layer_call_and_return_conditional_losses_758053

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿa:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
¶¾
Í^
"__inference__traced_restore_762160
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_552_kernel:P/
!assignvariableop_4_dense_552_bias:P>
0assignvariableop_5_batch_normalization_499_gamma:P=
/assignvariableop_6_batch_normalization_499_beta:PD
6assignvariableop_7_batch_normalization_499_moving_mean:PH
:assignvariableop_8_batch_normalization_499_moving_variance:P5
#assignvariableop_9_dense_553_kernel:PP0
"assignvariableop_10_dense_553_bias:P?
1assignvariableop_11_batch_normalization_500_gamma:P>
0assignvariableop_12_batch_normalization_500_beta:PE
7assignvariableop_13_batch_normalization_500_moving_mean:PI
;assignvariableop_14_batch_normalization_500_moving_variance:P6
$assignvariableop_15_dense_554_kernel:Pa0
"assignvariableop_16_dense_554_bias:a?
1assignvariableop_17_batch_normalization_501_gamma:a>
0assignvariableop_18_batch_normalization_501_beta:aE
7assignvariableop_19_batch_normalization_501_moving_mean:aI
;assignvariableop_20_batch_normalization_501_moving_variance:a6
$assignvariableop_21_dense_555_kernel:aa0
"assignvariableop_22_dense_555_bias:a?
1assignvariableop_23_batch_normalization_502_gamma:a>
0assignvariableop_24_batch_normalization_502_beta:aE
7assignvariableop_25_batch_normalization_502_moving_mean:aI
;assignvariableop_26_batch_normalization_502_moving_variance:a6
$assignvariableop_27_dense_556_kernel:aa0
"assignvariableop_28_dense_556_bias:a?
1assignvariableop_29_batch_normalization_503_gamma:a>
0assignvariableop_30_batch_normalization_503_beta:aE
7assignvariableop_31_batch_normalization_503_moving_mean:aI
;assignvariableop_32_batch_normalization_503_moving_variance:a6
$assignvariableop_33_dense_557_kernel:aa0
"assignvariableop_34_dense_557_bias:a?
1assignvariableop_35_batch_normalization_504_gamma:a>
0assignvariableop_36_batch_normalization_504_beta:aE
7assignvariableop_37_batch_normalization_504_moving_mean:aI
;assignvariableop_38_batch_normalization_504_moving_variance:a6
$assignvariableop_39_dense_558_kernel:a=0
"assignvariableop_40_dense_558_bias:=?
1assignvariableop_41_batch_normalization_505_gamma:=>
0assignvariableop_42_batch_normalization_505_beta:=E
7assignvariableop_43_batch_normalization_505_moving_mean:=I
;assignvariableop_44_batch_normalization_505_moving_variance:=6
$assignvariableop_45_dense_559_kernel:==0
"assignvariableop_46_dense_559_bias:=?
1assignvariableop_47_batch_normalization_506_gamma:=>
0assignvariableop_48_batch_normalization_506_beta:=E
7assignvariableop_49_batch_normalization_506_moving_mean:=I
;assignvariableop_50_batch_normalization_506_moving_variance:=6
$assignvariableop_51_dense_560_kernel:==0
"assignvariableop_52_dense_560_bias:=?
1assignvariableop_53_batch_normalization_507_gamma:=>
0assignvariableop_54_batch_normalization_507_beta:=E
7assignvariableop_55_batch_normalization_507_moving_mean:=I
;assignvariableop_56_batch_normalization_507_moving_variance:=6
$assignvariableop_57_dense_561_kernel:=0
"assignvariableop_58_dense_561_bias:'
assignvariableop_59_adam_iter:	 )
assignvariableop_60_adam_beta_1: )
assignvariableop_61_adam_beta_2: (
assignvariableop_62_adam_decay: #
assignvariableop_63_total: %
assignvariableop_64_count_1: =
+assignvariableop_65_adam_dense_552_kernel_m:P7
)assignvariableop_66_adam_dense_552_bias_m:PF
8assignvariableop_67_adam_batch_normalization_499_gamma_m:PE
7assignvariableop_68_adam_batch_normalization_499_beta_m:P=
+assignvariableop_69_adam_dense_553_kernel_m:PP7
)assignvariableop_70_adam_dense_553_bias_m:PF
8assignvariableop_71_adam_batch_normalization_500_gamma_m:PE
7assignvariableop_72_adam_batch_normalization_500_beta_m:P=
+assignvariableop_73_adam_dense_554_kernel_m:Pa7
)assignvariableop_74_adam_dense_554_bias_m:aF
8assignvariableop_75_adam_batch_normalization_501_gamma_m:aE
7assignvariableop_76_adam_batch_normalization_501_beta_m:a=
+assignvariableop_77_adam_dense_555_kernel_m:aa7
)assignvariableop_78_adam_dense_555_bias_m:aF
8assignvariableop_79_adam_batch_normalization_502_gamma_m:aE
7assignvariableop_80_adam_batch_normalization_502_beta_m:a=
+assignvariableop_81_adam_dense_556_kernel_m:aa7
)assignvariableop_82_adam_dense_556_bias_m:aF
8assignvariableop_83_adam_batch_normalization_503_gamma_m:aE
7assignvariableop_84_adam_batch_normalization_503_beta_m:a=
+assignvariableop_85_adam_dense_557_kernel_m:aa7
)assignvariableop_86_adam_dense_557_bias_m:aF
8assignvariableop_87_adam_batch_normalization_504_gamma_m:aE
7assignvariableop_88_adam_batch_normalization_504_beta_m:a=
+assignvariableop_89_adam_dense_558_kernel_m:a=7
)assignvariableop_90_adam_dense_558_bias_m:=F
8assignvariableop_91_adam_batch_normalization_505_gamma_m:=E
7assignvariableop_92_adam_batch_normalization_505_beta_m:==
+assignvariableop_93_adam_dense_559_kernel_m:==7
)assignvariableop_94_adam_dense_559_bias_m:=F
8assignvariableop_95_adam_batch_normalization_506_gamma_m:=E
7assignvariableop_96_adam_batch_normalization_506_beta_m:==
+assignvariableop_97_adam_dense_560_kernel_m:==7
)assignvariableop_98_adam_dense_560_bias_m:=F
8assignvariableop_99_adam_batch_normalization_507_gamma_m:=F
8assignvariableop_100_adam_batch_normalization_507_beta_m:=>
,assignvariableop_101_adam_dense_561_kernel_m:=8
*assignvariableop_102_adam_dense_561_bias_m:>
,assignvariableop_103_adam_dense_552_kernel_v:P8
*assignvariableop_104_adam_dense_552_bias_v:PG
9assignvariableop_105_adam_batch_normalization_499_gamma_v:PF
8assignvariableop_106_adam_batch_normalization_499_beta_v:P>
,assignvariableop_107_adam_dense_553_kernel_v:PP8
*assignvariableop_108_adam_dense_553_bias_v:PG
9assignvariableop_109_adam_batch_normalization_500_gamma_v:PF
8assignvariableop_110_adam_batch_normalization_500_beta_v:P>
,assignvariableop_111_adam_dense_554_kernel_v:Pa8
*assignvariableop_112_adam_dense_554_bias_v:aG
9assignvariableop_113_adam_batch_normalization_501_gamma_v:aF
8assignvariableop_114_adam_batch_normalization_501_beta_v:a>
,assignvariableop_115_adam_dense_555_kernel_v:aa8
*assignvariableop_116_adam_dense_555_bias_v:aG
9assignvariableop_117_adam_batch_normalization_502_gamma_v:aF
8assignvariableop_118_adam_batch_normalization_502_beta_v:a>
,assignvariableop_119_adam_dense_556_kernel_v:aa8
*assignvariableop_120_adam_dense_556_bias_v:aG
9assignvariableop_121_adam_batch_normalization_503_gamma_v:aF
8assignvariableop_122_adam_batch_normalization_503_beta_v:a>
,assignvariableop_123_adam_dense_557_kernel_v:aa8
*assignvariableop_124_adam_dense_557_bias_v:aG
9assignvariableop_125_adam_batch_normalization_504_gamma_v:aF
8assignvariableop_126_adam_batch_normalization_504_beta_v:a>
,assignvariableop_127_adam_dense_558_kernel_v:a=8
*assignvariableop_128_adam_dense_558_bias_v:=G
9assignvariableop_129_adam_batch_normalization_505_gamma_v:=F
8assignvariableop_130_adam_batch_normalization_505_beta_v:=>
,assignvariableop_131_adam_dense_559_kernel_v:==8
*assignvariableop_132_adam_dense_559_bias_v:=G
9assignvariableop_133_adam_batch_normalization_506_gamma_v:=F
8assignvariableop_134_adam_batch_normalization_506_beta_v:=>
,assignvariableop_135_adam_dense_560_kernel_v:==8
*assignvariableop_136_adam_dense_560_bias_v:=G
9assignvariableop_137_adam_batch_normalization_507_gamma_v:=F
8assignvariableop_138_adam_batch_normalization_507_beta_v:=>
,assignvariableop_139_adam_dense_561_kernel_v:=8
*assignvariableop_140_adam_dense_561_bias_v:
identity_142¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_111¢AssignVariableOp_112¢AssignVariableOp_113¢AssignVariableOp_114¢AssignVariableOp_115¢AssignVariableOp_116¢AssignVariableOp_117¢AssignVariableOp_118¢AssignVariableOp_119¢AssignVariableOp_12¢AssignVariableOp_120¢AssignVariableOp_121¢AssignVariableOp_122¢AssignVariableOp_123¢AssignVariableOp_124¢AssignVariableOp_125¢AssignVariableOp_126¢AssignVariableOp_127¢AssignVariableOp_128¢AssignVariableOp_129¢AssignVariableOp_13¢AssignVariableOp_130¢AssignVariableOp_131¢AssignVariableOp_132¢AssignVariableOp_133¢AssignVariableOp_134¢AssignVariableOp_135¢AssignVariableOp_136¢AssignVariableOp_137¢AssignVariableOp_138¢AssignVariableOp_139¢AssignVariableOp_14¢AssignVariableOp_140¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99ºO
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ßN
valueÕNBÒNB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*²
value¨B¥B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ë
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Î
_output_shapes»
¸::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes
2		[
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_552_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_552_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batch_normalization_499_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_499_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_499_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_8AssignVariableOp:assignvariableop_8_batch_normalization_499_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_553_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_553_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batch_normalization_500_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_500_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_500_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_14AssignVariableOp;assignvariableop_14_batch_normalization_500_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_554_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_554_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batch_normalization_501_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_501_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_501_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_20AssignVariableOp;assignvariableop_20_batch_normalization_501_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_555_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_555_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_23AssignVariableOp1assignvariableop_23_batch_normalization_502_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_502_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_25AssignVariableOp7assignvariableop_25_batch_normalization_502_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_26AssignVariableOp;assignvariableop_26_batch_normalization_502_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_556_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_556_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_29AssignVariableOp1assignvariableop_29_batch_normalization_503_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_503_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_31AssignVariableOp7assignvariableop_31_batch_normalization_503_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_32AssignVariableOp;assignvariableop_32_batch_normalization_503_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_557_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_557_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_35AssignVariableOp1assignvariableop_35_batch_normalization_504_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_504_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_37AssignVariableOp7assignvariableop_37_batch_normalization_504_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_38AssignVariableOp;assignvariableop_38_batch_normalization_504_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp$assignvariableop_39_dense_558_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_558_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_41AssignVariableOp1assignvariableop_41_batch_normalization_505_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_batch_normalization_505_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_43AssignVariableOp7assignvariableop_43_batch_normalization_505_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_44AssignVariableOp;assignvariableop_44_batch_normalization_505_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp$assignvariableop_45_dense_559_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_559_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_47AssignVariableOp1assignvariableop_47_batch_normalization_506_gammaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_48AssignVariableOp0assignvariableop_48_batch_normalization_506_betaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_49AssignVariableOp7assignvariableop_49_batch_normalization_506_moving_meanIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_50AssignVariableOp;assignvariableop_50_batch_normalization_506_moving_varianceIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp$assignvariableop_51_dense_560_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp"assignvariableop_52_dense_560_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_53AssignVariableOp1assignvariableop_53_batch_normalization_507_gammaIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_54AssignVariableOp0assignvariableop_54_batch_normalization_507_betaIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_55AssignVariableOp7assignvariableop_55_batch_normalization_507_moving_meanIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_56AssignVariableOp;assignvariableop_56_batch_normalization_507_moving_varianceIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp$assignvariableop_57_dense_561_kernelIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp"assignvariableop_58_dense_561_biasIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_59AssignVariableOpassignvariableop_59_adam_iterIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOpassignvariableop_60_adam_beta_1Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOpassignvariableop_61_adam_beta_2Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOpassignvariableop_62_adam_decayIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOpassignvariableop_63_totalIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOpassignvariableop_64_count_1Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_dense_552_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_dense_552_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_67AssignVariableOp8assignvariableop_67_adam_batch_normalization_499_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_batch_normalization_499_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_dense_553_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_dense_553_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_71AssignVariableOp8assignvariableop_71_adam_batch_normalization_500_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_72AssignVariableOp7assignvariableop_72_adam_batch_normalization_500_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_dense_554_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_dense_554_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_75AssignVariableOp8assignvariableop_75_adam_batch_normalization_501_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_76AssignVariableOp7assignvariableop_76_adam_batch_normalization_501_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_dense_555_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_dense_555_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_79AssignVariableOp8assignvariableop_79_adam_batch_normalization_502_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_80AssignVariableOp7assignvariableop_80_adam_batch_normalization_502_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_dense_556_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_dense_556_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_83AssignVariableOp8assignvariableop_83_adam_batch_normalization_503_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_84AssignVariableOp7assignvariableop_84_adam_batch_normalization_503_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_dense_557_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_dense_557_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_87AssignVariableOp8assignvariableop_87_adam_batch_normalization_504_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_88AssignVariableOp7assignvariableop_88_adam_batch_normalization_504_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_dense_558_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_dense_558_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_91AssignVariableOp8assignvariableop_91_adam_batch_normalization_505_gamma_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_92AssignVariableOp7assignvariableop_92_adam_batch_normalization_505_beta_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_dense_559_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_dense_559_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_95AssignVariableOp8assignvariableop_95_adam_batch_normalization_506_gamma_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_96AssignVariableOp7assignvariableop_96_adam_batch_normalization_506_beta_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_dense_560_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_dense_560_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_99AssignVariableOp8assignvariableop_99_adam_batch_normalization_507_gamma_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_100AssignVariableOp8assignvariableop_100_adam_batch_normalization_507_beta_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_dense_561_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_dense_561_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_103AssignVariableOp,assignvariableop_103_adam_dense_552_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_dense_552_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_105AssignVariableOp9assignvariableop_105_adam_batch_normalization_499_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_106AssignVariableOp8assignvariableop_106_adam_batch_normalization_499_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_dense_553_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_dense_553_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_109AssignVariableOp9assignvariableop_109_adam_batch_normalization_500_gamma_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_110AssignVariableOp8assignvariableop_110_adam_batch_normalization_500_beta_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_111AssignVariableOp,assignvariableop_111_adam_dense_554_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_112AssignVariableOp*assignvariableop_112_adam_dense_554_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_113AssignVariableOp9assignvariableop_113_adam_batch_normalization_501_gamma_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_114AssignVariableOp8assignvariableop_114_adam_batch_normalization_501_beta_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_115AssignVariableOp,assignvariableop_115_adam_dense_555_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_116AssignVariableOp*assignvariableop_116_adam_dense_555_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_117AssignVariableOp9assignvariableop_117_adam_batch_normalization_502_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_118AssignVariableOp8assignvariableop_118_adam_batch_normalization_502_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_119AssignVariableOp,assignvariableop_119_adam_dense_556_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_120AssignVariableOp*assignvariableop_120_adam_dense_556_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_121AssignVariableOp9assignvariableop_121_adam_batch_normalization_503_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_122AssignVariableOp8assignvariableop_122_adam_batch_normalization_503_beta_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_123AssignVariableOp,assignvariableop_123_adam_dense_557_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_124AssignVariableOp*assignvariableop_124_adam_dense_557_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_125AssignVariableOp9assignvariableop_125_adam_batch_normalization_504_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_126AssignVariableOp8assignvariableop_126_adam_batch_normalization_504_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_127AssignVariableOp,assignvariableop_127_adam_dense_558_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_128AssignVariableOp*assignvariableop_128_adam_dense_558_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_129AssignVariableOp9assignvariableop_129_adam_batch_normalization_505_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_130AssignVariableOp8assignvariableop_130_adam_batch_normalization_505_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_131AssignVariableOp,assignvariableop_131_adam_dense_559_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_132AssignVariableOp*assignvariableop_132_adam_dense_559_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_133AssignVariableOp9assignvariableop_133_adam_batch_normalization_506_gamma_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_134AssignVariableOp8assignvariableop_134_adam_batch_normalization_506_beta_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_135AssignVariableOp,assignvariableop_135_adam_dense_560_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_136AssignVariableOp*assignvariableop_136_adam_dense_560_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_137AssignVariableOp9assignvariableop_137_adam_batch_normalization_507_gamma_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_138AssignVariableOp8assignvariableop_138_adam_batch_normalization_507_beta_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_139AssignVariableOp,assignvariableop_139_adam_dense_561_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_140AssignVariableOp*assignvariableop_140_adam_dense_561_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_141Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_142IdentityIdentity_141:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_142Identity_142:output:0*±
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402*
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
È	
ö
E__inference_dense_558_layer_call_and_return_conditional_losses_758097

inputs0
matmul_readvariableop_resource:a=-
biasadd_readvariableop_resource:=
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:a=*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:=*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_757659

inputs/
!batchnorm_readvariableop_resource:=3
%batchnorm_mul_readvariableop_resource:=1
#batchnorm_readvariableop_1_resource:=1
#batchnorm_readvariableop_2_resource:=
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:=*
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
:=P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:=~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:=z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:=r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Õ
û
I__inference_sequential_53_layer_call_and_return_conditional_losses_759138
normalization_53_input
normalization_53_sub_y
normalization_53_sqrt_x"
dense_552_758997:P
dense_552_758999:P,
batch_normalization_499_759002:P,
batch_normalization_499_759004:P,
batch_normalization_499_759006:P,
batch_normalization_499_759008:P"
dense_553_759012:PP
dense_553_759014:P,
batch_normalization_500_759017:P,
batch_normalization_500_759019:P,
batch_normalization_500_759021:P,
batch_normalization_500_759023:P"
dense_554_759027:Pa
dense_554_759029:a,
batch_normalization_501_759032:a,
batch_normalization_501_759034:a,
batch_normalization_501_759036:a,
batch_normalization_501_759038:a"
dense_555_759042:aa
dense_555_759044:a,
batch_normalization_502_759047:a,
batch_normalization_502_759049:a,
batch_normalization_502_759051:a,
batch_normalization_502_759053:a"
dense_556_759057:aa
dense_556_759059:a,
batch_normalization_503_759062:a,
batch_normalization_503_759064:a,
batch_normalization_503_759066:a,
batch_normalization_503_759068:a"
dense_557_759072:aa
dense_557_759074:a,
batch_normalization_504_759077:a,
batch_normalization_504_759079:a,
batch_normalization_504_759081:a,
batch_normalization_504_759083:a"
dense_558_759087:a=
dense_558_759089:=,
batch_normalization_505_759092:=,
batch_normalization_505_759094:=,
batch_normalization_505_759096:=,
batch_normalization_505_759098:="
dense_559_759102:==
dense_559_759104:=,
batch_normalization_506_759107:=,
batch_normalization_506_759109:=,
batch_normalization_506_759111:=,
batch_normalization_506_759113:="
dense_560_759117:==
dense_560_759119:=,
batch_normalization_507_759122:=,
batch_normalization_507_759124:=,
batch_normalization_507_759126:=,
batch_normalization_507_759128:="
dense_561_759132:=
dense_561_759134:
identity¢/batch_normalization_499/StatefulPartitionedCall¢/batch_normalization_500/StatefulPartitionedCall¢/batch_normalization_501/StatefulPartitionedCall¢/batch_normalization_502/StatefulPartitionedCall¢/batch_normalization_503/StatefulPartitionedCall¢/batch_normalization_504/StatefulPartitionedCall¢/batch_normalization_505/StatefulPartitionedCall¢/batch_normalization_506/StatefulPartitionedCall¢/batch_normalization_507/StatefulPartitionedCall¢!dense_552/StatefulPartitionedCall¢!dense_553/StatefulPartitionedCall¢!dense_554/StatefulPartitionedCall¢!dense_555/StatefulPartitionedCall¢!dense_556/StatefulPartitionedCall¢!dense_557/StatefulPartitionedCall¢!dense_558/StatefulPartitionedCall¢!dense_559/StatefulPartitionedCall¢!dense_560/StatefulPartitionedCall¢!dense_561/StatefulPartitionedCall}
normalization_53/subSubnormalization_53_inputnormalization_53_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_53/SqrtSqrtnormalization_53_sqrt_x*
T0*
_output_shapes

:_
normalization_53/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_53/MaximumMaximumnormalization_53/Sqrt:y:0#normalization_53/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_53/truedivRealDivnormalization_53/sub:z:0normalization_53/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_552/StatefulPartitionedCallStatefulPartitionedCallnormalization_53/truediv:z:0dense_552_758997dense_552_758999*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_552_layer_call_and_return_conditional_losses_757905
/batch_normalization_499/StatefulPartitionedCallStatefulPartitionedCall*dense_552/StatefulPartitionedCall:output:0batch_normalization_499_759002batch_normalization_499_759004batch_normalization_499_759006batch_normalization_499_759008*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_757167ø
leaky_re_lu_499/PartitionedCallPartitionedCall8batch_normalization_499/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_499_layer_call_and_return_conditional_losses_757925
!dense_553/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_499/PartitionedCall:output:0dense_553_759012dense_553_759014*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_553_layer_call_and_return_conditional_losses_757937
/batch_normalization_500/StatefulPartitionedCallStatefulPartitionedCall*dense_553/StatefulPartitionedCall:output:0batch_normalization_500_759017batch_normalization_500_759019batch_normalization_500_759021batch_normalization_500_759023*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_757249ø
leaky_re_lu_500/PartitionedCallPartitionedCall8batch_normalization_500/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_500_layer_call_and_return_conditional_losses_757957
!dense_554/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_500/PartitionedCall:output:0dense_554_759027dense_554_759029*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_554_layer_call_and_return_conditional_losses_757969
/batch_normalization_501/StatefulPartitionedCallStatefulPartitionedCall*dense_554/StatefulPartitionedCall:output:0batch_normalization_501_759032batch_normalization_501_759034batch_normalization_501_759036batch_normalization_501_759038*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_757331ø
leaky_re_lu_501/PartitionedCallPartitionedCall8batch_normalization_501/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_501_layer_call_and_return_conditional_losses_757989
!dense_555/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_501/PartitionedCall:output:0dense_555_759042dense_555_759044*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_555_layer_call_and_return_conditional_losses_758001
/batch_normalization_502/StatefulPartitionedCallStatefulPartitionedCall*dense_555/StatefulPartitionedCall:output:0batch_normalization_502_759047batch_normalization_502_759049batch_normalization_502_759051batch_normalization_502_759053*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_757413ø
leaky_re_lu_502/PartitionedCallPartitionedCall8batch_normalization_502/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_502_layer_call_and_return_conditional_losses_758021
!dense_556/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_502/PartitionedCall:output:0dense_556_759057dense_556_759059*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_556_layer_call_and_return_conditional_losses_758033
/batch_normalization_503/StatefulPartitionedCallStatefulPartitionedCall*dense_556/StatefulPartitionedCall:output:0batch_normalization_503_759062batch_normalization_503_759064batch_normalization_503_759066batch_normalization_503_759068*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_757495ø
leaky_re_lu_503/PartitionedCallPartitionedCall8batch_normalization_503/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_503_layer_call_and_return_conditional_losses_758053
!dense_557/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_503/PartitionedCall:output:0dense_557_759072dense_557_759074*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_557_layer_call_and_return_conditional_losses_758065
/batch_normalization_504/StatefulPartitionedCallStatefulPartitionedCall*dense_557/StatefulPartitionedCall:output:0batch_normalization_504_759077batch_normalization_504_759079batch_normalization_504_759081batch_normalization_504_759083*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_757577ø
leaky_re_lu_504/PartitionedCallPartitionedCall8batch_normalization_504/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_758085
!dense_558/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_504/PartitionedCall:output:0dense_558_759087dense_558_759089*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_558_layer_call_and_return_conditional_losses_758097
/batch_normalization_505/StatefulPartitionedCallStatefulPartitionedCall*dense_558/StatefulPartitionedCall:output:0batch_normalization_505_759092batch_normalization_505_759094batch_normalization_505_759096batch_normalization_505_759098*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_757659ø
leaky_re_lu_505/PartitionedCallPartitionedCall8batch_normalization_505/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_758117
!dense_559/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_505/PartitionedCall:output:0dense_559_759102dense_559_759104*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_559_layer_call_and_return_conditional_losses_758129
/batch_normalization_506/StatefulPartitionedCallStatefulPartitionedCall*dense_559/StatefulPartitionedCall:output:0batch_normalization_506_759107batch_normalization_506_759109batch_normalization_506_759111batch_normalization_506_759113*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_757741ø
leaky_re_lu_506/PartitionedCallPartitionedCall8batch_normalization_506/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_758149
!dense_560/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_506/PartitionedCall:output:0dense_560_759117dense_560_759119*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_560_layer_call_and_return_conditional_losses_758161
/batch_normalization_507/StatefulPartitionedCallStatefulPartitionedCall*dense_560/StatefulPartitionedCall:output:0batch_normalization_507_759122batch_normalization_507_759124batch_normalization_507_759126batch_normalization_507_759128*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_757823ø
leaky_re_lu_507/PartitionedCallPartitionedCall8batch_normalization_507/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_758181
!dense_561/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_507/PartitionedCall:output:0dense_561_759132dense_561_759134*
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
E__inference_dense_561_layer_call_and_return_conditional_losses_758193y
IdentityIdentity*dense_561/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp0^batch_normalization_499/StatefulPartitionedCall0^batch_normalization_500/StatefulPartitionedCall0^batch_normalization_501/StatefulPartitionedCall0^batch_normalization_502/StatefulPartitionedCall0^batch_normalization_503/StatefulPartitionedCall0^batch_normalization_504/StatefulPartitionedCall0^batch_normalization_505/StatefulPartitionedCall0^batch_normalization_506/StatefulPartitionedCall0^batch_normalization_507/StatefulPartitionedCall"^dense_552/StatefulPartitionedCall"^dense_553/StatefulPartitionedCall"^dense_554/StatefulPartitionedCall"^dense_555/StatefulPartitionedCall"^dense_556/StatefulPartitionedCall"^dense_557/StatefulPartitionedCall"^dense_558/StatefulPartitionedCall"^dense_559/StatefulPartitionedCall"^dense_560/StatefulPartitionedCall"^dense_561/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_499/StatefulPartitionedCall/batch_normalization_499/StatefulPartitionedCall2b
/batch_normalization_500/StatefulPartitionedCall/batch_normalization_500/StatefulPartitionedCall2b
/batch_normalization_501/StatefulPartitionedCall/batch_normalization_501/StatefulPartitionedCall2b
/batch_normalization_502/StatefulPartitionedCall/batch_normalization_502/StatefulPartitionedCall2b
/batch_normalization_503/StatefulPartitionedCall/batch_normalization_503/StatefulPartitionedCall2b
/batch_normalization_504/StatefulPartitionedCall/batch_normalization_504/StatefulPartitionedCall2b
/batch_normalization_505/StatefulPartitionedCall/batch_normalization_505/StatefulPartitionedCall2b
/batch_normalization_506/StatefulPartitionedCall/batch_normalization_506/StatefulPartitionedCall2b
/batch_normalization_507/StatefulPartitionedCall/batch_normalization_507/StatefulPartitionedCall2F
!dense_552/StatefulPartitionedCall!dense_552/StatefulPartitionedCall2F
!dense_553/StatefulPartitionedCall!dense_553/StatefulPartitionedCall2F
!dense_554/StatefulPartitionedCall!dense_554/StatefulPartitionedCall2F
!dense_555/StatefulPartitionedCall!dense_555/StatefulPartitionedCall2F
!dense_556/StatefulPartitionedCall!dense_556/StatefulPartitionedCall2F
!dense_557/StatefulPartitionedCall!dense_557/StatefulPartitionedCall2F
!dense_558/StatefulPartitionedCall!dense_558/StatefulPartitionedCall2F
!dense_559/StatefulPartitionedCall!dense_559/StatefulPartitionedCall2F
!dense_560/StatefulPartitionedCall!dense_560/StatefulPartitionedCall2F
!dense_561/StatefulPartitionedCall!dense_561/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_53_input:$ 

_output_shapes

::$ 

_output_shapes

:
ª
Ó
8__inference_batch_normalization_502_layer_call_fn_760651

inputs
unknown:a
	unknown_0:a
	unknown_1:a
	unknown_2:a
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_757460o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
È	
ö
E__inference_dense_557_layer_call_and_return_conditional_losses_758065

inputs0
matmul_readvariableop_resource:aa-
biasadd_readvariableop_resource:a
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:aa*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿar
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:a*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_499_layer_call_fn_760324

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_757214o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_500_layer_call_fn_760420

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_757249o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_500_layer_call_fn_760433

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_757296o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
È	
ö
E__inference_dense_557_layer_call_and_return_conditional_losses_760843

inputs0
matmul_readvariableop_resource:aa-
biasadd_readvariableop_resource:a
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:aa*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿar
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:a*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_502_layer_call_and_return_conditional_losses_758021

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿa:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_760933

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿa:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
â
B
__inference__traced_save_761727
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_552_kernel_read_readvariableop-
)savev2_dense_552_bias_read_readvariableop<
8savev2_batch_normalization_499_gamma_read_readvariableop;
7savev2_batch_normalization_499_beta_read_readvariableopB
>savev2_batch_normalization_499_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_499_moving_variance_read_readvariableop/
+savev2_dense_553_kernel_read_readvariableop-
)savev2_dense_553_bias_read_readvariableop<
8savev2_batch_normalization_500_gamma_read_readvariableop;
7savev2_batch_normalization_500_beta_read_readvariableopB
>savev2_batch_normalization_500_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_500_moving_variance_read_readvariableop/
+savev2_dense_554_kernel_read_readvariableop-
)savev2_dense_554_bias_read_readvariableop<
8savev2_batch_normalization_501_gamma_read_readvariableop;
7savev2_batch_normalization_501_beta_read_readvariableopB
>savev2_batch_normalization_501_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_501_moving_variance_read_readvariableop/
+savev2_dense_555_kernel_read_readvariableop-
)savev2_dense_555_bias_read_readvariableop<
8savev2_batch_normalization_502_gamma_read_readvariableop;
7savev2_batch_normalization_502_beta_read_readvariableopB
>savev2_batch_normalization_502_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_502_moving_variance_read_readvariableop/
+savev2_dense_556_kernel_read_readvariableop-
)savev2_dense_556_bias_read_readvariableop<
8savev2_batch_normalization_503_gamma_read_readvariableop;
7savev2_batch_normalization_503_beta_read_readvariableopB
>savev2_batch_normalization_503_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_503_moving_variance_read_readvariableop/
+savev2_dense_557_kernel_read_readvariableop-
)savev2_dense_557_bias_read_readvariableop<
8savev2_batch_normalization_504_gamma_read_readvariableop;
7savev2_batch_normalization_504_beta_read_readvariableopB
>savev2_batch_normalization_504_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_504_moving_variance_read_readvariableop/
+savev2_dense_558_kernel_read_readvariableop-
)savev2_dense_558_bias_read_readvariableop<
8savev2_batch_normalization_505_gamma_read_readvariableop;
7savev2_batch_normalization_505_beta_read_readvariableopB
>savev2_batch_normalization_505_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_505_moving_variance_read_readvariableop/
+savev2_dense_559_kernel_read_readvariableop-
)savev2_dense_559_bias_read_readvariableop<
8savev2_batch_normalization_506_gamma_read_readvariableop;
7savev2_batch_normalization_506_beta_read_readvariableopB
>savev2_batch_normalization_506_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_506_moving_variance_read_readvariableop/
+savev2_dense_560_kernel_read_readvariableop-
)savev2_dense_560_bias_read_readvariableop<
8savev2_batch_normalization_507_gamma_read_readvariableop;
7savev2_batch_normalization_507_beta_read_readvariableopB
>savev2_batch_normalization_507_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_507_moving_variance_read_readvariableop/
+savev2_dense_561_kernel_read_readvariableop-
)savev2_dense_561_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_552_kernel_m_read_readvariableop4
0savev2_adam_dense_552_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_499_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_499_beta_m_read_readvariableop6
2savev2_adam_dense_553_kernel_m_read_readvariableop4
0savev2_adam_dense_553_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_500_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_500_beta_m_read_readvariableop6
2savev2_adam_dense_554_kernel_m_read_readvariableop4
0savev2_adam_dense_554_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_501_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_501_beta_m_read_readvariableop6
2savev2_adam_dense_555_kernel_m_read_readvariableop4
0savev2_adam_dense_555_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_502_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_502_beta_m_read_readvariableop6
2savev2_adam_dense_556_kernel_m_read_readvariableop4
0savev2_adam_dense_556_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_503_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_503_beta_m_read_readvariableop6
2savev2_adam_dense_557_kernel_m_read_readvariableop4
0savev2_adam_dense_557_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_504_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_504_beta_m_read_readvariableop6
2savev2_adam_dense_558_kernel_m_read_readvariableop4
0savev2_adam_dense_558_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_505_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_505_beta_m_read_readvariableop6
2savev2_adam_dense_559_kernel_m_read_readvariableop4
0savev2_adam_dense_559_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_506_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_506_beta_m_read_readvariableop6
2savev2_adam_dense_560_kernel_m_read_readvariableop4
0savev2_adam_dense_560_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_507_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_507_beta_m_read_readvariableop6
2savev2_adam_dense_561_kernel_m_read_readvariableop4
0savev2_adam_dense_561_bias_m_read_readvariableop6
2savev2_adam_dense_552_kernel_v_read_readvariableop4
0savev2_adam_dense_552_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_499_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_499_beta_v_read_readvariableop6
2savev2_adam_dense_553_kernel_v_read_readvariableop4
0savev2_adam_dense_553_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_500_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_500_beta_v_read_readvariableop6
2savev2_adam_dense_554_kernel_v_read_readvariableop4
0savev2_adam_dense_554_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_501_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_501_beta_v_read_readvariableop6
2savev2_adam_dense_555_kernel_v_read_readvariableop4
0savev2_adam_dense_555_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_502_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_502_beta_v_read_readvariableop6
2savev2_adam_dense_556_kernel_v_read_readvariableop4
0savev2_adam_dense_556_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_503_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_503_beta_v_read_readvariableop6
2savev2_adam_dense_557_kernel_v_read_readvariableop4
0savev2_adam_dense_557_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_504_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_504_beta_v_read_readvariableop6
2savev2_adam_dense_558_kernel_v_read_readvariableop4
0savev2_adam_dense_558_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_505_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_505_beta_v_read_readvariableop6
2savev2_adam_dense_559_kernel_v_read_readvariableop4
0savev2_adam_dense_559_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_506_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_506_beta_v_read_readvariableop6
2savev2_adam_dense_560_kernel_v_read_readvariableop4
0savev2_adam_dense_560_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_507_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_507_beta_v_read_readvariableop6
2savev2_adam_dense_561_kernel_v_read_readvariableop4
0savev2_adam_dense_561_bias_v_read_readvariableop
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
: ·O
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ßN
valueÕNBÒNB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*²
value¨B¥B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ·?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_552_kernel_read_readvariableop)savev2_dense_552_bias_read_readvariableop8savev2_batch_normalization_499_gamma_read_readvariableop7savev2_batch_normalization_499_beta_read_readvariableop>savev2_batch_normalization_499_moving_mean_read_readvariableopBsavev2_batch_normalization_499_moving_variance_read_readvariableop+savev2_dense_553_kernel_read_readvariableop)savev2_dense_553_bias_read_readvariableop8savev2_batch_normalization_500_gamma_read_readvariableop7savev2_batch_normalization_500_beta_read_readvariableop>savev2_batch_normalization_500_moving_mean_read_readvariableopBsavev2_batch_normalization_500_moving_variance_read_readvariableop+savev2_dense_554_kernel_read_readvariableop)savev2_dense_554_bias_read_readvariableop8savev2_batch_normalization_501_gamma_read_readvariableop7savev2_batch_normalization_501_beta_read_readvariableop>savev2_batch_normalization_501_moving_mean_read_readvariableopBsavev2_batch_normalization_501_moving_variance_read_readvariableop+savev2_dense_555_kernel_read_readvariableop)savev2_dense_555_bias_read_readvariableop8savev2_batch_normalization_502_gamma_read_readvariableop7savev2_batch_normalization_502_beta_read_readvariableop>savev2_batch_normalization_502_moving_mean_read_readvariableopBsavev2_batch_normalization_502_moving_variance_read_readvariableop+savev2_dense_556_kernel_read_readvariableop)savev2_dense_556_bias_read_readvariableop8savev2_batch_normalization_503_gamma_read_readvariableop7savev2_batch_normalization_503_beta_read_readvariableop>savev2_batch_normalization_503_moving_mean_read_readvariableopBsavev2_batch_normalization_503_moving_variance_read_readvariableop+savev2_dense_557_kernel_read_readvariableop)savev2_dense_557_bias_read_readvariableop8savev2_batch_normalization_504_gamma_read_readvariableop7savev2_batch_normalization_504_beta_read_readvariableop>savev2_batch_normalization_504_moving_mean_read_readvariableopBsavev2_batch_normalization_504_moving_variance_read_readvariableop+savev2_dense_558_kernel_read_readvariableop)savev2_dense_558_bias_read_readvariableop8savev2_batch_normalization_505_gamma_read_readvariableop7savev2_batch_normalization_505_beta_read_readvariableop>savev2_batch_normalization_505_moving_mean_read_readvariableopBsavev2_batch_normalization_505_moving_variance_read_readvariableop+savev2_dense_559_kernel_read_readvariableop)savev2_dense_559_bias_read_readvariableop8savev2_batch_normalization_506_gamma_read_readvariableop7savev2_batch_normalization_506_beta_read_readvariableop>savev2_batch_normalization_506_moving_mean_read_readvariableopBsavev2_batch_normalization_506_moving_variance_read_readvariableop+savev2_dense_560_kernel_read_readvariableop)savev2_dense_560_bias_read_readvariableop8savev2_batch_normalization_507_gamma_read_readvariableop7savev2_batch_normalization_507_beta_read_readvariableop>savev2_batch_normalization_507_moving_mean_read_readvariableopBsavev2_batch_normalization_507_moving_variance_read_readvariableop+savev2_dense_561_kernel_read_readvariableop)savev2_dense_561_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_552_kernel_m_read_readvariableop0savev2_adam_dense_552_bias_m_read_readvariableop?savev2_adam_batch_normalization_499_gamma_m_read_readvariableop>savev2_adam_batch_normalization_499_beta_m_read_readvariableop2savev2_adam_dense_553_kernel_m_read_readvariableop0savev2_adam_dense_553_bias_m_read_readvariableop?savev2_adam_batch_normalization_500_gamma_m_read_readvariableop>savev2_adam_batch_normalization_500_beta_m_read_readvariableop2savev2_adam_dense_554_kernel_m_read_readvariableop0savev2_adam_dense_554_bias_m_read_readvariableop?savev2_adam_batch_normalization_501_gamma_m_read_readvariableop>savev2_adam_batch_normalization_501_beta_m_read_readvariableop2savev2_adam_dense_555_kernel_m_read_readvariableop0savev2_adam_dense_555_bias_m_read_readvariableop?savev2_adam_batch_normalization_502_gamma_m_read_readvariableop>savev2_adam_batch_normalization_502_beta_m_read_readvariableop2savev2_adam_dense_556_kernel_m_read_readvariableop0savev2_adam_dense_556_bias_m_read_readvariableop?savev2_adam_batch_normalization_503_gamma_m_read_readvariableop>savev2_adam_batch_normalization_503_beta_m_read_readvariableop2savev2_adam_dense_557_kernel_m_read_readvariableop0savev2_adam_dense_557_bias_m_read_readvariableop?savev2_adam_batch_normalization_504_gamma_m_read_readvariableop>savev2_adam_batch_normalization_504_beta_m_read_readvariableop2savev2_adam_dense_558_kernel_m_read_readvariableop0savev2_adam_dense_558_bias_m_read_readvariableop?savev2_adam_batch_normalization_505_gamma_m_read_readvariableop>savev2_adam_batch_normalization_505_beta_m_read_readvariableop2savev2_adam_dense_559_kernel_m_read_readvariableop0savev2_adam_dense_559_bias_m_read_readvariableop?savev2_adam_batch_normalization_506_gamma_m_read_readvariableop>savev2_adam_batch_normalization_506_beta_m_read_readvariableop2savev2_adam_dense_560_kernel_m_read_readvariableop0savev2_adam_dense_560_bias_m_read_readvariableop?savev2_adam_batch_normalization_507_gamma_m_read_readvariableop>savev2_adam_batch_normalization_507_beta_m_read_readvariableop2savev2_adam_dense_561_kernel_m_read_readvariableop0savev2_adam_dense_561_bias_m_read_readvariableop2savev2_adam_dense_552_kernel_v_read_readvariableop0savev2_adam_dense_552_bias_v_read_readvariableop?savev2_adam_batch_normalization_499_gamma_v_read_readvariableop>savev2_adam_batch_normalization_499_beta_v_read_readvariableop2savev2_adam_dense_553_kernel_v_read_readvariableop0savev2_adam_dense_553_bias_v_read_readvariableop?savev2_adam_batch_normalization_500_gamma_v_read_readvariableop>savev2_adam_batch_normalization_500_beta_v_read_readvariableop2savev2_adam_dense_554_kernel_v_read_readvariableop0savev2_adam_dense_554_bias_v_read_readvariableop?savev2_adam_batch_normalization_501_gamma_v_read_readvariableop>savev2_adam_batch_normalization_501_beta_v_read_readvariableop2savev2_adam_dense_555_kernel_v_read_readvariableop0savev2_adam_dense_555_bias_v_read_readvariableop?savev2_adam_batch_normalization_502_gamma_v_read_readvariableop>savev2_adam_batch_normalization_502_beta_v_read_readvariableop2savev2_adam_dense_556_kernel_v_read_readvariableop0savev2_adam_dense_556_bias_v_read_readvariableop?savev2_adam_batch_normalization_503_gamma_v_read_readvariableop>savev2_adam_batch_normalization_503_beta_v_read_readvariableop2savev2_adam_dense_557_kernel_v_read_readvariableop0savev2_adam_dense_557_bias_v_read_readvariableop?savev2_adam_batch_normalization_504_gamma_v_read_readvariableop>savev2_adam_batch_normalization_504_beta_v_read_readvariableop2savev2_adam_dense_558_kernel_v_read_readvariableop0savev2_adam_dense_558_bias_v_read_readvariableop?savev2_adam_batch_normalization_505_gamma_v_read_readvariableop>savev2_adam_batch_normalization_505_beta_v_read_readvariableop2savev2_adam_dense_559_kernel_v_read_readvariableop0savev2_adam_dense_559_bias_v_read_readvariableop?savev2_adam_batch_normalization_506_gamma_v_read_readvariableop>savev2_adam_batch_normalization_506_beta_v_read_readvariableop2savev2_adam_dense_560_kernel_v_read_readvariableop0savev2_adam_dense_560_bias_v_read_readvariableop?savev2_adam_batch_normalization_507_gamma_v_read_readvariableop>savev2_adam_batch_normalization_507_beta_v_read_readvariableop2savev2_adam_dense_561_kernel_v_read_readvariableop0savev2_adam_dense_561_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *
dtypes
2		
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

identity_1Identity_1:output:0*Ã
_input_shapes±
®: ::: :P:P:P:P:P:P:PP:P:P:P:P:P:Pa:a:a:a:a:a:aa:a:a:a:a:a:aa:a:a:a:a:a:aa:a:a:a:a:a:a=:=:=:=:=:=:==:=:=:=:=:=:==:=:=:=:=:=:=:: : : : : : :P:P:P:P:PP:P:P:P:Pa:a:a:a:aa:a:a:a:aa:a:a:a:aa:a:a:a:a=:=:=:=:==:=:=:=:==:=:=:=:=::P:P:P:P:PP:P:P:P:Pa:a:a:a:aa:a:a:a:aa:a:a:a:aa:a:a:a:a=:=:=:=:==:=:=:=:==:=:=:=:=:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :$ 

_output_shapes

:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 	

_output_shapes
:P:$
 

_output_shapes

:PP: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P:$ 

_output_shapes

:Pa: 

_output_shapes
:a: 

_output_shapes
:a: 

_output_shapes
:a: 

_output_shapes
:a: 

_output_shapes
:a:$ 

_output_shapes

:aa: 

_output_shapes
:a: 

_output_shapes
:a: 

_output_shapes
:a: 

_output_shapes
:a: 

_output_shapes
:a:$ 

_output_shapes

:aa: 

_output_shapes
:a: 

_output_shapes
:a: 

_output_shapes
:a:  

_output_shapes
:a: !

_output_shapes
:a:$" 

_output_shapes

:aa: #

_output_shapes
:a: $

_output_shapes
:a: %

_output_shapes
:a: &

_output_shapes
:a: '

_output_shapes
:a:$( 

_output_shapes

:a=: )

_output_shapes
:=: *

_output_shapes
:=: +

_output_shapes
:=: ,

_output_shapes
:=: -

_output_shapes
:=:$. 

_output_shapes

:==: /

_output_shapes
:=: 0

_output_shapes
:=: 1

_output_shapes
:=: 2

_output_shapes
:=: 3

_output_shapes
:=:$4 

_output_shapes

:==: 5

_output_shapes
:=: 6

_output_shapes
:=: 7

_output_shapes
:=: 8

_output_shapes
:=: 9

_output_shapes
:=:$: 

_output_shapes

:=: ;

_output_shapes
::<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :$B 

_output_shapes

:P: C

_output_shapes
:P: D

_output_shapes
:P: E

_output_shapes
:P:$F 

_output_shapes

:PP: G

_output_shapes
:P: H

_output_shapes
:P: I

_output_shapes
:P:$J 

_output_shapes

:Pa: K

_output_shapes
:a: L

_output_shapes
:a: M

_output_shapes
:a:$N 

_output_shapes

:aa: O

_output_shapes
:a: P

_output_shapes
:a: Q

_output_shapes
:a:$R 

_output_shapes

:aa: S

_output_shapes
:a: T

_output_shapes
:a: U

_output_shapes
:a:$V 

_output_shapes

:aa: W

_output_shapes
:a: X

_output_shapes
:a: Y

_output_shapes
:a:$Z 

_output_shapes

:a=: [

_output_shapes
:=: \

_output_shapes
:=: ]

_output_shapes
:=:$^ 

_output_shapes

:==: _

_output_shapes
:=: `

_output_shapes
:=: a

_output_shapes
:=:$b 

_output_shapes

:==: c

_output_shapes
:=: d

_output_shapes
:=: e

_output_shapes
:=:$f 

_output_shapes

:=: g

_output_shapes
::$h 

_output_shapes

:P: i

_output_shapes
:P: j

_output_shapes
:P: k

_output_shapes
:P:$l 

_output_shapes

:PP: m

_output_shapes
:P: n

_output_shapes
:P: o

_output_shapes
:P:$p 

_output_shapes

:Pa: q

_output_shapes
:a: r

_output_shapes
:a: s

_output_shapes
:a:$t 

_output_shapes

:aa: u

_output_shapes
:a: v

_output_shapes
:a: w

_output_shapes
:a:$x 

_output_shapes

:aa: y

_output_shapes
:a: z

_output_shapes
:a: {

_output_shapes
:a:$| 

_output_shapes

:aa: }

_output_shapes
:a: ~

_output_shapes
:a: 

_output_shapes
:a:% 

_output_shapes

:a=:!

_output_shapes
:=:!

_output_shapes
:=:!

_output_shapes
:=:% 

_output_shapes

:==:!

_output_shapes
:=:!

_output_shapes
:=:!

_output_shapes
:=:% 

_output_shapes

:==:!

_output_shapes
:=:!

_output_shapes
:=:!

_output_shapes
:=:% 

_output_shapes

:=:!

_output_shapes
::

_output_shapes
: 
å
g
K__inference_leaky_re_lu_499_layer_call_and_return_conditional_losses_757925

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿP:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_761032

inputs5
'assignmovingavg_readvariableop_resource:=7
)assignmovingavg_1_readvariableop_resource:=3
%batchnorm_mul_readvariableop_resource:=/
!batchnorm_readvariableop_resource:=
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:=
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:=*
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
:=*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:=x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:=¬
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
:=*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:=~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:=´
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
:=P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:=~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:=v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:=r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_761216

inputs/
!batchnorm_readvariableop_resource:=3
%batchnorm_mul_readvariableop_resource:=1
#batchnorm_readvariableop_1_resource:=1
#batchnorm_readvariableop_2_resource:=
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:=*
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
:=P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:=~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:=z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:=r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_757624

inputs5
'assignmovingavg_readvariableop_resource:a7
)assignmovingavg_1_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a/
!batchnorm_readvariableop_resource:a
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:a
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿal
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:a*
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
:a*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a¬
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
:a*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:a~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a´
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿah
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_758085

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿa:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
È	
ö
E__inference_dense_552_layer_call_and_return_conditional_losses_757905

inputs0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:P
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿPw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
Ò
.__inference_sequential_53_layer_call_fn_759414

inputs
unknown
	unknown_0
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P

unknown_10:P

unknown_11:P

unknown_12:P

unknown_13:Pa

unknown_14:a

unknown_15:a

unknown_16:a

unknown_17:a

unknown_18:a

unknown_19:aa

unknown_20:a

unknown_21:a

unknown_22:a

unknown_23:a

unknown_24:a

unknown_25:aa

unknown_26:a

unknown_27:a

unknown_28:a

unknown_29:a

unknown_30:a

unknown_31:aa

unknown_32:a

unknown_33:a

unknown_34:a

unknown_35:a

unknown_36:a

unknown_37:a=

unknown_38:=

unknown_39:=

unknown_40:=

unknown_41:=

unknown_42:=

unknown_43:==

unknown_44:=

unknown_45:=

unknown_46:=

unknown_47:=

unknown_48:=

unknown_49:==

unknown_50:=

unknown_51:=

unknown_52:=

unknown_53:=

unknown_54:=

unknown_55:=

unknown_56:
identity¢StatefulPartitionedCallã
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
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./0123456789:*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_53_layer_call_and_return_conditional_losses_758200o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_502_layer_call_and_return_conditional_losses_760715

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿa:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
Å
Ò
.__inference_sequential_53_layer_call_fn_759535

inputs
unknown
	unknown_0
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P

unknown_10:P

unknown_11:P

unknown_12:P

unknown_13:Pa

unknown_14:a

unknown_15:a

unknown_16:a

unknown_17:a

unknown_18:a

unknown_19:aa

unknown_20:a

unknown_21:a

unknown_22:a

unknown_23:a

unknown_24:a

unknown_25:aa

unknown_26:a

unknown_27:a

unknown_28:a

unknown_29:a

unknown_30:a

unknown_31:aa

unknown_32:a

unknown_33:a

unknown_34:a

unknown_35:a

unknown_36:a

unknown_37:a=

unknown_38:=

unknown_39:=

unknown_40:=

unknown_41:=

unknown_42:=

unknown_43:==

unknown_44:=

unknown_45:=

unknown_46:=

unknown_47:=

unknown_48:=

unknown_49:==

unknown_50:=

unknown_51:=

unknown_52:=

unknown_53:=

unknown_54:=

unknown_55:=

unknown_56:
identity¢StatefulPartitionedCallÑ
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
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"%&'(+,-.1234789:*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_53_layer_call_and_return_conditional_losses_758747o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
È	
ö
E__inference_dense_561_layer_call_and_return_conditional_losses_761279

inputs0
matmul_readvariableop_resource:=-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:=*
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
:ÿÿÿÿÿÿÿÿÿ=: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_507_layer_call_fn_761196

inputs
unknown:=
	unknown_0:=
	unknown_1:=
	unknown_2:=
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_757870o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_760671

inputs/
!batchnorm_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a1
#batchnorm_readvariableop_1_resource:a1
#batchnorm_readvariableop_2_resource:a
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_502_layer_call_fn_760638

inputs
unknown:a
	unknown_0:a
	unknown_1:a
	unknown_2:a
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_757413o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_757823

inputs/
!batchnorm_readvariableop_resource:=3
%batchnorm_mul_readvariableop_resource:=1
#batchnorm_readvariableop_1_resource:=1
#batchnorm_readvariableop_2_resource:=
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:=*
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
:=P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:=~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:=z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:=r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_757495

inputs/
!batchnorm_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a1
#batchnorm_readvariableop_1_resource:a1
#batchnorm_readvariableop_2_resource:a
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_760562

inputs/
!batchnorm_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a1
#batchnorm_readvariableop_1_resource:a1
#batchnorm_readvariableop_2_resource:a
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_758181

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ="
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_761141

inputs5
'assignmovingavg_readvariableop_resource:=7
)assignmovingavg_1_readvariableop_resource:=3
%batchnorm_mul_readvariableop_resource:=/
!batchnorm_readvariableop_resource:=
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:=
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:=*
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
:=*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:=x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:=¬
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
:=*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:=~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:=´
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
:=P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:=~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:=v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:=r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_760705

inputs5
'assignmovingavg_readvariableop_resource:a7
)assignmovingavg_1_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a/
!batchnorm_readvariableop_resource:a
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:a
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿal
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:a*
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
:a*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a¬
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
:a*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:a~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a´
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿah
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
Ä

*__inference_dense_555_layer_call_fn_760615

inputs
unknown:aa
	unknown_0:a
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_555_layer_call_and_return_conditional_losses_758001o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_760889

inputs/
!batchnorm_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a1
#batchnorm_readvariableop_1_resource:a1
#batchnorm_readvariableop_2_resource:a
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_757460

inputs5
'assignmovingavg_readvariableop_resource:a7
)assignmovingavg_1_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a/
!batchnorm_readvariableop_resource:a
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:a
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿal
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:a*
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
:a*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a¬
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
:a*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:a~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a´
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿah
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
Ä

*__inference_dense_556_layer_call_fn_760724

inputs
unknown:aa
	unknown_0:a
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_556_layer_call_and_return_conditional_losses_758033o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_502_layer_call_fn_760710

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
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_502_layer_call_and_return_conditional_losses_758021`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿa:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs

ë
I__inference_sequential_53_layer_call_and_return_conditional_losses_758747

inputs
normalization_53_sub_y
normalization_53_sqrt_x"
dense_552_758606:P
dense_552_758608:P,
batch_normalization_499_758611:P,
batch_normalization_499_758613:P,
batch_normalization_499_758615:P,
batch_normalization_499_758617:P"
dense_553_758621:PP
dense_553_758623:P,
batch_normalization_500_758626:P,
batch_normalization_500_758628:P,
batch_normalization_500_758630:P,
batch_normalization_500_758632:P"
dense_554_758636:Pa
dense_554_758638:a,
batch_normalization_501_758641:a,
batch_normalization_501_758643:a,
batch_normalization_501_758645:a,
batch_normalization_501_758647:a"
dense_555_758651:aa
dense_555_758653:a,
batch_normalization_502_758656:a,
batch_normalization_502_758658:a,
batch_normalization_502_758660:a,
batch_normalization_502_758662:a"
dense_556_758666:aa
dense_556_758668:a,
batch_normalization_503_758671:a,
batch_normalization_503_758673:a,
batch_normalization_503_758675:a,
batch_normalization_503_758677:a"
dense_557_758681:aa
dense_557_758683:a,
batch_normalization_504_758686:a,
batch_normalization_504_758688:a,
batch_normalization_504_758690:a,
batch_normalization_504_758692:a"
dense_558_758696:a=
dense_558_758698:=,
batch_normalization_505_758701:=,
batch_normalization_505_758703:=,
batch_normalization_505_758705:=,
batch_normalization_505_758707:="
dense_559_758711:==
dense_559_758713:=,
batch_normalization_506_758716:=,
batch_normalization_506_758718:=,
batch_normalization_506_758720:=,
batch_normalization_506_758722:="
dense_560_758726:==
dense_560_758728:=,
batch_normalization_507_758731:=,
batch_normalization_507_758733:=,
batch_normalization_507_758735:=,
batch_normalization_507_758737:="
dense_561_758741:=
dense_561_758743:
identity¢/batch_normalization_499/StatefulPartitionedCall¢/batch_normalization_500/StatefulPartitionedCall¢/batch_normalization_501/StatefulPartitionedCall¢/batch_normalization_502/StatefulPartitionedCall¢/batch_normalization_503/StatefulPartitionedCall¢/batch_normalization_504/StatefulPartitionedCall¢/batch_normalization_505/StatefulPartitionedCall¢/batch_normalization_506/StatefulPartitionedCall¢/batch_normalization_507/StatefulPartitionedCall¢!dense_552/StatefulPartitionedCall¢!dense_553/StatefulPartitionedCall¢!dense_554/StatefulPartitionedCall¢!dense_555/StatefulPartitionedCall¢!dense_556/StatefulPartitionedCall¢!dense_557/StatefulPartitionedCall¢!dense_558/StatefulPartitionedCall¢!dense_559/StatefulPartitionedCall¢!dense_560/StatefulPartitionedCall¢!dense_561/StatefulPartitionedCallm
normalization_53/subSubinputsnormalization_53_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_53/SqrtSqrtnormalization_53_sqrt_x*
T0*
_output_shapes

:_
normalization_53/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_53/MaximumMaximumnormalization_53/Sqrt:y:0#normalization_53/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_53/truedivRealDivnormalization_53/sub:z:0normalization_53/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_552/StatefulPartitionedCallStatefulPartitionedCallnormalization_53/truediv:z:0dense_552_758606dense_552_758608*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_552_layer_call_and_return_conditional_losses_757905
/batch_normalization_499/StatefulPartitionedCallStatefulPartitionedCall*dense_552/StatefulPartitionedCall:output:0batch_normalization_499_758611batch_normalization_499_758613batch_normalization_499_758615batch_normalization_499_758617*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_757214ø
leaky_re_lu_499/PartitionedCallPartitionedCall8batch_normalization_499/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_499_layer_call_and_return_conditional_losses_757925
!dense_553/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_499/PartitionedCall:output:0dense_553_758621dense_553_758623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_553_layer_call_and_return_conditional_losses_757937
/batch_normalization_500/StatefulPartitionedCallStatefulPartitionedCall*dense_553/StatefulPartitionedCall:output:0batch_normalization_500_758626batch_normalization_500_758628batch_normalization_500_758630batch_normalization_500_758632*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_757296ø
leaky_re_lu_500/PartitionedCallPartitionedCall8batch_normalization_500/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_500_layer_call_and_return_conditional_losses_757957
!dense_554/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_500/PartitionedCall:output:0dense_554_758636dense_554_758638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_554_layer_call_and_return_conditional_losses_757969
/batch_normalization_501/StatefulPartitionedCallStatefulPartitionedCall*dense_554/StatefulPartitionedCall:output:0batch_normalization_501_758641batch_normalization_501_758643batch_normalization_501_758645batch_normalization_501_758647*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_757378ø
leaky_re_lu_501/PartitionedCallPartitionedCall8batch_normalization_501/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_501_layer_call_and_return_conditional_losses_757989
!dense_555/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_501/PartitionedCall:output:0dense_555_758651dense_555_758653*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_555_layer_call_and_return_conditional_losses_758001
/batch_normalization_502/StatefulPartitionedCallStatefulPartitionedCall*dense_555/StatefulPartitionedCall:output:0batch_normalization_502_758656batch_normalization_502_758658batch_normalization_502_758660batch_normalization_502_758662*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_757460ø
leaky_re_lu_502/PartitionedCallPartitionedCall8batch_normalization_502/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_502_layer_call_and_return_conditional_losses_758021
!dense_556/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_502/PartitionedCall:output:0dense_556_758666dense_556_758668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_556_layer_call_and_return_conditional_losses_758033
/batch_normalization_503/StatefulPartitionedCallStatefulPartitionedCall*dense_556/StatefulPartitionedCall:output:0batch_normalization_503_758671batch_normalization_503_758673batch_normalization_503_758675batch_normalization_503_758677*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_757542ø
leaky_re_lu_503/PartitionedCallPartitionedCall8batch_normalization_503/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_503_layer_call_and_return_conditional_losses_758053
!dense_557/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_503/PartitionedCall:output:0dense_557_758681dense_557_758683*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_557_layer_call_and_return_conditional_losses_758065
/batch_normalization_504/StatefulPartitionedCallStatefulPartitionedCall*dense_557/StatefulPartitionedCall:output:0batch_normalization_504_758686batch_normalization_504_758688batch_normalization_504_758690batch_normalization_504_758692*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_757624ø
leaky_re_lu_504/PartitionedCallPartitionedCall8batch_normalization_504/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_758085
!dense_558/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_504/PartitionedCall:output:0dense_558_758696dense_558_758698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_558_layer_call_and_return_conditional_losses_758097
/batch_normalization_505/StatefulPartitionedCallStatefulPartitionedCall*dense_558/StatefulPartitionedCall:output:0batch_normalization_505_758701batch_normalization_505_758703batch_normalization_505_758705batch_normalization_505_758707*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_757706ø
leaky_re_lu_505/PartitionedCallPartitionedCall8batch_normalization_505/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_758117
!dense_559/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_505/PartitionedCall:output:0dense_559_758711dense_559_758713*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_559_layer_call_and_return_conditional_losses_758129
/batch_normalization_506/StatefulPartitionedCallStatefulPartitionedCall*dense_559/StatefulPartitionedCall:output:0batch_normalization_506_758716batch_normalization_506_758718batch_normalization_506_758720batch_normalization_506_758722*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_757788ø
leaky_re_lu_506/PartitionedCallPartitionedCall8batch_normalization_506/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_758149
!dense_560/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_506/PartitionedCall:output:0dense_560_758726dense_560_758728*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_560_layer_call_and_return_conditional_losses_758161
/batch_normalization_507/StatefulPartitionedCallStatefulPartitionedCall*dense_560/StatefulPartitionedCall:output:0batch_normalization_507_758731batch_normalization_507_758733batch_normalization_507_758735batch_normalization_507_758737*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_757870ø
leaky_re_lu_507/PartitionedCallPartitionedCall8batch_normalization_507/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_758181
!dense_561/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_507/PartitionedCall:output:0dense_561_758741dense_561_758743*
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
E__inference_dense_561_layer_call_and_return_conditional_losses_758193y
IdentityIdentity*dense_561/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp0^batch_normalization_499/StatefulPartitionedCall0^batch_normalization_500/StatefulPartitionedCall0^batch_normalization_501/StatefulPartitionedCall0^batch_normalization_502/StatefulPartitionedCall0^batch_normalization_503/StatefulPartitionedCall0^batch_normalization_504/StatefulPartitionedCall0^batch_normalization_505/StatefulPartitionedCall0^batch_normalization_506/StatefulPartitionedCall0^batch_normalization_507/StatefulPartitionedCall"^dense_552/StatefulPartitionedCall"^dense_553/StatefulPartitionedCall"^dense_554/StatefulPartitionedCall"^dense_555/StatefulPartitionedCall"^dense_556/StatefulPartitionedCall"^dense_557/StatefulPartitionedCall"^dense_558/StatefulPartitionedCall"^dense_559/StatefulPartitionedCall"^dense_560/StatefulPartitionedCall"^dense_561/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_499/StatefulPartitionedCall/batch_normalization_499/StatefulPartitionedCall2b
/batch_normalization_500/StatefulPartitionedCall/batch_normalization_500/StatefulPartitionedCall2b
/batch_normalization_501/StatefulPartitionedCall/batch_normalization_501/StatefulPartitionedCall2b
/batch_normalization_502/StatefulPartitionedCall/batch_normalization_502/StatefulPartitionedCall2b
/batch_normalization_503/StatefulPartitionedCall/batch_normalization_503/StatefulPartitionedCall2b
/batch_normalization_504/StatefulPartitionedCall/batch_normalization_504/StatefulPartitionedCall2b
/batch_normalization_505/StatefulPartitionedCall/batch_normalization_505/StatefulPartitionedCall2b
/batch_normalization_506/StatefulPartitionedCall/batch_normalization_506/StatefulPartitionedCall2b
/batch_normalization_507/StatefulPartitionedCall/batch_normalization_507/StatefulPartitionedCall2F
!dense_552/StatefulPartitionedCall!dense_552/StatefulPartitionedCall2F
!dense_553/StatefulPartitionedCall!dense_553/StatefulPartitionedCall2F
!dense_554/StatefulPartitionedCall!dense_554/StatefulPartitionedCall2F
!dense_555/StatefulPartitionedCall!dense_555/StatefulPartitionedCall2F
!dense_556/StatefulPartitionedCall!dense_556/StatefulPartitionedCall2F
!dense_557/StatefulPartitionedCall!dense_557/StatefulPartitionedCall2F
!dense_558/StatefulPartitionedCall!dense_558/StatefulPartitionedCall2F
!dense_559/StatefulPartitionedCall!dense_559/StatefulPartitionedCall2F
!dense_560/StatefulPartitionedCall!dense_560/StatefulPartitionedCall2F
!dense_561/StatefulPartitionedCall!dense_561/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_501_layer_call_and_return_conditional_losses_757989

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿa:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_758117

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ="
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Ä

*__inference_dense_557_layer_call_fn_760833

inputs
unknown:aa
	unknown_0:a
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_557_layer_call_and_return_conditional_losses_758065o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
È	
ö
E__inference_dense_555_layer_call_and_return_conditional_losses_760625

inputs0
matmul_readvariableop_resource:aa-
biasadd_readvariableop_resource:a
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:aa*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿar
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:a*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_760596

inputs5
'assignmovingavg_readvariableop_resource:a7
)assignmovingavg_1_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a/
!batchnorm_readvariableop_resource:a
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:a
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿal
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:a*
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
:a*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a¬
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
:a*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:a~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a´
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿah
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_501_layer_call_fn_760601

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
:ÿÿÿÿÿÿÿÿÿa* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_501_layer_call_and_return_conditional_losses_757989`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿa:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_757870

inputs5
'assignmovingavg_readvariableop_resource:=7
)assignmovingavg_1_readvariableop_resource:=3
%batchnorm_mul_readvariableop_resource:=/
!batchnorm_readvariableop_resource:=
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:=
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:=*
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
:=*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:=x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:=¬
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
:=*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:=~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:=´
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
:=P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:=~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:=v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:=r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
È	
ö
E__inference_dense_559_layer_call_and_return_conditional_losses_761061

inputs0
matmul_readvariableop_resource:==-
biasadd_readvariableop_resource:=
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:==*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:=*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs

â
.__inference_sequential_53_layer_call_fn_758319
normalization_53_input
unknown
	unknown_0
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P

unknown_10:P

unknown_11:P

unknown_12:P

unknown_13:Pa

unknown_14:a

unknown_15:a

unknown_16:a

unknown_17:a

unknown_18:a

unknown_19:aa

unknown_20:a

unknown_21:a

unknown_22:a

unknown_23:a

unknown_24:a

unknown_25:aa

unknown_26:a

unknown_27:a

unknown_28:a

unknown_29:a

unknown_30:a

unknown_31:aa

unknown_32:a

unknown_33:a

unknown_34:a

unknown_35:a

unknown_36:a

unknown_37:a=

unknown_38:=

unknown_39:=

unknown_40:=

unknown_41:=

unknown_42:=

unknown_43:==

unknown_44:=

unknown_45:=

unknown_46:=

unknown_47:=

unknown_48:=

unknown_49:==

unknown_50:=

unknown_51:=

unknown_52:=

unknown_53:=

unknown_54:=

unknown_55:=

unknown_56:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallnormalization_53_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./0123456789:*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_53_layer_call_and_return_conditional_losses_758200o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_53_input:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_761151

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ="
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_761042

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ="
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
õ
â
.__inference_sequential_53_layer_call_fn_758987
normalization_53_input
unknown
	unknown_0
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P

unknown_10:P

unknown_11:P

unknown_12:P

unknown_13:Pa

unknown_14:a

unknown_15:a

unknown_16:a

unknown_17:a

unknown_18:a

unknown_19:aa

unknown_20:a

unknown_21:a

unknown_22:a

unknown_23:a

unknown_24:a

unknown_25:aa

unknown_26:a

unknown_27:a

unknown_28:a

unknown_29:a

unknown_30:a

unknown_31:aa

unknown_32:a

unknown_33:a

unknown_34:a

unknown_35:a

unknown_36:a

unknown_37:a=

unknown_38:=

unknown_39:=

unknown_40:=

unknown_41:=

unknown_42:=

unknown_43:==

unknown_44:=

unknown_45:=

unknown_46:=

unknown_47:=

unknown_48:=

unknown_49:==

unknown_50:=

unknown_51:=

unknown_52:=

unknown_53:=

unknown_54:=

unknown_55:=

unknown_56:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallnormalization_53_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"%&'(+,-.1234789:*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_53_layer_call_and_return_conditional_losses_758747o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_53_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ð
²
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_760780

inputs/
!batchnorm_readvariableop_resource:a3
%batchnorm_mul_readvariableop_resource:a1
#batchnorm_readvariableop_1_resource:a1
#batchnorm_readvariableop_2_resource:a
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:a*
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
:aP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:a~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿab
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿa: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
Ä

*__inference_dense_553_layer_call_fn_760397

inputs
unknown:PP
	unknown_0:P
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_553_layer_call_and_return_conditional_losses_757937o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿP: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_757741

inputs/
!batchnorm_readvariableop_resource:=3
%batchnorm_mul_readvariableop_resource:=1
#batchnorm_readvariableop_1_resource:=1
#batchnorm_readvariableop_2_resource:=
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:=*
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
:=P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:=~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:=z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:=r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Ä

*__inference_dense_552_layer_call_fn_760288

inputs
unknown:P
	unknown_0:P
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_552_layer_call_and_return_conditional_losses_757905o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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
normalization_53_input?
(serving_default_normalization_53_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_5610
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¤ø
	
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
layer_with_weights-18
layer-26
layer-27
layer_with_weights-19
layer-28
	optimizer
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_default_save_signature
&
signatures"
_tf_keras_sequential
Ó
'
_keep_axis
(_reduce_axis
)_reduce_axis_mask
*_broadcast_shape
+mean
+
adapt_mean
,variance
,adapt_variance
	-count
.	keras_api
/_adapt_function"
_tf_keras_layer
»

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
»

bkernel
cbias
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
jaxis
	kgamma
lbeta
mmoving_mean
nmoving_variance
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
¾

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
 moving_variance
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
«
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
­kernel
	®bias
¯	variables
°trainable_variables
±regularization_losses
²	keras_api
³__call__
+´&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	µaxis

¶gamma
	·beta
¸moving_mean
¹moving_variance
º	variables
»trainable_variables
¼regularization_losses
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"
_tf_keras_layer
«
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ækernel
	Çbias
È	variables
Étrainable_variables
Êregularization_losses
Ë	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	Îaxis

Ïgamma
	Ðbeta
Ñmoving_mean
Òmoving_variance
Ó	variables
Ôtrainable_variables
Õregularization_losses
Ö	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
ßkernel
	àbias
á	variables
âtrainable_variables
ãregularization_losses
ä	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	çaxis

ègamma
	ébeta
êmoving_mean
ëmoving_variance
ì	variables
ítrainable_variables
îregularization_losses
ï	keras_api
ð__call__
+ñ&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ò	variables
ótrainable_variables
ôregularization_losses
õ	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
økernel
	ùbias
ú	variables
ûtrainable_variables
üregularization_losses
ý	keras_api
þ__call__
+ÿ&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ì
	iter
beta_1
beta_2

decay0m´1mµ9m¶:m·Im¸Jm¹RmºSm»bm¼cm½km¾lm¿{mÀ|mÁ	mÂ	mÃ	mÄ	mÅ	mÆ	mÇ	­mÈ	®mÉ	¶mÊ	·mË	ÆmÌ	ÇmÍ	ÏmÎ	ÐmÏ	ßmÐ	àmÑ	èmÒ	émÓ	ømÔ	ùmÕ	mÖ	m×	mØ	mÙ0vÚ1vÛ9vÜ:vÝIvÞJvßRvàSvábvâcvãkvälvå{væ|vç	vè	vé	vê	vë	vì	ví	­vî	®vï	¶vð	·vñ	Ævò	Çvó	Ïvô	Ðvõ	ßvö	àv÷	èvø	évù	øvú	ùvû	vü	vý	vþ	vÿ"
	optimizer

+0
,1
-2
03
14
95
:6
;7
<8
I9
J10
R11
S12
T13
U14
b15
c16
k17
l18
m19
n20
{21
|22
23
24
25
26
27
28
29
30
31
 32
­33
®34
¶35
·36
¸37
¹38
Æ39
Ç40
Ï41
Ð42
Ñ43
Ò44
ß45
à46
è47
é48
ê49
ë50
ø51
ù52
53
54
55
56
57
58"
trackable_list_wrapper
Þ
00
11
92
:3
I4
J5
R6
S7
b8
c9
k10
l11
{12
|13
14
15
16
17
18
19
­20
®21
¶22
·23
Æ24
Ç25
Ï26
Ð27
ß28
à29
è30
é31
ø32
ù33
34
35
36
37"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
%_default_save_signature
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_sequential_53_layer_call_fn_758319
.__inference_sequential_53_layer_call_fn_759414
.__inference_sequential_53_layer_call_fn_759535
.__inference_sequential_53_layer_call_fn_758987À
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
I__inference_sequential_53_layer_call_and_return_conditional_losses_759759
I__inference_sequential_53_layer_call_and_return_conditional_losses_760109
I__inference_sequential_53_layer_call_and_return_conditional_losses_759138
I__inference_sequential_53_layer_call_and_return_conditional_losses_759289À
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
!__inference__wrapped_model_757143normalization_53_input"
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
¢serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
¿2¼
__inference_adapt_step_760279
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
": P2dense_552/kernel
:P2dense_552/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
²
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_552_layer_call_fn_760288¢
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
E__inference_dense_552_layer_call_and_return_conditional_losses_760298¢
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
+:)P2batch_normalization_499/gamma
*:(P2batch_normalization_499/beta
3:1P (2#batch_normalization_499/moving_mean
7:5P (2'batch_normalization_499/moving_variance
<
90
:1
;2
<3"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_499_layer_call_fn_760311
8__inference_batch_normalization_499_layer_call_fn_760324´
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
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_760344
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_760378´
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
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_499_layer_call_fn_760383¢
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
K__inference_leaky_re_lu_499_layer_call_and_return_conditional_losses_760388¢
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
": PP2dense_553/kernel
:P2dense_553/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_553_layer_call_fn_760397¢
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
E__inference_dense_553_layer_call_and_return_conditional_losses_760407¢
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
+:)P2batch_normalization_500/gamma
*:(P2batch_normalization_500/beta
3:1P (2#batch_normalization_500/moving_mean
7:5P (2'batch_normalization_500/moving_variance
<
R0
S1
T2
U3"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_500_layer_call_fn_760420
8__inference_batch_normalization_500_layer_call_fn_760433´
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
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_760453
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_760487´
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
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_500_layer_call_fn_760492¢
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
K__inference_leaky_re_lu_500_layer_call_and_return_conditional_losses_760497¢
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
": Pa2dense_554/kernel
:a2dense_554/bias
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_554_layer_call_fn_760506¢
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
E__inference_dense_554_layer_call_and_return_conditional_losses_760516¢
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
+:)a2batch_normalization_501/gamma
*:(a2batch_normalization_501/beta
3:1a (2#batch_normalization_501/moving_mean
7:5a (2'batch_normalization_501/moving_variance
<
k0
l1
m2
n3"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_501_layer_call_fn_760529
8__inference_batch_normalization_501_layer_call_fn_760542´
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
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_760562
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_760596´
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
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_501_layer_call_fn_760601¢
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
K__inference_leaky_re_lu_501_layer_call_and_return_conditional_losses_760606¢
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
": aa2dense_555/kernel
:a2dense_555/bias
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_555_layer_call_fn_760615¢
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
E__inference_dense_555_layer_call_and_return_conditional_losses_760625¢
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
+:)a2batch_normalization_502/gamma
*:(a2batch_normalization_502/beta
3:1a (2#batch_normalization_502/moving_mean
7:5a (2'batch_normalization_502/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_502_layer_call_fn_760638
8__inference_batch_normalization_502_layer_call_fn_760651´
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
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_760671
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_760705´
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
Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_502_layer_call_fn_760710¢
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
K__inference_leaky_re_lu_502_layer_call_and_return_conditional_losses_760715¢
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
": aa2dense_556/kernel
:a2dense_556/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_556_layer_call_fn_760724¢
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
E__inference_dense_556_layer_call_and_return_conditional_losses_760734¢
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
+:)a2batch_normalization_503/gamma
*:(a2batch_normalization_503/beta
3:1a (2#batch_normalization_503/moving_mean
7:5a (2'batch_normalization_503/moving_variance
@
0
1
2
 3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_503_layer_call_fn_760747
8__inference_batch_normalization_503_layer_call_fn_760760´
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
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_760780
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_760814´
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
énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_503_layer_call_fn_760819¢
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
K__inference_leaky_re_lu_503_layer_call_and_return_conditional_losses_760824¢
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
": aa2dense_557/kernel
:a2dense_557/bias
0
­0
®1"
trackable_list_wrapper
0
­0
®1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
¯	variables
°trainable_variables
±regularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_557_layer_call_fn_760833¢
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
E__inference_dense_557_layer_call_and_return_conditional_losses_760843¢
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
+:)a2batch_normalization_504/gamma
*:(a2batch_normalization_504/beta
3:1a (2#batch_normalization_504/moving_mean
7:5a (2'batch_normalization_504/moving_variance
@
¶0
·1
¸2
¹3"
trackable_list_wrapper
0
¶0
·1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_504_layer_call_fn_760856
8__inference_batch_normalization_504_layer_call_fn_760869´
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
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_760889
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_760923´
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
ønon_trainable_variables
ùlayers
úmetrics
 ûlayer_regularization_losses
ülayer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_504_layer_call_fn_760928¢
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
K__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_760933¢
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
": a=2dense_558/kernel
:=2dense_558/bias
0
Æ0
Ç1"
trackable_list_wrapper
0
Æ0
Ç1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
È	variables
Étrainable_variables
Êregularization_losses
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_558_layer_call_fn_760942¢
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
E__inference_dense_558_layer_call_and_return_conditional_losses_760952¢
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
+:)=2batch_normalization_505/gamma
*:(=2batch_normalization_505/beta
3:1= (2#batch_normalization_505/moving_mean
7:5= (2'batch_normalization_505/moving_variance
@
Ï0
Ð1
Ñ2
Ò3"
trackable_list_wrapper
0
Ï0
Ð1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ó	variables
Ôtrainable_variables
Õregularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_505_layer_call_fn_760965
8__inference_batch_normalization_505_layer_call_fn_760978´
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
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_760998
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_761032´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ù	variables
Útrainable_variables
Ûregularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_505_layer_call_fn_761037¢
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
K__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_761042¢
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
": ==2dense_559/kernel
:=2dense_559/bias
0
ß0
à1"
trackable_list_wrapper
0
ß0
à1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
á	variables
âtrainable_variables
ãregularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_559_layer_call_fn_761051¢
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
E__inference_dense_559_layer_call_and_return_conditional_losses_761061¢
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
+:)=2batch_normalization_506/gamma
*:(=2batch_normalization_506/beta
3:1= (2#batch_normalization_506/moving_mean
7:5= (2'batch_normalization_506/moving_variance
@
è0
é1
ê2
ë3"
trackable_list_wrapper
0
è0
é1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ì	variables
ítrainable_variables
îregularization_losses
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_506_layer_call_fn_761074
8__inference_batch_normalization_506_layer_call_fn_761087´
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
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_761107
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_761141´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ò	variables
ótrainable_variables
ôregularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_506_layer_call_fn_761146¢
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
K__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_761151¢
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
": ==2dense_560/kernel
:=2dense_560/bias
0
ø0
ù1"
trackable_list_wrapper
0
ø0
ù1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ú	variables
ûtrainable_variables
üregularization_losses
þ__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_560_layer_call_fn_761160¢
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
E__inference_dense_560_layer_call_and_return_conditional_losses_761170¢
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
+:)=2batch_normalization_507/gamma
*:(=2batch_normalization_507/beta
3:1= (2#batch_normalization_507/moving_mean
7:5= (2'batch_normalization_507/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_507_layer_call_fn_761183
8__inference_batch_normalization_507_layer_call_fn_761196´
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
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_761216
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_761250´
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
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_507_layer_call_fn_761255¢
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
K__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_761260¢
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
": =2dense_561/kernel
:2dense_561/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_561_layer_call_fn_761269¢
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
E__inference_dense_561_layer_call_and_return_conditional_losses_761279¢
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
Ê
+0
,1
-2
;3
<4
T5
U6
m7
n8
9
10
11
 12
¸13
¹14
Ñ15
Ò16
ê17
ë18
19
20"
trackable_list_wrapper
þ
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
25
26
27
28"
trackable_list_wrapper
(
¯0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
$__inference_signature_wrapper_760232normalization_53_input"
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
;0
<1"
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
T0
U1"
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
m0
n1"
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
0
1"
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
0
 1"
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
¸0
¹1"
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
Ñ0
Ò1"
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
ê0
ë1"
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
0
1"
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

°total

±count
²	variables
³	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
°0
±1"
trackable_list_wrapper
.
²	variables"
_generic_user_object
':%P2Adam/dense_552/kernel/m
!:P2Adam/dense_552/bias/m
0:.P2$Adam/batch_normalization_499/gamma/m
/:-P2#Adam/batch_normalization_499/beta/m
':%PP2Adam/dense_553/kernel/m
!:P2Adam/dense_553/bias/m
0:.P2$Adam/batch_normalization_500/gamma/m
/:-P2#Adam/batch_normalization_500/beta/m
':%Pa2Adam/dense_554/kernel/m
!:a2Adam/dense_554/bias/m
0:.a2$Adam/batch_normalization_501/gamma/m
/:-a2#Adam/batch_normalization_501/beta/m
':%aa2Adam/dense_555/kernel/m
!:a2Adam/dense_555/bias/m
0:.a2$Adam/batch_normalization_502/gamma/m
/:-a2#Adam/batch_normalization_502/beta/m
':%aa2Adam/dense_556/kernel/m
!:a2Adam/dense_556/bias/m
0:.a2$Adam/batch_normalization_503/gamma/m
/:-a2#Adam/batch_normalization_503/beta/m
':%aa2Adam/dense_557/kernel/m
!:a2Adam/dense_557/bias/m
0:.a2$Adam/batch_normalization_504/gamma/m
/:-a2#Adam/batch_normalization_504/beta/m
':%a=2Adam/dense_558/kernel/m
!:=2Adam/dense_558/bias/m
0:.=2$Adam/batch_normalization_505/gamma/m
/:-=2#Adam/batch_normalization_505/beta/m
':%==2Adam/dense_559/kernel/m
!:=2Adam/dense_559/bias/m
0:.=2$Adam/batch_normalization_506/gamma/m
/:-=2#Adam/batch_normalization_506/beta/m
':%==2Adam/dense_560/kernel/m
!:=2Adam/dense_560/bias/m
0:.=2$Adam/batch_normalization_507/gamma/m
/:-=2#Adam/batch_normalization_507/beta/m
':%=2Adam/dense_561/kernel/m
!:2Adam/dense_561/bias/m
':%P2Adam/dense_552/kernel/v
!:P2Adam/dense_552/bias/v
0:.P2$Adam/batch_normalization_499/gamma/v
/:-P2#Adam/batch_normalization_499/beta/v
':%PP2Adam/dense_553/kernel/v
!:P2Adam/dense_553/bias/v
0:.P2$Adam/batch_normalization_500/gamma/v
/:-P2#Adam/batch_normalization_500/beta/v
':%Pa2Adam/dense_554/kernel/v
!:a2Adam/dense_554/bias/v
0:.a2$Adam/batch_normalization_501/gamma/v
/:-a2#Adam/batch_normalization_501/beta/v
':%aa2Adam/dense_555/kernel/v
!:a2Adam/dense_555/bias/v
0:.a2$Adam/batch_normalization_502/gamma/v
/:-a2#Adam/batch_normalization_502/beta/v
':%aa2Adam/dense_556/kernel/v
!:a2Adam/dense_556/bias/v
0:.a2$Adam/batch_normalization_503/gamma/v
/:-a2#Adam/batch_normalization_503/beta/v
':%aa2Adam/dense_557/kernel/v
!:a2Adam/dense_557/bias/v
0:.a2$Adam/batch_normalization_504/gamma/v
/:-a2#Adam/batch_normalization_504/beta/v
':%a=2Adam/dense_558/kernel/v
!:=2Adam/dense_558/bias/v
0:.=2$Adam/batch_normalization_505/gamma/v
/:-=2#Adam/batch_normalization_505/beta/v
':%==2Adam/dense_559/kernel/v
!:=2Adam/dense_559/bias/v
0:.=2$Adam/batch_normalization_506/gamma/v
/:-=2#Adam/batch_normalization_506/beta/v
':%==2Adam/dense_560/kernel/v
!:=2Adam/dense_560/bias/v
0:.=2$Adam/batch_normalization_507/gamma/v
/:-=2#Adam/batch_normalization_507/beta/v
':%=2Adam/dense_561/kernel/v
!:2Adam/dense_561/bias/v
	J
Const
J	
Const_1
!__inference__wrapped_model_757143Ú`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøù?¢<
5¢2
0-
normalization_53_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_561# 
	dense_561ÿÿÿÿÿÿÿÿÿo
__inference_adapt_step_760279N-+,C¢@
9¢6
41¢
ÿÿÿÿÿÿÿÿÿ	IteratorSpec 
ª "
 ¹
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_760344b<9;:3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿP
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿP
 ¹
S__inference_batch_normalization_499_layer_call_and_return_conditional_losses_760378b;<9:3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿP
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿP
 
8__inference_batch_normalization_499_layer_call_fn_760311U<9;:3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿP
p 
ª "ÿÿÿÿÿÿÿÿÿP
8__inference_batch_normalization_499_layer_call_fn_760324U;<9:3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿP
p
ª "ÿÿÿÿÿÿÿÿÿP¹
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_760453bURTS3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿP
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿP
 ¹
S__inference_batch_normalization_500_layer_call_and_return_conditional_losses_760487bTURS3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿP
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿP
 
8__inference_batch_normalization_500_layer_call_fn_760420UURTS3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿP
p 
ª "ÿÿÿÿÿÿÿÿÿP
8__inference_batch_normalization_500_layer_call_fn_760433UTURS3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿP
p
ª "ÿÿÿÿÿÿÿÿÿP¹
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_760562bnkml3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 ¹
S__inference_batch_normalization_501_layer_call_and_return_conditional_losses_760596bmnkl3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
8__inference_batch_normalization_501_layer_call_fn_760529Unkml3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p 
ª "ÿÿÿÿÿÿÿÿÿa
8__inference_batch_normalization_501_layer_call_fn_760542Umnkl3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p
ª "ÿÿÿÿÿÿÿÿÿa½
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_760671f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 ½
S__inference_batch_normalization_502_layer_call_and_return_conditional_losses_760705f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
8__inference_batch_normalization_502_layer_call_fn_760638Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p 
ª "ÿÿÿÿÿÿÿÿÿa
8__inference_batch_normalization_502_layer_call_fn_760651Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p
ª "ÿÿÿÿÿÿÿÿÿa½
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_760780f 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 ½
S__inference_batch_normalization_503_layer_call_and_return_conditional_losses_760814f 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
8__inference_batch_normalization_503_layer_call_fn_760747Y 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p 
ª "ÿÿÿÿÿÿÿÿÿa
8__inference_batch_normalization_503_layer_call_fn_760760Y 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p
ª "ÿÿÿÿÿÿÿÿÿa½
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_760889f¹¶¸·3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 ½
S__inference_batch_normalization_504_layer_call_and_return_conditional_losses_760923f¸¹¶·3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
8__inference_batch_normalization_504_layer_call_fn_760856Y¹¶¸·3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p 
ª "ÿÿÿÿÿÿÿÿÿa
8__inference_batch_normalization_504_layer_call_fn_760869Y¸¹¶·3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p
ª "ÿÿÿÿÿÿÿÿÿa½
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_760998fÒÏÑÐ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 ½
S__inference_batch_normalization_505_layer_call_and_return_conditional_losses_761032fÑÒÏÐ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
8__inference_batch_normalization_505_layer_call_fn_760965YÒÏÑÐ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p 
ª "ÿÿÿÿÿÿÿÿÿ=
8__inference_batch_normalization_505_layer_call_fn_760978YÑÒÏÐ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p
ª "ÿÿÿÿÿÿÿÿÿ=½
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_761107fëèêé3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 ½
S__inference_batch_normalization_506_layer_call_and_return_conditional_losses_761141fêëèé3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
8__inference_batch_normalization_506_layer_call_fn_761074Yëèêé3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p 
ª "ÿÿÿÿÿÿÿÿÿ=
8__inference_batch_normalization_506_layer_call_fn_761087Yêëèé3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p
ª "ÿÿÿÿÿÿÿÿÿ=½
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_761216f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 ½
S__inference_batch_normalization_507_layer_call_and_return_conditional_losses_761250f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
8__inference_batch_normalization_507_layer_call_fn_761183Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p 
ª "ÿÿÿÿÿÿÿÿÿ=
8__inference_batch_normalization_507_layer_call_fn_761196Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p
ª "ÿÿÿÿÿÿÿÿÿ=¥
E__inference_dense_552_layer_call_and_return_conditional_losses_760298\01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿP
 }
*__inference_dense_552_layer_call_fn_760288O01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿP¥
E__inference_dense_553_layer_call_and_return_conditional_losses_760407\IJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿP
ª "%¢"

0ÿÿÿÿÿÿÿÿÿP
 }
*__inference_dense_553_layer_call_fn_760397OIJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿP
ª "ÿÿÿÿÿÿÿÿÿP¥
E__inference_dense_554_layer_call_and_return_conditional_losses_760516\bc/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿP
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 }
*__inference_dense_554_layer_call_fn_760506Obc/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿP
ª "ÿÿÿÿÿÿÿÿÿa¥
E__inference_dense_555_layer_call_and_return_conditional_losses_760625\{|/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 }
*__inference_dense_555_layer_call_fn_760615O{|/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "ÿÿÿÿÿÿÿÿÿa§
E__inference_dense_556_layer_call_and_return_conditional_losses_760734^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
*__inference_dense_556_layer_call_fn_760724Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "ÿÿÿÿÿÿÿÿÿa§
E__inference_dense_557_layer_call_and_return_conditional_losses_760843^­®/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
*__inference_dense_557_layer_call_fn_760833Q­®/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "ÿÿÿÿÿÿÿÿÿa§
E__inference_dense_558_layer_call_and_return_conditional_losses_760952^ÆÇ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
*__inference_dense_558_layer_call_fn_760942QÆÇ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "ÿÿÿÿÿÿÿÿÿ=§
E__inference_dense_559_layer_call_and_return_conditional_losses_761061^ßà/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
*__inference_dense_559_layer_call_fn_761051Qßà/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "ÿÿÿÿÿÿÿÿÿ=§
E__inference_dense_560_layer_call_and_return_conditional_losses_761170^øù/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
*__inference_dense_560_layer_call_fn_761160Qøù/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "ÿÿÿÿÿÿÿÿÿ=§
E__inference_dense_561_layer_call_and_return_conditional_losses_761279^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_561_layer_call_fn_761269Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_499_layer_call_and_return_conditional_losses_760388X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿP
ª "%¢"

0ÿÿÿÿÿÿÿÿÿP
 
0__inference_leaky_re_lu_499_layer_call_fn_760383K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿP
ª "ÿÿÿÿÿÿÿÿÿP§
K__inference_leaky_re_lu_500_layer_call_and_return_conditional_losses_760497X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿP
ª "%¢"

0ÿÿÿÿÿÿÿÿÿP
 
0__inference_leaky_re_lu_500_layer_call_fn_760492K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿP
ª "ÿÿÿÿÿÿÿÿÿP§
K__inference_leaky_re_lu_501_layer_call_and_return_conditional_losses_760606X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
0__inference_leaky_re_lu_501_layer_call_fn_760601K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "ÿÿÿÿÿÿÿÿÿa§
K__inference_leaky_re_lu_502_layer_call_and_return_conditional_losses_760715X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
0__inference_leaky_re_lu_502_layer_call_fn_760710K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "ÿÿÿÿÿÿÿÿÿa§
K__inference_leaky_re_lu_503_layer_call_and_return_conditional_losses_760824X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
0__inference_leaky_re_lu_503_layer_call_fn_760819K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "ÿÿÿÿÿÿÿÿÿa§
K__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_760933X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
0__inference_leaky_re_lu_504_layer_call_fn_760928K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "ÿÿÿÿÿÿÿÿÿa§
K__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_761042X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
0__inference_leaky_re_lu_505_layer_call_fn_761037K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "ÿÿÿÿÿÿÿÿÿ=§
K__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_761151X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
0__inference_leaky_re_lu_506_layer_call_fn_761146K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "ÿÿÿÿÿÿÿÿÿ=§
K__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_761260X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
0__inference_leaky_re_lu_507_layer_call_fn_761255K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "ÿÿÿÿÿÿÿÿÿ= 
I__inference_sequential_53_layer_call_and_return_conditional_losses_759138Ò`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøùG¢D
=¢:
0-
normalization_53_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
  
I__inference_sequential_53_layer_call_and_return_conditional_losses_759289Ò`01;<9:IJTURSbcmnkl{| ­®¸¹¶·ÆÇÑÒÏÐßàêëèéøùG¢D
=¢:
0-
normalization_53_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
I__inference_sequential_53_layer_call_and_return_conditional_losses_759759Â`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøù7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
I__inference_sequential_53_layer_call_and_return_conditional_losses_760109Â`01;<9:IJTURSbcmnkl{| ­®¸¹¶·ÆÇÑÒÏÐßàêëèéøù7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ø
.__inference_sequential_53_layer_call_fn_758319Å`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøùG¢D
=¢:
0-
normalization_53_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿø
.__inference_sequential_53_layer_call_fn_758987Å`01;<9:IJTURSbcmnkl{| ­®¸¹¶·ÆÇÑÒÏÐßàêëèéøùG¢D
=¢:
0-
normalization_53_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿè
.__inference_sequential_53_layer_call_fn_759414µ`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøù7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿè
.__inference_sequential_53_layer_call_fn_759535µ`01;<9:IJTURSbcmnkl{| ­®¸¹¶·ÆÇÑÒÏÐßàêëèéøù7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_signature_wrapper_760232ô`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøùY¢V
¢ 
OªL
J
normalization_53_input0-
normalization_53_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_561# 
	dense_561ÿÿÿÿÿÿÿÿÿ