·Ã5
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ò0
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
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
dense_619/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*!
shared_namedense_619/kernel
u
$dense_619/kernel/Read/ReadVariableOpReadVariableOpdense_619/kernel*
_output_shapes

:-*
dtype0
t
dense_619/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense_619/bias
m
"dense_619/bias/Read/ReadVariableOpReadVariableOpdense_619/bias*
_output_shapes
:-*
dtype0

batch_normalization_556/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*.
shared_namebatch_normalization_556/gamma

1batch_normalization_556/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_556/gamma*
_output_shapes
:-*
dtype0

batch_normalization_556/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*-
shared_namebatch_normalization_556/beta

0batch_normalization_556/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_556/beta*
_output_shapes
:-*
dtype0

#batch_normalization_556/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#batch_normalization_556/moving_mean

7batch_normalization_556/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_556/moving_mean*
_output_shapes
:-*
dtype0
¦
'batch_normalization_556/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*8
shared_name)'batch_normalization_556/moving_variance

;batch_normalization_556/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_556/moving_variance*
_output_shapes
:-*
dtype0
|
dense_620/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*!
shared_namedense_620/kernel
u
$dense_620/kernel/Read/ReadVariableOpReadVariableOpdense_620/kernel*
_output_shapes

:--*
dtype0
t
dense_620/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense_620/bias
m
"dense_620/bias/Read/ReadVariableOpReadVariableOpdense_620/bias*
_output_shapes
:-*
dtype0

batch_normalization_557/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*.
shared_namebatch_normalization_557/gamma

1batch_normalization_557/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_557/gamma*
_output_shapes
:-*
dtype0

batch_normalization_557/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*-
shared_namebatch_normalization_557/beta

0batch_normalization_557/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_557/beta*
_output_shapes
:-*
dtype0

#batch_normalization_557/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#batch_normalization_557/moving_mean

7batch_normalization_557/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_557/moving_mean*
_output_shapes
:-*
dtype0
¦
'batch_normalization_557/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*8
shared_name)'batch_normalization_557/moving_variance

;batch_normalization_557/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_557/moving_variance*
_output_shapes
:-*
dtype0
|
dense_621/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*!
shared_namedense_621/kernel
u
$dense_621/kernel/Read/ReadVariableOpReadVariableOpdense_621/kernel*
_output_shapes

:--*
dtype0
t
dense_621/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense_621/bias
m
"dense_621/bias/Read/ReadVariableOpReadVariableOpdense_621/bias*
_output_shapes
:-*
dtype0

batch_normalization_558/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*.
shared_namebatch_normalization_558/gamma

1batch_normalization_558/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_558/gamma*
_output_shapes
:-*
dtype0

batch_normalization_558/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*-
shared_namebatch_normalization_558/beta

0batch_normalization_558/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_558/beta*
_output_shapes
:-*
dtype0

#batch_normalization_558/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#batch_normalization_558/moving_mean

7batch_normalization_558/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_558/moving_mean*
_output_shapes
:-*
dtype0
¦
'batch_normalization_558/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*8
shared_name)'batch_normalization_558/moving_variance

;batch_normalization_558/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_558/moving_variance*
_output_shapes
:-*
dtype0
|
dense_622/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*!
shared_namedense_622/kernel
u
$dense_622/kernel/Read/ReadVariableOpReadVariableOpdense_622/kernel*
_output_shapes

:--*
dtype0
t
dense_622/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense_622/bias
m
"dense_622/bias/Read/ReadVariableOpReadVariableOpdense_622/bias*
_output_shapes
:-*
dtype0

batch_normalization_559/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*.
shared_namebatch_normalization_559/gamma

1batch_normalization_559/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_559/gamma*
_output_shapes
:-*
dtype0

batch_normalization_559/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*-
shared_namebatch_normalization_559/beta

0batch_normalization_559/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_559/beta*
_output_shapes
:-*
dtype0

#batch_normalization_559/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#batch_normalization_559/moving_mean

7batch_normalization_559/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_559/moving_mean*
_output_shapes
:-*
dtype0
¦
'batch_normalization_559/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*8
shared_name)'batch_normalization_559/moving_variance

;batch_normalization_559/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_559/moving_variance*
_output_shapes
:-*
dtype0
|
dense_623/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*!
shared_namedense_623/kernel
u
$dense_623/kernel/Read/ReadVariableOpReadVariableOpdense_623/kernel*
_output_shapes

:--*
dtype0
t
dense_623/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense_623/bias
m
"dense_623/bias/Read/ReadVariableOpReadVariableOpdense_623/bias*
_output_shapes
:-*
dtype0

batch_normalization_560/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*.
shared_namebatch_normalization_560/gamma

1batch_normalization_560/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_560/gamma*
_output_shapes
:-*
dtype0

batch_normalization_560/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*-
shared_namebatch_normalization_560/beta

0batch_normalization_560/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_560/beta*
_output_shapes
:-*
dtype0

#batch_normalization_560/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#batch_normalization_560/moving_mean

7batch_normalization_560/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_560/moving_mean*
_output_shapes
:-*
dtype0
¦
'batch_normalization_560/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*8
shared_name)'batch_normalization_560/moving_variance

;batch_normalization_560/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_560/moving_variance*
_output_shapes
:-*
dtype0
|
dense_624/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-l*!
shared_namedense_624/kernel
u
$dense_624/kernel/Read/ReadVariableOpReadVariableOpdense_624/kernel*
_output_shapes

:-l*
dtype0
t
dense_624/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*
shared_namedense_624/bias
m
"dense_624/bias/Read/ReadVariableOpReadVariableOpdense_624/bias*
_output_shapes
:l*
dtype0

batch_normalization_561/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*.
shared_namebatch_normalization_561/gamma

1batch_normalization_561/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_561/gamma*
_output_shapes
:l*
dtype0

batch_normalization_561/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*-
shared_namebatch_normalization_561/beta

0batch_normalization_561/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_561/beta*
_output_shapes
:l*
dtype0

#batch_normalization_561/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#batch_normalization_561/moving_mean

7batch_normalization_561/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_561/moving_mean*
_output_shapes
:l*
dtype0
¦
'batch_normalization_561/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*8
shared_name)'batch_normalization_561/moving_variance

;batch_normalization_561/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_561/moving_variance*
_output_shapes
:l*
dtype0
|
dense_625/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ll*!
shared_namedense_625/kernel
u
$dense_625/kernel/Read/ReadVariableOpReadVariableOpdense_625/kernel*
_output_shapes

:ll*
dtype0
t
dense_625/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*
shared_namedense_625/bias
m
"dense_625/bias/Read/ReadVariableOpReadVariableOpdense_625/bias*
_output_shapes
:l*
dtype0

batch_normalization_562/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*.
shared_namebatch_normalization_562/gamma

1batch_normalization_562/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_562/gamma*
_output_shapes
:l*
dtype0

batch_normalization_562/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*-
shared_namebatch_normalization_562/beta

0batch_normalization_562/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_562/beta*
_output_shapes
:l*
dtype0

#batch_normalization_562/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#batch_normalization_562/moving_mean

7batch_normalization_562/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_562/moving_mean*
_output_shapes
:l*
dtype0
¦
'batch_normalization_562/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*8
shared_name)'batch_normalization_562/moving_variance

;batch_normalization_562/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_562/moving_variance*
_output_shapes
:l*
dtype0
|
dense_626/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ll*!
shared_namedense_626/kernel
u
$dense_626/kernel/Read/ReadVariableOpReadVariableOpdense_626/kernel*
_output_shapes

:ll*
dtype0
t
dense_626/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*
shared_namedense_626/bias
m
"dense_626/bias/Read/ReadVariableOpReadVariableOpdense_626/bias*
_output_shapes
:l*
dtype0

batch_normalization_563/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*.
shared_namebatch_normalization_563/gamma

1batch_normalization_563/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_563/gamma*
_output_shapes
:l*
dtype0

batch_normalization_563/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*-
shared_namebatch_normalization_563/beta

0batch_normalization_563/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_563/beta*
_output_shapes
:l*
dtype0

#batch_normalization_563/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#batch_normalization_563/moving_mean

7batch_normalization_563/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_563/moving_mean*
_output_shapes
:l*
dtype0
¦
'batch_normalization_563/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*8
shared_name)'batch_normalization_563/moving_variance

;batch_normalization_563/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_563/moving_variance*
_output_shapes
:l*
dtype0
|
dense_627/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ll*!
shared_namedense_627/kernel
u
$dense_627/kernel/Read/ReadVariableOpReadVariableOpdense_627/kernel*
_output_shapes

:ll*
dtype0
t
dense_627/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*
shared_namedense_627/bias
m
"dense_627/bias/Read/ReadVariableOpReadVariableOpdense_627/bias*
_output_shapes
:l*
dtype0

batch_normalization_564/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*.
shared_namebatch_normalization_564/gamma

1batch_normalization_564/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_564/gamma*
_output_shapes
:l*
dtype0

batch_normalization_564/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*-
shared_namebatch_normalization_564/beta

0batch_normalization_564/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_564/beta*
_output_shapes
:l*
dtype0

#batch_normalization_564/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#batch_normalization_564/moving_mean

7batch_normalization_564/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_564/moving_mean*
_output_shapes
:l*
dtype0
¦
'batch_normalization_564/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*8
shared_name)'batch_normalization_564/moving_variance

;batch_normalization_564/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_564/moving_variance*
_output_shapes
:l*
dtype0
|
dense_628/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ll*!
shared_namedense_628/kernel
u
$dense_628/kernel/Read/ReadVariableOpReadVariableOpdense_628/kernel*
_output_shapes

:ll*
dtype0
t
dense_628/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*
shared_namedense_628/bias
m
"dense_628/bias/Read/ReadVariableOpReadVariableOpdense_628/bias*
_output_shapes
:l*
dtype0

batch_normalization_565/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*.
shared_namebatch_normalization_565/gamma

1batch_normalization_565/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_565/gamma*
_output_shapes
:l*
dtype0

batch_normalization_565/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*-
shared_namebatch_normalization_565/beta

0batch_normalization_565/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_565/beta*
_output_shapes
:l*
dtype0

#batch_normalization_565/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#batch_normalization_565/moving_mean

7batch_normalization_565/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_565/moving_mean*
_output_shapes
:l*
dtype0
¦
'batch_normalization_565/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*8
shared_name)'batch_normalization_565/moving_variance

;batch_normalization_565/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_565/moving_variance*
_output_shapes
:l*
dtype0
|
dense_629/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:lV*!
shared_namedense_629/kernel
u
$dense_629/kernel/Read/ReadVariableOpReadVariableOpdense_629/kernel*
_output_shapes

:lV*
dtype0
t
dense_629/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*
shared_namedense_629/bias
m
"dense_629/bias/Read/ReadVariableOpReadVariableOpdense_629/bias*
_output_shapes
:V*
dtype0

batch_normalization_566/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*.
shared_namebatch_normalization_566/gamma

1batch_normalization_566/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_566/gamma*
_output_shapes
:V*
dtype0

batch_normalization_566/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*-
shared_namebatch_normalization_566/beta

0batch_normalization_566/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_566/beta*
_output_shapes
:V*
dtype0

#batch_normalization_566/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*4
shared_name%#batch_normalization_566/moving_mean

7batch_normalization_566/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_566/moving_mean*
_output_shapes
:V*
dtype0
¦
'batch_normalization_566/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*8
shared_name)'batch_normalization_566/moving_variance

;batch_normalization_566/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_566/moving_variance*
_output_shapes
:V*
dtype0
|
dense_630/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:V*!
shared_namedense_630/kernel
u
$dense_630/kernel/Read/ReadVariableOpReadVariableOpdense_630/kernel*
_output_shapes

:V*
dtype0
t
dense_630/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_630/bias
m
"dense_630/bias/Read/ReadVariableOpReadVariableOpdense_630/bias*
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
Adam/dense_619/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*(
shared_nameAdam/dense_619/kernel/m

+Adam/dense_619/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_619/kernel/m*
_output_shapes

:-*
dtype0

Adam/dense_619/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameAdam/dense_619/bias/m
{
)Adam/dense_619/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_619/bias/m*
_output_shapes
:-*
dtype0
 
$Adam/batch_normalization_556/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*5
shared_name&$Adam/batch_normalization_556/gamma/m

8Adam/batch_normalization_556/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_556/gamma/m*
_output_shapes
:-*
dtype0

#Adam/batch_normalization_556/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#Adam/batch_normalization_556/beta/m

7Adam/batch_normalization_556/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_556/beta/m*
_output_shapes
:-*
dtype0

Adam/dense_620/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameAdam/dense_620/kernel/m

+Adam/dense_620/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_620/kernel/m*
_output_shapes

:--*
dtype0

Adam/dense_620/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameAdam/dense_620/bias/m
{
)Adam/dense_620/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_620/bias/m*
_output_shapes
:-*
dtype0
 
$Adam/batch_normalization_557/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*5
shared_name&$Adam/batch_normalization_557/gamma/m

8Adam/batch_normalization_557/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_557/gamma/m*
_output_shapes
:-*
dtype0

#Adam/batch_normalization_557/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#Adam/batch_normalization_557/beta/m

7Adam/batch_normalization_557/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_557/beta/m*
_output_shapes
:-*
dtype0

Adam/dense_621/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameAdam/dense_621/kernel/m

+Adam/dense_621/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_621/kernel/m*
_output_shapes

:--*
dtype0

Adam/dense_621/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameAdam/dense_621/bias/m
{
)Adam/dense_621/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_621/bias/m*
_output_shapes
:-*
dtype0
 
$Adam/batch_normalization_558/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*5
shared_name&$Adam/batch_normalization_558/gamma/m

8Adam/batch_normalization_558/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_558/gamma/m*
_output_shapes
:-*
dtype0

#Adam/batch_normalization_558/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#Adam/batch_normalization_558/beta/m

7Adam/batch_normalization_558/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_558/beta/m*
_output_shapes
:-*
dtype0

Adam/dense_622/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameAdam/dense_622/kernel/m

+Adam/dense_622/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_622/kernel/m*
_output_shapes

:--*
dtype0

Adam/dense_622/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameAdam/dense_622/bias/m
{
)Adam/dense_622/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_622/bias/m*
_output_shapes
:-*
dtype0
 
$Adam/batch_normalization_559/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*5
shared_name&$Adam/batch_normalization_559/gamma/m

8Adam/batch_normalization_559/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_559/gamma/m*
_output_shapes
:-*
dtype0

#Adam/batch_normalization_559/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#Adam/batch_normalization_559/beta/m

7Adam/batch_normalization_559/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_559/beta/m*
_output_shapes
:-*
dtype0

Adam/dense_623/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameAdam/dense_623/kernel/m

+Adam/dense_623/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_623/kernel/m*
_output_shapes

:--*
dtype0

Adam/dense_623/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameAdam/dense_623/bias/m
{
)Adam/dense_623/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_623/bias/m*
_output_shapes
:-*
dtype0
 
$Adam/batch_normalization_560/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*5
shared_name&$Adam/batch_normalization_560/gamma/m

8Adam/batch_normalization_560/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_560/gamma/m*
_output_shapes
:-*
dtype0

#Adam/batch_normalization_560/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#Adam/batch_normalization_560/beta/m

7Adam/batch_normalization_560/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_560/beta/m*
_output_shapes
:-*
dtype0

Adam/dense_624/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-l*(
shared_nameAdam/dense_624/kernel/m

+Adam/dense_624/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_624/kernel/m*
_output_shapes

:-l*
dtype0

Adam/dense_624/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*&
shared_nameAdam/dense_624/bias/m
{
)Adam/dense_624/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_624/bias/m*
_output_shapes
:l*
dtype0
 
$Adam/batch_normalization_561/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*5
shared_name&$Adam/batch_normalization_561/gamma/m

8Adam/batch_normalization_561/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_561/gamma/m*
_output_shapes
:l*
dtype0

#Adam/batch_normalization_561/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#Adam/batch_normalization_561/beta/m

7Adam/batch_normalization_561/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_561/beta/m*
_output_shapes
:l*
dtype0

Adam/dense_625/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ll*(
shared_nameAdam/dense_625/kernel/m

+Adam/dense_625/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_625/kernel/m*
_output_shapes

:ll*
dtype0

Adam/dense_625/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*&
shared_nameAdam/dense_625/bias/m
{
)Adam/dense_625/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_625/bias/m*
_output_shapes
:l*
dtype0
 
$Adam/batch_normalization_562/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*5
shared_name&$Adam/batch_normalization_562/gamma/m

8Adam/batch_normalization_562/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_562/gamma/m*
_output_shapes
:l*
dtype0

#Adam/batch_normalization_562/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#Adam/batch_normalization_562/beta/m

7Adam/batch_normalization_562/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_562/beta/m*
_output_shapes
:l*
dtype0

Adam/dense_626/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ll*(
shared_nameAdam/dense_626/kernel/m

+Adam/dense_626/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_626/kernel/m*
_output_shapes

:ll*
dtype0

Adam/dense_626/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*&
shared_nameAdam/dense_626/bias/m
{
)Adam/dense_626/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_626/bias/m*
_output_shapes
:l*
dtype0
 
$Adam/batch_normalization_563/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*5
shared_name&$Adam/batch_normalization_563/gamma/m

8Adam/batch_normalization_563/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_563/gamma/m*
_output_shapes
:l*
dtype0

#Adam/batch_normalization_563/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#Adam/batch_normalization_563/beta/m

7Adam/batch_normalization_563/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_563/beta/m*
_output_shapes
:l*
dtype0

Adam/dense_627/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ll*(
shared_nameAdam/dense_627/kernel/m

+Adam/dense_627/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_627/kernel/m*
_output_shapes

:ll*
dtype0

Adam/dense_627/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*&
shared_nameAdam/dense_627/bias/m
{
)Adam/dense_627/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_627/bias/m*
_output_shapes
:l*
dtype0
 
$Adam/batch_normalization_564/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*5
shared_name&$Adam/batch_normalization_564/gamma/m

8Adam/batch_normalization_564/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_564/gamma/m*
_output_shapes
:l*
dtype0

#Adam/batch_normalization_564/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#Adam/batch_normalization_564/beta/m

7Adam/batch_normalization_564/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_564/beta/m*
_output_shapes
:l*
dtype0

Adam/dense_628/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ll*(
shared_nameAdam/dense_628/kernel/m

+Adam/dense_628/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_628/kernel/m*
_output_shapes

:ll*
dtype0

Adam/dense_628/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*&
shared_nameAdam/dense_628/bias/m
{
)Adam/dense_628/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_628/bias/m*
_output_shapes
:l*
dtype0
 
$Adam/batch_normalization_565/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*5
shared_name&$Adam/batch_normalization_565/gamma/m

8Adam/batch_normalization_565/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_565/gamma/m*
_output_shapes
:l*
dtype0

#Adam/batch_normalization_565/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#Adam/batch_normalization_565/beta/m

7Adam/batch_normalization_565/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_565/beta/m*
_output_shapes
:l*
dtype0

Adam/dense_629/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:lV*(
shared_nameAdam/dense_629/kernel/m

+Adam/dense_629/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_629/kernel/m*
_output_shapes

:lV*
dtype0

Adam/dense_629/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*&
shared_nameAdam/dense_629/bias/m
{
)Adam/dense_629/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_629/bias/m*
_output_shapes
:V*
dtype0
 
$Adam/batch_normalization_566/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*5
shared_name&$Adam/batch_normalization_566/gamma/m

8Adam/batch_normalization_566/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_566/gamma/m*
_output_shapes
:V*
dtype0

#Adam/batch_normalization_566/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*4
shared_name%#Adam/batch_normalization_566/beta/m

7Adam/batch_normalization_566/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_566/beta/m*
_output_shapes
:V*
dtype0

Adam/dense_630/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:V*(
shared_nameAdam/dense_630/kernel/m

+Adam/dense_630/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_630/kernel/m*
_output_shapes

:V*
dtype0

Adam/dense_630/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_630/bias/m
{
)Adam/dense_630/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_630/bias/m*
_output_shapes
:*
dtype0

Adam/dense_619/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*(
shared_nameAdam/dense_619/kernel/v

+Adam/dense_619/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_619/kernel/v*
_output_shapes

:-*
dtype0

Adam/dense_619/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameAdam/dense_619/bias/v
{
)Adam/dense_619/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_619/bias/v*
_output_shapes
:-*
dtype0
 
$Adam/batch_normalization_556/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*5
shared_name&$Adam/batch_normalization_556/gamma/v

8Adam/batch_normalization_556/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_556/gamma/v*
_output_shapes
:-*
dtype0

#Adam/batch_normalization_556/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#Adam/batch_normalization_556/beta/v

7Adam/batch_normalization_556/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_556/beta/v*
_output_shapes
:-*
dtype0

Adam/dense_620/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameAdam/dense_620/kernel/v

+Adam/dense_620/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_620/kernel/v*
_output_shapes

:--*
dtype0

Adam/dense_620/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameAdam/dense_620/bias/v
{
)Adam/dense_620/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_620/bias/v*
_output_shapes
:-*
dtype0
 
$Adam/batch_normalization_557/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*5
shared_name&$Adam/batch_normalization_557/gamma/v

8Adam/batch_normalization_557/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_557/gamma/v*
_output_shapes
:-*
dtype0

#Adam/batch_normalization_557/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#Adam/batch_normalization_557/beta/v

7Adam/batch_normalization_557/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_557/beta/v*
_output_shapes
:-*
dtype0

Adam/dense_621/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameAdam/dense_621/kernel/v

+Adam/dense_621/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_621/kernel/v*
_output_shapes

:--*
dtype0

Adam/dense_621/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameAdam/dense_621/bias/v
{
)Adam/dense_621/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_621/bias/v*
_output_shapes
:-*
dtype0
 
$Adam/batch_normalization_558/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*5
shared_name&$Adam/batch_normalization_558/gamma/v

8Adam/batch_normalization_558/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_558/gamma/v*
_output_shapes
:-*
dtype0

#Adam/batch_normalization_558/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#Adam/batch_normalization_558/beta/v

7Adam/batch_normalization_558/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_558/beta/v*
_output_shapes
:-*
dtype0

Adam/dense_622/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameAdam/dense_622/kernel/v

+Adam/dense_622/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_622/kernel/v*
_output_shapes

:--*
dtype0

Adam/dense_622/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameAdam/dense_622/bias/v
{
)Adam/dense_622/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_622/bias/v*
_output_shapes
:-*
dtype0
 
$Adam/batch_normalization_559/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*5
shared_name&$Adam/batch_normalization_559/gamma/v

8Adam/batch_normalization_559/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_559/gamma/v*
_output_shapes
:-*
dtype0

#Adam/batch_normalization_559/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#Adam/batch_normalization_559/beta/v

7Adam/batch_normalization_559/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_559/beta/v*
_output_shapes
:-*
dtype0

Adam/dense_623/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameAdam/dense_623/kernel/v

+Adam/dense_623/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_623/kernel/v*
_output_shapes

:--*
dtype0

Adam/dense_623/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameAdam/dense_623/bias/v
{
)Adam/dense_623/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_623/bias/v*
_output_shapes
:-*
dtype0
 
$Adam/batch_normalization_560/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*5
shared_name&$Adam/batch_normalization_560/gamma/v

8Adam/batch_normalization_560/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_560/gamma/v*
_output_shapes
:-*
dtype0

#Adam/batch_normalization_560/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*4
shared_name%#Adam/batch_normalization_560/beta/v

7Adam/batch_normalization_560/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_560/beta/v*
_output_shapes
:-*
dtype0

Adam/dense_624/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-l*(
shared_nameAdam/dense_624/kernel/v

+Adam/dense_624/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_624/kernel/v*
_output_shapes

:-l*
dtype0

Adam/dense_624/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*&
shared_nameAdam/dense_624/bias/v
{
)Adam/dense_624/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_624/bias/v*
_output_shapes
:l*
dtype0
 
$Adam/batch_normalization_561/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*5
shared_name&$Adam/batch_normalization_561/gamma/v

8Adam/batch_normalization_561/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_561/gamma/v*
_output_shapes
:l*
dtype0

#Adam/batch_normalization_561/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#Adam/batch_normalization_561/beta/v

7Adam/batch_normalization_561/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_561/beta/v*
_output_shapes
:l*
dtype0

Adam/dense_625/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ll*(
shared_nameAdam/dense_625/kernel/v

+Adam/dense_625/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_625/kernel/v*
_output_shapes

:ll*
dtype0

Adam/dense_625/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*&
shared_nameAdam/dense_625/bias/v
{
)Adam/dense_625/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_625/bias/v*
_output_shapes
:l*
dtype0
 
$Adam/batch_normalization_562/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*5
shared_name&$Adam/batch_normalization_562/gamma/v

8Adam/batch_normalization_562/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_562/gamma/v*
_output_shapes
:l*
dtype0

#Adam/batch_normalization_562/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#Adam/batch_normalization_562/beta/v

7Adam/batch_normalization_562/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_562/beta/v*
_output_shapes
:l*
dtype0

Adam/dense_626/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ll*(
shared_nameAdam/dense_626/kernel/v

+Adam/dense_626/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_626/kernel/v*
_output_shapes

:ll*
dtype0

Adam/dense_626/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*&
shared_nameAdam/dense_626/bias/v
{
)Adam/dense_626/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_626/bias/v*
_output_shapes
:l*
dtype0
 
$Adam/batch_normalization_563/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*5
shared_name&$Adam/batch_normalization_563/gamma/v

8Adam/batch_normalization_563/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_563/gamma/v*
_output_shapes
:l*
dtype0

#Adam/batch_normalization_563/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#Adam/batch_normalization_563/beta/v

7Adam/batch_normalization_563/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_563/beta/v*
_output_shapes
:l*
dtype0

Adam/dense_627/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ll*(
shared_nameAdam/dense_627/kernel/v

+Adam/dense_627/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_627/kernel/v*
_output_shapes

:ll*
dtype0

Adam/dense_627/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*&
shared_nameAdam/dense_627/bias/v
{
)Adam/dense_627/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_627/bias/v*
_output_shapes
:l*
dtype0
 
$Adam/batch_normalization_564/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*5
shared_name&$Adam/batch_normalization_564/gamma/v

8Adam/batch_normalization_564/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_564/gamma/v*
_output_shapes
:l*
dtype0

#Adam/batch_normalization_564/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#Adam/batch_normalization_564/beta/v

7Adam/batch_normalization_564/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_564/beta/v*
_output_shapes
:l*
dtype0

Adam/dense_628/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ll*(
shared_nameAdam/dense_628/kernel/v

+Adam/dense_628/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_628/kernel/v*
_output_shapes

:ll*
dtype0

Adam/dense_628/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*&
shared_nameAdam/dense_628/bias/v
{
)Adam/dense_628/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_628/bias/v*
_output_shapes
:l*
dtype0
 
$Adam/batch_normalization_565/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*5
shared_name&$Adam/batch_normalization_565/gamma/v

8Adam/batch_normalization_565/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_565/gamma/v*
_output_shapes
:l*
dtype0

#Adam/batch_normalization_565/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:l*4
shared_name%#Adam/batch_normalization_565/beta/v

7Adam/batch_normalization_565/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_565/beta/v*
_output_shapes
:l*
dtype0

Adam/dense_629/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:lV*(
shared_nameAdam/dense_629/kernel/v

+Adam/dense_629/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_629/kernel/v*
_output_shapes

:lV*
dtype0

Adam/dense_629/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*&
shared_nameAdam/dense_629/bias/v
{
)Adam/dense_629/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_629/bias/v*
_output_shapes
:V*
dtype0
 
$Adam/batch_normalization_566/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*5
shared_name&$Adam/batch_normalization_566/gamma/v

8Adam/batch_normalization_566/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_566/gamma/v*
_output_shapes
:V*
dtype0

#Adam/batch_normalization_566/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*4
shared_name%#Adam/batch_normalization_566/beta/v

7Adam/batch_normalization_566/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_566/beta/v*
_output_shapes
:V*
dtype0

Adam/dense_630/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:V*(
shared_nameAdam/dense_630/kernel/v

+Adam/dense_630/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_630/kernel/v*
_output_shapes

:V*
dtype0

Adam/dense_630/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_630/bias/v
{
)Adam/dense_630/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_630/bias/v*
_output_shapes
:*
dtype0
n
ConstConst*
_output_shapes

:*
dtype0*1
value(B&"XUéBgföAeföA DA DA5>
p
Const_1Const*
_output_shapes

:*
dtype0*1
value(B&"4sE	×HD×HDÿ¿BÀB!=

NoOpNoOp
Ú
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*ºÙ
value¯ÙB«Ù B£Ù
ª

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
layer_with_weights-20
layer-29
layer-30
 layer_with_weights-21
 layer-31
!layer_with_weights-22
!layer-32
"layer-33
#layer_with_weights-23
#layer-34
$	optimizer
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_default_save_signature
,
signatures*
¾
-
_keep_axis
._reduce_axis
/_reduce_axis_mask
0_broadcast_shape
1mean
1
adapt_mean
2variance
2adapt_variance
	3count
4	keras_api
5_adapt_function*
¦

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses*
Õ
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses*

I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
¦

Okernel
Pbias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses*
Õ
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses*

b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses* 
¦

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses*
Õ
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses*

{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+¡&call_and_return_all_conditional_losses*
à
	¢axis

£gamma
	¤beta
¥moving_mean
¦moving_variance
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses*

­	variables
®trainable_variables
¯regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses* 
®
³kernel
	´bias
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api
¹__call__
+º&call_and_return_all_conditional_losses*
à
	»axis

¼gamma
	½beta
¾moving_mean
¿moving_variance
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses*

Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses* 
®
Ìkernel
	Íbias
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses*
à
	Ôaxis

Õgamma
	Öbeta
×moving_mean
Ømoving_variance
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses*

ß	variables
àtrainable_variables
áregularization_losses
â	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses* 
®
åkernel
	æbias
ç	variables
ètrainable_variables
éregularization_losses
ê	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses*
à
	íaxis

îgamma
	ïbeta
ðmoving_mean
ñmoving_variance
ò	variables
ótrainable_variables
ôregularization_losses
õ	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses*

ø	variables
ùtrainable_variables
úregularization_losses
û	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses* 
®
þkernel
	ÿbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

 gamma
	¡beta
¢moving_mean
£moving_variance
¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses*

ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses* 
®
°kernel
	±bias
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses*
à
	¸axis

¹gamma
	ºbeta
»moving_mean
¼moving_variance
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses*

Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses* 
®
Ékernel
	Êbias
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses*

	Ñiter
Òbeta_1
Óbeta_2

Ôdecay6m7m?m@mOmPmXmYmhmimqmrm	m	m	m	m	m	m	£m	¤m	³m	´m	¼m 	½m¡	Ìm¢	Ím£	Õm¤	Öm¥	åm¦	æm§	îm¨	ïm©	þmª	ÿm«	m¬	m­	m®	m¯	 m°	¡m±	°m²	±m³	¹m´	ºmµ	Ém¶	Êm·6v¸7v¹?vº@v»Ov¼Pv½Xv¾Yv¿hvÀivÁqvÂrvÃ	vÄ	vÅ	vÆ	vÇ	vÈ	vÉ	£vÊ	¤vË	³vÌ	´vÍ	¼vÎ	½vÏ	ÌvÐ	ÍvÑ	ÕvÒ	ÖvÓ	åvÔ	ævÕ	îvÖ	ïv×	þvØ	ÿvÙ	vÚ	vÛ	vÜ	vÝ	 vÞ	¡vß	°và	±vá	¹vâ	ºvã	Évä	Êvå*
ä
10
21
32
63
74
?5
@6
A7
B8
O9
P10
X11
Y12
Z13
[14
h15
i16
q17
r18
s19
t20
21
22
23
24
25
26
27
28
£29
¤30
¥31
¦32
³33
´34
¼35
½36
¾37
¿38
Ì39
Í40
Õ41
Ö42
×43
Ø44
å45
æ46
î47
ï48
ð49
ñ50
þ51
ÿ52
53
54
55
56
57
58
 59
¡60
¢61
£62
°63
±64
¹65
º66
»67
¼68
É69
Ê70*

60
71
?2
@3
O4
P5
X6
Y7
h8
i9
q10
r11
12
13
14
15
16
17
£18
¤19
³20
´21
¼22
½23
Ì24
Í25
Õ26
Ö27
å28
æ29
î30
ï31
þ32
ÿ33
34
35
36
37
 38
¡39
°40
±41
¹42
º43
É44
Ê45*
* 
µ
Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
+_default_save_signature
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*
* 
* 
* 

Úserving_default* 
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
VARIABLE_VALUEdense_619/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_619/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

60
71*

60
71*
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_556/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_556/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_556/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_556/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
?0
@1
A2
B3*

?0
@1*
* 

ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_620/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_620/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

O0
P1*

O0
P1*
* 

ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_557/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_557/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_557/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_557/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
X0
Y1
Z2
[3*

X0
Y1*
* 

ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_621/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_621/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

h0
i1*

h0
i1*
* 

ùnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_558/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_558/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_558/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_558/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
q0
r1
s2
t3*

q0
r1*
* 

þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_622/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_622/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_559/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_559/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_559/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_559/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_623/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_623/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_560/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_560/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_560/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_560/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
£0
¤1
¥2
¦3*

£0
¤1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
­	variables
®trainable_variables
¯regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_624/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_624/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

³0
´1*

³0
´1*
* 

¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
µ	variables
¶trainable_variables
·regularization_losses
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_561/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_561/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_561/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_561/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
¼0
½1
¾2
¿3*

¼0
½1*
* 

«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
Æ	variables
Çtrainable_variables
Èregularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_625/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_625/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ì0
Í1*

Ì0
Í1*
* 

µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_562/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_562/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_562/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_562/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
Õ0
Ö1
×2
Ø3*

Õ0
Ö1*
* 

ºnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
Ù	variables
Útrainable_variables
Ûregularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
ß	variables
àtrainable_variables
áregularization_losses
ã__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_626/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_626/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

å0
æ1*

å0
æ1*
* 

Änon_trainable_variables
Ålayers
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
ç	variables
ètrainable_variables
éregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_563/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_563/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_563/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_563/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
î0
ï1
ð2
ñ3*

î0
ï1*
* 

Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
ò	variables
ótrainable_variables
ôregularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Înon_trainable_variables
Ïlayers
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
ø	variables
ùtrainable_variables
úregularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_627/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_627/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*

þ0
ÿ1*

þ0
ÿ1*
* 

Ónon_trainable_variables
Ôlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_564/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_564/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_564/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_564/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

Ønon_trainable_variables
Ùlayers
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_628/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_628/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_565/gamma6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_565/beta5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_565/moving_mean<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_565/moving_variance@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
 0
¡1
¢2
£3*

 0
¡1*
* 

çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_629/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_629/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*

°0
±1*

°0
±1*
* 

ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_566/gamma6layer_with_weights-22/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_566/beta5layer_with_weights-22/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_566/moving_mean<layer_with_weights-22/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_566/moving_variance@layer_with_weights-22/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
¹0
º1
»2
¼3*

¹0
º1*
* 

önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_630/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_630/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE*

É0
Ê1*

É0
Ê1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses*
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
Ò
10
21
32
A3
B4
Z5
[6
s7
t8
9
10
¥11
¦12
¾13
¿14
×15
Ø16
ð17
ñ18
19
20
¢21
£22
»23
¼24*

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
28
29
30
 31
!32
"33
#34*

0*
* 
* 
* 
* 
* 
* 
* 
* 

A0
B1*
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
Z0
[1*
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
s0
t1*
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
0
1*
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
¥0
¦1*
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
¾0
¿1*
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
×0
Ø1*
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
ð0
ñ1*
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
0
1*
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
¢0
£1*
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
»0
¼1*
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

total

count
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
}
VARIABLE_VALUEAdam/dense_619/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_619/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_556/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_556/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_620/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_620/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_557/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_557/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_621/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_621/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_558/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_558/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_622/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_622/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_559/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_559/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_623/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_623/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_560/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_560/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_624/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_624/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_561/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_561/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_625/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_625/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_562/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_562/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_626/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_626/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_563/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_563/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_627/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_627/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_564/gamma/mRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_564/beta/mQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_628/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_628/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_565/gamma/mRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_565/beta/mQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_629/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_629/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_566/gamma/mRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_566/beta/mQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_630/kernel/mSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_630/bias/mQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_619/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_619/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_556/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_556/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_620/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_620/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_557/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_557/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_621/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_621/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_558/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_558/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_622/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_622/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_559/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_559/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_623/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_623/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_560/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_560/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_624/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_624/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_561/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_561/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_625/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_625/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_562/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_562/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_626/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_626/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_563/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_563/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_627/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_627/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_564/gamma/vRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_564/beta/vQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_628/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_628/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_565/gamma/vRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_565/beta/vQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_629/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_629/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_566/gamma/vRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_566/beta/vQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_630/kernel/vSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_630/bias/vQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

&serving_default_normalization_63_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ì
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_63_inputConstConst_1dense_619/kerneldense_619/bias'batch_normalization_556/moving_variancebatch_normalization_556/gamma#batch_normalization_556/moving_meanbatch_normalization_556/betadense_620/kerneldense_620/bias'batch_normalization_557/moving_variancebatch_normalization_557/gamma#batch_normalization_557/moving_meanbatch_normalization_557/betadense_621/kerneldense_621/bias'batch_normalization_558/moving_variancebatch_normalization_558/gamma#batch_normalization_558/moving_meanbatch_normalization_558/betadense_622/kerneldense_622/bias'batch_normalization_559/moving_variancebatch_normalization_559/gamma#batch_normalization_559/moving_meanbatch_normalization_559/betadense_623/kerneldense_623/bias'batch_normalization_560/moving_variancebatch_normalization_560/gamma#batch_normalization_560/moving_meanbatch_normalization_560/betadense_624/kerneldense_624/bias'batch_normalization_561/moving_variancebatch_normalization_561/gamma#batch_normalization_561/moving_meanbatch_normalization_561/betadense_625/kerneldense_625/bias'batch_normalization_562/moving_variancebatch_normalization_562/gamma#batch_normalization_562/moving_meanbatch_normalization_562/betadense_626/kerneldense_626/bias'batch_normalization_563/moving_variancebatch_normalization_563/gamma#batch_normalization_563/moving_meanbatch_normalization_563/betadense_627/kerneldense_627/bias'batch_normalization_564/moving_variancebatch_normalization_564/gamma#batch_normalization_564/moving_meanbatch_normalization_564/betadense_628/kerneldense_628/bias'batch_normalization_565/moving_variancebatch_normalization_565/gamma#batch_normalization_565/moving_meanbatch_normalization_565/betadense_629/kerneldense_629/bias'batch_normalization_566/moving_variancebatch_normalization_566/gamma#batch_normalization_566/moving_meanbatch_normalization_566/betadense_630/kerneldense_630/bias*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEF*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_833903
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ÙC
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_619/kernel/Read/ReadVariableOp"dense_619/bias/Read/ReadVariableOp1batch_normalization_556/gamma/Read/ReadVariableOp0batch_normalization_556/beta/Read/ReadVariableOp7batch_normalization_556/moving_mean/Read/ReadVariableOp;batch_normalization_556/moving_variance/Read/ReadVariableOp$dense_620/kernel/Read/ReadVariableOp"dense_620/bias/Read/ReadVariableOp1batch_normalization_557/gamma/Read/ReadVariableOp0batch_normalization_557/beta/Read/ReadVariableOp7batch_normalization_557/moving_mean/Read/ReadVariableOp;batch_normalization_557/moving_variance/Read/ReadVariableOp$dense_621/kernel/Read/ReadVariableOp"dense_621/bias/Read/ReadVariableOp1batch_normalization_558/gamma/Read/ReadVariableOp0batch_normalization_558/beta/Read/ReadVariableOp7batch_normalization_558/moving_mean/Read/ReadVariableOp;batch_normalization_558/moving_variance/Read/ReadVariableOp$dense_622/kernel/Read/ReadVariableOp"dense_622/bias/Read/ReadVariableOp1batch_normalization_559/gamma/Read/ReadVariableOp0batch_normalization_559/beta/Read/ReadVariableOp7batch_normalization_559/moving_mean/Read/ReadVariableOp;batch_normalization_559/moving_variance/Read/ReadVariableOp$dense_623/kernel/Read/ReadVariableOp"dense_623/bias/Read/ReadVariableOp1batch_normalization_560/gamma/Read/ReadVariableOp0batch_normalization_560/beta/Read/ReadVariableOp7batch_normalization_560/moving_mean/Read/ReadVariableOp;batch_normalization_560/moving_variance/Read/ReadVariableOp$dense_624/kernel/Read/ReadVariableOp"dense_624/bias/Read/ReadVariableOp1batch_normalization_561/gamma/Read/ReadVariableOp0batch_normalization_561/beta/Read/ReadVariableOp7batch_normalization_561/moving_mean/Read/ReadVariableOp;batch_normalization_561/moving_variance/Read/ReadVariableOp$dense_625/kernel/Read/ReadVariableOp"dense_625/bias/Read/ReadVariableOp1batch_normalization_562/gamma/Read/ReadVariableOp0batch_normalization_562/beta/Read/ReadVariableOp7batch_normalization_562/moving_mean/Read/ReadVariableOp;batch_normalization_562/moving_variance/Read/ReadVariableOp$dense_626/kernel/Read/ReadVariableOp"dense_626/bias/Read/ReadVariableOp1batch_normalization_563/gamma/Read/ReadVariableOp0batch_normalization_563/beta/Read/ReadVariableOp7batch_normalization_563/moving_mean/Read/ReadVariableOp;batch_normalization_563/moving_variance/Read/ReadVariableOp$dense_627/kernel/Read/ReadVariableOp"dense_627/bias/Read/ReadVariableOp1batch_normalization_564/gamma/Read/ReadVariableOp0batch_normalization_564/beta/Read/ReadVariableOp7batch_normalization_564/moving_mean/Read/ReadVariableOp;batch_normalization_564/moving_variance/Read/ReadVariableOp$dense_628/kernel/Read/ReadVariableOp"dense_628/bias/Read/ReadVariableOp1batch_normalization_565/gamma/Read/ReadVariableOp0batch_normalization_565/beta/Read/ReadVariableOp7batch_normalization_565/moving_mean/Read/ReadVariableOp;batch_normalization_565/moving_variance/Read/ReadVariableOp$dense_629/kernel/Read/ReadVariableOp"dense_629/bias/Read/ReadVariableOp1batch_normalization_566/gamma/Read/ReadVariableOp0batch_normalization_566/beta/Read/ReadVariableOp7batch_normalization_566/moving_mean/Read/ReadVariableOp;batch_normalization_566/moving_variance/Read/ReadVariableOp$dense_630/kernel/Read/ReadVariableOp"dense_630/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_619/kernel/m/Read/ReadVariableOp)Adam/dense_619/bias/m/Read/ReadVariableOp8Adam/batch_normalization_556/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_556/beta/m/Read/ReadVariableOp+Adam/dense_620/kernel/m/Read/ReadVariableOp)Adam/dense_620/bias/m/Read/ReadVariableOp8Adam/batch_normalization_557/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_557/beta/m/Read/ReadVariableOp+Adam/dense_621/kernel/m/Read/ReadVariableOp)Adam/dense_621/bias/m/Read/ReadVariableOp8Adam/batch_normalization_558/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_558/beta/m/Read/ReadVariableOp+Adam/dense_622/kernel/m/Read/ReadVariableOp)Adam/dense_622/bias/m/Read/ReadVariableOp8Adam/batch_normalization_559/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_559/beta/m/Read/ReadVariableOp+Adam/dense_623/kernel/m/Read/ReadVariableOp)Adam/dense_623/bias/m/Read/ReadVariableOp8Adam/batch_normalization_560/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_560/beta/m/Read/ReadVariableOp+Adam/dense_624/kernel/m/Read/ReadVariableOp)Adam/dense_624/bias/m/Read/ReadVariableOp8Adam/batch_normalization_561/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_561/beta/m/Read/ReadVariableOp+Adam/dense_625/kernel/m/Read/ReadVariableOp)Adam/dense_625/bias/m/Read/ReadVariableOp8Adam/batch_normalization_562/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_562/beta/m/Read/ReadVariableOp+Adam/dense_626/kernel/m/Read/ReadVariableOp)Adam/dense_626/bias/m/Read/ReadVariableOp8Adam/batch_normalization_563/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_563/beta/m/Read/ReadVariableOp+Adam/dense_627/kernel/m/Read/ReadVariableOp)Adam/dense_627/bias/m/Read/ReadVariableOp8Adam/batch_normalization_564/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_564/beta/m/Read/ReadVariableOp+Adam/dense_628/kernel/m/Read/ReadVariableOp)Adam/dense_628/bias/m/Read/ReadVariableOp8Adam/batch_normalization_565/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_565/beta/m/Read/ReadVariableOp+Adam/dense_629/kernel/m/Read/ReadVariableOp)Adam/dense_629/bias/m/Read/ReadVariableOp8Adam/batch_normalization_566/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_566/beta/m/Read/ReadVariableOp+Adam/dense_630/kernel/m/Read/ReadVariableOp)Adam/dense_630/bias/m/Read/ReadVariableOp+Adam/dense_619/kernel/v/Read/ReadVariableOp)Adam/dense_619/bias/v/Read/ReadVariableOp8Adam/batch_normalization_556/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_556/beta/v/Read/ReadVariableOp+Adam/dense_620/kernel/v/Read/ReadVariableOp)Adam/dense_620/bias/v/Read/ReadVariableOp8Adam/batch_normalization_557/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_557/beta/v/Read/ReadVariableOp+Adam/dense_621/kernel/v/Read/ReadVariableOp)Adam/dense_621/bias/v/Read/ReadVariableOp8Adam/batch_normalization_558/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_558/beta/v/Read/ReadVariableOp+Adam/dense_622/kernel/v/Read/ReadVariableOp)Adam/dense_622/bias/v/Read/ReadVariableOp8Adam/batch_normalization_559/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_559/beta/v/Read/ReadVariableOp+Adam/dense_623/kernel/v/Read/ReadVariableOp)Adam/dense_623/bias/v/Read/ReadVariableOp8Adam/batch_normalization_560/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_560/beta/v/Read/ReadVariableOp+Adam/dense_624/kernel/v/Read/ReadVariableOp)Adam/dense_624/bias/v/Read/ReadVariableOp8Adam/batch_normalization_561/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_561/beta/v/Read/ReadVariableOp+Adam/dense_625/kernel/v/Read/ReadVariableOp)Adam/dense_625/bias/v/Read/ReadVariableOp8Adam/batch_normalization_562/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_562/beta/v/Read/ReadVariableOp+Adam/dense_626/kernel/v/Read/ReadVariableOp)Adam/dense_626/bias/v/Read/ReadVariableOp8Adam/batch_normalization_563/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_563/beta/v/Read/ReadVariableOp+Adam/dense_627/kernel/v/Read/ReadVariableOp)Adam/dense_627/bias/v/Read/ReadVariableOp8Adam/batch_normalization_564/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_564/beta/v/Read/ReadVariableOp+Adam/dense_628/kernel/v/Read/ReadVariableOp)Adam/dense_628/bias/v/Read/ReadVariableOp8Adam/batch_normalization_565/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_565/beta/v/Read/ReadVariableOp+Adam/dense_629/kernel/v/Read/ReadVariableOp)Adam/dense_629/bias/v/Read/ReadVariableOp8Adam/batch_normalization_566/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_566/beta/v/Read/ReadVariableOp+Adam/dense_630/kernel/v/Read/ReadVariableOp)Adam/dense_630/bias/v/Read/ReadVariableOpConst_2*¹
Tin±
®2«		*
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
__inference__traced_save_835700
)
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_619/kerneldense_619/biasbatch_normalization_556/gammabatch_normalization_556/beta#batch_normalization_556/moving_mean'batch_normalization_556/moving_variancedense_620/kerneldense_620/biasbatch_normalization_557/gammabatch_normalization_557/beta#batch_normalization_557/moving_mean'batch_normalization_557/moving_variancedense_621/kerneldense_621/biasbatch_normalization_558/gammabatch_normalization_558/beta#batch_normalization_558/moving_mean'batch_normalization_558/moving_variancedense_622/kerneldense_622/biasbatch_normalization_559/gammabatch_normalization_559/beta#batch_normalization_559/moving_mean'batch_normalization_559/moving_variancedense_623/kerneldense_623/biasbatch_normalization_560/gammabatch_normalization_560/beta#batch_normalization_560/moving_mean'batch_normalization_560/moving_variancedense_624/kerneldense_624/biasbatch_normalization_561/gammabatch_normalization_561/beta#batch_normalization_561/moving_mean'batch_normalization_561/moving_variancedense_625/kerneldense_625/biasbatch_normalization_562/gammabatch_normalization_562/beta#batch_normalization_562/moving_mean'batch_normalization_562/moving_variancedense_626/kerneldense_626/biasbatch_normalization_563/gammabatch_normalization_563/beta#batch_normalization_563/moving_mean'batch_normalization_563/moving_variancedense_627/kerneldense_627/biasbatch_normalization_564/gammabatch_normalization_564/beta#batch_normalization_564/moving_mean'batch_normalization_564/moving_variancedense_628/kerneldense_628/biasbatch_normalization_565/gammabatch_normalization_565/beta#batch_normalization_565/moving_mean'batch_normalization_565/moving_variancedense_629/kerneldense_629/biasbatch_normalization_566/gammabatch_normalization_566/beta#batch_normalization_566/moving_mean'batch_normalization_566/moving_variancedense_630/kerneldense_630/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_619/kernel/mAdam/dense_619/bias/m$Adam/batch_normalization_556/gamma/m#Adam/batch_normalization_556/beta/mAdam/dense_620/kernel/mAdam/dense_620/bias/m$Adam/batch_normalization_557/gamma/m#Adam/batch_normalization_557/beta/mAdam/dense_621/kernel/mAdam/dense_621/bias/m$Adam/batch_normalization_558/gamma/m#Adam/batch_normalization_558/beta/mAdam/dense_622/kernel/mAdam/dense_622/bias/m$Adam/batch_normalization_559/gamma/m#Adam/batch_normalization_559/beta/mAdam/dense_623/kernel/mAdam/dense_623/bias/m$Adam/batch_normalization_560/gamma/m#Adam/batch_normalization_560/beta/mAdam/dense_624/kernel/mAdam/dense_624/bias/m$Adam/batch_normalization_561/gamma/m#Adam/batch_normalization_561/beta/mAdam/dense_625/kernel/mAdam/dense_625/bias/m$Adam/batch_normalization_562/gamma/m#Adam/batch_normalization_562/beta/mAdam/dense_626/kernel/mAdam/dense_626/bias/m$Adam/batch_normalization_563/gamma/m#Adam/batch_normalization_563/beta/mAdam/dense_627/kernel/mAdam/dense_627/bias/m$Adam/batch_normalization_564/gamma/m#Adam/batch_normalization_564/beta/mAdam/dense_628/kernel/mAdam/dense_628/bias/m$Adam/batch_normalization_565/gamma/m#Adam/batch_normalization_565/beta/mAdam/dense_629/kernel/mAdam/dense_629/bias/m$Adam/batch_normalization_566/gamma/m#Adam/batch_normalization_566/beta/mAdam/dense_630/kernel/mAdam/dense_630/bias/mAdam/dense_619/kernel/vAdam/dense_619/bias/v$Adam/batch_normalization_556/gamma/v#Adam/batch_normalization_556/beta/vAdam/dense_620/kernel/vAdam/dense_620/bias/v$Adam/batch_normalization_557/gamma/v#Adam/batch_normalization_557/beta/vAdam/dense_621/kernel/vAdam/dense_621/bias/v$Adam/batch_normalization_558/gamma/v#Adam/batch_normalization_558/beta/vAdam/dense_622/kernel/vAdam/dense_622/bias/v$Adam/batch_normalization_559/gamma/v#Adam/batch_normalization_559/beta/vAdam/dense_623/kernel/vAdam/dense_623/bias/v$Adam/batch_normalization_560/gamma/v#Adam/batch_normalization_560/beta/vAdam/dense_624/kernel/vAdam/dense_624/bias/v$Adam/batch_normalization_561/gamma/v#Adam/batch_normalization_561/beta/vAdam/dense_625/kernel/vAdam/dense_625/bias/v$Adam/batch_normalization_562/gamma/v#Adam/batch_normalization_562/beta/vAdam/dense_626/kernel/vAdam/dense_626/bias/v$Adam/batch_normalization_563/gamma/v#Adam/batch_normalization_563/beta/vAdam/dense_627/kernel/vAdam/dense_627/bias/v$Adam/batch_normalization_564/gamma/v#Adam/batch_normalization_564/beta/vAdam/dense_628/kernel/vAdam/dense_628/bias/v$Adam/batch_normalization_565/gamma/v#Adam/batch_normalization_565/beta/vAdam/dense_629/kernel/vAdam/dense_629/bias/v$Adam/batch_normalization_566/gamma/v#Adam/batch_normalization_566/beta/vAdam/dense_630/kernel/vAdam/dense_630/bias/v*¸
Tin°
­2ª*
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
"__inference__traced_restore_836217*
È	
ö
E__inference_dense_620_layer_call_and_return_conditional_losses_831134

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ä

*__inference_dense_625_layer_call_fn_834613

inputs
unknown:ll
	unknown_0:l
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_625_layer_call_and_return_conditional_losses_831294o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_834921

inputs5
'assignmovingavg_readvariableop_resource:l7
)assignmovingavg_1_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l/
!batchnorm_readvariableop_resource:l
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:l
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿll
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:l*
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
:l*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:lx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l¬
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
:l*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:l~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l´
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:lv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
È	
ö
E__inference_dense_629_layer_call_and_return_conditional_losses_835059

inputs0
matmul_readvariableop_resource:lV-
biasadd_readvariableop_resource:V
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:lV*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:V*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_835105

inputs/
!batchnorm_readvariableop_resource:V3
%batchnorm_mul_readvariableop_resource:V1
#batchnorm_readvariableop_1_resource:V1
#batchnorm_readvariableop_2_resource:V
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:V*
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
:VP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:V~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:V*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Vc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:V*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Vz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:V*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Vr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿV: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_830528

inputs/
!batchnorm_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-1
#batchnorm_readvariableop_1_resource:-1
#batchnorm_readvariableop_2_resource:-
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:-z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_561_layer_call_fn_834540

inputs
unknown:l
	unknown_0:l
	unknown_1:l
	unknown_2:l
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_830657o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_830329

inputs5
'assignmovingavg_readvariableop_resource:-7
)assignmovingavg_1_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-/
!batchnorm_readvariableop_resource:-
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:-
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:-*
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
:-*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:-x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-¬
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
:-*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:-~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-´
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:-v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_558_layer_call_fn_834272

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
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_558_layer_call_and_return_conditional_losses_831186`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_563_layer_call_fn_834758

inputs
unknown:l
	unknown_0:l
	unknown_1:l
	unknown_2:l
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_830821o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_834996

inputs/
!batchnorm_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l1
#batchnorm_readvariableop_1_resource:l1
#batchnorm_readvariableop_2_resource:l
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:lz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_566_layer_call_and_return_conditional_losses_835149

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿV:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_830575

inputs5
'assignmovingavg_readvariableop_resource:-7
)assignmovingavg_1_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-/
!batchnorm_readvariableop_resource:-
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:-
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:-*
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
:-*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:-x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-¬
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
:-*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:-~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-´
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:-v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_563_layer_call_fn_834745

inputs
unknown:l
	unknown_0:l
	unknown_1:l
	unknown_2:l
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_830774o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_830493

inputs5
'assignmovingavg_readvariableop_resource:-7
)assignmovingavg_1_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-/
!batchnorm_readvariableop_resource:-
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:-
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:-*
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
:-*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:-x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-¬
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
:-*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:-~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-´
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:-v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Á

.__inference_sequential_63_layer_call_fn_833062

inputs
unknown
	unknown_0
	unknown_1:-
	unknown_2:-
	unknown_3:-
	unknown_4:-
	unknown_5:-
	unknown_6:-
	unknown_7:--
	unknown_8:-
	unknown_9:-

unknown_10:-

unknown_11:-

unknown_12:-

unknown_13:--

unknown_14:-

unknown_15:-

unknown_16:-

unknown_17:-

unknown_18:-

unknown_19:--

unknown_20:-

unknown_21:-

unknown_22:-

unknown_23:-

unknown_24:-

unknown_25:--

unknown_26:-

unknown_27:-

unknown_28:-

unknown_29:-

unknown_30:-

unknown_31:-l

unknown_32:l

unknown_33:l

unknown_34:l

unknown_35:l

unknown_36:l

unknown_37:ll

unknown_38:l

unknown_39:l

unknown_40:l

unknown_41:l

unknown_42:l

unknown_43:ll

unknown_44:l

unknown_45:l

unknown_46:l

unknown_47:l

unknown_48:l

unknown_49:ll

unknown_50:l

unknown_51:l

unknown_52:l

unknown_53:l

unknown_54:l

unknown_55:ll

unknown_56:l

unknown_57:l

unknown_58:l

unknown_59:l

unknown_60:l

unknown_61:lV

unknown_62:V

unknown_63:V

unknown_64:V

unknown_65:V

unknown_66:V

unknown_67:V

unknown_68:
identity¢StatefulPartitionedCallõ	
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
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*P
_read_only_resource_inputs2
0.	
 !"%&'(+,-.1234789:=>?@CDEF*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_63_layer_call_and_return_conditional_losses_832118o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
%
ì
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_830903

inputs5
'assignmovingavg_readvariableop_resource:l7
)assignmovingavg_1_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l/
!batchnorm_readvariableop_resource:l
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:l
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿll
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:l*
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
:l*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:lx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l¬
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
:l*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:l~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l´
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:lv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
È	
ö
E__inference_dense_620_layer_call_and_return_conditional_losses_834078

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ä

*__inference_dense_619_layer_call_fn_833959

inputs
unknown:-
	unknown_0:-
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_619_layer_call_and_return_conditional_losses_831102o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_830821

inputs5
'assignmovingavg_readvariableop_resource:l7
)assignmovingavg_1_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l/
!batchnorm_readvariableop_resource:l
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:l
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿll
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:l*
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
:l*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:lx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l¬
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
:l*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:l~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l´
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:lv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_830938

inputs/
!batchnorm_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l1
#batchnorm_readvariableop_1_resource:l1
#batchnorm_readvariableop_2_resource:l
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:lz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
È	
ö
E__inference_dense_622_layer_call_and_return_conditional_losses_834296

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_835030

inputs5
'assignmovingavg_readvariableop_resource:l7
)assignmovingavg_1_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l/
!batchnorm_readvariableop_resource:l
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:l
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿll
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:l*
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
:l*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:lx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l¬
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
:l*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:l~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l´
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:lv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_830247

inputs5
'assignmovingavg_readvariableop_resource:-7
)assignmovingavg_1_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-/
!batchnorm_readvariableop_resource:-
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:-
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:-*
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
:-*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:-x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-¬
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
:-*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:-~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-´
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:-v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
È	
ö
E__inference_dense_619_layer_call_and_return_conditional_losses_833969

inputs0
matmul_readvariableop_resource:--
biasadd_readvariableop_resource:-
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:-*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ´

I__inference_sequential_63_layer_call_and_return_conditional_losses_832768
normalization_63_input
normalization_63_sub_y
normalization_63_sqrt_x"
dense_619_832597:-
dense_619_832599:-,
batch_normalization_556_832602:-,
batch_normalization_556_832604:-,
batch_normalization_556_832606:-,
batch_normalization_556_832608:-"
dense_620_832612:--
dense_620_832614:-,
batch_normalization_557_832617:-,
batch_normalization_557_832619:-,
batch_normalization_557_832621:-,
batch_normalization_557_832623:-"
dense_621_832627:--
dense_621_832629:-,
batch_normalization_558_832632:-,
batch_normalization_558_832634:-,
batch_normalization_558_832636:-,
batch_normalization_558_832638:-"
dense_622_832642:--
dense_622_832644:-,
batch_normalization_559_832647:-,
batch_normalization_559_832649:-,
batch_normalization_559_832651:-,
batch_normalization_559_832653:-"
dense_623_832657:--
dense_623_832659:-,
batch_normalization_560_832662:-,
batch_normalization_560_832664:-,
batch_normalization_560_832666:-,
batch_normalization_560_832668:-"
dense_624_832672:-l
dense_624_832674:l,
batch_normalization_561_832677:l,
batch_normalization_561_832679:l,
batch_normalization_561_832681:l,
batch_normalization_561_832683:l"
dense_625_832687:ll
dense_625_832689:l,
batch_normalization_562_832692:l,
batch_normalization_562_832694:l,
batch_normalization_562_832696:l,
batch_normalization_562_832698:l"
dense_626_832702:ll
dense_626_832704:l,
batch_normalization_563_832707:l,
batch_normalization_563_832709:l,
batch_normalization_563_832711:l,
batch_normalization_563_832713:l"
dense_627_832717:ll
dense_627_832719:l,
batch_normalization_564_832722:l,
batch_normalization_564_832724:l,
batch_normalization_564_832726:l,
batch_normalization_564_832728:l"
dense_628_832732:ll
dense_628_832734:l,
batch_normalization_565_832737:l,
batch_normalization_565_832739:l,
batch_normalization_565_832741:l,
batch_normalization_565_832743:l"
dense_629_832747:lV
dense_629_832749:V,
batch_normalization_566_832752:V,
batch_normalization_566_832754:V,
batch_normalization_566_832756:V,
batch_normalization_566_832758:V"
dense_630_832762:V
dense_630_832764:
identity¢/batch_normalization_556/StatefulPartitionedCall¢/batch_normalization_557/StatefulPartitionedCall¢/batch_normalization_558/StatefulPartitionedCall¢/batch_normalization_559/StatefulPartitionedCall¢/batch_normalization_560/StatefulPartitionedCall¢/batch_normalization_561/StatefulPartitionedCall¢/batch_normalization_562/StatefulPartitionedCall¢/batch_normalization_563/StatefulPartitionedCall¢/batch_normalization_564/StatefulPartitionedCall¢/batch_normalization_565/StatefulPartitionedCall¢/batch_normalization_566/StatefulPartitionedCall¢!dense_619/StatefulPartitionedCall¢!dense_620/StatefulPartitionedCall¢!dense_621/StatefulPartitionedCall¢!dense_622/StatefulPartitionedCall¢!dense_623/StatefulPartitionedCall¢!dense_624/StatefulPartitionedCall¢!dense_625/StatefulPartitionedCall¢!dense_626/StatefulPartitionedCall¢!dense_627/StatefulPartitionedCall¢!dense_628/StatefulPartitionedCall¢!dense_629/StatefulPartitionedCall¢!dense_630/StatefulPartitionedCall}
normalization_63/subSubnormalization_63_inputnormalization_63_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_63/SqrtSqrtnormalization_63_sqrt_x*
T0*
_output_shapes

:_
normalization_63/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_63/MaximumMaximumnormalization_63/Sqrt:y:0#normalization_63/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_63/truedivRealDivnormalization_63/sub:z:0normalization_63/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_619/StatefulPartitionedCallStatefulPartitionedCallnormalization_63/truediv:z:0dense_619_832597dense_619_832599*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_619_layer_call_and_return_conditional_losses_831102
/batch_normalization_556/StatefulPartitionedCallStatefulPartitionedCall*dense_619/StatefulPartitionedCall:output:0batch_normalization_556_832602batch_normalization_556_832604batch_normalization_556_832606batch_normalization_556_832608*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_830247ø
leaky_re_lu_556/PartitionedCallPartitionedCall8batch_normalization_556/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_556_layer_call_and_return_conditional_losses_831122
!dense_620/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_556/PartitionedCall:output:0dense_620_832612dense_620_832614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_620_layer_call_and_return_conditional_losses_831134
/batch_normalization_557/StatefulPartitionedCallStatefulPartitionedCall*dense_620/StatefulPartitionedCall:output:0batch_normalization_557_832617batch_normalization_557_832619batch_normalization_557_832621batch_normalization_557_832623*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_830329ø
leaky_re_lu_557/PartitionedCallPartitionedCall8batch_normalization_557/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_557_layer_call_and_return_conditional_losses_831154
!dense_621/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_557/PartitionedCall:output:0dense_621_832627dense_621_832629*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_621_layer_call_and_return_conditional_losses_831166
/batch_normalization_558/StatefulPartitionedCallStatefulPartitionedCall*dense_621/StatefulPartitionedCall:output:0batch_normalization_558_832632batch_normalization_558_832634batch_normalization_558_832636batch_normalization_558_832638*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_830411ø
leaky_re_lu_558/PartitionedCallPartitionedCall8batch_normalization_558/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_558_layer_call_and_return_conditional_losses_831186
!dense_622/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_558/PartitionedCall:output:0dense_622_832642dense_622_832644*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_622_layer_call_and_return_conditional_losses_831198
/batch_normalization_559/StatefulPartitionedCallStatefulPartitionedCall*dense_622/StatefulPartitionedCall:output:0batch_normalization_559_832647batch_normalization_559_832649batch_normalization_559_832651batch_normalization_559_832653*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_830493ø
leaky_re_lu_559/PartitionedCallPartitionedCall8batch_normalization_559/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_559_layer_call_and_return_conditional_losses_831218
!dense_623/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_559/PartitionedCall:output:0dense_623_832657dense_623_832659*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_623_layer_call_and_return_conditional_losses_831230
/batch_normalization_560/StatefulPartitionedCallStatefulPartitionedCall*dense_623/StatefulPartitionedCall:output:0batch_normalization_560_832662batch_normalization_560_832664batch_normalization_560_832666batch_normalization_560_832668*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_830575ø
leaky_re_lu_560/PartitionedCallPartitionedCall8batch_normalization_560/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_560_layer_call_and_return_conditional_losses_831250
!dense_624/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_560/PartitionedCall:output:0dense_624_832672dense_624_832674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_624_layer_call_and_return_conditional_losses_831262
/batch_normalization_561/StatefulPartitionedCallStatefulPartitionedCall*dense_624/StatefulPartitionedCall:output:0batch_normalization_561_832677batch_normalization_561_832679batch_normalization_561_832681batch_normalization_561_832683*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_830657ø
leaky_re_lu_561/PartitionedCallPartitionedCall8batch_normalization_561/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_561_layer_call_and_return_conditional_losses_831282
!dense_625/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_561/PartitionedCall:output:0dense_625_832687dense_625_832689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_625_layer_call_and_return_conditional_losses_831294
/batch_normalization_562/StatefulPartitionedCallStatefulPartitionedCall*dense_625/StatefulPartitionedCall:output:0batch_normalization_562_832692batch_normalization_562_832694batch_normalization_562_832696batch_normalization_562_832698*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_830739ø
leaky_re_lu_562/PartitionedCallPartitionedCall8batch_normalization_562/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_562_layer_call_and_return_conditional_losses_831314
!dense_626/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_562/PartitionedCall:output:0dense_626_832702dense_626_832704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_626_layer_call_and_return_conditional_losses_831326
/batch_normalization_563/StatefulPartitionedCallStatefulPartitionedCall*dense_626/StatefulPartitionedCall:output:0batch_normalization_563_832707batch_normalization_563_832709batch_normalization_563_832711batch_normalization_563_832713*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_830821ø
leaky_re_lu_563/PartitionedCallPartitionedCall8batch_normalization_563/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_563_layer_call_and_return_conditional_losses_831346
!dense_627/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_563/PartitionedCall:output:0dense_627_832717dense_627_832719*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_627_layer_call_and_return_conditional_losses_831358
/batch_normalization_564/StatefulPartitionedCallStatefulPartitionedCall*dense_627/StatefulPartitionedCall:output:0batch_normalization_564_832722batch_normalization_564_832724batch_normalization_564_832726batch_normalization_564_832728*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_830903ø
leaky_re_lu_564/PartitionedCallPartitionedCall8batch_normalization_564/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_564_layer_call_and_return_conditional_losses_831378
!dense_628/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_564/PartitionedCall:output:0dense_628_832732dense_628_832734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_628_layer_call_and_return_conditional_losses_831390
/batch_normalization_565/StatefulPartitionedCallStatefulPartitionedCall*dense_628/StatefulPartitionedCall:output:0batch_normalization_565_832737batch_normalization_565_832739batch_normalization_565_832741batch_normalization_565_832743*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_830985ø
leaky_re_lu_565/PartitionedCallPartitionedCall8batch_normalization_565/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_565_layer_call_and_return_conditional_losses_831410
!dense_629/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_565/PartitionedCall:output:0dense_629_832747dense_629_832749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_629_layer_call_and_return_conditional_losses_831422
/batch_normalization_566/StatefulPartitionedCallStatefulPartitionedCall*dense_629/StatefulPartitionedCall:output:0batch_normalization_566_832752batch_normalization_566_832754batch_normalization_566_832756batch_normalization_566_832758*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_831067ø
leaky_re_lu_566/PartitionedCallPartitionedCall8batch_normalization_566/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_566_layer_call_and_return_conditional_losses_831442
!dense_630/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_566/PartitionedCall:output:0dense_630_832762dense_630_832764*
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
E__inference_dense_630_layer_call_and_return_conditional_losses_831454y
IdentityIdentity*dense_630/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_556/StatefulPartitionedCall0^batch_normalization_557/StatefulPartitionedCall0^batch_normalization_558/StatefulPartitionedCall0^batch_normalization_559/StatefulPartitionedCall0^batch_normalization_560/StatefulPartitionedCall0^batch_normalization_561/StatefulPartitionedCall0^batch_normalization_562/StatefulPartitionedCall0^batch_normalization_563/StatefulPartitionedCall0^batch_normalization_564/StatefulPartitionedCall0^batch_normalization_565/StatefulPartitionedCall0^batch_normalization_566/StatefulPartitionedCall"^dense_619/StatefulPartitionedCall"^dense_620/StatefulPartitionedCall"^dense_621/StatefulPartitionedCall"^dense_622/StatefulPartitionedCall"^dense_623/StatefulPartitionedCall"^dense_624/StatefulPartitionedCall"^dense_625/StatefulPartitionedCall"^dense_626/StatefulPartitionedCall"^dense_627/StatefulPartitionedCall"^dense_628/StatefulPartitionedCall"^dense_629/StatefulPartitionedCall"^dense_630/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_556/StatefulPartitionedCall/batch_normalization_556/StatefulPartitionedCall2b
/batch_normalization_557/StatefulPartitionedCall/batch_normalization_557/StatefulPartitionedCall2b
/batch_normalization_558/StatefulPartitionedCall/batch_normalization_558/StatefulPartitionedCall2b
/batch_normalization_559/StatefulPartitionedCall/batch_normalization_559/StatefulPartitionedCall2b
/batch_normalization_560/StatefulPartitionedCall/batch_normalization_560/StatefulPartitionedCall2b
/batch_normalization_561/StatefulPartitionedCall/batch_normalization_561/StatefulPartitionedCall2b
/batch_normalization_562/StatefulPartitionedCall/batch_normalization_562/StatefulPartitionedCall2b
/batch_normalization_563/StatefulPartitionedCall/batch_normalization_563/StatefulPartitionedCall2b
/batch_normalization_564/StatefulPartitionedCall/batch_normalization_564/StatefulPartitionedCall2b
/batch_normalization_565/StatefulPartitionedCall/batch_normalization_565/StatefulPartitionedCall2b
/batch_normalization_566/StatefulPartitionedCall/batch_normalization_566/StatefulPartitionedCall2F
!dense_619/StatefulPartitionedCall!dense_619/StatefulPartitionedCall2F
!dense_620/StatefulPartitionedCall!dense_620/StatefulPartitionedCall2F
!dense_621/StatefulPartitionedCall!dense_621/StatefulPartitionedCall2F
!dense_622/StatefulPartitionedCall!dense_622/StatefulPartitionedCall2F
!dense_623/StatefulPartitionedCall!dense_623/StatefulPartitionedCall2F
!dense_624/StatefulPartitionedCall!dense_624/StatefulPartitionedCall2F
!dense_625/StatefulPartitionedCall!dense_625/StatefulPartitionedCall2F
!dense_626/StatefulPartitionedCall!dense_626/StatefulPartitionedCall2F
!dense_627/StatefulPartitionedCall!dense_627/StatefulPartitionedCall2F
!dense_628/StatefulPartitionedCall!dense_628/StatefulPartitionedCall2F
!dense_629/StatefulPartitionedCall!dense_629/StatefulPartitionedCall2F
!dense_630/StatefulPartitionedCall!dense_630/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_63_input:$ 

_output_shapes

::$ 

_output_shapes

:
»´

I__inference_sequential_63_layer_call_and_return_conditional_losses_831461

inputs
normalization_63_sub_y
normalization_63_sqrt_x"
dense_619_831103:-
dense_619_831105:-,
batch_normalization_556_831108:-,
batch_normalization_556_831110:-,
batch_normalization_556_831112:-,
batch_normalization_556_831114:-"
dense_620_831135:--
dense_620_831137:-,
batch_normalization_557_831140:-,
batch_normalization_557_831142:-,
batch_normalization_557_831144:-,
batch_normalization_557_831146:-"
dense_621_831167:--
dense_621_831169:-,
batch_normalization_558_831172:-,
batch_normalization_558_831174:-,
batch_normalization_558_831176:-,
batch_normalization_558_831178:-"
dense_622_831199:--
dense_622_831201:-,
batch_normalization_559_831204:-,
batch_normalization_559_831206:-,
batch_normalization_559_831208:-,
batch_normalization_559_831210:-"
dense_623_831231:--
dense_623_831233:-,
batch_normalization_560_831236:-,
batch_normalization_560_831238:-,
batch_normalization_560_831240:-,
batch_normalization_560_831242:-"
dense_624_831263:-l
dense_624_831265:l,
batch_normalization_561_831268:l,
batch_normalization_561_831270:l,
batch_normalization_561_831272:l,
batch_normalization_561_831274:l"
dense_625_831295:ll
dense_625_831297:l,
batch_normalization_562_831300:l,
batch_normalization_562_831302:l,
batch_normalization_562_831304:l,
batch_normalization_562_831306:l"
dense_626_831327:ll
dense_626_831329:l,
batch_normalization_563_831332:l,
batch_normalization_563_831334:l,
batch_normalization_563_831336:l,
batch_normalization_563_831338:l"
dense_627_831359:ll
dense_627_831361:l,
batch_normalization_564_831364:l,
batch_normalization_564_831366:l,
batch_normalization_564_831368:l,
batch_normalization_564_831370:l"
dense_628_831391:ll
dense_628_831393:l,
batch_normalization_565_831396:l,
batch_normalization_565_831398:l,
batch_normalization_565_831400:l,
batch_normalization_565_831402:l"
dense_629_831423:lV
dense_629_831425:V,
batch_normalization_566_831428:V,
batch_normalization_566_831430:V,
batch_normalization_566_831432:V,
batch_normalization_566_831434:V"
dense_630_831455:V
dense_630_831457:
identity¢/batch_normalization_556/StatefulPartitionedCall¢/batch_normalization_557/StatefulPartitionedCall¢/batch_normalization_558/StatefulPartitionedCall¢/batch_normalization_559/StatefulPartitionedCall¢/batch_normalization_560/StatefulPartitionedCall¢/batch_normalization_561/StatefulPartitionedCall¢/batch_normalization_562/StatefulPartitionedCall¢/batch_normalization_563/StatefulPartitionedCall¢/batch_normalization_564/StatefulPartitionedCall¢/batch_normalization_565/StatefulPartitionedCall¢/batch_normalization_566/StatefulPartitionedCall¢!dense_619/StatefulPartitionedCall¢!dense_620/StatefulPartitionedCall¢!dense_621/StatefulPartitionedCall¢!dense_622/StatefulPartitionedCall¢!dense_623/StatefulPartitionedCall¢!dense_624/StatefulPartitionedCall¢!dense_625/StatefulPartitionedCall¢!dense_626/StatefulPartitionedCall¢!dense_627/StatefulPartitionedCall¢!dense_628/StatefulPartitionedCall¢!dense_629/StatefulPartitionedCall¢!dense_630/StatefulPartitionedCallm
normalization_63/subSubinputsnormalization_63_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_63/SqrtSqrtnormalization_63_sqrt_x*
T0*
_output_shapes

:_
normalization_63/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_63/MaximumMaximumnormalization_63/Sqrt:y:0#normalization_63/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_63/truedivRealDivnormalization_63/sub:z:0normalization_63/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_619/StatefulPartitionedCallStatefulPartitionedCallnormalization_63/truediv:z:0dense_619_831103dense_619_831105*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_619_layer_call_and_return_conditional_losses_831102
/batch_normalization_556/StatefulPartitionedCallStatefulPartitionedCall*dense_619/StatefulPartitionedCall:output:0batch_normalization_556_831108batch_normalization_556_831110batch_normalization_556_831112batch_normalization_556_831114*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_830200ø
leaky_re_lu_556/PartitionedCallPartitionedCall8batch_normalization_556/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_556_layer_call_and_return_conditional_losses_831122
!dense_620/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_556/PartitionedCall:output:0dense_620_831135dense_620_831137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_620_layer_call_and_return_conditional_losses_831134
/batch_normalization_557/StatefulPartitionedCallStatefulPartitionedCall*dense_620/StatefulPartitionedCall:output:0batch_normalization_557_831140batch_normalization_557_831142batch_normalization_557_831144batch_normalization_557_831146*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_830282ø
leaky_re_lu_557/PartitionedCallPartitionedCall8batch_normalization_557/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_557_layer_call_and_return_conditional_losses_831154
!dense_621/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_557/PartitionedCall:output:0dense_621_831167dense_621_831169*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_621_layer_call_and_return_conditional_losses_831166
/batch_normalization_558/StatefulPartitionedCallStatefulPartitionedCall*dense_621/StatefulPartitionedCall:output:0batch_normalization_558_831172batch_normalization_558_831174batch_normalization_558_831176batch_normalization_558_831178*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_830364ø
leaky_re_lu_558/PartitionedCallPartitionedCall8batch_normalization_558/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_558_layer_call_and_return_conditional_losses_831186
!dense_622/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_558/PartitionedCall:output:0dense_622_831199dense_622_831201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_622_layer_call_and_return_conditional_losses_831198
/batch_normalization_559/StatefulPartitionedCallStatefulPartitionedCall*dense_622/StatefulPartitionedCall:output:0batch_normalization_559_831204batch_normalization_559_831206batch_normalization_559_831208batch_normalization_559_831210*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_830446ø
leaky_re_lu_559/PartitionedCallPartitionedCall8batch_normalization_559/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_559_layer_call_and_return_conditional_losses_831218
!dense_623/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_559/PartitionedCall:output:0dense_623_831231dense_623_831233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_623_layer_call_and_return_conditional_losses_831230
/batch_normalization_560/StatefulPartitionedCallStatefulPartitionedCall*dense_623/StatefulPartitionedCall:output:0batch_normalization_560_831236batch_normalization_560_831238batch_normalization_560_831240batch_normalization_560_831242*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_830528ø
leaky_re_lu_560/PartitionedCallPartitionedCall8batch_normalization_560/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_560_layer_call_and_return_conditional_losses_831250
!dense_624/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_560/PartitionedCall:output:0dense_624_831263dense_624_831265*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_624_layer_call_and_return_conditional_losses_831262
/batch_normalization_561/StatefulPartitionedCallStatefulPartitionedCall*dense_624/StatefulPartitionedCall:output:0batch_normalization_561_831268batch_normalization_561_831270batch_normalization_561_831272batch_normalization_561_831274*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_830610ø
leaky_re_lu_561/PartitionedCallPartitionedCall8batch_normalization_561/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_561_layer_call_and_return_conditional_losses_831282
!dense_625/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_561/PartitionedCall:output:0dense_625_831295dense_625_831297*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_625_layer_call_and_return_conditional_losses_831294
/batch_normalization_562/StatefulPartitionedCallStatefulPartitionedCall*dense_625/StatefulPartitionedCall:output:0batch_normalization_562_831300batch_normalization_562_831302batch_normalization_562_831304batch_normalization_562_831306*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_830692ø
leaky_re_lu_562/PartitionedCallPartitionedCall8batch_normalization_562/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_562_layer_call_and_return_conditional_losses_831314
!dense_626/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_562/PartitionedCall:output:0dense_626_831327dense_626_831329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_626_layer_call_and_return_conditional_losses_831326
/batch_normalization_563/StatefulPartitionedCallStatefulPartitionedCall*dense_626/StatefulPartitionedCall:output:0batch_normalization_563_831332batch_normalization_563_831334batch_normalization_563_831336batch_normalization_563_831338*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_830774ø
leaky_re_lu_563/PartitionedCallPartitionedCall8batch_normalization_563/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_563_layer_call_and_return_conditional_losses_831346
!dense_627/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_563/PartitionedCall:output:0dense_627_831359dense_627_831361*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_627_layer_call_and_return_conditional_losses_831358
/batch_normalization_564/StatefulPartitionedCallStatefulPartitionedCall*dense_627/StatefulPartitionedCall:output:0batch_normalization_564_831364batch_normalization_564_831366batch_normalization_564_831368batch_normalization_564_831370*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_830856ø
leaky_re_lu_564/PartitionedCallPartitionedCall8batch_normalization_564/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_564_layer_call_and_return_conditional_losses_831378
!dense_628/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_564/PartitionedCall:output:0dense_628_831391dense_628_831393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_628_layer_call_and_return_conditional_losses_831390
/batch_normalization_565/StatefulPartitionedCallStatefulPartitionedCall*dense_628/StatefulPartitionedCall:output:0batch_normalization_565_831396batch_normalization_565_831398batch_normalization_565_831400batch_normalization_565_831402*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_830938ø
leaky_re_lu_565/PartitionedCallPartitionedCall8batch_normalization_565/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_565_layer_call_and_return_conditional_losses_831410
!dense_629/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_565/PartitionedCall:output:0dense_629_831423dense_629_831425*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_629_layer_call_and_return_conditional_losses_831422
/batch_normalization_566/StatefulPartitionedCallStatefulPartitionedCall*dense_629/StatefulPartitionedCall:output:0batch_normalization_566_831428batch_normalization_566_831430batch_normalization_566_831432batch_normalization_566_831434*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_831020ø
leaky_re_lu_566/PartitionedCallPartitionedCall8batch_normalization_566/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_566_layer_call_and_return_conditional_losses_831442
!dense_630/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_566/PartitionedCall:output:0dense_630_831455dense_630_831457*
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
E__inference_dense_630_layer_call_and_return_conditional_losses_831454y
IdentityIdentity*dense_630/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_556/StatefulPartitionedCall0^batch_normalization_557/StatefulPartitionedCall0^batch_normalization_558/StatefulPartitionedCall0^batch_normalization_559/StatefulPartitionedCall0^batch_normalization_560/StatefulPartitionedCall0^batch_normalization_561/StatefulPartitionedCall0^batch_normalization_562/StatefulPartitionedCall0^batch_normalization_563/StatefulPartitionedCall0^batch_normalization_564/StatefulPartitionedCall0^batch_normalization_565/StatefulPartitionedCall0^batch_normalization_566/StatefulPartitionedCall"^dense_619/StatefulPartitionedCall"^dense_620/StatefulPartitionedCall"^dense_621/StatefulPartitionedCall"^dense_622/StatefulPartitionedCall"^dense_623/StatefulPartitionedCall"^dense_624/StatefulPartitionedCall"^dense_625/StatefulPartitionedCall"^dense_626/StatefulPartitionedCall"^dense_627/StatefulPartitionedCall"^dense_628/StatefulPartitionedCall"^dense_629/StatefulPartitionedCall"^dense_630/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_556/StatefulPartitionedCall/batch_normalization_556/StatefulPartitionedCall2b
/batch_normalization_557/StatefulPartitionedCall/batch_normalization_557/StatefulPartitionedCall2b
/batch_normalization_558/StatefulPartitionedCall/batch_normalization_558/StatefulPartitionedCall2b
/batch_normalization_559/StatefulPartitionedCall/batch_normalization_559/StatefulPartitionedCall2b
/batch_normalization_560/StatefulPartitionedCall/batch_normalization_560/StatefulPartitionedCall2b
/batch_normalization_561/StatefulPartitionedCall/batch_normalization_561/StatefulPartitionedCall2b
/batch_normalization_562/StatefulPartitionedCall/batch_normalization_562/StatefulPartitionedCall2b
/batch_normalization_563/StatefulPartitionedCall/batch_normalization_563/StatefulPartitionedCall2b
/batch_normalization_564/StatefulPartitionedCall/batch_normalization_564/StatefulPartitionedCall2b
/batch_normalization_565/StatefulPartitionedCall/batch_normalization_565/StatefulPartitionedCall2b
/batch_normalization_566/StatefulPartitionedCall/batch_normalization_566/StatefulPartitionedCall2F
!dense_619/StatefulPartitionedCall!dense_619/StatefulPartitionedCall2F
!dense_620/StatefulPartitionedCall!dense_620/StatefulPartitionedCall2F
!dense_621/StatefulPartitionedCall!dense_621/StatefulPartitionedCall2F
!dense_622/StatefulPartitionedCall!dense_622/StatefulPartitionedCall2F
!dense_623/StatefulPartitionedCall!dense_623/StatefulPartitionedCall2F
!dense_624/StatefulPartitionedCall!dense_624/StatefulPartitionedCall2F
!dense_625/StatefulPartitionedCall!dense_625/StatefulPartitionedCall2F
!dense_626/StatefulPartitionedCall!dense_626/StatefulPartitionedCall2F
!dense_627/StatefulPartitionedCall!dense_627/StatefulPartitionedCall2F
!dense_628/StatefulPartitionedCall!dense_628/StatefulPartitionedCall2F
!dense_629/StatefulPartitionedCall!dense_629/StatefulPartitionedCall2F
!dense_630/StatefulPartitionedCall!dense_630/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
«
L
0__inference_leaky_re_lu_562_layer_call_fn_834708

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
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_562_layer_call_and_return_conditional_losses_831314`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_561_layer_call_and_return_conditional_losses_831282

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_566_layer_call_fn_835144

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
:ÿÿÿÿÿÿÿÿÿV* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_566_layer_call_and_return_conditional_losses_831442`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿV:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_830282

inputs/
!batchnorm_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-1
#batchnorm_readvariableop_1_resource:-1
#batchnorm_readvariableop_2_resource:-
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:-z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_559_layer_call_and_return_conditional_losses_831218

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_565_layer_call_and_return_conditional_losses_835040

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_557_layer_call_fn_834104

inputs
unknown:-
	unknown_0:-
	unknown_1:-
	unknown_2:-
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_830329o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
î'
Ò
__inference_adapt_step_833950
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢IteratorGetNext¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢add/ReadVariableOp±
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
output_shapes
:ÿÿÿÿÿÿÿÿÿ*
output_types
2k
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0P
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:X
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:G
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0V
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0V
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:E
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:V
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @N
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:Z
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:I
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:I
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:
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
È	
ö
E__inference_dense_629_layer_call_and_return_conditional_losses_831422

inputs0
matmul_readvariableop_resource:lV-
biasadd_readvariableop_resource:V
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:lV*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:V*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
ÀÂ
ÀO
__inference__traced_save_835700
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_619_kernel_read_readvariableop-
)savev2_dense_619_bias_read_readvariableop<
8savev2_batch_normalization_556_gamma_read_readvariableop;
7savev2_batch_normalization_556_beta_read_readvariableopB
>savev2_batch_normalization_556_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_556_moving_variance_read_readvariableop/
+savev2_dense_620_kernel_read_readvariableop-
)savev2_dense_620_bias_read_readvariableop<
8savev2_batch_normalization_557_gamma_read_readvariableop;
7savev2_batch_normalization_557_beta_read_readvariableopB
>savev2_batch_normalization_557_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_557_moving_variance_read_readvariableop/
+savev2_dense_621_kernel_read_readvariableop-
)savev2_dense_621_bias_read_readvariableop<
8savev2_batch_normalization_558_gamma_read_readvariableop;
7savev2_batch_normalization_558_beta_read_readvariableopB
>savev2_batch_normalization_558_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_558_moving_variance_read_readvariableop/
+savev2_dense_622_kernel_read_readvariableop-
)savev2_dense_622_bias_read_readvariableop<
8savev2_batch_normalization_559_gamma_read_readvariableop;
7savev2_batch_normalization_559_beta_read_readvariableopB
>savev2_batch_normalization_559_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_559_moving_variance_read_readvariableop/
+savev2_dense_623_kernel_read_readvariableop-
)savev2_dense_623_bias_read_readvariableop<
8savev2_batch_normalization_560_gamma_read_readvariableop;
7savev2_batch_normalization_560_beta_read_readvariableopB
>savev2_batch_normalization_560_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_560_moving_variance_read_readvariableop/
+savev2_dense_624_kernel_read_readvariableop-
)savev2_dense_624_bias_read_readvariableop<
8savev2_batch_normalization_561_gamma_read_readvariableop;
7savev2_batch_normalization_561_beta_read_readvariableopB
>savev2_batch_normalization_561_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_561_moving_variance_read_readvariableop/
+savev2_dense_625_kernel_read_readvariableop-
)savev2_dense_625_bias_read_readvariableop<
8savev2_batch_normalization_562_gamma_read_readvariableop;
7savev2_batch_normalization_562_beta_read_readvariableopB
>savev2_batch_normalization_562_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_562_moving_variance_read_readvariableop/
+savev2_dense_626_kernel_read_readvariableop-
)savev2_dense_626_bias_read_readvariableop<
8savev2_batch_normalization_563_gamma_read_readvariableop;
7savev2_batch_normalization_563_beta_read_readvariableopB
>savev2_batch_normalization_563_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_563_moving_variance_read_readvariableop/
+savev2_dense_627_kernel_read_readvariableop-
)savev2_dense_627_bias_read_readvariableop<
8savev2_batch_normalization_564_gamma_read_readvariableop;
7savev2_batch_normalization_564_beta_read_readvariableopB
>savev2_batch_normalization_564_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_564_moving_variance_read_readvariableop/
+savev2_dense_628_kernel_read_readvariableop-
)savev2_dense_628_bias_read_readvariableop<
8savev2_batch_normalization_565_gamma_read_readvariableop;
7savev2_batch_normalization_565_beta_read_readvariableopB
>savev2_batch_normalization_565_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_565_moving_variance_read_readvariableop/
+savev2_dense_629_kernel_read_readvariableop-
)savev2_dense_629_bias_read_readvariableop<
8savev2_batch_normalization_566_gamma_read_readvariableop;
7savev2_batch_normalization_566_beta_read_readvariableopB
>savev2_batch_normalization_566_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_566_moving_variance_read_readvariableop/
+savev2_dense_630_kernel_read_readvariableop-
)savev2_dense_630_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_619_kernel_m_read_readvariableop4
0savev2_adam_dense_619_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_556_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_556_beta_m_read_readvariableop6
2savev2_adam_dense_620_kernel_m_read_readvariableop4
0savev2_adam_dense_620_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_557_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_557_beta_m_read_readvariableop6
2savev2_adam_dense_621_kernel_m_read_readvariableop4
0savev2_adam_dense_621_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_558_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_558_beta_m_read_readvariableop6
2savev2_adam_dense_622_kernel_m_read_readvariableop4
0savev2_adam_dense_622_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_559_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_559_beta_m_read_readvariableop6
2savev2_adam_dense_623_kernel_m_read_readvariableop4
0savev2_adam_dense_623_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_560_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_560_beta_m_read_readvariableop6
2savev2_adam_dense_624_kernel_m_read_readvariableop4
0savev2_adam_dense_624_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_561_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_561_beta_m_read_readvariableop6
2savev2_adam_dense_625_kernel_m_read_readvariableop4
0savev2_adam_dense_625_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_562_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_562_beta_m_read_readvariableop6
2savev2_adam_dense_626_kernel_m_read_readvariableop4
0savev2_adam_dense_626_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_563_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_563_beta_m_read_readvariableop6
2savev2_adam_dense_627_kernel_m_read_readvariableop4
0savev2_adam_dense_627_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_564_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_564_beta_m_read_readvariableop6
2savev2_adam_dense_628_kernel_m_read_readvariableop4
0savev2_adam_dense_628_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_565_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_565_beta_m_read_readvariableop6
2savev2_adam_dense_629_kernel_m_read_readvariableop4
0savev2_adam_dense_629_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_566_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_566_beta_m_read_readvariableop6
2savev2_adam_dense_630_kernel_m_read_readvariableop4
0savev2_adam_dense_630_bias_m_read_readvariableop6
2savev2_adam_dense_619_kernel_v_read_readvariableop4
0savev2_adam_dense_619_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_556_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_556_beta_v_read_readvariableop6
2savev2_adam_dense_620_kernel_v_read_readvariableop4
0savev2_adam_dense_620_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_557_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_557_beta_v_read_readvariableop6
2savev2_adam_dense_621_kernel_v_read_readvariableop4
0savev2_adam_dense_621_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_558_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_558_beta_v_read_readvariableop6
2savev2_adam_dense_622_kernel_v_read_readvariableop4
0savev2_adam_dense_622_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_559_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_559_beta_v_read_readvariableop6
2savev2_adam_dense_623_kernel_v_read_readvariableop4
0savev2_adam_dense_623_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_560_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_560_beta_v_read_readvariableop6
2savev2_adam_dense_624_kernel_v_read_readvariableop4
0savev2_adam_dense_624_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_561_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_561_beta_v_read_readvariableop6
2savev2_adam_dense_625_kernel_v_read_readvariableop4
0savev2_adam_dense_625_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_562_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_562_beta_v_read_readvariableop6
2savev2_adam_dense_626_kernel_v_read_readvariableop4
0savev2_adam_dense_626_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_563_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_563_beta_v_read_readvariableop6
2savev2_adam_dense_627_kernel_v_read_readvariableop4
0savev2_adam_dense_627_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_564_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_564_beta_v_read_readvariableop6
2savev2_adam_dense_628_kernel_v_read_readvariableop4
0savev2_adam_dense_628_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_565_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_565_beta_v_read_readvariableop6
2savev2_adam_dense_629_kernel_v_read_readvariableop4
0savev2_adam_dense_629_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_566_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_566_beta_v_read_readvariableop6
2savev2_adam_dense_630_kernel_v_read_readvariableop4
0savev2_adam_dense_630_bias_v_read_readvariableop
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
: ±_
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:ª*
dtype0*Ù^
valueÏ^BÌ^ªB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-22/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-22/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-22/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÆ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:ª*
dtype0*ê
valueàBÝªB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B L
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_619_kernel_read_readvariableop)savev2_dense_619_bias_read_readvariableop8savev2_batch_normalization_556_gamma_read_readvariableop7savev2_batch_normalization_556_beta_read_readvariableop>savev2_batch_normalization_556_moving_mean_read_readvariableopBsavev2_batch_normalization_556_moving_variance_read_readvariableop+savev2_dense_620_kernel_read_readvariableop)savev2_dense_620_bias_read_readvariableop8savev2_batch_normalization_557_gamma_read_readvariableop7savev2_batch_normalization_557_beta_read_readvariableop>savev2_batch_normalization_557_moving_mean_read_readvariableopBsavev2_batch_normalization_557_moving_variance_read_readvariableop+savev2_dense_621_kernel_read_readvariableop)savev2_dense_621_bias_read_readvariableop8savev2_batch_normalization_558_gamma_read_readvariableop7savev2_batch_normalization_558_beta_read_readvariableop>savev2_batch_normalization_558_moving_mean_read_readvariableopBsavev2_batch_normalization_558_moving_variance_read_readvariableop+savev2_dense_622_kernel_read_readvariableop)savev2_dense_622_bias_read_readvariableop8savev2_batch_normalization_559_gamma_read_readvariableop7savev2_batch_normalization_559_beta_read_readvariableop>savev2_batch_normalization_559_moving_mean_read_readvariableopBsavev2_batch_normalization_559_moving_variance_read_readvariableop+savev2_dense_623_kernel_read_readvariableop)savev2_dense_623_bias_read_readvariableop8savev2_batch_normalization_560_gamma_read_readvariableop7savev2_batch_normalization_560_beta_read_readvariableop>savev2_batch_normalization_560_moving_mean_read_readvariableopBsavev2_batch_normalization_560_moving_variance_read_readvariableop+savev2_dense_624_kernel_read_readvariableop)savev2_dense_624_bias_read_readvariableop8savev2_batch_normalization_561_gamma_read_readvariableop7savev2_batch_normalization_561_beta_read_readvariableop>savev2_batch_normalization_561_moving_mean_read_readvariableopBsavev2_batch_normalization_561_moving_variance_read_readvariableop+savev2_dense_625_kernel_read_readvariableop)savev2_dense_625_bias_read_readvariableop8savev2_batch_normalization_562_gamma_read_readvariableop7savev2_batch_normalization_562_beta_read_readvariableop>savev2_batch_normalization_562_moving_mean_read_readvariableopBsavev2_batch_normalization_562_moving_variance_read_readvariableop+savev2_dense_626_kernel_read_readvariableop)savev2_dense_626_bias_read_readvariableop8savev2_batch_normalization_563_gamma_read_readvariableop7savev2_batch_normalization_563_beta_read_readvariableop>savev2_batch_normalization_563_moving_mean_read_readvariableopBsavev2_batch_normalization_563_moving_variance_read_readvariableop+savev2_dense_627_kernel_read_readvariableop)savev2_dense_627_bias_read_readvariableop8savev2_batch_normalization_564_gamma_read_readvariableop7savev2_batch_normalization_564_beta_read_readvariableop>savev2_batch_normalization_564_moving_mean_read_readvariableopBsavev2_batch_normalization_564_moving_variance_read_readvariableop+savev2_dense_628_kernel_read_readvariableop)savev2_dense_628_bias_read_readvariableop8savev2_batch_normalization_565_gamma_read_readvariableop7savev2_batch_normalization_565_beta_read_readvariableop>savev2_batch_normalization_565_moving_mean_read_readvariableopBsavev2_batch_normalization_565_moving_variance_read_readvariableop+savev2_dense_629_kernel_read_readvariableop)savev2_dense_629_bias_read_readvariableop8savev2_batch_normalization_566_gamma_read_readvariableop7savev2_batch_normalization_566_beta_read_readvariableop>savev2_batch_normalization_566_moving_mean_read_readvariableopBsavev2_batch_normalization_566_moving_variance_read_readvariableop+savev2_dense_630_kernel_read_readvariableop)savev2_dense_630_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_619_kernel_m_read_readvariableop0savev2_adam_dense_619_bias_m_read_readvariableop?savev2_adam_batch_normalization_556_gamma_m_read_readvariableop>savev2_adam_batch_normalization_556_beta_m_read_readvariableop2savev2_adam_dense_620_kernel_m_read_readvariableop0savev2_adam_dense_620_bias_m_read_readvariableop?savev2_adam_batch_normalization_557_gamma_m_read_readvariableop>savev2_adam_batch_normalization_557_beta_m_read_readvariableop2savev2_adam_dense_621_kernel_m_read_readvariableop0savev2_adam_dense_621_bias_m_read_readvariableop?savev2_adam_batch_normalization_558_gamma_m_read_readvariableop>savev2_adam_batch_normalization_558_beta_m_read_readvariableop2savev2_adam_dense_622_kernel_m_read_readvariableop0savev2_adam_dense_622_bias_m_read_readvariableop?savev2_adam_batch_normalization_559_gamma_m_read_readvariableop>savev2_adam_batch_normalization_559_beta_m_read_readvariableop2savev2_adam_dense_623_kernel_m_read_readvariableop0savev2_adam_dense_623_bias_m_read_readvariableop?savev2_adam_batch_normalization_560_gamma_m_read_readvariableop>savev2_adam_batch_normalization_560_beta_m_read_readvariableop2savev2_adam_dense_624_kernel_m_read_readvariableop0savev2_adam_dense_624_bias_m_read_readvariableop?savev2_adam_batch_normalization_561_gamma_m_read_readvariableop>savev2_adam_batch_normalization_561_beta_m_read_readvariableop2savev2_adam_dense_625_kernel_m_read_readvariableop0savev2_adam_dense_625_bias_m_read_readvariableop?savev2_adam_batch_normalization_562_gamma_m_read_readvariableop>savev2_adam_batch_normalization_562_beta_m_read_readvariableop2savev2_adam_dense_626_kernel_m_read_readvariableop0savev2_adam_dense_626_bias_m_read_readvariableop?savev2_adam_batch_normalization_563_gamma_m_read_readvariableop>savev2_adam_batch_normalization_563_beta_m_read_readvariableop2savev2_adam_dense_627_kernel_m_read_readvariableop0savev2_adam_dense_627_bias_m_read_readvariableop?savev2_adam_batch_normalization_564_gamma_m_read_readvariableop>savev2_adam_batch_normalization_564_beta_m_read_readvariableop2savev2_adam_dense_628_kernel_m_read_readvariableop0savev2_adam_dense_628_bias_m_read_readvariableop?savev2_adam_batch_normalization_565_gamma_m_read_readvariableop>savev2_adam_batch_normalization_565_beta_m_read_readvariableop2savev2_adam_dense_629_kernel_m_read_readvariableop0savev2_adam_dense_629_bias_m_read_readvariableop?savev2_adam_batch_normalization_566_gamma_m_read_readvariableop>savev2_adam_batch_normalization_566_beta_m_read_readvariableop2savev2_adam_dense_630_kernel_m_read_readvariableop0savev2_adam_dense_630_bias_m_read_readvariableop2savev2_adam_dense_619_kernel_v_read_readvariableop0savev2_adam_dense_619_bias_v_read_readvariableop?savev2_adam_batch_normalization_556_gamma_v_read_readvariableop>savev2_adam_batch_normalization_556_beta_v_read_readvariableop2savev2_adam_dense_620_kernel_v_read_readvariableop0savev2_adam_dense_620_bias_v_read_readvariableop?savev2_adam_batch_normalization_557_gamma_v_read_readvariableop>savev2_adam_batch_normalization_557_beta_v_read_readvariableop2savev2_adam_dense_621_kernel_v_read_readvariableop0savev2_adam_dense_621_bias_v_read_readvariableop?savev2_adam_batch_normalization_558_gamma_v_read_readvariableop>savev2_adam_batch_normalization_558_beta_v_read_readvariableop2savev2_adam_dense_622_kernel_v_read_readvariableop0savev2_adam_dense_622_bias_v_read_readvariableop?savev2_adam_batch_normalization_559_gamma_v_read_readvariableop>savev2_adam_batch_normalization_559_beta_v_read_readvariableop2savev2_adam_dense_623_kernel_v_read_readvariableop0savev2_adam_dense_623_bias_v_read_readvariableop?savev2_adam_batch_normalization_560_gamma_v_read_readvariableop>savev2_adam_batch_normalization_560_beta_v_read_readvariableop2savev2_adam_dense_624_kernel_v_read_readvariableop0savev2_adam_dense_624_bias_v_read_readvariableop?savev2_adam_batch_normalization_561_gamma_v_read_readvariableop>savev2_adam_batch_normalization_561_beta_v_read_readvariableop2savev2_adam_dense_625_kernel_v_read_readvariableop0savev2_adam_dense_625_bias_v_read_readvariableop?savev2_adam_batch_normalization_562_gamma_v_read_readvariableop>savev2_adam_batch_normalization_562_beta_v_read_readvariableop2savev2_adam_dense_626_kernel_v_read_readvariableop0savev2_adam_dense_626_bias_v_read_readvariableop?savev2_adam_batch_normalization_563_gamma_v_read_readvariableop>savev2_adam_batch_normalization_563_beta_v_read_readvariableop2savev2_adam_dense_627_kernel_v_read_readvariableop0savev2_adam_dense_627_bias_v_read_readvariableop?savev2_adam_batch_normalization_564_gamma_v_read_readvariableop>savev2_adam_batch_normalization_564_beta_v_read_readvariableop2savev2_adam_dense_628_kernel_v_read_readvariableop0savev2_adam_dense_628_bias_v_read_readvariableop?savev2_adam_batch_normalization_565_gamma_v_read_readvariableop>savev2_adam_batch_normalization_565_beta_v_read_readvariableop2savev2_adam_dense_629_kernel_v_read_readvariableop0savev2_adam_dense_629_bias_v_read_readvariableop?savev2_adam_batch_normalization_566_gamma_v_read_readvariableop>savev2_adam_batch_normalization_566_beta_v_read_readvariableop2savev2_adam_dense_630_kernel_v_read_readvariableop0savev2_adam_dense_630_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *»
dtypes°
­2ª		
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

identity_1Identity_1:output:0*	
_input_shapesñ
î: ::: :-:-:-:-:-:-:--:-:-:-:-:-:--:-:-:-:-:-:--:-:-:-:-:-:--:-:-:-:-:-:-l:l:l:l:l:l:ll:l:l:l:l:l:ll:l:l:l:l:l:ll:l:l:l:l:l:ll:l:l:l:l:l:lV:V:V:V:V:V:V:: : : : : : :-:-:-:-:--:-:-:-:--:-:-:-:--:-:-:-:--:-:-:-:-l:l:l:l:ll:l:l:l:ll:l:l:l:ll:l:l:l:ll:l:l:l:lV:V:V:V:V::-:-:-:-:--:-:-:-:--:-:-:-:--:-:-:-:--:-:-:-:-l:l:l:l:ll:l:l:l:ll:l:l:l:ll:l:l:l:ll:l:l:l:lV:V:V:V:V:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :$ 

_output_shapes

:-: 

_output_shapes
:-: 

_output_shapes
:-: 

_output_shapes
:-: 

_output_shapes
:-: 	

_output_shapes
:-:$
 

_output_shapes

:--: 

_output_shapes
:-: 

_output_shapes
:-: 

_output_shapes
:-: 

_output_shapes
:-: 

_output_shapes
:-:$ 

_output_shapes

:--: 

_output_shapes
:-: 

_output_shapes
:-: 

_output_shapes
:-: 

_output_shapes
:-: 

_output_shapes
:-:$ 

_output_shapes

:--: 

_output_shapes
:-: 

_output_shapes
:-: 

_output_shapes
:-: 

_output_shapes
:-: 

_output_shapes
:-:$ 

_output_shapes

:--: 

_output_shapes
:-: 

_output_shapes
:-: 

_output_shapes
:-:  

_output_shapes
:-: !

_output_shapes
:-:$" 

_output_shapes

:-l: #

_output_shapes
:l: $

_output_shapes
:l: %

_output_shapes
:l: &

_output_shapes
:l: '

_output_shapes
:l:$( 

_output_shapes

:ll: )

_output_shapes
:l: *

_output_shapes
:l: +

_output_shapes
:l: ,

_output_shapes
:l: -

_output_shapes
:l:$. 

_output_shapes

:ll: /

_output_shapes
:l: 0

_output_shapes
:l: 1

_output_shapes
:l: 2

_output_shapes
:l: 3

_output_shapes
:l:$4 

_output_shapes

:ll: 5

_output_shapes
:l: 6

_output_shapes
:l: 7

_output_shapes
:l: 8

_output_shapes
:l: 9

_output_shapes
:l:$: 

_output_shapes

:ll: ;

_output_shapes
:l: <

_output_shapes
:l: =

_output_shapes
:l: >

_output_shapes
:l: ?

_output_shapes
:l:$@ 

_output_shapes

:lV: A

_output_shapes
:V: B

_output_shapes
:V: C

_output_shapes
:V: D

_output_shapes
:V: E

_output_shapes
:V:$F 

_output_shapes

:V: G

_output_shapes
::H

_output_shapes
: :I

_output_shapes
: :J

_output_shapes
: :K

_output_shapes
: :L

_output_shapes
: :M

_output_shapes
: :$N 

_output_shapes

:-: O

_output_shapes
:-: P

_output_shapes
:-: Q

_output_shapes
:-:$R 

_output_shapes

:--: S

_output_shapes
:-: T

_output_shapes
:-: U

_output_shapes
:-:$V 

_output_shapes

:--: W

_output_shapes
:-: X

_output_shapes
:-: Y

_output_shapes
:-:$Z 

_output_shapes

:--: [

_output_shapes
:-: \

_output_shapes
:-: ]

_output_shapes
:-:$^ 

_output_shapes

:--: _

_output_shapes
:-: `

_output_shapes
:-: a

_output_shapes
:-:$b 

_output_shapes

:-l: c

_output_shapes
:l: d

_output_shapes
:l: e

_output_shapes
:l:$f 

_output_shapes

:ll: g

_output_shapes
:l: h

_output_shapes
:l: i

_output_shapes
:l:$j 

_output_shapes

:ll: k

_output_shapes
:l: l

_output_shapes
:l: m

_output_shapes
:l:$n 

_output_shapes

:ll: o

_output_shapes
:l: p

_output_shapes
:l: q

_output_shapes
:l:$r 

_output_shapes

:ll: s

_output_shapes
:l: t

_output_shapes
:l: u

_output_shapes
:l:$v 

_output_shapes

:lV: w

_output_shapes
:V: x

_output_shapes
:V: y

_output_shapes
:V:$z 

_output_shapes

:V: {

_output_shapes
::$| 

_output_shapes

:-: }

_output_shapes
:-: ~

_output_shapes
:-: 

_output_shapes
:-:% 

_output_shapes

:--:!

_output_shapes
:-:!

_output_shapes
:-:!

_output_shapes
:-:% 

_output_shapes

:--:!

_output_shapes
:-:!

_output_shapes
:-:!

_output_shapes
:-:% 

_output_shapes

:--:!

_output_shapes
:-:!

_output_shapes
:-:!

_output_shapes
:-:% 

_output_shapes

:--:!

_output_shapes
:-:!

_output_shapes
:-:!

_output_shapes
:-:% 

_output_shapes

:-l:!

_output_shapes
:l:!

_output_shapes
:l:!

_output_shapes
:l:% 

_output_shapes

:ll:!

_output_shapes
:l:!

_output_shapes
:l:!

_output_shapes
:l:% 

_output_shapes

:ll:!

_output_shapes
:l:!

_output_shapes
:l:!

_output_shapes
:l:% 

_output_shapes

:ll:!

_output_shapes
:l:!

_output_shapes
:l:!

_output_shapes
:l:%  

_output_shapes

:ll:!¡

_output_shapes
:l:!¢

_output_shapes
:l:!£

_output_shapes
:l:%¤ 

_output_shapes

:lV:!¥

_output_shapes
:V:!¦

_output_shapes
:V:!§

_output_shapes
:V:%¨ 

_output_shapes

:V:!©

_output_shapes
::ª

_output_shapes
: 
È	
ö
E__inference_dense_624_layer_call_and_return_conditional_losses_831262

inputs0
matmul_readvariableop_resource:-l-
biasadd_readvariableop_resource:l
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:-l*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:l*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_562_layer_call_fn_834649

inputs
unknown:l
	unknown_0:l
	unknown_1:l
	unknown_2:l
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_830739o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_560_layer_call_and_return_conditional_losses_831250

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ä

*__inference_dense_628_layer_call_fn_834940

inputs
unknown:ll
	unknown_0:l
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_628_layer_call_and_return_conditional_losses_831390o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
È	
ö
E__inference_dense_621_layer_call_and_return_conditional_losses_834187

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ä

*__inference_dense_622_layer_call_fn_834286

inputs
unknown:--
	unknown_0:-
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_622_layer_call_and_return_conditional_losses_831198o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_566_layer_call_fn_835085

inputs
unknown:V
	unknown_0:V
	unknown_1:V
	unknown_2:V
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_831067o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿV: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_834669

inputs/
!batchnorm_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l1
#batchnorm_readvariableop_1_resource:l1
#batchnorm_readvariableop_2_resource:l
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:lz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_559_layer_call_and_return_conditional_losses_834386

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_830856

inputs/
!batchnorm_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l1
#batchnorm_readvariableop_1_resource:l1
#batchnorm_readvariableop_2_resource:l
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:lz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_557_layer_call_and_return_conditional_losses_831154

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_560_layer_call_and_return_conditional_losses_834495

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_558_layer_call_fn_834213

inputs
unknown:-
	unknown_0:-
	unknown_1:-
	unknown_2:-
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_830411o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_563_layer_call_and_return_conditional_losses_831346

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_563_layer_call_and_return_conditional_losses_834822

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
¥´

I__inference_sequential_63_layer_call_and_return_conditional_losses_832118

inputs
normalization_63_sub_y
normalization_63_sqrt_x"
dense_619_831947:-
dense_619_831949:-,
batch_normalization_556_831952:-,
batch_normalization_556_831954:-,
batch_normalization_556_831956:-,
batch_normalization_556_831958:-"
dense_620_831962:--
dense_620_831964:-,
batch_normalization_557_831967:-,
batch_normalization_557_831969:-,
batch_normalization_557_831971:-,
batch_normalization_557_831973:-"
dense_621_831977:--
dense_621_831979:-,
batch_normalization_558_831982:-,
batch_normalization_558_831984:-,
batch_normalization_558_831986:-,
batch_normalization_558_831988:-"
dense_622_831992:--
dense_622_831994:-,
batch_normalization_559_831997:-,
batch_normalization_559_831999:-,
batch_normalization_559_832001:-,
batch_normalization_559_832003:-"
dense_623_832007:--
dense_623_832009:-,
batch_normalization_560_832012:-,
batch_normalization_560_832014:-,
batch_normalization_560_832016:-,
batch_normalization_560_832018:-"
dense_624_832022:-l
dense_624_832024:l,
batch_normalization_561_832027:l,
batch_normalization_561_832029:l,
batch_normalization_561_832031:l,
batch_normalization_561_832033:l"
dense_625_832037:ll
dense_625_832039:l,
batch_normalization_562_832042:l,
batch_normalization_562_832044:l,
batch_normalization_562_832046:l,
batch_normalization_562_832048:l"
dense_626_832052:ll
dense_626_832054:l,
batch_normalization_563_832057:l,
batch_normalization_563_832059:l,
batch_normalization_563_832061:l,
batch_normalization_563_832063:l"
dense_627_832067:ll
dense_627_832069:l,
batch_normalization_564_832072:l,
batch_normalization_564_832074:l,
batch_normalization_564_832076:l,
batch_normalization_564_832078:l"
dense_628_832082:ll
dense_628_832084:l,
batch_normalization_565_832087:l,
batch_normalization_565_832089:l,
batch_normalization_565_832091:l,
batch_normalization_565_832093:l"
dense_629_832097:lV
dense_629_832099:V,
batch_normalization_566_832102:V,
batch_normalization_566_832104:V,
batch_normalization_566_832106:V,
batch_normalization_566_832108:V"
dense_630_832112:V
dense_630_832114:
identity¢/batch_normalization_556/StatefulPartitionedCall¢/batch_normalization_557/StatefulPartitionedCall¢/batch_normalization_558/StatefulPartitionedCall¢/batch_normalization_559/StatefulPartitionedCall¢/batch_normalization_560/StatefulPartitionedCall¢/batch_normalization_561/StatefulPartitionedCall¢/batch_normalization_562/StatefulPartitionedCall¢/batch_normalization_563/StatefulPartitionedCall¢/batch_normalization_564/StatefulPartitionedCall¢/batch_normalization_565/StatefulPartitionedCall¢/batch_normalization_566/StatefulPartitionedCall¢!dense_619/StatefulPartitionedCall¢!dense_620/StatefulPartitionedCall¢!dense_621/StatefulPartitionedCall¢!dense_622/StatefulPartitionedCall¢!dense_623/StatefulPartitionedCall¢!dense_624/StatefulPartitionedCall¢!dense_625/StatefulPartitionedCall¢!dense_626/StatefulPartitionedCall¢!dense_627/StatefulPartitionedCall¢!dense_628/StatefulPartitionedCall¢!dense_629/StatefulPartitionedCall¢!dense_630/StatefulPartitionedCallm
normalization_63/subSubinputsnormalization_63_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_63/SqrtSqrtnormalization_63_sqrt_x*
T0*
_output_shapes

:_
normalization_63/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_63/MaximumMaximumnormalization_63/Sqrt:y:0#normalization_63/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_63/truedivRealDivnormalization_63/sub:z:0normalization_63/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_619/StatefulPartitionedCallStatefulPartitionedCallnormalization_63/truediv:z:0dense_619_831947dense_619_831949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_619_layer_call_and_return_conditional_losses_831102
/batch_normalization_556/StatefulPartitionedCallStatefulPartitionedCall*dense_619/StatefulPartitionedCall:output:0batch_normalization_556_831952batch_normalization_556_831954batch_normalization_556_831956batch_normalization_556_831958*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_830247ø
leaky_re_lu_556/PartitionedCallPartitionedCall8batch_normalization_556/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_556_layer_call_and_return_conditional_losses_831122
!dense_620/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_556/PartitionedCall:output:0dense_620_831962dense_620_831964*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_620_layer_call_and_return_conditional_losses_831134
/batch_normalization_557/StatefulPartitionedCallStatefulPartitionedCall*dense_620/StatefulPartitionedCall:output:0batch_normalization_557_831967batch_normalization_557_831969batch_normalization_557_831971batch_normalization_557_831973*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_830329ø
leaky_re_lu_557/PartitionedCallPartitionedCall8batch_normalization_557/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_557_layer_call_and_return_conditional_losses_831154
!dense_621/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_557/PartitionedCall:output:0dense_621_831977dense_621_831979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_621_layer_call_and_return_conditional_losses_831166
/batch_normalization_558/StatefulPartitionedCallStatefulPartitionedCall*dense_621/StatefulPartitionedCall:output:0batch_normalization_558_831982batch_normalization_558_831984batch_normalization_558_831986batch_normalization_558_831988*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_830411ø
leaky_re_lu_558/PartitionedCallPartitionedCall8batch_normalization_558/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_558_layer_call_and_return_conditional_losses_831186
!dense_622/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_558/PartitionedCall:output:0dense_622_831992dense_622_831994*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_622_layer_call_and_return_conditional_losses_831198
/batch_normalization_559/StatefulPartitionedCallStatefulPartitionedCall*dense_622/StatefulPartitionedCall:output:0batch_normalization_559_831997batch_normalization_559_831999batch_normalization_559_832001batch_normalization_559_832003*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_830493ø
leaky_re_lu_559/PartitionedCallPartitionedCall8batch_normalization_559/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_559_layer_call_and_return_conditional_losses_831218
!dense_623/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_559/PartitionedCall:output:0dense_623_832007dense_623_832009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_623_layer_call_and_return_conditional_losses_831230
/batch_normalization_560/StatefulPartitionedCallStatefulPartitionedCall*dense_623/StatefulPartitionedCall:output:0batch_normalization_560_832012batch_normalization_560_832014batch_normalization_560_832016batch_normalization_560_832018*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_830575ø
leaky_re_lu_560/PartitionedCallPartitionedCall8batch_normalization_560/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_560_layer_call_and_return_conditional_losses_831250
!dense_624/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_560/PartitionedCall:output:0dense_624_832022dense_624_832024*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_624_layer_call_and_return_conditional_losses_831262
/batch_normalization_561/StatefulPartitionedCallStatefulPartitionedCall*dense_624/StatefulPartitionedCall:output:0batch_normalization_561_832027batch_normalization_561_832029batch_normalization_561_832031batch_normalization_561_832033*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_830657ø
leaky_re_lu_561/PartitionedCallPartitionedCall8batch_normalization_561/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_561_layer_call_and_return_conditional_losses_831282
!dense_625/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_561/PartitionedCall:output:0dense_625_832037dense_625_832039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_625_layer_call_and_return_conditional_losses_831294
/batch_normalization_562/StatefulPartitionedCallStatefulPartitionedCall*dense_625/StatefulPartitionedCall:output:0batch_normalization_562_832042batch_normalization_562_832044batch_normalization_562_832046batch_normalization_562_832048*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_830739ø
leaky_re_lu_562/PartitionedCallPartitionedCall8batch_normalization_562/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_562_layer_call_and_return_conditional_losses_831314
!dense_626/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_562/PartitionedCall:output:0dense_626_832052dense_626_832054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_626_layer_call_and_return_conditional_losses_831326
/batch_normalization_563/StatefulPartitionedCallStatefulPartitionedCall*dense_626/StatefulPartitionedCall:output:0batch_normalization_563_832057batch_normalization_563_832059batch_normalization_563_832061batch_normalization_563_832063*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_830821ø
leaky_re_lu_563/PartitionedCallPartitionedCall8batch_normalization_563/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_563_layer_call_and_return_conditional_losses_831346
!dense_627/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_563/PartitionedCall:output:0dense_627_832067dense_627_832069*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_627_layer_call_and_return_conditional_losses_831358
/batch_normalization_564/StatefulPartitionedCallStatefulPartitionedCall*dense_627/StatefulPartitionedCall:output:0batch_normalization_564_832072batch_normalization_564_832074batch_normalization_564_832076batch_normalization_564_832078*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_830903ø
leaky_re_lu_564/PartitionedCallPartitionedCall8batch_normalization_564/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_564_layer_call_and_return_conditional_losses_831378
!dense_628/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_564/PartitionedCall:output:0dense_628_832082dense_628_832084*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_628_layer_call_and_return_conditional_losses_831390
/batch_normalization_565/StatefulPartitionedCallStatefulPartitionedCall*dense_628/StatefulPartitionedCall:output:0batch_normalization_565_832087batch_normalization_565_832089batch_normalization_565_832091batch_normalization_565_832093*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_830985ø
leaky_re_lu_565/PartitionedCallPartitionedCall8batch_normalization_565/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_565_layer_call_and_return_conditional_losses_831410
!dense_629/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_565/PartitionedCall:output:0dense_629_832097dense_629_832099*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_629_layer_call_and_return_conditional_losses_831422
/batch_normalization_566/StatefulPartitionedCallStatefulPartitionedCall*dense_629/StatefulPartitionedCall:output:0batch_normalization_566_832102batch_normalization_566_832104batch_normalization_566_832106batch_normalization_566_832108*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_831067ø
leaky_re_lu_566/PartitionedCallPartitionedCall8batch_normalization_566/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_566_layer_call_and_return_conditional_losses_831442
!dense_630/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_566/PartitionedCall:output:0dense_630_832112dense_630_832114*
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
E__inference_dense_630_layer_call_and_return_conditional_losses_831454y
IdentityIdentity*dense_630/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_556/StatefulPartitionedCall0^batch_normalization_557/StatefulPartitionedCall0^batch_normalization_558/StatefulPartitionedCall0^batch_normalization_559/StatefulPartitionedCall0^batch_normalization_560/StatefulPartitionedCall0^batch_normalization_561/StatefulPartitionedCall0^batch_normalization_562/StatefulPartitionedCall0^batch_normalization_563/StatefulPartitionedCall0^batch_normalization_564/StatefulPartitionedCall0^batch_normalization_565/StatefulPartitionedCall0^batch_normalization_566/StatefulPartitionedCall"^dense_619/StatefulPartitionedCall"^dense_620/StatefulPartitionedCall"^dense_621/StatefulPartitionedCall"^dense_622/StatefulPartitionedCall"^dense_623/StatefulPartitionedCall"^dense_624/StatefulPartitionedCall"^dense_625/StatefulPartitionedCall"^dense_626/StatefulPartitionedCall"^dense_627/StatefulPartitionedCall"^dense_628/StatefulPartitionedCall"^dense_629/StatefulPartitionedCall"^dense_630/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_556/StatefulPartitionedCall/batch_normalization_556/StatefulPartitionedCall2b
/batch_normalization_557/StatefulPartitionedCall/batch_normalization_557/StatefulPartitionedCall2b
/batch_normalization_558/StatefulPartitionedCall/batch_normalization_558/StatefulPartitionedCall2b
/batch_normalization_559/StatefulPartitionedCall/batch_normalization_559/StatefulPartitionedCall2b
/batch_normalization_560/StatefulPartitionedCall/batch_normalization_560/StatefulPartitionedCall2b
/batch_normalization_561/StatefulPartitionedCall/batch_normalization_561/StatefulPartitionedCall2b
/batch_normalization_562/StatefulPartitionedCall/batch_normalization_562/StatefulPartitionedCall2b
/batch_normalization_563/StatefulPartitionedCall/batch_normalization_563/StatefulPartitionedCall2b
/batch_normalization_564/StatefulPartitionedCall/batch_normalization_564/StatefulPartitionedCall2b
/batch_normalization_565/StatefulPartitionedCall/batch_normalization_565/StatefulPartitionedCall2b
/batch_normalization_566/StatefulPartitionedCall/batch_normalization_566/StatefulPartitionedCall2F
!dense_619/StatefulPartitionedCall!dense_619/StatefulPartitionedCall2F
!dense_620/StatefulPartitionedCall!dense_620/StatefulPartitionedCall2F
!dense_621/StatefulPartitionedCall!dense_621/StatefulPartitionedCall2F
!dense_622/StatefulPartitionedCall!dense_622/StatefulPartitionedCall2F
!dense_623/StatefulPartitionedCall!dense_623/StatefulPartitionedCall2F
!dense_624/StatefulPartitionedCall!dense_624/StatefulPartitionedCall2F
!dense_625/StatefulPartitionedCall!dense_625/StatefulPartitionedCall2F
!dense_626/StatefulPartitionedCall!dense_626/StatefulPartitionedCall2F
!dense_627/StatefulPartitionedCall!dense_627/StatefulPartitionedCall2F
!dense_628/StatefulPartitionedCall!dense_628/StatefulPartitionedCall2F
!dense_629/StatefulPartitionedCall!dense_629/StatefulPartitionedCall2F
!dense_630/StatefulPartitionedCall!dense_630/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
%
ì
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_834485

inputs5
'assignmovingavg_readvariableop_resource:-7
)assignmovingavg_1_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-/
!batchnorm_readvariableop_resource:-
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:-
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:-*
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
:-*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:-x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-¬
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
:-*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:-~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-´
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:-v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
²
r
"__inference__traced_restore_836217
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_619_kernel:-/
!assignvariableop_4_dense_619_bias:->
0assignvariableop_5_batch_normalization_556_gamma:-=
/assignvariableop_6_batch_normalization_556_beta:-D
6assignvariableop_7_batch_normalization_556_moving_mean:-H
:assignvariableop_8_batch_normalization_556_moving_variance:-5
#assignvariableop_9_dense_620_kernel:--0
"assignvariableop_10_dense_620_bias:-?
1assignvariableop_11_batch_normalization_557_gamma:->
0assignvariableop_12_batch_normalization_557_beta:-E
7assignvariableop_13_batch_normalization_557_moving_mean:-I
;assignvariableop_14_batch_normalization_557_moving_variance:-6
$assignvariableop_15_dense_621_kernel:--0
"assignvariableop_16_dense_621_bias:-?
1assignvariableop_17_batch_normalization_558_gamma:->
0assignvariableop_18_batch_normalization_558_beta:-E
7assignvariableop_19_batch_normalization_558_moving_mean:-I
;assignvariableop_20_batch_normalization_558_moving_variance:-6
$assignvariableop_21_dense_622_kernel:--0
"assignvariableop_22_dense_622_bias:-?
1assignvariableop_23_batch_normalization_559_gamma:->
0assignvariableop_24_batch_normalization_559_beta:-E
7assignvariableop_25_batch_normalization_559_moving_mean:-I
;assignvariableop_26_batch_normalization_559_moving_variance:-6
$assignvariableop_27_dense_623_kernel:--0
"assignvariableop_28_dense_623_bias:-?
1assignvariableop_29_batch_normalization_560_gamma:->
0assignvariableop_30_batch_normalization_560_beta:-E
7assignvariableop_31_batch_normalization_560_moving_mean:-I
;assignvariableop_32_batch_normalization_560_moving_variance:-6
$assignvariableop_33_dense_624_kernel:-l0
"assignvariableop_34_dense_624_bias:l?
1assignvariableop_35_batch_normalization_561_gamma:l>
0assignvariableop_36_batch_normalization_561_beta:lE
7assignvariableop_37_batch_normalization_561_moving_mean:lI
;assignvariableop_38_batch_normalization_561_moving_variance:l6
$assignvariableop_39_dense_625_kernel:ll0
"assignvariableop_40_dense_625_bias:l?
1assignvariableop_41_batch_normalization_562_gamma:l>
0assignvariableop_42_batch_normalization_562_beta:lE
7assignvariableop_43_batch_normalization_562_moving_mean:lI
;assignvariableop_44_batch_normalization_562_moving_variance:l6
$assignvariableop_45_dense_626_kernel:ll0
"assignvariableop_46_dense_626_bias:l?
1assignvariableop_47_batch_normalization_563_gamma:l>
0assignvariableop_48_batch_normalization_563_beta:lE
7assignvariableop_49_batch_normalization_563_moving_mean:lI
;assignvariableop_50_batch_normalization_563_moving_variance:l6
$assignvariableop_51_dense_627_kernel:ll0
"assignvariableop_52_dense_627_bias:l?
1assignvariableop_53_batch_normalization_564_gamma:l>
0assignvariableop_54_batch_normalization_564_beta:lE
7assignvariableop_55_batch_normalization_564_moving_mean:lI
;assignvariableop_56_batch_normalization_564_moving_variance:l6
$assignvariableop_57_dense_628_kernel:ll0
"assignvariableop_58_dense_628_bias:l?
1assignvariableop_59_batch_normalization_565_gamma:l>
0assignvariableop_60_batch_normalization_565_beta:lE
7assignvariableop_61_batch_normalization_565_moving_mean:lI
;assignvariableop_62_batch_normalization_565_moving_variance:l6
$assignvariableop_63_dense_629_kernel:lV0
"assignvariableop_64_dense_629_bias:V?
1assignvariableop_65_batch_normalization_566_gamma:V>
0assignvariableop_66_batch_normalization_566_beta:VE
7assignvariableop_67_batch_normalization_566_moving_mean:VI
;assignvariableop_68_batch_normalization_566_moving_variance:V6
$assignvariableop_69_dense_630_kernel:V0
"assignvariableop_70_dense_630_bias:'
assignvariableop_71_adam_iter:	 )
assignvariableop_72_adam_beta_1: )
assignvariableop_73_adam_beta_2: (
assignvariableop_74_adam_decay: #
assignvariableop_75_total: %
assignvariableop_76_count_1: =
+assignvariableop_77_adam_dense_619_kernel_m:-7
)assignvariableop_78_adam_dense_619_bias_m:-F
8assignvariableop_79_adam_batch_normalization_556_gamma_m:-E
7assignvariableop_80_adam_batch_normalization_556_beta_m:-=
+assignvariableop_81_adam_dense_620_kernel_m:--7
)assignvariableop_82_adam_dense_620_bias_m:-F
8assignvariableop_83_adam_batch_normalization_557_gamma_m:-E
7assignvariableop_84_adam_batch_normalization_557_beta_m:-=
+assignvariableop_85_adam_dense_621_kernel_m:--7
)assignvariableop_86_adam_dense_621_bias_m:-F
8assignvariableop_87_adam_batch_normalization_558_gamma_m:-E
7assignvariableop_88_adam_batch_normalization_558_beta_m:-=
+assignvariableop_89_adam_dense_622_kernel_m:--7
)assignvariableop_90_adam_dense_622_bias_m:-F
8assignvariableop_91_adam_batch_normalization_559_gamma_m:-E
7assignvariableop_92_adam_batch_normalization_559_beta_m:-=
+assignvariableop_93_adam_dense_623_kernel_m:--7
)assignvariableop_94_adam_dense_623_bias_m:-F
8assignvariableop_95_adam_batch_normalization_560_gamma_m:-E
7assignvariableop_96_adam_batch_normalization_560_beta_m:-=
+assignvariableop_97_adam_dense_624_kernel_m:-l7
)assignvariableop_98_adam_dense_624_bias_m:lF
8assignvariableop_99_adam_batch_normalization_561_gamma_m:lF
8assignvariableop_100_adam_batch_normalization_561_beta_m:l>
,assignvariableop_101_adam_dense_625_kernel_m:ll8
*assignvariableop_102_adam_dense_625_bias_m:lG
9assignvariableop_103_adam_batch_normalization_562_gamma_m:lF
8assignvariableop_104_adam_batch_normalization_562_beta_m:l>
,assignvariableop_105_adam_dense_626_kernel_m:ll8
*assignvariableop_106_adam_dense_626_bias_m:lG
9assignvariableop_107_adam_batch_normalization_563_gamma_m:lF
8assignvariableop_108_adam_batch_normalization_563_beta_m:l>
,assignvariableop_109_adam_dense_627_kernel_m:ll8
*assignvariableop_110_adam_dense_627_bias_m:lG
9assignvariableop_111_adam_batch_normalization_564_gamma_m:lF
8assignvariableop_112_adam_batch_normalization_564_beta_m:l>
,assignvariableop_113_adam_dense_628_kernel_m:ll8
*assignvariableop_114_adam_dense_628_bias_m:lG
9assignvariableop_115_adam_batch_normalization_565_gamma_m:lF
8assignvariableop_116_adam_batch_normalization_565_beta_m:l>
,assignvariableop_117_adam_dense_629_kernel_m:lV8
*assignvariableop_118_adam_dense_629_bias_m:VG
9assignvariableop_119_adam_batch_normalization_566_gamma_m:VF
8assignvariableop_120_adam_batch_normalization_566_beta_m:V>
,assignvariableop_121_adam_dense_630_kernel_m:V8
*assignvariableop_122_adam_dense_630_bias_m:>
,assignvariableop_123_adam_dense_619_kernel_v:-8
*assignvariableop_124_adam_dense_619_bias_v:-G
9assignvariableop_125_adam_batch_normalization_556_gamma_v:-F
8assignvariableop_126_adam_batch_normalization_556_beta_v:->
,assignvariableop_127_adam_dense_620_kernel_v:--8
*assignvariableop_128_adam_dense_620_bias_v:-G
9assignvariableop_129_adam_batch_normalization_557_gamma_v:-F
8assignvariableop_130_adam_batch_normalization_557_beta_v:->
,assignvariableop_131_adam_dense_621_kernel_v:--8
*assignvariableop_132_adam_dense_621_bias_v:-G
9assignvariableop_133_adam_batch_normalization_558_gamma_v:-F
8assignvariableop_134_adam_batch_normalization_558_beta_v:->
,assignvariableop_135_adam_dense_622_kernel_v:--8
*assignvariableop_136_adam_dense_622_bias_v:-G
9assignvariableop_137_adam_batch_normalization_559_gamma_v:-F
8assignvariableop_138_adam_batch_normalization_559_beta_v:->
,assignvariableop_139_adam_dense_623_kernel_v:--8
*assignvariableop_140_adam_dense_623_bias_v:-G
9assignvariableop_141_adam_batch_normalization_560_gamma_v:-F
8assignvariableop_142_adam_batch_normalization_560_beta_v:->
,assignvariableop_143_adam_dense_624_kernel_v:-l8
*assignvariableop_144_adam_dense_624_bias_v:lG
9assignvariableop_145_adam_batch_normalization_561_gamma_v:lF
8assignvariableop_146_adam_batch_normalization_561_beta_v:l>
,assignvariableop_147_adam_dense_625_kernel_v:ll8
*assignvariableop_148_adam_dense_625_bias_v:lG
9assignvariableop_149_adam_batch_normalization_562_gamma_v:lF
8assignvariableop_150_adam_batch_normalization_562_beta_v:l>
,assignvariableop_151_adam_dense_626_kernel_v:ll8
*assignvariableop_152_adam_dense_626_bias_v:lG
9assignvariableop_153_adam_batch_normalization_563_gamma_v:lF
8assignvariableop_154_adam_batch_normalization_563_beta_v:l>
,assignvariableop_155_adam_dense_627_kernel_v:ll8
*assignvariableop_156_adam_dense_627_bias_v:lG
9assignvariableop_157_adam_batch_normalization_564_gamma_v:lF
8assignvariableop_158_adam_batch_normalization_564_beta_v:l>
,assignvariableop_159_adam_dense_628_kernel_v:ll8
*assignvariableop_160_adam_dense_628_bias_v:lG
9assignvariableop_161_adam_batch_normalization_565_gamma_v:lF
8assignvariableop_162_adam_batch_normalization_565_beta_v:l>
,assignvariableop_163_adam_dense_629_kernel_v:lV8
*assignvariableop_164_adam_dense_629_bias_v:VG
9assignvariableop_165_adam_batch_normalization_566_gamma_v:VF
8assignvariableop_166_adam_batch_normalization_566_beta_v:V>
,assignvariableop_167_adam_dense_630_kernel_v:V8
*assignvariableop_168_adam_dense_630_bias_v:
identity_170¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_111¢AssignVariableOp_112¢AssignVariableOp_113¢AssignVariableOp_114¢AssignVariableOp_115¢AssignVariableOp_116¢AssignVariableOp_117¢AssignVariableOp_118¢AssignVariableOp_119¢AssignVariableOp_12¢AssignVariableOp_120¢AssignVariableOp_121¢AssignVariableOp_122¢AssignVariableOp_123¢AssignVariableOp_124¢AssignVariableOp_125¢AssignVariableOp_126¢AssignVariableOp_127¢AssignVariableOp_128¢AssignVariableOp_129¢AssignVariableOp_13¢AssignVariableOp_130¢AssignVariableOp_131¢AssignVariableOp_132¢AssignVariableOp_133¢AssignVariableOp_134¢AssignVariableOp_135¢AssignVariableOp_136¢AssignVariableOp_137¢AssignVariableOp_138¢AssignVariableOp_139¢AssignVariableOp_14¢AssignVariableOp_140¢AssignVariableOp_141¢AssignVariableOp_142¢AssignVariableOp_143¢AssignVariableOp_144¢AssignVariableOp_145¢AssignVariableOp_146¢AssignVariableOp_147¢AssignVariableOp_148¢AssignVariableOp_149¢AssignVariableOp_15¢AssignVariableOp_150¢AssignVariableOp_151¢AssignVariableOp_152¢AssignVariableOp_153¢AssignVariableOp_154¢AssignVariableOp_155¢AssignVariableOp_156¢AssignVariableOp_157¢AssignVariableOp_158¢AssignVariableOp_159¢AssignVariableOp_16¢AssignVariableOp_160¢AssignVariableOp_161¢AssignVariableOp_162¢AssignVariableOp_163¢AssignVariableOp_164¢AssignVariableOp_165¢AssignVariableOp_166¢AssignVariableOp_167¢AssignVariableOp_168¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99´_
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:ª*
dtype0*Ù^
valueÏ^BÌ^ªB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-22/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-22/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-22/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÉ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:ª*
dtype0*ê
valueàBÝªB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ÷
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¾
_output_shapes«
¨::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*»
dtypes°
­2ª		[
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_619_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_619_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batch_normalization_556_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_556_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_556_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_8AssignVariableOp:assignvariableop_8_batch_normalization_556_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_620_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_620_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batch_normalization_557_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_557_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_557_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_14AssignVariableOp;assignvariableop_14_batch_normalization_557_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_621_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_621_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batch_normalization_558_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_558_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_558_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_20AssignVariableOp;assignvariableop_20_batch_normalization_558_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_622_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_622_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_23AssignVariableOp1assignvariableop_23_batch_normalization_559_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_559_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_25AssignVariableOp7assignvariableop_25_batch_normalization_559_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_26AssignVariableOp;assignvariableop_26_batch_normalization_559_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_623_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_623_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_29AssignVariableOp1assignvariableop_29_batch_normalization_560_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_560_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_31AssignVariableOp7assignvariableop_31_batch_normalization_560_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_32AssignVariableOp;assignvariableop_32_batch_normalization_560_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_624_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_624_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_35AssignVariableOp1assignvariableop_35_batch_normalization_561_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_561_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_37AssignVariableOp7assignvariableop_37_batch_normalization_561_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_38AssignVariableOp;assignvariableop_38_batch_normalization_561_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp$assignvariableop_39_dense_625_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_625_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_41AssignVariableOp1assignvariableop_41_batch_normalization_562_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_batch_normalization_562_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_43AssignVariableOp7assignvariableop_43_batch_normalization_562_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_44AssignVariableOp;assignvariableop_44_batch_normalization_562_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp$assignvariableop_45_dense_626_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_626_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_47AssignVariableOp1assignvariableop_47_batch_normalization_563_gammaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_48AssignVariableOp0assignvariableop_48_batch_normalization_563_betaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_49AssignVariableOp7assignvariableop_49_batch_normalization_563_moving_meanIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_50AssignVariableOp;assignvariableop_50_batch_normalization_563_moving_varianceIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp$assignvariableop_51_dense_627_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp"assignvariableop_52_dense_627_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_53AssignVariableOp1assignvariableop_53_batch_normalization_564_gammaIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_54AssignVariableOp0assignvariableop_54_batch_normalization_564_betaIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_55AssignVariableOp7assignvariableop_55_batch_normalization_564_moving_meanIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_56AssignVariableOp;assignvariableop_56_batch_normalization_564_moving_varianceIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp$assignvariableop_57_dense_628_kernelIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp"assignvariableop_58_dense_628_biasIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_59AssignVariableOp1assignvariableop_59_batch_normalization_565_gammaIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_60AssignVariableOp0assignvariableop_60_batch_normalization_565_betaIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_61AssignVariableOp7assignvariableop_61_batch_normalization_565_moving_meanIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_62AssignVariableOp;assignvariableop_62_batch_normalization_565_moving_varianceIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp$assignvariableop_63_dense_629_kernelIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp"assignvariableop_64_dense_629_biasIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_65AssignVariableOp1assignvariableop_65_batch_normalization_566_gammaIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_66AssignVariableOp0assignvariableop_66_batch_normalization_566_betaIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_67AssignVariableOp7assignvariableop_67_batch_normalization_566_moving_meanIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_68AssignVariableOp;assignvariableop_68_batch_normalization_566_moving_varianceIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp$assignvariableop_69_dense_630_kernelIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp"assignvariableop_70_dense_630_biasIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_71AssignVariableOpassignvariableop_71_adam_iterIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOpassignvariableop_72_adam_beta_1Identity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOpassignvariableop_73_adam_beta_2Identity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_74AssignVariableOpassignvariableop_74_adam_decayIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_75AssignVariableOpassignvariableop_75_totalIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_76AssignVariableOpassignvariableop_76_count_1Identity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_dense_619_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_dense_619_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_79AssignVariableOp8assignvariableop_79_adam_batch_normalization_556_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_80AssignVariableOp7assignvariableop_80_adam_batch_normalization_556_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_dense_620_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_dense_620_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_83AssignVariableOp8assignvariableop_83_adam_batch_normalization_557_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_84AssignVariableOp7assignvariableop_84_adam_batch_normalization_557_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_dense_621_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_dense_621_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_87AssignVariableOp8assignvariableop_87_adam_batch_normalization_558_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_88AssignVariableOp7assignvariableop_88_adam_batch_normalization_558_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_dense_622_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_dense_622_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_91AssignVariableOp8assignvariableop_91_adam_batch_normalization_559_gamma_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_92AssignVariableOp7assignvariableop_92_adam_batch_normalization_559_beta_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_dense_623_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_dense_623_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_95AssignVariableOp8assignvariableop_95_adam_batch_normalization_560_gamma_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_96AssignVariableOp7assignvariableop_96_adam_batch_normalization_560_beta_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_dense_624_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_dense_624_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_99AssignVariableOp8assignvariableop_99_adam_batch_normalization_561_gamma_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_100AssignVariableOp8assignvariableop_100_adam_batch_normalization_561_beta_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_dense_625_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_dense_625_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_103AssignVariableOp9assignvariableop_103_adam_batch_normalization_562_gamma_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_104AssignVariableOp8assignvariableop_104_adam_batch_normalization_562_beta_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_dense_626_kernel_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_dense_626_bias_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_107AssignVariableOp9assignvariableop_107_adam_batch_normalization_563_gamma_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_108AssignVariableOp8assignvariableop_108_adam_batch_normalization_563_beta_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_dense_627_kernel_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_dense_627_bias_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_111AssignVariableOp9assignvariableop_111_adam_batch_normalization_564_gamma_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_112AssignVariableOp8assignvariableop_112_adam_batch_normalization_564_beta_mIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_dense_628_kernel_mIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_dense_628_bias_mIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_115AssignVariableOp9assignvariableop_115_adam_batch_normalization_565_gamma_mIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_116AssignVariableOp8assignvariableop_116_adam_batch_normalization_565_beta_mIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_dense_629_kernel_mIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_dense_629_bias_mIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_119AssignVariableOp9assignvariableop_119_adam_batch_normalization_566_gamma_mIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_120AssignVariableOp8assignvariableop_120_adam_batch_normalization_566_beta_mIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_121AssignVariableOp,assignvariableop_121_adam_dense_630_kernel_mIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_122AssignVariableOp*assignvariableop_122_adam_dense_630_bias_mIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_123AssignVariableOp,assignvariableop_123_adam_dense_619_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_124AssignVariableOp*assignvariableop_124_adam_dense_619_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_125AssignVariableOp9assignvariableop_125_adam_batch_normalization_556_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_126AssignVariableOp8assignvariableop_126_adam_batch_normalization_556_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_127AssignVariableOp,assignvariableop_127_adam_dense_620_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_128AssignVariableOp*assignvariableop_128_adam_dense_620_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_129AssignVariableOp9assignvariableop_129_adam_batch_normalization_557_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_130AssignVariableOp8assignvariableop_130_adam_batch_normalization_557_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_131AssignVariableOp,assignvariableop_131_adam_dense_621_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_132AssignVariableOp*assignvariableop_132_adam_dense_621_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_133AssignVariableOp9assignvariableop_133_adam_batch_normalization_558_gamma_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_134AssignVariableOp8assignvariableop_134_adam_batch_normalization_558_beta_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_135AssignVariableOp,assignvariableop_135_adam_dense_622_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_136AssignVariableOp*assignvariableop_136_adam_dense_622_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_137AssignVariableOp9assignvariableop_137_adam_batch_normalization_559_gamma_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_138AssignVariableOp8assignvariableop_138_adam_batch_normalization_559_beta_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_139AssignVariableOp,assignvariableop_139_adam_dense_623_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_140AssignVariableOp*assignvariableop_140_adam_dense_623_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_141AssignVariableOp9assignvariableop_141_adam_batch_normalization_560_gamma_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_142AssignVariableOp8assignvariableop_142_adam_batch_normalization_560_beta_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_143AssignVariableOp,assignvariableop_143_adam_dense_624_kernel_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_144AssignVariableOp*assignvariableop_144_adam_dense_624_bias_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_145AssignVariableOp9assignvariableop_145_adam_batch_normalization_561_gamma_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_146AssignVariableOp8assignvariableop_146_adam_batch_normalization_561_beta_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_147AssignVariableOp,assignvariableop_147_adam_dense_625_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_148AssignVariableOp*assignvariableop_148_adam_dense_625_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_149AssignVariableOp9assignvariableop_149_adam_batch_normalization_562_gamma_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_150AssignVariableOp8assignvariableop_150_adam_batch_normalization_562_beta_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_151AssignVariableOp,assignvariableop_151_adam_dense_626_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_152AssignVariableOp*assignvariableop_152_adam_dense_626_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_153AssignVariableOp9assignvariableop_153_adam_batch_normalization_563_gamma_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_154AssignVariableOp8assignvariableop_154_adam_batch_normalization_563_beta_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_155AssignVariableOp,assignvariableop_155_adam_dense_627_kernel_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_156AssignVariableOp*assignvariableop_156_adam_dense_627_bias_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_157AssignVariableOp9assignvariableop_157_adam_batch_normalization_564_gamma_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_158AssignVariableOp8assignvariableop_158_adam_batch_normalization_564_beta_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_159AssignVariableOp,assignvariableop_159_adam_dense_628_kernel_vIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_160AssignVariableOp*assignvariableop_160_adam_dense_628_bias_vIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_161AssignVariableOp9assignvariableop_161_adam_batch_normalization_565_gamma_vIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_162AssignVariableOp8assignvariableop_162_adam_batch_normalization_565_beta_vIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_163AssignVariableOp,assignvariableop_163_adam_dense_629_kernel_vIdentity_163:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_164AssignVariableOp*assignvariableop_164_adam_dense_629_bias_vIdentity_164:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_165IdentityRestoreV2:tensors:165"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_165AssignVariableOp9assignvariableop_165_adam_batch_normalization_566_gamma_vIdentity_165:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_166IdentityRestoreV2:tensors:166"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_166AssignVariableOp8assignvariableop_166_adam_batch_normalization_566_beta_vIdentity_166:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_167IdentityRestoreV2:tensors:167"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_167AssignVariableOp,assignvariableop_167_adam_dense_630_kernel_vIdentity_167:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_168IdentityRestoreV2:tensors:168"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_168AssignVariableOp*assignvariableop_168_adam_dense_630_bias_vIdentity_168:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_169Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_170IdentityIdentity_169:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_170Identity_170:output:0*é
_input_shapes×
Ô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632,
AssignVariableOp_164AssignVariableOp_1642,
AssignVariableOp_165AssignVariableOp_1652,
AssignVariableOp_166AssignVariableOp_1662,
AssignVariableOp_167AssignVariableOp_1672,
AssignVariableOp_168AssignVariableOp_1682*
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
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_834233

inputs/
!batchnorm_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-1
#batchnorm_readvariableop_1_resource:-1
#batchnorm_readvariableop_2_resource:-
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:-z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
È	
ö
E__inference_dense_628_layer_call_and_return_conditional_losses_831390

inputs0
matmul_readvariableop_resource:ll-
biasadd_readvariableop_resource:l
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ll*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:l*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
ë´

I__inference_sequential_63_layer_call_and_return_conditional_losses_832587
normalization_63_input
normalization_63_sub_y
normalization_63_sqrt_x"
dense_619_832416:-
dense_619_832418:-,
batch_normalization_556_832421:-,
batch_normalization_556_832423:-,
batch_normalization_556_832425:-,
batch_normalization_556_832427:-"
dense_620_832431:--
dense_620_832433:-,
batch_normalization_557_832436:-,
batch_normalization_557_832438:-,
batch_normalization_557_832440:-,
batch_normalization_557_832442:-"
dense_621_832446:--
dense_621_832448:-,
batch_normalization_558_832451:-,
batch_normalization_558_832453:-,
batch_normalization_558_832455:-,
batch_normalization_558_832457:-"
dense_622_832461:--
dense_622_832463:-,
batch_normalization_559_832466:-,
batch_normalization_559_832468:-,
batch_normalization_559_832470:-,
batch_normalization_559_832472:-"
dense_623_832476:--
dense_623_832478:-,
batch_normalization_560_832481:-,
batch_normalization_560_832483:-,
batch_normalization_560_832485:-,
batch_normalization_560_832487:-"
dense_624_832491:-l
dense_624_832493:l,
batch_normalization_561_832496:l,
batch_normalization_561_832498:l,
batch_normalization_561_832500:l,
batch_normalization_561_832502:l"
dense_625_832506:ll
dense_625_832508:l,
batch_normalization_562_832511:l,
batch_normalization_562_832513:l,
batch_normalization_562_832515:l,
batch_normalization_562_832517:l"
dense_626_832521:ll
dense_626_832523:l,
batch_normalization_563_832526:l,
batch_normalization_563_832528:l,
batch_normalization_563_832530:l,
batch_normalization_563_832532:l"
dense_627_832536:ll
dense_627_832538:l,
batch_normalization_564_832541:l,
batch_normalization_564_832543:l,
batch_normalization_564_832545:l,
batch_normalization_564_832547:l"
dense_628_832551:ll
dense_628_832553:l,
batch_normalization_565_832556:l,
batch_normalization_565_832558:l,
batch_normalization_565_832560:l,
batch_normalization_565_832562:l"
dense_629_832566:lV
dense_629_832568:V,
batch_normalization_566_832571:V,
batch_normalization_566_832573:V,
batch_normalization_566_832575:V,
batch_normalization_566_832577:V"
dense_630_832581:V
dense_630_832583:
identity¢/batch_normalization_556/StatefulPartitionedCall¢/batch_normalization_557/StatefulPartitionedCall¢/batch_normalization_558/StatefulPartitionedCall¢/batch_normalization_559/StatefulPartitionedCall¢/batch_normalization_560/StatefulPartitionedCall¢/batch_normalization_561/StatefulPartitionedCall¢/batch_normalization_562/StatefulPartitionedCall¢/batch_normalization_563/StatefulPartitionedCall¢/batch_normalization_564/StatefulPartitionedCall¢/batch_normalization_565/StatefulPartitionedCall¢/batch_normalization_566/StatefulPartitionedCall¢!dense_619/StatefulPartitionedCall¢!dense_620/StatefulPartitionedCall¢!dense_621/StatefulPartitionedCall¢!dense_622/StatefulPartitionedCall¢!dense_623/StatefulPartitionedCall¢!dense_624/StatefulPartitionedCall¢!dense_625/StatefulPartitionedCall¢!dense_626/StatefulPartitionedCall¢!dense_627/StatefulPartitionedCall¢!dense_628/StatefulPartitionedCall¢!dense_629/StatefulPartitionedCall¢!dense_630/StatefulPartitionedCall}
normalization_63/subSubnormalization_63_inputnormalization_63_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_63/SqrtSqrtnormalization_63_sqrt_x*
T0*
_output_shapes

:_
normalization_63/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_63/MaximumMaximumnormalization_63/Sqrt:y:0#normalization_63/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_63/truedivRealDivnormalization_63/sub:z:0normalization_63/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_619/StatefulPartitionedCallStatefulPartitionedCallnormalization_63/truediv:z:0dense_619_832416dense_619_832418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_619_layer_call_and_return_conditional_losses_831102
/batch_normalization_556/StatefulPartitionedCallStatefulPartitionedCall*dense_619/StatefulPartitionedCall:output:0batch_normalization_556_832421batch_normalization_556_832423batch_normalization_556_832425batch_normalization_556_832427*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_830200ø
leaky_re_lu_556/PartitionedCallPartitionedCall8batch_normalization_556/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_556_layer_call_and_return_conditional_losses_831122
!dense_620/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_556/PartitionedCall:output:0dense_620_832431dense_620_832433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_620_layer_call_and_return_conditional_losses_831134
/batch_normalization_557/StatefulPartitionedCallStatefulPartitionedCall*dense_620/StatefulPartitionedCall:output:0batch_normalization_557_832436batch_normalization_557_832438batch_normalization_557_832440batch_normalization_557_832442*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_830282ø
leaky_re_lu_557/PartitionedCallPartitionedCall8batch_normalization_557/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_557_layer_call_and_return_conditional_losses_831154
!dense_621/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_557/PartitionedCall:output:0dense_621_832446dense_621_832448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_621_layer_call_and_return_conditional_losses_831166
/batch_normalization_558/StatefulPartitionedCallStatefulPartitionedCall*dense_621/StatefulPartitionedCall:output:0batch_normalization_558_832451batch_normalization_558_832453batch_normalization_558_832455batch_normalization_558_832457*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_830364ø
leaky_re_lu_558/PartitionedCallPartitionedCall8batch_normalization_558/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_558_layer_call_and_return_conditional_losses_831186
!dense_622/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_558/PartitionedCall:output:0dense_622_832461dense_622_832463*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_622_layer_call_and_return_conditional_losses_831198
/batch_normalization_559/StatefulPartitionedCallStatefulPartitionedCall*dense_622/StatefulPartitionedCall:output:0batch_normalization_559_832466batch_normalization_559_832468batch_normalization_559_832470batch_normalization_559_832472*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_830446ø
leaky_re_lu_559/PartitionedCallPartitionedCall8batch_normalization_559/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_559_layer_call_and_return_conditional_losses_831218
!dense_623/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_559/PartitionedCall:output:0dense_623_832476dense_623_832478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_623_layer_call_and_return_conditional_losses_831230
/batch_normalization_560/StatefulPartitionedCallStatefulPartitionedCall*dense_623/StatefulPartitionedCall:output:0batch_normalization_560_832481batch_normalization_560_832483batch_normalization_560_832485batch_normalization_560_832487*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_830528ø
leaky_re_lu_560/PartitionedCallPartitionedCall8batch_normalization_560/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_560_layer_call_and_return_conditional_losses_831250
!dense_624/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_560/PartitionedCall:output:0dense_624_832491dense_624_832493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_624_layer_call_and_return_conditional_losses_831262
/batch_normalization_561/StatefulPartitionedCallStatefulPartitionedCall*dense_624/StatefulPartitionedCall:output:0batch_normalization_561_832496batch_normalization_561_832498batch_normalization_561_832500batch_normalization_561_832502*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_830610ø
leaky_re_lu_561/PartitionedCallPartitionedCall8batch_normalization_561/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_561_layer_call_and_return_conditional_losses_831282
!dense_625/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_561/PartitionedCall:output:0dense_625_832506dense_625_832508*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_625_layer_call_and_return_conditional_losses_831294
/batch_normalization_562/StatefulPartitionedCallStatefulPartitionedCall*dense_625/StatefulPartitionedCall:output:0batch_normalization_562_832511batch_normalization_562_832513batch_normalization_562_832515batch_normalization_562_832517*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_830692ø
leaky_re_lu_562/PartitionedCallPartitionedCall8batch_normalization_562/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_562_layer_call_and_return_conditional_losses_831314
!dense_626/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_562/PartitionedCall:output:0dense_626_832521dense_626_832523*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_626_layer_call_and_return_conditional_losses_831326
/batch_normalization_563/StatefulPartitionedCallStatefulPartitionedCall*dense_626/StatefulPartitionedCall:output:0batch_normalization_563_832526batch_normalization_563_832528batch_normalization_563_832530batch_normalization_563_832532*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_830774ø
leaky_re_lu_563/PartitionedCallPartitionedCall8batch_normalization_563/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_563_layer_call_and_return_conditional_losses_831346
!dense_627/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_563/PartitionedCall:output:0dense_627_832536dense_627_832538*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_627_layer_call_and_return_conditional_losses_831358
/batch_normalization_564/StatefulPartitionedCallStatefulPartitionedCall*dense_627/StatefulPartitionedCall:output:0batch_normalization_564_832541batch_normalization_564_832543batch_normalization_564_832545batch_normalization_564_832547*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_830856ø
leaky_re_lu_564/PartitionedCallPartitionedCall8batch_normalization_564/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_564_layer_call_and_return_conditional_losses_831378
!dense_628/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_564/PartitionedCall:output:0dense_628_832551dense_628_832553*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_628_layer_call_and_return_conditional_losses_831390
/batch_normalization_565/StatefulPartitionedCallStatefulPartitionedCall*dense_628/StatefulPartitionedCall:output:0batch_normalization_565_832556batch_normalization_565_832558batch_normalization_565_832560batch_normalization_565_832562*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_830938ø
leaky_re_lu_565/PartitionedCallPartitionedCall8batch_normalization_565/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_565_layer_call_and_return_conditional_losses_831410
!dense_629/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_565/PartitionedCall:output:0dense_629_832566dense_629_832568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_629_layer_call_and_return_conditional_losses_831422
/batch_normalization_566/StatefulPartitionedCallStatefulPartitionedCall*dense_629/StatefulPartitionedCall:output:0batch_normalization_566_832571batch_normalization_566_832573batch_normalization_566_832575batch_normalization_566_832577*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_831020ø
leaky_re_lu_566/PartitionedCallPartitionedCall8batch_normalization_566/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_566_layer_call_and_return_conditional_losses_831442
!dense_630/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_566/PartitionedCall:output:0dense_630_832581dense_630_832583*
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
E__inference_dense_630_layer_call_and_return_conditional_losses_831454y
IdentityIdentity*dense_630/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_556/StatefulPartitionedCall0^batch_normalization_557/StatefulPartitionedCall0^batch_normalization_558/StatefulPartitionedCall0^batch_normalization_559/StatefulPartitionedCall0^batch_normalization_560/StatefulPartitionedCall0^batch_normalization_561/StatefulPartitionedCall0^batch_normalization_562/StatefulPartitionedCall0^batch_normalization_563/StatefulPartitionedCall0^batch_normalization_564/StatefulPartitionedCall0^batch_normalization_565/StatefulPartitionedCall0^batch_normalization_566/StatefulPartitionedCall"^dense_619/StatefulPartitionedCall"^dense_620/StatefulPartitionedCall"^dense_621/StatefulPartitionedCall"^dense_622/StatefulPartitionedCall"^dense_623/StatefulPartitionedCall"^dense_624/StatefulPartitionedCall"^dense_625/StatefulPartitionedCall"^dense_626/StatefulPartitionedCall"^dense_627/StatefulPartitionedCall"^dense_628/StatefulPartitionedCall"^dense_629/StatefulPartitionedCall"^dense_630/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_556/StatefulPartitionedCall/batch_normalization_556/StatefulPartitionedCall2b
/batch_normalization_557/StatefulPartitionedCall/batch_normalization_557/StatefulPartitionedCall2b
/batch_normalization_558/StatefulPartitionedCall/batch_normalization_558/StatefulPartitionedCall2b
/batch_normalization_559/StatefulPartitionedCall/batch_normalization_559/StatefulPartitionedCall2b
/batch_normalization_560/StatefulPartitionedCall/batch_normalization_560/StatefulPartitionedCall2b
/batch_normalization_561/StatefulPartitionedCall/batch_normalization_561/StatefulPartitionedCall2b
/batch_normalization_562/StatefulPartitionedCall/batch_normalization_562/StatefulPartitionedCall2b
/batch_normalization_563/StatefulPartitionedCall/batch_normalization_563/StatefulPartitionedCall2b
/batch_normalization_564/StatefulPartitionedCall/batch_normalization_564/StatefulPartitionedCall2b
/batch_normalization_565/StatefulPartitionedCall/batch_normalization_565/StatefulPartitionedCall2b
/batch_normalization_566/StatefulPartitionedCall/batch_normalization_566/StatefulPartitionedCall2F
!dense_619/StatefulPartitionedCall!dense_619/StatefulPartitionedCall2F
!dense_620/StatefulPartitionedCall!dense_620/StatefulPartitionedCall2F
!dense_621/StatefulPartitionedCall!dense_621/StatefulPartitionedCall2F
!dense_622/StatefulPartitionedCall!dense_622/StatefulPartitionedCall2F
!dense_623/StatefulPartitionedCall!dense_623/StatefulPartitionedCall2F
!dense_624/StatefulPartitionedCall!dense_624/StatefulPartitionedCall2F
!dense_625/StatefulPartitionedCall!dense_625/StatefulPartitionedCall2F
!dense_626/StatefulPartitionedCall!dense_626/StatefulPartitionedCall2F
!dense_627/StatefulPartitionedCall!dense_627/StatefulPartitionedCall2F
!dense_628/StatefulPartitionedCall!dense_628/StatefulPartitionedCall2F
!dense_629/StatefulPartitionedCall!dense_629/StatefulPartitionedCall2F
!dense_630/StatefulPartitionedCall!dense_630/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_63_input:$ 

_output_shapes

::$ 

_output_shapes

:
ª
Ó
8__inference_batch_normalization_559_layer_call_fn_834322

inputs
unknown:-
	unknown_0:-
	unknown_1:-
	unknown_2:-
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_830493o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_558_layer_call_and_return_conditional_losses_831186

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_563_layer_call_fn_834817

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
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_563_layer_call_and_return_conditional_losses_831346`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
È	
ö
E__inference_dense_623_layer_call_and_return_conditional_losses_834405

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_561_layer_call_fn_834527

inputs
unknown:l
	unknown_0:l
	unknown_1:l
	unknown_2:l
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_830610o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
È	
ö
E__inference_dense_625_layer_call_and_return_conditional_losses_831294

inputs0
matmul_readvariableop_resource:ll-
biasadd_readvariableop_resource:l
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ll*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:l*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_834594

inputs5
'assignmovingavg_readvariableop_resource:l7
)assignmovingavg_1_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l/
!batchnorm_readvariableop_resource:l
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:l
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿll
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:l*
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
:l*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:lx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l¬
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
:l*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:l~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l´
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:lv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_834158

inputs5
'assignmovingavg_readvariableop_resource:-7
)assignmovingavg_1_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-/
!batchnorm_readvariableop_resource:-
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:-
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:-*
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
:-*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:-x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-¬
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
:-*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:-~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-´
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:-v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
È	
ö
E__inference_dense_625_layer_call_and_return_conditional_losses_834623

inputs0
matmul_readvariableop_resource:ll-
biasadd_readvariableop_resource:l
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ll*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:l*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
È	
ö
E__inference_dense_623_layer_call_and_return_conditional_losses_831230

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_834015

inputs/
!batchnorm_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-1
#batchnorm_readvariableop_1_resource:-1
#batchnorm_readvariableop_2_resource:-
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:-z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_564_layer_call_and_return_conditional_losses_834931

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_830411

inputs5
'assignmovingavg_readvariableop_resource:-7
)assignmovingavg_1_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-/
!batchnorm_readvariableop_resource:-
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:-
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:-*
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
:-*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:-x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-¬
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
:-*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:-~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-´
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:-v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_831067

inputs5
'assignmovingavg_readvariableop_resource:V7
)assignmovingavg_1_readvariableop_resource:V3
%batchnorm_mul_readvariableop_resource:V/
!batchnorm_readvariableop_resource:V
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:V*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:V
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:V*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:V*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:V*
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
:V*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Vx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:V¬
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
:V*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:V~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:V´
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
:VP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:V~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:V*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Vc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Vv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:V*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Vr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿV: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_834124

inputs/
!batchnorm_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-1
#batchnorm_readvariableop_1_resource:-1
#batchnorm_readvariableop_2_resource:-
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:-z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
È	
ö
E__inference_dense_627_layer_call_and_return_conditional_losses_831358

inputs0
matmul_readvariableop_resource:ll-
biasadd_readvariableop_resource:l
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ll*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:l*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
ñ
¢
.__inference_sequential_63_layer_call_fn_832406
normalization_63_input
unknown
	unknown_0
	unknown_1:-
	unknown_2:-
	unknown_3:-
	unknown_4:-
	unknown_5:-
	unknown_6:-
	unknown_7:--
	unknown_8:-
	unknown_9:-

unknown_10:-

unknown_11:-

unknown_12:-

unknown_13:--

unknown_14:-

unknown_15:-

unknown_16:-

unknown_17:-

unknown_18:-

unknown_19:--

unknown_20:-

unknown_21:-

unknown_22:-

unknown_23:-

unknown_24:-

unknown_25:--

unknown_26:-

unknown_27:-

unknown_28:-

unknown_29:-

unknown_30:-

unknown_31:-l

unknown_32:l

unknown_33:l

unknown_34:l

unknown_35:l

unknown_36:l

unknown_37:ll

unknown_38:l

unknown_39:l

unknown_40:l

unknown_41:l

unknown_42:l

unknown_43:ll

unknown_44:l

unknown_45:l

unknown_46:l

unknown_47:l

unknown_48:l

unknown_49:ll

unknown_50:l

unknown_51:l

unknown_52:l

unknown_53:l

unknown_54:l

unknown_55:ll

unknown_56:l

unknown_57:l

unknown_58:l

unknown_59:l

unknown_60:l

unknown_61:lV

unknown_62:V

unknown_63:V

unknown_64:V

unknown_65:V

unknown_66:V

unknown_67:V

unknown_68:
identity¢StatefulPartitionedCall

StatefulPartitionedCallStatefulPartitionedCallnormalization_63_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*P
_read_only_resource_inputs2
0.	
 !"%&'(+,-.1234789:=>?@CDEF*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_63_layer_call_and_return_conditional_losses_832118o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_63_input:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_565_layer_call_and_return_conditional_losses_831410

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_557_layer_call_fn_834163

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
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_557_layer_call_and_return_conditional_losses_831154`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_556_layer_call_fn_834054

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
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_556_layer_call_and_return_conditional_losses_831122`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_562_layer_call_fn_834636

inputs
unknown:l
	unknown_0:l
	unknown_1:l
	unknown_2:l
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_830692o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_566_layer_call_and_return_conditional_losses_831442

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿV:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 
_user_specified_nameinputs
Ú§
çG
I__inference_sequential_63_layer_call_and_return_conditional_losses_833756

inputs
normalization_63_sub_y
normalization_63_sqrt_x:
(dense_619_matmul_readvariableop_resource:-7
)dense_619_biasadd_readvariableop_resource:-M
?batch_normalization_556_assignmovingavg_readvariableop_resource:-O
Abatch_normalization_556_assignmovingavg_1_readvariableop_resource:-K
=batch_normalization_556_batchnorm_mul_readvariableop_resource:-G
9batch_normalization_556_batchnorm_readvariableop_resource:-:
(dense_620_matmul_readvariableop_resource:--7
)dense_620_biasadd_readvariableop_resource:-M
?batch_normalization_557_assignmovingavg_readvariableop_resource:-O
Abatch_normalization_557_assignmovingavg_1_readvariableop_resource:-K
=batch_normalization_557_batchnorm_mul_readvariableop_resource:-G
9batch_normalization_557_batchnorm_readvariableop_resource:-:
(dense_621_matmul_readvariableop_resource:--7
)dense_621_biasadd_readvariableop_resource:-M
?batch_normalization_558_assignmovingavg_readvariableop_resource:-O
Abatch_normalization_558_assignmovingavg_1_readvariableop_resource:-K
=batch_normalization_558_batchnorm_mul_readvariableop_resource:-G
9batch_normalization_558_batchnorm_readvariableop_resource:-:
(dense_622_matmul_readvariableop_resource:--7
)dense_622_biasadd_readvariableop_resource:-M
?batch_normalization_559_assignmovingavg_readvariableop_resource:-O
Abatch_normalization_559_assignmovingavg_1_readvariableop_resource:-K
=batch_normalization_559_batchnorm_mul_readvariableop_resource:-G
9batch_normalization_559_batchnorm_readvariableop_resource:-:
(dense_623_matmul_readvariableop_resource:--7
)dense_623_biasadd_readvariableop_resource:-M
?batch_normalization_560_assignmovingavg_readvariableop_resource:-O
Abatch_normalization_560_assignmovingavg_1_readvariableop_resource:-K
=batch_normalization_560_batchnorm_mul_readvariableop_resource:-G
9batch_normalization_560_batchnorm_readvariableop_resource:-:
(dense_624_matmul_readvariableop_resource:-l7
)dense_624_biasadd_readvariableop_resource:lM
?batch_normalization_561_assignmovingavg_readvariableop_resource:lO
Abatch_normalization_561_assignmovingavg_1_readvariableop_resource:lK
=batch_normalization_561_batchnorm_mul_readvariableop_resource:lG
9batch_normalization_561_batchnorm_readvariableop_resource:l:
(dense_625_matmul_readvariableop_resource:ll7
)dense_625_biasadd_readvariableop_resource:lM
?batch_normalization_562_assignmovingavg_readvariableop_resource:lO
Abatch_normalization_562_assignmovingavg_1_readvariableop_resource:lK
=batch_normalization_562_batchnorm_mul_readvariableop_resource:lG
9batch_normalization_562_batchnorm_readvariableop_resource:l:
(dense_626_matmul_readvariableop_resource:ll7
)dense_626_biasadd_readvariableop_resource:lM
?batch_normalization_563_assignmovingavg_readvariableop_resource:lO
Abatch_normalization_563_assignmovingavg_1_readvariableop_resource:lK
=batch_normalization_563_batchnorm_mul_readvariableop_resource:lG
9batch_normalization_563_batchnorm_readvariableop_resource:l:
(dense_627_matmul_readvariableop_resource:ll7
)dense_627_biasadd_readvariableop_resource:lM
?batch_normalization_564_assignmovingavg_readvariableop_resource:lO
Abatch_normalization_564_assignmovingavg_1_readvariableop_resource:lK
=batch_normalization_564_batchnorm_mul_readvariableop_resource:lG
9batch_normalization_564_batchnorm_readvariableop_resource:l:
(dense_628_matmul_readvariableop_resource:ll7
)dense_628_biasadd_readvariableop_resource:lM
?batch_normalization_565_assignmovingavg_readvariableop_resource:lO
Abatch_normalization_565_assignmovingavg_1_readvariableop_resource:lK
=batch_normalization_565_batchnorm_mul_readvariableop_resource:lG
9batch_normalization_565_batchnorm_readvariableop_resource:l:
(dense_629_matmul_readvariableop_resource:lV7
)dense_629_biasadd_readvariableop_resource:VM
?batch_normalization_566_assignmovingavg_readvariableop_resource:VO
Abatch_normalization_566_assignmovingavg_1_readvariableop_resource:VK
=batch_normalization_566_batchnorm_mul_readvariableop_resource:VG
9batch_normalization_566_batchnorm_readvariableop_resource:V:
(dense_630_matmul_readvariableop_resource:V7
)dense_630_biasadd_readvariableop_resource:
identity¢'batch_normalization_556/AssignMovingAvg¢6batch_normalization_556/AssignMovingAvg/ReadVariableOp¢)batch_normalization_556/AssignMovingAvg_1¢8batch_normalization_556/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_556/batchnorm/ReadVariableOp¢4batch_normalization_556/batchnorm/mul/ReadVariableOp¢'batch_normalization_557/AssignMovingAvg¢6batch_normalization_557/AssignMovingAvg/ReadVariableOp¢)batch_normalization_557/AssignMovingAvg_1¢8batch_normalization_557/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_557/batchnorm/ReadVariableOp¢4batch_normalization_557/batchnorm/mul/ReadVariableOp¢'batch_normalization_558/AssignMovingAvg¢6batch_normalization_558/AssignMovingAvg/ReadVariableOp¢)batch_normalization_558/AssignMovingAvg_1¢8batch_normalization_558/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_558/batchnorm/ReadVariableOp¢4batch_normalization_558/batchnorm/mul/ReadVariableOp¢'batch_normalization_559/AssignMovingAvg¢6batch_normalization_559/AssignMovingAvg/ReadVariableOp¢)batch_normalization_559/AssignMovingAvg_1¢8batch_normalization_559/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_559/batchnorm/ReadVariableOp¢4batch_normalization_559/batchnorm/mul/ReadVariableOp¢'batch_normalization_560/AssignMovingAvg¢6batch_normalization_560/AssignMovingAvg/ReadVariableOp¢)batch_normalization_560/AssignMovingAvg_1¢8batch_normalization_560/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_560/batchnorm/ReadVariableOp¢4batch_normalization_560/batchnorm/mul/ReadVariableOp¢'batch_normalization_561/AssignMovingAvg¢6batch_normalization_561/AssignMovingAvg/ReadVariableOp¢)batch_normalization_561/AssignMovingAvg_1¢8batch_normalization_561/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_561/batchnorm/ReadVariableOp¢4batch_normalization_561/batchnorm/mul/ReadVariableOp¢'batch_normalization_562/AssignMovingAvg¢6batch_normalization_562/AssignMovingAvg/ReadVariableOp¢)batch_normalization_562/AssignMovingAvg_1¢8batch_normalization_562/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_562/batchnorm/ReadVariableOp¢4batch_normalization_562/batchnorm/mul/ReadVariableOp¢'batch_normalization_563/AssignMovingAvg¢6batch_normalization_563/AssignMovingAvg/ReadVariableOp¢)batch_normalization_563/AssignMovingAvg_1¢8batch_normalization_563/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_563/batchnorm/ReadVariableOp¢4batch_normalization_563/batchnorm/mul/ReadVariableOp¢'batch_normalization_564/AssignMovingAvg¢6batch_normalization_564/AssignMovingAvg/ReadVariableOp¢)batch_normalization_564/AssignMovingAvg_1¢8batch_normalization_564/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_564/batchnorm/ReadVariableOp¢4batch_normalization_564/batchnorm/mul/ReadVariableOp¢'batch_normalization_565/AssignMovingAvg¢6batch_normalization_565/AssignMovingAvg/ReadVariableOp¢)batch_normalization_565/AssignMovingAvg_1¢8batch_normalization_565/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_565/batchnorm/ReadVariableOp¢4batch_normalization_565/batchnorm/mul/ReadVariableOp¢'batch_normalization_566/AssignMovingAvg¢6batch_normalization_566/AssignMovingAvg/ReadVariableOp¢)batch_normalization_566/AssignMovingAvg_1¢8batch_normalization_566/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_566/batchnorm/ReadVariableOp¢4batch_normalization_566/batchnorm/mul/ReadVariableOp¢ dense_619/BiasAdd/ReadVariableOp¢dense_619/MatMul/ReadVariableOp¢ dense_620/BiasAdd/ReadVariableOp¢dense_620/MatMul/ReadVariableOp¢ dense_621/BiasAdd/ReadVariableOp¢dense_621/MatMul/ReadVariableOp¢ dense_622/BiasAdd/ReadVariableOp¢dense_622/MatMul/ReadVariableOp¢ dense_623/BiasAdd/ReadVariableOp¢dense_623/MatMul/ReadVariableOp¢ dense_624/BiasAdd/ReadVariableOp¢dense_624/MatMul/ReadVariableOp¢ dense_625/BiasAdd/ReadVariableOp¢dense_625/MatMul/ReadVariableOp¢ dense_626/BiasAdd/ReadVariableOp¢dense_626/MatMul/ReadVariableOp¢ dense_627/BiasAdd/ReadVariableOp¢dense_627/MatMul/ReadVariableOp¢ dense_628/BiasAdd/ReadVariableOp¢dense_628/MatMul/ReadVariableOp¢ dense_629/BiasAdd/ReadVariableOp¢dense_629/MatMul/ReadVariableOp¢ dense_630/BiasAdd/ReadVariableOp¢dense_630/MatMul/ReadVariableOpm
normalization_63/subSubinputsnormalization_63_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_63/SqrtSqrtnormalization_63_sqrt_x*
T0*
_output_shapes

:_
normalization_63/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_63/MaximumMaximumnormalization_63/Sqrt:y:0#normalization_63/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_63/truedivRealDivnormalization_63/sub:z:0normalization_63/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_619/MatMul/ReadVariableOpReadVariableOp(dense_619_matmul_readvariableop_resource*
_output_shapes

:-*
dtype0
dense_619/MatMulMatMulnormalization_63/truediv:z:0'dense_619/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 dense_619/BiasAdd/ReadVariableOpReadVariableOp)dense_619_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0
dense_619/BiasAddBiasAdddense_619/MatMul:product:0(dense_619/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
6batch_normalization_556/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_556/moments/meanMeandense_619/BiasAdd:output:0?batch_normalization_556/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(
,batch_normalization_556/moments/StopGradientStopGradient-batch_normalization_556/moments/mean:output:0*
T0*
_output_shapes

:-Ë
1batch_normalization_556/moments/SquaredDifferenceSquaredDifferencedense_619/BiasAdd:output:05batch_normalization_556/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
:batch_normalization_556/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_556/moments/varianceMean5batch_normalization_556/moments/SquaredDifference:z:0Cbatch_normalization_556/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(
'batch_normalization_556/moments/SqueezeSqueeze-batch_normalization_556/moments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 £
)batch_normalization_556/moments/Squeeze_1Squeeze1batch_normalization_556/moments/variance:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 r
-batch_normalization_556/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_556/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_556_assignmovingavg_readvariableop_resource*
_output_shapes
:-*
dtype0É
+batch_normalization_556/AssignMovingAvg/subSub>batch_normalization_556/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_556/moments/Squeeze:output:0*
T0*
_output_shapes
:-À
+batch_normalization_556/AssignMovingAvg/mulMul/batch_normalization_556/AssignMovingAvg/sub:z:06batch_normalization_556/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-
'batch_normalization_556/AssignMovingAvgAssignSubVariableOp?batch_normalization_556_assignmovingavg_readvariableop_resource/batch_normalization_556/AssignMovingAvg/mul:z:07^batch_normalization_556/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_556/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_556/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_556_assignmovingavg_1_readvariableop_resource*
_output_shapes
:-*
dtype0Ï
-batch_normalization_556/AssignMovingAvg_1/subSub@batch_normalization_556/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_556/moments/Squeeze_1:output:0*
T0*
_output_shapes
:-Æ
-batch_normalization_556/AssignMovingAvg_1/mulMul1batch_normalization_556/AssignMovingAvg_1/sub:z:08batch_normalization_556/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-
)batch_normalization_556/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_556_assignmovingavg_1_readvariableop_resource1batch_normalization_556/AssignMovingAvg_1/mul:z:09^batch_normalization_556/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_556/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_556/batchnorm/addAddV22batch_normalization_556/moments/Squeeze_1:output:00batch_normalization_556/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
'batch_normalization_556/batchnorm/RsqrtRsqrt)batch_normalization_556/batchnorm/add:z:0*
T0*
_output_shapes
:-®
4batch_normalization_556/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_556_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0¼
%batch_normalization_556/batchnorm/mulMul+batch_normalization_556/batchnorm/Rsqrt:y:0<batch_normalization_556/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-§
'batch_normalization_556/batchnorm/mul_1Muldense_619/BiasAdd:output:0)batch_normalization_556/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-°
'batch_normalization_556/batchnorm/mul_2Mul0batch_normalization_556/moments/Squeeze:output:0)batch_normalization_556/batchnorm/mul:z:0*
T0*
_output_shapes
:-¦
0batch_normalization_556/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_556_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0¸
%batch_normalization_556/batchnorm/subSub8batch_normalization_556/batchnorm/ReadVariableOp:value:0+batch_normalization_556/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-º
'batch_normalization_556/batchnorm/add_1AddV2+batch_normalization_556/batchnorm/mul_1:z:0)batch_normalization_556/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
leaky_re_lu_556/LeakyRelu	LeakyRelu+batch_normalization_556/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>
dense_620/MatMul/ReadVariableOpReadVariableOp(dense_620_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0
dense_620/MatMulMatMul'leaky_re_lu_556/LeakyRelu:activations:0'dense_620/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 dense_620/BiasAdd/ReadVariableOpReadVariableOp)dense_620_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0
dense_620/BiasAddBiasAdddense_620/MatMul:product:0(dense_620/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
6batch_normalization_557/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_557/moments/meanMeandense_620/BiasAdd:output:0?batch_normalization_557/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(
,batch_normalization_557/moments/StopGradientStopGradient-batch_normalization_557/moments/mean:output:0*
T0*
_output_shapes

:-Ë
1batch_normalization_557/moments/SquaredDifferenceSquaredDifferencedense_620/BiasAdd:output:05batch_normalization_557/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
:batch_normalization_557/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_557/moments/varianceMean5batch_normalization_557/moments/SquaredDifference:z:0Cbatch_normalization_557/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(
'batch_normalization_557/moments/SqueezeSqueeze-batch_normalization_557/moments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 £
)batch_normalization_557/moments/Squeeze_1Squeeze1batch_normalization_557/moments/variance:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 r
-batch_normalization_557/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_557/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_557_assignmovingavg_readvariableop_resource*
_output_shapes
:-*
dtype0É
+batch_normalization_557/AssignMovingAvg/subSub>batch_normalization_557/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_557/moments/Squeeze:output:0*
T0*
_output_shapes
:-À
+batch_normalization_557/AssignMovingAvg/mulMul/batch_normalization_557/AssignMovingAvg/sub:z:06batch_normalization_557/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-
'batch_normalization_557/AssignMovingAvgAssignSubVariableOp?batch_normalization_557_assignmovingavg_readvariableop_resource/batch_normalization_557/AssignMovingAvg/mul:z:07^batch_normalization_557/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_557/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_557/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_557_assignmovingavg_1_readvariableop_resource*
_output_shapes
:-*
dtype0Ï
-batch_normalization_557/AssignMovingAvg_1/subSub@batch_normalization_557/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_557/moments/Squeeze_1:output:0*
T0*
_output_shapes
:-Æ
-batch_normalization_557/AssignMovingAvg_1/mulMul1batch_normalization_557/AssignMovingAvg_1/sub:z:08batch_normalization_557/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-
)batch_normalization_557/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_557_assignmovingavg_1_readvariableop_resource1batch_normalization_557/AssignMovingAvg_1/mul:z:09^batch_normalization_557/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_557/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_557/batchnorm/addAddV22batch_normalization_557/moments/Squeeze_1:output:00batch_normalization_557/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
'batch_normalization_557/batchnorm/RsqrtRsqrt)batch_normalization_557/batchnorm/add:z:0*
T0*
_output_shapes
:-®
4batch_normalization_557/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_557_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0¼
%batch_normalization_557/batchnorm/mulMul+batch_normalization_557/batchnorm/Rsqrt:y:0<batch_normalization_557/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-§
'batch_normalization_557/batchnorm/mul_1Muldense_620/BiasAdd:output:0)batch_normalization_557/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-°
'batch_normalization_557/batchnorm/mul_2Mul0batch_normalization_557/moments/Squeeze:output:0)batch_normalization_557/batchnorm/mul:z:0*
T0*
_output_shapes
:-¦
0batch_normalization_557/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_557_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0¸
%batch_normalization_557/batchnorm/subSub8batch_normalization_557/batchnorm/ReadVariableOp:value:0+batch_normalization_557/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-º
'batch_normalization_557/batchnorm/add_1AddV2+batch_normalization_557/batchnorm/mul_1:z:0)batch_normalization_557/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
leaky_re_lu_557/LeakyRelu	LeakyRelu+batch_normalization_557/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>
dense_621/MatMul/ReadVariableOpReadVariableOp(dense_621_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0
dense_621/MatMulMatMul'leaky_re_lu_557/LeakyRelu:activations:0'dense_621/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 dense_621/BiasAdd/ReadVariableOpReadVariableOp)dense_621_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0
dense_621/BiasAddBiasAdddense_621/MatMul:product:0(dense_621/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
6batch_normalization_558/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_558/moments/meanMeandense_621/BiasAdd:output:0?batch_normalization_558/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(
,batch_normalization_558/moments/StopGradientStopGradient-batch_normalization_558/moments/mean:output:0*
T0*
_output_shapes

:-Ë
1batch_normalization_558/moments/SquaredDifferenceSquaredDifferencedense_621/BiasAdd:output:05batch_normalization_558/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
:batch_normalization_558/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_558/moments/varianceMean5batch_normalization_558/moments/SquaredDifference:z:0Cbatch_normalization_558/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(
'batch_normalization_558/moments/SqueezeSqueeze-batch_normalization_558/moments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 £
)batch_normalization_558/moments/Squeeze_1Squeeze1batch_normalization_558/moments/variance:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 r
-batch_normalization_558/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_558/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_558_assignmovingavg_readvariableop_resource*
_output_shapes
:-*
dtype0É
+batch_normalization_558/AssignMovingAvg/subSub>batch_normalization_558/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_558/moments/Squeeze:output:0*
T0*
_output_shapes
:-À
+batch_normalization_558/AssignMovingAvg/mulMul/batch_normalization_558/AssignMovingAvg/sub:z:06batch_normalization_558/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-
'batch_normalization_558/AssignMovingAvgAssignSubVariableOp?batch_normalization_558_assignmovingavg_readvariableop_resource/batch_normalization_558/AssignMovingAvg/mul:z:07^batch_normalization_558/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_558/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_558/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_558_assignmovingavg_1_readvariableop_resource*
_output_shapes
:-*
dtype0Ï
-batch_normalization_558/AssignMovingAvg_1/subSub@batch_normalization_558/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_558/moments/Squeeze_1:output:0*
T0*
_output_shapes
:-Æ
-batch_normalization_558/AssignMovingAvg_1/mulMul1batch_normalization_558/AssignMovingAvg_1/sub:z:08batch_normalization_558/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-
)batch_normalization_558/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_558_assignmovingavg_1_readvariableop_resource1batch_normalization_558/AssignMovingAvg_1/mul:z:09^batch_normalization_558/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_558/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_558/batchnorm/addAddV22batch_normalization_558/moments/Squeeze_1:output:00batch_normalization_558/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
'batch_normalization_558/batchnorm/RsqrtRsqrt)batch_normalization_558/batchnorm/add:z:0*
T0*
_output_shapes
:-®
4batch_normalization_558/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_558_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0¼
%batch_normalization_558/batchnorm/mulMul+batch_normalization_558/batchnorm/Rsqrt:y:0<batch_normalization_558/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-§
'batch_normalization_558/batchnorm/mul_1Muldense_621/BiasAdd:output:0)batch_normalization_558/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-°
'batch_normalization_558/batchnorm/mul_2Mul0batch_normalization_558/moments/Squeeze:output:0)batch_normalization_558/batchnorm/mul:z:0*
T0*
_output_shapes
:-¦
0batch_normalization_558/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_558_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0¸
%batch_normalization_558/batchnorm/subSub8batch_normalization_558/batchnorm/ReadVariableOp:value:0+batch_normalization_558/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-º
'batch_normalization_558/batchnorm/add_1AddV2+batch_normalization_558/batchnorm/mul_1:z:0)batch_normalization_558/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
leaky_re_lu_558/LeakyRelu	LeakyRelu+batch_normalization_558/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>
dense_622/MatMul/ReadVariableOpReadVariableOp(dense_622_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0
dense_622/MatMulMatMul'leaky_re_lu_558/LeakyRelu:activations:0'dense_622/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 dense_622/BiasAdd/ReadVariableOpReadVariableOp)dense_622_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0
dense_622/BiasAddBiasAdddense_622/MatMul:product:0(dense_622/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
6batch_normalization_559/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_559/moments/meanMeandense_622/BiasAdd:output:0?batch_normalization_559/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(
,batch_normalization_559/moments/StopGradientStopGradient-batch_normalization_559/moments/mean:output:0*
T0*
_output_shapes

:-Ë
1batch_normalization_559/moments/SquaredDifferenceSquaredDifferencedense_622/BiasAdd:output:05batch_normalization_559/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
:batch_normalization_559/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_559/moments/varianceMean5batch_normalization_559/moments/SquaredDifference:z:0Cbatch_normalization_559/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(
'batch_normalization_559/moments/SqueezeSqueeze-batch_normalization_559/moments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 £
)batch_normalization_559/moments/Squeeze_1Squeeze1batch_normalization_559/moments/variance:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 r
-batch_normalization_559/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_559/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_559_assignmovingavg_readvariableop_resource*
_output_shapes
:-*
dtype0É
+batch_normalization_559/AssignMovingAvg/subSub>batch_normalization_559/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_559/moments/Squeeze:output:0*
T0*
_output_shapes
:-À
+batch_normalization_559/AssignMovingAvg/mulMul/batch_normalization_559/AssignMovingAvg/sub:z:06batch_normalization_559/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-
'batch_normalization_559/AssignMovingAvgAssignSubVariableOp?batch_normalization_559_assignmovingavg_readvariableop_resource/batch_normalization_559/AssignMovingAvg/mul:z:07^batch_normalization_559/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_559/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_559/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_559_assignmovingavg_1_readvariableop_resource*
_output_shapes
:-*
dtype0Ï
-batch_normalization_559/AssignMovingAvg_1/subSub@batch_normalization_559/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_559/moments/Squeeze_1:output:0*
T0*
_output_shapes
:-Æ
-batch_normalization_559/AssignMovingAvg_1/mulMul1batch_normalization_559/AssignMovingAvg_1/sub:z:08batch_normalization_559/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-
)batch_normalization_559/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_559_assignmovingavg_1_readvariableop_resource1batch_normalization_559/AssignMovingAvg_1/mul:z:09^batch_normalization_559/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_559/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_559/batchnorm/addAddV22batch_normalization_559/moments/Squeeze_1:output:00batch_normalization_559/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
'batch_normalization_559/batchnorm/RsqrtRsqrt)batch_normalization_559/batchnorm/add:z:0*
T0*
_output_shapes
:-®
4batch_normalization_559/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_559_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0¼
%batch_normalization_559/batchnorm/mulMul+batch_normalization_559/batchnorm/Rsqrt:y:0<batch_normalization_559/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-§
'batch_normalization_559/batchnorm/mul_1Muldense_622/BiasAdd:output:0)batch_normalization_559/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-°
'batch_normalization_559/batchnorm/mul_2Mul0batch_normalization_559/moments/Squeeze:output:0)batch_normalization_559/batchnorm/mul:z:0*
T0*
_output_shapes
:-¦
0batch_normalization_559/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_559_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0¸
%batch_normalization_559/batchnorm/subSub8batch_normalization_559/batchnorm/ReadVariableOp:value:0+batch_normalization_559/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-º
'batch_normalization_559/batchnorm/add_1AddV2+batch_normalization_559/batchnorm/mul_1:z:0)batch_normalization_559/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
leaky_re_lu_559/LeakyRelu	LeakyRelu+batch_normalization_559/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>
dense_623/MatMul/ReadVariableOpReadVariableOp(dense_623_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0
dense_623/MatMulMatMul'leaky_re_lu_559/LeakyRelu:activations:0'dense_623/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 dense_623/BiasAdd/ReadVariableOpReadVariableOp)dense_623_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0
dense_623/BiasAddBiasAdddense_623/MatMul:product:0(dense_623/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
6batch_normalization_560/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_560/moments/meanMeandense_623/BiasAdd:output:0?batch_normalization_560/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(
,batch_normalization_560/moments/StopGradientStopGradient-batch_normalization_560/moments/mean:output:0*
T0*
_output_shapes

:-Ë
1batch_normalization_560/moments/SquaredDifferenceSquaredDifferencedense_623/BiasAdd:output:05batch_normalization_560/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
:batch_normalization_560/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_560/moments/varianceMean5batch_normalization_560/moments/SquaredDifference:z:0Cbatch_normalization_560/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(
'batch_normalization_560/moments/SqueezeSqueeze-batch_normalization_560/moments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 £
)batch_normalization_560/moments/Squeeze_1Squeeze1batch_normalization_560/moments/variance:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 r
-batch_normalization_560/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_560/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_560_assignmovingavg_readvariableop_resource*
_output_shapes
:-*
dtype0É
+batch_normalization_560/AssignMovingAvg/subSub>batch_normalization_560/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_560/moments/Squeeze:output:0*
T0*
_output_shapes
:-À
+batch_normalization_560/AssignMovingAvg/mulMul/batch_normalization_560/AssignMovingAvg/sub:z:06batch_normalization_560/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-
'batch_normalization_560/AssignMovingAvgAssignSubVariableOp?batch_normalization_560_assignmovingavg_readvariableop_resource/batch_normalization_560/AssignMovingAvg/mul:z:07^batch_normalization_560/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_560/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_560/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_560_assignmovingavg_1_readvariableop_resource*
_output_shapes
:-*
dtype0Ï
-batch_normalization_560/AssignMovingAvg_1/subSub@batch_normalization_560/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_560/moments/Squeeze_1:output:0*
T0*
_output_shapes
:-Æ
-batch_normalization_560/AssignMovingAvg_1/mulMul1batch_normalization_560/AssignMovingAvg_1/sub:z:08batch_normalization_560/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-
)batch_normalization_560/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_560_assignmovingavg_1_readvariableop_resource1batch_normalization_560/AssignMovingAvg_1/mul:z:09^batch_normalization_560/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_560/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_560/batchnorm/addAddV22batch_normalization_560/moments/Squeeze_1:output:00batch_normalization_560/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
'batch_normalization_560/batchnorm/RsqrtRsqrt)batch_normalization_560/batchnorm/add:z:0*
T0*
_output_shapes
:-®
4batch_normalization_560/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_560_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0¼
%batch_normalization_560/batchnorm/mulMul+batch_normalization_560/batchnorm/Rsqrt:y:0<batch_normalization_560/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-§
'batch_normalization_560/batchnorm/mul_1Muldense_623/BiasAdd:output:0)batch_normalization_560/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-°
'batch_normalization_560/batchnorm/mul_2Mul0batch_normalization_560/moments/Squeeze:output:0)batch_normalization_560/batchnorm/mul:z:0*
T0*
_output_shapes
:-¦
0batch_normalization_560/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_560_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0¸
%batch_normalization_560/batchnorm/subSub8batch_normalization_560/batchnorm/ReadVariableOp:value:0+batch_normalization_560/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-º
'batch_normalization_560/batchnorm/add_1AddV2+batch_normalization_560/batchnorm/mul_1:z:0)batch_normalization_560/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
leaky_re_lu_560/LeakyRelu	LeakyRelu+batch_normalization_560/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>
dense_624/MatMul/ReadVariableOpReadVariableOp(dense_624_matmul_readvariableop_resource*
_output_shapes

:-l*
dtype0
dense_624/MatMulMatMul'leaky_re_lu_560/LeakyRelu:activations:0'dense_624/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 dense_624/BiasAdd/ReadVariableOpReadVariableOp)dense_624_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0
dense_624/BiasAddBiasAdddense_624/MatMul:product:0(dense_624/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
6batch_normalization_561/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_561/moments/meanMeandense_624/BiasAdd:output:0?batch_normalization_561/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(
,batch_normalization_561/moments/StopGradientStopGradient-batch_normalization_561/moments/mean:output:0*
T0*
_output_shapes

:lË
1batch_normalization_561/moments/SquaredDifferenceSquaredDifferencedense_624/BiasAdd:output:05batch_normalization_561/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
:batch_normalization_561/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_561/moments/varianceMean5batch_normalization_561/moments/SquaredDifference:z:0Cbatch_normalization_561/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(
'batch_normalization_561/moments/SqueezeSqueeze-batch_normalization_561/moments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 £
)batch_normalization_561/moments/Squeeze_1Squeeze1batch_normalization_561/moments/variance:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 r
-batch_normalization_561/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_561/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_561_assignmovingavg_readvariableop_resource*
_output_shapes
:l*
dtype0É
+batch_normalization_561/AssignMovingAvg/subSub>batch_normalization_561/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_561/moments/Squeeze:output:0*
T0*
_output_shapes
:lÀ
+batch_normalization_561/AssignMovingAvg/mulMul/batch_normalization_561/AssignMovingAvg/sub:z:06batch_normalization_561/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l
'batch_normalization_561/AssignMovingAvgAssignSubVariableOp?batch_normalization_561_assignmovingavg_readvariableop_resource/batch_normalization_561/AssignMovingAvg/mul:z:07^batch_normalization_561/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_561/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_561/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_561_assignmovingavg_1_readvariableop_resource*
_output_shapes
:l*
dtype0Ï
-batch_normalization_561/AssignMovingAvg_1/subSub@batch_normalization_561/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_561/moments/Squeeze_1:output:0*
T0*
_output_shapes
:lÆ
-batch_normalization_561/AssignMovingAvg_1/mulMul1batch_normalization_561/AssignMovingAvg_1/sub:z:08batch_normalization_561/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l
)batch_normalization_561/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_561_assignmovingavg_1_readvariableop_resource1batch_normalization_561/AssignMovingAvg_1/mul:z:09^batch_normalization_561/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_561/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_561/batchnorm/addAddV22batch_normalization_561/moments/Squeeze_1:output:00batch_normalization_561/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
'batch_normalization_561/batchnorm/RsqrtRsqrt)batch_normalization_561/batchnorm/add:z:0*
T0*
_output_shapes
:l®
4batch_normalization_561/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_561_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0¼
%batch_normalization_561/batchnorm/mulMul+batch_normalization_561/batchnorm/Rsqrt:y:0<batch_normalization_561/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l§
'batch_normalization_561/batchnorm/mul_1Muldense_624/BiasAdd:output:0)batch_normalization_561/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl°
'batch_normalization_561/batchnorm/mul_2Mul0batch_normalization_561/moments/Squeeze:output:0)batch_normalization_561/batchnorm/mul:z:0*
T0*
_output_shapes
:l¦
0batch_normalization_561/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_561_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0¸
%batch_normalization_561/batchnorm/subSub8batch_normalization_561/batchnorm/ReadVariableOp:value:0+batch_normalization_561/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lº
'batch_normalization_561/batchnorm/add_1AddV2+batch_normalization_561/batchnorm/mul_1:z:0)batch_normalization_561/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
leaky_re_lu_561/LeakyRelu	LeakyRelu+batch_normalization_561/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>
dense_625/MatMul/ReadVariableOpReadVariableOp(dense_625_matmul_readvariableop_resource*
_output_shapes

:ll*
dtype0
dense_625/MatMulMatMul'leaky_re_lu_561/LeakyRelu:activations:0'dense_625/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 dense_625/BiasAdd/ReadVariableOpReadVariableOp)dense_625_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0
dense_625/BiasAddBiasAdddense_625/MatMul:product:0(dense_625/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
6batch_normalization_562/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_562/moments/meanMeandense_625/BiasAdd:output:0?batch_normalization_562/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(
,batch_normalization_562/moments/StopGradientStopGradient-batch_normalization_562/moments/mean:output:0*
T0*
_output_shapes

:lË
1batch_normalization_562/moments/SquaredDifferenceSquaredDifferencedense_625/BiasAdd:output:05batch_normalization_562/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
:batch_normalization_562/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_562/moments/varianceMean5batch_normalization_562/moments/SquaredDifference:z:0Cbatch_normalization_562/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(
'batch_normalization_562/moments/SqueezeSqueeze-batch_normalization_562/moments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 £
)batch_normalization_562/moments/Squeeze_1Squeeze1batch_normalization_562/moments/variance:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 r
-batch_normalization_562/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_562/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_562_assignmovingavg_readvariableop_resource*
_output_shapes
:l*
dtype0É
+batch_normalization_562/AssignMovingAvg/subSub>batch_normalization_562/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_562/moments/Squeeze:output:0*
T0*
_output_shapes
:lÀ
+batch_normalization_562/AssignMovingAvg/mulMul/batch_normalization_562/AssignMovingAvg/sub:z:06batch_normalization_562/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l
'batch_normalization_562/AssignMovingAvgAssignSubVariableOp?batch_normalization_562_assignmovingavg_readvariableop_resource/batch_normalization_562/AssignMovingAvg/mul:z:07^batch_normalization_562/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_562/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_562/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_562_assignmovingavg_1_readvariableop_resource*
_output_shapes
:l*
dtype0Ï
-batch_normalization_562/AssignMovingAvg_1/subSub@batch_normalization_562/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_562/moments/Squeeze_1:output:0*
T0*
_output_shapes
:lÆ
-batch_normalization_562/AssignMovingAvg_1/mulMul1batch_normalization_562/AssignMovingAvg_1/sub:z:08batch_normalization_562/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l
)batch_normalization_562/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_562_assignmovingavg_1_readvariableop_resource1batch_normalization_562/AssignMovingAvg_1/mul:z:09^batch_normalization_562/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_562/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_562/batchnorm/addAddV22batch_normalization_562/moments/Squeeze_1:output:00batch_normalization_562/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
'batch_normalization_562/batchnorm/RsqrtRsqrt)batch_normalization_562/batchnorm/add:z:0*
T0*
_output_shapes
:l®
4batch_normalization_562/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_562_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0¼
%batch_normalization_562/batchnorm/mulMul+batch_normalization_562/batchnorm/Rsqrt:y:0<batch_normalization_562/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l§
'batch_normalization_562/batchnorm/mul_1Muldense_625/BiasAdd:output:0)batch_normalization_562/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl°
'batch_normalization_562/batchnorm/mul_2Mul0batch_normalization_562/moments/Squeeze:output:0)batch_normalization_562/batchnorm/mul:z:0*
T0*
_output_shapes
:l¦
0batch_normalization_562/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_562_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0¸
%batch_normalization_562/batchnorm/subSub8batch_normalization_562/batchnorm/ReadVariableOp:value:0+batch_normalization_562/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lº
'batch_normalization_562/batchnorm/add_1AddV2+batch_normalization_562/batchnorm/mul_1:z:0)batch_normalization_562/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
leaky_re_lu_562/LeakyRelu	LeakyRelu+batch_normalization_562/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>
dense_626/MatMul/ReadVariableOpReadVariableOp(dense_626_matmul_readvariableop_resource*
_output_shapes

:ll*
dtype0
dense_626/MatMulMatMul'leaky_re_lu_562/LeakyRelu:activations:0'dense_626/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 dense_626/BiasAdd/ReadVariableOpReadVariableOp)dense_626_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0
dense_626/BiasAddBiasAdddense_626/MatMul:product:0(dense_626/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
6batch_normalization_563/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_563/moments/meanMeandense_626/BiasAdd:output:0?batch_normalization_563/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(
,batch_normalization_563/moments/StopGradientStopGradient-batch_normalization_563/moments/mean:output:0*
T0*
_output_shapes

:lË
1batch_normalization_563/moments/SquaredDifferenceSquaredDifferencedense_626/BiasAdd:output:05batch_normalization_563/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
:batch_normalization_563/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_563/moments/varianceMean5batch_normalization_563/moments/SquaredDifference:z:0Cbatch_normalization_563/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(
'batch_normalization_563/moments/SqueezeSqueeze-batch_normalization_563/moments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 £
)batch_normalization_563/moments/Squeeze_1Squeeze1batch_normalization_563/moments/variance:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 r
-batch_normalization_563/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_563/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_563_assignmovingavg_readvariableop_resource*
_output_shapes
:l*
dtype0É
+batch_normalization_563/AssignMovingAvg/subSub>batch_normalization_563/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_563/moments/Squeeze:output:0*
T0*
_output_shapes
:lÀ
+batch_normalization_563/AssignMovingAvg/mulMul/batch_normalization_563/AssignMovingAvg/sub:z:06batch_normalization_563/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l
'batch_normalization_563/AssignMovingAvgAssignSubVariableOp?batch_normalization_563_assignmovingavg_readvariableop_resource/batch_normalization_563/AssignMovingAvg/mul:z:07^batch_normalization_563/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_563/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_563/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_563_assignmovingavg_1_readvariableop_resource*
_output_shapes
:l*
dtype0Ï
-batch_normalization_563/AssignMovingAvg_1/subSub@batch_normalization_563/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_563/moments/Squeeze_1:output:0*
T0*
_output_shapes
:lÆ
-batch_normalization_563/AssignMovingAvg_1/mulMul1batch_normalization_563/AssignMovingAvg_1/sub:z:08batch_normalization_563/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l
)batch_normalization_563/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_563_assignmovingavg_1_readvariableop_resource1batch_normalization_563/AssignMovingAvg_1/mul:z:09^batch_normalization_563/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_563/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_563/batchnorm/addAddV22batch_normalization_563/moments/Squeeze_1:output:00batch_normalization_563/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
'batch_normalization_563/batchnorm/RsqrtRsqrt)batch_normalization_563/batchnorm/add:z:0*
T0*
_output_shapes
:l®
4batch_normalization_563/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_563_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0¼
%batch_normalization_563/batchnorm/mulMul+batch_normalization_563/batchnorm/Rsqrt:y:0<batch_normalization_563/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l§
'batch_normalization_563/batchnorm/mul_1Muldense_626/BiasAdd:output:0)batch_normalization_563/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl°
'batch_normalization_563/batchnorm/mul_2Mul0batch_normalization_563/moments/Squeeze:output:0)batch_normalization_563/batchnorm/mul:z:0*
T0*
_output_shapes
:l¦
0batch_normalization_563/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_563_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0¸
%batch_normalization_563/batchnorm/subSub8batch_normalization_563/batchnorm/ReadVariableOp:value:0+batch_normalization_563/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lº
'batch_normalization_563/batchnorm/add_1AddV2+batch_normalization_563/batchnorm/mul_1:z:0)batch_normalization_563/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
leaky_re_lu_563/LeakyRelu	LeakyRelu+batch_normalization_563/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>
dense_627/MatMul/ReadVariableOpReadVariableOp(dense_627_matmul_readvariableop_resource*
_output_shapes

:ll*
dtype0
dense_627/MatMulMatMul'leaky_re_lu_563/LeakyRelu:activations:0'dense_627/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 dense_627/BiasAdd/ReadVariableOpReadVariableOp)dense_627_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0
dense_627/BiasAddBiasAdddense_627/MatMul:product:0(dense_627/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
6batch_normalization_564/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_564/moments/meanMeandense_627/BiasAdd:output:0?batch_normalization_564/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(
,batch_normalization_564/moments/StopGradientStopGradient-batch_normalization_564/moments/mean:output:0*
T0*
_output_shapes

:lË
1batch_normalization_564/moments/SquaredDifferenceSquaredDifferencedense_627/BiasAdd:output:05batch_normalization_564/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
:batch_normalization_564/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_564/moments/varianceMean5batch_normalization_564/moments/SquaredDifference:z:0Cbatch_normalization_564/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(
'batch_normalization_564/moments/SqueezeSqueeze-batch_normalization_564/moments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 £
)batch_normalization_564/moments/Squeeze_1Squeeze1batch_normalization_564/moments/variance:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 r
-batch_normalization_564/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_564/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_564_assignmovingavg_readvariableop_resource*
_output_shapes
:l*
dtype0É
+batch_normalization_564/AssignMovingAvg/subSub>batch_normalization_564/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_564/moments/Squeeze:output:0*
T0*
_output_shapes
:lÀ
+batch_normalization_564/AssignMovingAvg/mulMul/batch_normalization_564/AssignMovingAvg/sub:z:06batch_normalization_564/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l
'batch_normalization_564/AssignMovingAvgAssignSubVariableOp?batch_normalization_564_assignmovingavg_readvariableop_resource/batch_normalization_564/AssignMovingAvg/mul:z:07^batch_normalization_564/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_564/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_564/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_564_assignmovingavg_1_readvariableop_resource*
_output_shapes
:l*
dtype0Ï
-batch_normalization_564/AssignMovingAvg_1/subSub@batch_normalization_564/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_564/moments/Squeeze_1:output:0*
T0*
_output_shapes
:lÆ
-batch_normalization_564/AssignMovingAvg_1/mulMul1batch_normalization_564/AssignMovingAvg_1/sub:z:08batch_normalization_564/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l
)batch_normalization_564/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_564_assignmovingavg_1_readvariableop_resource1batch_normalization_564/AssignMovingAvg_1/mul:z:09^batch_normalization_564/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_564/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_564/batchnorm/addAddV22batch_normalization_564/moments/Squeeze_1:output:00batch_normalization_564/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
'batch_normalization_564/batchnorm/RsqrtRsqrt)batch_normalization_564/batchnorm/add:z:0*
T0*
_output_shapes
:l®
4batch_normalization_564/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_564_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0¼
%batch_normalization_564/batchnorm/mulMul+batch_normalization_564/batchnorm/Rsqrt:y:0<batch_normalization_564/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l§
'batch_normalization_564/batchnorm/mul_1Muldense_627/BiasAdd:output:0)batch_normalization_564/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl°
'batch_normalization_564/batchnorm/mul_2Mul0batch_normalization_564/moments/Squeeze:output:0)batch_normalization_564/batchnorm/mul:z:0*
T0*
_output_shapes
:l¦
0batch_normalization_564/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_564_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0¸
%batch_normalization_564/batchnorm/subSub8batch_normalization_564/batchnorm/ReadVariableOp:value:0+batch_normalization_564/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lº
'batch_normalization_564/batchnorm/add_1AddV2+batch_normalization_564/batchnorm/mul_1:z:0)batch_normalization_564/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
leaky_re_lu_564/LeakyRelu	LeakyRelu+batch_normalization_564/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>
dense_628/MatMul/ReadVariableOpReadVariableOp(dense_628_matmul_readvariableop_resource*
_output_shapes

:ll*
dtype0
dense_628/MatMulMatMul'leaky_re_lu_564/LeakyRelu:activations:0'dense_628/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 dense_628/BiasAdd/ReadVariableOpReadVariableOp)dense_628_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0
dense_628/BiasAddBiasAdddense_628/MatMul:product:0(dense_628/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
6batch_normalization_565/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_565/moments/meanMeandense_628/BiasAdd:output:0?batch_normalization_565/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(
,batch_normalization_565/moments/StopGradientStopGradient-batch_normalization_565/moments/mean:output:0*
T0*
_output_shapes

:lË
1batch_normalization_565/moments/SquaredDifferenceSquaredDifferencedense_628/BiasAdd:output:05batch_normalization_565/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
:batch_normalization_565/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_565/moments/varianceMean5batch_normalization_565/moments/SquaredDifference:z:0Cbatch_normalization_565/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(
'batch_normalization_565/moments/SqueezeSqueeze-batch_normalization_565/moments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 £
)batch_normalization_565/moments/Squeeze_1Squeeze1batch_normalization_565/moments/variance:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 r
-batch_normalization_565/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_565/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_565_assignmovingavg_readvariableop_resource*
_output_shapes
:l*
dtype0É
+batch_normalization_565/AssignMovingAvg/subSub>batch_normalization_565/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_565/moments/Squeeze:output:0*
T0*
_output_shapes
:lÀ
+batch_normalization_565/AssignMovingAvg/mulMul/batch_normalization_565/AssignMovingAvg/sub:z:06batch_normalization_565/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l
'batch_normalization_565/AssignMovingAvgAssignSubVariableOp?batch_normalization_565_assignmovingavg_readvariableop_resource/batch_normalization_565/AssignMovingAvg/mul:z:07^batch_normalization_565/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_565/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_565/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_565_assignmovingavg_1_readvariableop_resource*
_output_shapes
:l*
dtype0Ï
-batch_normalization_565/AssignMovingAvg_1/subSub@batch_normalization_565/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_565/moments/Squeeze_1:output:0*
T0*
_output_shapes
:lÆ
-batch_normalization_565/AssignMovingAvg_1/mulMul1batch_normalization_565/AssignMovingAvg_1/sub:z:08batch_normalization_565/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l
)batch_normalization_565/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_565_assignmovingavg_1_readvariableop_resource1batch_normalization_565/AssignMovingAvg_1/mul:z:09^batch_normalization_565/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_565/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_565/batchnorm/addAddV22batch_normalization_565/moments/Squeeze_1:output:00batch_normalization_565/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
'batch_normalization_565/batchnorm/RsqrtRsqrt)batch_normalization_565/batchnorm/add:z:0*
T0*
_output_shapes
:l®
4batch_normalization_565/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_565_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0¼
%batch_normalization_565/batchnorm/mulMul+batch_normalization_565/batchnorm/Rsqrt:y:0<batch_normalization_565/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l§
'batch_normalization_565/batchnorm/mul_1Muldense_628/BiasAdd:output:0)batch_normalization_565/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl°
'batch_normalization_565/batchnorm/mul_2Mul0batch_normalization_565/moments/Squeeze:output:0)batch_normalization_565/batchnorm/mul:z:0*
T0*
_output_shapes
:l¦
0batch_normalization_565/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_565_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0¸
%batch_normalization_565/batchnorm/subSub8batch_normalization_565/batchnorm/ReadVariableOp:value:0+batch_normalization_565/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lº
'batch_normalization_565/batchnorm/add_1AddV2+batch_normalization_565/batchnorm/mul_1:z:0)batch_normalization_565/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
leaky_re_lu_565/LeakyRelu	LeakyRelu+batch_normalization_565/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>
dense_629/MatMul/ReadVariableOpReadVariableOp(dense_629_matmul_readvariableop_resource*
_output_shapes

:lV*
dtype0
dense_629/MatMulMatMul'leaky_re_lu_565/LeakyRelu:activations:0'dense_629/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 dense_629/BiasAdd/ReadVariableOpReadVariableOp)dense_629_biasadd_readvariableop_resource*
_output_shapes
:V*
dtype0
dense_629/BiasAddBiasAdddense_629/MatMul:product:0(dense_629/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
6batch_normalization_566/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_566/moments/meanMeandense_629/BiasAdd:output:0?batch_normalization_566/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:V*
	keep_dims(
,batch_normalization_566/moments/StopGradientStopGradient-batch_normalization_566/moments/mean:output:0*
T0*
_output_shapes

:VË
1batch_normalization_566/moments/SquaredDifferenceSquaredDifferencedense_629/BiasAdd:output:05batch_normalization_566/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
:batch_normalization_566/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_566/moments/varianceMean5batch_normalization_566/moments/SquaredDifference:z:0Cbatch_normalization_566/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:V*
	keep_dims(
'batch_normalization_566/moments/SqueezeSqueeze-batch_normalization_566/moments/mean:output:0*
T0*
_output_shapes
:V*
squeeze_dims
 £
)batch_normalization_566/moments/Squeeze_1Squeeze1batch_normalization_566/moments/variance:output:0*
T0*
_output_shapes
:V*
squeeze_dims
 r
-batch_normalization_566/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_566/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_566_assignmovingavg_readvariableop_resource*
_output_shapes
:V*
dtype0É
+batch_normalization_566/AssignMovingAvg/subSub>batch_normalization_566/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_566/moments/Squeeze:output:0*
T0*
_output_shapes
:VÀ
+batch_normalization_566/AssignMovingAvg/mulMul/batch_normalization_566/AssignMovingAvg/sub:z:06batch_normalization_566/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:V
'batch_normalization_566/AssignMovingAvgAssignSubVariableOp?batch_normalization_566_assignmovingavg_readvariableop_resource/batch_normalization_566/AssignMovingAvg/mul:z:07^batch_normalization_566/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_566/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_566/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_566_assignmovingavg_1_readvariableop_resource*
_output_shapes
:V*
dtype0Ï
-batch_normalization_566/AssignMovingAvg_1/subSub@batch_normalization_566/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_566/moments/Squeeze_1:output:0*
T0*
_output_shapes
:VÆ
-batch_normalization_566/AssignMovingAvg_1/mulMul1batch_normalization_566/AssignMovingAvg_1/sub:z:08batch_normalization_566/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:V
)batch_normalization_566/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_566_assignmovingavg_1_readvariableop_resource1batch_normalization_566/AssignMovingAvg_1/mul:z:09^batch_normalization_566/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_566/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_566/batchnorm/addAddV22batch_normalization_566/moments/Squeeze_1:output:00batch_normalization_566/batchnorm/add/y:output:0*
T0*
_output_shapes
:V
'batch_normalization_566/batchnorm/RsqrtRsqrt)batch_normalization_566/batchnorm/add:z:0*
T0*
_output_shapes
:V®
4batch_normalization_566/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_566_batchnorm_mul_readvariableop_resource*
_output_shapes
:V*
dtype0¼
%batch_normalization_566/batchnorm/mulMul+batch_normalization_566/batchnorm/Rsqrt:y:0<batch_normalization_566/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:V§
'batch_normalization_566/batchnorm/mul_1Muldense_629/BiasAdd:output:0)batch_normalization_566/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV°
'batch_normalization_566/batchnorm/mul_2Mul0batch_normalization_566/moments/Squeeze:output:0)batch_normalization_566/batchnorm/mul:z:0*
T0*
_output_shapes
:V¦
0batch_normalization_566/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_566_batchnorm_readvariableop_resource*
_output_shapes
:V*
dtype0¸
%batch_normalization_566/batchnorm/subSub8batch_normalization_566/batchnorm/ReadVariableOp:value:0+batch_normalization_566/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Vº
'batch_normalization_566/batchnorm/add_1AddV2+batch_normalization_566/batchnorm/mul_1:z:0)batch_normalization_566/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
leaky_re_lu_566/LeakyRelu	LeakyRelu+batch_normalization_566/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*
alpha%>
dense_630/MatMul/ReadVariableOpReadVariableOp(dense_630_matmul_readvariableop_resource*
_output_shapes

:V*
dtype0
dense_630/MatMulMatMul'leaky_re_lu_566/LeakyRelu:activations:0'dense_630/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_630/BiasAdd/ReadVariableOpReadVariableOp)dense_630_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_630/BiasAddBiasAdddense_630/MatMul:product:0(dense_630/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_630/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾!
NoOpNoOp(^batch_normalization_556/AssignMovingAvg7^batch_normalization_556/AssignMovingAvg/ReadVariableOp*^batch_normalization_556/AssignMovingAvg_19^batch_normalization_556/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_556/batchnorm/ReadVariableOp5^batch_normalization_556/batchnorm/mul/ReadVariableOp(^batch_normalization_557/AssignMovingAvg7^batch_normalization_557/AssignMovingAvg/ReadVariableOp*^batch_normalization_557/AssignMovingAvg_19^batch_normalization_557/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_557/batchnorm/ReadVariableOp5^batch_normalization_557/batchnorm/mul/ReadVariableOp(^batch_normalization_558/AssignMovingAvg7^batch_normalization_558/AssignMovingAvg/ReadVariableOp*^batch_normalization_558/AssignMovingAvg_19^batch_normalization_558/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_558/batchnorm/ReadVariableOp5^batch_normalization_558/batchnorm/mul/ReadVariableOp(^batch_normalization_559/AssignMovingAvg7^batch_normalization_559/AssignMovingAvg/ReadVariableOp*^batch_normalization_559/AssignMovingAvg_19^batch_normalization_559/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_559/batchnorm/ReadVariableOp5^batch_normalization_559/batchnorm/mul/ReadVariableOp(^batch_normalization_560/AssignMovingAvg7^batch_normalization_560/AssignMovingAvg/ReadVariableOp*^batch_normalization_560/AssignMovingAvg_19^batch_normalization_560/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_560/batchnorm/ReadVariableOp5^batch_normalization_560/batchnorm/mul/ReadVariableOp(^batch_normalization_561/AssignMovingAvg7^batch_normalization_561/AssignMovingAvg/ReadVariableOp*^batch_normalization_561/AssignMovingAvg_19^batch_normalization_561/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_561/batchnorm/ReadVariableOp5^batch_normalization_561/batchnorm/mul/ReadVariableOp(^batch_normalization_562/AssignMovingAvg7^batch_normalization_562/AssignMovingAvg/ReadVariableOp*^batch_normalization_562/AssignMovingAvg_19^batch_normalization_562/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_562/batchnorm/ReadVariableOp5^batch_normalization_562/batchnorm/mul/ReadVariableOp(^batch_normalization_563/AssignMovingAvg7^batch_normalization_563/AssignMovingAvg/ReadVariableOp*^batch_normalization_563/AssignMovingAvg_19^batch_normalization_563/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_563/batchnorm/ReadVariableOp5^batch_normalization_563/batchnorm/mul/ReadVariableOp(^batch_normalization_564/AssignMovingAvg7^batch_normalization_564/AssignMovingAvg/ReadVariableOp*^batch_normalization_564/AssignMovingAvg_19^batch_normalization_564/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_564/batchnorm/ReadVariableOp5^batch_normalization_564/batchnorm/mul/ReadVariableOp(^batch_normalization_565/AssignMovingAvg7^batch_normalization_565/AssignMovingAvg/ReadVariableOp*^batch_normalization_565/AssignMovingAvg_19^batch_normalization_565/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_565/batchnorm/ReadVariableOp5^batch_normalization_565/batchnorm/mul/ReadVariableOp(^batch_normalization_566/AssignMovingAvg7^batch_normalization_566/AssignMovingAvg/ReadVariableOp*^batch_normalization_566/AssignMovingAvg_19^batch_normalization_566/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_566/batchnorm/ReadVariableOp5^batch_normalization_566/batchnorm/mul/ReadVariableOp!^dense_619/BiasAdd/ReadVariableOp ^dense_619/MatMul/ReadVariableOp!^dense_620/BiasAdd/ReadVariableOp ^dense_620/MatMul/ReadVariableOp!^dense_621/BiasAdd/ReadVariableOp ^dense_621/MatMul/ReadVariableOp!^dense_622/BiasAdd/ReadVariableOp ^dense_622/MatMul/ReadVariableOp!^dense_623/BiasAdd/ReadVariableOp ^dense_623/MatMul/ReadVariableOp!^dense_624/BiasAdd/ReadVariableOp ^dense_624/MatMul/ReadVariableOp!^dense_625/BiasAdd/ReadVariableOp ^dense_625/MatMul/ReadVariableOp!^dense_626/BiasAdd/ReadVariableOp ^dense_626/MatMul/ReadVariableOp!^dense_627/BiasAdd/ReadVariableOp ^dense_627/MatMul/ReadVariableOp!^dense_628/BiasAdd/ReadVariableOp ^dense_628/MatMul/ReadVariableOp!^dense_629/BiasAdd/ReadVariableOp ^dense_629/MatMul/ReadVariableOp!^dense_630/BiasAdd/ReadVariableOp ^dense_630/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_556/AssignMovingAvg'batch_normalization_556/AssignMovingAvg2p
6batch_normalization_556/AssignMovingAvg/ReadVariableOp6batch_normalization_556/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_556/AssignMovingAvg_1)batch_normalization_556/AssignMovingAvg_12t
8batch_normalization_556/AssignMovingAvg_1/ReadVariableOp8batch_normalization_556/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_556/batchnorm/ReadVariableOp0batch_normalization_556/batchnorm/ReadVariableOp2l
4batch_normalization_556/batchnorm/mul/ReadVariableOp4batch_normalization_556/batchnorm/mul/ReadVariableOp2R
'batch_normalization_557/AssignMovingAvg'batch_normalization_557/AssignMovingAvg2p
6batch_normalization_557/AssignMovingAvg/ReadVariableOp6batch_normalization_557/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_557/AssignMovingAvg_1)batch_normalization_557/AssignMovingAvg_12t
8batch_normalization_557/AssignMovingAvg_1/ReadVariableOp8batch_normalization_557/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_557/batchnorm/ReadVariableOp0batch_normalization_557/batchnorm/ReadVariableOp2l
4batch_normalization_557/batchnorm/mul/ReadVariableOp4batch_normalization_557/batchnorm/mul/ReadVariableOp2R
'batch_normalization_558/AssignMovingAvg'batch_normalization_558/AssignMovingAvg2p
6batch_normalization_558/AssignMovingAvg/ReadVariableOp6batch_normalization_558/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_558/AssignMovingAvg_1)batch_normalization_558/AssignMovingAvg_12t
8batch_normalization_558/AssignMovingAvg_1/ReadVariableOp8batch_normalization_558/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_558/batchnorm/ReadVariableOp0batch_normalization_558/batchnorm/ReadVariableOp2l
4batch_normalization_558/batchnorm/mul/ReadVariableOp4batch_normalization_558/batchnorm/mul/ReadVariableOp2R
'batch_normalization_559/AssignMovingAvg'batch_normalization_559/AssignMovingAvg2p
6batch_normalization_559/AssignMovingAvg/ReadVariableOp6batch_normalization_559/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_559/AssignMovingAvg_1)batch_normalization_559/AssignMovingAvg_12t
8batch_normalization_559/AssignMovingAvg_1/ReadVariableOp8batch_normalization_559/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_559/batchnorm/ReadVariableOp0batch_normalization_559/batchnorm/ReadVariableOp2l
4batch_normalization_559/batchnorm/mul/ReadVariableOp4batch_normalization_559/batchnorm/mul/ReadVariableOp2R
'batch_normalization_560/AssignMovingAvg'batch_normalization_560/AssignMovingAvg2p
6batch_normalization_560/AssignMovingAvg/ReadVariableOp6batch_normalization_560/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_560/AssignMovingAvg_1)batch_normalization_560/AssignMovingAvg_12t
8batch_normalization_560/AssignMovingAvg_1/ReadVariableOp8batch_normalization_560/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_560/batchnorm/ReadVariableOp0batch_normalization_560/batchnorm/ReadVariableOp2l
4batch_normalization_560/batchnorm/mul/ReadVariableOp4batch_normalization_560/batchnorm/mul/ReadVariableOp2R
'batch_normalization_561/AssignMovingAvg'batch_normalization_561/AssignMovingAvg2p
6batch_normalization_561/AssignMovingAvg/ReadVariableOp6batch_normalization_561/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_561/AssignMovingAvg_1)batch_normalization_561/AssignMovingAvg_12t
8batch_normalization_561/AssignMovingAvg_1/ReadVariableOp8batch_normalization_561/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_561/batchnorm/ReadVariableOp0batch_normalization_561/batchnorm/ReadVariableOp2l
4batch_normalization_561/batchnorm/mul/ReadVariableOp4batch_normalization_561/batchnorm/mul/ReadVariableOp2R
'batch_normalization_562/AssignMovingAvg'batch_normalization_562/AssignMovingAvg2p
6batch_normalization_562/AssignMovingAvg/ReadVariableOp6batch_normalization_562/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_562/AssignMovingAvg_1)batch_normalization_562/AssignMovingAvg_12t
8batch_normalization_562/AssignMovingAvg_1/ReadVariableOp8batch_normalization_562/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_562/batchnorm/ReadVariableOp0batch_normalization_562/batchnorm/ReadVariableOp2l
4batch_normalization_562/batchnorm/mul/ReadVariableOp4batch_normalization_562/batchnorm/mul/ReadVariableOp2R
'batch_normalization_563/AssignMovingAvg'batch_normalization_563/AssignMovingAvg2p
6batch_normalization_563/AssignMovingAvg/ReadVariableOp6batch_normalization_563/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_563/AssignMovingAvg_1)batch_normalization_563/AssignMovingAvg_12t
8batch_normalization_563/AssignMovingAvg_1/ReadVariableOp8batch_normalization_563/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_563/batchnorm/ReadVariableOp0batch_normalization_563/batchnorm/ReadVariableOp2l
4batch_normalization_563/batchnorm/mul/ReadVariableOp4batch_normalization_563/batchnorm/mul/ReadVariableOp2R
'batch_normalization_564/AssignMovingAvg'batch_normalization_564/AssignMovingAvg2p
6batch_normalization_564/AssignMovingAvg/ReadVariableOp6batch_normalization_564/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_564/AssignMovingAvg_1)batch_normalization_564/AssignMovingAvg_12t
8batch_normalization_564/AssignMovingAvg_1/ReadVariableOp8batch_normalization_564/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_564/batchnorm/ReadVariableOp0batch_normalization_564/batchnorm/ReadVariableOp2l
4batch_normalization_564/batchnorm/mul/ReadVariableOp4batch_normalization_564/batchnorm/mul/ReadVariableOp2R
'batch_normalization_565/AssignMovingAvg'batch_normalization_565/AssignMovingAvg2p
6batch_normalization_565/AssignMovingAvg/ReadVariableOp6batch_normalization_565/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_565/AssignMovingAvg_1)batch_normalization_565/AssignMovingAvg_12t
8batch_normalization_565/AssignMovingAvg_1/ReadVariableOp8batch_normalization_565/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_565/batchnorm/ReadVariableOp0batch_normalization_565/batchnorm/ReadVariableOp2l
4batch_normalization_565/batchnorm/mul/ReadVariableOp4batch_normalization_565/batchnorm/mul/ReadVariableOp2R
'batch_normalization_566/AssignMovingAvg'batch_normalization_566/AssignMovingAvg2p
6batch_normalization_566/AssignMovingAvg/ReadVariableOp6batch_normalization_566/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_566/AssignMovingAvg_1)batch_normalization_566/AssignMovingAvg_12t
8batch_normalization_566/AssignMovingAvg_1/ReadVariableOp8batch_normalization_566/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_566/batchnorm/ReadVariableOp0batch_normalization_566/batchnorm/ReadVariableOp2l
4batch_normalization_566/batchnorm/mul/ReadVariableOp4batch_normalization_566/batchnorm/mul/ReadVariableOp2D
 dense_619/BiasAdd/ReadVariableOp dense_619/BiasAdd/ReadVariableOp2B
dense_619/MatMul/ReadVariableOpdense_619/MatMul/ReadVariableOp2D
 dense_620/BiasAdd/ReadVariableOp dense_620/BiasAdd/ReadVariableOp2B
dense_620/MatMul/ReadVariableOpdense_620/MatMul/ReadVariableOp2D
 dense_621/BiasAdd/ReadVariableOp dense_621/BiasAdd/ReadVariableOp2B
dense_621/MatMul/ReadVariableOpdense_621/MatMul/ReadVariableOp2D
 dense_622/BiasAdd/ReadVariableOp dense_622/BiasAdd/ReadVariableOp2B
dense_622/MatMul/ReadVariableOpdense_622/MatMul/ReadVariableOp2D
 dense_623/BiasAdd/ReadVariableOp dense_623/BiasAdd/ReadVariableOp2B
dense_623/MatMul/ReadVariableOpdense_623/MatMul/ReadVariableOp2D
 dense_624/BiasAdd/ReadVariableOp dense_624/BiasAdd/ReadVariableOp2B
dense_624/MatMul/ReadVariableOpdense_624/MatMul/ReadVariableOp2D
 dense_625/BiasAdd/ReadVariableOp dense_625/BiasAdd/ReadVariableOp2B
dense_625/MatMul/ReadVariableOpdense_625/MatMul/ReadVariableOp2D
 dense_626/BiasAdd/ReadVariableOp dense_626/BiasAdd/ReadVariableOp2B
dense_626/MatMul/ReadVariableOpdense_626/MatMul/ReadVariableOp2D
 dense_627/BiasAdd/ReadVariableOp dense_627/BiasAdd/ReadVariableOp2B
dense_627/MatMul/ReadVariableOpdense_627/MatMul/ReadVariableOp2D
 dense_628/BiasAdd/ReadVariableOp dense_628/BiasAdd/ReadVariableOp2B
dense_628/MatMul/ReadVariableOpdense_628/MatMul/ReadVariableOp2D
 dense_629/BiasAdd/ReadVariableOp dense_629/BiasAdd/ReadVariableOp2B
dense_629/MatMul/ReadVariableOpdense_629/MatMul/ReadVariableOp2D
 dense_630/BiasAdd/ReadVariableOp dense_630/BiasAdd/ReadVariableOp2B
dense_630/MatMul/ReadVariableOpdense_630/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
È	
ö
E__inference_dense_626_layer_call_and_return_conditional_losses_831326

inputs0
matmul_readvariableop_resource:ll-
biasadd_readvariableop_resource:l
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ll*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:l*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_560_layer_call_fn_834418

inputs
unknown:-
	unknown_0:-
	unknown_1:-
	unknown_2:-
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_830528o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_561_layer_call_fn_834599

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
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_561_layer_call_and_return_conditional_losses_831282`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_834778

inputs/
!batchnorm_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l1
#batchnorm_readvariableop_1_resource:l1
#batchnorm_readvariableop_2_resource:l
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:lz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ä

*__inference_dense_621_layer_call_fn_834177

inputs
unknown:--
	unknown_0:-
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_621_layer_call_and_return_conditional_losses_831166o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ä

*__inference_dense_627_layer_call_fn_834831

inputs
unknown:ll
	unknown_0:l
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_627_layer_call_and_return_conditional_losses_831358o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
È	
ö
E__inference_dense_621_layer_call_and_return_conditional_losses_831166

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
È	
ö
E__inference_dense_627_layer_call_and_return_conditional_losses_834841

inputs0
matmul_readvariableop_resource:ll-
biasadd_readvariableop_resource:l
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ll*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:l*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_566_layer_call_fn_835072

inputs
unknown:V
	unknown_0:V
	unknown_1:V
	unknown_2:V
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_831020o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿV: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_556_layer_call_fn_833995

inputs
unknown:-
	unknown_0:-
	unknown_1:-
	unknown_2:-
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_830247o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_557_layer_call_fn_834091

inputs
unknown:-
	unknown_0:-
	unknown_1:-
	unknown_2:-
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_830282o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
È	
ö
E__inference_dense_630_layer_call_and_return_conditional_losses_835168

inputs0
matmul_readvariableop_resource:V-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:V*
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
:ÿÿÿÿÿÿÿÿÿV: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 
_user_specified_nameinputs
È	
ö
E__inference_dense_630_layer_call_and_return_conditional_losses_831454

inputs0
matmul_readvariableop_resource:V-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:V*
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
:ÿÿÿÿÿÿÿÿÿV: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_556_layer_call_fn_833982

inputs
unknown:-
	unknown_0:-
	unknown_1:-
	unknown_2:-
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_830200o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_560_layer_call_fn_834431

inputs
unknown:-
	unknown_0:-
	unknown_1:-
	unknown_2:-
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_830575o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_830692

inputs/
!batchnorm_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l1
#batchnorm_readvariableop_1_resource:l1
#batchnorm_readvariableop_2_resource:l
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:lz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_564_layer_call_and_return_conditional_losses_831378

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_834342

inputs/
!batchnorm_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-1
#batchnorm_readvariableop_1_resource:-1
#batchnorm_readvariableop_2_resource:-
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:-z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ä

*__inference_dense_626_layer_call_fn_834722

inputs
unknown:ll
	unknown_0:l
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_626_layer_call_and_return_conditional_losses_831326o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_834451

inputs/
!batchnorm_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-1
#batchnorm_readvariableop_1_resource:-1
#batchnorm_readvariableop_2_resource:-
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:-z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
È	
ö
E__inference_dense_628_layer_call_and_return_conditional_losses_834950

inputs0
matmul_readvariableop_resource:ll-
biasadd_readvariableop_resource:l
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ll*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:l*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_834560

inputs/
!batchnorm_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l1
#batchnorm_readvariableop_1_resource:l1
#batchnorm_readvariableop_2_resource:l
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:lz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs

Ù>
I__inference_sequential_63_layer_call_and_return_conditional_losses_833332

inputs
normalization_63_sub_y
normalization_63_sqrt_x:
(dense_619_matmul_readvariableop_resource:-7
)dense_619_biasadd_readvariableop_resource:-G
9batch_normalization_556_batchnorm_readvariableop_resource:-K
=batch_normalization_556_batchnorm_mul_readvariableop_resource:-I
;batch_normalization_556_batchnorm_readvariableop_1_resource:-I
;batch_normalization_556_batchnorm_readvariableop_2_resource:-:
(dense_620_matmul_readvariableop_resource:--7
)dense_620_biasadd_readvariableop_resource:-G
9batch_normalization_557_batchnorm_readvariableop_resource:-K
=batch_normalization_557_batchnorm_mul_readvariableop_resource:-I
;batch_normalization_557_batchnorm_readvariableop_1_resource:-I
;batch_normalization_557_batchnorm_readvariableop_2_resource:-:
(dense_621_matmul_readvariableop_resource:--7
)dense_621_biasadd_readvariableop_resource:-G
9batch_normalization_558_batchnorm_readvariableop_resource:-K
=batch_normalization_558_batchnorm_mul_readvariableop_resource:-I
;batch_normalization_558_batchnorm_readvariableop_1_resource:-I
;batch_normalization_558_batchnorm_readvariableop_2_resource:-:
(dense_622_matmul_readvariableop_resource:--7
)dense_622_biasadd_readvariableop_resource:-G
9batch_normalization_559_batchnorm_readvariableop_resource:-K
=batch_normalization_559_batchnorm_mul_readvariableop_resource:-I
;batch_normalization_559_batchnorm_readvariableop_1_resource:-I
;batch_normalization_559_batchnorm_readvariableop_2_resource:-:
(dense_623_matmul_readvariableop_resource:--7
)dense_623_biasadd_readvariableop_resource:-G
9batch_normalization_560_batchnorm_readvariableop_resource:-K
=batch_normalization_560_batchnorm_mul_readvariableop_resource:-I
;batch_normalization_560_batchnorm_readvariableop_1_resource:-I
;batch_normalization_560_batchnorm_readvariableop_2_resource:-:
(dense_624_matmul_readvariableop_resource:-l7
)dense_624_biasadd_readvariableop_resource:lG
9batch_normalization_561_batchnorm_readvariableop_resource:lK
=batch_normalization_561_batchnorm_mul_readvariableop_resource:lI
;batch_normalization_561_batchnorm_readvariableop_1_resource:lI
;batch_normalization_561_batchnorm_readvariableop_2_resource:l:
(dense_625_matmul_readvariableop_resource:ll7
)dense_625_biasadd_readvariableop_resource:lG
9batch_normalization_562_batchnorm_readvariableop_resource:lK
=batch_normalization_562_batchnorm_mul_readvariableop_resource:lI
;batch_normalization_562_batchnorm_readvariableop_1_resource:lI
;batch_normalization_562_batchnorm_readvariableop_2_resource:l:
(dense_626_matmul_readvariableop_resource:ll7
)dense_626_biasadd_readvariableop_resource:lG
9batch_normalization_563_batchnorm_readvariableop_resource:lK
=batch_normalization_563_batchnorm_mul_readvariableop_resource:lI
;batch_normalization_563_batchnorm_readvariableop_1_resource:lI
;batch_normalization_563_batchnorm_readvariableop_2_resource:l:
(dense_627_matmul_readvariableop_resource:ll7
)dense_627_biasadd_readvariableop_resource:lG
9batch_normalization_564_batchnorm_readvariableop_resource:lK
=batch_normalization_564_batchnorm_mul_readvariableop_resource:lI
;batch_normalization_564_batchnorm_readvariableop_1_resource:lI
;batch_normalization_564_batchnorm_readvariableop_2_resource:l:
(dense_628_matmul_readvariableop_resource:ll7
)dense_628_biasadd_readvariableop_resource:lG
9batch_normalization_565_batchnorm_readvariableop_resource:lK
=batch_normalization_565_batchnorm_mul_readvariableop_resource:lI
;batch_normalization_565_batchnorm_readvariableop_1_resource:lI
;batch_normalization_565_batchnorm_readvariableop_2_resource:l:
(dense_629_matmul_readvariableop_resource:lV7
)dense_629_biasadd_readvariableop_resource:VG
9batch_normalization_566_batchnorm_readvariableop_resource:VK
=batch_normalization_566_batchnorm_mul_readvariableop_resource:VI
;batch_normalization_566_batchnorm_readvariableop_1_resource:VI
;batch_normalization_566_batchnorm_readvariableop_2_resource:V:
(dense_630_matmul_readvariableop_resource:V7
)dense_630_biasadd_readvariableop_resource:
identity¢0batch_normalization_556/batchnorm/ReadVariableOp¢2batch_normalization_556/batchnorm/ReadVariableOp_1¢2batch_normalization_556/batchnorm/ReadVariableOp_2¢4batch_normalization_556/batchnorm/mul/ReadVariableOp¢0batch_normalization_557/batchnorm/ReadVariableOp¢2batch_normalization_557/batchnorm/ReadVariableOp_1¢2batch_normalization_557/batchnorm/ReadVariableOp_2¢4batch_normalization_557/batchnorm/mul/ReadVariableOp¢0batch_normalization_558/batchnorm/ReadVariableOp¢2batch_normalization_558/batchnorm/ReadVariableOp_1¢2batch_normalization_558/batchnorm/ReadVariableOp_2¢4batch_normalization_558/batchnorm/mul/ReadVariableOp¢0batch_normalization_559/batchnorm/ReadVariableOp¢2batch_normalization_559/batchnorm/ReadVariableOp_1¢2batch_normalization_559/batchnorm/ReadVariableOp_2¢4batch_normalization_559/batchnorm/mul/ReadVariableOp¢0batch_normalization_560/batchnorm/ReadVariableOp¢2batch_normalization_560/batchnorm/ReadVariableOp_1¢2batch_normalization_560/batchnorm/ReadVariableOp_2¢4batch_normalization_560/batchnorm/mul/ReadVariableOp¢0batch_normalization_561/batchnorm/ReadVariableOp¢2batch_normalization_561/batchnorm/ReadVariableOp_1¢2batch_normalization_561/batchnorm/ReadVariableOp_2¢4batch_normalization_561/batchnorm/mul/ReadVariableOp¢0batch_normalization_562/batchnorm/ReadVariableOp¢2batch_normalization_562/batchnorm/ReadVariableOp_1¢2batch_normalization_562/batchnorm/ReadVariableOp_2¢4batch_normalization_562/batchnorm/mul/ReadVariableOp¢0batch_normalization_563/batchnorm/ReadVariableOp¢2batch_normalization_563/batchnorm/ReadVariableOp_1¢2batch_normalization_563/batchnorm/ReadVariableOp_2¢4batch_normalization_563/batchnorm/mul/ReadVariableOp¢0batch_normalization_564/batchnorm/ReadVariableOp¢2batch_normalization_564/batchnorm/ReadVariableOp_1¢2batch_normalization_564/batchnorm/ReadVariableOp_2¢4batch_normalization_564/batchnorm/mul/ReadVariableOp¢0batch_normalization_565/batchnorm/ReadVariableOp¢2batch_normalization_565/batchnorm/ReadVariableOp_1¢2batch_normalization_565/batchnorm/ReadVariableOp_2¢4batch_normalization_565/batchnorm/mul/ReadVariableOp¢0batch_normalization_566/batchnorm/ReadVariableOp¢2batch_normalization_566/batchnorm/ReadVariableOp_1¢2batch_normalization_566/batchnorm/ReadVariableOp_2¢4batch_normalization_566/batchnorm/mul/ReadVariableOp¢ dense_619/BiasAdd/ReadVariableOp¢dense_619/MatMul/ReadVariableOp¢ dense_620/BiasAdd/ReadVariableOp¢dense_620/MatMul/ReadVariableOp¢ dense_621/BiasAdd/ReadVariableOp¢dense_621/MatMul/ReadVariableOp¢ dense_622/BiasAdd/ReadVariableOp¢dense_622/MatMul/ReadVariableOp¢ dense_623/BiasAdd/ReadVariableOp¢dense_623/MatMul/ReadVariableOp¢ dense_624/BiasAdd/ReadVariableOp¢dense_624/MatMul/ReadVariableOp¢ dense_625/BiasAdd/ReadVariableOp¢dense_625/MatMul/ReadVariableOp¢ dense_626/BiasAdd/ReadVariableOp¢dense_626/MatMul/ReadVariableOp¢ dense_627/BiasAdd/ReadVariableOp¢dense_627/MatMul/ReadVariableOp¢ dense_628/BiasAdd/ReadVariableOp¢dense_628/MatMul/ReadVariableOp¢ dense_629/BiasAdd/ReadVariableOp¢dense_629/MatMul/ReadVariableOp¢ dense_630/BiasAdd/ReadVariableOp¢dense_630/MatMul/ReadVariableOpm
normalization_63/subSubinputsnormalization_63_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_63/SqrtSqrtnormalization_63_sqrt_x*
T0*
_output_shapes

:_
normalization_63/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_63/MaximumMaximumnormalization_63/Sqrt:y:0#normalization_63/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_63/truedivRealDivnormalization_63/sub:z:0normalization_63/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_619/MatMul/ReadVariableOpReadVariableOp(dense_619_matmul_readvariableop_resource*
_output_shapes

:-*
dtype0
dense_619/MatMulMatMulnormalization_63/truediv:z:0'dense_619/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 dense_619/BiasAdd/ReadVariableOpReadVariableOp)dense_619_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0
dense_619/BiasAddBiasAdddense_619/MatMul:product:0(dense_619/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¦
0batch_normalization_556/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_556_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0l
'batch_normalization_556/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_556/batchnorm/addAddV28batch_normalization_556/batchnorm/ReadVariableOp:value:00batch_normalization_556/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
'batch_normalization_556/batchnorm/RsqrtRsqrt)batch_normalization_556/batchnorm/add:z:0*
T0*
_output_shapes
:-®
4batch_normalization_556/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_556_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0¼
%batch_normalization_556/batchnorm/mulMul+batch_normalization_556/batchnorm/Rsqrt:y:0<batch_normalization_556/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-§
'batch_normalization_556/batchnorm/mul_1Muldense_619/BiasAdd:output:0)batch_normalization_556/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ª
2batch_normalization_556/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_556_batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0º
'batch_normalization_556/batchnorm/mul_2Mul:batch_normalization_556/batchnorm/ReadVariableOp_1:value:0)batch_normalization_556/batchnorm/mul:z:0*
T0*
_output_shapes
:-ª
2batch_normalization_556/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_556_batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0º
%batch_normalization_556/batchnorm/subSub:batch_normalization_556/batchnorm/ReadVariableOp_2:value:0+batch_normalization_556/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-º
'batch_normalization_556/batchnorm/add_1AddV2+batch_normalization_556/batchnorm/mul_1:z:0)batch_normalization_556/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
leaky_re_lu_556/LeakyRelu	LeakyRelu+batch_normalization_556/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>
dense_620/MatMul/ReadVariableOpReadVariableOp(dense_620_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0
dense_620/MatMulMatMul'leaky_re_lu_556/LeakyRelu:activations:0'dense_620/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 dense_620/BiasAdd/ReadVariableOpReadVariableOp)dense_620_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0
dense_620/BiasAddBiasAdddense_620/MatMul:product:0(dense_620/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¦
0batch_normalization_557/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_557_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0l
'batch_normalization_557/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_557/batchnorm/addAddV28batch_normalization_557/batchnorm/ReadVariableOp:value:00batch_normalization_557/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
'batch_normalization_557/batchnorm/RsqrtRsqrt)batch_normalization_557/batchnorm/add:z:0*
T0*
_output_shapes
:-®
4batch_normalization_557/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_557_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0¼
%batch_normalization_557/batchnorm/mulMul+batch_normalization_557/batchnorm/Rsqrt:y:0<batch_normalization_557/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-§
'batch_normalization_557/batchnorm/mul_1Muldense_620/BiasAdd:output:0)batch_normalization_557/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ª
2batch_normalization_557/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_557_batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0º
'batch_normalization_557/batchnorm/mul_2Mul:batch_normalization_557/batchnorm/ReadVariableOp_1:value:0)batch_normalization_557/batchnorm/mul:z:0*
T0*
_output_shapes
:-ª
2batch_normalization_557/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_557_batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0º
%batch_normalization_557/batchnorm/subSub:batch_normalization_557/batchnorm/ReadVariableOp_2:value:0+batch_normalization_557/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-º
'batch_normalization_557/batchnorm/add_1AddV2+batch_normalization_557/batchnorm/mul_1:z:0)batch_normalization_557/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
leaky_re_lu_557/LeakyRelu	LeakyRelu+batch_normalization_557/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>
dense_621/MatMul/ReadVariableOpReadVariableOp(dense_621_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0
dense_621/MatMulMatMul'leaky_re_lu_557/LeakyRelu:activations:0'dense_621/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 dense_621/BiasAdd/ReadVariableOpReadVariableOp)dense_621_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0
dense_621/BiasAddBiasAdddense_621/MatMul:product:0(dense_621/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¦
0batch_normalization_558/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_558_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0l
'batch_normalization_558/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_558/batchnorm/addAddV28batch_normalization_558/batchnorm/ReadVariableOp:value:00batch_normalization_558/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
'batch_normalization_558/batchnorm/RsqrtRsqrt)batch_normalization_558/batchnorm/add:z:0*
T0*
_output_shapes
:-®
4batch_normalization_558/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_558_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0¼
%batch_normalization_558/batchnorm/mulMul+batch_normalization_558/batchnorm/Rsqrt:y:0<batch_normalization_558/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-§
'batch_normalization_558/batchnorm/mul_1Muldense_621/BiasAdd:output:0)batch_normalization_558/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ª
2batch_normalization_558/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_558_batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0º
'batch_normalization_558/batchnorm/mul_2Mul:batch_normalization_558/batchnorm/ReadVariableOp_1:value:0)batch_normalization_558/batchnorm/mul:z:0*
T0*
_output_shapes
:-ª
2batch_normalization_558/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_558_batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0º
%batch_normalization_558/batchnorm/subSub:batch_normalization_558/batchnorm/ReadVariableOp_2:value:0+batch_normalization_558/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-º
'batch_normalization_558/batchnorm/add_1AddV2+batch_normalization_558/batchnorm/mul_1:z:0)batch_normalization_558/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
leaky_re_lu_558/LeakyRelu	LeakyRelu+batch_normalization_558/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>
dense_622/MatMul/ReadVariableOpReadVariableOp(dense_622_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0
dense_622/MatMulMatMul'leaky_re_lu_558/LeakyRelu:activations:0'dense_622/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 dense_622/BiasAdd/ReadVariableOpReadVariableOp)dense_622_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0
dense_622/BiasAddBiasAdddense_622/MatMul:product:0(dense_622/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¦
0batch_normalization_559/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_559_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0l
'batch_normalization_559/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_559/batchnorm/addAddV28batch_normalization_559/batchnorm/ReadVariableOp:value:00batch_normalization_559/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
'batch_normalization_559/batchnorm/RsqrtRsqrt)batch_normalization_559/batchnorm/add:z:0*
T0*
_output_shapes
:-®
4batch_normalization_559/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_559_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0¼
%batch_normalization_559/batchnorm/mulMul+batch_normalization_559/batchnorm/Rsqrt:y:0<batch_normalization_559/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-§
'batch_normalization_559/batchnorm/mul_1Muldense_622/BiasAdd:output:0)batch_normalization_559/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ª
2batch_normalization_559/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_559_batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0º
'batch_normalization_559/batchnorm/mul_2Mul:batch_normalization_559/batchnorm/ReadVariableOp_1:value:0)batch_normalization_559/batchnorm/mul:z:0*
T0*
_output_shapes
:-ª
2batch_normalization_559/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_559_batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0º
%batch_normalization_559/batchnorm/subSub:batch_normalization_559/batchnorm/ReadVariableOp_2:value:0+batch_normalization_559/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-º
'batch_normalization_559/batchnorm/add_1AddV2+batch_normalization_559/batchnorm/mul_1:z:0)batch_normalization_559/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
leaky_re_lu_559/LeakyRelu	LeakyRelu+batch_normalization_559/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>
dense_623/MatMul/ReadVariableOpReadVariableOp(dense_623_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0
dense_623/MatMulMatMul'leaky_re_lu_559/LeakyRelu:activations:0'dense_623/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 dense_623/BiasAdd/ReadVariableOpReadVariableOp)dense_623_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0
dense_623/BiasAddBiasAdddense_623/MatMul:product:0(dense_623/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¦
0batch_normalization_560/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_560_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0l
'batch_normalization_560/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_560/batchnorm/addAddV28batch_normalization_560/batchnorm/ReadVariableOp:value:00batch_normalization_560/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
'batch_normalization_560/batchnorm/RsqrtRsqrt)batch_normalization_560/batchnorm/add:z:0*
T0*
_output_shapes
:-®
4batch_normalization_560/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_560_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0¼
%batch_normalization_560/batchnorm/mulMul+batch_normalization_560/batchnorm/Rsqrt:y:0<batch_normalization_560/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-§
'batch_normalization_560/batchnorm/mul_1Muldense_623/BiasAdd:output:0)batch_normalization_560/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ª
2batch_normalization_560/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_560_batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0º
'batch_normalization_560/batchnorm/mul_2Mul:batch_normalization_560/batchnorm/ReadVariableOp_1:value:0)batch_normalization_560/batchnorm/mul:z:0*
T0*
_output_shapes
:-ª
2batch_normalization_560/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_560_batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0º
%batch_normalization_560/batchnorm/subSub:batch_normalization_560/batchnorm/ReadVariableOp_2:value:0+batch_normalization_560/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-º
'batch_normalization_560/batchnorm/add_1AddV2+batch_normalization_560/batchnorm/mul_1:z:0)batch_normalization_560/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
leaky_re_lu_560/LeakyRelu	LeakyRelu+batch_normalization_560/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>
dense_624/MatMul/ReadVariableOpReadVariableOp(dense_624_matmul_readvariableop_resource*
_output_shapes

:-l*
dtype0
dense_624/MatMulMatMul'leaky_re_lu_560/LeakyRelu:activations:0'dense_624/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 dense_624/BiasAdd/ReadVariableOpReadVariableOp)dense_624_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0
dense_624/BiasAddBiasAdddense_624/MatMul:product:0(dense_624/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¦
0batch_normalization_561/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_561_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0l
'batch_normalization_561/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_561/batchnorm/addAddV28batch_normalization_561/batchnorm/ReadVariableOp:value:00batch_normalization_561/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
'batch_normalization_561/batchnorm/RsqrtRsqrt)batch_normalization_561/batchnorm/add:z:0*
T0*
_output_shapes
:l®
4batch_normalization_561/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_561_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0¼
%batch_normalization_561/batchnorm/mulMul+batch_normalization_561/batchnorm/Rsqrt:y:0<batch_normalization_561/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l§
'batch_normalization_561/batchnorm/mul_1Muldense_624/BiasAdd:output:0)batch_normalization_561/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlª
2batch_normalization_561/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_561_batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0º
'batch_normalization_561/batchnorm/mul_2Mul:batch_normalization_561/batchnorm/ReadVariableOp_1:value:0)batch_normalization_561/batchnorm/mul:z:0*
T0*
_output_shapes
:lª
2batch_normalization_561/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_561_batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0º
%batch_normalization_561/batchnorm/subSub:batch_normalization_561/batchnorm/ReadVariableOp_2:value:0+batch_normalization_561/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lº
'batch_normalization_561/batchnorm/add_1AddV2+batch_normalization_561/batchnorm/mul_1:z:0)batch_normalization_561/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
leaky_re_lu_561/LeakyRelu	LeakyRelu+batch_normalization_561/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>
dense_625/MatMul/ReadVariableOpReadVariableOp(dense_625_matmul_readvariableop_resource*
_output_shapes

:ll*
dtype0
dense_625/MatMulMatMul'leaky_re_lu_561/LeakyRelu:activations:0'dense_625/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 dense_625/BiasAdd/ReadVariableOpReadVariableOp)dense_625_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0
dense_625/BiasAddBiasAdddense_625/MatMul:product:0(dense_625/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¦
0batch_normalization_562/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_562_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0l
'batch_normalization_562/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_562/batchnorm/addAddV28batch_normalization_562/batchnorm/ReadVariableOp:value:00batch_normalization_562/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
'batch_normalization_562/batchnorm/RsqrtRsqrt)batch_normalization_562/batchnorm/add:z:0*
T0*
_output_shapes
:l®
4batch_normalization_562/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_562_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0¼
%batch_normalization_562/batchnorm/mulMul+batch_normalization_562/batchnorm/Rsqrt:y:0<batch_normalization_562/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l§
'batch_normalization_562/batchnorm/mul_1Muldense_625/BiasAdd:output:0)batch_normalization_562/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlª
2batch_normalization_562/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_562_batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0º
'batch_normalization_562/batchnorm/mul_2Mul:batch_normalization_562/batchnorm/ReadVariableOp_1:value:0)batch_normalization_562/batchnorm/mul:z:0*
T0*
_output_shapes
:lª
2batch_normalization_562/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_562_batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0º
%batch_normalization_562/batchnorm/subSub:batch_normalization_562/batchnorm/ReadVariableOp_2:value:0+batch_normalization_562/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lº
'batch_normalization_562/batchnorm/add_1AddV2+batch_normalization_562/batchnorm/mul_1:z:0)batch_normalization_562/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
leaky_re_lu_562/LeakyRelu	LeakyRelu+batch_normalization_562/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>
dense_626/MatMul/ReadVariableOpReadVariableOp(dense_626_matmul_readvariableop_resource*
_output_shapes

:ll*
dtype0
dense_626/MatMulMatMul'leaky_re_lu_562/LeakyRelu:activations:0'dense_626/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 dense_626/BiasAdd/ReadVariableOpReadVariableOp)dense_626_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0
dense_626/BiasAddBiasAdddense_626/MatMul:product:0(dense_626/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¦
0batch_normalization_563/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_563_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0l
'batch_normalization_563/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_563/batchnorm/addAddV28batch_normalization_563/batchnorm/ReadVariableOp:value:00batch_normalization_563/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
'batch_normalization_563/batchnorm/RsqrtRsqrt)batch_normalization_563/batchnorm/add:z:0*
T0*
_output_shapes
:l®
4batch_normalization_563/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_563_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0¼
%batch_normalization_563/batchnorm/mulMul+batch_normalization_563/batchnorm/Rsqrt:y:0<batch_normalization_563/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l§
'batch_normalization_563/batchnorm/mul_1Muldense_626/BiasAdd:output:0)batch_normalization_563/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlª
2batch_normalization_563/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_563_batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0º
'batch_normalization_563/batchnorm/mul_2Mul:batch_normalization_563/batchnorm/ReadVariableOp_1:value:0)batch_normalization_563/batchnorm/mul:z:0*
T0*
_output_shapes
:lª
2batch_normalization_563/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_563_batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0º
%batch_normalization_563/batchnorm/subSub:batch_normalization_563/batchnorm/ReadVariableOp_2:value:0+batch_normalization_563/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lº
'batch_normalization_563/batchnorm/add_1AddV2+batch_normalization_563/batchnorm/mul_1:z:0)batch_normalization_563/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
leaky_re_lu_563/LeakyRelu	LeakyRelu+batch_normalization_563/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>
dense_627/MatMul/ReadVariableOpReadVariableOp(dense_627_matmul_readvariableop_resource*
_output_shapes

:ll*
dtype0
dense_627/MatMulMatMul'leaky_re_lu_563/LeakyRelu:activations:0'dense_627/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 dense_627/BiasAdd/ReadVariableOpReadVariableOp)dense_627_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0
dense_627/BiasAddBiasAdddense_627/MatMul:product:0(dense_627/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¦
0batch_normalization_564/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_564_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0l
'batch_normalization_564/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_564/batchnorm/addAddV28batch_normalization_564/batchnorm/ReadVariableOp:value:00batch_normalization_564/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
'batch_normalization_564/batchnorm/RsqrtRsqrt)batch_normalization_564/batchnorm/add:z:0*
T0*
_output_shapes
:l®
4batch_normalization_564/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_564_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0¼
%batch_normalization_564/batchnorm/mulMul+batch_normalization_564/batchnorm/Rsqrt:y:0<batch_normalization_564/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l§
'batch_normalization_564/batchnorm/mul_1Muldense_627/BiasAdd:output:0)batch_normalization_564/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlª
2batch_normalization_564/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_564_batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0º
'batch_normalization_564/batchnorm/mul_2Mul:batch_normalization_564/batchnorm/ReadVariableOp_1:value:0)batch_normalization_564/batchnorm/mul:z:0*
T0*
_output_shapes
:lª
2batch_normalization_564/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_564_batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0º
%batch_normalization_564/batchnorm/subSub:batch_normalization_564/batchnorm/ReadVariableOp_2:value:0+batch_normalization_564/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lº
'batch_normalization_564/batchnorm/add_1AddV2+batch_normalization_564/batchnorm/mul_1:z:0)batch_normalization_564/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
leaky_re_lu_564/LeakyRelu	LeakyRelu+batch_normalization_564/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>
dense_628/MatMul/ReadVariableOpReadVariableOp(dense_628_matmul_readvariableop_resource*
_output_shapes

:ll*
dtype0
dense_628/MatMulMatMul'leaky_re_lu_564/LeakyRelu:activations:0'dense_628/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 dense_628/BiasAdd/ReadVariableOpReadVariableOp)dense_628_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0
dense_628/BiasAddBiasAdddense_628/MatMul:product:0(dense_628/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¦
0batch_normalization_565/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_565_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0l
'batch_normalization_565/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_565/batchnorm/addAddV28batch_normalization_565/batchnorm/ReadVariableOp:value:00batch_normalization_565/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
'batch_normalization_565/batchnorm/RsqrtRsqrt)batch_normalization_565/batchnorm/add:z:0*
T0*
_output_shapes
:l®
4batch_normalization_565/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_565_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0¼
%batch_normalization_565/batchnorm/mulMul+batch_normalization_565/batchnorm/Rsqrt:y:0<batch_normalization_565/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l§
'batch_normalization_565/batchnorm/mul_1Muldense_628/BiasAdd:output:0)batch_normalization_565/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlª
2batch_normalization_565/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_565_batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0º
'batch_normalization_565/batchnorm/mul_2Mul:batch_normalization_565/batchnorm/ReadVariableOp_1:value:0)batch_normalization_565/batchnorm/mul:z:0*
T0*
_output_shapes
:lª
2batch_normalization_565/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_565_batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0º
%batch_normalization_565/batchnorm/subSub:batch_normalization_565/batchnorm/ReadVariableOp_2:value:0+batch_normalization_565/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lº
'batch_normalization_565/batchnorm/add_1AddV2+batch_normalization_565/batchnorm/mul_1:z:0)batch_normalization_565/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
leaky_re_lu_565/LeakyRelu	LeakyRelu+batch_normalization_565/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>
dense_629/MatMul/ReadVariableOpReadVariableOp(dense_629_matmul_readvariableop_resource*
_output_shapes

:lV*
dtype0
dense_629/MatMulMatMul'leaky_re_lu_565/LeakyRelu:activations:0'dense_629/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 dense_629/BiasAdd/ReadVariableOpReadVariableOp)dense_629_biasadd_readvariableop_resource*
_output_shapes
:V*
dtype0
dense_629/BiasAddBiasAdddense_629/MatMul:product:0(dense_629/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV¦
0batch_normalization_566/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_566_batchnorm_readvariableop_resource*
_output_shapes
:V*
dtype0l
'batch_normalization_566/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_566/batchnorm/addAddV28batch_normalization_566/batchnorm/ReadVariableOp:value:00batch_normalization_566/batchnorm/add/y:output:0*
T0*
_output_shapes
:V
'batch_normalization_566/batchnorm/RsqrtRsqrt)batch_normalization_566/batchnorm/add:z:0*
T0*
_output_shapes
:V®
4batch_normalization_566/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_566_batchnorm_mul_readvariableop_resource*
_output_shapes
:V*
dtype0¼
%batch_normalization_566/batchnorm/mulMul+batch_normalization_566/batchnorm/Rsqrt:y:0<batch_normalization_566/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:V§
'batch_normalization_566/batchnorm/mul_1Muldense_629/BiasAdd:output:0)batch_normalization_566/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVª
2batch_normalization_566/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_566_batchnorm_readvariableop_1_resource*
_output_shapes
:V*
dtype0º
'batch_normalization_566/batchnorm/mul_2Mul:batch_normalization_566/batchnorm/ReadVariableOp_1:value:0)batch_normalization_566/batchnorm/mul:z:0*
T0*
_output_shapes
:Vª
2batch_normalization_566/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_566_batchnorm_readvariableop_2_resource*
_output_shapes
:V*
dtype0º
%batch_normalization_566/batchnorm/subSub:batch_normalization_566/batchnorm/ReadVariableOp_2:value:0+batch_normalization_566/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Vº
'batch_normalization_566/batchnorm/add_1AddV2+batch_normalization_566/batchnorm/mul_1:z:0)batch_normalization_566/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
leaky_re_lu_566/LeakyRelu	LeakyRelu+batch_normalization_566/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*
alpha%>
dense_630/MatMul/ReadVariableOpReadVariableOp(dense_630_matmul_readvariableop_resource*
_output_shapes

:V*
dtype0
dense_630/MatMulMatMul'leaky_re_lu_566/LeakyRelu:activations:0'dense_630/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_630/BiasAdd/ReadVariableOpReadVariableOp)dense_630_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_630/BiasAddBiasAdddense_630/MatMul:product:0(dense_630/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_630/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp1^batch_normalization_556/batchnorm/ReadVariableOp3^batch_normalization_556/batchnorm/ReadVariableOp_13^batch_normalization_556/batchnorm/ReadVariableOp_25^batch_normalization_556/batchnorm/mul/ReadVariableOp1^batch_normalization_557/batchnorm/ReadVariableOp3^batch_normalization_557/batchnorm/ReadVariableOp_13^batch_normalization_557/batchnorm/ReadVariableOp_25^batch_normalization_557/batchnorm/mul/ReadVariableOp1^batch_normalization_558/batchnorm/ReadVariableOp3^batch_normalization_558/batchnorm/ReadVariableOp_13^batch_normalization_558/batchnorm/ReadVariableOp_25^batch_normalization_558/batchnorm/mul/ReadVariableOp1^batch_normalization_559/batchnorm/ReadVariableOp3^batch_normalization_559/batchnorm/ReadVariableOp_13^batch_normalization_559/batchnorm/ReadVariableOp_25^batch_normalization_559/batchnorm/mul/ReadVariableOp1^batch_normalization_560/batchnorm/ReadVariableOp3^batch_normalization_560/batchnorm/ReadVariableOp_13^batch_normalization_560/batchnorm/ReadVariableOp_25^batch_normalization_560/batchnorm/mul/ReadVariableOp1^batch_normalization_561/batchnorm/ReadVariableOp3^batch_normalization_561/batchnorm/ReadVariableOp_13^batch_normalization_561/batchnorm/ReadVariableOp_25^batch_normalization_561/batchnorm/mul/ReadVariableOp1^batch_normalization_562/batchnorm/ReadVariableOp3^batch_normalization_562/batchnorm/ReadVariableOp_13^batch_normalization_562/batchnorm/ReadVariableOp_25^batch_normalization_562/batchnorm/mul/ReadVariableOp1^batch_normalization_563/batchnorm/ReadVariableOp3^batch_normalization_563/batchnorm/ReadVariableOp_13^batch_normalization_563/batchnorm/ReadVariableOp_25^batch_normalization_563/batchnorm/mul/ReadVariableOp1^batch_normalization_564/batchnorm/ReadVariableOp3^batch_normalization_564/batchnorm/ReadVariableOp_13^batch_normalization_564/batchnorm/ReadVariableOp_25^batch_normalization_564/batchnorm/mul/ReadVariableOp1^batch_normalization_565/batchnorm/ReadVariableOp3^batch_normalization_565/batchnorm/ReadVariableOp_13^batch_normalization_565/batchnorm/ReadVariableOp_25^batch_normalization_565/batchnorm/mul/ReadVariableOp1^batch_normalization_566/batchnorm/ReadVariableOp3^batch_normalization_566/batchnorm/ReadVariableOp_13^batch_normalization_566/batchnorm/ReadVariableOp_25^batch_normalization_566/batchnorm/mul/ReadVariableOp!^dense_619/BiasAdd/ReadVariableOp ^dense_619/MatMul/ReadVariableOp!^dense_620/BiasAdd/ReadVariableOp ^dense_620/MatMul/ReadVariableOp!^dense_621/BiasAdd/ReadVariableOp ^dense_621/MatMul/ReadVariableOp!^dense_622/BiasAdd/ReadVariableOp ^dense_622/MatMul/ReadVariableOp!^dense_623/BiasAdd/ReadVariableOp ^dense_623/MatMul/ReadVariableOp!^dense_624/BiasAdd/ReadVariableOp ^dense_624/MatMul/ReadVariableOp!^dense_625/BiasAdd/ReadVariableOp ^dense_625/MatMul/ReadVariableOp!^dense_626/BiasAdd/ReadVariableOp ^dense_626/MatMul/ReadVariableOp!^dense_627/BiasAdd/ReadVariableOp ^dense_627/MatMul/ReadVariableOp!^dense_628/BiasAdd/ReadVariableOp ^dense_628/MatMul/ReadVariableOp!^dense_629/BiasAdd/ReadVariableOp ^dense_629/MatMul/ReadVariableOp!^dense_630/BiasAdd/ReadVariableOp ^dense_630/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_556/batchnorm/ReadVariableOp0batch_normalization_556/batchnorm/ReadVariableOp2h
2batch_normalization_556/batchnorm/ReadVariableOp_12batch_normalization_556/batchnorm/ReadVariableOp_12h
2batch_normalization_556/batchnorm/ReadVariableOp_22batch_normalization_556/batchnorm/ReadVariableOp_22l
4batch_normalization_556/batchnorm/mul/ReadVariableOp4batch_normalization_556/batchnorm/mul/ReadVariableOp2d
0batch_normalization_557/batchnorm/ReadVariableOp0batch_normalization_557/batchnorm/ReadVariableOp2h
2batch_normalization_557/batchnorm/ReadVariableOp_12batch_normalization_557/batchnorm/ReadVariableOp_12h
2batch_normalization_557/batchnorm/ReadVariableOp_22batch_normalization_557/batchnorm/ReadVariableOp_22l
4batch_normalization_557/batchnorm/mul/ReadVariableOp4batch_normalization_557/batchnorm/mul/ReadVariableOp2d
0batch_normalization_558/batchnorm/ReadVariableOp0batch_normalization_558/batchnorm/ReadVariableOp2h
2batch_normalization_558/batchnorm/ReadVariableOp_12batch_normalization_558/batchnorm/ReadVariableOp_12h
2batch_normalization_558/batchnorm/ReadVariableOp_22batch_normalization_558/batchnorm/ReadVariableOp_22l
4batch_normalization_558/batchnorm/mul/ReadVariableOp4batch_normalization_558/batchnorm/mul/ReadVariableOp2d
0batch_normalization_559/batchnorm/ReadVariableOp0batch_normalization_559/batchnorm/ReadVariableOp2h
2batch_normalization_559/batchnorm/ReadVariableOp_12batch_normalization_559/batchnorm/ReadVariableOp_12h
2batch_normalization_559/batchnorm/ReadVariableOp_22batch_normalization_559/batchnorm/ReadVariableOp_22l
4batch_normalization_559/batchnorm/mul/ReadVariableOp4batch_normalization_559/batchnorm/mul/ReadVariableOp2d
0batch_normalization_560/batchnorm/ReadVariableOp0batch_normalization_560/batchnorm/ReadVariableOp2h
2batch_normalization_560/batchnorm/ReadVariableOp_12batch_normalization_560/batchnorm/ReadVariableOp_12h
2batch_normalization_560/batchnorm/ReadVariableOp_22batch_normalization_560/batchnorm/ReadVariableOp_22l
4batch_normalization_560/batchnorm/mul/ReadVariableOp4batch_normalization_560/batchnorm/mul/ReadVariableOp2d
0batch_normalization_561/batchnorm/ReadVariableOp0batch_normalization_561/batchnorm/ReadVariableOp2h
2batch_normalization_561/batchnorm/ReadVariableOp_12batch_normalization_561/batchnorm/ReadVariableOp_12h
2batch_normalization_561/batchnorm/ReadVariableOp_22batch_normalization_561/batchnorm/ReadVariableOp_22l
4batch_normalization_561/batchnorm/mul/ReadVariableOp4batch_normalization_561/batchnorm/mul/ReadVariableOp2d
0batch_normalization_562/batchnorm/ReadVariableOp0batch_normalization_562/batchnorm/ReadVariableOp2h
2batch_normalization_562/batchnorm/ReadVariableOp_12batch_normalization_562/batchnorm/ReadVariableOp_12h
2batch_normalization_562/batchnorm/ReadVariableOp_22batch_normalization_562/batchnorm/ReadVariableOp_22l
4batch_normalization_562/batchnorm/mul/ReadVariableOp4batch_normalization_562/batchnorm/mul/ReadVariableOp2d
0batch_normalization_563/batchnorm/ReadVariableOp0batch_normalization_563/batchnorm/ReadVariableOp2h
2batch_normalization_563/batchnorm/ReadVariableOp_12batch_normalization_563/batchnorm/ReadVariableOp_12h
2batch_normalization_563/batchnorm/ReadVariableOp_22batch_normalization_563/batchnorm/ReadVariableOp_22l
4batch_normalization_563/batchnorm/mul/ReadVariableOp4batch_normalization_563/batchnorm/mul/ReadVariableOp2d
0batch_normalization_564/batchnorm/ReadVariableOp0batch_normalization_564/batchnorm/ReadVariableOp2h
2batch_normalization_564/batchnorm/ReadVariableOp_12batch_normalization_564/batchnorm/ReadVariableOp_12h
2batch_normalization_564/batchnorm/ReadVariableOp_22batch_normalization_564/batchnorm/ReadVariableOp_22l
4batch_normalization_564/batchnorm/mul/ReadVariableOp4batch_normalization_564/batchnorm/mul/ReadVariableOp2d
0batch_normalization_565/batchnorm/ReadVariableOp0batch_normalization_565/batchnorm/ReadVariableOp2h
2batch_normalization_565/batchnorm/ReadVariableOp_12batch_normalization_565/batchnorm/ReadVariableOp_12h
2batch_normalization_565/batchnorm/ReadVariableOp_22batch_normalization_565/batchnorm/ReadVariableOp_22l
4batch_normalization_565/batchnorm/mul/ReadVariableOp4batch_normalization_565/batchnorm/mul/ReadVariableOp2d
0batch_normalization_566/batchnorm/ReadVariableOp0batch_normalization_566/batchnorm/ReadVariableOp2h
2batch_normalization_566/batchnorm/ReadVariableOp_12batch_normalization_566/batchnorm/ReadVariableOp_12h
2batch_normalization_566/batchnorm/ReadVariableOp_22batch_normalization_566/batchnorm/ReadVariableOp_22l
4batch_normalization_566/batchnorm/mul/ReadVariableOp4batch_normalization_566/batchnorm/mul/ReadVariableOp2D
 dense_619/BiasAdd/ReadVariableOp dense_619/BiasAdd/ReadVariableOp2B
dense_619/MatMul/ReadVariableOpdense_619/MatMul/ReadVariableOp2D
 dense_620/BiasAdd/ReadVariableOp dense_620/BiasAdd/ReadVariableOp2B
dense_620/MatMul/ReadVariableOpdense_620/MatMul/ReadVariableOp2D
 dense_621/BiasAdd/ReadVariableOp dense_621/BiasAdd/ReadVariableOp2B
dense_621/MatMul/ReadVariableOpdense_621/MatMul/ReadVariableOp2D
 dense_622/BiasAdd/ReadVariableOp dense_622/BiasAdd/ReadVariableOp2B
dense_622/MatMul/ReadVariableOpdense_622/MatMul/ReadVariableOp2D
 dense_623/BiasAdd/ReadVariableOp dense_623/BiasAdd/ReadVariableOp2B
dense_623/MatMul/ReadVariableOpdense_623/MatMul/ReadVariableOp2D
 dense_624/BiasAdd/ReadVariableOp dense_624/BiasAdd/ReadVariableOp2B
dense_624/MatMul/ReadVariableOpdense_624/MatMul/ReadVariableOp2D
 dense_625/BiasAdd/ReadVariableOp dense_625/BiasAdd/ReadVariableOp2B
dense_625/MatMul/ReadVariableOpdense_625/MatMul/ReadVariableOp2D
 dense_626/BiasAdd/ReadVariableOp dense_626/BiasAdd/ReadVariableOp2B
dense_626/MatMul/ReadVariableOpdense_626/MatMul/ReadVariableOp2D
 dense_627/BiasAdd/ReadVariableOp dense_627/BiasAdd/ReadVariableOp2B
dense_627/MatMul/ReadVariableOpdense_627/MatMul/ReadVariableOp2D
 dense_628/BiasAdd/ReadVariableOp dense_628/BiasAdd/ReadVariableOp2B
dense_628/MatMul/ReadVariableOpdense_628/MatMul/ReadVariableOp2D
 dense_629/BiasAdd/ReadVariableOp dense_629/BiasAdd/ReadVariableOp2B
dense_629/MatMul/ReadVariableOpdense_629/MatMul/ReadVariableOp2D
 dense_630/BiasAdd/ReadVariableOp dense_630/BiasAdd/ReadVariableOp2B
dense_630/MatMul/ReadVariableOpdense_630/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
¬
Ó
8__inference_batch_normalization_558_layer_call_fn_834200

inputs
unknown:-
	unknown_0:-
	unknown_1:-
	unknown_2:-
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_830364o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ä

*__inference_dense_630_layer_call_fn_835158

inputs
unknown:V
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
E__inference_dense_630_layer_call_and_return_conditional_losses_831454o
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
:ÿÿÿÿÿÿÿÿÿV: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_558_layer_call_and_return_conditional_losses_834277

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs

¢
.__inference_sequential_63_layer_call_fn_831604
normalization_63_input
unknown
	unknown_0
	unknown_1:-
	unknown_2:-
	unknown_3:-
	unknown_4:-
	unknown_5:-
	unknown_6:-
	unknown_7:--
	unknown_8:-
	unknown_9:-

unknown_10:-

unknown_11:-

unknown_12:-

unknown_13:--

unknown_14:-

unknown_15:-

unknown_16:-

unknown_17:-

unknown_18:-

unknown_19:--

unknown_20:-

unknown_21:-

unknown_22:-

unknown_23:-

unknown_24:-

unknown_25:--

unknown_26:-

unknown_27:-

unknown_28:-

unknown_29:-

unknown_30:-

unknown_31:-l

unknown_32:l

unknown_33:l

unknown_34:l

unknown_35:l

unknown_36:l

unknown_37:ll

unknown_38:l

unknown_39:l

unknown_40:l

unknown_41:l

unknown_42:l

unknown_43:ll

unknown_44:l

unknown_45:l

unknown_46:l

unknown_47:l

unknown_48:l

unknown_49:ll

unknown_50:l

unknown_51:l

unknown_52:l

unknown_53:l

unknown_54:l

unknown_55:ll

unknown_56:l

unknown_57:l

unknown_58:l

unknown_59:l

unknown_60:l

unknown_61:lV

unknown_62:V

unknown_63:V

unknown_64:V

unknown_65:V

unknown_66:V

unknown_67:V

unknown_68:
identity¢StatefulPartitionedCall

StatefulPartitionedCallStatefulPartitionedCallnormalization_63_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEF*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_63_layer_call_and_return_conditional_losses_831461o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_63_input:$ 

_output_shapes

::$ 

_output_shapes

:
«
L
0__inference_leaky_re_lu_565_layer_call_fn_835035

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
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_565_layer_call_and_return_conditional_losses_831410`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ä

*__inference_dense_623_layer_call_fn_834395

inputs
unknown:--
	unknown_0:-
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_623_layer_call_and_return_conditional_losses_831230o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_564_layer_call_fn_834926

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
:ÿÿÿÿÿÿÿÿÿl* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_564_layer_call_and_return_conditional_losses_831378`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_565_layer_call_fn_834976

inputs
unknown:l
	unknown_0:l
	unknown_1:l
	unknown_2:l
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_830985o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ä

*__inference_dense_629_layer_call_fn_835049

inputs
unknown:lV
	unknown_0:V
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_629_layer_call_and_return_conditional_losses_831422o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_834049

inputs5
'assignmovingavg_readvariableop_resource:-7
)assignmovingavg_1_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-/
!batchnorm_readvariableop_resource:-
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:-
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:-*
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
:-*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:-x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-¬
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
:-*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:-~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-´
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:-v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_559_layer_call_fn_834381

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
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_559_layer_call_and_return_conditional_losses_831218`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_834376

inputs5
'assignmovingavg_readvariableop_resource:-7
)assignmovingavg_1_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-/
!batchnorm_readvariableop_resource:-
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:-
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:-*
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
:-*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:-x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-¬
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
:-*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:-~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-´
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:-v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_834703

inputs5
'assignmovingavg_readvariableop_resource:l7
)assignmovingavg_1_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l/
!batchnorm_readvariableop_resource:l
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:l
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿll
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:l*
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
:l*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:lx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l¬
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
:l*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:l~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l´
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:lv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_834887

inputs/
!batchnorm_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l1
#batchnorm_readvariableop_1_resource:l1
#batchnorm_readvariableop_2_resource:l
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:lz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_834267

inputs5
'assignmovingavg_readvariableop_resource:-7
)assignmovingavg_1_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-/
!batchnorm_readvariableop_resource:-
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:-
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:-*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:-*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:-*
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
:-*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:-x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:-¬
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
:-*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:-~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:-´
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:-v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_561_layer_call_and_return_conditional_losses_834604

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
È	
ö
E__inference_dense_622_layer_call_and_return_conditional_losses_831198

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
È	
ö
E__inference_dense_626_layer_call_and_return_conditional_losses_834732

inputs0
matmul_readvariableop_resource:ll-
biasadd_readvariableop_resource:l
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ll*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:l*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_565_layer_call_fn_834963

inputs
unknown:l
	unknown_0:l
	unknown_1:l
	unknown_2:l
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_830938o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
×

.__inference_sequential_63_layer_call_fn_832917

inputs
unknown
	unknown_0
	unknown_1:-
	unknown_2:-
	unknown_3:-
	unknown_4:-
	unknown_5:-
	unknown_6:-
	unknown_7:--
	unknown_8:-
	unknown_9:-

unknown_10:-

unknown_11:-

unknown_12:-

unknown_13:--

unknown_14:-

unknown_15:-

unknown_16:-

unknown_17:-

unknown_18:-

unknown_19:--

unknown_20:-

unknown_21:-

unknown_22:-

unknown_23:-

unknown_24:-

unknown_25:--

unknown_26:-

unknown_27:-

unknown_28:-

unknown_29:-

unknown_30:-

unknown_31:-l

unknown_32:l

unknown_33:l

unknown_34:l

unknown_35:l

unknown_36:l

unknown_37:ll

unknown_38:l

unknown_39:l

unknown_40:l

unknown_41:l

unknown_42:l

unknown_43:ll

unknown_44:l

unknown_45:l

unknown_46:l

unknown_47:l

unknown_48:l

unknown_49:ll

unknown_50:l

unknown_51:l

unknown_52:l

unknown_53:l

unknown_54:l

unknown_55:ll

unknown_56:l

unknown_57:l

unknown_58:l

unknown_59:l

unknown_60:l

unknown_61:lV

unknown_62:V

unknown_63:V

unknown_64:V

unknown_65:V

unknown_66:V

unknown_67:V

unknown_68:
identity¢StatefulPartitionedCall

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
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEF*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_63_layer_call_and_return_conditional_losses_831461o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
%
ì
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_830985

inputs5
'assignmovingavg_readvariableop_resource:l7
)assignmovingavg_1_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l/
!batchnorm_readvariableop_resource:l
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:l
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿll
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:l*
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
:l*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:lx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l¬
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
:l*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:l~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l´
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:lv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_564_layer_call_fn_834854

inputs
unknown:l
	unknown_0:l
	unknown_1:l
	unknown_2:l
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_830856o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_556_layer_call_and_return_conditional_losses_834059

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Óß
ÍM
!__inference__wrapped_model_830176
normalization_63_input(
$sequential_63_normalization_63_sub_y)
%sequential_63_normalization_63_sqrt_xH
6sequential_63_dense_619_matmul_readvariableop_resource:-E
7sequential_63_dense_619_biasadd_readvariableop_resource:-U
Gsequential_63_batch_normalization_556_batchnorm_readvariableop_resource:-Y
Ksequential_63_batch_normalization_556_batchnorm_mul_readvariableop_resource:-W
Isequential_63_batch_normalization_556_batchnorm_readvariableop_1_resource:-W
Isequential_63_batch_normalization_556_batchnorm_readvariableop_2_resource:-H
6sequential_63_dense_620_matmul_readvariableop_resource:--E
7sequential_63_dense_620_biasadd_readvariableop_resource:-U
Gsequential_63_batch_normalization_557_batchnorm_readvariableop_resource:-Y
Ksequential_63_batch_normalization_557_batchnorm_mul_readvariableop_resource:-W
Isequential_63_batch_normalization_557_batchnorm_readvariableop_1_resource:-W
Isequential_63_batch_normalization_557_batchnorm_readvariableop_2_resource:-H
6sequential_63_dense_621_matmul_readvariableop_resource:--E
7sequential_63_dense_621_biasadd_readvariableop_resource:-U
Gsequential_63_batch_normalization_558_batchnorm_readvariableop_resource:-Y
Ksequential_63_batch_normalization_558_batchnorm_mul_readvariableop_resource:-W
Isequential_63_batch_normalization_558_batchnorm_readvariableop_1_resource:-W
Isequential_63_batch_normalization_558_batchnorm_readvariableop_2_resource:-H
6sequential_63_dense_622_matmul_readvariableop_resource:--E
7sequential_63_dense_622_biasadd_readvariableop_resource:-U
Gsequential_63_batch_normalization_559_batchnorm_readvariableop_resource:-Y
Ksequential_63_batch_normalization_559_batchnorm_mul_readvariableop_resource:-W
Isequential_63_batch_normalization_559_batchnorm_readvariableop_1_resource:-W
Isequential_63_batch_normalization_559_batchnorm_readvariableop_2_resource:-H
6sequential_63_dense_623_matmul_readvariableop_resource:--E
7sequential_63_dense_623_biasadd_readvariableop_resource:-U
Gsequential_63_batch_normalization_560_batchnorm_readvariableop_resource:-Y
Ksequential_63_batch_normalization_560_batchnorm_mul_readvariableop_resource:-W
Isequential_63_batch_normalization_560_batchnorm_readvariableop_1_resource:-W
Isequential_63_batch_normalization_560_batchnorm_readvariableop_2_resource:-H
6sequential_63_dense_624_matmul_readvariableop_resource:-lE
7sequential_63_dense_624_biasadd_readvariableop_resource:lU
Gsequential_63_batch_normalization_561_batchnorm_readvariableop_resource:lY
Ksequential_63_batch_normalization_561_batchnorm_mul_readvariableop_resource:lW
Isequential_63_batch_normalization_561_batchnorm_readvariableop_1_resource:lW
Isequential_63_batch_normalization_561_batchnorm_readvariableop_2_resource:lH
6sequential_63_dense_625_matmul_readvariableop_resource:llE
7sequential_63_dense_625_biasadd_readvariableop_resource:lU
Gsequential_63_batch_normalization_562_batchnorm_readvariableop_resource:lY
Ksequential_63_batch_normalization_562_batchnorm_mul_readvariableop_resource:lW
Isequential_63_batch_normalization_562_batchnorm_readvariableop_1_resource:lW
Isequential_63_batch_normalization_562_batchnorm_readvariableop_2_resource:lH
6sequential_63_dense_626_matmul_readvariableop_resource:llE
7sequential_63_dense_626_biasadd_readvariableop_resource:lU
Gsequential_63_batch_normalization_563_batchnorm_readvariableop_resource:lY
Ksequential_63_batch_normalization_563_batchnorm_mul_readvariableop_resource:lW
Isequential_63_batch_normalization_563_batchnorm_readvariableop_1_resource:lW
Isequential_63_batch_normalization_563_batchnorm_readvariableop_2_resource:lH
6sequential_63_dense_627_matmul_readvariableop_resource:llE
7sequential_63_dense_627_biasadd_readvariableop_resource:lU
Gsequential_63_batch_normalization_564_batchnorm_readvariableop_resource:lY
Ksequential_63_batch_normalization_564_batchnorm_mul_readvariableop_resource:lW
Isequential_63_batch_normalization_564_batchnorm_readvariableop_1_resource:lW
Isequential_63_batch_normalization_564_batchnorm_readvariableop_2_resource:lH
6sequential_63_dense_628_matmul_readvariableop_resource:llE
7sequential_63_dense_628_biasadd_readvariableop_resource:lU
Gsequential_63_batch_normalization_565_batchnorm_readvariableop_resource:lY
Ksequential_63_batch_normalization_565_batchnorm_mul_readvariableop_resource:lW
Isequential_63_batch_normalization_565_batchnorm_readvariableop_1_resource:lW
Isequential_63_batch_normalization_565_batchnorm_readvariableop_2_resource:lH
6sequential_63_dense_629_matmul_readvariableop_resource:lVE
7sequential_63_dense_629_biasadd_readvariableop_resource:VU
Gsequential_63_batch_normalization_566_batchnorm_readvariableop_resource:VY
Ksequential_63_batch_normalization_566_batchnorm_mul_readvariableop_resource:VW
Isequential_63_batch_normalization_566_batchnorm_readvariableop_1_resource:VW
Isequential_63_batch_normalization_566_batchnorm_readvariableop_2_resource:VH
6sequential_63_dense_630_matmul_readvariableop_resource:VE
7sequential_63_dense_630_biasadd_readvariableop_resource:
identity¢>sequential_63/batch_normalization_556/batchnorm/ReadVariableOp¢@sequential_63/batch_normalization_556/batchnorm/ReadVariableOp_1¢@sequential_63/batch_normalization_556/batchnorm/ReadVariableOp_2¢Bsequential_63/batch_normalization_556/batchnorm/mul/ReadVariableOp¢>sequential_63/batch_normalization_557/batchnorm/ReadVariableOp¢@sequential_63/batch_normalization_557/batchnorm/ReadVariableOp_1¢@sequential_63/batch_normalization_557/batchnorm/ReadVariableOp_2¢Bsequential_63/batch_normalization_557/batchnorm/mul/ReadVariableOp¢>sequential_63/batch_normalization_558/batchnorm/ReadVariableOp¢@sequential_63/batch_normalization_558/batchnorm/ReadVariableOp_1¢@sequential_63/batch_normalization_558/batchnorm/ReadVariableOp_2¢Bsequential_63/batch_normalization_558/batchnorm/mul/ReadVariableOp¢>sequential_63/batch_normalization_559/batchnorm/ReadVariableOp¢@sequential_63/batch_normalization_559/batchnorm/ReadVariableOp_1¢@sequential_63/batch_normalization_559/batchnorm/ReadVariableOp_2¢Bsequential_63/batch_normalization_559/batchnorm/mul/ReadVariableOp¢>sequential_63/batch_normalization_560/batchnorm/ReadVariableOp¢@sequential_63/batch_normalization_560/batchnorm/ReadVariableOp_1¢@sequential_63/batch_normalization_560/batchnorm/ReadVariableOp_2¢Bsequential_63/batch_normalization_560/batchnorm/mul/ReadVariableOp¢>sequential_63/batch_normalization_561/batchnorm/ReadVariableOp¢@sequential_63/batch_normalization_561/batchnorm/ReadVariableOp_1¢@sequential_63/batch_normalization_561/batchnorm/ReadVariableOp_2¢Bsequential_63/batch_normalization_561/batchnorm/mul/ReadVariableOp¢>sequential_63/batch_normalization_562/batchnorm/ReadVariableOp¢@sequential_63/batch_normalization_562/batchnorm/ReadVariableOp_1¢@sequential_63/batch_normalization_562/batchnorm/ReadVariableOp_2¢Bsequential_63/batch_normalization_562/batchnorm/mul/ReadVariableOp¢>sequential_63/batch_normalization_563/batchnorm/ReadVariableOp¢@sequential_63/batch_normalization_563/batchnorm/ReadVariableOp_1¢@sequential_63/batch_normalization_563/batchnorm/ReadVariableOp_2¢Bsequential_63/batch_normalization_563/batchnorm/mul/ReadVariableOp¢>sequential_63/batch_normalization_564/batchnorm/ReadVariableOp¢@sequential_63/batch_normalization_564/batchnorm/ReadVariableOp_1¢@sequential_63/batch_normalization_564/batchnorm/ReadVariableOp_2¢Bsequential_63/batch_normalization_564/batchnorm/mul/ReadVariableOp¢>sequential_63/batch_normalization_565/batchnorm/ReadVariableOp¢@sequential_63/batch_normalization_565/batchnorm/ReadVariableOp_1¢@sequential_63/batch_normalization_565/batchnorm/ReadVariableOp_2¢Bsequential_63/batch_normalization_565/batchnorm/mul/ReadVariableOp¢>sequential_63/batch_normalization_566/batchnorm/ReadVariableOp¢@sequential_63/batch_normalization_566/batchnorm/ReadVariableOp_1¢@sequential_63/batch_normalization_566/batchnorm/ReadVariableOp_2¢Bsequential_63/batch_normalization_566/batchnorm/mul/ReadVariableOp¢.sequential_63/dense_619/BiasAdd/ReadVariableOp¢-sequential_63/dense_619/MatMul/ReadVariableOp¢.sequential_63/dense_620/BiasAdd/ReadVariableOp¢-sequential_63/dense_620/MatMul/ReadVariableOp¢.sequential_63/dense_621/BiasAdd/ReadVariableOp¢-sequential_63/dense_621/MatMul/ReadVariableOp¢.sequential_63/dense_622/BiasAdd/ReadVariableOp¢-sequential_63/dense_622/MatMul/ReadVariableOp¢.sequential_63/dense_623/BiasAdd/ReadVariableOp¢-sequential_63/dense_623/MatMul/ReadVariableOp¢.sequential_63/dense_624/BiasAdd/ReadVariableOp¢-sequential_63/dense_624/MatMul/ReadVariableOp¢.sequential_63/dense_625/BiasAdd/ReadVariableOp¢-sequential_63/dense_625/MatMul/ReadVariableOp¢.sequential_63/dense_626/BiasAdd/ReadVariableOp¢-sequential_63/dense_626/MatMul/ReadVariableOp¢.sequential_63/dense_627/BiasAdd/ReadVariableOp¢-sequential_63/dense_627/MatMul/ReadVariableOp¢.sequential_63/dense_628/BiasAdd/ReadVariableOp¢-sequential_63/dense_628/MatMul/ReadVariableOp¢.sequential_63/dense_629/BiasAdd/ReadVariableOp¢-sequential_63/dense_629/MatMul/ReadVariableOp¢.sequential_63/dense_630/BiasAdd/ReadVariableOp¢-sequential_63/dense_630/MatMul/ReadVariableOp
"sequential_63/normalization_63/subSubnormalization_63_input$sequential_63_normalization_63_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#sequential_63/normalization_63/SqrtSqrt%sequential_63_normalization_63_sqrt_x*
T0*
_output_shapes

:m
(sequential_63/normalization_63/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¶
&sequential_63/normalization_63/MaximumMaximum'sequential_63/normalization_63/Sqrt:y:01sequential_63/normalization_63/Maximum/y:output:0*
T0*
_output_shapes

:·
&sequential_63/normalization_63/truedivRealDiv&sequential_63/normalization_63/sub:z:0*sequential_63/normalization_63/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_63/dense_619/MatMul/ReadVariableOpReadVariableOp6sequential_63_dense_619_matmul_readvariableop_resource*
_output_shapes

:-*
dtype0½
sequential_63/dense_619/MatMulMatMul*sequential_63/normalization_63/truediv:z:05sequential_63/dense_619/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¢
.sequential_63/dense_619/BiasAdd/ReadVariableOpReadVariableOp7sequential_63_dense_619_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0¾
sequential_63/dense_619/BiasAddBiasAdd(sequential_63/dense_619/MatMul:product:06sequential_63/dense_619/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-Â
>sequential_63/batch_normalization_556/batchnorm/ReadVariableOpReadVariableOpGsequential_63_batch_normalization_556_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0z
5sequential_63/batch_normalization_556/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_63/batch_normalization_556/batchnorm/addAddV2Fsequential_63/batch_normalization_556/batchnorm/ReadVariableOp:value:0>sequential_63/batch_normalization_556/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
5sequential_63/batch_normalization_556/batchnorm/RsqrtRsqrt7sequential_63/batch_normalization_556/batchnorm/add:z:0*
T0*
_output_shapes
:-Ê
Bsequential_63/batch_normalization_556/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_63_batch_normalization_556_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0æ
3sequential_63/batch_normalization_556/batchnorm/mulMul9sequential_63/batch_normalization_556/batchnorm/Rsqrt:y:0Jsequential_63/batch_normalization_556/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-Ñ
5sequential_63/batch_normalization_556/batchnorm/mul_1Mul(sequential_63/dense_619/BiasAdd:output:07sequential_63/batch_normalization_556/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-Æ
@sequential_63/batch_normalization_556/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_63_batch_normalization_556_batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0ä
5sequential_63/batch_normalization_556/batchnorm/mul_2MulHsequential_63/batch_normalization_556/batchnorm/ReadVariableOp_1:value:07sequential_63/batch_normalization_556/batchnorm/mul:z:0*
T0*
_output_shapes
:-Æ
@sequential_63/batch_normalization_556/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_63_batch_normalization_556_batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0ä
3sequential_63/batch_normalization_556/batchnorm/subSubHsequential_63/batch_normalization_556/batchnorm/ReadVariableOp_2:value:09sequential_63/batch_normalization_556/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-ä
5sequential_63/batch_normalization_556/batchnorm/add_1AddV29sequential_63/batch_normalization_556/batchnorm/mul_1:z:07sequential_63/batch_normalization_556/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¨
'sequential_63/leaky_re_lu_556/LeakyRelu	LeakyRelu9sequential_63/batch_normalization_556/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>¤
-sequential_63/dense_620/MatMul/ReadVariableOpReadVariableOp6sequential_63_dense_620_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0È
sequential_63/dense_620/MatMulMatMul5sequential_63/leaky_re_lu_556/LeakyRelu:activations:05sequential_63/dense_620/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¢
.sequential_63/dense_620/BiasAdd/ReadVariableOpReadVariableOp7sequential_63_dense_620_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0¾
sequential_63/dense_620/BiasAddBiasAdd(sequential_63/dense_620/MatMul:product:06sequential_63/dense_620/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-Â
>sequential_63/batch_normalization_557/batchnorm/ReadVariableOpReadVariableOpGsequential_63_batch_normalization_557_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0z
5sequential_63/batch_normalization_557/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_63/batch_normalization_557/batchnorm/addAddV2Fsequential_63/batch_normalization_557/batchnorm/ReadVariableOp:value:0>sequential_63/batch_normalization_557/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
5sequential_63/batch_normalization_557/batchnorm/RsqrtRsqrt7sequential_63/batch_normalization_557/batchnorm/add:z:0*
T0*
_output_shapes
:-Ê
Bsequential_63/batch_normalization_557/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_63_batch_normalization_557_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0æ
3sequential_63/batch_normalization_557/batchnorm/mulMul9sequential_63/batch_normalization_557/batchnorm/Rsqrt:y:0Jsequential_63/batch_normalization_557/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-Ñ
5sequential_63/batch_normalization_557/batchnorm/mul_1Mul(sequential_63/dense_620/BiasAdd:output:07sequential_63/batch_normalization_557/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-Æ
@sequential_63/batch_normalization_557/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_63_batch_normalization_557_batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0ä
5sequential_63/batch_normalization_557/batchnorm/mul_2MulHsequential_63/batch_normalization_557/batchnorm/ReadVariableOp_1:value:07sequential_63/batch_normalization_557/batchnorm/mul:z:0*
T0*
_output_shapes
:-Æ
@sequential_63/batch_normalization_557/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_63_batch_normalization_557_batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0ä
3sequential_63/batch_normalization_557/batchnorm/subSubHsequential_63/batch_normalization_557/batchnorm/ReadVariableOp_2:value:09sequential_63/batch_normalization_557/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-ä
5sequential_63/batch_normalization_557/batchnorm/add_1AddV29sequential_63/batch_normalization_557/batchnorm/mul_1:z:07sequential_63/batch_normalization_557/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¨
'sequential_63/leaky_re_lu_557/LeakyRelu	LeakyRelu9sequential_63/batch_normalization_557/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>¤
-sequential_63/dense_621/MatMul/ReadVariableOpReadVariableOp6sequential_63_dense_621_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0È
sequential_63/dense_621/MatMulMatMul5sequential_63/leaky_re_lu_557/LeakyRelu:activations:05sequential_63/dense_621/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¢
.sequential_63/dense_621/BiasAdd/ReadVariableOpReadVariableOp7sequential_63_dense_621_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0¾
sequential_63/dense_621/BiasAddBiasAdd(sequential_63/dense_621/MatMul:product:06sequential_63/dense_621/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-Â
>sequential_63/batch_normalization_558/batchnorm/ReadVariableOpReadVariableOpGsequential_63_batch_normalization_558_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0z
5sequential_63/batch_normalization_558/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_63/batch_normalization_558/batchnorm/addAddV2Fsequential_63/batch_normalization_558/batchnorm/ReadVariableOp:value:0>sequential_63/batch_normalization_558/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
5sequential_63/batch_normalization_558/batchnorm/RsqrtRsqrt7sequential_63/batch_normalization_558/batchnorm/add:z:0*
T0*
_output_shapes
:-Ê
Bsequential_63/batch_normalization_558/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_63_batch_normalization_558_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0æ
3sequential_63/batch_normalization_558/batchnorm/mulMul9sequential_63/batch_normalization_558/batchnorm/Rsqrt:y:0Jsequential_63/batch_normalization_558/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-Ñ
5sequential_63/batch_normalization_558/batchnorm/mul_1Mul(sequential_63/dense_621/BiasAdd:output:07sequential_63/batch_normalization_558/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-Æ
@sequential_63/batch_normalization_558/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_63_batch_normalization_558_batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0ä
5sequential_63/batch_normalization_558/batchnorm/mul_2MulHsequential_63/batch_normalization_558/batchnorm/ReadVariableOp_1:value:07sequential_63/batch_normalization_558/batchnorm/mul:z:0*
T0*
_output_shapes
:-Æ
@sequential_63/batch_normalization_558/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_63_batch_normalization_558_batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0ä
3sequential_63/batch_normalization_558/batchnorm/subSubHsequential_63/batch_normalization_558/batchnorm/ReadVariableOp_2:value:09sequential_63/batch_normalization_558/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-ä
5sequential_63/batch_normalization_558/batchnorm/add_1AddV29sequential_63/batch_normalization_558/batchnorm/mul_1:z:07sequential_63/batch_normalization_558/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¨
'sequential_63/leaky_re_lu_558/LeakyRelu	LeakyRelu9sequential_63/batch_normalization_558/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>¤
-sequential_63/dense_622/MatMul/ReadVariableOpReadVariableOp6sequential_63_dense_622_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0È
sequential_63/dense_622/MatMulMatMul5sequential_63/leaky_re_lu_558/LeakyRelu:activations:05sequential_63/dense_622/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¢
.sequential_63/dense_622/BiasAdd/ReadVariableOpReadVariableOp7sequential_63_dense_622_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0¾
sequential_63/dense_622/BiasAddBiasAdd(sequential_63/dense_622/MatMul:product:06sequential_63/dense_622/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-Â
>sequential_63/batch_normalization_559/batchnorm/ReadVariableOpReadVariableOpGsequential_63_batch_normalization_559_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0z
5sequential_63/batch_normalization_559/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_63/batch_normalization_559/batchnorm/addAddV2Fsequential_63/batch_normalization_559/batchnorm/ReadVariableOp:value:0>sequential_63/batch_normalization_559/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
5sequential_63/batch_normalization_559/batchnorm/RsqrtRsqrt7sequential_63/batch_normalization_559/batchnorm/add:z:0*
T0*
_output_shapes
:-Ê
Bsequential_63/batch_normalization_559/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_63_batch_normalization_559_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0æ
3sequential_63/batch_normalization_559/batchnorm/mulMul9sequential_63/batch_normalization_559/batchnorm/Rsqrt:y:0Jsequential_63/batch_normalization_559/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-Ñ
5sequential_63/batch_normalization_559/batchnorm/mul_1Mul(sequential_63/dense_622/BiasAdd:output:07sequential_63/batch_normalization_559/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-Æ
@sequential_63/batch_normalization_559/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_63_batch_normalization_559_batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0ä
5sequential_63/batch_normalization_559/batchnorm/mul_2MulHsequential_63/batch_normalization_559/batchnorm/ReadVariableOp_1:value:07sequential_63/batch_normalization_559/batchnorm/mul:z:0*
T0*
_output_shapes
:-Æ
@sequential_63/batch_normalization_559/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_63_batch_normalization_559_batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0ä
3sequential_63/batch_normalization_559/batchnorm/subSubHsequential_63/batch_normalization_559/batchnorm/ReadVariableOp_2:value:09sequential_63/batch_normalization_559/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-ä
5sequential_63/batch_normalization_559/batchnorm/add_1AddV29sequential_63/batch_normalization_559/batchnorm/mul_1:z:07sequential_63/batch_normalization_559/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¨
'sequential_63/leaky_re_lu_559/LeakyRelu	LeakyRelu9sequential_63/batch_normalization_559/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>¤
-sequential_63/dense_623/MatMul/ReadVariableOpReadVariableOp6sequential_63_dense_623_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0È
sequential_63/dense_623/MatMulMatMul5sequential_63/leaky_re_lu_559/LeakyRelu:activations:05sequential_63/dense_623/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¢
.sequential_63/dense_623/BiasAdd/ReadVariableOpReadVariableOp7sequential_63_dense_623_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0¾
sequential_63/dense_623/BiasAddBiasAdd(sequential_63/dense_623/MatMul:product:06sequential_63/dense_623/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-Â
>sequential_63/batch_normalization_560/batchnorm/ReadVariableOpReadVariableOpGsequential_63_batch_normalization_560_batchnorm_readvariableop_resource*
_output_shapes
:-*
dtype0z
5sequential_63/batch_normalization_560/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_63/batch_normalization_560/batchnorm/addAddV2Fsequential_63/batch_normalization_560/batchnorm/ReadVariableOp:value:0>sequential_63/batch_normalization_560/batchnorm/add/y:output:0*
T0*
_output_shapes
:-
5sequential_63/batch_normalization_560/batchnorm/RsqrtRsqrt7sequential_63/batch_normalization_560/batchnorm/add:z:0*
T0*
_output_shapes
:-Ê
Bsequential_63/batch_normalization_560/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_63_batch_normalization_560_batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0æ
3sequential_63/batch_normalization_560/batchnorm/mulMul9sequential_63/batch_normalization_560/batchnorm/Rsqrt:y:0Jsequential_63/batch_normalization_560/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-Ñ
5sequential_63/batch_normalization_560/batchnorm/mul_1Mul(sequential_63/dense_623/BiasAdd:output:07sequential_63/batch_normalization_560/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-Æ
@sequential_63/batch_normalization_560/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_63_batch_normalization_560_batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0ä
5sequential_63/batch_normalization_560/batchnorm/mul_2MulHsequential_63/batch_normalization_560/batchnorm/ReadVariableOp_1:value:07sequential_63/batch_normalization_560/batchnorm/mul:z:0*
T0*
_output_shapes
:-Æ
@sequential_63/batch_normalization_560/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_63_batch_normalization_560_batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0ä
3sequential_63/batch_normalization_560/batchnorm/subSubHsequential_63/batch_normalization_560/batchnorm/ReadVariableOp_2:value:09sequential_63/batch_normalization_560/batchnorm/mul_2:z:0*
T0*
_output_shapes
:-ä
5sequential_63/batch_normalization_560/batchnorm/add_1AddV29sequential_63/batch_normalization_560/batchnorm/mul_1:z:07sequential_63/batch_normalization_560/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-¨
'sequential_63/leaky_re_lu_560/LeakyRelu	LeakyRelu9sequential_63/batch_normalization_560/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>¤
-sequential_63/dense_624/MatMul/ReadVariableOpReadVariableOp6sequential_63_dense_624_matmul_readvariableop_resource*
_output_shapes

:-l*
dtype0È
sequential_63/dense_624/MatMulMatMul5sequential_63/leaky_re_lu_560/LeakyRelu:activations:05sequential_63/dense_624/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¢
.sequential_63/dense_624/BiasAdd/ReadVariableOpReadVariableOp7sequential_63_dense_624_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0¾
sequential_63/dense_624/BiasAddBiasAdd(sequential_63/dense_624/MatMul:product:06sequential_63/dense_624/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlÂ
>sequential_63/batch_normalization_561/batchnorm/ReadVariableOpReadVariableOpGsequential_63_batch_normalization_561_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0z
5sequential_63/batch_normalization_561/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_63/batch_normalization_561/batchnorm/addAddV2Fsequential_63/batch_normalization_561/batchnorm/ReadVariableOp:value:0>sequential_63/batch_normalization_561/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
5sequential_63/batch_normalization_561/batchnorm/RsqrtRsqrt7sequential_63/batch_normalization_561/batchnorm/add:z:0*
T0*
_output_shapes
:lÊ
Bsequential_63/batch_normalization_561/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_63_batch_normalization_561_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0æ
3sequential_63/batch_normalization_561/batchnorm/mulMul9sequential_63/batch_normalization_561/batchnorm/Rsqrt:y:0Jsequential_63/batch_normalization_561/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lÑ
5sequential_63/batch_normalization_561/batchnorm/mul_1Mul(sequential_63/dense_624/BiasAdd:output:07sequential_63/batch_normalization_561/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlÆ
@sequential_63/batch_normalization_561/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_63_batch_normalization_561_batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0ä
5sequential_63/batch_normalization_561/batchnorm/mul_2MulHsequential_63/batch_normalization_561/batchnorm/ReadVariableOp_1:value:07sequential_63/batch_normalization_561/batchnorm/mul:z:0*
T0*
_output_shapes
:lÆ
@sequential_63/batch_normalization_561/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_63_batch_normalization_561_batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0ä
3sequential_63/batch_normalization_561/batchnorm/subSubHsequential_63/batch_normalization_561/batchnorm/ReadVariableOp_2:value:09sequential_63/batch_normalization_561/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lä
5sequential_63/batch_normalization_561/batchnorm/add_1AddV29sequential_63/batch_normalization_561/batchnorm/mul_1:z:07sequential_63/batch_normalization_561/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¨
'sequential_63/leaky_re_lu_561/LeakyRelu	LeakyRelu9sequential_63/batch_normalization_561/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>¤
-sequential_63/dense_625/MatMul/ReadVariableOpReadVariableOp6sequential_63_dense_625_matmul_readvariableop_resource*
_output_shapes

:ll*
dtype0È
sequential_63/dense_625/MatMulMatMul5sequential_63/leaky_re_lu_561/LeakyRelu:activations:05sequential_63/dense_625/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¢
.sequential_63/dense_625/BiasAdd/ReadVariableOpReadVariableOp7sequential_63_dense_625_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0¾
sequential_63/dense_625/BiasAddBiasAdd(sequential_63/dense_625/MatMul:product:06sequential_63/dense_625/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlÂ
>sequential_63/batch_normalization_562/batchnorm/ReadVariableOpReadVariableOpGsequential_63_batch_normalization_562_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0z
5sequential_63/batch_normalization_562/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_63/batch_normalization_562/batchnorm/addAddV2Fsequential_63/batch_normalization_562/batchnorm/ReadVariableOp:value:0>sequential_63/batch_normalization_562/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
5sequential_63/batch_normalization_562/batchnorm/RsqrtRsqrt7sequential_63/batch_normalization_562/batchnorm/add:z:0*
T0*
_output_shapes
:lÊ
Bsequential_63/batch_normalization_562/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_63_batch_normalization_562_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0æ
3sequential_63/batch_normalization_562/batchnorm/mulMul9sequential_63/batch_normalization_562/batchnorm/Rsqrt:y:0Jsequential_63/batch_normalization_562/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lÑ
5sequential_63/batch_normalization_562/batchnorm/mul_1Mul(sequential_63/dense_625/BiasAdd:output:07sequential_63/batch_normalization_562/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlÆ
@sequential_63/batch_normalization_562/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_63_batch_normalization_562_batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0ä
5sequential_63/batch_normalization_562/batchnorm/mul_2MulHsequential_63/batch_normalization_562/batchnorm/ReadVariableOp_1:value:07sequential_63/batch_normalization_562/batchnorm/mul:z:0*
T0*
_output_shapes
:lÆ
@sequential_63/batch_normalization_562/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_63_batch_normalization_562_batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0ä
3sequential_63/batch_normalization_562/batchnorm/subSubHsequential_63/batch_normalization_562/batchnorm/ReadVariableOp_2:value:09sequential_63/batch_normalization_562/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lä
5sequential_63/batch_normalization_562/batchnorm/add_1AddV29sequential_63/batch_normalization_562/batchnorm/mul_1:z:07sequential_63/batch_normalization_562/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¨
'sequential_63/leaky_re_lu_562/LeakyRelu	LeakyRelu9sequential_63/batch_normalization_562/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>¤
-sequential_63/dense_626/MatMul/ReadVariableOpReadVariableOp6sequential_63_dense_626_matmul_readvariableop_resource*
_output_shapes

:ll*
dtype0È
sequential_63/dense_626/MatMulMatMul5sequential_63/leaky_re_lu_562/LeakyRelu:activations:05sequential_63/dense_626/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¢
.sequential_63/dense_626/BiasAdd/ReadVariableOpReadVariableOp7sequential_63_dense_626_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0¾
sequential_63/dense_626/BiasAddBiasAdd(sequential_63/dense_626/MatMul:product:06sequential_63/dense_626/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlÂ
>sequential_63/batch_normalization_563/batchnorm/ReadVariableOpReadVariableOpGsequential_63_batch_normalization_563_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0z
5sequential_63/batch_normalization_563/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_63/batch_normalization_563/batchnorm/addAddV2Fsequential_63/batch_normalization_563/batchnorm/ReadVariableOp:value:0>sequential_63/batch_normalization_563/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
5sequential_63/batch_normalization_563/batchnorm/RsqrtRsqrt7sequential_63/batch_normalization_563/batchnorm/add:z:0*
T0*
_output_shapes
:lÊ
Bsequential_63/batch_normalization_563/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_63_batch_normalization_563_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0æ
3sequential_63/batch_normalization_563/batchnorm/mulMul9sequential_63/batch_normalization_563/batchnorm/Rsqrt:y:0Jsequential_63/batch_normalization_563/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lÑ
5sequential_63/batch_normalization_563/batchnorm/mul_1Mul(sequential_63/dense_626/BiasAdd:output:07sequential_63/batch_normalization_563/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlÆ
@sequential_63/batch_normalization_563/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_63_batch_normalization_563_batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0ä
5sequential_63/batch_normalization_563/batchnorm/mul_2MulHsequential_63/batch_normalization_563/batchnorm/ReadVariableOp_1:value:07sequential_63/batch_normalization_563/batchnorm/mul:z:0*
T0*
_output_shapes
:lÆ
@sequential_63/batch_normalization_563/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_63_batch_normalization_563_batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0ä
3sequential_63/batch_normalization_563/batchnorm/subSubHsequential_63/batch_normalization_563/batchnorm/ReadVariableOp_2:value:09sequential_63/batch_normalization_563/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lä
5sequential_63/batch_normalization_563/batchnorm/add_1AddV29sequential_63/batch_normalization_563/batchnorm/mul_1:z:07sequential_63/batch_normalization_563/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¨
'sequential_63/leaky_re_lu_563/LeakyRelu	LeakyRelu9sequential_63/batch_normalization_563/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>¤
-sequential_63/dense_627/MatMul/ReadVariableOpReadVariableOp6sequential_63_dense_627_matmul_readvariableop_resource*
_output_shapes

:ll*
dtype0È
sequential_63/dense_627/MatMulMatMul5sequential_63/leaky_re_lu_563/LeakyRelu:activations:05sequential_63/dense_627/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¢
.sequential_63/dense_627/BiasAdd/ReadVariableOpReadVariableOp7sequential_63_dense_627_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0¾
sequential_63/dense_627/BiasAddBiasAdd(sequential_63/dense_627/MatMul:product:06sequential_63/dense_627/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlÂ
>sequential_63/batch_normalization_564/batchnorm/ReadVariableOpReadVariableOpGsequential_63_batch_normalization_564_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0z
5sequential_63/batch_normalization_564/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_63/batch_normalization_564/batchnorm/addAddV2Fsequential_63/batch_normalization_564/batchnorm/ReadVariableOp:value:0>sequential_63/batch_normalization_564/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
5sequential_63/batch_normalization_564/batchnorm/RsqrtRsqrt7sequential_63/batch_normalization_564/batchnorm/add:z:0*
T0*
_output_shapes
:lÊ
Bsequential_63/batch_normalization_564/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_63_batch_normalization_564_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0æ
3sequential_63/batch_normalization_564/batchnorm/mulMul9sequential_63/batch_normalization_564/batchnorm/Rsqrt:y:0Jsequential_63/batch_normalization_564/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lÑ
5sequential_63/batch_normalization_564/batchnorm/mul_1Mul(sequential_63/dense_627/BiasAdd:output:07sequential_63/batch_normalization_564/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlÆ
@sequential_63/batch_normalization_564/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_63_batch_normalization_564_batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0ä
5sequential_63/batch_normalization_564/batchnorm/mul_2MulHsequential_63/batch_normalization_564/batchnorm/ReadVariableOp_1:value:07sequential_63/batch_normalization_564/batchnorm/mul:z:0*
T0*
_output_shapes
:lÆ
@sequential_63/batch_normalization_564/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_63_batch_normalization_564_batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0ä
3sequential_63/batch_normalization_564/batchnorm/subSubHsequential_63/batch_normalization_564/batchnorm/ReadVariableOp_2:value:09sequential_63/batch_normalization_564/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lä
5sequential_63/batch_normalization_564/batchnorm/add_1AddV29sequential_63/batch_normalization_564/batchnorm/mul_1:z:07sequential_63/batch_normalization_564/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¨
'sequential_63/leaky_re_lu_564/LeakyRelu	LeakyRelu9sequential_63/batch_normalization_564/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>¤
-sequential_63/dense_628/MatMul/ReadVariableOpReadVariableOp6sequential_63_dense_628_matmul_readvariableop_resource*
_output_shapes

:ll*
dtype0È
sequential_63/dense_628/MatMulMatMul5sequential_63/leaky_re_lu_564/LeakyRelu:activations:05sequential_63/dense_628/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¢
.sequential_63/dense_628/BiasAdd/ReadVariableOpReadVariableOp7sequential_63_dense_628_biasadd_readvariableop_resource*
_output_shapes
:l*
dtype0¾
sequential_63/dense_628/BiasAddBiasAdd(sequential_63/dense_628/MatMul:product:06sequential_63/dense_628/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlÂ
>sequential_63/batch_normalization_565/batchnorm/ReadVariableOpReadVariableOpGsequential_63_batch_normalization_565_batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0z
5sequential_63/batch_normalization_565/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_63/batch_normalization_565/batchnorm/addAddV2Fsequential_63/batch_normalization_565/batchnorm/ReadVariableOp:value:0>sequential_63/batch_normalization_565/batchnorm/add/y:output:0*
T0*
_output_shapes
:l
5sequential_63/batch_normalization_565/batchnorm/RsqrtRsqrt7sequential_63/batch_normalization_565/batchnorm/add:z:0*
T0*
_output_shapes
:lÊ
Bsequential_63/batch_normalization_565/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_63_batch_normalization_565_batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0æ
3sequential_63/batch_normalization_565/batchnorm/mulMul9sequential_63/batch_normalization_565/batchnorm/Rsqrt:y:0Jsequential_63/batch_normalization_565/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lÑ
5sequential_63/batch_normalization_565/batchnorm/mul_1Mul(sequential_63/dense_628/BiasAdd:output:07sequential_63/batch_normalization_565/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlÆ
@sequential_63/batch_normalization_565/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_63_batch_normalization_565_batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0ä
5sequential_63/batch_normalization_565/batchnorm/mul_2MulHsequential_63/batch_normalization_565/batchnorm/ReadVariableOp_1:value:07sequential_63/batch_normalization_565/batchnorm/mul:z:0*
T0*
_output_shapes
:lÆ
@sequential_63/batch_normalization_565/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_63_batch_normalization_565_batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0ä
3sequential_63/batch_normalization_565/batchnorm/subSubHsequential_63/batch_normalization_565/batchnorm/ReadVariableOp_2:value:09sequential_63/batch_normalization_565/batchnorm/mul_2:z:0*
T0*
_output_shapes
:lä
5sequential_63/batch_normalization_565/batchnorm/add_1AddV29sequential_63/batch_normalization_565/batchnorm/mul_1:z:07sequential_63/batch_normalization_565/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl¨
'sequential_63/leaky_re_lu_565/LeakyRelu	LeakyRelu9sequential_63/batch_normalization_565/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>¤
-sequential_63/dense_629/MatMul/ReadVariableOpReadVariableOp6sequential_63_dense_629_matmul_readvariableop_resource*
_output_shapes

:lV*
dtype0È
sequential_63/dense_629/MatMulMatMul5sequential_63/leaky_re_lu_565/LeakyRelu:activations:05sequential_63/dense_629/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV¢
.sequential_63/dense_629/BiasAdd/ReadVariableOpReadVariableOp7sequential_63_dense_629_biasadd_readvariableop_resource*
_output_shapes
:V*
dtype0¾
sequential_63/dense_629/BiasAddBiasAdd(sequential_63/dense_629/MatMul:product:06sequential_63/dense_629/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVÂ
>sequential_63/batch_normalization_566/batchnorm/ReadVariableOpReadVariableOpGsequential_63_batch_normalization_566_batchnorm_readvariableop_resource*
_output_shapes
:V*
dtype0z
5sequential_63/batch_normalization_566/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_63/batch_normalization_566/batchnorm/addAddV2Fsequential_63/batch_normalization_566/batchnorm/ReadVariableOp:value:0>sequential_63/batch_normalization_566/batchnorm/add/y:output:0*
T0*
_output_shapes
:V
5sequential_63/batch_normalization_566/batchnorm/RsqrtRsqrt7sequential_63/batch_normalization_566/batchnorm/add:z:0*
T0*
_output_shapes
:VÊ
Bsequential_63/batch_normalization_566/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_63_batch_normalization_566_batchnorm_mul_readvariableop_resource*
_output_shapes
:V*
dtype0æ
3sequential_63/batch_normalization_566/batchnorm/mulMul9sequential_63/batch_normalization_566/batchnorm/Rsqrt:y:0Jsequential_63/batch_normalization_566/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:VÑ
5sequential_63/batch_normalization_566/batchnorm/mul_1Mul(sequential_63/dense_629/BiasAdd:output:07sequential_63/batch_normalization_566/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVÆ
@sequential_63/batch_normalization_566/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_63_batch_normalization_566_batchnorm_readvariableop_1_resource*
_output_shapes
:V*
dtype0ä
5sequential_63/batch_normalization_566/batchnorm/mul_2MulHsequential_63/batch_normalization_566/batchnorm/ReadVariableOp_1:value:07sequential_63/batch_normalization_566/batchnorm/mul:z:0*
T0*
_output_shapes
:VÆ
@sequential_63/batch_normalization_566/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_63_batch_normalization_566_batchnorm_readvariableop_2_resource*
_output_shapes
:V*
dtype0ä
3sequential_63/batch_normalization_566/batchnorm/subSubHsequential_63/batch_normalization_566/batchnorm/ReadVariableOp_2:value:09sequential_63/batch_normalization_566/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Vä
5sequential_63/batch_normalization_566/batchnorm/add_1AddV29sequential_63/batch_normalization_566/batchnorm/mul_1:z:07sequential_63/batch_normalization_566/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV¨
'sequential_63/leaky_re_lu_566/LeakyRelu	LeakyRelu9sequential_63/batch_normalization_566/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV*
alpha%>¤
-sequential_63/dense_630/MatMul/ReadVariableOpReadVariableOp6sequential_63_dense_630_matmul_readvariableop_resource*
_output_shapes

:V*
dtype0È
sequential_63/dense_630/MatMulMatMul5sequential_63/leaky_re_lu_566/LeakyRelu:activations:05sequential_63/dense_630/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_63/dense_630/BiasAdd/ReadVariableOpReadVariableOp7sequential_63_dense_630_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_63/dense_630/BiasAddBiasAdd(sequential_63/dense_630/MatMul:product:06sequential_63/dense_630/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_63/dense_630/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ 
NoOpNoOp?^sequential_63/batch_normalization_556/batchnorm/ReadVariableOpA^sequential_63/batch_normalization_556/batchnorm/ReadVariableOp_1A^sequential_63/batch_normalization_556/batchnorm/ReadVariableOp_2C^sequential_63/batch_normalization_556/batchnorm/mul/ReadVariableOp?^sequential_63/batch_normalization_557/batchnorm/ReadVariableOpA^sequential_63/batch_normalization_557/batchnorm/ReadVariableOp_1A^sequential_63/batch_normalization_557/batchnorm/ReadVariableOp_2C^sequential_63/batch_normalization_557/batchnorm/mul/ReadVariableOp?^sequential_63/batch_normalization_558/batchnorm/ReadVariableOpA^sequential_63/batch_normalization_558/batchnorm/ReadVariableOp_1A^sequential_63/batch_normalization_558/batchnorm/ReadVariableOp_2C^sequential_63/batch_normalization_558/batchnorm/mul/ReadVariableOp?^sequential_63/batch_normalization_559/batchnorm/ReadVariableOpA^sequential_63/batch_normalization_559/batchnorm/ReadVariableOp_1A^sequential_63/batch_normalization_559/batchnorm/ReadVariableOp_2C^sequential_63/batch_normalization_559/batchnorm/mul/ReadVariableOp?^sequential_63/batch_normalization_560/batchnorm/ReadVariableOpA^sequential_63/batch_normalization_560/batchnorm/ReadVariableOp_1A^sequential_63/batch_normalization_560/batchnorm/ReadVariableOp_2C^sequential_63/batch_normalization_560/batchnorm/mul/ReadVariableOp?^sequential_63/batch_normalization_561/batchnorm/ReadVariableOpA^sequential_63/batch_normalization_561/batchnorm/ReadVariableOp_1A^sequential_63/batch_normalization_561/batchnorm/ReadVariableOp_2C^sequential_63/batch_normalization_561/batchnorm/mul/ReadVariableOp?^sequential_63/batch_normalization_562/batchnorm/ReadVariableOpA^sequential_63/batch_normalization_562/batchnorm/ReadVariableOp_1A^sequential_63/batch_normalization_562/batchnorm/ReadVariableOp_2C^sequential_63/batch_normalization_562/batchnorm/mul/ReadVariableOp?^sequential_63/batch_normalization_563/batchnorm/ReadVariableOpA^sequential_63/batch_normalization_563/batchnorm/ReadVariableOp_1A^sequential_63/batch_normalization_563/batchnorm/ReadVariableOp_2C^sequential_63/batch_normalization_563/batchnorm/mul/ReadVariableOp?^sequential_63/batch_normalization_564/batchnorm/ReadVariableOpA^sequential_63/batch_normalization_564/batchnorm/ReadVariableOp_1A^sequential_63/batch_normalization_564/batchnorm/ReadVariableOp_2C^sequential_63/batch_normalization_564/batchnorm/mul/ReadVariableOp?^sequential_63/batch_normalization_565/batchnorm/ReadVariableOpA^sequential_63/batch_normalization_565/batchnorm/ReadVariableOp_1A^sequential_63/batch_normalization_565/batchnorm/ReadVariableOp_2C^sequential_63/batch_normalization_565/batchnorm/mul/ReadVariableOp?^sequential_63/batch_normalization_566/batchnorm/ReadVariableOpA^sequential_63/batch_normalization_566/batchnorm/ReadVariableOp_1A^sequential_63/batch_normalization_566/batchnorm/ReadVariableOp_2C^sequential_63/batch_normalization_566/batchnorm/mul/ReadVariableOp/^sequential_63/dense_619/BiasAdd/ReadVariableOp.^sequential_63/dense_619/MatMul/ReadVariableOp/^sequential_63/dense_620/BiasAdd/ReadVariableOp.^sequential_63/dense_620/MatMul/ReadVariableOp/^sequential_63/dense_621/BiasAdd/ReadVariableOp.^sequential_63/dense_621/MatMul/ReadVariableOp/^sequential_63/dense_622/BiasAdd/ReadVariableOp.^sequential_63/dense_622/MatMul/ReadVariableOp/^sequential_63/dense_623/BiasAdd/ReadVariableOp.^sequential_63/dense_623/MatMul/ReadVariableOp/^sequential_63/dense_624/BiasAdd/ReadVariableOp.^sequential_63/dense_624/MatMul/ReadVariableOp/^sequential_63/dense_625/BiasAdd/ReadVariableOp.^sequential_63/dense_625/MatMul/ReadVariableOp/^sequential_63/dense_626/BiasAdd/ReadVariableOp.^sequential_63/dense_626/MatMul/ReadVariableOp/^sequential_63/dense_627/BiasAdd/ReadVariableOp.^sequential_63/dense_627/MatMul/ReadVariableOp/^sequential_63/dense_628/BiasAdd/ReadVariableOp.^sequential_63/dense_628/MatMul/ReadVariableOp/^sequential_63/dense_629/BiasAdd/ReadVariableOp.^sequential_63/dense_629/MatMul/ReadVariableOp/^sequential_63/dense_630/BiasAdd/ReadVariableOp.^sequential_63/dense_630/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>sequential_63/batch_normalization_556/batchnorm/ReadVariableOp>sequential_63/batch_normalization_556/batchnorm/ReadVariableOp2
@sequential_63/batch_normalization_556/batchnorm/ReadVariableOp_1@sequential_63/batch_normalization_556/batchnorm/ReadVariableOp_12
@sequential_63/batch_normalization_556/batchnorm/ReadVariableOp_2@sequential_63/batch_normalization_556/batchnorm/ReadVariableOp_22
Bsequential_63/batch_normalization_556/batchnorm/mul/ReadVariableOpBsequential_63/batch_normalization_556/batchnorm/mul/ReadVariableOp2
>sequential_63/batch_normalization_557/batchnorm/ReadVariableOp>sequential_63/batch_normalization_557/batchnorm/ReadVariableOp2
@sequential_63/batch_normalization_557/batchnorm/ReadVariableOp_1@sequential_63/batch_normalization_557/batchnorm/ReadVariableOp_12
@sequential_63/batch_normalization_557/batchnorm/ReadVariableOp_2@sequential_63/batch_normalization_557/batchnorm/ReadVariableOp_22
Bsequential_63/batch_normalization_557/batchnorm/mul/ReadVariableOpBsequential_63/batch_normalization_557/batchnorm/mul/ReadVariableOp2
>sequential_63/batch_normalization_558/batchnorm/ReadVariableOp>sequential_63/batch_normalization_558/batchnorm/ReadVariableOp2
@sequential_63/batch_normalization_558/batchnorm/ReadVariableOp_1@sequential_63/batch_normalization_558/batchnorm/ReadVariableOp_12
@sequential_63/batch_normalization_558/batchnorm/ReadVariableOp_2@sequential_63/batch_normalization_558/batchnorm/ReadVariableOp_22
Bsequential_63/batch_normalization_558/batchnorm/mul/ReadVariableOpBsequential_63/batch_normalization_558/batchnorm/mul/ReadVariableOp2
>sequential_63/batch_normalization_559/batchnorm/ReadVariableOp>sequential_63/batch_normalization_559/batchnorm/ReadVariableOp2
@sequential_63/batch_normalization_559/batchnorm/ReadVariableOp_1@sequential_63/batch_normalization_559/batchnorm/ReadVariableOp_12
@sequential_63/batch_normalization_559/batchnorm/ReadVariableOp_2@sequential_63/batch_normalization_559/batchnorm/ReadVariableOp_22
Bsequential_63/batch_normalization_559/batchnorm/mul/ReadVariableOpBsequential_63/batch_normalization_559/batchnorm/mul/ReadVariableOp2
>sequential_63/batch_normalization_560/batchnorm/ReadVariableOp>sequential_63/batch_normalization_560/batchnorm/ReadVariableOp2
@sequential_63/batch_normalization_560/batchnorm/ReadVariableOp_1@sequential_63/batch_normalization_560/batchnorm/ReadVariableOp_12
@sequential_63/batch_normalization_560/batchnorm/ReadVariableOp_2@sequential_63/batch_normalization_560/batchnorm/ReadVariableOp_22
Bsequential_63/batch_normalization_560/batchnorm/mul/ReadVariableOpBsequential_63/batch_normalization_560/batchnorm/mul/ReadVariableOp2
>sequential_63/batch_normalization_561/batchnorm/ReadVariableOp>sequential_63/batch_normalization_561/batchnorm/ReadVariableOp2
@sequential_63/batch_normalization_561/batchnorm/ReadVariableOp_1@sequential_63/batch_normalization_561/batchnorm/ReadVariableOp_12
@sequential_63/batch_normalization_561/batchnorm/ReadVariableOp_2@sequential_63/batch_normalization_561/batchnorm/ReadVariableOp_22
Bsequential_63/batch_normalization_561/batchnorm/mul/ReadVariableOpBsequential_63/batch_normalization_561/batchnorm/mul/ReadVariableOp2
>sequential_63/batch_normalization_562/batchnorm/ReadVariableOp>sequential_63/batch_normalization_562/batchnorm/ReadVariableOp2
@sequential_63/batch_normalization_562/batchnorm/ReadVariableOp_1@sequential_63/batch_normalization_562/batchnorm/ReadVariableOp_12
@sequential_63/batch_normalization_562/batchnorm/ReadVariableOp_2@sequential_63/batch_normalization_562/batchnorm/ReadVariableOp_22
Bsequential_63/batch_normalization_562/batchnorm/mul/ReadVariableOpBsequential_63/batch_normalization_562/batchnorm/mul/ReadVariableOp2
>sequential_63/batch_normalization_563/batchnorm/ReadVariableOp>sequential_63/batch_normalization_563/batchnorm/ReadVariableOp2
@sequential_63/batch_normalization_563/batchnorm/ReadVariableOp_1@sequential_63/batch_normalization_563/batchnorm/ReadVariableOp_12
@sequential_63/batch_normalization_563/batchnorm/ReadVariableOp_2@sequential_63/batch_normalization_563/batchnorm/ReadVariableOp_22
Bsequential_63/batch_normalization_563/batchnorm/mul/ReadVariableOpBsequential_63/batch_normalization_563/batchnorm/mul/ReadVariableOp2
>sequential_63/batch_normalization_564/batchnorm/ReadVariableOp>sequential_63/batch_normalization_564/batchnorm/ReadVariableOp2
@sequential_63/batch_normalization_564/batchnorm/ReadVariableOp_1@sequential_63/batch_normalization_564/batchnorm/ReadVariableOp_12
@sequential_63/batch_normalization_564/batchnorm/ReadVariableOp_2@sequential_63/batch_normalization_564/batchnorm/ReadVariableOp_22
Bsequential_63/batch_normalization_564/batchnorm/mul/ReadVariableOpBsequential_63/batch_normalization_564/batchnorm/mul/ReadVariableOp2
>sequential_63/batch_normalization_565/batchnorm/ReadVariableOp>sequential_63/batch_normalization_565/batchnorm/ReadVariableOp2
@sequential_63/batch_normalization_565/batchnorm/ReadVariableOp_1@sequential_63/batch_normalization_565/batchnorm/ReadVariableOp_12
@sequential_63/batch_normalization_565/batchnorm/ReadVariableOp_2@sequential_63/batch_normalization_565/batchnorm/ReadVariableOp_22
Bsequential_63/batch_normalization_565/batchnorm/mul/ReadVariableOpBsequential_63/batch_normalization_565/batchnorm/mul/ReadVariableOp2
>sequential_63/batch_normalization_566/batchnorm/ReadVariableOp>sequential_63/batch_normalization_566/batchnorm/ReadVariableOp2
@sequential_63/batch_normalization_566/batchnorm/ReadVariableOp_1@sequential_63/batch_normalization_566/batchnorm/ReadVariableOp_12
@sequential_63/batch_normalization_566/batchnorm/ReadVariableOp_2@sequential_63/batch_normalization_566/batchnorm/ReadVariableOp_22
Bsequential_63/batch_normalization_566/batchnorm/mul/ReadVariableOpBsequential_63/batch_normalization_566/batchnorm/mul/ReadVariableOp2`
.sequential_63/dense_619/BiasAdd/ReadVariableOp.sequential_63/dense_619/BiasAdd/ReadVariableOp2^
-sequential_63/dense_619/MatMul/ReadVariableOp-sequential_63/dense_619/MatMul/ReadVariableOp2`
.sequential_63/dense_620/BiasAdd/ReadVariableOp.sequential_63/dense_620/BiasAdd/ReadVariableOp2^
-sequential_63/dense_620/MatMul/ReadVariableOp-sequential_63/dense_620/MatMul/ReadVariableOp2`
.sequential_63/dense_621/BiasAdd/ReadVariableOp.sequential_63/dense_621/BiasAdd/ReadVariableOp2^
-sequential_63/dense_621/MatMul/ReadVariableOp-sequential_63/dense_621/MatMul/ReadVariableOp2`
.sequential_63/dense_622/BiasAdd/ReadVariableOp.sequential_63/dense_622/BiasAdd/ReadVariableOp2^
-sequential_63/dense_622/MatMul/ReadVariableOp-sequential_63/dense_622/MatMul/ReadVariableOp2`
.sequential_63/dense_623/BiasAdd/ReadVariableOp.sequential_63/dense_623/BiasAdd/ReadVariableOp2^
-sequential_63/dense_623/MatMul/ReadVariableOp-sequential_63/dense_623/MatMul/ReadVariableOp2`
.sequential_63/dense_624/BiasAdd/ReadVariableOp.sequential_63/dense_624/BiasAdd/ReadVariableOp2^
-sequential_63/dense_624/MatMul/ReadVariableOp-sequential_63/dense_624/MatMul/ReadVariableOp2`
.sequential_63/dense_625/BiasAdd/ReadVariableOp.sequential_63/dense_625/BiasAdd/ReadVariableOp2^
-sequential_63/dense_625/MatMul/ReadVariableOp-sequential_63/dense_625/MatMul/ReadVariableOp2`
.sequential_63/dense_626/BiasAdd/ReadVariableOp.sequential_63/dense_626/BiasAdd/ReadVariableOp2^
-sequential_63/dense_626/MatMul/ReadVariableOp-sequential_63/dense_626/MatMul/ReadVariableOp2`
.sequential_63/dense_627/BiasAdd/ReadVariableOp.sequential_63/dense_627/BiasAdd/ReadVariableOp2^
-sequential_63/dense_627/MatMul/ReadVariableOp-sequential_63/dense_627/MatMul/ReadVariableOp2`
.sequential_63/dense_628/BiasAdd/ReadVariableOp.sequential_63/dense_628/BiasAdd/ReadVariableOp2^
-sequential_63/dense_628/MatMul/ReadVariableOp-sequential_63/dense_628/MatMul/ReadVariableOp2`
.sequential_63/dense_629/BiasAdd/ReadVariableOp.sequential_63/dense_629/BiasAdd/ReadVariableOp2^
-sequential_63/dense_629/MatMul/ReadVariableOp-sequential_63/dense_629/MatMul/ReadVariableOp2`
.sequential_63/dense_630/BiasAdd/ReadVariableOp.sequential_63/dense_630/BiasAdd/ReadVariableOp2^
-sequential_63/dense_630/MatMul/ReadVariableOp-sequential_63/dense_630/MatMul/ReadVariableOp:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_63_input:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_562_layer_call_and_return_conditional_losses_834713

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Õ

$__inference_signature_wrapper_833903
normalization_63_input
unknown
	unknown_0
	unknown_1:-
	unknown_2:-
	unknown_3:-
	unknown_4:-
	unknown_5:-
	unknown_6:-
	unknown_7:--
	unknown_8:-
	unknown_9:-

unknown_10:-

unknown_11:-

unknown_12:-

unknown_13:--

unknown_14:-

unknown_15:-

unknown_16:-

unknown_17:-

unknown_18:-

unknown_19:--

unknown_20:-

unknown_21:-

unknown_22:-

unknown_23:-

unknown_24:-

unknown_25:--

unknown_26:-

unknown_27:-

unknown_28:-

unknown_29:-

unknown_30:-

unknown_31:-l

unknown_32:l

unknown_33:l

unknown_34:l

unknown_35:l

unknown_36:l

unknown_37:ll

unknown_38:l

unknown_39:l

unknown_40:l

unknown_41:l

unknown_42:l

unknown_43:ll

unknown_44:l

unknown_45:l

unknown_46:l

unknown_47:l

unknown_48:l

unknown_49:ll

unknown_50:l

unknown_51:l

unknown_52:l

unknown_53:l

unknown_54:l

unknown_55:ll

unknown_56:l

unknown_57:l

unknown_58:l

unknown_59:l

unknown_60:l

unknown_61:lV

unknown_62:V

unknown_63:V

unknown_64:V

unknown_65:V

unknown_66:V

unknown_67:V

unknown_68:
identity¢StatefulPartitionedCalló	
StatefulPartitionedCallStatefulPartitionedCallnormalization_63_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEF*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_830176o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_63_input:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_557_layer_call_and_return_conditional_losses_834168

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_835139

inputs5
'assignmovingavg_readvariableop_resource:V7
)assignmovingavg_1_readvariableop_resource:V3
%batchnorm_mul_readvariableop_resource:V/
!batchnorm_readvariableop_resource:V
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:V*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:V
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:V*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:V*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:V*
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
:V*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Vx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:V¬
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
:V*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:V~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:V´
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
:VP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:V~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:V*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Vc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Vv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:V*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Vr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿV: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_830610

inputs/
!batchnorm_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l1
#batchnorm_readvariableop_1_resource:l1
#batchnorm_readvariableop_2_resource:l
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:lz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_830446

inputs/
!batchnorm_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-1
#batchnorm_readvariableop_1_resource:-1
#batchnorm_readvariableop_2_resource:-
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:-z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_830739

inputs5
'assignmovingavg_readvariableop_resource:l7
)assignmovingavg_1_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l/
!batchnorm_readvariableop_resource:l
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:l
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿll
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:l*
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
:l*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:lx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l¬
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
:l*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:l~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l´
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:lv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ä

*__inference_dense_620_layer_call_fn_834068

inputs
unknown:--
	unknown_0:-
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_620_layer_call_and_return_conditional_losses_831134o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
È	
ö
E__inference_dense_619_layer_call_and_return_conditional_losses_831102

inputs0
matmul_readvariableop_resource:--
biasadd_readvariableop_resource:-
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:-*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_830364

inputs/
!batchnorm_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-1
#batchnorm_readvariableop_1_resource:-1
#batchnorm_readvariableop_2_resource:-
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:-z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_831020

inputs/
!batchnorm_readvariableop_resource:V3
%batchnorm_mul_readvariableop_resource:V1
#batchnorm_readvariableop_1_resource:V1
#batchnorm_readvariableop_2_resource:V
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:V*
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
:VP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:V~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:V*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Vc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:V*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Vz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:V*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Vr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿVº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿV: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_564_layer_call_fn_834867

inputs
unknown:l
	unknown_0:l
	unknown_1:l
	unknown_2:l
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_830903o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_834812

inputs5
'assignmovingavg_readvariableop_resource:l7
)assignmovingavg_1_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l/
!batchnorm_readvariableop_resource:l
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:l
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿll
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:l*
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
:l*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:lx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l¬
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
:l*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:l~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l´
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:lv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_560_layer_call_fn_834490

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
:ÿÿÿÿÿÿÿÿÿ-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_560_layer_call_and_return_conditional_losses_831250`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_830657

inputs5
'assignmovingavg_readvariableop_resource:l7
)assignmovingavg_1_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l/
!batchnorm_readvariableop_resource:l
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:l
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿll
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:l*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:l*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:l*
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
:l*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:lx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:l¬
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
:l*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:l~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:l´
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:lv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_559_layer_call_fn_834309

inputs
unknown:-
	unknown_0:-
	unknown_1:-
	unknown_2:-
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_830446o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_556_layer_call_and_return_conditional_losses_831122

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_562_layer_call_and_return_conditional_losses_831314

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿl:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_830774

inputs/
!batchnorm_readvariableop_resource:l3
%batchnorm_mul_readvariableop_resource:l1
#batchnorm_readvariableop_1_resource:l1
#batchnorm_readvariableop_2_resource:l
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:l*
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
:lP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:l~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:l*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:lc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:l*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:lz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:l*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:lr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿl: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
 
_user_specified_nameinputs
È	
ö
E__inference_dense_624_layer_call_and_return_conditional_losses_834514

inputs0
matmul_readvariableop_resource:-l-
biasadd_readvariableop_resource:l
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:-l*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:l*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿlw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ä

*__inference_dense_624_layer_call_fn_834504

inputs
unknown:-l
	unknown_0:l
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_624_layer_call_and_return_conditional_losses_831262o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_830200

inputs/
!batchnorm_readvariableop_resource:-3
%batchnorm_mul_readvariableop_resource:-1
#batchnorm_readvariableop_1_resource:-1
#batchnorm_readvariableop_2_resource:-
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:-*
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
:-P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:-~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:-*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:-c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:-*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:-z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:-*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:-r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ-: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-
 
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
normalization_63_input?
(serving_default_normalization_63_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_6300
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ý
Ä

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
layer_with_weights-20
layer-29
layer-30
 layer_with_weights-21
 layer-31
!layer_with_weights-22
!layer-32
"layer-33
#layer_with_weights-23
#layer-34
$	optimizer
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_default_save_signature
,
signatures"
_tf_keras_sequential
Ó
-
_keep_axis
._reduce_axis
/_reduce_axis_mask
0_broadcast_shape
1mean
1
adapt_mean
2variance
2adapt_variance
	3count
4	keras_api
5_adapt_function"
_tf_keras_layer
»

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Okernel
Pbias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
»

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
¦
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	¢axis

£gamma
	¤beta
¥moving_mean
¦moving_variance
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
«
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
³kernel
	´bias
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	»axis

¼gamma
	½beta
¾moving_mean
¿moving_variance
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ìkernel
	Íbias
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	Ôaxis

Õgamma
	Öbeta
×moving_mean
Ømoving_variance
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ß	variables
àtrainable_variables
áregularization_losses
â	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
åkernel
	æbias
ç	variables
ètrainable_variables
éregularization_losses
ê	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	íaxis

îgamma
	ïbeta
ðmoving_mean
ñmoving_variance
ò	variables
ótrainable_variables
ôregularization_losses
õ	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ø	variables
ùtrainable_variables
úregularization_losses
û	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
þkernel
	ÿbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

 gamma
	¡beta
¢moving_mean
£moving_variance
¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
°kernel
	±bias
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	¸axis

¹gamma
	ºbeta
»moving_mean
¼moving_variance
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ékernel
	Êbias
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"
_tf_keras_layer
 
	Ñiter
Òbeta_1
Óbeta_2

Ôdecay6m7m?m@mOmPmXmYmhmimqmrm	m	m	m	m	m	m	£m	¤m	³m	´m	¼m 	½m¡	Ìm¢	Ím£	Õm¤	Öm¥	åm¦	æm§	îm¨	ïm©	þmª	ÿm«	m¬	m­	m®	m¯	 m°	¡m±	°m²	±m³	¹m´	ºmµ	Ém¶	Êm·6v¸7v¹?vº@v»Ov¼Pv½Xv¾Yv¿hvÀivÁqvÂrvÃ	vÄ	vÅ	vÆ	vÇ	vÈ	vÉ	£vÊ	¤vË	³vÌ	´vÍ	¼vÎ	½vÏ	ÌvÐ	ÍvÑ	ÕvÒ	ÖvÓ	åvÔ	ævÕ	îvÖ	ïv×	þvØ	ÿvÙ	vÚ	vÛ	vÜ	vÝ	 vÞ	¡vß	°và	±vá	¹vâ	ºvã	Évä	Êvå"
	optimizer

10
21
32
63
74
?5
@6
A7
B8
O9
P10
X11
Y12
Z13
[14
h15
i16
q17
r18
s19
t20
21
22
23
24
25
26
27
28
£29
¤30
¥31
¦32
³33
´34
¼35
½36
¾37
¿38
Ì39
Í40
Õ41
Ö42
×43
Ø44
å45
æ46
î47
ï48
ð49
ñ50
þ51
ÿ52
53
54
55
56
57
58
 59
¡60
¢61
£62
°63
±64
¹65
º66
»67
¼68
É69
Ê70"
trackable_list_wrapper
¨
60
71
?2
@3
O4
P5
X6
Y7
h8
i9
q10
r11
12
13
14
15
16
17
£18
¤19
³20
´21
¼22
½23
Ì24
Í25
Õ26
Ö27
å28
æ29
î30
ï31
þ32
ÿ33
34
35
36
37
 38
¡39
°40
±41
¹42
º43
É44
Ê45"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
+_default_save_signature
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_sequential_63_layer_call_fn_831604
.__inference_sequential_63_layer_call_fn_832917
.__inference_sequential_63_layer_call_fn_833062
.__inference_sequential_63_layer_call_fn_832406À
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
I__inference_sequential_63_layer_call_and_return_conditional_losses_833332
I__inference_sequential_63_layer_call_and_return_conditional_losses_833756
I__inference_sequential_63_layer_call_and_return_conditional_losses_832587
I__inference_sequential_63_layer_call_and_return_conditional_losses_832768À
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
!__inference__wrapped_model_830176normalization_63_input"
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
Úserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
¿2¼
__inference_adapt_step_833950
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
": -2dense_619/kernel
:-2dense_619/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_619_layer_call_fn_833959¢
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
E__inference_dense_619_layer_call_and_return_conditional_losses_833969¢
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
+:)-2batch_normalization_556/gamma
*:(-2batch_normalization_556/beta
3:1- (2#batch_normalization_556/moving_mean
7:5- (2'batch_normalization_556/moving_variance
<
?0
@1
A2
B3"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_556_layer_call_fn_833982
8__inference_batch_normalization_556_layer_call_fn_833995´
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
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_834015
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_834049´
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
ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_556_layer_call_fn_834054¢
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
K__inference_leaky_re_lu_556_layer_call_and_return_conditional_losses_834059¢
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
": --2dense_620/kernel
:-2dense_620/bias
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_620_layer_call_fn_834068¢
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
E__inference_dense_620_layer_call_and_return_conditional_losses_834078¢
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
+:)-2batch_normalization_557/gamma
*:(-2batch_normalization_557/beta
3:1- (2#batch_normalization_557/moving_mean
7:5- (2'batch_normalization_557/moving_variance
<
X0
Y1
Z2
[3"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_557_layer_call_fn_834091
8__inference_batch_normalization_557_layer_call_fn_834104´
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
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_834124
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_834158´
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
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_557_layer_call_fn_834163¢
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
K__inference_leaky_re_lu_557_layer_call_and_return_conditional_losses_834168¢
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
": --2dense_621/kernel
:-2dense_621/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ùnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_621_layer_call_fn_834177¢
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
E__inference_dense_621_layer_call_and_return_conditional_losses_834187¢
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
+:)-2batch_normalization_558/gamma
*:(-2batch_normalization_558/beta
3:1- (2#batch_normalization_558/moving_mean
7:5- (2'batch_normalization_558/moving_variance
<
q0
r1
s2
t3"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_558_layer_call_fn_834200
8__inference_batch_normalization_558_layer_call_fn_834213´
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
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_834233
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_834267´
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
´
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_558_layer_call_fn_834272¢
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
K__inference_leaky_re_lu_558_layer_call_and_return_conditional_losses_834277¢
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
": --2dense_622/kernel
:-2dense_622/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_622_layer_call_fn_834286¢
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
E__inference_dense_622_layer_call_and_return_conditional_losses_834296¢
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
+:)-2batch_normalization_559/gamma
*:(-2batch_normalization_559/beta
3:1- (2#batch_normalization_559/moving_mean
7:5- (2'batch_normalization_559/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_559_layer_call_fn_834309
8__inference_batch_normalization_559_layer_call_fn_834322´
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
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_834342
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_834376´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_559_layer_call_fn_834381¢
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
K__inference_leaky_re_lu_559_layer_call_and_return_conditional_losses_834386¢
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
": --2dense_623/kernel
:-2dense_623/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_623_layer_call_fn_834395¢
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
E__inference_dense_623_layer_call_and_return_conditional_losses_834405¢
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
+:)-2batch_normalization_560/gamma
*:(-2batch_normalization_560/beta
3:1- (2#batch_normalization_560/moving_mean
7:5- (2'batch_normalization_560/moving_variance
@
£0
¤1
¥2
¦3"
trackable_list_wrapper
0
£0
¤1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_560_layer_call_fn_834418
8__inference_batch_normalization_560_layer_call_fn_834431´
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
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_834451
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_834485´
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
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
­	variables
®trainable_variables
¯regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_560_layer_call_fn_834490¢
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
K__inference_leaky_re_lu_560_layer_call_and_return_conditional_losses_834495¢
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
": -l2dense_624/kernel
:l2dense_624/bias
0
³0
´1"
trackable_list_wrapper
0
³0
´1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
µ	variables
¶trainable_variables
·regularization_losses
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_624_layer_call_fn_834504¢
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
E__inference_dense_624_layer_call_and_return_conditional_losses_834514¢
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
+:)l2batch_normalization_561/gamma
*:(l2batch_normalization_561/beta
3:1l (2#batch_normalization_561/moving_mean
7:5l (2'batch_normalization_561/moving_variance
@
¼0
½1
¾2
¿3"
trackable_list_wrapper
0
¼0
½1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_561_layer_call_fn_834527
8__inference_batch_normalization_561_layer_call_fn_834540´
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
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_834560
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_834594´
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
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
Æ	variables
Çtrainable_variables
Èregularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_561_layer_call_fn_834599¢
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
K__inference_leaky_re_lu_561_layer_call_and_return_conditional_losses_834604¢
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
": ll2dense_625/kernel
:l2dense_625/bias
0
Ì0
Í1"
trackable_list_wrapper
0
Ì0
Í1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_625_layer_call_fn_834613¢
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
E__inference_dense_625_layer_call_and_return_conditional_losses_834623¢
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
+:)l2batch_normalization_562/gamma
*:(l2batch_normalization_562/beta
3:1l (2#batch_normalization_562/moving_mean
7:5l (2'batch_normalization_562/moving_variance
@
Õ0
Ö1
×2
Ø3"
trackable_list_wrapper
0
Õ0
Ö1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ºnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
Ù	variables
Útrainable_variables
Ûregularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_562_layer_call_fn_834636
8__inference_batch_normalization_562_layer_call_fn_834649´
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
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_834669
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_834703´
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
¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
ß	variables
àtrainable_variables
áregularization_losses
ã__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_562_layer_call_fn_834708¢
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
K__inference_leaky_re_lu_562_layer_call_and_return_conditional_losses_834713¢
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
": ll2dense_626/kernel
:l2dense_626/bias
0
å0
æ1"
trackable_list_wrapper
0
å0
æ1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Änon_trainable_variables
Ålayers
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
ç	variables
ètrainable_variables
éregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_626_layer_call_fn_834722¢
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
E__inference_dense_626_layer_call_and_return_conditional_losses_834732¢
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
+:)l2batch_normalization_563/gamma
*:(l2batch_normalization_563/beta
3:1l (2#batch_normalization_563/moving_mean
7:5l (2'batch_normalization_563/moving_variance
@
î0
ï1
ð2
ñ3"
trackable_list_wrapper
0
î0
ï1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
ò	variables
ótrainable_variables
ôregularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_563_layer_call_fn_834745
8__inference_batch_normalization_563_layer_call_fn_834758´
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
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_834778
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_834812´
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
Înon_trainable_variables
Ïlayers
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
ø	variables
ùtrainable_variables
úregularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_563_layer_call_fn_834817¢
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
K__inference_leaky_re_lu_563_layer_call_and_return_conditional_losses_834822¢
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
": ll2dense_627/kernel
:l2dense_627/bias
0
þ0
ÿ1"
trackable_list_wrapper
0
þ0
ÿ1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ónon_trainable_variables
Ôlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_627_layer_call_fn_834831¢
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
E__inference_dense_627_layer_call_and_return_conditional_losses_834841¢
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
+:)l2batch_normalization_564/gamma
*:(l2batch_normalization_564/beta
3:1l (2#batch_normalization_564/moving_mean
7:5l (2'batch_normalization_564/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ønon_trainable_variables
Ùlayers
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_564_layer_call_fn_834854
8__inference_batch_normalization_564_layer_call_fn_834867´
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
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_834887
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_834921´
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
Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_564_layer_call_fn_834926¢
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
K__inference_leaky_re_lu_564_layer_call_and_return_conditional_losses_834931¢
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
": ll2dense_628/kernel
:l2dense_628/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_628_layer_call_fn_834940¢
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
E__inference_dense_628_layer_call_and_return_conditional_losses_834950¢
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
+:)l2batch_normalization_565/gamma
*:(l2batch_normalization_565/beta
3:1l (2#batch_normalization_565/moving_mean
7:5l (2'batch_normalization_565/moving_variance
@
 0
¡1
¢2
£3"
trackable_list_wrapper
0
 0
¡1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_565_layer_call_fn_834963
8__inference_batch_normalization_565_layer_call_fn_834976´
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
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_834996
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_835030´
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
ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_565_layer_call_fn_835035¢
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
K__inference_leaky_re_lu_565_layer_call_and_return_conditional_losses_835040¢
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
": lV2dense_629/kernel
:V2dense_629/bias
0
°0
±1"
trackable_list_wrapper
0
°0
±1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_629_layer_call_fn_835049¢
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
E__inference_dense_629_layer_call_and_return_conditional_losses_835059¢
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
+:)V2batch_normalization_566/gamma
*:(V2batch_normalization_566/beta
3:1V (2#batch_normalization_566/moving_mean
7:5V (2'batch_normalization_566/moving_variance
@
¹0
º1
»2
¼3"
trackable_list_wrapper
0
¹0
º1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_566_layer_call_fn_835072
8__inference_batch_normalization_566_layer_call_fn_835085´
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
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_835105
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_835139´
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
ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_566_layer_call_fn_835144¢
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
K__inference_leaky_re_lu_566_layer_call_and_return_conditional_losses_835149¢
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
": V2dense_630/kernel
:2dense_630/bias
0
É0
Ê1"
trackable_list_wrapper
0
É0
Ê1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_630_layer_call_fn_835158¢
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
E__inference_dense_630_layer_call_and_return_conditional_losses_835168¢
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
î
10
21
32
A3
B4
Z5
[6
s7
t8
9
10
¥11
¦12
¾13
¿14
×15
Ø16
ð17
ñ18
19
20
¢21
£22
»23
¼24"
trackable_list_wrapper
®
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
28
29
30
 31
!32
"33
#34"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
$__inference_signature_wrapper_833903normalization_63_input"
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
A0
B1"
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
Z0
[1"
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
s0
t1"
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
0
1"
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
¥0
¦1"
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
¾0
¿1"
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
×0
Ø1"
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
ð0
ñ1"
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
0
1"
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
¢0
£1"
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
»0
¼1"
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

total

count
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
':%-2Adam/dense_619/kernel/m
!:-2Adam/dense_619/bias/m
0:.-2$Adam/batch_normalization_556/gamma/m
/:--2#Adam/batch_normalization_556/beta/m
':%--2Adam/dense_620/kernel/m
!:-2Adam/dense_620/bias/m
0:.-2$Adam/batch_normalization_557/gamma/m
/:--2#Adam/batch_normalization_557/beta/m
':%--2Adam/dense_621/kernel/m
!:-2Adam/dense_621/bias/m
0:.-2$Adam/batch_normalization_558/gamma/m
/:--2#Adam/batch_normalization_558/beta/m
':%--2Adam/dense_622/kernel/m
!:-2Adam/dense_622/bias/m
0:.-2$Adam/batch_normalization_559/gamma/m
/:--2#Adam/batch_normalization_559/beta/m
':%--2Adam/dense_623/kernel/m
!:-2Adam/dense_623/bias/m
0:.-2$Adam/batch_normalization_560/gamma/m
/:--2#Adam/batch_normalization_560/beta/m
':%-l2Adam/dense_624/kernel/m
!:l2Adam/dense_624/bias/m
0:.l2$Adam/batch_normalization_561/gamma/m
/:-l2#Adam/batch_normalization_561/beta/m
':%ll2Adam/dense_625/kernel/m
!:l2Adam/dense_625/bias/m
0:.l2$Adam/batch_normalization_562/gamma/m
/:-l2#Adam/batch_normalization_562/beta/m
':%ll2Adam/dense_626/kernel/m
!:l2Adam/dense_626/bias/m
0:.l2$Adam/batch_normalization_563/gamma/m
/:-l2#Adam/batch_normalization_563/beta/m
':%ll2Adam/dense_627/kernel/m
!:l2Adam/dense_627/bias/m
0:.l2$Adam/batch_normalization_564/gamma/m
/:-l2#Adam/batch_normalization_564/beta/m
':%ll2Adam/dense_628/kernel/m
!:l2Adam/dense_628/bias/m
0:.l2$Adam/batch_normalization_565/gamma/m
/:-l2#Adam/batch_normalization_565/beta/m
':%lV2Adam/dense_629/kernel/m
!:V2Adam/dense_629/bias/m
0:.V2$Adam/batch_normalization_566/gamma/m
/:-V2#Adam/batch_normalization_566/beta/m
':%V2Adam/dense_630/kernel/m
!:2Adam/dense_630/bias/m
':%-2Adam/dense_619/kernel/v
!:-2Adam/dense_619/bias/v
0:.-2$Adam/batch_normalization_556/gamma/v
/:--2#Adam/batch_normalization_556/beta/v
':%--2Adam/dense_620/kernel/v
!:-2Adam/dense_620/bias/v
0:.-2$Adam/batch_normalization_557/gamma/v
/:--2#Adam/batch_normalization_557/beta/v
':%--2Adam/dense_621/kernel/v
!:-2Adam/dense_621/bias/v
0:.-2$Adam/batch_normalization_558/gamma/v
/:--2#Adam/batch_normalization_558/beta/v
':%--2Adam/dense_622/kernel/v
!:-2Adam/dense_622/bias/v
0:.-2$Adam/batch_normalization_559/gamma/v
/:--2#Adam/batch_normalization_559/beta/v
':%--2Adam/dense_623/kernel/v
!:-2Adam/dense_623/bias/v
0:.-2$Adam/batch_normalization_560/gamma/v
/:--2#Adam/batch_normalization_560/beta/v
':%-l2Adam/dense_624/kernel/v
!:l2Adam/dense_624/bias/v
0:.l2$Adam/batch_normalization_561/gamma/v
/:-l2#Adam/batch_normalization_561/beta/v
':%ll2Adam/dense_625/kernel/v
!:l2Adam/dense_625/bias/v
0:.l2$Adam/batch_normalization_562/gamma/v
/:-l2#Adam/batch_normalization_562/beta/v
':%ll2Adam/dense_626/kernel/v
!:l2Adam/dense_626/bias/v
0:.l2$Adam/batch_normalization_563/gamma/v
/:-l2#Adam/batch_normalization_563/beta/v
':%ll2Adam/dense_627/kernel/v
!:l2Adam/dense_627/bias/v
0:.l2$Adam/batch_normalization_564/gamma/v
/:-l2#Adam/batch_normalization_564/beta/v
':%ll2Adam/dense_628/kernel/v
!:l2Adam/dense_628/bias/v
0:.l2$Adam/batch_normalization_565/gamma/v
/:-l2#Adam/batch_normalization_565/beta/v
':%lV2Adam/dense_629/kernel/v
!:V2Adam/dense_629/bias/v
0:.V2$Adam/batch_normalization_566/gamma/v
/:-V2#Adam/batch_normalization_566/beta/v
':%V2Adam/dense_630/kernel/v
!:2Adam/dense_630/bias/v
	J
Const
J	
Const_1
!__inference__wrapped_model_830176ôzæç67B?A@OP[XZYhitqsr¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæñîðïþÿ£ ¢¡°±¼¹»ºÉÊ?¢<
5¢2
0-
normalization_63_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_630# 
	dense_630ÿÿÿÿÿÿÿÿÿo
__inference_adapt_step_833950N312C¢@
9¢6
41¢
ÿÿÿÿÿÿÿÿÿIteratorSpec 
ª "
 ¹
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_834015bB?A@3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 ¹
S__inference_batch_normalization_556_layer_call_and_return_conditional_losses_834049bAB?@3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 
8__inference_batch_normalization_556_layer_call_fn_833982UB?A@3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p 
ª "ÿÿÿÿÿÿÿÿÿ-
8__inference_batch_normalization_556_layer_call_fn_833995UAB?@3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p
ª "ÿÿÿÿÿÿÿÿÿ-¹
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_834124b[XZY3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 ¹
S__inference_batch_normalization_557_layer_call_and_return_conditional_losses_834158bZ[XY3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 
8__inference_batch_normalization_557_layer_call_fn_834091U[XZY3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p 
ª "ÿÿÿÿÿÿÿÿÿ-
8__inference_batch_normalization_557_layer_call_fn_834104UZ[XY3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p
ª "ÿÿÿÿÿÿÿÿÿ-¹
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_834233btqsr3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 ¹
S__inference_batch_normalization_558_layer_call_and_return_conditional_losses_834267bstqr3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 
8__inference_batch_normalization_558_layer_call_fn_834200Utqsr3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p 
ª "ÿÿÿÿÿÿÿÿÿ-
8__inference_batch_normalization_558_layer_call_fn_834213Ustqr3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p
ª "ÿÿÿÿÿÿÿÿÿ-½
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_834342f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 ½
S__inference_batch_normalization_559_layer_call_and_return_conditional_losses_834376f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 
8__inference_batch_normalization_559_layer_call_fn_834309Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p 
ª "ÿÿÿÿÿÿÿÿÿ-
8__inference_batch_normalization_559_layer_call_fn_834322Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p
ª "ÿÿÿÿÿÿÿÿÿ-½
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_834451f¦£¥¤3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 ½
S__inference_batch_normalization_560_layer_call_and_return_conditional_losses_834485f¥¦£¤3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 
8__inference_batch_normalization_560_layer_call_fn_834418Y¦£¥¤3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p 
ª "ÿÿÿÿÿÿÿÿÿ-
8__inference_batch_normalization_560_layer_call_fn_834431Y¥¦£¤3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ-
p
ª "ÿÿÿÿÿÿÿÿÿ-½
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_834560f¿¼¾½3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 ½
S__inference_batch_normalization_561_layer_call_and_return_conditional_losses_834594f¾¿¼½3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
8__inference_batch_normalization_561_layer_call_fn_834527Y¿¼¾½3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p 
ª "ÿÿÿÿÿÿÿÿÿl
8__inference_batch_normalization_561_layer_call_fn_834540Y¾¿¼½3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p
ª "ÿÿÿÿÿÿÿÿÿl½
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_834669fØÕ×Ö3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 ½
S__inference_batch_normalization_562_layer_call_and_return_conditional_losses_834703f×ØÕÖ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
8__inference_batch_normalization_562_layer_call_fn_834636YØÕ×Ö3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p 
ª "ÿÿÿÿÿÿÿÿÿl
8__inference_batch_normalization_562_layer_call_fn_834649Y×ØÕÖ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p
ª "ÿÿÿÿÿÿÿÿÿl½
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_834778fñîðï3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 ½
S__inference_batch_normalization_563_layer_call_and_return_conditional_losses_834812fðñîï3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
8__inference_batch_normalization_563_layer_call_fn_834745Yñîðï3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p 
ª "ÿÿÿÿÿÿÿÿÿl
8__inference_batch_normalization_563_layer_call_fn_834758Yðñîï3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p
ª "ÿÿÿÿÿÿÿÿÿl½
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_834887f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 ½
S__inference_batch_normalization_564_layer_call_and_return_conditional_losses_834921f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
8__inference_batch_normalization_564_layer_call_fn_834854Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p 
ª "ÿÿÿÿÿÿÿÿÿl
8__inference_batch_normalization_564_layer_call_fn_834867Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p
ª "ÿÿÿÿÿÿÿÿÿl½
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_834996f£ ¢¡3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 ½
S__inference_batch_normalization_565_layer_call_and_return_conditional_losses_835030f¢£ ¡3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
8__inference_batch_normalization_565_layer_call_fn_834963Y£ ¢¡3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p 
ª "ÿÿÿÿÿÿÿÿÿl
8__inference_batch_normalization_565_layer_call_fn_834976Y¢£ ¡3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿl
p
ª "ÿÿÿÿÿÿÿÿÿl½
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_835105f¼¹»º3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿV
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿV
 ½
S__inference_batch_normalization_566_layer_call_and_return_conditional_losses_835139f»¼¹º3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿV
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿV
 
8__inference_batch_normalization_566_layer_call_fn_835072Y¼¹»º3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿV
p 
ª "ÿÿÿÿÿÿÿÿÿV
8__inference_batch_normalization_566_layer_call_fn_835085Y»¼¹º3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿV
p
ª "ÿÿÿÿÿÿÿÿÿV¥
E__inference_dense_619_layer_call_and_return_conditional_losses_833969\67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 }
*__inference_dense_619_layer_call_fn_833959O67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ-¥
E__inference_dense_620_layer_call_and_return_conditional_losses_834078\OP/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 }
*__inference_dense_620_layer_call_fn_834068OOP/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "ÿÿÿÿÿÿÿÿÿ-¥
E__inference_dense_621_layer_call_and_return_conditional_losses_834187\hi/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 }
*__inference_dense_621_layer_call_fn_834177Ohi/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "ÿÿÿÿÿÿÿÿÿ-§
E__inference_dense_622_layer_call_and_return_conditional_losses_834296^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 
*__inference_dense_622_layer_call_fn_834286Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "ÿÿÿÿÿÿÿÿÿ-§
E__inference_dense_623_layer_call_and_return_conditional_losses_834405^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 
*__inference_dense_623_layer_call_fn_834395Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "ÿÿÿÿÿÿÿÿÿ-§
E__inference_dense_624_layer_call_and_return_conditional_losses_834514^³´/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
*__inference_dense_624_layer_call_fn_834504Q³´/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "ÿÿÿÿÿÿÿÿÿl§
E__inference_dense_625_layer_call_and_return_conditional_losses_834623^ÌÍ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
*__inference_dense_625_layer_call_fn_834613QÌÍ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "ÿÿÿÿÿÿÿÿÿl§
E__inference_dense_626_layer_call_and_return_conditional_losses_834732^åæ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
*__inference_dense_626_layer_call_fn_834722Qåæ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "ÿÿÿÿÿÿÿÿÿl§
E__inference_dense_627_layer_call_and_return_conditional_losses_834841^þÿ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
*__inference_dense_627_layer_call_fn_834831Qþÿ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "ÿÿÿÿÿÿÿÿÿl§
E__inference_dense_628_layer_call_and_return_conditional_losses_834950^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
*__inference_dense_628_layer_call_fn_834940Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "ÿÿÿÿÿÿÿÿÿl§
E__inference_dense_629_layer_call_and_return_conditional_losses_835059^°±/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "%¢"

0ÿÿÿÿÿÿÿÿÿV
 
*__inference_dense_629_layer_call_fn_835049Q°±/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "ÿÿÿÿÿÿÿÿÿV§
E__inference_dense_630_layer_call_and_return_conditional_losses_835168^ÉÊ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿV
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_630_layer_call_fn_835158QÉÊ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿV
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_556_layer_call_and_return_conditional_losses_834059X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 
0__inference_leaky_re_lu_556_layer_call_fn_834054K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "ÿÿÿÿÿÿÿÿÿ-§
K__inference_leaky_re_lu_557_layer_call_and_return_conditional_losses_834168X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 
0__inference_leaky_re_lu_557_layer_call_fn_834163K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "ÿÿÿÿÿÿÿÿÿ-§
K__inference_leaky_re_lu_558_layer_call_and_return_conditional_losses_834277X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 
0__inference_leaky_re_lu_558_layer_call_fn_834272K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "ÿÿÿÿÿÿÿÿÿ-§
K__inference_leaky_re_lu_559_layer_call_and_return_conditional_losses_834386X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 
0__inference_leaky_re_lu_559_layer_call_fn_834381K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "ÿÿÿÿÿÿÿÿÿ-§
K__inference_leaky_re_lu_560_layer_call_and_return_conditional_losses_834495X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ-
 
0__inference_leaky_re_lu_560_layer_call_fn_834490K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ-
ª "ÿÿÿÿÿÿÿÿÿ-§
K__inference_leaky_re_lu_561_layer_call_and_return_conditional_losses_834604X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
0__inference_leaky_re_lu_561_layer_call_fn_834599K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "ÿÿÿÿÿÿÿÿÿl§
K__inference_leaky_re_lu_562_layer_call_and_return_conditional_losses_834713X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
0__inference_leaky_re_lu_562_layer_call_fn_834708K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "ÿÿÿÿÿÿÿÿÿl§
K__inference_leaky_re_lu_563_layer_call_and_return_conditional_losses_834822X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
0__inference_leaky_re_lu_563_layer_call_fn_834817K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "ÿÿÿÿÿÿÿÿÿl§
K__inference_leaky_re_lu_564_layer_call_and_return_conditional_losses_834931X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
0__inference_leaky_re_lu_564_layer_call_fn_834926K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "ÿÿÿÿÿÿÿÿÿl§
K__inference_leaky_re_lu_565_layer_call_and_return_conditional_losses_835040X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "%¢"

0ÿÿÿÿÿÿÿÿÿl
 
0__inference_leaky_re_lu_565_layer_call_fn_835035K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿl
ª "ÿÿÿÿÿÿÿÿÿl§
K__inference_leaky_re_lu_566_layer_call_and_return_conditional_losses_835149X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿV
ª "%¢"

0ÿÿÿÿÿÿÿÿÿV
 
0__inference_leaky_re_lu_566_layer_call_fn_835144K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿV
ª "ÿÿÿÿÿÿÿÿÿVº
I__inference_sequential_63_layer_call_and_return_conditional_losses_832587ìzæç67B?A@OP[XZYhitqsr¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæñîðïþÿ£ ¢¡°±¼¹»ºÉÊG¢D
=¢:
0-
normalization_63_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
I__inference_sequential_63_layer_call_and_return_conditional_losses_832768ìzæç67AB?@OPZ[XYhistqr¥¦£¤³´¾¿¼½ÌÍ×ØÕÖåæðñîïþÿ¢£ ¡°±»¼¹ºÉÊG¢D
=¢:
0-
normalization_63_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ª
I__inference_sequential_63_layer_call_and_return_conditional_losses_833332Üzæç67B?A@OP[XZYhitqsr¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæñîðïþÿ£ ¢¡°±¼¹»ºÉÊ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ª
I__inference_sequential_63_layer_call_and_return_conditional_losses_833756Üzæç67AB?@OPZ[XYhistqr¥¦£¤³´¾¿¼½ÌÍ×ØÕÖåæðñîïþÿ¢£ ¡°±»¼¹ºÉÊ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_sequential_63_layer_call_fn_831604ßzæç67B?A@OP[XZYhitqsr¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæñîðïþÿ£ ¢¡°±¼¹»ºÉÊG¢D
=¢:
0-
normalization_63_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_63_layer_call_fn_832406ßzæç67AB?@OPZ[XYhistqr¥¦£¤³´¾¿¼½ÌÍ×ØÕÖåæðñîïþÿ¢£ ¡°±»¼¹ºÉÊG¢D
=¢:
0-
normalization_63_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_63_layer_call_fn_832917Ïzæç67B?A@OP[XZYhitqsr¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæñîðïþÿ£ ¢¡°±¼¹»ºÉÊ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_63_layer_call_fn_833062Ïzæç67AB?@OPZ[XYhistqr¥¦£¤³´¾¿¼½ÌÍ×ØÕÖåæðñîïþÿ¢£ ¡°±»¼¹ºÉÊ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ·
$__inference_signature_wrapper_833903zæç67B?A@OP[XZYhitqsr¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæñîðïþÿ£ ¢¡°±¼¹»ºÉÊY¢V
¢ 
OªL
J
normalization_63_input0-
normalization_63_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_630# 
	dense_630ÿÿÿÿÿÿÿÿÿ