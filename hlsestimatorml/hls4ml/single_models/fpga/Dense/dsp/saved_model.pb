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
dense_945/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:h*!
shared_namedense_945/kernel
u
$dense_945/kernel/Read/ReadVariableOpReadVariableOpdense_945/kernel*
_output_shapes

:h*
dtype0
t
dense_945/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*
shared_namedense_945/bias
m
"dense_945/bias/Read/ReadVariableOpReadVariableOpdense_945/bias*
_output_shapes
:h*
dtype0

batch_normalization_855/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*.
shared_namebatch_normalization_855/gamma

1batch_normalization_855/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_855/gamma*
_output_shapes
:h*
dtype0

batch_normalization_855/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*-
shared_namebatch_normalization_855/beta

0batch_normalization_855/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_855/beta*
_output_shapes
:h*
dtype0

#batch_normalization_855/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#batch_normalization_855/moving_mean

7batch_normalization_855/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_855/moving_mean*
_output_shapes
:h*
dtype0
¦
'batch_normalization_855/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*8
shared_name)'batch_normalization_855/moving_variance

;batch_normalization_855/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_855/moving_variance*
_output_shapes
:h*
dtype0
|
dense_946/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:hh*!
shared_namedense_946/kernel
u
$dense_946/kernel/Read/ReadVariableOpReadVariableOpdense_946/kernel*
_output_shapes

:hh*
dtype0
t
dense_946/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*
shared_namedense_946/bias
m
"dense_946/bias/Read/ReadVariableOpReadVariableOpdense_946/bias*
_output_shapes
:h*
dtype0

batch_normalization_856/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*.
shared_namebatch_normalization_856/gamma

1batch_normalization_856/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_856/gamma*
_output_shapes
:h*
dtype0

batch_normalization_856/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*-
shared_namebatch_normalization_856/beta

0batch_normalization_856/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_856/beta*
_output_shapes
:h*
dtype0

#batch_normalization_856/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#batch_normalization_856/moving_mean

7batch_normalization_856/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_856/moving_mean*
_output_shapes
:h*
dtype0
¦
'batch_normalization_856/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*8
shared_name)'batch_normalization_856/moving_variance

;batch_normalization_856/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_856/moving_variance*
_output_shapes
:h*
dtype0
|
dense_947/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:hh*!
shared_namedense_947/kernel
u
$dense_947/kernel/Read/ReadVariableOpReadVariableOpdense_947/kernel*
_output_shapes

:hh*
dtype0
t
dense_947/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*
shared_namedense_947/bias
m
"dense_947/bias/Read/ReadVariableOpReadVariableOpdense_947/bias*
_output_shapes
:h*
dtype0

batch_normalization_857/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*.
shared_namebatch_normalization_857/gamma

1batch_normalization_857/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_857/gamma*
_output_shapes
:h*
dtype0

batch_normalization_857/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*-
shared_namebatch_normalization_857/beta

0batch_normalization_857/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_857/beta*
_output_shapes
:h*
dtype0

#batch_normalization_857/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#batch_normalization_857/moving_mean

7batch_normalization_857/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_857/moving_mean*
_output_shapes
:h*
dtype0
¦
'batch_normalization_857/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*8
shared_name)'batch_normalization_857/moving_variance

;batch_normalization_857/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_857/moving_variance*
_output_shapes
:h*
dtype0
|
dense_948/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:hh*!
shared_namedense_948/kernel
u
$dense_948/kernel/Read/ReadVariableOpReadVariableOpdense_948/kernel*
_output_shapes

:hh*
dtype0
t
dense_948/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*
shared_namedense_948/bias
m
"dense_948/bias/Read/ReadVariableOpReadVariableOpdense_948/bias*
_output_shapes
:h*
dtype0

batch_normalization_858/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*.
shared_namebatch_normalization_858/gamma

1batch_normalization_858/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_858/gamma*
_output_shapes
:h*
dtype0

batch_normalization_858/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*-
shared_namebatch_normalization_858/beta

0batch_normalization_858/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_858/beta*
_output_shapes
:h*
dtype0

#batch_normalization_858/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#batch_normalization_858/moving_mean

7batch_normalization_858/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_858/moving_mean*
_output_shapes
:h*
dtype0
¦
'batch_normalization_858/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*8
shared_name)'batch_normalization_858/moving_variance

;batch_normalization_858/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_858/moving_variance*
_output_shapes
:h*
dtype0
|
dense_949/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:hh*!
shared_namedense_949/kernel
u
$dense_949/kernel/Read/ReadVariableOpReadVariableOpdense_949/kernel*
_output_shapes

:hh*
dtype0
t
dense_949/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*
shared_namedense_949/bias
m
"dense_949/bias/Read/ReadVariableOpReadVariableOpdense_949/bias*
_output_shapes
:h*
dtype0

batch_normalization_859/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*.
shared_namebatch_normalization_859/gamma

1batch_normalization_859/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_859/gamma*
_output_shapes
:h*
dtype0

batch_normalization_859/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*-
shared_namebatch_normalization_859/beta

0batch_normalization_859/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_859/beta*
_output_shapes
:h*
dtype0

#batch_normalization_859/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#batch_normalization_859/moving_mean

7batch_normalization_859/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_859/moving_mean*
_output_shapes
:h*
dtype0
¦
'batch_normalization_859/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*8
shared_name)'batch_normalization_859/moving_variance

;batch_normalization_859/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_859/moving_variance*
_output_shapes
:h*
dtype0
|
dense_950/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:h/*!
shared_namedense_950/kernel
u
$dense_950/kernel/Read/ReadVariableOpReadVariableOpdense_950/kernel*
_output_shapes

:h/*
dtype0
t
dense_950/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_950/bias
m
"dense_950/bias/Read/ReadVariableOpReadVariableOpdense_950/bias*
_output_shapes
:/*
dtype0

batch_normalization_860/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*.
shared_namebatch_normalization_860/gamma

1batch_normalization_860/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_860/gamma*
_output_shapes
:/*
dtype0

batch_normalization_860/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*-
shared_namebatch_normalization_860/beta

0batch_normalization_860/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_860/beta*
_output_shapes
:/*
dtype0

#batch_normalization_860/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*4
shared_name%#batch_normalization_860/moving_mean

7batch_normalization_860/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_860/moving_mean*
_output_shapes
:/*
dtype0
¦
'batch_normalization_860/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*8
shared_name)'batch_normalization_860/moving_variance

;batch_normalization_860/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_860/moving_variance*
_output_shapes
:/*
dtype0
|
dense_951/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*!
shared_namedense_951/kernel
u
$dense_951/kernel/Read/ReadVariableOpReadVariableOpdense_951/kernel*
_output_shapes

://*
dtype0
t
dense_951/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_951/bias
m
"dense_951/bias/Read/ReadVariableOpReadVariableOpdense_951/bias*
_output_shapes
:/*
dtype0

batch_normalization_861/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*.
shared_namebatch_normalization_861/gamma

1batch_normalization_861/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_861/gamma*
_output_shapes
:/*
dtype0

batch_normalization_861/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*-
shared_namebatch_normalization_861/beta

0batch_normalization_861/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_861/beta*
_output_shapes
:/*
dtype0

#batch_normalization_861/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*4
shared_name%#batch_normalization_861/moving_mean

7batch_normalization_861/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_861/moving_mean*
_output_shapes
:/*
dtype0
¦
'batch_normalization_861/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*8
shared_name)'batch_normalization_861/moving_variance

;batch_normalization_861/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_861/moving_variance*
_output_shapes
:/*
dtype0
|
dense_952/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*!
shared_namedense_952/kernel
u
$dense_952/kernel/Read/ReadVariableOpReadVariableOpdense_952/kernel*
_output_shapes

:/*
dtype0
t
dense_952/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_952/bias
m
"dense_952/bias/Read/ReadVariableOpReadVariableOpdense_952/bias*
_output_shapes
:*
dtype0

batch_normalization_862/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_862/gamma

1batch_normalization_862/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_862/gamma*
_output_shapes
:*
dtype0

batch_normalization_862/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_862/beta

0batch_normalization_862/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_862/beta*
_output_shapes
:*
dtype0

#batch_normalization_862/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_862/moving_mean

7batch_normalization_862/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_862/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_862/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_862/moving_variance

;batch_normalization_862/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_862/moving_variance*
_output_shapes
:*
dtype0
|
dense_953/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_953/kernel
u
$dense_953/kernel/Read/ReadVariableOpReadVariableOpdense_953/kernel*
_output_shapes

:*
dtype0
t
dense_953/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_953/bias
m
"dense_953/bias/Read/ReadVariableOpReadVariableOpdense_953/bias*
_output_shapes
:*
dtype0

batch_normalization_863/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_863/gamma

1batch_normalization_863/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_863/gamma*
_output_shapes
:*
dtype0

batch_normalization_863/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_863/beta

0batch_normalization_863/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_863/beta*
_output_shapes
:*
dtype0

#batch_normalization_863/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_863/moving_mean

7batch_normalization_863/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_863/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_863/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_863/moving_variance

;batch_normalization_863/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_863/moving_variance*
_output_shapes
:*
dtype0
|
dense_954/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_954/kernel
u
$dense_954/kernel/Read/ReadVariableOpReadVariableOpdense_954/kernel*
_output_shapes

:*
dtype0
t
dense_954/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_954/bias
m
"dense_954/bias/Read/ReadVariableOpReadVariableOpdense_954/bias*
_output_shapes
:*
dtype0

batch_normalization_864/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_864/gamma

1batch_normalization_864/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_864/gamma*
_output_shapes
:*
dtype0

batch_normalization_864/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_864/beta

0batch_normalization_864/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_864/beta*
_output_shapes
:*
dtype0

#batch_normalization_864/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_864/moving_mean

7batch_normalization_864/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_864/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_864/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_864/moving_variance

;batch_normalization_864/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_864/moving_variance*
_output_shapes
:*
dtype0
|
dense_955/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_955/kernel
u
$dense_955/kernel/Read/ReadVariableOpReadVariableOpdense_955/kernel*
_output_shapes

:*
dtype0
t
dense_955/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_955/bias
m
"dense_955/bias/Read/ReadVariableOpReadVariableOpdense_955/bias*
_output_shapes
:*
dtype0

batch_normalization_865/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_865/gamma

1batch_normalization_865/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_865/gamma*
_output_shapes
:*
dtype0

batch_normalization_865/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_865/beta

0batch_normalization_865/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_865/beta*
_output_shapes
:*
dtype0

#batch_normalization_865/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_865/moving_mean

7batch_normalization_865/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_865/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_865/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_865/moving_variance

;batch_normalization_865/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_865/moving_variance*
_output_shapes
:*
dtype0
|
dense_956/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_956/kernel
u
$dense_956/kernel/Read/ReadVariableOpReadVariableOpdense_956/kernel*
_output_shapes

:*
dtype0
t
dense_956/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_956/bias
m
"dense_956/bias/Read/ReadVariableOpReadVariableOpdense_956/bias*
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
Adam/dense_945/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:h*(
shared_nameAdam/dense_945/kernel/m

+Adam/dense_945/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_945/kernel/m*
_output_shapes

:h*
dtype0

Adam/dense_945/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*&
shared_nameAdam/dense_945/bias/m
{
)Adam/dense_945/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_945/bias/m*
_output_shapes
:h*
dtype0
 
$Adam/batch_normalization_855/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*5
shared_name&$Adam/batch_normalization_855/gamma/m

8Adam/batch_normalization_855/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_855/gamma/m*
_output_shapes
:h*
dtype0

#Adam/batch_normalization_855/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#Adam/batch_normalization_855/beta/m

7Adam/batch_normalization_855/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_855/beta/m*
_output_shapes
:h*
dtype0

Adam/dense_946/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:hh*(
shared_nameAdam/dense_946/kernel/m

+Adam/dense_946/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_946/kernel/m*
_output_shapes

:hh*
dtype0

Adam/dense_946/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*&
shared_nameAdam/dense_946/bias/m
{
)Adam/dense_946/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_946/bias/m*
_output_shapes
:h*
dtype0
 
$Adam/batch_normalization_856/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*5
shared_name&$Adam/batch_normalization_856/gamma/m

8Adam/batch_normalization_856/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_856/gamma/m*
_output_shapes
:h*
dtype0

#Adam/batch_normalization_856/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#Adam/batch_normalization_856/beta/m

7Adam/batch_normalization_856/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_856/beta/m*
_output_shapes
:h*
dtype0

Adam/dense_947/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:hh*(
shared_nameAdam/dense_947/kernel/m

+Adam/dense_947/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_947/kernel/m*
_output_shapes

:hh*
dtype0

Adam/dense_947/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*&
shared_nameAdam/dense_947/bias/m
{
)Adam/dense_947/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_947/bias/m*
_output_shapes
:h*
dtype0
 
$Adam/batch_normalization_857/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*5
shared_name&$Adam/batch_normalization_857/gamma/m

8Adam/batch_normalization_857/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_857/gamma/m*
_output_shapes
:h*
dtype0

#Adam/batch_normalization_857/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#Adam/batch_normalization_857/beta/m

7Adam/batch_normalization_857/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_857/beta/m*
_output_shapes
:h*
dtype0

Adam/dense_948/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:hh*(
shared_nameAdam/dense_948/kernel/m

+Adam/dense_948/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_948/kernel/m*
_output_shapes

:hh*
dtype0

Adam/dense_948/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*&
shared_nameAdam/dense_948/bias/m
{
)Adam/dense_948/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_948/bias/m*
_output_shapes
:h*
dtype0
 
$Adam/batch_normalization_858/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*5
shared_name&$Adam/batch_normalization_858/gamma/m

8Adam/batch_normalization_858/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_858/gamma/m*
_output_shapes
:h*
dtype0

#Adam/batch_normalization_858/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#Adam/batch_normalization_858/beta/m

7Adam/batch_normalization_858/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_858/beta/m*
_output_shapes
:h*
dtype0

Adam/dense_949/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:hh*(
shared_nameAdam/dense_949/kernel/m

+Adam/dense_949/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_949/kernel/m*
_output_shapes

:hh*
dtype0

Adam/dense_949/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*&
shared_nameAdam/dense_949/bias/m
{
)Adam/dense_949/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_949/bias/m*
_output_shapes
:h*
dtype0
 
$Adam/batch_normalization_859/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*5
shared_name&$Adam/batch_normalization_859/gamma/m

8Adam/batch_normalization_859/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_859/gamma/m*
_output_shapes
:h*
dtype0

#Adam/batch_normalization_859/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#Adam/batch_normalization_859/beta/m

7Adam/batch_normalization_859/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_859/beta/m*
_output_shapes
:h*
dtype0

Adam/dense_950/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:h/*(
shared_nameAdam/dense_950/kernel/m

+Adam/dense_950/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_950/kernel/m*
_output_shapes

:h/*
dtype0

Adam/dense_950/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*&
shared_nameAdam/dense_950/bias/m
{
)Adam/dense_950/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_950/bias/m*
_output_shapes
:/*
dtype0
 
$Adam/batch_normalization_860/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*5
shared_name&$Adam/batch_normalization_860/gamma/m

8Adam/batch_normalization_860/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_860/gamma/m*
_output_shapes
:/*
dtype0

#Adam/batch_normalization_860/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*4
shared_name%#Adam/batch_normalization_860/beta/m

7Adam/batch_normalization_860/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_860/beta/m*
_output_shapes
:/*
dtype0

Adam/dense_951/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*(
shared_nameAdam/dense_951/kernel/m

+Adam/dense_951/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_951/kernel/m*
_output_shapes

://*
dtype0

Adam/dense_951/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*&
shared_nameAdam/dense_951/bias/m
{
)Adam/dense_951/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_951/bias/m*
_output_shapes
:/*
dtype0
 
$Adam/batch_normalization_861/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*5
shared_name&$Adam/batch_normalization_861/gamma/m

8Adam/batch_normalization_861/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_861/gamma/m*
_output_shapes
:/*
dtype0

#Adam/batch_normalization_861/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*4
shared_name%#Adam/batch_normalization_861/beta/m

7Adam/batch_normalization_861/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_861/beta/m*
_output_shapes
:/*
dtype0

Adam/dense_952/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*(
shared_nameAdam/dense_952/kernel/m

+Adam/dense_952/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_952/kernel/m*
_output_shapes

:/*
dtype0

Adam/dense_952/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_952/bias/m
{
)Adam/dense_952/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_952/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_862/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_862/gamma/m

8Adam/batch_normalization_862/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_862/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_862/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_862/beta/m

7Adam/batch_normalization_862/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_862/beta/m*
_output_shapes
:*
dtype0

Adam/dense_953/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_953/kernel/m

+Adam/dense_953/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_953/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_953/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_953/bias/m
{
)Adam/dense_953/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_953/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_863/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_863/gamma/m

8Adam/batch_normalization_863/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_863/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_863/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_863/beta/m

7Adam/batch_normalization_863/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_863/beta/m*
_output_shapes
:*
dtype0

Adam/dense_954/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_954/kernel/m

+Adam/dense_954/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_954/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_954/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_954/bias/m
{
)Adam/dense_954/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_954/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_864/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_864/gamma/m

8Adam/batch_normalization_864/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_864/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_864/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_864/beta/m

7Adam/batch_normalization_864/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_864/beta/m*
_output_shapes
:*
dtype0

Adam/dense_955/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_955/kernel/m

+Adam/dense_955/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_955/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_955/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_955/bias/m
{
)Adam/dense_955/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_955/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_865/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_865/gamma/m

8Adam/batch_normalization_865/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_865/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_865/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_865/beta/m

7Adam/batch_normalization_865/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_865/beta/m*
_output_shapes
:*
dtype0

Adam/dense_956/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_956/kernel/m

+Adam/dense_956/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_956/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_956/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_956/bias/m
{
)Adam/dense_956/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_956/bias/m*
_output_shapes
:*
dtype0

Adam/dense_945/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:h*(
shared_nameAdam/dense_945/kernel/v

+Adam/dense_945/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_945/kernel/v*
_output_shapes

:h*
dtype0

Adam/dense_945/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*&
shared_nameAdam/dense_945/bias/v
{
)Adam/dense_945/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_945/bias/v*
_output_shapes
:h*
dtype0
 
$Adam/batch_normalization_855/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*5
shared_name&$Adam/batch_normalization_855/gamma/v

8Adam/batch_normalization_855/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_855/gamma/v*
_output_shapes
:h*
dtype0

#Adam/batch_normalization_855/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#Adam/batch_normalization_855/beta/v

7Adam/batch_normalization_855/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_855/beta/v*
_output_shapes
:h*
dtype0

Adam/dense_946/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:hh*(
shared_nameAdam/dense_946/kernel/v

+Adam/dense_946/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_946/kernel/v*
_output_shapes

:hh*
dtype0

Adam/dense_946/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*&
shared_nameAdam/dense_946/bias/v
{
)Adam/dense_946/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_946/bias/v*
_output_shapes
:h*
dtype0
 
$Adam/batch_normalization_856/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*5
shared_name&$Adam/batch_normalization_856/gamma/v

8Adam/batch_normalization_856/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_856/gamma/v*
_output_shapes
:h*
dtype0

#Adam/batch_normalization_856/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#Adam/batch_normalization_856/beta/v

7Adam/batch_normalization_856/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_856/beta/v*
_output_shapes
:h*
dtype0

Adam/dense_947/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:hh*(
shared_nameAdam/dense_947/kernel/v

+Adam/dense_947/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_947/kernel/v*
_output_shapes

:hh*
dtype0

Adam/dense_947/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*&
shared_nameAdam/dense_947/bias/v
{
)Adam/dense_947/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_947/bias/v*
_output_shapes
:h*
dtype0
 
$Adam/batch_normalization_857/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*5
shared_name&$Adam/batch_normalization_857/gamma/v

8Adam/batch_normalization_857/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_857/gamma/v*
_output_shapes
:h*
dtype0

#Adam/batch_normalization_857/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#Adam/batch_normalization_857/beta/v

7Adam/batch_normalization_857/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_857/beta/v*
_output_shapes
:h*
dtype0

Adam/dense_948/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:hh*(
shared_nameAdam/dense_948/kernel/v

+Adam/dense_948/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_948/kernel/v*
_output_shapes

:hh*
dtype0

Adam/dense_948/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*&
shared_nameAdam/dense_948/bias/v
{
)Adam/dense_948/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_948/bias/v*
_output_shapes
:h*
dtype0
 
$Adam/batch_normalization_858/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*5
shared_name&$Adam/batch_normalization_858/gamma/v

8Adam/batch_normalization_858/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_858/gamma/v*
_output_shapes
:h*
dtype0

#Adam/batch_normalization_858/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#Adam/batch_normalization_858/beta/v

7Adam/batch_normalization_858/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_858/beta/v*
_output_shapes
:h*
dtype0

Adam/dense_949/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:hh*(
shared_nameAdam/dense_949/kernel/v

+Adam/dense_949/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_949/kernel/v*
_output_shapes

:hh*
dtype0

Adam/dense_949/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*&
shared_nameAdam/dense_949/bias/v
{
)Adam/dense_949/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_949/bias/v*
_output_shapes
:h*
dtype0
 
$Adam/batch_normalization_859/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*5
shared_name&$Adam/batch_normalization_859/gamma/v

8Adam/batch_normalization_859/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_859/gamma/v*
_output_shapes
:h*
dtype0

#Adam/batch_normalization_859/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:h*4
shared_name%#Adam/batch_normalization_859/beta/v

7Adam/batch_normalization_859/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_859/beta/v*
_output_shapes
:h*
dtype0

Adam/dense_950/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:h/*(
shared_nameAdam/dense_950/kernel/v

+Adam/dense_950/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_950/kernel/v*
_output_shapes

:h/*
dtype0

Adam/dense_950/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*&
shared_nameAdam/dense_950/bias/v
{
)Adam/dense_950/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_950/bias/v*
_output_shapes
:/*
dtype0
 
$Adam/batch_normalization_860/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*5
shared_name&$Adam/batch_normalization_860/gamma/v

8Adam/batch_normalization_860/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_860/gamma/v*
_output_shapes
:/*
dtype0

#Adam/batch_normalization_860/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*4
shared_name%#Adam/batch_normalization_860/beta/v

7Adam/batch_normalization_860/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_860/beta/v*
_output_shapes
:/*
dtype0

Adam/dense_951/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*(
shared_nameAdam/dense_951/kernel/v

+Adam/dense_951/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_951/kernel/v*
_output_shapes

://*
dtype0

Adam/dense_951/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*&
shared_nameAdam/dense_951/bias/v
{
)Adam/dense_951/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_951/bias/v*
_output_shapes
:/*
dtype0
 
$Adam/batch_normalization_861/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*5
shared_name&$Adam/batch_normalization_861/gamma/v

8Adam/batch_normalization_861/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_861/gamma/v*
_output_shapes
:/*
dtype0

#Adam/batch_normalization_861/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*4
shared_name%#Adam/batch_normalization_861/beta/v

7Adam/batch_normalization_861/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_861/beta/v*
_output_shapes
:/*
dtype0

Adam/dense_952/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*(
shared_nameAdam/dense_952/kernel/v

+Adam/dense_952/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_952/kernel/v*
_output_shapes

:/*
dtype0

Adam/dense_952/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_952/bias/v
{
)Adam/dense_952/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_952/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_862/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_862/gamma/v

8Adam/batch_normalization_862/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_862/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_862/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_862/beta/v

7Adam/batch_normalization_862/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_862/beta/v*
_output_shapes
:*
dtype0

Adam/dense_953/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_953/kernel/v

+Adam/dense_953/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_953/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_953/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_953/bias/v
{
)Adam/dense_953/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_953/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_863/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_863/gamma/v

8Adam/batch_normalization_863/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_863/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_863/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_863/beta/v

7Adam/batch_normalization_863/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_863/beta/v*
_output_shapes
:*
dtype0

Adam/dense_954/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_954/kernel/v

+Adam/dense_954/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_954/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_954/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_954/bias/v
{
)Adam/dense_954/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_954/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_864/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_864/gamma/v

8Adam/batch_normalization_864/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_864/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_864/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_864/beta/v

7Adam/batch_normalization_864/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_864/beta/v*
_output_shapes
:*
dtype0

Adam/dense_955/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_955/kernel/v

+Adam/dense_955/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_955/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_955/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_955/bias/v
{
)Adam/dense_955/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_955/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_865/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_865/gamma/v

8Adam/batch_normalization_865/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_865/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_865/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_865/beta/v

7Adam/batch_normalization_865/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_865/beta/v*
_output_shapes
:*
dtype0

Adam/dense_956/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_956/kernel/v

+Adam/dense_956/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_956/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_956/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_956/bias/v
{
)Adam/dense_956/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_956/bias/v*
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
VARIABLE_VALUEdense_945/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_945/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_855/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_855/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_855/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_855/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEdense_946/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_946/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_856/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_856/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_856/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_856/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEdense_947/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_947/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_857/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_857/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_857/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_857/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEdense_948/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_948/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_858/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_858/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_858/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_858/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_949/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_949/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_859/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_859/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_859/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_859/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_950/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_950/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_860/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_860/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_860/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_860/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_951/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_951/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_861/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_861/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_861/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_861/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_952/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_952/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_862/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_862/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_862/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_862/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_953/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_953/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_863/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_863/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_863/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_863/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_954/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_954/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_864/gamma6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_864/beta5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_864/moving_mean<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_864/moving_variance@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_955/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_955/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_865/gamma6layer_with_weights-22/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_865/beta5layer_with_weights-22/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_865/moving_mean<layer_with_weights-22/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_865/moving_variance@layer_with_weights-22/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_956/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_956/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_945/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_945/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_855/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_855/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_946/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_946/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_856/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_856/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_947/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_947/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_857/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_857/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_948/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_948/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_858/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_858/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_949/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_949/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_859/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_859/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_950/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_950/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_860/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_860/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_951/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_951/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_861/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_861/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_952/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_952/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_862/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_862/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_953/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_953/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_863/gamma/mRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_863/beta/mQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_954/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_954/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_864/gamma/mRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_864/beta/mQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_955/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_955/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_865/gamma/mRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_865/beta/mQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_956/kernel/mSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_956/bias/mQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_945/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_945/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_855/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_855/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_946/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_946/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_856/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_856/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_947/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_947/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_857/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_857/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_948/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_948/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_858/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_858/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_949/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_949/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_859/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_859/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_950/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_950/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_860/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_860/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_951/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_951/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_861/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_861/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_952/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_952/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_862/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_862/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_953/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_953/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_863/gamma/vRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_863/beta/vQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_954/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_954/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_864/gamma/vRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_864/beta/vQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_955/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_955/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_865/gamma/vRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_865/beta/vQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_956/kernel/vSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_956/bias/vQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

&serving_default_normalization_90_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ì
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_90_inputConstConst_1dense_945/kerneldense_945/bias'batch_normalization_855/moving_variancebatch_normalization_855/gamma#batch_normalization_855/moving_meanbatch_normalization_855/betadense_946/kerneldense_946/bias'batch_normalization_856/moving_variancebatch_normalization_856/gamma#batch_normalization_856/moving_meanbatch_normalization_856/betadense_947/kerneldense_947/bias'batch_normalization_857/moving_variancebatch_normalization_857/gamma#batch_normalization_857/moving_meanbatch_normalization_857/betadense_948/kerneldense_948/bias'batch_normalization_858/moving_variancebatch_normalization_858/gamma#batch_normalization_858/moving_meanbatch_normalization_858/betadense_949/kerneldense_949/bias'batch_normalization_859/moving_variancebatch_normalization_859/gamma#batch_normalization_859/moving_meanbatch_normalization_859/betadense_950/kerneldense_950/bias'batch_normalization_860/moving_variancebatch_normalization_860/gamma#batch_normalization_860/moving_meanbatch_normalization_860/betadense_951/kerneldense_951/bias'batch_normalization_861/moving_variancebatch_normalization_861/gamma#batch_normalization_861/moving_meanbatch_normalization_861/betadense_952/kerneldense_952/bias'batch_normalization_862/moving_variancebatch_normalization_862/gamma#batch_normalization_862/moving_meanbatch_normalization_862/betadense_953/kerneldense_953/bias'batch_normalization_863/moving_variancebatch_normalization_863/gamma#batch_normalization_863/moving_meanbatch_normalization_863/betadense_954/kerneldense_954/bias'batch_normalization_864/moving_variancebatch_normalization_864/gamma#batch_normalization_864/moving_meanbatch_normalization_864/betadense_955/kerneldense_955/bias'batch_normalization_865/moving_variancebatch_normalization_865/gamma#batch_normalization_865/moving_meanbatch_normalization_865/betadense_956/kerneldense_956/bias*R
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
$__inference_signature_wrapper_839553
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ÙC
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_945/kernel/Read/ReadVariableOp"dense_945/bias/Read/ReadVariableOp1batch_normalization_855/gamma/Read/ReadVariableOp0batch_normalization_855/beta/Read/ReadVariableOp7batch_normalization_855/moving_mean/Read/ReadVariableOp;batch_normalization_855/moving_variance/Read/ReadVariableOp$dense_946/kernel/Read/ReadVariableOp"dense_946/bias/Read/ReadVariableOp1batch_normalization_856/gamma/Read/ReadVariableOp0batch_normalization_856/beta/Read/ReadVariableOp7batch_normalization_856/moving_mean/Read/ReadVariableOp;batch_normalization_856/moving_variance/Read/ReadVariableOp$dense_947/kernel/Read/ReadVariableOp"dense_947/bias/Read/ReadVariableOp1batch_normalization_857/gamma/Read/ReadVariableOp0batch_normalization_857/beta/Read/ReadVariableOp7batch_normalization_857/moving_mean/Read/ReadVariableOp;batch_normalization_857/moving_variance/Read/ReadVariableOp$dense_948/kernel/Read/ReadVariableOp"dense_948/bias/Read/ReadVariableOp1batch_normalization_858/gamma/Read/ReadVariableOp0batch_normalization_858/beta/Read/ReadVariableOp7batch_normalization_858/moving_mean/Read/ReadVariableOp;batch_normalization_858/moving_variance/Read/ReadVariableOp$dense_949/kernel/Read/ReadVariableOp"dense_949/bias/Read/ReadVariableOp1batch_normalization_859/gamma/Read/ReadVariableOp0batch_normalization_859/beta/Read/ReadVariableOp7batch_normalization_859/moving_mean/Read/ReadVariableOp;batch_normalization_859/moving_variance/Read/ReadVariableOp$dense_950/kernel/Read/ReadVariableOp"dense_950/bias/Read/ReadVariableOp1batch_normalization_860/gamma/Read/ReadVariableOp0batch_normalization_860/beta/Read/ReadVariableOp7batch_normalization_860/moving_mean/Read/ReadVariableOp;batch_normalization_860/moving_variance/Read/ReadVariableOp$dense_951/kernel/Read/ReadVariableOp"dense_951/bias/Read/ReadVariableOp1batch_normalization_861/gamma/Read/ReadVariableOp0batch_normalization_861/beta/Read/ReadVariableOp7batch_normalization_861/moving_mean/Read/ReadVariableOp;batch_normalization_861/moving_variance/Read/ReadVariableOp$dense_952/kernel/Read/ReadVariableOp"dense_952/bias/Read/ReadVariableOp1batch_normalization_862/gamma/Read/ReadVariableOp0batch_normalization_862/beta/Read/ReadVariableOp7batch_normalization_862/moving_mean/Read/ReadVariableOp;batch_normalization_862/moving_variance/Read/ReadVariableOp$dense_953/kernel/Read/ReadVariableOp"dense_953/bias/Read/ReadVariableOp1batch_normalization_863/gamma/Read/ReadVariableOp0batch_normalization_863/beta/Read/ReadVariableOp7batch_normalization_863/moving_mean/Read/ReadVariableOp;batch_normalization_863/moving_variance/Read/ReadVariableOp$dense_954/kernel/Read/ReadVariableOp"dense_954/bias/Read/ReadVariableOp1batch_normalization_864/gamma/Read/ReadVariableOp0batch_normalization_864/beta/Read/ReadVariableOp7batch_normalization_864/moving_mean/Read/ReadVariableOp;batch_normalization_864/moving_variance/Read/ReadVariableOp$dense_955/kernel/Read/ReadVariableOp"dense_955/bias/Read/ReadVariableOp1batch_normalization_865/gamma/Read/ReadVariableOp0batch_normalization_865/beta/Read/ReadVariableOp7batch_normalization_865/moving_mean/Read/ReadVariableOp;batch_normalization_865/moving_variance/Read/ReadVariableOp$dense_956/kernel/Read/ReadVariableOp"dense_956/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_945/kernel/m/Read/ReadVariableOp)Adam/dense_945/bias/m/Read/ReadVariableOp8Adam/batch_normalization_855/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_855/beta/m/Read/ReadVariableOp+Adam/dense_946/kernel/m/Read/ReadVariableOp)Adam/dense_946/bias/m/Read/ReadVariableOp8Adam/batch_normalization_856/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_856/beta/m/Read/ReadVariableOp+Adam/dense_947/kernel/m/Read/ReadVariableOp)Adam/dense_947/bias/m/Read/ReadVariableOp8Adam/batch_normalization_857/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_857/beta/m/Read/ReadVariableOp+Adam/dense_948/kernel/m/Read/ReadVariableOp)Adam/dense_948/bias/m/Read/ReadVariableOp8Adam/batch_normalization_858/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_858/beta/m/Read/ReadVariableOp+Adam/dense_949/kernel/m/Read/ReadVariableOp)Adam/dense_949/bias/m/Read/ReadVariableOp8Adam/batch_normalization_859/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_859/beta/m/Read/ReadVariableOp+Adam/dense_950/kernel/m/Read/ReadVariableOp)Adam/dense_950/bias/m/Read/ReadVariableOp8Adam/batch_normalization_860/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_860/beta/m/Read/ReadVariableOp+Adam/dense_951/kernel/m/Read/ReadVariableOp)Adam/dense_951/bias/m/Read/ReadVariableOp8Adam/batch_normalization_861/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_861/beta/m/Read/ReadVariableOp+Adam/dense_952/kernel/m/Read/ReadVariableOp)Adam/dense_952/bias/m/Read/ReadVariableOp8Adam/batch_normalization_862/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_862/beta/m/Read/ReadVariableOp+Adam/dense_953/kernel/m/Read/ReadVariableOp)Adam/dense_953/bias/m/Read/ReadVariableOp8Adam/batch_normalization_863/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_863/beta/m/Read/ReadVariableOp+Adam/dense_954/kernel/m/Read/ReadVariableOp)Adam/dense_954/bias/m/Read/ReadVariableOp8Adam/batch_normalization_864/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_864/beta/m/Read/ReadVariableOp+Adam/dense_955/kernel/m/Read/ReadVariableOp)Adam/dense_955/bias/m/Read/ReadVariableOp8Adam/batch_normalization_865/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_865/beta/m/Read/ReadVariableOp+Adam/dense_956/kernel/m/Read/ReadVariableOp)Adam/dense_956/bias/m/Read/ReadVariableOp+Adam/dense_945/kernel/v/Read/ReadVariableOp)Adam/dense_945/bias/v/Read/ReadVariableOp8Adam/batch_normalization_855/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_855/beta/v/Read/ReadVariableOp+Adam/dense_946/kernel/v/Read/ReadVariableOp)Adam/dense_946/bias/v/Read/ReadVariableOp8Adam/batch_normalization_856/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_856/beta/v/Read/ReadVariableOp+Adam/dense_947/kernel/v/Read/ReadVariableOp)Adam/dense_947/bias/v/Read/ReadVariableOp8Adam/batch_normalization_857/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_857/beta/v/Read/ReadVariableOp+Adam/dense_948/kernel/v/Read/ReadVariableOp)Adam/dense_948/bias/v/Read/ReadVariableOp8Adam/batch_normalization_858/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_858/beta/v/Read/ReadVariableOp+Adam/dense_949/kernel/v/Read/ReadVariableOp)Adam/dense_949/bias/v/Read/ReadVariableOp8Adam/batch_normalization_859/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_859/beta/v/Read/ReadVariableOp+Adam/dense_950/kernel/v/Read/ReadVariableOp)Adam/dense_950/bias/v/Read/ReadVariableOp8Adam/batch_normalization_860/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_860/beta/v/Read/ReadVariableOp+Adam/dense_951/kernel/v/Read/ReadVariableOp)Adam/dense_951/bias/v/Read/ReadVariableOp8Adam/batch_normalization_861/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_861/beta/v/Read/ReadVariableOp+Adam/dense_952/kernel/v/Read/ReadVariableOp)Adam/dense_952/bias/v/Read/ReadVariableOp8Adam/batch_normalization_862/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_862/beta/v/Read/ReadVariableOp+Adam/dense_953/kernel/v/Read/ReadVariableOp)Adam/dense_953/bias/v/Read/ReadVariableOp8Adam/batch_normalization_863/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_863/beta/v/Read/ReadVariableOp+Adam/dense_954/kernel/v/Read/ReadVariableOp)Adam/dense_954/bias/v/Read/ReadVariableOp8Adam/batch_normalization_864/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_864/beta/v/Read/ReadVariableOp+Adam/dense_955/kernel/v/Read/ReadVariableOp)Adam/dense_955/bias/v/Read/ReadVariableOp8Adam/batch_normalization_865/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_865/beta/v/Read/ReadVariableOp+Adam/dense_956/kernel/v/Read/ReadVariableOp)Adam/dense_956/bias/v/Read/ReadVariableOpConst_2*¹
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
__inference__traced_save_841350
)
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_945/kerneldense_945/biasbatch_normalization_855/gammabatch_normalization_855/beta#batch_normalization_855/moving_mean'batch_normalization_855/moving_variancedense_946/kerneldense_946/biasbatch_normalization_856/gammabatch_normalization_856/beta#batch_normalization_856/moving_mean'batch_normalization_856/moving_variancedense_947/kerneldense_947/biasbatch_normalization_857/gammabatch_normalization_857/beta#batch_normalization_857/moving_mean'batch_normalization_857/moving_variancedense_948/kerneldense_948/biasbatch_normalization_858/gammabatch_normalization_858/beta#batch_normalization_858/moving_mean'batch_normalization_858/moving_variancedense_949/kerneldense_949/biasbatch_normalization_859/gammabatch_normalization_859/beta#batch_normalization_859/moving_mean'batch_normalization_859/moving_variancedense_950/kerneldense_950/biasbatch_normalization_860/gammabatch_normalization_860/beta#batch_normalization_860/moving_mean'batch_normalization_860/moving_variancedense_951/kerneldense_951/biasbatch_normalization_861/gammabatch_normalization_861/beta#batch_normalization_861/moving_mean'batch_normalization_861/moving_variancedense_952/kerneldense_952/biasbatch_normalization_862/gammabatch_normalization_862/beta#batch_normalization_862/moving_mean'batch_normalization_862/moving_variancedense_953/kerneldense_953/biasbatch_normalization_863/gammabatch_normalization_863/beta#batch_normalization_863/moving_mean'batch_normalization_863/moving_variancedense_954/kerneldense_954/biasbatch_normalization_864/gammabatch_normalization_864/beta#batch_normalization_864/moving_mean'batch_normalization_864/moving_variancedense_955/kerneldense_955/biasbatch_normalization_865/gammabatch_normalization_865/beta#batch_normalization_865/moving_mean'batch_normalization_865/moving_variancedense_956/kerneldense_956/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_945/kernel/mAdam/dense_945/bias/m$Adam/batch_normalization_855/gamma/m#Adam/batch_normalization_855/beta/mAdam/dense_946/kernel/mAdam/dense_946/bias/m$Adam/batch_normalization_856/gamma/m#Adam/batch_normalization_856/beta/mAdam/dense_947/kernel/mAdam/dense_947/bias/m$Adam/batch_normalization_857/gamma/m#Adam/batch_normalization_857/beta/mAdam/dense_948/kernel/mAdam/dense_948/bias/m$Adam/batch_normalization_858/gamma/m#Adam/batch_normalization_858/beta/mAdam/dense_949/kernel/mAdam/dense_949/bias/m$Adam/batch_normalization_859/gamma/m#Adam/batch_normalization_859/beta/mAdam/dense_950/kernel/mAdam/dense_950/bias/m$Adam/batch_normalization_860/gamma/m#Adam/batch_normalization_860/beta/mAdam/dense_951/kernel/mAdam/dense_951/bias/m$Adam/batch_normalization_861/gamma/m#Adam/batch_normalization_861/beta/mAdam/dense_952/kernel/mAdam/dense_952/bias/m$Adam/batch_normalization_862/gamma/m#Adam/batch_normalization_862/beta/mAdam/dense_953/kernel/mAdam/dense_953/bias/m$Adam/batch_normalization_863/gamma/m#Adam/batch_normalization_863/beta/mAdam/dense_954/kernel/mAdam/dense_954/bias/m$Adam/batch_normalization_864/gamma/m#Adam/batch_normalization_864/beta/mAdam/dense_955/kernel/mAdam/dense_955/bias/m$Adam/batch_normalization_865/gamma/m#Adam/batch_normalization_865/beta/mAdam/dense_956/kernel/mAdam/dense_956/bias/mAdam/dense_945/kernel/vAdam/dense_945/bias/v$Adam/batch_normalization_855/gamma/v#Adam/batch_normalization_855/beta/vAdam/dense_946/kernel/vAdam/dense_946/bias/v$Adam/batch_normalization_856/gamma/v#Adam/batch_normalization_856/beta/vAdam/dense_947/kernel/vAdam/dense_947/bias/v$Adam/batch_normalization_857/gamma/v#Adam/batch_normalization_857/beta/vAdam/dense_948/kernel/vAdam/dense_948/bias/v$Adam/batch_normalization_858/gamma/v#Adam/batch_normalization_858/beta/vAdam/dense_949/kernel/vAdam/dense_949/bias/v$Adam/batch_normalization_859/gamma/v#Adam/batch_normalization_859/beta/vAdam/dense_950/kernel/vAdam/dense_950/bias/v$Adam/batch_normalization_860/gamma/v#Adam/batch_normalization_860/beta/vAdam/dense_951/kernel/vAdam/dense_951/bias/v$Adam/batch_normalization_861/gamma/v#Adam/batch_normalization_861/beta/vAdam/dense_952/kernel/vAdam/dense_952/bias/v$Adam/batch_normalization_862/gamma/v#Adam/batch_normalization_862/beta/vAdam/dense_953/kernel/vAdam/dense_953/bias/v$Adam/batch_normalization_863/gamma/v#Adam/batch_normalization_863/beta/vAdam/dense_954/kernel/vAdam/dense_954/bias/v$Adam/batch_normalization_864/gamma/v#Adam/batch_normalization_864/beta/vAdam/dense_955/kernel/vAdam/dense_955/bias/v$Adam/batch_normalization_865/gamma/v#Adam/batch_normalization_865/beta/vAdam/dense_956/kernel/vAdam/dense_956/bias/v*¸
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
"__inference__traced_restore_841867*
¥´

I__inference_sequential_90_layer_call_and_return_conditional_losses_837768

inputs
normalization_90_sub_y
normalization_90_sqrt_x"
dense_945_837597:h
dense_945_837599:h,
batch_normalization_855_837602:h,
batch_normalization_855_837604:h,
batch_normalization_855_837606:h,
batch_normalization_855_837608:h"
dense_946_837612:hh
dense_946_837614:h,
batch_normalization_856_837617:h,
batch_normalization_856_837619:h,
batch_normalization_856_837621:h,
batch_normalization_856_837623:h"
dense_947_837627:hh
dense_947_837629:h,
batch_normalization_857_837632:h,
batch_normalization_857_837634:h,
batch_normalization_857_837636:h,
batch_normalization_857_837638:h"
dense_948_837642:hh
dense_948_837644:h,
batch_normalization_858_837647:h,
batch_normalization_858_837649:h,
batch_normalization_858_837651:h,
batch_normalization_858_837653:h"
dense_949_837657:hh
dense_949_837659:h,
batch_normalization_859_837662:h,
batch_normalization_859_837664:h,
batch_normalization_859_837666:h,
batch_normalization_859_837668:h"
dense_950_837672:h/
dense_950_837674:/,
batch_normalization_860_837677:/,
batch_normalization_860_837679:/,
batch_normalization_860_837681:/,
batch_normalization_860_837683:/"
dense_951_837687://
dense_951_837689:/,
batch_normalization_861_837692:/,
batch_normalization_861_837694:/,
batch_normalization_861_837696:/,
batch_normalization_861_837698:/"
dense_952_837702:/
dense_952_837704:,
batch_normalization_862_837707:,
batch_normalization_862_837709:,
batch_normalization_862_837711:,
batch_normalization_862_837713:"
dense_953_837717:
dense_953_837719:,
batch_normalization_863_837722:,
batch_normalization_863_837724:,
batch_normalization_863_837726:,
batch_normalization_863_837728:"
dense_954_837732:
dense_954_837734:,
batch_normalization_864_837737:,
batch_normalization_864_837739:,
batch_normalization_864_837741:,
batch_normalization_864_837743:"
dense_955_837747:
dense_955_837749:,
batch_normalization_865_837752:,
batch_normalization_865_837754:,
batch_normalization_865_837756:,
batch_normalization_865_837758:"
dense_956_837762:
dense_956_837764:
identity¢/batch_normalization_855/StatefulPartitionedCall¢/batch_normalization_856/StatefulPartitionedCall¢/batch_normalization_857/StatefulPartitionedCall¢/batch_normalization_858/StatefulPartitionedCall¢/batch_normalization_859/StatefulPartitionedCall¢/batch_normalization_860/StatefulPartitionedCall¢/batch_normalization_861/StatefulPartitionedCall¢/batch_normalization_862/StatefulPartitionedCall¢/batch_normalization_863/StatefulPartitionedCall¢/batch_normalization_864/StatefulPartitionedCall¢/batch_normalization_865/StatefulPartitionedCall¢!dense_945/StatefulPartitionedCall¢!dense_946/StatefulPartitionedCall¢!dense_947/StatefulPartitionedCall¢!dense_948/StatefulPartitionedCall¢!dense_949/StatefulPartitionedCall¢!dense_950/StatefulPartitionedCall¢!dense_951/StatefulPartitionedCall¢!dense_952/StatefulPartitionedCall¢!dense_953/StatefulPartitionedCall¢!dense_954/StatefulPartitionedCall¢!dense_955/StatefulPartitionedCall¢!dense_956/StatefulPartitionedCallm
normalization_90/subSubinputsnormalization_90_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_90/SqrtSqrtnormalization_90_sqrt_x*
T0*
_output_shapes

:_
normalization_90/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_90/MaximumMaximumnormalization_90/Sqrt:y:0#normalization_90/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_90/truedivRealDivnormalization_90/sub:z:0normalization_90/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_945/StatefulPartitionedCallStatefulPartitionedCallnormalization_90/truediv:z:0dense_945_837597dense_945_837599*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_945_layer_call_and_return_conditional_losses_836752
/batch_normalization_855/StatefulPartitionedCallStatefulPartitionedCall*dense_945/StatefulPartitionedCall:output:0batch_normalization_855_837602batch_normalization_855_837604batch_normalization_855_837606batch_normalization_855_837608*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_835897ø
leaky_re_lu_855/PartitionedCallPartitionedCall8batch_normalization_855/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_855_layer_call_and_return_conditional_losses_836772
!dense_946/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_855/PartitionedCall:output:0dense_946_837612dense_946_837614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_946_layer_call_and_return_conditional_losses_836784
/batch_normalization_856/StatefulPartitionedCallStatefulPartitionedCall*dense_946/StatefulPartitionedCall:output:0batch_normalization_856_837617batch_normalization_856_837619batch_normalization_856_837621batch_normalization_856_837623*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_835979ø
leaky_re_lu_856/PartitionedCallPartitionedCall8batch_normalization_856/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_856_layer_call_and_return_conditional_losses_836804
!dense_947/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_856/PartitionedCall:output:0dense_947_837627dense_947_837629*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_947_layer_call_and_return_conditional_losses_836816
/batch_normalization_857/StatefulPartitionedCallStatefulPartitionedCall*dense_947/StatefulPartitionedCall:output:0batch_normalization_857_837632batch_normalization_857_837634batch_normalization_857_837636batch_normalization_857_837638*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_836061ø
leaky_re_lu_857/PartitionedCallPartitionedCall8batch_normalization_857/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_857_layer_call_and_return_conditional_losses_836836
!dense_948/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_857/PartitionedCall:output:0dense_948_837642dense_948_837644*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_948_layer_call_and_return_conditional_losses_836848
/batch_normalization_858/StatefulPartitionedCallStatefulPartitionedCall*dense_948/StatefulPartitionedCall:output:0batch_normalization_858_837647batch_normalization_858_837649batch_normalization_858_837651batch_normalization_858_837653*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_836143ø
leaky_re_lu_858/PartitionedCallPartitionedCall8batch_normalization_858/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_858_layer_call_and_return_conditional_losses_836868
!dense_949/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_858/PartitionedCall:output:0dense_949_837657dense_949_837659*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_949_layer_call_and_return_conditional_losses_836880
/batch_normalization_859/StatefulPartitionedCallStatefulPartitionedCall*dense_949/StatefulPartitionedCall:output:0batch_normalization_859_837662batch_normalization_859_837664batch_normalization_859_837666batch_normalization_859_837668*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_836225ø
leaky_re_lu_859/PartitionedCallPartitionedCall8batch_normalization_859/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_859_layer_call_and_return_conditional_losses_836900
!dense_950/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_859/PartitionedCall:output:0dense_950_837672dense_950_837674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_950_layer_call_and_return_conditional_losses_836912
/batch_normalization_860/StatefulPartitionedCallStatefulPartitionedCall*dense_950/StatefulPartitionedCall:output:0batch_normalization_860_837677batch_normalization_860_837679batch_normalization_860_837681batch_normalization_860_837683*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_836307ø
leaky_re_lu_860/PartitionedCallPartitionedCall8batch_normalization_860/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_860_layer_call_and_return_conditional_losses_836932
!dense_951/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_860/PartitionedCall:output:0dense_951_837687dense_951_837689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_951_layer_call_and_return_conditional_losses_836944
/batch_normalization_861/StatefulPartitionedCallStatefulPartitionedCall*dense_951/StatefulPartitionedCall:output:0batch_normalization_861_837692batch_normalization_861_837694batch_normalization_861_837696batch_normalization_861_837698*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_836389ø
leaky_re_lu_861/PartitionedCallPartitionedCall8batch_normalization_861/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_861_layer_call_and_return_conditional_losses_836964
!dense_952/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_861/PartitionedCall:output:0dense_952_837702dense_952_837704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_952_layer_call_and_return_conditional_losses_836976
/batch_normalization_862/StatefulPartitionedCallStatefulPartitionedCall*dense_952/StatefulPartitionedCall:output:0batch_normalization_862_837707batch_normalization_862_837709batch_normalization_862_837711batch_normalization_862_837713*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_836471ø
leaky_re_lu_862/PartitionedCallPartitionedCall8batch_normalization_862/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_862_layer_call_and_return_conditional_losses_836996
!dense_953/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_862/PartitionedCall:output:0dense_953_837717dense_953_837719*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_953_layer_call_and_return_conditional_losses_837008
/batch_normalization_863/StatefulPartitionedCallStatefulPartitionedCall*dense_953/StatefulPartitionedCall:output:0batch_normalization_863_837722batch_normalization_863_837724batch_normalization_863_837726batch_normalization_863_837728*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_836553ø
leaky_re_lu_863/PartitionedCallPartitionedCall8batch_normalization_863/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_863_layer_call_and_return_conditional_losses_837028
!dense_954/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_863/PartitionedCall:output:0dense_954_837732dense_954_837734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_954_layer_call_and_return_conditional_losses_837040
/batch_normalization_864/StatefulPartitionedCallStatefulPartitionedCall*dense_954/StatefulPartitionedCall:output:0batch_normalization_864_837737batch_normalization_864_837739batch_normalization_864_837741batch_normalization_864_837743*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_836635ø
leaky_re_lu_864/PartitionedCallPartitionedCall8batch_normalization_864/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_864_layer_call_and_return_conditional_losses_837060
!dense_955/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_864/PartitionedCall:output:0dense_955_837747dense_955_837749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_955_layer_call_and_return_conditional_losses_837072
/batch_normalization_865/StatefulPartitionedCallStatefulPartitionedCall*dense_955/StatefulPartitionedCall:output:0batch_normalization_865_837752batch_normalization_865_837754batch_normalization_865_837756batch_normalization_865_837758*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_836717ø
leaky_re_lu_865/PartitionedCallPartitionedCall8batch_normalization_865/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_837092
!dense_956/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_865/PartitionedCall:output:0dense_956_837762dense_956_837764*
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
E__inference_dense_956_layer_call_and_return_conditional_losses_837104y
IdentityIdentity*dense_956/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_855/StatefulPartitionedCall0^batch_normalization_856/StatefulPartitionedCall0^batch_normalization_857/StatefulPartitionedCall0^batch_normalization_858/StatefulPartitionedCall0^batch_normalization_859/StatefulPartitionedCall0^batch_normalization_860/StatefulPartitionedCall0^batch_normalization_861/StatefulPartitionedCall0^batch_normalization_862/StatefulPartitionedCall0^batch_normalization_863/StatefulPartitionedCall0^batch_normalization_864/StatefulPartitionedCall0^batch_normalization_865/StatefulPartitionedCall"^dense_945/StatefulPartitionedCall"^dense_946/StatefulPartitionedCall"^dense_947/StatefulPartitionedCall"^dense_948/StatefulPartitionedCall"^dense_949/StatefulPartitionedCall"^dense_950/StatefulPartitionedCall"^dense_951/StatefulPartitionedCall"^dense_952/StatefulPartitionedCall"^dense_953/StatefulPartitionedCall"^dense_954/StatefulPartitionedCall"^dense_955/StatefulPartitionedCall"^dense_956/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_855/StatefulPartitionedCall/batch_normalization_855/StatefulPartitionedCall2b
/batch_normalization_856/StatefulPartitionedCall/batch_normalization_856/StatefulPartitionedCall2b
/batch_normalization_857/StatefulPartitionedCall/batch_normalization_857/StatefulPartitionedCall2b
/batch_normalization_858/StatefulPartitionedCall/batch_normalization_858/StatefulPartitionedCall2b
/batch_normalization_859/StatefulPartitionedCall/batch_normalization_859/StatefulPartitionedCall2b
/batch_normalization_860/StatefulPartitionedCall/batch_normalization_860/StatefulPartitionedCall2b
/batch_normalization_861/StatefulPartitionedCall/batch_normalization_861/StatefulPartitionedCall2b
/batch_normalization_862/StatefulPartitionedCall/batch_normalization_862/StatefulPartitionedCall2b
/batch_normalization_863/StatefulPartitionedCall/batch_normalization_863/StatefulPartitionedCall2b
/batch_normalization_864/StatefulPartitionedCall/batch_normalization_864/StatefulPartitionedCall2b
/batch_normalization_865/StatefulPartitionedCall/batch_normalization_865/StatefulPartitionedCall2F
!dense_945/StatefulPartitionedCall!dense_945/StatefulPartitionedCall2F
!dense_946/StatefulPartitionedCall!dense_946/StatefulPartitionedCall2F
!dense_947/StatefulPartitionedCall!dense_947/StatefulPartitionedCall2F
!dense_948/StatefulPartitionedCall!dense_948/StatefulPartitionedCall2F
!dense_949/StatefulPartitionedCall!dense_949/StatefulPartitionedCall2F
!dense_950/StatefulPartitionedCall!dense_950/StatefulPartitionedCall2F
!dense_951/StatefulPartitionedCall!dense_951/StatefulPartitionedCall2F
!dense_952/StatefulPartitionedCall!dense_952/StatefulPartitionedCall2F
!dense_953/StatefulPartitionedCall!dense_953/StatefulPartitionedCall2F
!dense_954/StatefulPartitionedCall!dense_954/StatefulPartitionedCall2F
!dense_955/StatefulPartitionedCall!dense_955/StatefulPartitionedCall2F
!dense_956/StatefulPartitionedCall!dense_956/StatefulPartitionedCall:O K
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
E__inference_dense_947_layer_call_and_return_conditional_losses_839837

inputs0
matmul_readvariableop_resource:hh-
biasadd_readvariableop_resource:h
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:hh*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:h*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
È	
ö
E__inference_dense_949_layer_call_and_return_conditional_losses_840055

inputs0
matmul_readvariableop_resource:hh-
biasadd_readvariableop_resource:h
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:hh*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:h*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ä

*__inference_dense_954_layer_call_fn_840590

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_954_layer_call_and_return_conditional_losses_837040o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¢
.__inference_sequential_90_layer_call_fn_837254
normalization_90_input
unknown
	unknown_0
	unknown_1:h
	unknown_2:h
	unknown_3:h
	unknown_4:h
	unknown_5:h
	unknown_6:h
	unknown_7:hh
	unknown_8:h
	unknown_9:h

unknown_10:h

unknown_11:h

unknown_12:h

unknown_13:hh

unknown_14:h

unknown_15:h

unknown_16:h

unknown_17:h

unknown_18:h

unknown_19:hh

unknown_20:h

unknown_21:h

unknown_22:h

unknown_23:h

unknown_24:h

unknown_25:hh

unknown_26:h

unknown_27:h

unknown_28:h

unknown_29:h

unknown_30:h

unknown_31:h/

unknown_32:/

unknown_33:/

unknown_34:/

unknown_35:/

unknown_36:/

unknown_37://

unknown_38:/

unknown_39:/

unknown_40:/

unknown_41:/

unknown_42:/

unknown_43:/

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:

unknown_61:

unknown_62:

unknown_63:

unknown_64:

unknown_65:

unknown_66:

unknown_67:

unknown_68:
identity¢StatefulPartitionedCall

StatefulPartitionedCallStatefulPartitionedCallnormalization_90_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_90_layer_call_and_return_conditional_losses_837111o
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
_user_specified_namenormalization_90_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ð
²
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_836506

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_856_layer_call_fn_839754

inputs
unknown:h
	unknown_0:h
	unknown_1:h
	unknown_2:h
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_835979o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ä

*__inference_dense_951_layer_call_fn_840263

inputs
unknown://
	unknown_0:/
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_951_layer_call_and_return_conditional_losses_836944o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_836342

inputs/
!batchnorm_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource:/1
#batchnorm_readvariableop_1_resource:/1
#batchnorm_readvariableop_2_resource:/
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:/z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_861_layer_call_and_return_conditional_losses_836964

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_836061

inputs5
'assignmovingavg_readvariableop_resource:h7
)assignmovingavg_1_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h/
!batchnorm_readvariableop_resource:h
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:h
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:h*
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
:h*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:hx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h¬
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
:h*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:h~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h´
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:hv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
È	
ö
E__inference_dense_953_layer_call_and_return_conditional_losses_840491

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_955_layer_call_and_return_conditional_losses_840709

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_949_layer_call_and_return_conditional_losses_836880

inputs0
matmul_readvariableop_resource:hh-
biasadd_readvariableop_resource:h
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:hh*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:h*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_861_layer_call_and_return_conditional_losses_840363

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_864_layer_call_fn_840613

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_836588o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_856_layer_call_and_return_conditional_losses_836804

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_839699

inputs5
'assignmovingavg_readvariableop_resource:h7
)assignmovingavg_1_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h/
!batchnorm_readvariableop_resource:h
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:h
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:h*
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
:h*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:hx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h¬
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
:h*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:h~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h´
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:hv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_839665

inputs/
!batchnorm_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h1
#batchnorm_readvariableop_1_resource:h1
#batchnorm_readvariableop_2_resource:h
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:hz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
î'
Ò
__inference_adapt_step_839600
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
%
ì
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_839808

inputs5
'assignmovingavg_readvariableop_resource:h7
)assignmovingavg_1_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h/
!batchnorm_readvariableop_resource:h
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:h
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:h*
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
:h*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:hx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h¬
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
:h*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:h~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h´
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:hv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_840755

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_836307

inputs5
'assignmovingavg_readvariableop_resource:/7
)assignmovingavg_1_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource://
!batchnorm_readvariableop_resource:/
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:/
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:/*
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
:/*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:/x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:/¬
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
:/*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:/~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:/´
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:/v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
Ä

*__inference_dense_946_layer_call_fn_839718

inputs
unknown:hh
	unknown_0:h
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_946_layer_call_and_return_conditional_losses_836784o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_836260

inputs/
!batchnorm_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource:/1
#batchnorm_readvariableop_1_resource:/1
#batchnorm_readvariableop_2_resource:/
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:/z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_840571

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_953_layer_call_and_return_conditional_losses_837008

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_950_layer_call_and_return_conditional_losses_840164

inputs0
matmul_readvariableop_resource:h/-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:h/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_861_layer_call_fn_840358

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
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_861_layer_call_and_return_conditional_losses_836964`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_855_layer_call_fn_839645

inputs
unknown:h
	unknown_0:h
	unknown_1:h
	unknown_2:h
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_835897o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_836670

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_863_layer_call_fn_840517

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_836553o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_840210

inputs/
!batchnorm_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource:/1
#batchnorm_readvariableop_1_resource:/1
#batchnorm_readvariableop_2_resource:/
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:/z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_863_layer_call_and_return_conditional_losses_840581

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_951_layer_call_and_return_conditional_losses_840273

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
È	
ö
E__inference_dense_947_layer_call_and_return_conditional_losses_836816

inputs0
matmul_readvariableop_resource:hh-
biasadd_readvariableop_resource:h
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:hh*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:h*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
È	
ö
E__inference_dense_952_layer_call_and_return_conditional_losses_836976

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_858_layer_call_fn_839972

inputs
unknown:h
	unknown_0:h
	unknown_1:h
	unknown_2:h
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_836143o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
È	
ö
E__inference_dense_948_layer_call_and_return_conditional_losses_839946

inputs0
matmul_readvariableop_resource:hh-
biasadd_readvariableop_resource:h
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:hh*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:h*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_860_layer_call_and_return_conditional_losses_840254

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
È	
ö
E__inference_dense_954_layer_call_and_return_conditional_losses_837040

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_864_layer_call_fn_840626

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_836635o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
¢
.__inference_sequential_90_layer_call_fn_838056
normalization_90_input
unknown
	unknown_0
	unknown_1:h
	unknown_2:h
	unknown_3:h
	unknown_4:h
	unknown_5:h
	unknown_6:h
	unknown_7:hh
	unknown_8:h
	unknown_9:h

unknown_10:h

unknown_11:h

unknown_12:h

unknown_13:hh

unknown_14:h

unknown_15:h

unknown_16:h

unknown_17:h

unknown_18:h

unknown_19:hh

unknown_20:h

unknown_21:h

unknown_22:h

unknown_23:h

unknown_24:h

unknown_25:hh

unknown_26:h

unknown_27:h

unknown_28:h

unknown_29:h

unknown_30:h

unknown_31:h/

unknown_32:/

unknown_33:/

unknown_34:/

unknown_35:/

unknown_36:/

unknown_37://

unknown_38:/

unknown_39:/

unknown_40:/

unknown_41:/

unknown_42:/

unknown_43:/

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:

unknown_61:

unknown_62:

unknown_63:

unknown_64:

unknown_65:

unknown_66:

unknown_67:

unknown_68:
identity¢StatefulPartitionedCall

StatefulPartitionedCallStatefulPartitionedCallnormalization_90_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_90_layer_call_and_return_conditional_losses_837768o
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
_user_specified_namenormalization_90_input:$ 

_output_shapes

::$ 

_output_shapes

:
ª
Ó
8__inference_batch_normalization_859_layer_call_fn_840081

inputs
unknown:h
	unknown_0:h
	unknown_1:h
	unknown_2:h
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_836225o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ä

*__inference_dense_950_layer_call_fn_840154

inputs
unknown:h/
	unknown_0:/
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_950_layer_call_and_return_conditional_losses_836912o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_857_layer_call_fn_839863

inputs
unknown:h
	unknown_0:h
	unknown_1:h
	unknown_2:h
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_836061o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_857_layer_call_and_return_conditional_losses_836836

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
È	
ö
E__inference_dense_950_layer_call_and_return_conditional_losses_836912

inputs0
matmul_readvariableop_resource:h/-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:h/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_861_layer_call_fn_840286

inputs
unknown:/
	unknown_0:/
	unknown_1:/
	unknown_2:/
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_836342o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
ÀÂ
ÀO
__inference__traced_save_841350
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_945_kernel_read_readvariableop-
)savev2_dense_945_bias_read_readvariableop<
8savev2_batch_normalization_855_gamma_read_readvariableop;
7savev2_batch_normalization_855_beta_read_readvariableopB
>savev2_batch_normalization_855_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_855_moving_variance_read_readvariableop/
+savev2_dense_946_kernel_read_readvariableop-
)savev2_dense_946_bias_read_readvariableop<
8savev2_batch_normalization_856_gamma_read_readvariableop;
7savev2_batch_normalization_856_beta_read_readvariableopB
>savev2_batch_normalization_856_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_856_moving_variance_read_readvariableop/
+savev2_dense_947_kernel_read_readvariableop-
)savev2_dense_947_bias_read_readvariableop<
8savev2_batch_normalization_857_gamma_read_readvariableop;
7savev2_batch_normalization_857_beta_read_readvariableopB
>savev2_batch_normalization_857_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_857_moving_variance_read_readvariableop/
+savev2_dense_948_kernel_read_readvariableop-
)savev2_dense_948_bias_read_readvariableop<
8savev2_batch_normalization_858_gamma_read_readvariableop;
7savev2_batch_normalization_858_beta_read_readvariableopB
>savev2_batch_normalization_858_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_858_moving_variance_read_readvariableop/
+savev2_dense_949_kernel_read_readvariableop-
)savev2_dense_949_bias_read_readvariableop<
8savev2_batch_normalization_859_gamma_read_readvariableop;
7savev2_batch_normalization_859_beta_read_readvariableopB
>savev2_batch_normalization_859_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_859_moving_variance_read_readvariableop/
+savev2_dense_950_kernel_read_readvariableop-
)savev2_dense_950_bias_read_readvariableop<
8savev2_batch_normalization_860_gamma_read_readvariableop;
7savev2_batch_normalization_860_beta_read_readvariableopB
>savev2_batch_normalization_860_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_860_moving_variance_read_readvariableop/
+savev2_dense_951_kernel_read_readvariableop-
)savev2_dense_951_bias_read_readvariableop<
8savev2_batch_normalization_861_gamma_read_readvariableop;
7savev2_batch_normalization_861_beta_read_readvariableopB
>savev2_batch_normalization_861_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_861_moving_variance_read_readvariableop/
+savev2_dense_952_kernel_read_readvariableop-
)savev2_dense_952_bias_read_readvariableop<
8savev2_batch_normalization_862_gamma_read_readvariableop;
7savev2_batch_normalization_862_beta_read_readvariableopB
>savev2_batch_normalization_862_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_862_moving_variance_read_readvariableop/
+savev2_dense_953_kernel_read_readvariableop-
)savev2_dense_953_bias_read_readvariableop<
8savev2_batch_normalization_863_gamma_read_readvariableop;
7savev2_batch_normalization_863_beta_read_readvariableopB
>savev2_batch_normalization_863_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_863_moving_variance_read_readvariableop/
+savev2_dense_954_kernel_read_readvariableop-
)savev2_dense_954_bias_read_readvariableop<
8savev2_batch_normalization_864_gamma_read_readvariableop;
7savev2_batch_normalization_864_beta_read_readvariableopB
>savev2_batch_normalization_864_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_864_moving_variance_read_readvariableop/
+savev2_dense_955_kernel_read_readvariableop-
)savev2_dense_955_bias_read_readvariableop<
8savev2_batch_normalization_865_gamma_read_readvariableop;
7savev2_batch_normalization_865_beta_read_readvariableopB
>savev2_batch_normalization_865_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_865_moving_variance_read_readvariableop/
+savev2_dense_956_kernel_read_readvariableop-
)savev2_dense_956_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_945_kernel_m_read_readvariableop4
0savev2_adam_dense_945_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_855_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_855_beta_m_read_readvariableop6
2savev2_adam_dense_946_kernel_m_read_readvariableop4
0savev2_adam_dense_946_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_856_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_856_beta_m_read_readvariableop6
2savev2_adam_dense_947_kernel_m_read_readvariableop4
0savev2_adam_dense_947_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_857_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_857_beta_m_read_readvariableop6
2savev2_adam_dense_948_kernel_m_read_readvariableop4
0savev2_adam_dense_948_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_858_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_858_beta_m_read_readvariableop6
2savev2_adam_dense_949_kernel_m_read_readvariableop4
0savev2_adam_dense_949_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_859_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_859_beta_m_read_readvariableop6
2savev2_adam_dense_950_kernel_m_read_readvariableop4
0savev2_adam_dense_950_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_860_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_860_beta_m_read_readvariableop6
2savev2_adam_dense_951_kernel_m_read_readvariableop4
0savev2_adam_dense_951_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_861_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_861_beta_m_read_readvariableop6
2savev2_adam_dense_952_kernel_m_read_readvariableop4
0savev2_adam_dense_952_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_862_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_862_beta_m_read_readvariableop6
2savev2_adam_dense_953_kernel_m_read_readvariableop4
0savev2_adam_dense_953_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_863_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_863_beta_m_read_readvariableop6
2savev2_adam_dense_954_kernel_m_read_readvariableop4
0savev2_adam_dense_954_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_864_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_864_beta_m_read_readvariableop6
2savev2_adam_dense_955_kernel_m_read_readvariableop4
0savev2_adam_dense_955_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_865_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_865_beta_m_read_readvariableop6
2savev2_adam_dense_956_kernel_m_read_readvariableop4
0savev2_adam_dense_956_bias_m_read_readvariableop6
2savev2_adam_dense_945_kernel_v_read_readvariableop4
0savev2_adam_dense_945_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_855_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_855_beta_v_read_readvariableop6
2savev2_adam_dense_946_kernel_v_read_readvariableop4
0savev2_adam_dense_946_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_856_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_856_beta_v_read_readvariableop6
2savev2_adam_dense_947_kernel_v_read_readvariableop4
0savev2_adam_dense_947_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_857_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_857_beta_v_read_readvariableop6
2savev2_adam_dense_948_kernel_v_read_readvariableop4
0savev2_adam_dense_948_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_858_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_858_beta_v_read_readvariableop6
2savev2_adam_dense_949_kernel_v_read_readvariableop4
0savev2_adam_dense_949_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_859_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_859_beta_v_read_readvariableop6
2savev2_adam_dense_950_kernel_v_read_readvariableop4
0savev2_adam_dense_950_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_860_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_860_beta_v_read_readvariableop6
2savev2_adam_dense_951_kernel_v_read_readvariableop4
0savev2_adam_dense_951_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_861_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_861_beta_v_read_readvariableop6
2savev2_adam_dense_952_kernel_v_read_readvariableop4
0savev2_adam_dense_952_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_862_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_862_beta_v_read_readvariableop6
2savev2_adam_dense_953_kernel_v_read_readvariableop4
0savev2_adam_dense_953_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_863_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_863_beta_v_read_readvariableop6
2savev2_adam_dense_954_kernel_v_read_readvariableop4
0savev2_adam_dense_954_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_864_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_864_beta_v_read_readvariableop6
2savev2_adam_dense_955_kernel_v_read_readvariableop4
0savev2_adam_dense_955_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_865_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_865_beta_v_read_readvariableop6
2savev2_adam_dense_956_kernel_v_read_readvariableop4
0savev2_adam_dense_956_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_945_kernel_read_readvariableop)savev2_dense_945_bias_read_readvariableop8savev2_batch_normalization_855_gamma_read_readvariableop7savev2_batch_normalization_855_beta_read_readvariableop>savev2_batch_normalization_855_moving_mean_read_readvariableopBsavev2_batch_normalization_855_moving_variance_read_readvariableop+savev2_dense_946_kernel_read_readvariableop)savev2_dense_946_bias_read_readvariableop8savev2_batch_normalization_856_gamma_read_readvariableop7savev2_batch_normalization_856_beta_read_readvariableop>savev2_batch_normalization_856_moving_mean_read_readvariableopBsavev2_batch_normalization_856_moving_variance_read_readvariableop+savev2_dense_947_kernel_read_readvariableop)savev2_dense_947_bias_read_readvariableop8savev2_batch_normalization_857_gamma_read_readvariableop7savev2_batch_normalization_857_beta_read_readvariableop>savev2_batch_normalization_857_moving_mean_read_readvariableopBsavev2_batch_normalization_857_moving_variance_read_readvariableop+savev2_dense_948_kernel_read_readvariableop)savev2_dense_948_bias_read_readvariableop8savev2_batch_normalization_858_gamma_read_readvariableop7savev2_batch_normalization_858_beta_read_readvariableop>savev2_batch_normalization_858_moving_mean_read_readvariableopBsavev2_batch_normalization_858_moving_variance_read_readvariableop+savev2_dense_949_kernel_read_readvariableop)savev2_dense_949_bias_read_readvariableop8savev2_batch_normalization_859_gamma_read_readvariableop7savev2_batch_normalization_859_beta_read_readvariableop>savev2_batch_normalization_859_moving_mean_read_readvariableopBsavev2_batch_normalization_859_moving_variance_read_readvariableop+savev2_dense_950_kernel_read_readvariableop)savev2_dense_950_bias_read_readvariableop8savev2_batch_normalization_860_gamma_read_readvariableop7savev2_batch_normalization_860_beta_read_readvariableop>savev2_batch_normalization_860_moving_mean_read_readvariableopBsavev2_batch_normalization_860_moving_variance_read_readvariableop+savev2_dense_951_kernel_read_readvariableop)savev2_dense_951_bias_read_readvariableop8savev2_batch_normalization_861_gamma_read_readvariableop7savev2_batch_normalization_861_beta_read_readvariableop>savev2_batch_normalization_861_moving_mean_read_readvariableopBsavev2_batch_normalization_861_moving_variance_read_readvariableop+savev2_dense_952_kernel_read_readvariableop)savev2_dense_952_bias_read_readvariableop8savev2_batch_normalization_862_gamma_read_readvariableop7savev2_batch_normalization_862_beta_read_readvariableop>savev2_batch_normalization_862_moving_mean_read_readvariableopBsavev2_batch_normalization_862_moving_variance_read_readvariableop+savev2_dense_953_kernel_read_readvariableop)savev2_dense_953_bias_read_readvariableop8savev2_batch_normalization_863_gamma_read_readvariableop7savev2_batch_normalization_863_beta_read_readvariableop>savev2_batch_normalization_863_moving_mean_read_readvariableopBsavev2_batch_normalization_863_moving_variance_read_readvariableop+savev2_dense_954_kernel_read_readvariableop)savev2_dense_954_bias_read_readvariableop8savev2_batch_normalization_864_gamma_read_readvariableop7savev2_batch_normalization_864_beta_read_readvariableop>savev2_batch_normalization_864_moving_mean_read_readvariableopBsavev2_batch_normalization_864_moving_variance_read_readvariableop+savev2_dense_955_kernel_read_readvariableop)savev2_dense_955_bias_read_readvariableop8savev2_batch_normalization_865_gamma_read_readvariableop7savev2_batch_normalization_865_beta_read_readvariableop>savev2_batch_normalization_865_moving_mean_read_readvariableopBsavev2_batch_normalization_865_moving_variance_read_readvariableop+savev2_dense_956_kernel_read_readvariableop)savev2_dense_956_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_945_kernel_m_read_readvariableop0savev2_adam_dense_945_bias_m_read_readvariableop?savev2_adam_batch_normalization_855_gamma_m_read_readvariableop>savev2_adam_batch_normalization_855_beta_m_read_readvariableop2savev2_adam_dense_946_kernel_m_read_readvariableop0savev2_adam_dense_946_bias_m_read_readvariableop?savev2_adam_batch_normalization_856_gamma_m_read_readvariableop>savev2_adam_batch_normalization_856_beta_m_read_readvariableop2savev2_adam_dense_947_kernel_m_read_readvariableop0savev2_adam_dense_947_bias_m_read_readvariableop?savev2_adam_batch_normalization_857_gamma_m_read_readvariableop>savev2_adam_batch_normalization_857_beta_m_read_readvariableop2savev2_adam_dense_948_kernel_m_read_readvariableop0savev2_adam_dense_948_bias_m_read_readvariableop?savev2_adam_batch_normalization_858_gamma_m_read_readvariableop>savev2_adam_batch_normalization_858_beta_m_read_readvariableop2savev2_adam_dense_949_kernel_m_read_readvariableop0savev2_adam_dense_949_bias_m_read_readvariableop?savev2_adam_batch_normalization_859_gamma_m_read_readvariableop>savev2_adam_batch_normalization_859_beta_m_read_readvariableop2savev2_adam_dense_950_kernel_m_read_readvariableop0savev2_adam_dense_950_bias_m_read_readvariableop?savev2_adam_batch_normalization_860_gamma_m_read_readvariableop>savev2_adam_batch_normalization_860_beta_m_read_readvariableop2savev2_adam_dense_951_kernel_m_read_readvariableop0savev2_adam_dense_951_bias_m_read_readvariableop?savev2_adam_batch_normalization_861_gamma_m_read_readvariableop>savev2_adam_batch_normalization_861_beta_m_read_readvariableop2savev2_adam_dense_952_kernel_m_read_readvariableop0savev2_adam_dense_952_bias_m_read_readvariableop?savev2_adam_batch_normalization_862_gamma_m_read_readvariableop>savev2_adam_batch_normalization_862_beta_m_read_readvariableop2savev2_adam_dense_953_kernel_m_read_readvariableop0savev2_adam_dense_953_bias_m_read_readvariableop?savev2_adam_batch_normalization_863_gamma_m_read_readvariableop>savev2_adam_batch_normalization_863_beta_m_read_readvariableop2savev2_adam_dense_954_kernel_m_read_readvariableop0savev2_adam_dense_954_bias_m_read_readvariableop?savev2_adam_batch_normalization_864_gamma_m_read_readvariableop>savev2_adam_batch_normalization_864_beta_m_read_readvariableop2savev2_adam_dense_955_kernel_m_read_readvariableop0savev2_adam_dense_955_bias_m_read_readvariableop?savev2_adam_batch_normalization_865_gamma_m_read_readvariableop>savev2_adam_batch_normalization_865_beta_m_read_readvariableop2savev2_adam_dense_956_kernel_m_read_readvariableop0savev2_adam_dense_956_bias_m_read_readvariableop2savev2_adam_dense_945_kernel_v_read_readvariableop0savev2_adam_dense_945_bias_v_read_readvariableop?savev2_adam_batch_normalization_855_gamma_v_read_readvariableop>savev2_adam_batch_normalization_855_beta_v_read_readvariableop2savev2_adam_dense_946_kernel_v_read_readvariableop0savev2_adam_dense_946_bias_v_read_readvariableop?savev2_adam_batch_normalization_856_gamma_v_read_readvariableop>savev2_adam_batch_normalization_856_beta_v_read_readvariableop2savev2_adam_dense_947_kernel_v_read_readvariableop0savev2_adam_dense_947_bias_v_read_readvariableop?savev2_adam_batch_normalization_857_gamma_v_read_readvariableop>savev2_adam_batch_normalization_857_beta_v_read_readvariableop2savev2_adam_dense_948_kernel_v_read_readvariableop0savev2_adam_dense_948_bias_v_read_readvariableop?savev2_adam_batch_normalization_858_gamma_v_read_readvariableop>savev2_adam_batch_normalization_858_beta_v_read_readvariableop2savev2_adam_dense_949_kernel_v_read_readvariableop0savev2_adam_dense_949_bias_v_read_readvariableop?savev2_adam_batch_normalization_859_gamma_v_read_readvariableop>savev2_adam_batch_normalization_859_beta_v_read_readvariableop2savev2_adam_dense_950_kernel_v_read_readvariableop0savev2_adam_dense_950_bias_v_read_readvariableop?savev2_adam_batch_normalization_860_gamma_v_read_readvariableop>savev2_adam_batch_normalization_860_beta_v_read_readvariableop2savev2_adam_dense_951_kernel_v_read_readvariableop0savev2_adam_dense_951_bias_v_read_readvariableop?savev2_adam_batch_normalization_861_gamma_v_read_readvariableop>savev2_adam_batch_normalization_861_beta_v_read_readvariableop2savev2_adam_dense_952_kernel_v_read_readvariableop0savev2_adam_dense_952_bias_v_read_readvariableop?savev2_adam_batch_normalization_862_gamma_v_read_readvariableop>savev2_adam_batch_normalization_862_beta_v_read_readvariableop2savev2_adam_dense_953_kernel_v_read_readvariableop0savev2_adam_dense_953_bias_v_read_readvariableop?savev2_adam_batch_normalization_863_gamma_v_read_readvariableop>savev2_adam_batch_normalization_863_beta_v_read_readvariableop2savev2_adam_dense_954_kernel_v_read_readvariableop0savev2_adam_dense_954_bias_v_read_readvariableop?savev2_adam_batch_normalization_864_gamma_v_read_readvariableop>savev2_adam_batch_normalization_864_beta_v_read_readvariableop2savev2_adam_dense_955_kernel_v_read_readvariableop0savev2_adam_dense_955_bias_v_read_readvariableop?savev2_adam_batch_normalization_865_gamma_v_read_readvariableop>savev2_adam_batch_normalization_865_beta_v_read_readvariableop2savev2_adam_dense_956_kernel_v_read_readvariableop0savev2_adam_dense_956_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
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
î: ::: :h:h:h:h:h:h:hh:h:h:h:h:h:hh:h:h:h:h:h:hh:h:h:h:h:h:hh:h:h:h:h:h:h/:/:/:/:/:/://:/:/:/:/:/:/:::::::::::::::::::::::::: : : : : : :h:h:h:h:hh:h:h:h:hh:h:h:h:hh:h:h:h:hh:h:h:h:h/:/:/:/://:/:/:/:/::::::::::::::::::h:h:h:h:hh:h:h:h:hh:h:h:h:hh:h:h:h:hh:h:h:h:h/:/:/:/://:/:/:/:/:::::::::::::::::: 2(
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

:h: 

_output_shapes
:h: 

_output_shapes
:h: 

_output_shapes
:h: 

_output_shapes
:h: 	

_output_shapes
:h:$
 

_output_shapes

:hh: 

_output_shapes
:h: 

_output_shapes
:h: 

_output_shapes
:h: 

_output_shapes
:h: 

_output_shapes
:h:$ 

_output_shapes

:hh: 

_output_shapes
:h: 

_output_shapes
:h: 

_output_shapes
:h: 

_output_shapes
:h: 

_output_shapes
:h:$ 

_output_shapes

:hh: 

_output_shapes
:h: 

_output_shapes
:h: 

_output_shapes
:h: 

_output_shapes
:h: 

_output_shapes
:h:$ 

_output_shapes

:hh: 

_output_shapes
:h: 

_output_shapes
:h: 

_output_shapes
:h:  

_output_shapes
:h: !

_output_shapes
:h:$" 

_output_shapes

:h/: #

_output_shapes
:/: $

_output_shapes
:/: %

_output_shapes
:/: &

_output_shapes
:/: '

_output_shapes
:/:$( 

_output_shapes

://: )

_output_shapes
:/: *

_output_shapes
:/: +

_output_shapes
:/: ,

_output_shapes
:/: -

_output_shapes
:/:$. 

_output_shapes

:/: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
::$4 

_output_shapes

:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
::$: 

_output_shapes

:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
::$@ 

_output_shapes

:: A

_output_shapes
:: B

_output_shapes
:: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
::$F 

_output_shapes

:: G
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

:h: O

_output_shapes
:h: P

_output_shapes
:h: Q

_output_shapes
:h:$R 

_output_shapes

:hh: S

_output_shapes
:h: T

_output_shapes
:h: U

_output_shapes
:h:$V 

_output_shapes

:hh: W

_output_shapes
:h: X

_output_shapes
:h: Y

_output_shapes
:h:$Z 

_output_shapes

:hh: [

_output_shapes
:h: \

_output_shapes
:h: ]

_output_shapes
:h:$^ 

_output_shapes

:hh: _

_output_shapes
:h: `

_output_shapes
:h: a

_output_shapes
:h:$b 

_output_shapes

:h/: c

_output_shapes
:/: d

_output_shapes
:/: e

_output_shapes
:/:$f 

_output_shapes

://: g

_output_shapes
:/: h

_output_shapes
:/: i

_output_shapes
:/:$j 

_output_shapes

:/: k

_output_shapes
:: l

_output_shapes
:: m

_output_shapes
::$n 

_output_shapes

:: o

_output_shapes
:: p

_output_shapes
:: q

_output_shapes
::$r 

_output_shapes

:: s

_output_shapes
:: t

_output_shapes
:: u

_output_shapes
::$v 

_output_shapes

:: w

_output_shapes
:: x

_output_shapes
:: y

_output_shapes
::$z 

_output_shapes

:: {

_output_shapes
::$| 

_output_shapes

:h: }

_output_shapes
:h: ~

_output_shapes
:h: 

_output_shapes
:h:% 

_output_shapes

:hh:!

_output_shapes
:h:!

_output_shapes
:h:!

_output_shapes
:h:% 

_output_shapes

:hh:!

_output_shapes
:h:!

_output_shapes
:h:!

_output_shapes
:h:% 

_output_shapes

:hh:!

_output_shapes
:h:!

_output_shapes
:h:!

_output_shapes
:h:% 

_output_shapes

:hh:!

_output_shapes
:h:!

_output_shapes
:h:!

_output_shapes
:h:% 

_output_shapes

:h/:!

_output_shapes
:/:!

_output_shapes
:/:!

_output_shapes
:/:% 

_output_shapes

://:!

_output_shapes
:/:!

_output_shapes
:/:!

_output_shapes
:/:% 

_output_shapes

:/:!

_output_shapes
::!

_output_shapes
::!

_output_shapes
::% 

_output_shapes

::!

_output_shapes
::!

_output_shapes
::!

_output_shapes
::%  

_output_shapes

::!¡

_output_shapes
::!¢

_output_shapes
::!£

_output_shapes
::%¤ 

_output_shapes

::!¥

_output_shapes
::!¦

_output_shapes
::!§

_output_shapes
::%¨ 

_output_shapes

::!©

_output_shapes
::ª

_output_shapes
: 
ª
Ó
8__inference_batch_normalization_860_layer_call_fn_840190

inputs
unknown:/
	unknown_0:/
	unknown_1:/
	unknown_2:/
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_836307o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_858_layer_call_and_return_conditional_losses_836868

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_836553

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_856_layer_call_fn_839813

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
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_856_layer_call_and_return_conditional_losses_836804`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_836717

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_864_layer_call_fn_840685

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_864_layer_call_and_return_conditional_losses_837060`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_840244

inputs5
'assignmovingavg_readvariableop_resource:/7
)assignmovingavg_1_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource://
!batchnorm_readvariableop_resource:/
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:/
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:/*
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
:/*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:/x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:/¬
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
:/*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:/~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:/´
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:/v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_857_layer_call_fn_839850

inputs
unknown:h
	unknown_0:h
	unknown_1:h
	unknown_2:h
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_836014o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_860_layer_call_and_return_conditional_losses_836932

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_835897

inputs5
'assignmovingavg_readvariableop_resource:h7
)assignmovingavg_1_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h/
!batchnorm_readvariableop_resource:h
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:h
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:h*
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
:h*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:hx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h¬
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
:h*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:h~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h´
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:hv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_862_layer_call_and_return_conditional_losses_836996

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_952_layer_call_fn_840372

inputs
unknown:/
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_952_layer_call_and_return_conditional_losses_836976o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_836225

inputs5
'assignmovingavg_readvariableop_resource:h7
)assignmovingavg_1_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h/
!batchnorm_readvariableop_resource:h
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:h
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:h*
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
:h*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:hx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h¬
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
:h*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:h~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h´
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:hv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_840026

inputs5
'assignmovingavg_readvariableop_resource:h7
)assignmovingavg_1_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h/
!batchnorm_readvariableop_resource:h
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:h
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:h*
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
:h*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:hx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h¬
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
:h*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:h~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h´
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:hv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Óß
ÍM
!__inference__wrapped_model_835826
normalization_90_input(
$sequential_90_normalization_90_sub_y)
%sequential_90_normalization_90_sqrt_xH
6sequential_90_dense_945_matmul_readvariableop_resource:hE
7sequential_90_dense_945_biasadd_readvariableop_resource:hU
Gsequential_90_batch_normalization_855_batchnorm_readvariableop_resource:hY
Ksequential_90_batch_normalization_855_batchnorm_mul_readvariableop_resource:hW
Isequential_90_batch_normalization_855_batchnorm_readvariableop_1_resource:hW
Isequential_90_batch_normalization_855_batchnorm_readvariableop_2_resource:hH
6sequential_90_dense_946_matmul_readvariableop_resource:hhE
7sequential_90_dense_946_biasadd_readvariableop_resource:hU
Gsequential_90_batch_normalization_856_batchnorm_readvariableop_resource:hY
Ksequential_90_batch_normalization_856_batchnorm_mul_readvariableop_resource:hW
Isequential_90_batch_normalization_856_batchnorm_readvariableop_1_resource:hW
Isequential_90_batch_normalization_856_batchnorm_readvariableop_2_resource:hH
6sequential_90_dense_947_matmul_readvariableop_resource:hhE
7sequential_90_dense_947_biasadd_readvariableop_resource:hU
Gsequential_90_batch_normalization_857_batchnorm_readvariableop_resource:hY
Ksequential_90_batch_normalization_857_batchnorm_mul_readvariableop_resource:hW
Isequential_90_batch_normalization_857_batchnorm_readvariableop_1_resource:hW
Isequential_90_batch_normalization_857_batchnorm_readvariableop_2_resource:hH
6sequential_90_dense_948_matmul_readvariableop_resource:hhE
7sequential_90_dense_948_biasadd_readvariableop_resource:hU
Gsequential_90_batch_normalization_858_batchnorm_readvariableop_resource:hY
Ksequential_90_batch_normalization_858_batchnorm_mul_readvariableop_resource:hW
Isequential_90_batch_normalization_858_batchnorm_readvariableop_1_resource:hW
Isequential_90_batch_normalization_858_batchnorm_readvariableop_2_resource:hH
6sequential_90_dense_949_matmul_readvariableop_resource:hhE
7sequential_90_dense_949_biasadd_readvariableop_resource:hU
Gsequential_90_batch_normalization_859_batchnorm_readvariableop_resource:hY
Ksequential_90_batch_normalization_859_batchnorm_mul_readvariableop_resource:hW
Isequential_90_batch_normalization_859_batchnorm_readvariableop_1_resource:hW
Isequential_90_batch_normalization_859_batchnorm_readvariableop_2_resource:hH
6sequential_90_dense_950_matmul_readvariableop_resource:h/E
7sequential_90_dense_950_biasadd_readvariableop_resource:/U
Gsequential_90_batch_normalization_860_batchnorm_readvariableop_resource:/Y
Ksequential_90_batch_normalization_860_batchnorm_mul_readvariableop_resource:/W
Isequential_90_batch_normalization_860_batchnorm_readvariableop_1_resource:/W
Isequential_90_batch_normalization_860_batchnorm_readvariableop_2_resource:/H
6sequential_90_dense_951_matmul_readvariableop_resource://E
7sequential_90_dense_951_biasadd_readvariableop_resource:/U
Gsequential_90_batch_normalization_861_batchnorm_readvariableop_resource:/Y
Ksequential_90_batch_normalization_861_batchnorm_mul_readvariableop_resource:/W
Isequential_90_batch_normalization_861_batchnorm_readvariableop_1_resource:/W
Isequential_90_batch_normalization_861_batchnorm_readvariableop_2_resource:/H
6sequential_90_dense_952_matmul_readvariableop_resource:/E
7sequential_90_dense_952_biasadd_readvariableop_resource:U
Gsequential_90_batch_normalization_862_batchnorm_readvariableop_resource:Y
Ksequential_90_batch_normalization_862_batchnorm_mul_readvariableop_resource:W
Isequential_90_batch_normalization_862_batchnorm_readvariableop_1_resource:W
Isequential_90_batch_normalization_862_batchnorm_readvariableop_2_resource:H
6sequential_90_dense_953_matmul_readvariableop_resource:E
7sequential_90_dense_953_biasadd_readvariableop_resource:U
Gsequential_90_batch_normalization_863_batchnorm_readvariableop_resource:Y
Ksequential_90_batch_normalization_863_batchnorm_mul_readvariableop_resource:W
Isequential_90_batch_normalization_863_batchnorm_readvariableop_1_resource:W
Isequential_90_batch_normalization_863_batchnorm_readvariableop_2_resource:H
6sequential_90_dense_954_matmul_readvariableop_resource:E
7sequential_90_dense_954_biasadd_readvariableop_resource:U
Gsequential_90_batch_normalization_864_batchnorm_readvariableop_resource:Y
Ksequential_90_batch_normalization_864_batchnorm_mul_readvariableop_resource:W
Isequential_90_batch_normalization_864_batchnorm_readvariableop_1_resource:W
Isequential_90_batch_normalization_864_batchnorm_readvariableop_2_resource:H
6sequential_90_dense_955_matmul_readvariableop_resource:E
7sequential_90_dense_955_biasadd_readvariableop_resource:U
Gsequential_90_batch_normalization_865_batchnorm_readvariableop_resource:Y
Ksequential_90_batch_normalization_865_batchnorm_mul_readvariableop_resource:W
Isequential_90_batch_normalization_865_batchnorm_readvariableop_1_resource:W
Isequential_90_batch_normalization_865_batchnorm_readvariableop_2_resource:H
6sequential_90_dense_956_matmul_readvariableop_resource:E
7sequential_90_dense_956_biasadd_readvariableop_resource:
identity¢>sequential_90/batch_normalization_855/batchnorm/ReadVariableOp¢@sequential_90/batch_normalization_855/batchnorm/ReadVariableOp_1¢@sequential_90/batch_normalization_855/batchnorm/ReadVariableOp_2¢Bsequential_90/batch_normalization_855/batchnorm/mul/ReadVariableOp¢>sequential_90/batch_normalization_856/batchnorm/ReadVariableOp¢@sequential_90/batch_normalization_856/batchnorm/ReadVariableOp_1¢@sequential_90/batch_normalization_856/batchnorm/ReadVariableOp_2¢Bsequential_90/batch_normalization_856/batchnorm/mul/ReadVariableOp¢>sequential_90/batch_normalization_857/batchnorm/ReadVariableOp¢@sequential_90/batch_normalization_857/batchnorm/ReadVariableOp_1¢@sequential_90/batch_normalization_857/batchnorm/ReadVariableOp_2¢Bsequential_90/batch_normalization_857/batchnorm/mul/ReadVariableOp¢>sequential_90/batch_normalization_858/batchnorm/ReadVariableOp¢@sequential_90/batch_normalization_858/batchnorm/ReadVariableOp_1¢@sequential_90/batch_normalization_858/batchnorm/ReadVariableOp_2¢Bsequential_90/batch_normalization_858/batchnorm/mul/ReadVariableOp¢>sequential_90/batch_normalization_859/batchnorm/ReadVariableOp¢@sequential_90/batch_normalization_859/batchnorm/ReadVariableOp_1¢@sequential_90/batch_normalization_859/batchnorm/ReadVariableOp_2¢Bsequential_90/batch_normalization_859/batchnorm/mul/ReadVariableOp¢>sequential_90/batch_normalization_860/batchnorm/ReadVariableOp¢@sequential_90/batch_normalization_860/batchnorm/ReadVariableOp_1¢@sequential_90/batch_normalization_860/batchnorm/ReadVariableOp_2¢Bsequential_90/batch_normalization_860/batchnorm/mul/ReadVariableOp¢>sequential_90/batch_normalization_861/batchnorm/ReadVariableOp¢@sequential_90/batch_normalization_861/batchnorm/ReadVariableOp_1¢@sequential_90/batch_normalization_861/batchnorm/ReadVariableOp_2¢Bsequential_90/batch_normalization_861/batchnorm/mul/ReadVariableOp¢>sequential_90/batch_normalization_862/batchnorm/ReadVariableOp¢@sequential_90/batch_normalization_862/batchnorm/ReadVariableOp_1¢@sequential_90/batch_normalization_862/batchnorm/ReadVariableOp_2¢Bsequential_90/batch_normalization_862/batchnorm/mul/ReadVariableOp¢>sequential_90/batch_normalization_863/batchnorm/ReadVariableOp¢@sequential_90/batch_normalization_863/batchnorm/ReadVariableOp_1¢@sequential_90/batch_normalization_863/batchnorm/ReadVariableOp_2¢Bsequential_90/batch_normalization_863/batchnorm/mul/ReadVariableOp¢>sequential_90/batch_normalization_864/batchnorm/ReadVariableOp¢@sequential_90/batch_normalization_864/batchnorm/ReadVariableOp_1¢@sequential_90/batch_normalization_864/batchnorm/ReadVariableOp_2¢Bsequential_90/batch_normalization_864/batchnorm/mul/ReadVariableOp¢>sequential_90/batch_normalization_865/batchnorm/ReadVariableOp¢@sequential_90/batch_normalization_865/batchnorm/ReadVariableOp_1¢@sequential_90/batch_normalization_865/batchnorm/ReadVariableOp_2¢Bsequential_90/batch_normalization_865/batchnorm/mul/ReadVariableOp¢.sequential_90/dense_945/BiasAdd/ReadVariableOp¢-sequential_90/dense_945/MatMul/ReadVariableOp¢.sequential_90/dense_946/BiasAdd/ReadVariableOp¢-sequential_90/dense_946/MatMul/ReadVariableOp¢.sequential_90/dense_947/BiasAdd/ReadVariableOp¢-sequential_90/dense_947/MatMul/ReadVariableOp¢.sequential_90/dense_948/BiasAdd/ReadVariableOp¢-sequential_90/dense_948/MatMul/ReadVariableOp¢.sequential_90/dense_949/BiasAdd/ReadVariableOp¢-sequential_90/dense_949/MatMul/ReadVariableOp¢.sequential_90/dense_950/BiasAdd/ReadVariableOp¢-sequential_90/dense_950/MatMul/ReadVariableOp¢.sequential_90/dense_951/BiasAdd/ReadVariableOp¢-sequential_90/dense_951/MatMul/ReadVariableOp¢.sequential_90/dense_952/BiasAdd/ReadVariableOp¢-sequential_90/dense_952/MatMul/ReadVariableOp¢.sequential_90/dense_953/BiasAdd/ReadVariableOp¢-sequential_90/dense_953/MatMul/ReadVariableOp¢.sequential_90/dense_954/BiasAdd/ReadVariableOp¢-sequential_90/dense_954/MatMul/ReadVariableOp¢.sequential_90/dense_955/BiasAdd/ReadVariableOp¢-sequential_90/dense_955/MatMul/ReadVariableOp¢.sequential_90/dense_956/BiasAdd/ReadVariableOp¢-sequential_90/dense_956/MatMul/ReadVariableOp
"sequential_90/normalization_90/subSubnormalization_90_input$sequential_90_normalization_90_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#sequential_90/normalization_90/SqrtSqrt%sequential_90_normalization_90_sqrt_x*
T0*
_output_shapes

:m
(sequential_90/normalization_90/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¶
&sequential_90/normalization_90/MaximumMaximum'sequential_90/normalization_90/Sqrt:y:01sequential_90/normalization_90/Maximum/y:output:0*
T0*
_output_shapes

:·
&sequential_90/normalization_90/truedivRealDiv&sequential_90/normalization_90/sub:z:0*sequential_90/normalization_90/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_90/dense_945/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_945_matmul_readvariableop_resource*
_output_shapes

:h*
dtype0½
sequential_90/dense_945/MatMulMatMul*sequential_90/normalization_90/truediv:z:05sequential_90/dense_945/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¢
.sequential_90/dense_945/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_945_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0¾
sequential_90/dense_945/BiasAddBiasAdd(sequential_90/dense_945/MatMul:product:06sequential_90/dense_945/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhÂ
>sequential_90/batch_normalization_855/batchnorm/ReadVariableOpReadVariableOpGsequential_90_batch_normalization_855_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0z
5sequential_90/batch_normalization_855/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_90/batch_normalization_855/batchnorm/addAddV2Fsequential_90/batch_normalization_855/batchnorm/ReadVariableOp:value:0>sequential_90/batch_normalization_855/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
5sequential_90/batch_normalization_855/batchnorm/RsqrtRsqrt7sequential_90/batch_normalization_855/batchnorm/add:z:0*
T0*
_output_shapes
:hÊ
Bsequential_90/batch_normalization_855/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_90_batch_normalization_855_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0æ
3sequential_90/batch_normalization_855/batchnorm/mulMul9sequential_90/batch_normalization_855/batchnorm/Rsqrt:y:0Jsequential_90/batch_normalization_855/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hÑ
5sequential_90/batch_normalization_855/batchnorm/mul_1Mul(sequential_90/dense_945/BiasAdd:output:07sequential_90/batch_normalization_855/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhÆ
@sequential_90/batch_normalization_855/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_90_batch_normalization_855_batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0ä
5sequential_90/batch_normalization_855/batchnorm/mul_2MulHsequential_90/batch_normalization_855/batchnorm/ReadVariableOp_1:value:07sequential_90/batch_normalization_855/batchnorm/mul:z:0*
T0*
_output_shapes
:hÆ
@sequential_90/batch_normalization_855/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_90_batch_normalization_855_batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0ä
3sequential_90/batch_normalization_855/batchnorm/subSubHsequential_90/batch_normalization_855/batchnorm/ReadVariableOp_2:value:09sequential_90/batch_normalization_855/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hä
5sequential_90/batch_normalization_855/batchnorm/add_1AddV29sequential_90/batch_normalization_855/batchnorm/mul_1:z:07sequential_90/batch_normalization_855/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¨
'sequential_90/leaky_re_lu_855/LeakyRelu	LeakyRelu9sequential_90/batch_normalization_855/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>¤
-sequential_90/dense_946/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_946_matmul_readvariableop_resource*
_output_shapes

:hh*
dtype0È
sequential_90/dense_946/MatMulMatMul5sequential_90/leaky_re_lu_855/LeakyRelu:activations:05sequential_90/dense_946/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¢
.sequential_90/dense_946/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_946_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0¾
sequential_90/dense_946/BiasAddBiasAdd(sequential_90/dense_946/MatMul:product:06sequential_90/dense_946/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhÂ
>sequential_90/batch_normalization_856/batchnorm/ReadVariableOpReadVariableOpGsequential_90_batch_normalization_856_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0z
5sequential_90/batch_normalization_856/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_90/batch_normalization_856/batchnorm/addAddV2Fsequential_90/batch_normalization_856/batchnorm/ReadVariableOp:value:0>sequential_90/batch_normalization_856/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
5sequential_90/batch_normalization_856/batchnorm/RsqrtRsqrt7sequential_90/batch_normalization_856/batchnorm/add:z:0*
T0*
_output_shapes
:hÊ
Bsequential_90/batch_normalization_856/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_90_batch_normalization_856_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0æ
3sequential_90/batch_normalization_856/batchnorm/mulMul9sequential_90/batch_normalization_856/batchnorm/Rsqrt:y:0Jsequential_90/batch_normalization_856/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hÑ
5sequential_90/batch_normalization_856/batchnorm/mul_1Mul(sequential_90/dense_946/BiasAdd:output:07sequential_90/batch_normalization_856/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhÆ
@sequential_90/batch_normalization_856/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_90_batch_normalization_856_batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0ä
5sequential_90/batch_normalization_856/batchnorm/mul_2MulHsequential_90/batch_normalization_856/batchnorm/ReadVariableOp_1:value:07sequential_90/batch_normalization_856/batchnorm/mul:z:0*
T0*
_output_shapes
:hÆ
@sequential_90/batch_normalization_856/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_90_batch_normalization_856_batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0ä
3sequential_90/batch_normalization_856/batchnorm/subSubHsequential_90/batch_normalization_856/batchnorm/ReadVariableOp_2:value:09sequential_90/batch_normalization_856/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hä
5sequential_90/batch_normalization_856/batchnorm/add_1AddV29sequential_90/batch_normalization_856/batchnorm/mul_1:z:07sequential_90/batch_normalization_856/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¨
'sequential_90/leaky_re_lu_856/LeakyRelu	LeakyRelu9sequential_90/batch_normalization_856/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>¤
-sequential_90/dense_947/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_947_matmul_readvariableop_resource*
_output_shapes

:hh*
dtype0È
sequential_90/dense_947/MatMulMatMul5sequential_90/leaky_re_lu_856/LeakyRelu:activations:05sequential_90/dense_947/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¢
.sequential_90/dense_947/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_947_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0¾
sequential_90/dense_947/BiasAddBiasAdd(sequential_90/dense_947/MatMul:product:06sequential_90/dense_947/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhÂ
>sequential_90/batch_normalization_857/batchnorm/ReadVariableOpReadVariableOpGsequential_90_batch_normalization_857_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0z
5sequential_90/batch_normalization_857/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_90/batch_normalization_857/batchnorm/addAddV2Fsequential_90/batch_normalization_857/batchnorm/ReadVariableOp:value:0>sequential_90/batch_normalization_857/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
5sequential_90/batch_normalization_857/batchnorm/RsqrtRsqrt7sequential_90/batch_normalization_857/batchnorm/add:z:0*
T0*
_output_shapes
:hÊ
Bsequential_90/batch_normalization_857/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_90_batch_normalization_857_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0æ
3sequential_90/batch_normalization_857/batchnorm/mulMul9sequential_90/batch_normalization_857/batchnorm/Rsqrt:y:0Jsequential_90/batch_normalization_857/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hÑ
5sequential_90/batch_normalization_857/batchnorm/mul_1Mul(sequential_90/dense_947/BiasAdd:output:07sequential_90/batch_normalization_857/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhÆ
@sequential_90/batch_normalization_857/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_90_batch_normalization_857_batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0ä
5sequential_90/batch_normalization_857/batchnorm/mul_2MulHsequential_90/batch_normalization_857/batchnorm/ReadVariableOp_1:value:07sequential_90/batch_normalization_857/batchnorm/mul:z:0*
T0*
_output_shapes
:hÆ
@sequential_90/batch_normalization_857/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_90_batch_normalization_857_batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0ä
3sequential_90/batch_normalization_857/batchnorm/subSubHsequential_90/batch_normalization_857/batchnorm/ReadVariableOp_2:value:09sequential_90/batch_normalization_857/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hä
5sequential_90/batch_normalization_857/batchnorm/add_1AddV29sequential_90/batch_normalization_857/batchnorm/mul_1:z:07sequential_90/batch_normalization_857/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¨
'sequential_90/leaky_re_lu_857/LeakyRelu	LeakyRelu9sequential_90/batch_normalization_857/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>¤
-sequential_90/dense_948/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_948_matmul_readvariableop_resource*
_output_shapes

:hh*
dtype0È
sequential_90/dense_948/MatMulMatMul5sequential_90/leaky_re_lu_857/LeakyRelu:activations:05sequential_90/dense_948/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¢
.sequential_90/dense_948/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_948_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0¾
sequential_90/dense_948/BiasAddBiasAdd(sequential_90/dense_948/MatMul:product:06sequential_90/dense_948/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhÂ
>sequential_90/batch_normalization_858/batchnorm/ReadVariableOpReadVariableOpGsequential_90_batch_normalization_858_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0z
5sequential_90/batch_normalization_858/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_90/batch_normalization_858/batchnorm/addAddV2Fsequential_90/batch_normalization_858/batchnorm/ReadVariableOp:value:0>sequential_90/batch_normalization_858/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
5sequential_90/batch_normalization_858/batchnorm/RsqrtRsqrt7sequential_90/batch_normalization_858/batchnorm/add:z:0*
T0*
_output_shapes
:hÊ
Bsequential_90/batch_normalization_858/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_90_batch_normalization_858_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0æ
3sequential_90/batch_normalization_858/batchnorm/mulMul9sequential_90/batch_normalization_858/batchnorm/Rsqrt:y:0Jsequential_90/batch_normalization_858/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hÑ
5sequential_90/batch_normalization_858/batchnorm/mul_1Mul(sequential_90/dense_948/BiasAdd:output:07sequential_90/batch_normalization_858/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhÆ
@sequential_90/batch_normalization_858/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_90_batch_normalization_858_batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0ä
5sequential_90/batch_normalization_858/batchnorm/mul_2MulHsequential_90/batch_normalization_858/batchnorm/ReadVariableOp_1:value:07sequential_90/batch_normalization_858/batchnorm/mul:z:0*
T0*
_output_shapes
:hÆ
@sequential_90/batch_normalization_858/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_90_batch_normalization_858_batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0ä
3sequential_90/batch_normalization_858/batchnorm/subSubHsequential_90/batch_normalization_858/batchnorm/ReadVariableOp_2:value:09sequential_90/batch_normalization_858/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hä
5sequential_90/batch_normalization_858/batchnorm/add_1AddV29sequential_90/batch_normalization_858/batchnorm/mul_1:z:07sequential_90/batch_normalization_858/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¨
'sequential_90/leaky_re_lu_858/LeakyRelu	LeakyRelu9sequential_90/batch_normalization_858/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>¤
-sequential_90/dense_949/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_949_matmul_readvariableop_resource*
_output_shapes

:hh*
dtype0È
sequential_90/dense_949/MatMulMatMul5sequential_90/leaky_re_lu_858/LeakyRelu:activations:05sequential_90/dense_949/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¢
.sequential_90/dense_949/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_949_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0¾
sequential_90/dense_949/BiasAddBiasAdd(sequential_90/dense_949/MatMul:product:06sequential_90/dense_949/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhÂ
>sequential_90/batch_normalization_859/batchnorm/ReadVariableOpReadVariableOpGsequential_90_batch_normalization_859_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0z
5sequential_90/batch_normalization_859/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_90/batch_normalization_859/batchnorm/addAddV2Fsequential_90/batch_normalization_859/batchnorm/ReadVariableOp:value:0>sequential_90/batch_normalization_859/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
5sequential_90/batch_normalization_859/batchnorm/RsqrtRsqrt7sequential_90/batch_normalization_859/batchnorm/add:z:0*
T0*
_output_shapes
:hÊ
Bsequential_90/batch_normalization_859/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_90_batch_normalization_859_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0æ
3sequential_90/batch_normalization_859/batchnorm/mulMul9sequential_90/batch_normalization_859/batchnorm/Rsqrt:y:0Jsequential_90/batch_normalization_859/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hÑ
5sequential_90/batch_normalization_859/batchnorm/mul_1Mul(sequential_90/dense_949/BiasAdd:output:07sequential_90/batch_normalization_859/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhÆ
@sequential_90/batch_normalization_859/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_90_batch_normalization_859_batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0ä
5sequential_90/batch_normalization_859/batchnorm/mul_2MulHsequential_90/batch_normalization_859/batchnorm/ReadVariableOp_1:value:07sequential_90/batch_normalization_859/batchnorm/mul:z:0*
T0*
_output_shapes
:hÆ
@sequential_90/batch_normalization_859/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_90_batch_normalization_859_batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0ä
3sequential_90/batch_normalization_859/batchnorm/subSubHsequential_90/batch_normalization_859/batchnorm/ReadVariableOp_2:value:09sequential_90/batch_normalization_859/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hä
5sequential_90/batch_normalization_859/batchnorm/add_1AddV29sequential_90/batch_normalization_859/batchnorm/mul_1:z:07sequential_90/batch_normalization_859/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¨
'sequential_90/leaky_re_lu_859/LeakyRelu	LeakyRelu9sequential_90/batch_normalization_859/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>¤
-sequential_90/dense_950/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_950_matmul_readvariableop_resource*
_output_shapes

:h/*
dtype0È
sequential_90/dense_950/MatMulMatMul5sequential_90/leaky_re_lu_859/LeakyRelu:activations:05sequential_90/dense_950/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/¢
.sequential_90/dense_950/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_950_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0¾
sequential_90/dense_950/BiasAddBiasAdd(sequential_90/dense_950/MatMul:product:06sequential_90/dense_950/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/Â
>sequential_90/batch_normalization_860/batchnorm/ReadVariableOpReadVariableOpGsequential_90_batch_normalization_860_batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0z
5sequential_90/batch_normalization_860/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_90/batch_normalization_860/batchnorm/addAddV2Fsequential_90/batch_normalization_860/batchnorm/ReadVariableOp:value:0>sequential_90/batch_normalization_860/batchnorm/add/y:output:0*
T0*
_output_shapes
:/
5sequential_90/batch_normalization_860/batchnorm/RsqrtRsqrt7sequential_90/batch_normalization_860/batchnorm/add:z:0*
T0*
_output_shapes
:/Ê
Bsequential_90/batch_normalization_860/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_90_batch_normalization_860_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0æ
3sequential_90/batch_normalization_860/batchnorm/mulMul9sequential_90/batch_normalization_860/batchnorm/Rsqrt:y:0Jsequential_90/batch_normalization_860/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/Ñ
5sequential_90/batch_normalization_860/batchnorm/mul_1Mul(sequential_90/dense_950/BiasAdd:output:07sequential_90/batch_normalization_860/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/Æ
@sequential_90/batch_normalization_860/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_90_batch_normalization_860_batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0ä
5sequential_90/batch_normalization_860/batchnorm/mul_2MulHsequential_90/batch_normalization_860/batchnorm/ReadVariableOp_1:value:07sequential_90/batch_normalization_860/batchnorm/mul:z:0*
T0*
_output_shapes
:/Æ
@sequential_90/batch_normalization_860/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_90_batch_normalization_860_batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0ä
3sequential_90/batch_normalization_860/batchnorm/subSubHsequential_90/batch_normalization_860/batchnorm/ReadVariableOp_2:value:09sequential_90/batch_normalization_860/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/ä
5sequential_90/batch_normalization_860/batchnorm/add_1AddV29sequential_90/batch_normalization_860/batchnorm/mul_1:z:07sequential_90/batch_normalization_860/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/¨
'sequential_90/leaky_re_lu_860/LeakyRelu	LeakyRelu9sequential_90/batch_normalization_860/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*
alpha%>¤
-sequential_90/dense_951/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_951_matmul_readvariableop_resource*
_output_shapes

://*
dtype0È
sequential_90/dense_951/MatMulMatMul5sequential_90/leaky_re_lu_860/LeakyRelu:activations:05sequential_90/dense_951/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/¢
.sequential_90/dense_951/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_951_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0¾
sequential_90/dense_951/BiasAddBiasAdd(sequential_90/dense_951/MatMul:product:06sequential_90/dense_951/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/Â
>sequential_90/batch_normalization_861/batchnorm/ReadVariableOpReadVariableOpGsequential_90_batch_normalization_861_batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0z
5sequential_90/batch_normalization_861/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_90/batch_normalization_861/batchnorm/addAddV2Fsequential_90/batch_normalization_861/batchnorm/ReadVariableOp:value:0>sequential_90/batch_normalization_861/batchnorm/add/y:output:0*
T0*
_output_shapes
:/
5sequential_90/batch_normalization_861/batchnorm/RsqrtRsqrt7sequential_90/batch_normalization_861/batchnorm/add:z:0*
T0*
_output_shapes
:/Ê
Bsequential_90/batch_normalization_861/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_90_batch_normalization_861_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0æ
3sequential_90/batch_normalization_861/batchnorm/mulMul9sequential_90/batch_normalization_861/batchnorm/Rsqrt:y:0Jsequential_90/batch_normalization_861/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/Ñ
5sequential_90/batch_normalization_861/batchnorm/mul_1Mul(sequential_90/dense_951/BiasAdd:output:07sequential_90/batch_normalization_861/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/Æ
@sequential_90/batch_normalization_861/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_90_batch_normalization_861_batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0ä
5sequential_90/batch_normalization_861/batchnorm/mul_2MulHsequential_90/batch_normalization_861/batchnorm/ReadVariableOp_1:value:07sequential_90/batch_normalization_861/batchnorm/mul:z:0*
T0*
_output_shapes
:/Æ
@sequential_90/batch_normalization_861/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_90_batch_normalization_861_batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0ä
3sequential_90/batch_normalization_861/batchnorm/subSubHsequential_90/batch_normalization_861/batchnorm/ReadVariableOp_2:value:09sequential_90/batch_normalization_861/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/ä
5sequential_90/batch_normalization_861/batchnorm/add_1AddV29sequential_90/batch_normalization_861/batchnorm/mul_1:z:07sequential_90/batch_normalization_861/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/¨
'sequential_90/leaky_re_lu_861/LeakyRelu	LeakyRelu9sequential_90/batch_normalization_861/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*
alpha%>¤
-sequential_90/dense_952/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_952_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0È
sequential_90/dense_952/MatMulMatMul5sequential_90/leaky_re_lu_861/LeakyRelu:activations:05sequential_90/dense_952/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_90/dense_952/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_952_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_90/dense_952/BiasAddBiasAdd(sequential_90/dense_952/MatMul:product:06sequential_90/dense_952/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_90/batch_normalization_862/batchnorm/ReadVariableOpReadVariableOpGsequential_90_batch_normalization_862_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_90/batch_normalization_862/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_90/batch_normalization_862/batchnorm/addAddV2Fsequential_90/batch_normalization_862/batchnorm/ReadVariableOp:value:0>sequential_90/batch_normalization_862/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_90/batch_normalization_862/batchnorm/RsqrtRsqrt7sequential_90/batch_normalization_862/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_90/batch_normalization_862/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_90_batch_normalization_862_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_90/batch_normalization_862/batchnorm/mulMul9sequential_90/batch_normalization_862/batchnorm/Rsqrt:y:0Jsequential_90/batch_normalization_862/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ñ
5sequential_90/batch_normalization_862/batchnorm/mul_1Mul(sequential_90/dense_952/BiasAdd:output:07sequential_90/batch_normalization_862/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_90/batch_normalization_862/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_90_batch_normalization_862_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_90/batch_normalization_862/batchnorm/mul_2MulHsequential_90/batch_normalization_862/batchnorm/ReadVariableOp_1:value:07sequential_90/batch_normalization_862/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_90/batch_normalization_862/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_90_batch_normalization_862_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_90/batch_normalization_862/batchnorm/subSubHsequential_90/batch_normalization_862/batchnorm/ReadVariableOp_2:value:09sequential_90/batch_normalization_862/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_90/batch_normalization_862/batchnorm/add_1AddV29sequential_90/batch_normalization_862/batchnorm/mul_1:z:07sequential_90/batch_normalization_862/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_90/leaky_re_lu_862/LeakyRelu	LeakyRelu9sequential_90/batch_normalization_862/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¤
-sequential_90/dense_953/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_953_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
sequential_90/dense_953/MatMulMatMul5sequential_90/leaky_re_lu_862/LeakyRelu:activations:05sequential_90/dense_953/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_90/dense_953/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_953_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_90/dense_953/BiasAddBiasAdd(sequential_90/dense_953/MatMul:product:06sequential_90/dense_953/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_90/batch_normalization_863/batchnorm/ReadVariableOpReadVariableOpGsequential_90_batch_normalization_863_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_90/batch_normalization_863/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_90/batch_normalization_863/batchnorm/addAddV2Fsequential_90/batch_normalization_863/batchnorm/ReadVariableOp:value:0>sequential_90/batch_normalization_863/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_90/batch_normalization_863/batchnorm/RsqrtRsqrt7sequential_90/batch_normalization_863/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_90/batch_normalization_863/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_90_batch_normalization_863_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_90/batch_normalization_863/batchnorm/mulMul9sequential_90/batch_normalization_863/batchnorm/Rsqrt:y:0Jsequential_90/batch_normalization_863/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ñ
5sequential_90/batch_normalization_863/batchnorm/mul_1Mul(sequential_90/dense_953/BiasAdd:output:07sequential_90/batch_normalization_863/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_90/batch_normalization_863/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_90_batch_normalization_863_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_90/batch_normalization_863/batchnorm/mul_2MulHsequential_90/batch_normalization_863/batchnorm/ReadVariableOp_1:value:07sequential_90/batch_normalization_863/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_90/batch_normalization_863/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_90_batch_normalization_863_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_90/batch_normalization_863/batchnorm/subSubHsequential_90/batch_normalization_863/batchnorm/ReadVariableOp_2:value:09sequential_90/batch_normalization_863/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_90/batch_normalization_863/batchnorm/add_1AddV29sequential_90/batch_normalization_863/batchnorm/mul_1:z:07sequential_90/batch_normalization_863/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_90/leaky_re_lu_863/LeakyRelu	LeakyRelu9sequential_90/batch_normalization_863/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¤
-sequential_90/dense_954/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_954_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
sequential_90/dense_954/MatMulMatMul5sequential_90/leaky_re_lu_863/LeakyRelu:activations:05sequential_90/dense_954/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_90/dense_954/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_954_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_90/dense_954/BiasAddBiasAdd(sequential_90/dense_954/MatMul:product:06sequential_90/dense_954/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_90/batch_normalization_864/batchnorm/ReadVariableOpReadVariableOpGsequential_90_batch_normalization_864_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_90/batch_normalization_864/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_90/batch_normalization_864/batchnorm/addAddV2Fsequential_90/batch_normalization_864/batchnorm/ReadVariableOp:value:0>sequential_90/batch_normalization_864/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_90/batch_normalization_864/batchnorm/RsqrtRsqrt7sequential_90/batch_normalization_864/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_90/batch_normalization_864/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_90_batch_normalization_864_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_90/batch_normalization_864/batchnorm/mulMul9sequential_90/batch_normalization_864/batchnorm/Rsqrt:y:0Jsequential_90/batch_normalization_864/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ñ
5sequential_90/batch_normalization_864/batchnorm/mul_1Mul(sequential_90/dense_954/BiasAdd:output:07sequential_90/batch_normalization_864/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_90/batch_normalization_864/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_90_batch_normalization_864_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_90/batch_normalization_864/batchnorm/mul_2MulHsequential_90/batch_normalization_864/batchnorm/ReadVariableOp_1:value:07sequential_90/batch_normalization_864/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_90/batch_normalization_864/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_90_batch_normalization_864_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_90/batch_normalization_864/batchnorm/subSubHsequential_90/batch_normalization_864/batchnorm/ReadVariableOp_2:value:09sequential_90/batch_normalization_864/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_90/batch_normalization_864/batchnorm/add_1AddV29sequential_90/batch_normalization_864/batchnorm/mul_1:z:07sequential_90/batch_normalization_864/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_90/leaky_re_lu_864/LeakyRelu	LeakyRelu9sequential_90/batch_normalization_864/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¤
-sequential_90/dense_955/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_955_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
sequential_90/dense_955/MatMulMatMul5sequential_90/leaky_re_lu_864/LeakyRelu:activations:05sequential_90/dense_955/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_90/dense_955/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_955_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_90/dense_955/BiasAddBiasAdd(sequential_90/dense_955/MatMul:product:06sequential_90/dense_955/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_90/batch_normalization_865/batchnorm/ReadVariableOpReadVariableOpGsequential_90_batch_normalization_865_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_90/batch_normalization_865/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_90/batch_normalization_865/batchnorm/addAddV2Fsequential_90/batch_normalization_865/batchnorm/ReadVariableOp:value:0>sequential_90/batch_normalization_865/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_90/batch_normalization_865/batchnorm/RsqrtRsqrt7sequential_90/batch_normalization_865/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_90/batch_normalization_865/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_90_batch_normalization_865_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_90/batch_normalization_865/batchnorm/mulMul9sequential_90/batch_normalization_865/batchnorm/Rsqrt:y:0Jsequential_90/batch_normalization_865/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ñ
5sequential_90/batch_normalization_865/batchnorm/mul_1Mul(sequential_90/dense_955/BiasAdd:output:07sequential_90/batch_normalization_865/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_90/batch_normalization_865/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_90_batch_normalization_865_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_90/batch_normalization_865/batchnorm/mul_2MulHsequential_90/batch_normalization_865/batchnorm/ReadVariableOp_1:value:07sequential_90/batch_normalization_865/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_90/batch_normalization_865/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_90_batch_normalization_865_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_90/batch_normalization_865/batchnorm/subSubHsequential_90/batch_normalization_865/batchnorm/ReadVariableOp_2:value:09sequential_90/batch_normalization_865/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_90/batch_normalization_865/batchnorm/add_1AddV29sequential_90/batch_normalization_865/batchnorm/mul_1:z:07sequential_90/batch_normalization_865/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_90/leaky_re_lu_865/LeakyRelu	LeakyRelu9sequential_90/batch_normalization_865/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¤
-sequential_90/dense_956/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_956_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
sequential_90/dense_956/MatMulMatMul5sequential_90/leaky_re_lu_865/LeakyRelu:activations:05sequential_90/dense_956/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_90/dense_956/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_956_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_90/dense_956/BiasAddBiasAdd(sequential_90/dense_956/MatMul:product:06sequential_90/dense_956/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_90/dense_956/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ 
NoOpNoOp?^sequential_90/batch_normalization_855/batchnorm/ReadVariableOpA^sequential_90/batch_normalization_855/batchnorm/ReadVariableOp_1A^sequential_90/batch_normalization_855/batchnorm/ReadVariableOp_2C^sequential_90/batch_normalization_855/batchnorm/mul/ReadVariableOp?^sequential_90/batch_normalization_856/batchnorm/ReadVariableOpA^sequential_90/batch_normalization_856/batchnorm/ReadVariableOp_1A^sequential_90/batch_normalization_856/batchnorm/ReadVariableOp_2C^sequential_90/batch_normalization_856/batchnorm/mul/ReadVariableOp?^sequential_90/batch_normalization_857/batchnorm/ReadVariableOpA^sequential_90/batch_normalization_857/batchnorm/ReadVariableOp_1A^sequential_90/batch_normalization_857/batchnorm/ReadVariableOp_2C^sequential_90/batch_normalization_857/batchnorm/mul/ReadVariableOp?^sequential_90/batch_normalization_858/batchnorm/ReadVariableOpA^sequential_90/batch_normalization_858/batchnorm/ReadVariableOp_1A^sequential_90/batch_normalization_858/batchnorm/ReadVariableOp_2C^sequential_90/batch_normalization_858/batchnorm/mul/ReadVariableOp?^sequential_90/batch_normalization_859/batchnorm/ReadVariableOpA^sequential_90/batch_normalization_859/batchnorm/ReadVariableOp_1A^sequential_90/batch_normalization_859/batchnorm/ReadVariableOp_2C^sequential_90/batch_normalization_859/batchnorm/mul/ReadVariableOp?^sequential_90/batch_normalization_860/batchnorm/ReadVariableOpA^sequential_90/batch_normalization_860/batchnorm/ReadVariableOp_1A^sequential_90/batch_normalization_860/batchnorm/ReadVariableOp_2C^sequential_90/batch_normalization_860/batchnorm/mul/ReadVariableOp?^sequential_90/batch_normalization_861/batchnorm/ReadVariableOpA^sequential_90/batch_normalization_861/batchnorm/ReadVariableOp_1A^sequential_90/batch_normalization_861/batchnorm/ReadVariableOp_2C^sequential_90/batch_normalization_861/batchnorm/mul/ReadVariableOp?^sequential_90/batch_normalization_862/batchnorm/ReadVariableOpA^sequential_90/batch_normalization_862/batchnorm/ReadVariableOp_1A^sequential_90/batch_normalization_862/batchnorm/ReadVariableOp_2C^sequential_90/batch_normalization_862/batchnorm/mul/ReadVariableOp?^sequential_90/batch_normalization_863/batchnorm/ReadVariableOpA^sequential_90/batch_normalization_863/batchnorm/ReadVariableOp_1A^sequential_90/batch_normalization_863/batchnorm/ReadVariableOp_2C^sequential_90/batch_normalization_863/batchnorm/mul/ReadVariableOp?^sequential_90/batch_normalization_864/batchnorm/ReadVariableOpA^sequential_90/batch_normalization_864/batchnorm/ReadVariableOp_1A^sequential_90/batch_normalization_864/batchnorm/ReadVariableOp_2C^sequential_90/batch_normalization_864/batchnorm/mul/ReadVariableOp?^sequential_90/batch_normalization_865/batchnorm/ReadVariableOpA^sequential_90/batch_normalization_865/batchnorm/ReadVariableOp_1A^sequential_90/batch_normalization_865/batchnorm/ReadVariableOp_2C^sequential_90/batch_normalization_865/batchnorm/mul/ReadVariableOp/^sequential_90/dense_945/BiasAdd/ReadVariableOp.^sequential_90/dense_945/MatMul/ReadVariableOp/^sequential_90/dense_946/BiasAdd/ReadVariableOp.^sequential_90/dense_946/MatMul/ReadVariableOp/^sequential_90/dense_947/BiasAdd/ReadVariableOp.^sequential_90/dense_947/MatMul/ReadVariableOp/^sequential_90/dense_948/BiasAdd/ReadVariableOp.^sequential_90/dense_948/MatMul/ReadVariableOp/^sequential_90/dense_949/BiasAdd/ReadVariableOp.^sequential_90/dense_949/MatMul/ReadVariableOp/^sequential_90/dense_950/BiasAdd/ReadVariableOp.^sequential_90/dense_950/MatMul/ReadVariableOp/^sequential_90/dense_951/BiasAdd/ReadVariableOp.^sequential_90/dense_951/MatMul/ReadVariableOp/^sequential_90/dense_952/BiasAdd/ReadVariableOp.^sequential_90/dense_952/MatMul/ReadVariableOp/^sequential_90/dense_953/BiasAdd/ReadVariableOp.^sequential_90/dense_953/MatMul/ReadVariableOp/^sequential_90/dense_954/BiasAdd/ReadVariableOp.^sequential_90/dense_954/MatMul/ReadVariableOp/^sequential_90/dense_955/BiasAdd/ReadVariableOp.^sequential_90/dense_955/MatMul/ReadVariableOp/^sequential_90/dense_956/BiasAdd/ReadVariableOp.^sequential_90/dense_956/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>sequential_90/batch_normalization_855/batchnorm/ReadVariableOp>sequential_90/batch_normalization_855/batchnorm/ReadVariableOp2
@sequential_90/batch_normalization_855/batchnorm/ReadVariableOp_1@sequential_90/batch_normalization_855/batchnorm/ReadVariableOp_12
@sequential_90/batch_normalization_855/batchnorm/ReadVariableOp_2@sequential_90/batch_normalization_855/batchnorm/ReadVariableOp_22
Bsequential_90/batch_normalization_855/batchnorm/mul/ReadVariableOpBsequential_90/batch_normalization_855/batchnorm/mul/ReadVariableOp2
>sequential_90/batch_normalization_856/batchnorm/ReadVariableOp>sequential_90/batch_normalization_856/batchnorm/ReadVariableOp2
@sequential_90/batch_normalization_856/batchnorm/ReadVariableOp_1@sequential_90/batch_normalization_856/batchnorm/ReadVariableOp_12
@sequential_90/batch_normalization_856/batchnorm/ReadVariableOp_2@sequential_90/batch_normalization_856/batchnorm/ReadVariableOp_22
Bsequential_90/batch_normalization_856/batchnorm/mul/ReadVariableOpBsequential_90/batch_normalization_856/batchnorm/mul/ReadVariableOp2
>sequential_90/batch_normalization_857/batchnorm/ReadVariableOp>sequential_90/batch_normalization_857/batchnorm/ReadVariableOp2
@sequential_90/batch_normalization_857/batchnorm/ReadVariableOp_1@sequential_90/batch_normalization_857/batchnorm/ReadVariableOp_12
@sequential_90/batch_normalization_857/batchnorm/ReadVariableOp_2@sequential_90/batch_normalization_857/batchnorm/ReadVariableOp_22
Bsequential_90/batch_normalization_857/batchnorm/mul/ReadVariableOpBsequential_90/batch_normalization_857/batchnorm/mul/ReadVariableOp2
>sequential_90/batch_normalization_858/batchnorm/ReadVariableOp>sequential_90/batch_normalization_858/batchnorm/ReadVariableOp2
@sequential_90/batch_normalization_858/batchnorm/ReadVariableOp_1@sequential_90/batch_normalization_858/batchnorm/ReadVariableOp_12
@sequential_90/batch_normalization_858/batchnorm/ReadVariableOp_2@sequential_90/batch_normalization_858/batchnorm/ReadVariableOp_22
Bsequential_90/batch_normalization_858/batchnorm/mul/ReadVariableOpBsequential_90/batch_normalization_858/batchnorm/mul/ReadVariableOp2
>sequential_90/batch_normalization_859/batchnorm/ReadVariableOp>sequential_90/batch_normalization_859/batchnorm/ReadVariableOp2
@sequential_90/batch_normalization_859/batchnorm/ReadVariableOp_1@sequential_90/batch_normalization_859/batchnorm/ReadVariableOp_12
@sequential_90/batch_normalization_859/batchnorm/ReadVariableOp_2@sequential_90/batch_normalization_859/batchnorm/ReadVariableOp_22
Bsequential_90/batch_normalization_859/batchnorm/mul/ReadVariableOpBsequential_90/batch_normalization_859/batchnorm/mul/ReadVariableOp2
>sequential_90/batch_normalization_860/batchnorm/ReadVariableOp>sequential_90/batch_normalization_860/batchnorm/ReadVariableOp2
@sequential_90/batch_normalization_860/batchnorm/ReadVariableOp_1@sequential_90/batch_normalization_860/batchnorm/ReadVariableOp_12
@sequential_90/batch_normalization_860/batchnorm/ReadVariableOp_2@sequential_90/batch_normalization_860/batchnorm/ReadVariableOp_22
Bsequential_90/batch_normalization_860/batchnorm/mul/ReadVariableOpBsequential_90/batch_normalization_860/batchnorm/mul/ReadVariableOp2
>sequential_90/batch_normalization_861/batchnorm/ReadVariableOp>sequential_90/batch_normalization_861/batchnorm/ReadVariableOp2
@sequential_90/batch_normalization_861/batchnorm/ReadVariableOp_1@sequential_90/batch_normalization_861/batchnorm/ReadVariableOp_12
@sequential_90/batch_normalization_861/batchnorm/ReadVariableOp_2@sequential_90/batch_normalization_861/batchnorm/ReadVariableOp_22
Bsequential_90/batch_normalization_861/batchnorm/mul/ReadVariableOpBsequential_90/batch_normalization_861/batchnorm/mul/ReadVariableOp2
>sequential_90/batch_normalization_862/batchnorm/ReadVariableOp>sequential_90/batch_normalization_862/batchnorm/ReadVariableOp2
@sequential_90/batch_normalization_862/batchnorm/ReadVariableOp_1@sequential_90/batch_normalization_862/batchnorm/ReadVariableOp_12
@sequential_90/batch_normalization_862/batchnorm/ReadVariableOp_2@sequential_90/batch_normalization_862/batchnorm/ReadVariableOp_22
Bsequential_90/batch_normalization_862/batchnorm/mul/ReadVariableOpBsequential_90/batch_normalization_862/batchnorm/mul/ReadVariableOp2
>sequential_90/batch_normalization_863/batchnorm/ReadVariableOp>sequential_90/batch_normalization_863/batchnorm/ReadVariableOp2
@sequential_90/batch_normalization_863/batchnorm/ReadVariableOp_1@sequential_90/batch_normalization_863/batchnorm/ReadVariableOp_12
@sequential_90/batch_normalization_863/batchnorm/ReadVariableOp_2@sequential_90/batch_normalization_863/batchnorm/ReadVariableOp_22
Bsequential_90/batch_normalization_863/batchnorm/mul/ReadVariableOpBsequential_90/batch_normalization_863/batchnorm/mul/ReadVariableOp2
>sequential_90/batch_normalization_864/batchnorm/ReadVariableOp>sequential_90/batch_normalization_864/batchnorm/ReadVariableOp2
@sequential_90/batch_normalization_864/batchnorm/ReadVariableOp_1@sequential_90/batch_normalization_864/batchnorm/ReadVariableOp_12
@sequential_90/batch_normalization_864/batchnorm/ReadVariableOp_2@sequential_90/batch_normalization_864/batchnorm/ReadVariableOp_22
Bsequential_90/batch_normalization_864/batchnorm/mul/ReadVariableOpBsequential_90/batch_normalization_864/batchnorm/mul/ReadVariableOp2
>sequential_90/batch_normalization_865/batchnorm/ReadVariableOp>sequential_90/batch_normalization_865/batchnorm/ReadVariableOp2
@sequential_90/batch_normalization_865/batchnorm/ReadVariableOp_1@sequential_90/batch_normalization_865/batchnorm/ReadVariableOp_12
@sequential_90/batch_normalization_865/batchnorm/ReadVariableOp_2@sequential_90/batch_normalization_865/batchnorm/ReadVariableOp_22
Bsequential_90/batch_normalization_865/batchnorm/mul/ReadVariableOpBsequential_90/batch_normalization_865/batchnorm/mul/ReadVariableOp2`
.sequential_90/dense_945/BiasAdd/ReadVariableOp.sequential_90/dense_945/BiasAdd/ReadVariableOp2^
-sequential_90/dense_945/MatMul/ReadVariableOp-sequential_90/dense_945/MatMul/ReadVariableOp2`
.sequential_90/dense_946/BiasAdd/ReadVariableOp.sequential_90/dense_946/BiasAdd/ReadVariableOp2^
-sequential_90/dense_946/MatMul/ReadVariableOp-sequential_90/dense_946/MatMul/ReadVariableOp2`
.sequential_90/dense_947/BiasAdd/ReadVariableOp.sequential_90/dense_947/BiasAdd/ReadVariableOp2^
-sequential_90/dense_947/MatMul/ReadVariableOp-sequential_90/dense_947/MatMul/ReadVariableOp2`
.sequential_90/dense_948/BiasAdd/ReadVariableOp.sequential_90/dense_948/BiasAdd/ReadVariableOp2^
-sequential_90/dense_948/MatMul/ReadVariableOp-sequential_90/dense_948/MatMul/ReadVariableOp2`
.sequential_90/dense_949/BiasAdd/ReadVariableOp.sequential_90/dense_949/BiasAdd/ReadVariableOp2^
-sequential_90/dense_949/MatMul/ReadVariableOp-sequential_90/dense_949/MatMul/ReadVariableOp2`
.sequential_90/dense_950/BiasAdd/ReadVariableOp.sequential_90/dense_950/BiasAdd/ReadVariableOp2^
-sequential_90/dense_950/MatMul/ReadVariableOp-sequential_90/dense_950/MatMul/ReadVariableOp2`
.sequential_90/dense_951/BiasAdd/ReadVariableOp.sequential_90/dense_951/BiasAdd/ReadVariableOp2^
-sequential_90/dense_951/MatMul/ReadVariableOp-sequential_90/dense_951/MatMul/ReadVariableOp2`
.sequential_90/dense_952/BiasAdd/ReadVariableOp.sequential_90/dense_952/BiasAdd/ReadVariableOp2^
-sequential_90/dense_952/MatMul/ReadVariableOp-sequential_90/dense_952/MatMul/ReadVariableOp2`
.sequential_90/dense_953/BiasAdd/ReadVariableOp.sequential_90/dense_953/BiasAdd/ReadVariableOp2^
-sequential_90/dense_953/MatMul/ReadVariableOp-sequential_90/dense_953/MatMul/ReadVariableOp2`
.sequential_90/dense_954/BiasAdd/ReadVariableOp.sequential_90/dense_954/BiasAdd/ReadVariableOp2^
-sequential_90/dense_954/MatMul/ReadVariableOp-sequential_90/dense_954/MatMul/ReadVariableOp2`
.sequential_90/dense_955/BiasAdd/ReadVariableOp.sequential_90/dense_955/BiasAdd/ReadVariableOp2^
-sequential_90/dense_955/MatMul/ReadVariableOp-sequential_90/dense_955/MatMul/ReadVariableOp2`
.sequential_90/dense_956/BiasAdd/ReadVariableOp.sequential_90/dense_956/BiasAdd/ReadVariableOp2^
-sequential_90/dense_956/MatMul/ReadVariableOp-sequential_90/dense_956/MatMul/ReadVariableOp:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_90_input:$ 

_output_shapes

::$ 

_output_shapes

:
%
ì
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_839917

inputs5
'assignmovingavg_readvariableop_resource:h7
)assignmovingavg_1_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h/
!batchnorm_readvariableop_resource:h
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:h
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:h*
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
:h*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:hx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h¬
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
:h*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:h~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h´
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:hv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_835932

inputs/
!batchnorm_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h1
#batchnorm_readvariableop_1_resource:h1
#batchnorm_readvariableop_2_resource:h
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:hz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_836096

inputs/
!batchnorm_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h1
#batchnorm_readvariableop_1_resource:h1
#batchnorm_readvariableop_2_resource:h
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:hz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
È	
ö
E__inference_dense_945_layer_call_and_return_conditional_losses_839619

inputs0
matmul_readvariableop_resource:h-
biasadd_readvariableop_resource:h
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:h*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:h*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhw
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
¬
Ó
8__inference_batch_normalization_855_layer_call_fn_839632

inputs
unknown:h
	unknown_0:h
	unknown_1:h
	unknown_2:h
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_835850o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_835979

inputs5
'assignmovingavg_readvariableop_resource:h7
)assignmovingavg_1_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h/
!batchnorm_readvariableop_resource:h
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:h
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:h*
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
:h*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:hx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h¬
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
:h*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:h~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h´
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:hv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_840537

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_948_layer_call_and_return_conditional_losses_836848

inputs0
matmul_readvariableop_resource:hh-
biasadd_readvariableop_resource:h
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:hh*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:h*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ä

*__inference_dense_947_layer_call_fn_839827

inputs
unknown:hh
	unknown_0:h
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_947_layer_call_and_return_conditional_losses_836816o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_836143

inputs5
'assignmovingavg_readvariableop_resource:h7
)assignmovingavg_1_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h/
!batchnorm_readvariableop_resource:h
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:h
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:h*
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
:h*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:hx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h¬
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
:h*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:h~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h´
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:hv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_860_layer_call_fn_840249

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
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_860_layer_call_and_return_conditional_losses_836932`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_839992

inputs/
!batchnorm_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h1
#batchnorm_readvariableop_1_resource:h1
#batchnorm_readvariableop_2_resource:h
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:hz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_861_layer_call_fn_840299

inputs
unknown:/
	unknown_0:/
	unknown_1:/
	unknown_2:/
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_836389o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_836635

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_835850

inputs/
!batchnorm_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h1
#batchnorm_readvariableop_1_resource:h1
#batchnorm_readvariableop_2_resource:h
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:hz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_836471

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_865_layer_call_fn_840735

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_836717o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_863_layer_call_and_return_conditional_losses_837028

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_864_layer_call_and_return_conditional_losses_840690

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_839774

inputs/
!batchnorm_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h1
#batchnorm_readvariableop_1_resource:h1
#batchnorm_readvariableop_2_resource:h
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:hz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
È	
ö
E__inference_dense_946_layer_call_and_return_conditional_losses_836784

inputs0
matmul_readvariableop_resource:hh-
biasadd_readvariableop_resource:h
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:hh*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:h*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Á

.__inference_sequential_90_layer_call_fn_838712

inputs
unknown
	unknown_0
	unknown_1:h
	unknown_2:h
	unknown_3:h
	unknown_4:h
	unknown_5:h
	unknown_6:h
	unknown_7:hh
	unknown_8:h
	unknown_9:h

unknown_10:h

unknown_11:h

unknown_12:h

unknown_13:hh

unknown_14:h

unknown_15:h

unknown_16:h

unknown_17:h

unknown_18:h

unknown_19:hh

unknown_20:h

unknown_21:h

unknown_22:h

unknown_23:h

unknown_24:h

unknown_25:hh

unknown_26:h

unknown_27:h

unknown_28:h

unknown_29:h

unknown_30:h

unknown_31:h/

unknown_32:/

unknown_33:/

unknown_34:/

unknown_35:/

unknown_36:/

unknown_37://

unknown_38:/

unknown_39:/

unknown_40:/

unknown_41:/

unknown_42:/

unknown_43:/

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:

unknown_61:

unknown_62:

unknown_63:

unknown_64:

unknown_65:

unknown_66:

unknown_67:

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
I__inference_sequential_90_layer_call_and_return_conditional_losses_837768o
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
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_840680

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_862_layer_call_fn_840408

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_836471o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_839883

inputs/
!batchnorm_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h1
#batchnorm_readvariableop_1_resource:h1
#batchnorm_readvariableop_2_resource:h
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:hz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_865_layer_call_fn_840722

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_836670o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_863_layer_call_fn_840576

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_863_layer_call_and_return_conditional_losses_837028`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_948_layer_call_fn_839936

inputs
unknown:hh
	unknown_0:h
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_948_layer_call_and_return_conditional_losses_836848o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_836588

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×

.__inference_sequential_90_layer_call_fn_838567

inputs
unknown
	unknown_0
	unknown_1:h
	unknown_2:h
	unknown_3:h
	unknown_4:h
	unknown_5:h
	unknown_6:h
	unknown_7:hh
	unknown_8:h
	unknown_9:h

unknown_10:h

unknown_11:h

unknown_12:h

unknown_13:hh

unknown_14:h

unknown_15:h

unknown_16:h

unknown_17:h

unknown_18:h

unknown_19:hh

unknown_20:h

unknown_21:h

unknown_22:h

unknown_23:h

unknown_24:h

unknown_25:hh

unknown_26:h

unknown_27:h

unknown_28:h

unknown_29:h

unknown_30:h

unknown_31:h/

unknown_32:/

unknown_33:/

unknown_34:/

unknown_35:/

unknown_36:/

unknown_37://

unknown_38:/

unknown_39:/

unknown_40:/

unknown_41:/

unknown_42:/

unknown_43:/

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:

unknown_61:

unknown_62:

unknown_63:

unknown_64:

unknown_65:

unknown_66:

unknown_67:

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
I__inference_sequential_90_layer_call_and_return_conditional_losses_837111o
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
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_840135

inputs5
'assignmovingavg_readvariableop_resource:h7
)assignmovingavg_1_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h/
!batchnorm_readvariableop_resource:h
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:h
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:h*
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
:h*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:hx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h¬
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
:h*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:h~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h´
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:hv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_840428

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_955_layer_call_fn_840699

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_955_layer_call_and_return_conditional_losses_837072o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_862_layer_call_fn_840395

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_836424o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_840353

inputs5
'assignmovingavg_readvariableop_resource:/7
)assignmovingavg_1_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource://
!batchnorm_readvariableop_resource:/
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:/
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:/*
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
:/*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:/x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:/¬
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
:/*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:/~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:/´
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:/v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_836424

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_858_layer_call_fn_839959

inputs
unknown:h
	unknown_0:h
	unknown_1:h
	unknown_2:h
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_836096o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
È	
ö
E__inference_dense_952_layer_call_and_return_conditional_losses_840382

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
»´

I__inference_sequential_90_layer_call_and_return_conditional_losses_837111

inputs
normalization_90_sub_y
normalization_90_sqrt_x"
dense_945_836753:h
dense_945_836755:h,
batch_normalization_855_836758:h,
batch_normalization_855_836760:h,
batch_normalization_855_836762:h,
batch_normalization_855_836764:h"
dense_946_836785:hh
dense_946_836787:h,
batch_normalization_856_836790:h,
batch_normalization_856_836792:h,
batch_normalization_856_836794:h,
batch_normalization_856_836796:h"
dense_947_836817:hh
dense_947_836819:h,
batch_normalization_857_836822:h,
batch_normalization_857_836824:h,
batch_normalization_857_836826:h,
batch_normalization_857_836828:h"
dense_948_836849:hh
dense_948_836851:h,
batch_normalization_858_836854:h,
batch_normalization_858_836856:h,
batch_normalization_858_836858:h,
batch_normalization_858_836860:h"
dense_949_836881:hh
dense_949_836883:h,
batch_normalization_859_836886:h,
batch_normalization_859_836888:h,
batch_normalization_859_836890:h,
batch_normalization_859_836892:h"
dense_950_836913:h/
dense_950_836915:/,
batch_normalization_860_836918:/,
batch_normalization_860_836920:/,
batch_normalization_860_836922:/,
batch_normalization_860_836924:/"
dense_951_836945://
dense_951_836947:/,
batch_normalization_861_836950:/,
batch_normalization_861_836952:/,
batch_normalization_861_836954:/,
batch_normalization_861_836956:/"
dense_952_836977:/
dense_952_836979:,
batch_normalization_862_836982:,
batch_normalization_862_836984:,
batch_normalization_862_836986:,
batch_normalization_862_836988:"
dense_953_837009:
dense_953_837011:,
batch_normalization_863_837014:,
batch_normalization_863_837016:,
batch_normalization_863_837018:,
batch_normalization_863_837020:"
dense_954_837041:
dense_954_837043:,
batch_normalization_864_837046:,
batch_normalization_864_837048:,
batch_normalization_864_837050:,
batch_normalization_864_837052:"
dense_955_837073:
dense_955_837075:,
batch_normalization_865_837078:,
batch_normalization_865_837080:,
batch_normalization_865_837082:,
batch_normalization_865_837084:"
dense_956_837105:
dense_956_837107:
identity¢/batch_normalization_855/StatefulPartitionedCall¢/batch_normalization_856/StatefulPartitionedCall¢/batch_normalization_857/StatefulPartitionedCall¢/batch_normalization_858/StatefulPartitionedCall¢/batch_normalization_859/StatefulPartitionedCall¢/batch_normalization_860/StatefulPartitionedCall¢/batch_normalization_861/StatefulPartitionedCall¢/batch_normalization_862/StatefulPartitionedCall¢/batch_normalization_863/StatefulPartitionedCall¢/batch_normalization_864/StatefulPartitionedCall¢/batch_normalization_865/StatefulPartitionedCall¢!dense_945/StatefulPartitionedCall¢!dense_946/StatefulPartitionedCall¢!dense_947/StatefulPartitionedCall¢!dense_948/StatefulPartitionedCall¢!dense_949/StatefulPartitionedCall¢!dense_950/StatefulPartitionedCall¢!dense_951/StatefulPartitionedCall¢!dense_952/StatefulPartitionedCall¢!dense_953/StatefulPartitionedCall¢!dense_954/StatefulPartitionedCall¢!dense_955/StatefulPartitionedCall¢!dense_956/StatefulPartitionedCallm
normalization_90/subSubinputsnormalization_90_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_90/SqrtSqrtnormalization_90_sqrt_x*
T0*
_output_shapes

:_
normalization_90/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_90/MaximumMaximumnormalization_90/Sqrt:y:0#normalization_90/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_90/truedivRealDivnormalization_90/sub:z:0normalization_90/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_945/StatefulPartitionedCallStatefulPartitionedCallnormalization_90/truediv:z:0dense_945_836753dense_945_836755*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_945_layer_call_and_return_conditional_losses_836752
/batch_normalization_855/StatefulPartitionedCallStatefulPartitionedCall*dense_945/StatefulPartitionedCall:output:0batch_normalization_855_836758batch_normalization_855_836760batch_normalization_855_836762batch_normalization_855_836764*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_835850ø
leaky_re_lu_855/PartitionedCallPartitionedCall8batch_normalization_855/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_855_layer_call_and_return_conditional_losses_836772
!dense_946/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_855/PartitionedCall:output:0dense_946_836785dense_946_836787*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_946_layer_call_and_return_conditional_losses_836784
/batch_normalization_856/StatefulPartitionedCallStatefulPartitionedCall*dense_946/StatefulPartitionedCall:output:0batch_normalization_856_836790batch_normalization_856_836792batch_normalization_856_836794batch_normalization_856_836796*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_835932ø
leaky_re_lu_856/PartitionedCallPartitionedCall8batch_normalization_856/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_856_layer_call_and_return_conditional_losses_836804
!dense_947/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_856/PartitionedCall:output:0dense_947_836817dense_947_836819*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_947_layer_call_and_return_conditional_losses_836816
/batch_normalization_857/StatefulPartitionedCallStatefulPartitionedCall*dense_947/StatefulPartitionedCall:output:0batch_normalization_857_836822batch_normalization_857_836824batch_normalization_857_836826batch_normalization_857_836828*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_836014ø
leaky_re_lu_857/PartitionedCallPartitionedCall8batch_normalization_857/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_857_layer_call_and_return_conditional_losses_836836
!dense_948/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_857/PartitionedCall:output:0dense_948_836849dense_948_836851*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_948_layer_call_and_return_conditional_losses_836848
/batch_normalization_858/StatefulPartitionedCallStatefulPartitionedCall*dense_948/StatefulPartitionedCall:output:0batch_normalization_858_836854batch_normalization_858_836856batch_normalization_858_836858batch_normalization_858_836860*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_836096ø
leaky_re_lu_858/PartitionedCallPartitionedCall8batch_normalization_858/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_858_layer_call_and_return_conditional_losses_836868
!dense_949/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_858/PartitionedCall:output:0dense_949_836881dense_949_836883*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_949_layer_call_and_return_conditional_losses_836880
/batch_normalization_859/StatefulPartitionedCallStatefulPartitionedCall*dense_949/StatefulPartitionedCall:output:0batch_normalization_859_836886batch_normalization_859_836888batch_normalization_859_836890batch_normalization_859_836892*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_836178ø
leaky_re_lu_859/PartitionedCallPartitionedCall8batch_normalization_859/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_859_layer_call_and_return_conditional_losses_836900
!dense_950/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_859/PartitionedCall:output:0dense_950_836913dense_950_836915*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_950_layer_call_and_return_conditional_losses_836912
/batch_normalization_860/StatefulPartitionedCallStatefulPartitionedCall*dense_950/StatefulPartitionedCall:output:0batch_normalization_860_836918batch_normalization_860_836920batch_normalization_860_836922batch_normalization_860_836924*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_836260ø
leaky_re_lu_860/PartitionedCallPartitionedCall8batch_normalization_860/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_860_layer_call_and_return_conditional_losses_836932
!dense_951/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_860/PartitionedCall:output:0dense_951_836945dense_951_836947*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_951_layer_call_and_return_conditional_losses_836944
/batch_normalization_861/StatefulPartitionedCallStatefulPartitionedCall*dense_951/StatefulPartitionedCall:output:0batch_normalization_861_836950batch_normalization_861_836952batch_normalization_861_836954batch_normalization_861_836956*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_836342ø
leaky_re_lu_861/PartitionedCallPartitionedCall8batch_normalization_861/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_861_layer_call_and_return_conditional_losses_836964
!dense_952/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_861/PartitionedCall:output:0dense_952_836977dense_952_836979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_952_layer_call_and_return_conditional_losses_836976
/batch_normalization_862/StatefulPartitionedCallStatefulPartitionedCall*dense_952/StatefulPartitionedCall:output:0batch_normalization_862_836982batch_normalization_862_836984batch_normalization_862_836986batch_normalization_862_836988*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_836424ø
leaky_re_lu_862/PartitionedCallPartitionedCall8batch_normalization_862/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_862_layer_call_and_return_conditional_losses_836996
!dense_953/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_862/PartitionedCall:output:0dense_953_837009dense_953_837011*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_953_layer_call_and_return_conditional_losses_837008
/batch_normalization_863/StatefulPartitionedCallStatefulPartitionedCall*dense_953/StatefulPartitionedCall:output:0batch_normalization_863_837014batch_normalization_863_837016batch_normalization_863_837018batch_normalization_863_837020*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_836506ø
leaky_re_lu_863/PartitionedCallPartitionedCall8batch_normalization_863/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_863_layer_call_and_return_conditional_losses_837028
!dense_954/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_863/PartitionedCall:output:0dense_954_837041dense_954_837043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_954_layer_call_and_return_conditional_losses_837040
/batch_normalization_864/StatefulPartitionedCallStatefulPartitionedCall*dense_954/StatefulPartitionedCall:output:0batch_normalization_864_837046batch_normalization_864_837048batch_normalization_864_837050batch_normalization_864_837052*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_836588ø
leaky_re_lu_864/PartitionedCallPartitionedCall8batch_normalization_864/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_864_layer_call_and_return_conditional_losses_837060
!dense_955/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_864/PartitionedCall:output:0dense_955_837073dense_955_837075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_955_layer_call_and_return_conditional_losses_837072
/batch_normalization_865/StatefulPartitionedCallStatefulPartitionedCall*dense_955/StatefulPartitionedCall:output:0batch_normalization_865_837078batch_normalization_865_837080batch_normalization_865_837082batch_normalization_865_837084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_836670ø
leaky_re_lu_865/PartitionedCallPartitionedCall8batch_normalization_865/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_837092
!dense_956/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_865/PartitionedCall:output:0dense_956_837105dense_956_837107*
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
E__inference_dense_956_layer_call_and_return_conditional_losses_837104y
IdentityIdentity*dense_956/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_855/StatefulPartitionedCall0^batch_normalization_856/StatefulPartitionedCall0^batch_normalization_857/StatefulPartitionedCall0^batch_normalization_858/StatefulPartitionedCall0^batch_normalization_859/StatefulPartitionedCall0^batch_normalization_860/StatefulPartitionedCall0^batch_normalization_861/StatefulPartitionedCall0^batch_normalization_862/StatefulPartitionedCall0^batch_normalization_863/StatefulPartitionedCall0^batch_normalization_864/StatefulPartitionedCall0^batch_normalization_865/StatefulPartitionedCall"^dense_945/StatefulPartitionedCall"^dense_946/StatefulPartitionedCall"^dense_947/StatefulPartitionedCall"^dense_948/StatefulPartitionedCall"^dense_949/StatefulPartitionedCall"^dense_950/StatefulPartitionedCall"^dense_951/StatefulPartitionedCall"^dense_952/StatefulPartitionedCall"^dense_953/StatefulPartitionedCall"^dense_954/StatefulPartitionedCall"^dense_955/StatefulPartitionedCall"^dense_956/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_855/StatefulPartitionedCall/batch_normalization_855/StatefulPartitionedCall2b
/batch_normalization_856/StatefulPartitionedCall/batch_normalization_856/StatefulPartitionedCall2b
/batch_normalization_857/StatefulPartitionedCall/batch_normalization_857/StatefulPartitionedCall2b
/batch_normalization_858/StatefulPartitionedCall/batch_normalization_858/StatefulPartitionedCall2b
/batch_normalization_859/StatefulPartitionedCall/batch_normalization_859/StatefulPartitionedCall2b
/batch_normalization_860/StatefulPartitionedCall/batch_normalization_860/StatefulPartitionedCall2b
/batch_normalization_861/StatefulPartitionedCall/batch_normalization_861/StatefulPartitionedCall2b
/batch_normalization_862/StatefulPartitionedCall/batch_normalization_862/StatefulPartitionedCall2b
/batch_normalization_863/StatefulPartitionedCall/batch_normalization_863/StatefulPartitionedCall2b
/batch_normalization_864/StatefulPartitionedCall/batch_normalization_864/StatefulPartitionedCall2b
/batch_normalization_865/StatefulPartitionedCall/batch_normalization_865/StatefulPartitionedCall2F
!dense_945/StatefulPartitionedCall!dense_945/StatefulPartitionedCall2F
!dense_946/StatefulPartitionedCall!dense_946/StatefulPartitionedCall2F
!dense_947/StatefulPartitionedCall!dense_947/StatefulPartitionedCall2F
!dense_948/StatefulPartitionedCall!dense_948/StatefulPartitionedCall2F
!dense_949/StatefulPartitionedCall!dense_949/StatefulPartitionedCall2F
!dense_950/StatefulPartitionedCall!dense_950/StatefulPartitionedCall2F
!dense_951/StatefulPartitionedCall!dense_951/StatefulPartitionedCall2F
!dense_952/StatefulPartitionedCall!dense_952/StatefulPartitionedCall2F
!dense_953/StatefulPartitionedCall!dense_953/StatefulPartitionedCall2F
!dense_954/StatefulPartitionedCall!dense_954/StatefulPartitionedCall2F
!dense_955/StatefulPartitionedCall!dense_955/StatefulPartitionedCall2F
!dense_956/StatefulPartitionedCall!dense_956/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ä

*__inference_dense_945_layer_call_fn_839609

inputs
unknown:h
	unknown_0:h
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_945_layer_call_and_return_conditional_losses_836752o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
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
«
L
0__inference_leaky_re_lu_865_layer_call_fn_840794

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_837092`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_953_layer_call_fn_840481

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_953_layer_call_and_return_conditional_losses_837008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_856_layer_call_and_return_conditional_losses_839818

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_855_layer_call_fn_839704

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
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_855_layer_call_and_return_conditional_losses_836772`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ä

*__inference_dense_956_layer_call_fn_840808

inputs
unknown:
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
E__inference_dense_956_layer_call_and_return_conditional_losses_837104o
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
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ

$__inference_signature_wrapper_839553
normalization_90_input
unknown
	unknown_0
	unknown_1:h
	unknown_2:h
	unknown_3:h
	unknown_4:h
	unknown_5:h
	unknown_6:h
	unknown_7:hh
	unknown_8:h
	unknown_9:h

unknown_10:h

unknown_11:h

unknown_12:h

unknown_13:hh

unknown_14:h

unknown_15:h

unknown_16:h

unknown_17:h

unknown_18:h

unknown_19:hh

unknown_20:h

unknown_21:h

unknown_22:h

unknown_23:h

unknown_24:h

unknown_25:hh

unknown_26:h

unknown_27:h

unknown_28:h

unknown_29:h

unknown_30:h

unknown_31:h/

unknown_32:/

unknown_33:/

unknown_34:/

unknown_35:/

unknown_36:/

unknown_37://

unknown_38:/

unknown_39:/

unknown_40:/

unknown_41:/

unknown_42:/

unknown_43:/

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:

unknown_61:

unknown_62:

unknown_63:

unknown_64:

unknown_65:

unknown_66:

unknown_67:

unknown_68:
identity¢StatefulPartitionedCalló	
StatefulPartitionedCallStatefulPartitionedCallnormalization_90_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_835826o
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
_user_specified_namenormalization_90_input:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_837092

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ´

I__inference_sequential_90_layer_call_and_return_conditional_losses_838418
normalization_90_input
normalization_90_sub_y
normalization_90_sqrt_x"
dense_945_838247:h
dense_945_838249:h,
batch_normalization_855_838252:h,
batch_normalization_855_838254:h,
batch_normalization_855_838256:h,
batch_normalization_855_838258:h"
dense_946_838262:hh
dense_946_838264:h,
batch_normalization_856_838267:h,
batch_normalization_856_838269:h,
batch_normalization_856_838271:h,
batch_normalization_856_838273:h"
dense_947_838277:hh
dense_947_838279:h,
batch_normalization_857_838282:h,
batch_normalization_857_838284:h,
batch_normalization_857_838286:h,
batch_normalization_857_838288:h"
dense_948_838292:hh
dense_948_838294:h,
batch_normalization_858_838297:h,
batch_normalization_858_838299:h,
batch_normalization_858_838301:h,
batch_normalization_858_838303:h"
dense_949_838307:hh
dense_949_838309:h,
batch_normalization_859_838312:h,
batch_normalization_859_838314:h,
batch_normalization_859_838316:h,
batch_normalization_859_838318:h"
dense_950_838322:h/
dense_950_838324:/,
batch_normalization_860_838327:/,
batch_normalization_860_838329:/,
batch_normalization_860_838331:/,
batch_normalization_860_838333:/"
dense_951_838337://
dense_951_838339:/,
batch_normalization_861_838342:/,
batch_normalization_861_838344:/,
batch_normalization_861_838346:/,
batch_normalization_861_838348:/"
dense_952_838352:/
dense_952_838354:,
batch_normalization_862_838357:,
batch_normalization_862_838359:,
batch_normalization_862_838361:,
batch_normalization_862_838363:"
dense_953_838367:
dense_953_838369:,
batch_normalization_863_838372:,
batch_normalization_863_838374:,
batch_normalization_863_838376:,
batch_normalization_863_838378:"
dense_954_838382:
dense_954_838384:,
batch_normalization_864_838387:,
batch_normalization_864_838389:,
batch_normalization_864_838391:,
batch_normalization_864_838393:"
dense_955_838397:
dense_955_838399:,
batch_normalization_865_838402:,
batch_normalization_865_838404:,
batch_normalization_865_838406:,
batch_normalization_865_838408:"
dense_956_838412:
dense_956_838414:
identity¢/batch_normalization_855/StatefulPartitionedCall¢/batch_normalization_856/StatefulPartitionedCall¢/batch_normalization_857/StatefulPartitionedCall¢/batch_normalization_858/StatefulPartitionedCall¢/batch_normalization_859/StatefulPartitionedCall¢/batch_normalization_860/StatefulPartitionedCall¢/batch_normalization_861/StatefulPartitionedCall¢/batch_normalization_862/StatefulPartitionedCall¢/batch_normalization_863/StatefulPartitionedCall¢/batch_normalization_864/StatefulPartitionedCall¢/batch_normalization_865/StatefulPartitionedCall¢!dense_945/StatefulPartitionedCall¢!dense_946/StatefulPartitionedCall¢!dense_947/StatefulPartitionedCall¢!dense_948/StatefulPartitionedCall¢!dense_949/StatefulPartitionedCall¢!dense_950/StatefulPartitionedCall¢!dense_951/StatefulPartitionedCall¢!dense_952/StatefulPartitionedCall¢!dense_953/StatefulPartitionedCall¢!dense_954/StatefulPartitionedCall¢!dense_955/StatefulPartitionedCall¢!dense_956/StatefulPartitionedCall}
normalization_90/subSubnormalization_90_inputnormalization_90_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_90/SqrtSqrtnormalization_90_sqrt_x*
T0*
_output_shapes

:_
normalization_90/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_90/MaximumMaximumnormalization_90/Sqrt:y:0#normalization_90/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_90/truedivRealDivnormalization_90/sub:z:0normalization_90/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_945/StatefulPartitionedCallStatefulPartitionedCallnormalization_90/truediv:z:0dense_945_838247dense_945_838249*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_945_layer_call_and_return_conditional_losses_836752
/batch_normalization_855/StatefulPartitionedCallStatefulPartitionedCall*dense_945/StatefulPartitionedCall:output:0batch_normalization_855_838252batch_normalization_855_838254batch_normalization_855_838256batch_normalization_855_838258*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_835897ø
leaky_re_lu_855/PartitionedCallPartitionedCall8batch_normalization_855/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_855_layer_call_and_return_conditional_losses_836772
!dense_946/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_855/PartitionedCall:output:0dense_946_838262dense_946_838264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_946_layer_call_and_return_conditional_losses_836784
/batch_normalization_856/StatefulPartitionedCallStatefulPartitionedCall*dense_946/StatefulPartitionedCall:output:0batch_normalization_856_838267batch_normalization_856_838269batch_normalization_856_838271batch_normalization_856_838273*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_835979ø
leaky_re_lu_856/PartitionedCallPartitionedCall8batch_normalization_856/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_856_layer_call_and_return_conditional_losses_836804
!dense_947/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_856/PartitionedCall:output:0dense_947_838277dense_947_838279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_947_layer_call_and_return_conditional_losses_836816
/batch_normalization_857/StatefulPartitionedCallStatefulPartitionedCall*dense_947/StatefulPartitionedCall:output:0batch_normalization_857_838282batch_normalization_857_838284batch_normalization_857_838286batch_normalization_857_838288*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_836061ø
leaky_re_lu_857/PartitionedCallPartitionedCall8batch_normalization_857/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_857_layer_call_and_return_conditional_losses_836836
!dense_948/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_857/PartitionedCall:output:0dense_948_838292dense_948_838294*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_948_layer_call_and_return_conditional_losses_836848
/batch_normalization_858/StatefulPartitionedCallStatefulPartitionedCall*dense_948/StatefulPartitionedCall:output:0batch_normalization_858_838297batch_normalization_858_838299batch_normalization_858_838301batch_normalization_858_838303*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_836143ø
leaky_re_lu_858/PartitionedCallPartitionedCall8batch_normalization_858/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_858_layer_call_and_return_conditional_losses_836868
!dense_949/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_858/PartitionedCall:output:0dense_949_838307dense_949_838309*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_949_layer_call_and_return_conditional_losses_836880
/batch_normalization_859/StatefulPartitionedCallStatefulPartitionedCall*dense_949/StatefulPartitionedCall:output:0batch_normalization_859_838312batch_normalization_859_838314batch_normalization_859_838316batch_normalization_859_838318*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_836225ø
leaky_re_lu_859/PartitionedCallPartitionedCall8batch_normalization_859/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_859_layer_call_and_return_conditional_losses_836900
!dense_950/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_859/PartitionedCall:output:0dense_950_838322dense_950_838324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_950_layer_call_and_return_conditional_losses_836912
/batch_normalization_860/StatefulPartitionedCallStatefulPartitionedCall*dense_950/StatefulPartitionedCall:output:0batch_normalization_860_838327batch_normalization_860_838329batch_normalization_860_838331batch_normalization_860_838333*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_836307ø
leaky_re_lu_860/PartitionedCallPartitionedCall8batch_normalization_860/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_860_layer_call_and_return_conditional_losses_836932
!dense_951/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_860/PartitionedCall:output:0dense_951_838337dense_951_838339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_951_layer_call_and_return_conditional_losses_836944
/batch_normalization_861/StatefulPartitionedCallStatefulPartitionedCall*dense_951/StatefulPartitionedCall:output:0batch_normalization_861_838342batch_normalization_861_838344batch_normalization_861_838346batch_normalization_861_838348*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_836389ø
leaky_re_lu_861/PartitionedCallPartitionedCall8batch_normalization_861/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_861_layer_call_and_return_conditional_losses_836964
!dense_952/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_861/PartitionedCall:output:0dense_952_838352dense_952_838354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_952_layer_call_and_return_conditional_losses_836976
/batch_normalization_862/StatefulPartitionedCallStatefulPartitionedCall*dense_952/StatefulPartitionedCall:output:0batch_normalization_862_838357batch_normalization_862_838359batch_normalization_862_838361batch_normalization_862_838363*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_836471ø
leaky_re_lu_862/PartitionedCallPartitionedCall8batch_normalization_862/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_862_layer_call_and_return_conditional_losses_836996
!dense_953/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_862/PartitionedCall:output:0dense_953_838367dense_953_838369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_953_layer_call_and_return_conditional_losses_837008
/batch_normalization_863/StatefulPartitionedCallStatefulPartitionedCall*dense_953/StatefulPartitionedCall:output:0batch_normalization_863_838372batch_normalization_863_838374batch_normalization_863_838376batch_normalization_863_838378*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_836553ø
leaky_re_lu_863/PartitionedCallPartitionedCall8batch_normalization_863/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_863_layer_call_and_return_conditional_losses_837028
!dense_954/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_863/PartitionedCall:output:0dense_954_838382dense_954_838384*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_954_layer_call_and_return_conditional_losses_837040
/batch_normalization_864/StatefulPartitionedCallStatefulPartitionedCall*dense_954/StatefulPartitionedCall:output:0batch_normalization_864_838387batch_normalization_864_838389batch_normalization_864_838391batch_normalization_864_838393*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_836635ø
leaky_re_lu_864/PartitionedCallPartitionedCall8batch_normalization_864/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_864_layer_call_and_return_conditional_losses_837060
!dense_955/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_864/PartitionedCall:output:0dense_955_838397dense_955_838399*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_955_layer_call_and_return_conditional_losses_837072
/batch_normalization_865/StatefulPartitionedCallStatefulPartitionedCall*dense_955/StatefulPartitionedCall:output:0batch_normalization_865_838402batch_normalization_865_838404batch_normalization_865_838406batch_normalization_865_838408*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_836717ø
leaky_re_lu_865/PartitionedCallPartitionedCall8batch_normalization_865/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_837092
!dense_956/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_865/PartitionedCall:output:0dense_956_838412dense_956_838414*
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
E__inference_dense_956_layer_call_and_return_conditional_losses_837104y
IdentityIdentity*dense_956/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_855/StatefulPartitionedCall0^batch_normalization_856/StatefulPartitionedCall0^batch_normalization_857/StatefulPartitionedCall0^batch_normalization_858/StatefulPartitionedCall0^batch_normalization_859/StatefulPartitionedCall0^batch_normalization_860/StatefulPartitionedCall0^batch_normalization_861/StatefulPartitionedCall0^batch_normalization_862/StatefulPartitionedCall0^batch_normalization_863/StatefulPartitionedCall0^batch_normalization_864/StatefulPartitionedCall0^batch_normalization_865/StatefulPartitionedCall"^dense_945/StatefulPartitionedCall"^dense_946/StatefulPartitionedCall"^dense_947/StatefulPartitionedCall"^dense_948/StatefulPartitionedCall"^dense_949/StatefulPartitionedCall"^dense_950/StatefulPartitionedCall"^dense_951/StatefulPartitionedCall"^dense_952/StatefulPartitionedCall"^dense_953/StatefulPartitionedCall"^dense_954/StatefulPartitionedCall"^dense_955/StatefulPartitionedCall"^dense_956/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_855/StatefulPartitionedCall/batch_normalization_855/StatefulPartitionedCall2b
/batch_normalization_856/StatefulPartitionedCall/batch_normalization_856/StatefulPartitionedCall2b
/batch_normalization_857/StatefulPartitionedCall/batch_normalization_857/StatefulPartitionedCall2b
/batch_normalization_858/StatefulPartitionedCall/batch_normalization_858/StatefulPartitionedCall2b
/batch_normalization_859/StatefulPartitionedCall/batch_normalization_859/StatefulPartitionedCall2b
/batch_normalization_860/StatefulPartitionedCall/batch_normalization_860/StatefulPartitionedCall2b
/batch_normalization_861/StatefulPartitionedCall/batch_normalization_861/StatefulPartitionedCall2b
/batch_normalization_862/StatefulPartitionedCall/batch_normalization_862/StatefulPartitionedCall2b
/batch_normalization_863/StatefulPartitionedCall/batch_normalization_863/StatefulPartitionedCall2b
/batch_normalization_864/StatefulPartitionedCall/batch_normalization_864/StatefulPartitionedCall2b
/batch_normalization_865/StatefulPartitionedCall/batch_normalization_865/StatefulPartitionedCall2F
!dense_945/StatefulPartitionedCall!dense_945/StatefulPartitionedCall2F
!dense_946/StatefulPartitionedCall!dense_946/StatefulPartitionedCall2F
!dense_947/StatefulPartitionedCall!dense_947/StatefulPartitionedCall2F
!dense_948/StatefulPartitionedCall!dense_948/StatefulPartitionedCall2F
!dense_949/StatefulPartitionedCall!dense_949/StatefulPartitionedCall2F
!dense_950/StatefulPartitionedCall!dense_950/StatefulPartitionedCall2F
!dense_951/StatefulPartitionedCall!dense_951/StatefulPartitionedCall2F
!dense_952/StatefulPartitionedCall!dense_952/StatefulPartitionedCall2F
!dense_953/StatefulPartitionedCall!dense_953/StatefulPartitionedCall2F
!dense_954/StatefulPartitionedCall!dense_954/StatefulPartitionedCall2F
!dense_955/StatefulPartitionedCall!dense_955/StatefulPartitionedCall2F
!dense_956/StatefulPartitionedCall!dense_956/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_90_input:$ 

_output_shapes

::$ 

_output_shapes

:
«
L
0__inference_leaky_re_lu_858_layer_call_fn_840031

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
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_858_layer_call_and_return_conditional_losses_836868`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
ë´

I__inference_sequential_90_layer_call_and_return_conditional_losses_838237
normalization_90_input
normalization_90_sub_y
normalization_90_sqrt_x"
dense_945_838066:h
dense_945_838068:h,
batch_normalization_855_838071:h,
batch_normalization_855_838073:h,
batch_normalization_855_838075:h,
batch_normalization_855_838077:h"
dense_946_838081:hh
dense_946_838083:h,
batch_normalization_856_838086:h,
batch_normalization_856_838088:h,
batch_normalization_856_838090:h,
batch_normalization_856_838092:h"
dense_947_838096:hh
dense_947_838098:h,
batch_normalization_857_838101:h,
batch_normalization_857_838103:h,
batch_normalization_857_838105:h,
batch_normalization_857_838107:h"
dense_948_838111:hh
dense_948_838113:h,
batch_normalization_858_838116:h,
batch_normalization_858_838118:h,
batch_normalization_858_838120:h,
batch_normalization_858_838122:h"
dense_949_838126:hh
dense_949_838128:h,
batch_normalization_859_838131:h,
batch_normalization_859_838133:h,
batch_normalization_859_838135:h,
batch_normalization_859_838137:h"
dense_950_838141:h/
dense_950_838143:/,
batch_normalization_860_838146:/,
batch_normalization_860_838148:/,
batch_normalization_860_838150:/,
batch_normalization_860_838152:/"
dense_951_838156://
dense_951_838158:/,
batch_normalization_861_838161:/,
batch_normalization_861_838163:/,
batch_normalization_861_838165:/,
batch_normalization_861_838167:/"
dense_952_838171:/
dense_952_838173:,
batch_normalization_862_838176:,
batch_normalization_862_838178:,
batch_normalization_862_838180:,
batch_normalization_862_838182:"
dense_953_838186:
dense_953_838188:,
batch_normalization_863_838191:,
batch_normalization_863_838193:,
batch_normalization_863_838195:,
batch_normalization_863_838197:"
dense_954_838201:
dense_954_838203:,
batch_normalization_864_838206:,
batch_normalization_864_838208:,
batch_normalization_864_838210:,
batch_normalization_864_838212:"
dense_955_838216:
dense_955_838218:,
batch_normalization_865_838221:,
batch_normalization_865_838223:,
batch_normalization_865_838225:,
batch_normalization_865_838227:"
dense_956_838231:
dense_956_838233:
identity¢/batch_normalization_855/StatefulPartitionedCall¢/batch_normalization_856/StatefulPartitionedCall¢/batch_normalization_857/StatefulPartitionedCall¢/batch_normalization_858/StatefulPartitionedCall¢/batch_normalization_859/StatefulPartitionedCall¢/batch_normalization_860/StatefulPartitionedCall¢/batch_normalization_861/StatefulPartitionedCall¢/batch_normalization_862/StatefulPartitionedCall¢/batch_normalization_863/StatefulPartitionedCall¢/batch_normalization_864/StatefulPartitionedCall¢/batch_normalization_865/StatefulPartitionedCall¢!dense_945/StatefulPartitionedCall¢!dense_946/StatefulPartitionedCall¢!dense_947/StatefulPartitionedCall¢!dense_948/StatefulPartitionedCall¢!dense_949/StatefulPartitionedCall¢!dense_950/StatefulPartitionedCall¢!dense_951/StatefulPartitionedCall¢!dense_952/StatefulPartitionedCall¢!dense_953/StatefulPartitionedCall¢!dense_954/StatefulPartitionedCall¢!dense_955/StatefulPartitionedCall¢!dense_956/StatefulPartitionedCall}
normalization_90/subSubnormalization_90_inputnormalization_90_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_90/SqrtSqrtnormalization_90_sqrt_x*
T0*
_output_shapes

:_
normalization_90/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_90/MaximumMaximumnormalization_90/Sqrt:y:0#normalization_90/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_90/truedivRealDivnormalization_90/sub:z:0normalization_90/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_945/StatefulPartitionedCallStatefulPartitionedCallnormalization_90/truediv:z:0dense_945_838066dense_945_838068*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_945_layer_call_and_return_conditional_losses_836752
/batch_normalization_855/StatefulPartitionedCallStatefulPartitionedCall*dense_945/StatefulPartitionedCall:output:0batch_normalization_855_838071batch_normalization_855_838073batch_normalization_855_838075batch_normalization_855_838077*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_835850ø
leaky_re_lu_855/PartitionedCallPartitionedCall8batch_normalization_855/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_855_layer_call_and_return_conditional_losses_836772
!dense_946/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_855/PartitionedCall:output:0dense_946_838081dense_946_838083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_946_layer_call_and_return_conditional_losses_836784
/batch_normalization_856/StatefulPartitionedCallStatefulPartitionedCall*dense_946/StatefulPartitionedCall:output:0batch_normalization_856_838086batch_normalization_856_838088batch_normalization_856_838090batch_normalization_856_838092*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_835932ø
leaky_re_lu_856/PartitionedCallPartitionedCall8batch_normalization_856/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_856_layer_call_and_return_conditional_losses_836804
!dense_947/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_856/PartitionedCall:output:0dense_947_838096dense_947_838098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_947_layer_call_and_return_conditional_losses_836816
/batch_normalization_857/StatefulPartitionedCallStatefulPartitionedCall*dense_947/StatefulPartitionedCall:output:0batch_normalization_857_838101batch_normalization_857_838103batch_normalization_857_838105batch_normalization_857_838107*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_836014ø
leaky_re_lu_857/PartitionedCallPartitionedCall8batch_normalization_857/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_857_layer_call_and_return_conditional_losses_836836
!dense_948/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_857/PartitionedCall:output:0dense_948_838111dense_948_838113*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_948_layer_call_and_return_conditional_losses_836848
/batch_normalization_858/StatefulPartitionedCallStatefulPartitionedCall*dense_948/StatefulPartitionedCall:output:0batch_normalization_858_838116batch_normalization_858_838118batch_normalization_858_838120batch_normalization_858_838122*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_836096ø
leaky_re_lu_858/PartitionedCallPartitionedCall8batch_normalization_858/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_858_layer_call_and_return_conditional_losses_836868
!dense_949/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_858/PartitionedCall:output:0dense_949_838126dense_949_838128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_949_layer_call_and_return_conditional_losses_836880
/batch_normalization_859/StatefulPartitionedCallStatefulPartitionedCall*dense_949/StatefulPartitionedCall:output:0batch_normalization_859_838131batch_normalization_859_838133batch_normalization_859_838135batch_normalization_859_838137*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_836178ø
leaky_re_lu_859/PartitionedCallPartitionedCall8batch_normalization_859/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_859_layer_call_and_return_conditional_losses_836900
!dense_950/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_859/PartitionedCall:output:0dense_950_838141dense_950_838143*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_950_layer_call_and_return_conditional_losses_836912
/batch_normalization_860/StatefulPartitionedCallStatefulPartitionedCall*dense_950/StatefulPartitionedCall:output:0batch_normalization_860_838146batch_normalization_860_838148batch_normalization_860_838150batch_normalization_860_838152*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_836260ø
leaky_re_lu_860/PartitionedCallPartitionedCall8batch_normalization_860/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_860_layer_call_and_return_conditional_losses_836932
!dense_951/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_860/PartitionedCall:output:0dense_951_838156dense_951_838158*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_951_layer_call_and_return_conditional_losses_836944
/batch_normalization_861/StatefulPartitionedCallStatefulPartitionedCall*dense_951/StatefulPartitionedCall:output:0batch_normalization_861_838161batch_normalization_861_838163batch_normalization_861_838165batch_normalization_861_838167*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_836342ø
leaky_re_lu_861/PartitionedCallPartitionedCall8batch_normalization_861/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_861_layer_call_and_return_conditional_losses_836964
!dense_952/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_861/PartitionedCall:output:0dense_952_838171dense_952_838173*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_952_layer_call_and_return_conditional_losses_836976
/batch_normalization_862/StatefulPartitionedCallStatefulPartitionedCall*dense_952/StatefulPartitionedCall:output:0batch_normalization_862_838176batch_normalization_862_838178batch_normalization_862_838180batch_normalization_862_838182*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_836424ø
leaky_re_lu_862/PartitionedCallPartitionedCall8batch_normalization_862/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_862_layer_call_and_return_conditional_losses_836996
!dense_953/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_862/PartitionedCall:output:0dense_953_838186dense_953_838188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_953_layer_call_and_return_conditional_losses_837008
/batch_normalization_863/StatefulPartitionedCallStatefulPartitionedCall*dense_953/StatefulPartitionedCall:output:0batch_normalization_863_838191batch_normalization_863_838193batch_normalization_863_838195batch_normalization_863_838197*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_836506ø
leaky_re_lu_863/PartitionedCallPartitionedCall8batch_normalization_863/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_863_layer_call_and_return_conditional_losses_837028
!dense_954/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_863/PartitionedCall:output:0dense_954_838201dense_954_838203*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_954_layer_call_and_return_conditional_losses_837040
/batch_normalization_864/StatefulPartitionedCallStatefulPartitionedCall*dense_954/StatefulPartitionedCall:output:0batch_normalization_864_838206batch_normalization_864_838208batch_normalization_864_838210batch_normalization_864_838212*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_836588ø
leaky_re_lu_864/PartitionedCallPartitionedCall8batch_normalization_864/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_864_layer_call_and_return_conditional_losses_837060
!dense_955/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_864/PartitionedCall:output:0dense_955_838216dense_955_838218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_955_layer_call_and_return_conditional_losses_837072
/batch_normalization_865/StatefulPartitionedCallStatefulPartitionedCall*dense_955/StatefulPartitionedCall:output:0batch_normalization_865_838221batch_normalization_865_838223batch_normalization_865_838225batch_normalization_865_838227*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_836670ø
leaky_re_lu_865/PartitionedCallPartitionedCall8batch_normalization_865/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_837092
!dense_956/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_865/PartitionedCall:output:0dense_956_838231dense_956_838233*
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
E__inference_dense_956_layer_call_and_return_conditional_losses_837104y
IdentityIdentity*dense_956/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_855/StatefulPartitionedCall0^batch_normalization_856/StatefulPartitionedCall0^batch_normalization_857/StatefulPartitionedCall0^batch_normalization_858/StatefulPartitionedCall0^batch_normalization_859/StatefulPartitionedCall0^batch_normalization_860/StatefulPartitionedCall0^batch_normalization_861/StatefulPartitionedCall0^batch_normalization_862/StatefulPartitionedCall0^batch_normalization_863/StatefulPartitionedCall0^batch_normalization_864/StatefulPartitionedCall0^batch_normalization_865/StatefulPartitionedCall"^dense_945/StatefulPartitionedCall"^dense_946/StatefulPartitionedCall"^dense_947/StatefulPartitionedCall"^dense_948/StatefulPartitionedCall"^dense_949/StatefulPartitionedCall"^dense_950/StatefulPartitionedCall"^dense_951/StatefulPartitionedCall"^dense_952/StatefulPartitionedCall"^dense_953/StatefulPartitionedCall"^dense_954/StatefulPartitionedCall"^dense_955/StatefulPartitionedCall"^dense_956/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_855/StatefulPartitionedCall/batch_normalization_855/StatefulPartitionedCall2b
/batch_normalization_856/StatefulPartitionedCall/batch_normalization_856/StatefulPartitionedCall2b
/batch_normalization_857/StatefulPartitionedCall/batch_normalization_857/StatefulPartitionedCall2b
/batch_normalization_858/StatefulPartitionedCall/batch_normalization_858/StatefulPartitionedCall2b
/batch_normalization_859/StatefulPartitionedCall/batch_normalization_859/StatefulPartitionedCall2b
/batch_normalization_860/StatefulPartitionedCall/batch_normalization_860/StatefulPartitionedCall2b
/batch_normalization_861/StatefulPartitionedCall/batch_normalization_861/StatefulPartitionedCall2b
/batch_normalization_862/StatefulPartitionedCall/batch_normalization_862/StatefulPartitionedCall2b
/batch_normalization_863/StatefulPartitionedCall/batch_normalization_863/StatefulPartitionedCall2b
/batch_normalization_864/StatefulPartitionedCall/batch_normalization_864/StatefulPartitionedCall2b
/batch_normalization_865/StatefulPartitionedCall/batch_normalization_865/StatefulPartitionedCall2F
!dense_945/StatefulPartitionedCall!dense_945/StatefulPartitionedCall2F
!dense_946/StatefulPartitionedCall!dense_946/StatefulPartitionedCall2F
!dense_947/StatefulPartitionedCall!dense_947/StatefulPartitionedCall2F
!dense_948/StatefulPartitionedCall!dense_948/StatefulPartitionedCall2F
!dense_949/StatefulPartitionedCall!dense_949/StatefulPartitionedCall2F
!dense_950/StatefulPartitionedCall!dense_950/StatefulPartitionedCall2F
!dense_951/StatefulPartitionedCall!dense_951/StatefulPartitionedCall2F
!dense_952/StatefulPartitionedCall!dense_952/StatefulPartitionedCall2F
!dense_953/StatefulPartitionedCall!dense_953/StatefulPartitionedCall2F
!dense_954/StatefulPartitionedCall!dense_954/StatefulPartitionedCall2F
!dense_955/StatefulPartitionedCall!dense_955/StatefulPartitionedCall2F
!dense_956/StatefulPartitionedCall!dense_956/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_90_input:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_859_layer_call_and_return_conditional_losses_840145

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_859_layer_call_fn_840140

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
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_859_layer_call_and_return_conditional_losses_836900`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_864_layer_call_and_return_conditional_losses_837060

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_857_layer_call_fn_839922

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
:ÿÿÿÿÿÿÿÿÿh* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_857_layer_call_and_return_conditional_losses_836836`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_862_layer_call_fn_840467

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_862_layer_call_and_return_conditional_losses_836996`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú§
çG
I__inference_sequential_90_layer_call_and_return_conditional_losses_839406

inputs
normalization_90_sub_y
normalization_90_sqrt_x:
(dense_945_matmul_readvariableop_resource:h7
)dense_945_biasadd_readvariableop_resource:hM
?batch_normalization_855_assignmovingavg_readvariableop_resource:hO
Abatch_normalization_855_assignmovingavg_1_readvariableop_resource:hK
=batch_normalization_855_batchnorm_mul_readvariableop_resource:hG
9batch_normalization_855_batchnorm_readvariableop_resource:h:
(dense_946_matmul_readvariableop_resource:hh7
)dense_946_biasadd_readvariableop_resource:hM
?batch_normalization_856_assignmovingavg_readvariableop_resource:hO
Abatch_normalization_856_assignmovingavg_1_readvariableop_resource:hK
=batch_normalization_856_batchnorm_mul_readvariableop_resource:hG
9batch_normalization_856_batchnorm_readvariableop_resource:h:
(dense_947_matmul_readvariableop_resource:hh7
)dense_947_biasadd_readvariableop_resource:hM
?batch_normalization_857_assignmovingavg_readvariableop_resource:hO
Abatch_normalization_857_assignmovingavg_1_readvariableop_resource:hK
=batch_normalization_857_batchnorm_mul_readvariableop_resource:hG
9batch_normalization_857_batchnorm_readvariableop_resource:h:
(dense_948_matmul_readvariableop_resource:hh7
)dense_948_biasadd_readvariableop_resource:hM
?batch_normalization_858_assignmovingavg_readvariableop_resource:hO
Abatch_normalization_858_assignmovingavg_1_readvariableop_resource:hK
=batch_normalization_858_batchnorm_mul_readvariableop_resource:hG
9batch_normalization_858_batchnorm_readvariableop_resource:h:
(dense_949_matmul_readvariableop_resource:hh7
)dense_949_biasadd_readvariableop_resource:hM
?batch_normalization_859_assignmovingavg_readvariableop_resource:hO
Abatch_normalization_859_assignmovingavg_1_readvariableop_resource:hK
=batch_normalization_859_batchnorm_mul_readvariableop_resource:hG
9batch_normalization_859_batchnorm_readvariableop_resource:h:
(dense_950_matmul_readvariableop_resource:h/7
)dense_950_biasadd_readvariableop_resource:/M
?batch_normalization_860_assignmovingavg_readvariableop_resource:/O
Abatch_normalization_860_assignmovingavg_1_readvariableop_resource:/K
=batch_normalization_860_batchnorm_mul_readvariableop_resource:/G
9batch_normalization_860_batchnorm_readvariableop_resource:/:
(dense_951_matmul_readvariableop_resource://7
)dense_951_biasadd_readvariableop_resource:/M
?batch_normalization_861_assignmovingavg_readvariableop_resource:/O
Abatch_normalization_861_assignmovingavg_1_readvariableop_resource:/K
=batch_normalization_861_batchnorm_mul_readvariableop_resource:/G
9batch_normalization_861_batchnorm_readvariableop_resource:/:
(dense_952_matmul_readvariableop_resource:/7
)dense_952_biasadd_readvariableop_resource:M
?batch_normalization_862_assignmovingavg_readvariableop_resource:O
Abatch_normalization_862_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_862_batchnorm_mul_readvariableop_resource:G
9batch_normalization_862_batchnorm_readvariableop_resource::
(dense_953_matmul_readvariableop_resource:7
)dense_953_biasadd_readvariableop_resource:M
?batch_normalization_863_assignmovingavg_readvariableop_resource:O
Abatch_normalization_863_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_863_batchnorm_mul_readvariableop_resource:G
9batch_normalization_863_batchnorm_readvariableop_resource::
(dense_954_matmul_readvariableop_resource:7
)dense_954_biasadd_readvariableop_resource:M
?batch_normalization_864_assignmovingavg_readvariableop_resource:O
Abatch_normalization_864_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_864_batchnorm_mul_readvariableop_resource:G
9batch_normalization_864_batchnorm_readvariableop_resource::
(dense_955_matmul_readvariableop_resource:7
)dense_955_biasadd_readvariableop_resource:M
?batch_normalization_865_assignmovingavg_readvariableop_resource:O
Abatch_normalization_865_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_865_batchnorm_mul_readvariableop_resource:G
9batch_normalization_865_batchnorm_readvariableop_resource::
(dense_956_matmul_readvariableop_resource:7
)dense_956_biasadd_readvariableop_resource:
identity¢'batch_normalization_855/AssignMovingAvg¢6batch_normalization_855/AssignMovingAvg/ReadVariableOp¢)batch_normalization_855/AssignMovingAvg_1¢8batch_normalization_855/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_855/batchnorm/ReadVariableOp¢4batch_normalization_855/batchnorm/mul/ReadVariableOp¢'batch_normalization_856/AssignMovingAvg¢6batch_normalization_856/AssignMovingAvg/ReadVariableOp¢)batch_normalization_856/AssignMovingAvg_1¢8batch_normalization_856/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_856/batchnorm/ReadVariableOp¢4batch_normalization_856/batchnorm/mul/ReadVariableOp¢'batch_normalization_857/AssignMovingAvg¢6batch_normalization_857/AssignMovingAvg/ReadVariableOp¢)batch_normalization_857/AssignMovingAvg_1¢8batch_normalization_857/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_857/batchnorm/ReadVariableOp¢4batch_normalization_857/batchnorm/mul/ReadVariableOp¢'batch_normalization_858/AssignMovingAvg¢6batch_normalization_858/AssignMovingAvg/ReadVariableOp¢)batch_normalization_858/AssignMovingAvg_1¢8batch_normalization_858/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_858/batchnorm/ReadVariableOp¢4batch_normalization_858/batchnorm/mul/ReadVariableOp¢'batch_normalization_859/AssignMovingAvg¢6batch_normalization_859/AssignMovingAvg/ReadVariableOp¢)batch_normalization_859/AssignMovingAvg_1¢8batch_normalization_859/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_859/batchnorm/ReadVariableOp¢4batch_normalization_859/batchnorm/mul/ReadVariableOp¢'batch_normalization_860/AssignMovingAvg¢6batch_normalization_860/AssignMovingAvg/ReadVariableOp¢)batch_normalization_860/AssignMovingAvg_1¢8batch_normalization_860/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_860/batchnorm/ReadVariableOp¢4batch_normalization_860/batchnorm/mul/ReadVariableOp¢'batch_normalization_861/AssignMovingAvg¢6batch_normalization_861/AssignMovingAvg/ReadVariableOp¢)batch_normalization_861/AssignMovingAvg_1¢8batch_normalization_861/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_861/batchnorm/ReadVariableOp¢4batch_normalization_861/batchnorm/mul/ReadVariableOp¢'batch_normalization_862/AssignMovingAvg¢6batch_normalization_862/AssignMovingAvg/ReadVariableOp¢)batch_normalization_862/AssignMovingAvg_1¢8batch_normalization_862/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_862/batchnorm/ReadVariableOp¢4batch_normalization_862/batchnorm/mul/ReadVariableOp¢'batch_normalization_863/AssignMovingAvg¢6batch_normalization_863/AssignMovingAvg/ReadVariableOp¢)batch_normalization_863/AssignMovingAvg_1¢8batch_normalization_863/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_863/batchnorm/ReadVariableOp¢4batch_normalization_863/batchnorm/mul/ReadVariableOp¢'batch_normalization_864/AssignMovingAvg¢6batch_normalization_864/AssignMovingAvg/ReadVariableOp¢)batch_normalization_864/AssignMovingAvg_1¢8batch_normalization_864/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_864/batchnorm/ReadVariableOp¢4batch_normalization_864/batchnorm/mul/ReadVariableOp¢'batch_normalization_865/AssignMovingAvg¢6batch_normalization_865/AssignMovingAvg/ReadVariableOp¢)batch_normalization_865/AssignMovingAvg_1¢8batch_normalization_865/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_865/batchnorm/ReadVariableOp¢4batch_normalization_865/batchnorm/mul/ReadVariableOp¢ dense_945/BiasAdd/ReadVariableOp¢dense_945/MatMul/ReadVariableOp¢ dense_946/BiasAdd/ReadVariableOp¢dense_946/MatMul/ReadVariableOp¢ dense_947/BiasAdd/ReadVariableOp¢dense_947/MatMul/ReadVariableOp¢ dense_948/BiasAdd/ReadVariableOp¢dense_948/MatMul/ReadVariableOp¢ dense_949/BiasAdd/ReadVariableOp¢dense_949/MatMul/ReadVariableOp¢ dense_950/BiasAdd/ReadVariableOp¢dense_950/MatMul/ReadVariableOp¢ dense_951/BiasAdd/ReadVariableOp¢dense_951/MatMul/ReadVariableOp¢ dense_952/BiasAdd/ReadVariableOp¢dense_952/MatMul/ReadVariableOp¢ dense_953/BiasAdd/ReadVariableOp¢dense_953/MatMul/ReadVariableOp¢ dense_954/BiasAdd/ReadVariableOp¢dense_954/MatMul/ReadVariableOp¢ dense_955/BiasAdd/ReadVariableOp¢dense_955/MatMul/ReadVariableOp¢ dense_956/BiasAdd/ReadVariableOp¢dense_956/MatMul/ReadVariableOpm
normalization_90/subSubinputsnormalization_90_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_90/SqrtSqrtnormalization_90_sqrt_x*
T0*
_output_shapes

:_
normalization_90/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_90/MaximumMaximumnormalization_90/Sqrt:y:0#normalization_90/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_90/truedivRealDivnormalization_90/sub:z:0normalization_90/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_945/MatMul/ReadVariableOpReadVariableOp(dense_945_matmul_readvariableop_resource*
_output_shapes

:h*
dtype0
dense_945/MatMulMatMulnormalization_90/truediv:z:0'dense_945/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 dense_945/BiasAdd/ReadVariableOpReadVariableOp)dense_945_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0
dense_945/BiasAddBiasAdddense_945/MatMul:product:0(dense_945/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
6batch_normalization_855/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_855/moments/meanMeandense_945/BiasAdd:output:0?batch_normalization_855/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(
,batch_normalization_855/moments/StopGradientStopGradient-batch_normalization_855/moments/mean:output:0*
T0*
_output_shapes

:hË
1batch_normalization_855/moments/SquaredDifferenceSquaredDifferencedense_945/BiasAdd:output:05batch_normalization_855/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
:batch_normalization_855/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_855/moments/varianceMean5batch_normalization_855/moments/SquaredDifference:z:0Cbatch_normalization_855/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(
'batch_normalization_855/moments/SqueezeSqueeze-batch_normalization_855/moments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 £
)batch_normalization_855/moments/Squeeze_1Squeeze1batch_normalization_855/moments/variance:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 r
-batch_normalization_855/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_855/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_855_assignmovingavg_readvariableop_resource*
_output_shapes
:h*
dtype0É
+batch_normalization_855/AssignMovingAvg/subSub>batch_normalization_855/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_855/moments/Squeeze:output:0*
T0*
_output_shapes
:hÀ
+batch_normalization_855/AssignMovingAvg/mulMul/batch_normalization_855/AssignMovingAvg/sub:z:06batch_normalization_855/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h
'batch_normalization_855/AssignMovingAvgAssignSubVariableOp?batch_normalization_855_assignmovingavg_readvariableop_resource/batch_normalization_855/AssignMovingAvg/mul:z:07^batch_normalization_855/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_855/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_855/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_855_assignmovingavg_1_readvariableop_resource*
_output_shapes
:h*
dtype0Ï
-batch_normalization_855/AssignMovingAvg_1/subSub@batch_normalization_855/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_855/moments/Squeeze_1:output:0*
T0*
_output_shapes
:hÆ
-batch_normalization_855/AssignMovingAvg_1/mulMul1batch_normalization_855/AssignMovingAvg_1/sub:z:08batch_normalization_855/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h
)batch_normalization_855/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_855_assignmovingavg_1_readvariableop_resource1batch_normalization_855/AssignMovingAvg_1/mul:z:09^batch_normalization_855/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_855/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_855/batchnorm/addAddV22batch_normalization_855/moments/Squeeze_1:output:00batch_normalization_855/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
'batch_normalization_855/batchnorm/RsqrtRsqrt)batch_normalization_855/batchnorm/add:z:0*
T0*
_output_shapes
:h®
4batch_normalization_855/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_855_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0¼
%batch_normalization_855/batchnorm/mulMul+batch_normalization_855/batchnorm/Rsqrt:y:0<batch_normalization_855/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:h§
'batch_normalization_855/batchnorm/mul_1Muldense_945/BiasAdd:output:0)batch_normalization_855/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh°
'batch_normalization_855/batchnorm/mul_2Mul0batch_normalization_855/moments/Squeeze:output:0)batch_normalization_855/batchnorm/mul:z:0*
T0*
_output_shapes
:h¦
0batch_normalization_855/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_855_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0¸
%batch_normalization_855/batchnorm/subSub8batch_normalization_855/batchnorm/ReadVariableOp:value:0+batch_normalization_855/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hº
'batch_normalization_855/batchnorm/add_1AddV2+batch_normalization_855/batchnorm/mul_1:z:0)batch_normalization_855/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
leaky_re_lu_855/LeakyRelu	LeakyRelu+batch_normalization_855/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>
dense_946/MatMul/ReadVariableOpReadVariableOp(dense_946_matmul_readvariableop_resource*
_output_shapes

:hh*
dtype0
dense_946/MatMulMatMul'leaky_re_lu_855/LeakyRelu:activations:0'dense_946/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 dense_946/BiasAdd/ReadVariableOpReadVariableOp)dense_946_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0
dense_946/BiasAddBiasAdddense_946/MatMul:product:0(dense_946/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
6batch_normalization_856/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_856/moments/meanMeandense_946/BiasAdd:output:0?batch_normalization_856/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(
,batch_normalization_856/moments/StopGradientStopGradient-batch_normalization_856/moments/mean:output:0*
T0*
_output_shapes

:hË
1batch_normalization_856/moments/SquaredDifferenceSquaredDifferencedense_946/BiasAdd:output:05batch_normalization_856/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
:batch_normalization_856/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_856/moments/varianceMean5batch_normalization_856/moments/SquaredDifference:z:0Cbatch_normalization_856/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(
'batch_normalization_856/moments/SqueezeSqueeze-batch_normalization_856/moments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 £
)batch_normalization_856/moments/Squeeze_1Squeeze1batch_normalization_856/moments/variance:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 r
-batch_normalization_856/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_856/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_856_assignmovingavg_readvariableop_resource*
_output_shapes
:h*
dtype0É
+batch_normalization_856/AssignMovingAvg/subSub>batch_normalization_856/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_856/moments/Squeeze:output:0*
T0*
_output_shapes
:hÀ
+batch_normalization_856/AssignMovingAvg/mulMul/batch_normalization_856/AssignMovingAvg/sub:z:06batch_normalization_856/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h
'batch_normalization_856/AssignMovingAvgAssignSubVariableOp?batch_normalization_856_assignmovingavg_readvariableop_resource/batch_normalization_856/AssignMovingAvg/mul:z:07^batch_normalization_856/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_856/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_856/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_856_assignmovingavg_1_readvariableop_resource*
_output_shapes
:h*
dtype0Ï
-batch_normalization_856/AssignMovingAvg_1/subSub@batch_normalization_856/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_856/moments/Squeeze_1:output:0*
T0*
_output_shapes
:hÆ
-batch_normalization_856/AssignMovingAvg_1/mulMul1batch_normalization_856/AssignMovingAvg_1/sub:z:08batch_normalization_856/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h
)batch_normalization_856/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_856_assignmovingavg_1_readvariableop_resource1batch_normalization_856/AssignMovingAvg_1/mul:z:09^batch_normalization_856/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_856/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_856/batchnorm/addAddV22batch_normalization_856/moments/Squeeze_1:output:00batch_normalization_856/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
'batch_normalization_856/batchnorm/RsqrtRsqrt)batch_normalization_856/batchnorm/add:z:0*
T0*
_output_shapes
:h®
4batch_normalization_856/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_856_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0¼
%batch_normalization_856/batchnorm/mulMul+batch_normalization_856/batchnorm/Rsqrt:y:0<batch_normalization_856/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:h§
'batch_normalization_856/batchnorm/mul_1Muldense_946/BiasAdd:output:0)batch_normalization_856/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh°
'batch_normalization_856/batchnorm/mul_2Mul0batch_normalization_856/moments/Squeeze:output:0)batch_normalization_856/batchnorm/mul:z:0*
T0*
_output_shapes
:h¦
0batch_normalization_856/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_856_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0¸
%batch_normalization_856/batchnorm/subSub8batch_normalization_856/batchnorm/ReadVariableOp:value:0+batch_normalization_856/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hº
'batch_normalization_856/batchnorm/add_1AddV2+batch_normalization_856/batchnorm/mul_1:z:0)batch_normalization_856/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
leaky_re_lu_856/LeakyRelu	LeakyRelu+batch_normalization_856/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>
dense_947/MatMul/ReadVariableOpReadVariableOp(dense_947_matmul_readvariableop_resource*
_output_shapes

:hh*
dtype0
dense_947/MatMulMatMul'leaky_re_lu_856/LeakyRelu:activations:0'dense_947/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 dense_947/BiasAdd/ReadVariableOpReadVariableOp)dense_947_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0
dense_947/BiasAddBiasAdddense_947/MatMul:product:0(dense_947/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
6batch_normalization_857/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_857/moments/meanMeandense_947/BiasAdd:output:0?batch_normalization_857/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(
,batch_normalization_857/moments/StopGradientStopGradient-batch_normalization_857/moments/mean:output:0*
T0*
_output_shapes

:hË
1batch_normalization_857/moments/SquaredDifferenceSquaredDifferencedense_947/BiasAdd:output:05batch_normalization_857/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
:batch_normalization_857/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_857/moments/varianceMean5batch_normalization_857/moments/SquaredDifference:z:0Cbatch_normalization_857/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(
'batch_normalization_857/moments/SqueezeSqueeze-batch_normalization_857/moments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 £
)batch_normalization_857/moments/Squeeze_1Squeeze1batch_normalization_857/moments/variance:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 r
-batch_normalization_857/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_857/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_857_assignmovingavg_readvariableop_resource*
_output_shapes
:h*
dtype0É
+batch_normalization_857/AssignMovingAvg/subSub>batch_normalization_857/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_857/moments/Squeeze:output:0*
T0*
_output_shapes
:hÀ
+batch_normalization_857/AssignMovingAvg/mulMul/batch_normalization_857/AssignMovingAvg/sub:z:06batch_normalization_857/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h
'batch_normalization_857/AssignMovingAvgAssignSubVariableOp?batch_normalization_857_assignmovingavg_readvariableop_resource/batch_normalization_857/AssignMovingAvg/mul:z:07^batch_normalization_857/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_857/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_857/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_857_assignmovingavg_1_readvariableop_resource*
_output_shapes
:h*
dtype0Ï
-batch_normalization_857/AssignMovingAvg_1/subSub@batch_normalization_857/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_857/moments/Squeeze_1:output:0*
T0*
_output_shapes
:hÆ
-batch_normalization_857/AssignMovingAvg_1/mulMul1batch_normalization_857/AssignMovingAvg_1/sub:z:08batch_normalization_857/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h
)batch_normalization_857/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_857_assignmovingavg_1_readvariableop_resource1batch_normalization_857/AssignMovingAvg_1/mul:z:09^batch_normalization_857/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_857/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_857/batchnorm/addAddV22batch_normalization_857/moments/Squeeze_1:output:00batch_normalization_857/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
'batch_normalization_857/batchnorm/RsqrtRsqrt)batch_normalization_857/batchnorm/add:z:0*
T0*
_output_shapes
:h®
4batch_normalization_857/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_857_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0¼
%batch_normalization_857/batchnorm/mulMul+batch_normalization_857/batchnorm/Rsqrt:y:0<batch_normalization_857/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:h§
'batch_normalization_857/batchnorm/mul_1Muldense_947/BiasAdd:output:0)batch_normalization_857/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh°
'batch_normalization_857/batchnorm/mul_2Mul0batch_normalization_857/moments/Squeeze:output:0)batch_normalization_857/batchnorm/mul:z:0*
T0*
_output_shapes
:h¦
0batch_normalization_857/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_857_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0¸
%batch_normalization_857/batchnorm/subSub8batch_normalization_857/batchnorm/ReadVariableOp:value:0+batch_normalization_857/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hº
'batch_normalization_857/batchnorm/add_1AddV2+batch_normalization_857/batchnorm/mul_1:z:0)batch_normalization_857/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
leaky_re_lu_857/LeakyRelu	LeakyRelu+batch_normalization_857/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>
dense_948/MatMul/ReadVariableOpReadVariableOp(dense_948_matmul_readvariableop_resource*
_output_shapes

:hh*
dtype0
dense_948/MatMulMatMul'leaky_re_lu_857/LeakyRelu:activations:0'dense_948/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 dense_948/BiasAdd/ReadVariableOpReadVariableOp)dense_948_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0
dense_948/BiasAddBiasAdddense_948/MatMul:product:0(dense_948/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
6batch_normalization_858/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_858/moments/meanMeandense_948/BiasAdd:output:0?batch_normalization_858/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(
,batch_normalization_858/moments/StopGradientStopGradient-batch_normalization_858/moments/mean:output:0*
T0*
_output_shapes

:hË
1batch_normalization_858/moments/SquaredDifferenceSquaredDifferencedense_948/BiasAdd:output:05batch_normalization_858/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
:batch_normalization_858/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_858/moments/varianceMean5batch_normalization_858/moments/SquaredDifference:z:0Cbatch_normalization_858/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(
'batch_normalization_858/moments/SqueezeSqueeze-batch_normalization_858/moments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 £
)batch_normalization_858/moments/Squeeze_1Squeeze1batch_normalization_858/moments/variance:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 r
-batch_normalization_858/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_858/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_858_assignmovingavg_readvariableop_resource*
_output_shapes
:h*
dtype0É
+batch_normalization_858/AssignMovingAvg/subSub>batch_normalization_858/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_858/moments/Squeeze:output:0*
T0*
_output_shapes
:hÀ
+batch_normalization_858/AssignMovingAvg/mulMul/batch_normalization_858/AssignMovingAvg/sub:z:06batch_normalization_858/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h
'batch_normalization_858/AssignMovingAvgAssignSubVariableOp?batch_normalization_858_assignmovingavg_readvariableop_resource/batch_normalization_858/AssignMovingAvg/mul:z:07^batch_normalization_858/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_858/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_858/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_858_assignmovingavg_1_readvariableop_resource*
_output_shapes
:h*
dtype0Ï
-batch_normalization_858/AssignMovingAvg_1/subSub@batch_normalization_858/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_858/moments/Squeeze_1:output:0*
T0*
_output_shapes
:hÆ
-batch_normalization_858/AssignMovingAvg_1/mulMul1batch_normalization_858/AssignMovingAvg_1/sub:z:08batch_normalization_858/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h
)batch_normalization_858/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_858_assignmovingavg_1_readvariableop_resource1batch_normalization_858/AssignMovingAvg_1/mul:z:09^batch_normalization_858/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_858/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_858/batchnorm/addAddV22batch_normalization_858/moments/Squeeze_1:output:00batch_normalization_858/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
'batch_normalization_858/batchnorm/RsqrtRsqrt)batch_normalization_858/batchnorm/add:z:0*
T0*
_output_shapes
:h®
4batch_normalization_858/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_858_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0¼
%batch_normalization_858/batchnorm/mulMul+batch_normalization_858/batchnorm/Rsqrt:y:0<batch_normalization_858/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:h§
'batch_normalization_858/batchnorm/mul_1Muldense_948/BiasAdd:output:0)batch_normalization_858/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh°
'batch_normalization_858/batchnorm/mul_2Mul0batch_normalization_858/moments/Squeeze:output:0)batch_normalization_858/batchnorm/mul:z:0*
T0*
_output_shapes
:h¦
0batch_normalization_858/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_858_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0¸
%batch_normalization_858/batchnorm/subSub8batch_normalization_858/batchnorm/ReadVariableOp:value:0+batch_normalization_858/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hº
'batch_normalization_858/batchnorm/add_1AddV2+batch_normalization_858/batchnorm/mul_1:z:0)batch_normalization_858/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
leaky_re_lu_858/LeakyRelu	LeakyRelu+batch_normalization_858/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>
dense_949/MatMul/ReadVariableOpReadVariableOp(dense_949_matmul_readvariableop_resource*
_output_shapes

:hh*
dtype0
dense_949/MatMulMatMul'leaky_re_lu_858/LeakyRelu:activations:0'dense_949/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 dense_949/BiasAdd/ReadVariableOpReadVariableOp)dense_949_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0
dense_949/BiasAddBiasAdddense_949/MatMul:product:0(dense_949/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
6batch_normalization_859/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_859/moments/meanMeandense_949/BiasAdd:output:0?batch_normalization_859/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(
,batch_normalization_859/moments/StopGradientStopGradient-batch_normalization_859/moments/mean:output:0*
T0*
_output_shapes

:hË
1batch_normalization_859/moments/SquaredDifferenceSquaredDifferencedense_949/BiasAdd:output:05batch_normalization_859/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
:batch_normalization_859/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_859/moments/varianceMean5batch_normalization_859/moments/SquaredDifference:z:0Cbatch_normalization_859/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:h*
	keep_dims(
'batch_normalization_859/moments/SqueezeSqueeze-batch_normalization_859/moments/mean:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 £
)batch_normalization_859/moments/Squeeze_1Squeeze1batch_normalization_859/moments/variance:output:0*
T0*
_output_shapes
:h*
squeeze_dims
 r
-batch_normalization_859/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_859/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_859_assignmovingavg_readvariableop_resource*
_output_shapes
:h*
dtype0É
+batch_normalization_859/AssignMovingAvg/subSub>batch_normalization_859/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_859/moments/Squeeze:output:0*
T0*
_output_shapes
:hÀ
+batch_normalization_859/AssignMovingAvg/mulMul/batch_normalization_859/AssignMovingAvg/sub:z:06batch_normalization_859/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:h
'batch_normalization_859/AssignMovingAvgAssignSubVariableOp?batch_normalization_859_assignmovingavg_readvariableop_resource/batch_normalization_859/AssignMovingAvg/mul:z:07^batch_normalization_859/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_859/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_859/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_859_assignmovingavg_1_readvariableop_resource*
_output_shapes
:h*
dtype0Ï
-batch_normalization_859/AssignMovingAvg_1/subSub@batch_normalization_859/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_859/moments/Squeeze_1:output:0*
T0*
_output_shapes
:hÆ
-batch_normalization_859/AssignMovingAvg_1/mulMul1batch_normalization_859/AssignMovingAvg_1/sub:z:08batch_normalization_859/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:h
)batch_normalization_859/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_859_assignmovingavg_1_readvariableop_resource1batch_normalization_859/AssignMovingAvg_1/mul:z:09^batch_normalization_859/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_859/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_859/batchnorm/addAddV22batch_normalization_859/moments/Squeeze_1:output:00batch_normalization_859/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
'batch_normalization_859/batchnorm/RsqrtRsqrt)batch_normalization_859/batchnorm/add:z:0*
T0*
_output_shapes
:h®
4batch_normalization_859/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_859_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0¼
%batch_normalization_859/batchnorm/mulMul+batch_normalization_859/batchnorm/Rsqrt:y:0<batch_normalization_859/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:h§
'batch_normalization_859/batchnorm/mul_1Muldense_949/BiasAdd:output:0)batch_normalization_859/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh°
'batch_normalization_859/batchnorm/mul_2Mul0batch_normalization_859/moments/Squeeze:output:0)batch_normalization_859/batchnorm/mul:z:0*
T0*
_output_shapes
:h¦
0batch_normalization_859/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_859_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0¸
%batch_normalization_859/batchnorm/subSub8batch_normalization_859/batchnorm/ReadVariableOp:value:0+batch_normalization_859/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hº
'batch_normalization_859/batchnorm/add_1AddV2+batch_normalization_859/batchnorm/mul_1:z:0)batch_normalization_859/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
leaky_re_lu_859/LeakyRelu	LeakyRelu+batch_normalization_859/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>
dense_950/MatMul/ReadVariableOpReadVariableOp(dense_950_matmul_readvariableop_resource*
_output_shapes

:h/*
dtype0
dense_950/MatMulMatMul'leaky_re_lu_859/LeakyRelu:activations:0'dense_950/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 dense_950/BiasAdd/ReadVariableOpReadVariableOp)dense_950_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_950/BiasAddBiasAdddense_950/MatMul:product:0(dense_950/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
6batch_normalization_860/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_860/moments/meanMeandense_950/BiasAdd:output:0?batch_normalization_860/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(
,batch_normalization_860/moments/StopGradientStopGradient-batch_normalization_860/moments/mean:output:0*
T0*
_output_shapes

:/Ë
1batch_normalization_860/moments/SquaredDifferenceSquaredDifferencedense_950/BiasAdd:output:05batch_normalization_860/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
:batch_normalization_860/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_860/moments/varianceMean5batch_normalization_860/moments/SquaredDifference:z:0Cbatch_normalization_860/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(
'batch_normalization_860/moments/SqueezeSqueeze-batch_normalization_860/moments/mean:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 £
)batch_normalization_860/moments/Squeeze_1Squeeze1batch_normalization_860/moments/variance:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 r
-batch_normalization_860/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_860/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_860_assignmovingavg_readvariableop_resource*
_output_shapes
:/*
dtype0É
+batch_normalization_860/AssignMovingAvg/subSub>batch_normalization_860/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_860/moments/Squeeze:output:0*
T0*
_output_shapes
:/À
+batch_normalization_860/AssignMovingAvg/mulMul/batch_normalization_860/AssignMovingAvg/sub:z:06batch_normalization_860/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:/
'batch_normalization_860/AssignMovingAvgAssignSubVariableOp?batch_normalization_860_assignmovingavg_readvariableop_resource/batch_normalization_860/AssignMovingAvg/mul:z:07^batch_normalization_860/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_860/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_860/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_860_assignmovingavg_1_readvariableop_resource*
_output_shapes
:/*
dtype0Ï
-batch_normalization_860/AssignMovingAvg_1/subSub@batch_normalization_860/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_860/moments/Squeeze_1:output:0*
T0*
_output_shapes
:/Æ
-batch_normalization_860/AssignMovingAvg_1/mulMul1batch_normalization_860/AssignMovingAvg_1/sub:z:08batch_normalization_860/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:/
)batch_normalization_860/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_860_assignmovingavg_1_readvariableop_resource1batch_normalization_860/AssignMovingAvg_1/mul:z:09^batch_normalization_860/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_860/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_860/batchnorm/addAddV22batch_normalization_860/moments/Squeeze_1:output:00batch_normalization_860/batchnorm/add/y:output:0*
T0*
_output_shapes
:/
'batch_normalization_860/batchnorm/RsqrtRsqrt)batch_normalization_860/batchnorm/add:z:0*
T0*
_output_shapes
:/®
4batch_normalization_860/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_860_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0¼
%batch_normalization_860/batchnorm/mulMul+batch_normalization_860/batchnorm/Rsqrt:y:0<batch_normalization_860/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/§
'batch_normalization_860/batchnorm/mul_1Muldense_950/BiasAdd:output:0)batch_normalization_860/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/°
'batch_normalization_860/batchnorm/mul_2Mul0batch_normalization_860/moments/Squeeze:output:0)batch_normalization_860/batchnorm/mul:z:0*
T0*
_output_shapes
:/¦
0batch_normalization_860/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_860_batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0¸
%batch_normalization_860/batchnorm/subSub8batch_normalization_860/batchnorm/ReadVariableOp:value:0+batch_normalization_860/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/º
'batch_normalization_860/batchnorm/add_1AddV2+batch_normalization_860/batchnorm/mul_1:z:0)batch_normalization_860/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
leaky_re_lu_860/LeakyRelu	LeakyRelu+batch_normalization_860/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*
alpha%>
dense_951/MatMul/ReadVariableOpReadVariableOp(dense_951_matmul_readvariableop_resource*
_output_shapes

://*
dtype0
dense_951/MatMulMatMul'leaky_re_lu_860/LeakyRelu:activations:0'dense_951/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 dense_951/BiasAdd/ReadVariableOpReadVariableOp)dense_951_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_951/BiasAddBiasAdddense_951/MatMul:product:0(dense_951/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
6batch_normalization_861/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_861/moments/meanMeandense_951/BiasAdd:output:0?batch_normalization_861/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(
,batch_normalization_861/moments/StopGradientStopGradient-batch_normalization_861/moments/mean:output:0*
T0*
_output_shapes

:/Ë
1batch_normalization_861/moments/SquaredDifferenceSquaredDifferencedense_951/BiasAdd:output:05batch_normalization_861/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
:batch_normalization_861/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_861/moments/varianceMean5batch_normalization_861/moments/SquaredDifference:z:0Cbatch_normalization_861/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(
'batch_normalization_861/moments/SqueezeSqueeze-batch_normalization_861/moments/mean:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 £
)batch_normalization_861/moments/Squeeze_1Squeeze1batch_normalization_861/moments/variance:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 r
-batch_normalization_861/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_861/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_861_assignmovingavg_readvariableop_resource*
_output_shapes
:/*
dtype0É
+batch_normalization_861/AssignMovingAvg/subSub>batch_normalization_861/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_861/moments/Squeeze:output:0*
T0*
_output_shapes
:/À
+batch_normalization_861/AssignMovingAvg/mulMul/batch_normalization_861/AssignMovingAvg/sub:z:06batch_normalization_861/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:/
'batch_normalization_861/AssignMovingAvgAssignSubVariableOp?batch_normalization_861_assignmovingavg_readvariableop_resource/batch_normalization_861/AssignMovingAvg/mul:z:07^batch_normalization_861/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_861/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_861/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_861_assignmovingavg_1_readvariableop_resource*
_output_shapes
:/*
dtype0Ï
-batch_normalization_861/AssignMovingAvg_1/subSub@batch_normalization_861/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_861/moments/Squeeze_1:output:0*
T0*
_output_shapes
:/Æ
-batch_normalization_861/AssignMovingAvg_1/mulMul1batch_normalization_861/AssignMovingAvg_1/sub:z:08batch_normalization_861/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:/
)batch_normalization_861/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_861_assignmovingavg_1_readvariableop_resource1batch_normalization_861/AssignMovingAvg_1/mul:z:09^batch_normalization_861/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_861/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_861/batchnorm/addAddV22batch_normalization_861/moments/Squeeze_1:output:00batch_normalization_861/batchnorm/add/y:output:0*
T0*
_output_shapes
:/
'batch_normalization_861/batchnorm/RsqrtRsqrt)batch_normalization_861/batchnorm/add:z:0*
T0*
_output_shapes
:/®
4batch_normalization_861/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_861_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0¼
%batch_normalization_861/batchnorm/mulMul+batch_normalization_861/batchnorm/Rsqrt:y:0<batch_normalization_861/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/§
'batch_normalization_861/batchnorm/mul_1Muldense_951/BiasAdd:output:0)batch_normalization_861/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/°
'batch_normalization_861/batchnorm/mul_2Mul0batch_normalization_861/moments/Squeeze:output:0)batch_normalization_861/batchnorm/mul:z:0*
T0*
_output_shapes
:/¦
0batch_normalization_861/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_861_batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0¸
%batch_normalization_861/batchnorm/subSub8batch_normalization_861/batchnorm/ReadVariableOp:value:0+batch_normalization_861/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/º
'batch_normalization_861/batchnorm/add_1AddV2+batch_normalization_861/batchnorm/mul_1:z:0)batch_normalization_861/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
leaky_re_lu_861/LeakyRelu	LeakyRelu+batch_normalization_861/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*
alpha%>
dense_952/MatMul/ReadVariableOpReadVariableOp(dense_952_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0
dense_952/MatMulMatMul'leaky_re_lu_861/LeakyRelu:activations:0'dense_952/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_952/BiasAdd/ReadVariableOpReadVariableOp)dense_952_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_952/BiasAddBiasAdddense_952/MatMul:product:0(dense_952/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_862/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_862/moments/meanMeandense_952/BiasAdd:output:0?batch_normalization_862/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_862/moments/StopGradientStopGradient-batch_normalization_862/moments/mean:output:0*
T0*
_output_shapes

:Ë
1batch_normalization_862/moments/SquaredDifferenceSquaredDifferencedense_952/BiasAdd:output:05batch_normalization_862/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_862/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_862/moments/varianceMean5batch_normalization_862/moments/SquaredDifference:z:0Cbatch_normalization_862/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_862/moments/SqueezeSqueeze-batch_normalization_862/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_862/moments/Squeeze_1Squeeze1batch_normalization_862/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_862/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_862/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_862_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_862/AssignMovingAvg/subSub>batch_normalization_862/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_862/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_862/AssignMovingAvg/mulMul/batch_normalization_862/AssignMovingAvg/sub:z:06batch_normalization_862/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_862/AssignMovingAvgAssignSubVariableOp?batch_normalization_862_assignmovingavg_readvariableop_resource/batch_normalization_862/AssignMovingAvg/mul:z:07^batch_normalization_862/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_862/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_862/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_862_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_862/AssignMovingAvg_1/subSub@batch_normalization_862/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_862/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_862/AssignMovingAvg_1/mulMul1batch_normalization_862/AssignMovingAvg_1/sub:z:08batch_normalization_862/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_862/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_862_assignmovingavg_1_readvariableop_resource1batch_normalization_862/AssignMovingAvg_1/mul:z:09^batch_normalization_862/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_862/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_862/batchnorm/addAddV22batch_normalization_862/moments/Squeeze_1:output:00batch_normalization_862/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_862/batchnorm/RsqrtRsqrt)batch_normalization_862/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_862/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_862_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_862/batchnorm/mulMul+batch_normalization_862/batchnorm/Rsqrt:y:0<batch_normalization_862/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_862/batchnorm/mul_1Muldense_952/BiasAdd:output:0)batch_normalization_862/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_862/batchnorm/mul_2Mul0batch_normalization_862/moments/Squeeze:output:0)batch_normalization_862/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_862/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_862_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_862/batchnorm/subSub8batch_normalization_862/batchnorm/ReadVariableOp:value:0+batch_normalization_862/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_862/batchnorm/add_1AddV2+batch_normalization_862/batchnorm/mul_1:z:0)batch_normalization_862/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_862/LeakyRelu	LeakyRelu+batch_normalization_862/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_953/MatMul/ReadVariableOpReadVariableOp(dense_953_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_953/MatMulMatMul'leaky_re_lu_862/LeakyRelu:activations:0'dense_953/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_953/BiasAdd/ReadVariableOpReadVariableOp)dense_953_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_953/BiasAddBiasAdddense_953/MatMul:product:0(dense_953/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_863/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_863/moments/meanMeandense_953/BiasAdd:output:0?batch_normalization_863/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_863/moments/StopGradientStopGradient-batch_normalization_863/moments/mean:output:0*
T0*
_output_shapes

:Ë
1batch_normalization_863/moments/SquaredDifferenceSquaredDifferencedense_953/BiasAdd:output:05batch_normalization_863/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_863/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_863/moments/varianceMean5batch_normalization_863/moments/SquaredDifference:z:0Cbatch_normalization_863/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_863/moments/SqueezeSqueeze-batch_normalization_863/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_863/moments/Squeeze_1Squeeze1batch_normalization_863/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_863/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_863/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_863_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_863/AssignMovingAvg/subSub>batch_normalization_863/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_863/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_863/AssignMovingAvg/mulMul/batch_normalization_863/AssignMovingAvg/sub:z:06batch_normalization_863/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_863/AssignMovingAvgAssignSubVariableOp?batch_normalization_863_assignmovingavg_readvariableop_resource/batch_normalization_863/AssignMovingAvg/mul:z:07^batch_normalization_863/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_863/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_863/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_863_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_863/AssignMovingAvg_1/subSub@batch_normalization_863/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_863/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_863/AssignMovingAvg_1/mulMul1batch_normalization_863/AssignMovingAvg_1/sub:z:08batch_normalization_863/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_863/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_863_assignmovingavg_1_readvariableop_resource1batch_normalization_863/AssignMovingAvg_1/mul:z:09^batch_normalization_863/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_863/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_863/batchnorm/addAddV22batch_normalization_863/moments/Squeeze_1:output:00batch_normalization_863/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_863/batchnorm/RsqrtRsqrt)batch_normalization_863/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_863/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_863_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_863/batchnorm/mulMul+batch_normalization_863/batchnorm/Rsqrt:y:0<batch_normalization_863/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_863/batchnorm/mul_1Muldense_953/BiasAdd:output:0)batch_normalization_863/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_863/batchnorm/mul_2Mul0batch_normalization_863/moments/Squeeze:output:0)batch_normalization_863/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_863/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_863_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_863/batchnorm/subSub8batch_normalization_863/batchnorm/ReadVariableOp:value:0+batch_normalization_863/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_863/batchnorm/add_1AddV2+batch_normalization_863/batchnorm/mul_1:z:0)batch_normalization_863/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_863/LeakyRelu	LeakyRelu+batch_normalization_863/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_954/MatMul/ReadVariableOpReadVariableOp(dense_954_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_954/MatMulMatMul'leaky_re_lu_863/LeakyRelu:activations:0'dense_954/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_954/BiasAdd/ReadVariableOpReadVariableOp)dense_954_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_954/BiasAddBiasAdddense_954/MatMul:product:0(dense_954/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_864/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_864/moments/meanMeandense_954/BiasAdd:output:0?batch_normalization_864/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_864/moments/StopGradientStopGradient-batch_normalization_864/moments/mean:output:0*
T0*
_output_shapes

:Ë
1batch_normalization_864/moments/SquaredDifferenceSquaredDifferencedense_954/BiasAdd:output:05batch_normalization_864/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_864/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_864/moments/varianceMean5batch_normalization_864/moments/SquaredDifference:z:0Cbatch_normalization_864/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_864/moments/SqueezeSqueeze-batch_normalization_864/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_864/moments/Squeeze_1Squeeze1batch_normalization_864/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_864/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_864/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_864_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_864/AssignMovingAvg/subSub>batch_normalization_864/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_864/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_864/AssignMovingAvg/mulMul/batch_normalization_864/AssignMovingAvg/sub:z:06batch_normalization_864/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_864/AssignMovingAvgAssignSubVariableOp?batch_normalization_864_assignmovingavg_readvariableop_resource/batch_normalization_864/AssignMovingAvg/mul:z:07^batch_normalization_864/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_864/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_864/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_864_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_864/AssignMovingAvg_1/subSub@batch_normalization_864/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_864/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_864/AssignMovingAvg_1/mulMul1batch_normalization_864/AssignMovingAvg_1/sub:z:08batch_normalization_864/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_864/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_864_assignmovingavg_1_readvariableop_resource1batch_normalization_864/AssignMovingAvg_1/mul:z:09^batch_normalization_864/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_864/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_864/batchnorm/addAddV22batch_normalization_864/moments/Squeeze_1:output:00batch_normalization_864/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_864/batchnorm/RsqrtRsqrt)batch_normalization_864/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_864/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_864_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_864/batchnorm/mulMul+batch_normalization_864/batchnorm/Rsqrt:y:0<batch_normalization_864/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_864/batchnorm/mul_1Muldense_954/BiasAdd:output:0)batch_normalization_864/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_864/batchnorm/mul_2Mul0batch_normalization_864/moments/Squeeze:output:0)batch_normalization_864/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_864/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_864_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_864/batchnorm/subSub8batch_normalization_864/batchnorm/ReadVariableOp:value:0+batch_normalization_864/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_864/batchnorm/add_1AddV2+batch_normalization_864/batchnorm/mul_1:z:0)batch_normalization_864/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_864/LeakyRelu	LeakyRelu+batch_normalization_864/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_955/MatMul/ReadVariableOpReadVariableOp(dense_955_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_955/MatMulMatMul'leaky_re_lu_864/LeakyRelu:activations:0'dense_955/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_955/BiasAdd/ReadVariableOpReadVariableOp)dense_955_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_955/BiasAddBiasAdddense_955/MatMul:product:0(dense_955/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_865/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_865/moments/meanMeandense_955/BiasAdd:output:0?batch_normalization_865/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_865/moments/StopGradientStopGradient-batch_normalization_865/moments/mean:output:0*
T0*
_output_shapes

:Ë
1batch_normalization_865/moments/SquaredDifferenceSquaredDifferencedense_955/BiasAdd:output:05batch_normalization_865/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_865/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_865/moments/varianceMean5batch_normalization_865/moments/SquaredDifference:z:0Cbatch_normalization_865/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_865/moments/SqueezeSqueeze-batch_normalization_865/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_865/moments/Squeeze_1Squeeze1batch_normalization_865/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_865/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_865/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_865_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_865/AssignMovingAvg/subSub>batch_normalization_865/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_865/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_865/AssignMovingAvg/mulMul/batch_normalization_865/AssignMovingAvg/sub:z:06batch_normalization_865/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
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
×#<¶
8batch_normalization_865/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_865_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_865/AssignMovingAvg_1/subSub@batch_normalization_865/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_865/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_865/AssignMovingAvg_1/mulMul1batch_normalization_865/AssignMovingAvg_1/sub:z:08batch_normalization_865/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_865/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_865_assignmovingavg_1_readvariableop_resource1batch_normalization_865/AssignMovingAvg_1/mul:z:09^batch_normalization_865/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_865/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_865/batchnorm/addAddV22batch_normalization_865/moments/Squeeze_1:output:00batch_normalization_865/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_865/batchnorm/RsqrtRsqrt)batch_normalization_865/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_865/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_865_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_865/batchnorm/mulMul+batch_normalization_865/batchnorm/Rsqrt:y:0<batch_normalization_865/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_865/batchnorm/mul_1Muldense_955/BiasAdd:output:0)batch_normalization_865/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_865/batchnorm/mul_2Mul0batch_normalization_865/moments/Squeeze:output:0)batch_normalization_865/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_865/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_865_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_865/batchnorm/subSub8batch_normalization_865/batchnorm/ReadVariableOp:value:0+batch_normalization_865/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_865/batchnorm/add_1AddV2+batch_normalization_865/batchnorm/mul_1:z:0)batch_normalization_865/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_865/LeakyRelu	LeakyRelu+batch_normalization_865/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_956/MatMul/ReadVariableOpReadVariableOp(dense_956_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_956/MatMulMatMul'leaky_re_lu_865/LeakyRelu:activations:0'dense_956/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_956/BiasAdd/ReadVariableOpReadVariableOp)dense_956_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_956/BiasAddBiasAdddense_956/MatMul:product:0(dense_956/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_956/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾!
NoOpNoOp(^batch_normalization_855/AssignMovingAvg7^batch_normalization_855/AssignMovingAvg/ReadVariableOp*^batch_normalization_855/AssignMovingAvg_19^batch_normalization_855/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_855/batchnorm/ReadVariableOp5^batch_normalization_855/batchnorm/mul/ReadVariableOp(^batch_normalization_856/AssignMovingAvg7^batch_normalization_856/AssignMovingAvg/ReadVariableOp*^batch_normalization_856/AssignMovingAvg_19^batch_normalization_856/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_856/batchnorm/ReadVariableOp5^batch_normalization_856/batchnorm/mul/ReadVariableOp(^batch_normalization_857/AssignMovingAvg7^batch_normalization_857/AssignMovingAvg/ReadVariableOp*^batch_normalization_857/AssignMovingAvg_19^batch_normalization_857/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_857/batchnorm/ReadVariableOp5^batch_normalization_857/batchnorm/mul/ReadVariableOp(^batch_normalization_858/AssignMovingAvg7^batch_normalization_858/AssignMovingAvg/ReadVariableOp*^batch_normalization_858/AssignMovingAvg_19^batch_normalization_858/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_858/batchnorm/ReadVariableOp5^batch_normalization_858/batchnorm/mul/ReadVariableOp(^batch_normalization_859/AssignMovingAvg7^batch_normalization_859/AssignMovingAvg/ReadVariableOp*^batch_normalization_859/AssignMovingAvg_19^batch_normalization_859/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_859/batchnorm/ReadVariableOp5^batch_normalization_859/batchnorm/mul/ReadVariableOp(^batch_normalization_860/AssignMovingAvg7^batch_normalization_860/AssignMovingAvg/ReadVariableOp*^batch_normalization_860/AssignMovingAvg_19^batch_normalization_860/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_860/batchnorm/ReadVariableOp5^batch_normalization_860/batchnorm/mul/ReadVariableOp(^batch_normalization_861/AssignMovingAvg7^batch_normalization_861/AssignMovingAvg/ReadVariableOp*^batch_normalization_861/AssignMovingAvg_19^batch_normalization_861/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_861/batchnorm/ReadVariableOp5^batch_normalization_861/batchnorm/mul/ReadVariableOp(^batch_normalization_862/AssignMovingAvg7^batch_normalization_862/AssignMovingAvg/ReadVariableOp*^batch_normalization_862/AssignMovingAvg_19^batch_normalization_862/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_862/batchnorm/ReadVariableOp5^batch_normalization_862/batchnorm/mul/ReadVariableOp(^batch_normalization_863/AssignMovingAvg7^batch_normalization_863/AssignMovingAvg/ReadVariableOp*^batch_normalization_863/AssignMovingAvg_19^batch_normalization_863/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_863/batchnorm/ReadVariableOp5^batch_normalization_863/batchnorm/mul/ReadVariableOp(^batch_normalization_864/AssignMovingAvg7^batch_normalization_864/AssignMovingAvg/ReadVariableOp*^batch_normalization_864/AssignMovingAvg_19^batch_normalization_864/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_864/batchnorm/ReadVariableOp5^batch_normalization_864/batchnorm/mul/ReadVariableOp(^batch_normalization_865/AssignMovingAvg7^batch_normalization_865/AssignMovingAvg/ReadVariableOp*^batch_normalization_865/AssignMovingAvg_19^batch_normalization_865/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_865/batchnorm/ReadVariableOp5^batch_normalization_865/batchnorm/mul/ReadVariableOp!^dense_945/BiasAdd/ReadVariableOp ^dense_945/MatMul/ReadVariableOp!^dense_946/BiasAdd/ReadVariableOp ^dense_946/MatMul/ReadVariableOp!^dense_947/BiasAdd/ReadVariableOp ^dense_947/MatMul/ReadVariableOp!^dense_948/BiasAdd/ReadVariableOp ^dense_948/MatMul/ReadVariableOp!^dense_949/BiasAdd/ReadVariableOp ^dense_949/MatMul/ReadVariableOp!^dense_950/BiasAdd/ReadVariableOp ^dense_950/MatMul/ReadVariableOp!^dense_951/BiasAdd/ReadVariableOp ^dense_951/MatMul/ReadVariableOp!^dense_952/BiasAdd/ReadVariableOp ^dense_952/MatMul/ReadVariableOp!^dense_953/BiasAdd/ReadVariableOp ^dense_953/MatMul/ReadVariableOp!^dense_954/BiasAdd/ReadVariableOp ^dense_954/MatMul/ReadVariableOp!^dense_955/BiasAdd/ReadVariableOp ^dense_955/MatMul/ReadVariableOp!^dense_956/BiasAdd/ReadVariableOp ^dense_956/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_855/AssignMovingAvg'batch_normalization_855/AssignMovingAvg2p
6batch_normalization_855/AssignMovingAvg/ReadVariableOp6batch_normalization_855/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_855/AssignMovingAvg_1)batch_normalization_855/AssignMovingAvg_12t
8batch_normalization_855/AssignMovingAvg_1/ReadVariableOp8batch_normalization_855/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_855/batchnorm/ReadVariableOp0batch_normalization_855/batchnorm/ReadVariableOp2l
4batch_normalization_855/batchnorm/mul/ReadVariableOp4batch_normalization_855/batchnorm/mul/ReadVariableOp2R
'batch_normalization_856/AssignMovingAvg'batch_normalization_856/AssignMovingAvg2p
6batch_normalization_856/AssignMovingAvg/ReadVariableOp6batch_normalization_856/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_856/AssignMovingAvg_1)batch_normalization_856/AssignMovingAvg_12t
8batch_normalization_856/AssignMovingAvg_1/ReadVariableOp8batch_normalization_856/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_856/batchnorm/ReadVariableOp0batch_normalization_856/batchnorm/ReadVariableOp2l
4batch_normalization_856/batchnorm/mul/ReadVariableOp4batch_normalization_856/batchnorm/mul/ReadVariableOp2R
'batch_normalization_857/AssignMovingAvg'batch_normalization_857/AssignMovingAvg2p
6batch_normalization_857/AssignMovingAvg/ReadVariableOp6batch_normalization_857/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_857/AssignMovingAvg_1)batch_normalization_857/AssignMovingAvg_12t
8batch_normalization_857/AssignMovingAvg_1/ReadVariableOp8batch_normalization_857/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_857/batchnorm/ReadVariableOp0batch_normalization_857/batchnorm/ReadVariableOp2l
4batch_normalization_857/batchnorm/mul/ReadVariableOp4batch_normalization_857/batchnorm/mul/ReadVariableOp2R
'batch_normalization_858/AssignMovingAvg'batch_normalization_858/AssignMovingAvg2p
6batch_normalization_858/AssignMovingAvg/ReadVariableOp6batch_normalization_858/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_858/AssignMovingAvg_1)batch_normalization_858/AssignMovingAvg_12t
8batch_normalization_858/AssignMovingAvg_1/ReadVariableOp8batch_normalization_858/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_858/batchnorm/ReadVariableOp0batch_normalization_858/batchnorm/ReadVariableOp2l
4batch_normalization_858/batchnorm/mul/ReadVariableOp4batch_normalization_858/batchnorm/mul/ReadVariableOp2R
'batch_normalization_859/AssignMovingAvg'batch_normalization_859/AssignMovingAvg2p
6batch_normalization_859/AssignMovingAvg/ReadVariableOp6batch_normalization_859/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_859/AssignMovingAvg_1)batch_normalization_859/AssignMovingAvg_12t
8batch_normalization_859/AssignMovingAvg_1/ReadVariableOp8batch_normalization_859/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_859/batchnorm/ReadVariableOp0batch_normalization_859/batchnorm/ReadVariableOp2l
4batch_normalization_859/batchnorm/mul/ReadVariableOp4batch_normalization_859/batchnorm/mul/ReadVariableOp2R
'batch_normalization_860/AssignMovingAvg'batch_normalization_860/AssignMovingAvg2p
6batch_normalization_860/AssignMovingAvg/ReadVariableOp6batch_normalization_860/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_860/AssignMovingAvg_1)batch_normalization_860/AssignMovingAvg_12t
8batch_normalization_860/AssignMovingAvg_1/ReadVariableOp8batch_normalization_860/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_860/batchnorm/ReadVariableOp0batch_normalization_860/batchnorm/ReadVariableOp2l
4batch_normalization_860/batchnorm/mul/ReadVariableOp4batch_normalization_860/batchnorm/mul/ReadVariableOp2R
'batch_normalization_861/AssignMovingAvg'batch_normalization_861/AssignMovingAvg2p
6batch_normalization_861/AssignMovingAvg/ReadVariableOp6batch_normalization_861/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_861/AssignMovingAvg_1)batch_normalization_861/AssignMovingAvg_12t
8batch_normalization_861/AssignMovingAvg_1/ReadVariableOp8batch_normalization_861/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_861/batchnorm/ReadVariableOp0batch_normalization_861/batchnorm/ReadVariableOp2l
4batch_normalization_861/batchnorm/mul/ReadVariableOp4batch_normalization_861/batchnorm/mul/ReadVariableOp2R
'batch_normalization_862/AssignMovingAvg'batch_normalization_862/AssignMovingAvg2p
6batch_normalization_862/AssignMovingAvg/ReadVariableOp6batch_normalization_862/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_862/AssignMovingAvg_1)batch_normalization_862/AssignMovingAvg_12t
8batch_normalization_862/AssignMovingAvg_1/ReadVariableOp8batch_normalization_862/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_862/batchnorm/ReadVariableOp0batch_normalization_862/batchnorm/ReadVariableOp2l
4batch_normalization_862/batchnorm/mul/ReadVariableOp4batch_normalization_862/batchnorm/mul/ReadVariableOp2R
'batch_normalization_863/AssignMovingAvg'batch_normalization_863/AssignMovingAvg2p
6batch_normalization_863/AssignMovingAvg/ReadVariableOp6batch_normalization_863/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_863/AssignMovingAvg_1)batch_normalization_863/AssignMovingAvg_12t
8batch_normalization_863/AssignMovingAvg_1/ReadVariableOp8batch_normalization_863/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_863/batchnorm/ReadVariableOp0batch_normalization_863/batchnorm/ReadVariableOp2l
4batch_normalization_863/batchnorm/mul/ReadVariableOp4batch_normalization_863/batchnorm/mul/ReadVariableOp2R
'batch_normalization_864/AssignMovingAvg'batch_normalization_864/AssignMovingAvg2p
6batch_normalization_864/AssignMovingAvg/ReadVariableOp6batch_normalization_864/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_864/AssignMovingAvg_1)batch_normalization_864/AssignMovingAvg_12t
8batch_normalization_864/AssignMovingAvg_1/ReadVariableOp8batch_normalization_864/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_864/batchnorm/ReadVariableOp0batch_normalization_864/batchnorm/ReadVariableOp2l
4batch_normalization_864/batchnorm/mul/ReadVariableOp4batch_normalization_864/batchnorm/mul/ReadVariableOp2R
'batch_normalization_865/AssignMovingAvg'batch_normalization_865/AssignMovingAvg2p
6batch_normalization_865/AssignMovingAvg/ReadVariableOp6batch_normalization_865/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_865/AssignMovingAvg_1)batch_normalization_865/AssignMovingAvg_12t
8batch_normalization_865/AssignMovingAvg_1/ReadVariableOp8batch_normalization_865/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_865/batchnorm/ReadVariableOp0batch_normalization_865/batchnorm/ReadVariableOp2l
4batch_normalization_865/batchnorm/mul/ReadVariableOp4batch_normalization_865/batchnorm/mul/ReadVariableOp2D
 dense_945/BiasAdd/ReadVariableOp dense_945/BiasAdd/ReadVariableOp2B
dense_945/MatMul/ReadVariableOpdense_945/MatMul/ReadVariableOp2D
 dense_946/BiasAdd/ReadVariableOp dense_946/BiasAdd/ReadVariableOp2B
dense_946/MatMul/ReadVariableOpdense_946/MatMul/ReadVariableOp2D
 dense_947/BiasAdd/ReadVariableOp dense_947/BiasAdd/ReadVariableOp2B
dense_947/MatMul/ReadVariableOpdense_947/MatMul/ReadVariableOp2D
 dense_948/BiasAdd/ReadVariableOp dense_948/BiasAdd/ReadVariableOp2B
dense_948/MatMul/ReadVariableOpdense_948/MatMul/ReadVariableOp2D
 dense_949/BiasAdd/ReadVariableOp dense_949/BiasAdd/ReadVariableOp2B
dense_949/MatMul/ReadVariableOpdense_949/MatMul/ReadVariableOp2D
 dense_950/BiasAdd/ReadVariableOp dense_950/BiasAdd/ReadVariableOp2B
dense_950/MatMul/ReadVariableOpdense_950/MatMul/ReadVariableOp2D
 dense_951/BiasAdd/ReadVariableOp dense_951/BiasAdd/ReadVariableOp2B
dense_951/MatMul/ReadVariableOpdense_951/MatMul/ReadVariableOp2D
 dense_952/BiasAdd/ReadVariableOp dense_952/BiasAdd/ReadVariableOp2B
dense_952/MatMul/ReadVariableOpdense_952/MatMul/ReadVariableOp2D
 dense_953/BiasAdd/ReadVariableOp dense_953/BiasAdd/ReadVariableOp2B
dense_953/MatMul/ReadVariableOpdense_953/MatMul/ReadVariableOp2D
 dense_954/BiasAdd/ReadVariableOp dense_954/BiasAdd/ReadVariableOp2B
dense_954/MatMul/ReadVariableOpdense_954/MatMul/ReadVariableOp2D
 dense_955/BiasAdd/ReadVariableOp dense_955/BiasAdd/ReadVariableOp2B
dense_955/MatMul/ReadVariableOpdense_955/MatMul/ReadVariableOp2D
 dense_956/BiasAdd/ReadVariableOp dense_956/BiasAdd/ReadVariableOp2B
dense_956/MatMul/ReadVariableOpdense_956/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:

Ù>
I__inference_sequential_90_layer_call_and_return_conditional_losses_838982

inputs
normalization_90_sub_y
normalization_90_sqrt_x:
(dense_945_matmul_readvariableop_resource:h7
)dense_945_biasadd_readvariableop_resource:hG
9batch_normalization_855_batchnorm_readvariableop_resource:hK
=batch_normalization_855_batchnorm_mul_readvariableop_resource:hI
;batch_normalization_855_batchnorm_readvariableop_1_resource:hI
;batch_normalization_855_batchnorm_readvariableop_2_resource:h:
(dense_946_matmul_readvariableop_resource:hh7
)dense_946_biasadd_readvariableop_resource:hG
9batch_normalization_856_batchnorm_readvariableop_resource:hK
=batch_normalization_856_batchnorm_mul_readvariableop_resource:hI
;batch_normalization_856_batchnorm_readvariableop_1_resource:hI
;batch_normalization_856_batchnorm_readvariableop_2_resource:h:
(dense_947_matmul_readvariableop_resource:hh7
)dense_947_biasadd_readvariableop_resource:hG
9batch_normalization_857_batchnorm_readvariableop_resource:hK
=batch_normalization_857_batchnorm_mul_readvariableop_resource:hI
;batch_normalization_857_batchnorm_readvariableop_1_resource:hI
;batch_normalization_857_batchnorm_readvariableop_2_resource:h:
(dense_948_matmul_readvariableop_resource:hh7
)dense_948_biasadd_readvariableop_resource:hG
9batch_normalization_858_batchnorm_readvariableop_resource:hK
=batch_normalization_858_batchnorm_mul_readvariableop_resource:hI
;batch_normalization_858_batchnorm_readvariableop_1_resource:hI
;batch_normalization_858_batchnorm_readvariableop_2_resource:h:
(dense_949_matmul_readvariableop_resource:hh7
)dense_949_biasadd_readvariableop_resource:hG
9batch_normalization_859_batchnorm_readvariableop_resource:hK
=batch_normalization_859_batchnorm_mul_readvariableop_resource:hI
;batch_normalization_859_batchnorm_readvariableop_1_resource:hI
;batch_normalization_859_batchnorm_readvariableop_2_resource:h:
(dense_950_matmul_readvariableop_resource:h/7
)dense_950_biasadd_readvariableop_resource:/G
9batch_normalization_860_batchnorm_readvariableop_resource:/K
=batch_normalization_860_batchnorm_mul_readvariableop_resource:/I
;batch_normalization_860_batchnorm_readvariableop_1_resource:/I
;batch_normalization_860_batchnorm_readvariableop_2_resource:/:
(dense_951_matmul_readvariableop_resource://7
)dense_951_biasadd_readvariableop_resource:/G
9batch_normalization_861_batchnorm_readvariableop_resource:/K
=batch_normalization_861_batchnorm_mul_readvariableop_resource:/I
;batch_normalization_861_batchnorm_readvariableop_1_resource:/I
;batch_normalization_861_batchnorm_readvariableop_2_resource:/:
(dense_952_matmul_readvariableop_resource:/7
)dense_952_biasadd_readvariableop_resource:G
9batch_normalization_862_batchnorm_readvariableop_resource:K
=batch_normalization_862_batchnorm_mul_readvariableop_resource:I
;batch_normalization_862_batchnorm_readvariableop_1_resource:I
;batch_normalization_862_batchnorm_readvariableop_2_resource::
(dense_953_matmul_readvariableop_resource:7
)dense_953_biasadd_readvariableop_resource:G
9batch_normalization_863_batchnorm_readvariableop_resource:K
=batch_normalization_863_batchnorm_mul_readvariableop_resource:I
;batch_normalization_863_batchnorm_readvariableop_1_resource:I
;batch_normalization_863_batchnorm_readvariableop_2_resource::
(dense_954_matmul_readvariableop_resource:7
)dense_954_biasadd_readvariableop_resource:G
9batch_normalization_864_batchnorm_readvariableop_resource:K
=batch_normalization_864_batchnorm_mul_readvariableop_resource:I
;batch_normalization_864_batchnorm_readvariableop_1_resource:I
;batch_normalization_864_batchnorm_readvariableop_2_resource::
(dense_955_matmul_readvariableop_resource:7
)dense_955_biasadd_readvariableop_resource:G
9batch_normalization_865_batchnorm_readvariableop_resource:K
=batch_normalization_865_batchnorm_mul_readvariableop_resource:I
;batch_normalization_865_batchnorm_readvariableop_1_resource:I
;batch_normalization_865_batchnorm_readvariableop_2_resource::
(dense_956_matmul_readvariableop_resource:7
)dense_956_biasadd_readvariableop_resource:
identity¢0batch_normalization_855/batchnorm/ReadVariableOp¢2batch_normalization_855/batchnorm/ReadVariableOp_1¢2batch_normalization_855/batchnorm/ReadVariableOp_2¢4batch_normalization_855/batchnorm/mul/ReadVariableOp¢0batch_normalization_856/batchnorm/ReadVariableOp¢2batch_normalization_856/batchnorm/ReadVariableOp_1¢2batch_normalization_856/batchnorm/ReadVariableOp_2¢4batch_normalization_856/batchnorm/mul/ReadVariableOp¢0batch_normalization_857/batchnorm/ReadVariableOp¢2batch_normalization_857/batchnorm/ReadVariableOp_1¢2batch_normalization_857/batchnorm/ReadVariableOp_2¢4batch_normalization_857/batchnorm/mul/ReadVariableOp¢0batch_normalization_858/batchnorm/ReadVariableOp¢2batch_normalization_858/batchnorm/ReadVariableOp_1¢2batch_normalization_858/batchnorm/ReadVariableOp_2¢4batch_normalization_858/batchnorm/mul/ReadVariableOp¢0batch_normalization_859/batchnorm/ReadVariableOp¢2batch_normalization_859/batchnorm/ReadVariableOp_1¢2batch_normalization_859/batchnorm/ReadVariableOp_2¢4batch_normalization_859/batchnorm/mul/ReadVariableOp¢0batch_normalization_860/batchnorm/ReadVariableOp¢2batch_normalization_860/batchnorm/ReadVariableOp_1¢2batch_normalization_860/batchnorm/ReadVariableOp_2¢4batch_normalization_860/batchnorm/mul/ReadVariableOp¢0batch_normalization_861/batchnorm/ReadVariableOp¢2batch_normalization_861/batchnorm/ReadVariableOp_1¢2batch_normalization_861/batchnorm/ReadVariableOp_2¢4batch_normalization_861/batchnorm/mul/ReadVariableOp¢0batch_normalization_862/batchnorm/ReadVariableOp¢2batch_normalization_862/batchnorm/ReadVariableOp_1¢2batch_normalization_862/batchnorm/ReadVariableOp_2¢4batch_normalization_862/batchnorm/mul/ReadVariableOp¢0batch_normalization_863/batchnorm/ReadVariableOp¢2batch_normalization_863/batchnorm/ReadVariableOp_1¢2batch_normalization_863/batchnorm/ReadVariableOp_2¢4batch_normalization_863/batchnorm/mul/ReadVariableOp¢0batch_normalization_864/batchnorm/ReadVariableOp¢2batch_normalization_864/batchnorm/ReadVariableOp_1¢2batch_normalization_864/batchnorm/ReadVariableOp_2¢4batch_normalization_864/batchnorm/mul/ReadVariableOp¢0batch_normalization_865/batchnorm/ReadVariableOp¢2batch_normalization_865/batchnorm/ReadVariableOp_1¢2batch_normalization_865/batchnorm/ReadVariableOp_2¢4batch_normalization_865/batchnorm/mul/ReadVariableOp¢ dense_945/BiasAdd/ReadVariableOp¢dense_945/MatMul/ReadVariableOp¢ dense_946/BiasAdd/ReadVariableOp¢dense_946/MatMul/ReadVariableOp¢ dense_947/BiasAdd/ReadVariableOp¢dense_947/MatMul/ReadVariableOp¢ dense_948/BiasAdd/ReadVariableOp¢dense_948/MatMul/ReadVariableOp¢ dense_949/BiasAdd/ReadVariableOp¢dense_949/MatMul/ReadVariableOp¢ dense_950/BiasAdd/ReadVariableOp¢dense_950/MatMul/ReadVariableOp¢ dense_951/BiasAdd/ReadVariableOp¢dense_951/MatMul/ReadVariableOp¢ dense_952/BiasAdd/ReadVariableOp¢dense_952/MatMul/ReadVariableOp¢ dense_953/BiasAdd/ReadVariableOp¢dense_953/MatMul/ReadVariableOp¢ dense_954/BiasAdd/ReadVariableOp¢dense_954/MatMul/ReadVariableOp¢ dense_955/BiasAdd/ReadVariableOp¢dense_955/MatMul/ReadVariableOp¢ dense_956/BiasAdd/ReadVariableOp¢dense_956/MatMul/ReadVariableOpm
normalization_90/subSubinputsnormalization_90_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_90/SqrtSqrtnormalization_90_sqrt_x*
T0*
_output_shapes

:_
normalization_90/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_90/MaximumMaximumnormalization_90/Sqrt:y:0#normalization_90/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_90/truedivRealDivnormalization_90/sub:z:0normalization_90/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_945/MatMul/ReadVariableOpReadVariableOp(dense_945_matmul_readvariableop_resource*
_output_shapes

:h*
dtype0
dense_945/MatMulMatMulnormalization_90/truediv:z:0'dense_945/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 dense_945/BiasAdd/ReadVariableOpReadVariableOp)dense_945_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0
dense_945/BiasAddBiasAdddense_945/MatMul:product:0(dense_945/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¦
0batch_normalization_855/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_855_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0l
'batch_normalization_855/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_855/batchnorm/addAddV28batch_normalization_855/batchnorm/ReadVariableOp:value:00batch_normalization_855/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
'batch_normalization_855/batchnorm/RsqrtRsqrt)batch_normalization_855/batchnorm/add:z:0*
T0*
_output_shapes
:h®
4batch_normalization_855/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_855_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0¼
%batch_normalization_855/batchnorm/mulMul+batch_normalization_855/batchnorm/Rsqrt:y:0<batch_normalization_855/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:h§
'batch_normalization_855/batchnorm/mul_1Muldense_945/BiasAdd:output:0)batch_normalization_855/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhª
2batch_normalization_855/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_855_batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0º
'batch_normalization_855/batchnorm/mul_2Mul:batch_normalization_855/batchnorm/ReadVariableOp_1:value:0)batch_normalization_855/batchnorm/mul:z:0*
T0*
_output_shapes
:hª
2batch_normalization_855/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_855_batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0º
%batch_normalization_855/batchnorm/subSub:batch_normalization_855/batchnorm/ReadVariableOp_2:value:0+batch_normalization_855/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hº
'batch_normalization_855/batchnorm/add_1AddV2+batch_normalization_855/batchnorm/mul_1:z:0)batch_normalization_855/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
leaky_re_lu_855/LeakyRelu	LeakyRelu+batch_normalization_855/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>
dense_946/MatMul/ReadVariableOpReadVariableOp(dense_946_matmul_readvariableop_resource*
_output_shapes

:hh*
dtype0
dense_946/MatMulMatMul'leaky_re_lu_855/LeakyRelu:activations:0'dense_946/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 dense_946/BiasAdd/ReadVariableOpReadVariableOp)dense_946_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0
dense_946/BiasAddBiasAdddense_946/MatMul:product:0(dense_946/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¦
0batch_normalization_856/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_856_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0l
'batch_normalization_856/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_856/batchnorm/addAddV28batch_normalization_856/batchnorm/ReadVariableOp:value:00batch_normalization_856/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
'batch_normalization_856/batchnorm/RsqrtRsqrt)batch_normalization_856/batchnorm/add:z:0*
T0*
_output_shapes
:h®
4batch_normalization_856/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_856_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0¼
%batch_normalization_856/batchnorm/mulMul+batch_normalization_856/batchnorm/Rsqrt:y:0<batch_normalization_856/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:h§
'batch_normalization_856/batchnorm/mul_1Muldense_946/BiasAdd:output:0)batch_normalization_856/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhª
2batch_normalization_856/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_856_batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0º
'batch_normalization_856/batchnorm/mul_2Mul:batch_normalization_856/batchnorm/ReadVariableOp_1:value:0)batch_normalization_856/batchnorm/mul:z:0*
T0*
_output_shapes
:hª
2batch_normalization_856/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_856_batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0º
%batch_normalization_856/batchnorm/subSub:batch_normalization_856/batchnorm/ReadVariableOp_2:value:0+batch_normalization_856/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hº
'batch_normalization_856/batchnorm/add_1AddV2+batch_normalization_856/batchnorm/mul_1:z:0)batch_normalization_856/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
leaky_re_lu_856/LeakyRelu	LeakyRelu+batch_normalization_856/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>
dense_947/MatMul/ReadVariableOpReadVariableOp(dense_947_matmul_readvariableop_resource*
_output_shapes

:hh*
dtype0
dense_947/MatMulMatMul'leaky_re_lu_856/LeakyRelu:activations:0'dense_947/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 dense_947/BiasAdd/ReadVariableOpReadVariableOp)dense_947_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0
dense_947/BiasAddBiasAdddense_947/MatMul:product:0(dense_947/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¦
0batch_normalization_857/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_857_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0l
'batch_normalization_857/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_857/batchnorm/addAddV28batch_normalization_857/batchnorm/ReadVariableOp:value:00batch_normalization_857/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
'batch_normalization_857/batchnorm/RsqrtRsqrt)batch_normalization_857/batchnorm/add:z:0*
T0*
_output_shapes
:h®
4batch_normalization_857/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_857_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0¼
%batch_normalization_857/batchnorm/mulMul+batch_normalization_857/batchnorm/Rsqrt:y:0<batch_normalization_857/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:h§
'batch_normalization_857/batchnorm/mul_1Muldense_947/BiasAdd:output:0)batch_normalization_857/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhª
2batch_normalization_857/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_857_batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0º
'batch_normalization_857/batchnorm/mul_2Mul:batch_normalization_857/batchnorm/ReadVariableOp_1:value:0)batch_normalization_857/batchnorm/mul:z:0*
T0*
_output_shapes
:hª
2batch_normalization_857/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_857_batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0º
%batch_normalization_857/batchnorm/subSub:batch_normalization_857/batchnorm/ReadVariableOp_2:value:0+batch_normalization_857/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hº
'batch_normalization_857/batchnorm/add_1AddV2+batch_normalization_857/batchnorm/mul_1:z:0)batch_normalization_857/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
leaky_re_lu_857/LeakyRelu	LeakyRelu+batch_normalization_857/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>
dense_948/MatMul/ReadVariableOpReadVariableOp(dense_948_matmul_readvariableop_resource*
_output_shapes

:hh*
dtype0
dense_948/MatMulMatMul'leaky_re_lu_857/LeakyRelu:activations:0'dense_948/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 dense_948/BiasAdd/ReadVariableOpReadVariableOp)dense_948_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0
dense_948/BiasAddBiasAdddense_948/MatMul:product:0(dense_948/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¦
0batch_normalization_858/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_858_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0l
'batch_normalization_858/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_858/batchnorm/addAddV28batch_normalization_858/batchnorm/ReadVariableOp:value:00batch_normalization_858/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
'batch_normalization_858/batchnorm/RsqrtRsqrt)batch_normalization_858/batchnorm/add:z:0*
T0*
_output_shapes
:h®
4batch_normalization_858/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_858_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0¼
%batch_normalization_858/batchnorm/mulMul+batch_normalization_858/batchnorm/Rsqrt:y:0<batch_normalization_858/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:h§
'batch_normalization_858/batchnorm/mul_1Muldense_948/BiasAdd:output:0)batch_normalization_858/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhª
2batch_normalization_858/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_858_batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0º
'batch_normalization_858/batchnorm/mul_2Mul:batch_normalization_858/batchnorm/ReadVariableOp_1:value:0)batch_normalization_858/batchnorm/mul:z:0*
T0*
_output_shapes
:hª
2batch_normalization_858/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_858_batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0º
%batch_normalization_858/batchnorm/subSub:batch_normalization_858/batchnorm/ReadVariableOp_2:value:0+batch_normalization_858/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hº
'batch_normalization_858/batchnorm/add_1AddV2+batch_normalization_858/batchnorm/mul_1:z:0)batch_normalization_858/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
leaky_re_lu_858/LeakyRelu	LeakyRelu+batch_normalization_858/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>
dense_949/MatMul/ReadVariableOpReadVariableOp(dense_949_matmul_readvariableop_resource*
_output_shapes

:hh*
dtype0
dense_949/MatMulMatMul'leaky_re_lu_858/LeakyRelu:activations:0'dense_949/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 dense_949/BiasAdd/ReadVariableOpReadVariableOp)dense_949_biasadd_readvariableop_resource*
_output_shapes
:h*
dtype0
dense_949/BiasAddBiasAdddense_949/MatMul:product:0(dense_949/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh¦
0batch_normalization_859/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_859_batchnorm_readvariableop_resource*
_output_shapes
:h*
dtype0l
'batch_normalization_859/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_859/batchnorm/addAddV28batch_normalization_859/batchnorm/ReadVariableOp:value:00batch_normalization_859/batchnorm/add/y:output:0*
T0*
_output_shapes
:h
'batch_normalization_859/batchnorm/RsqrtRsqrt)batch_normalization_859/batchnorm/add:z:0*
T0*
_output_shapes
:h®
4batch_normalization_859/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_859_batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0¼
%batch_normalization_859/batchnorm/mulMul+batch_normalization_859/batchnorm/Rsqrt:y:0<batch_normalization_859/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:h§
'batch_normalization_859/batchnorm/mul_1Muldense_949/BiasAdd:output:0)batch_normalization_859/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhª
2batch_normalization_859/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_859_batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0º
'batch_normalization_859/batchnorm/mul_2Mul:batch_normalization_859/batchnorm/ReadVariableOp_1:value:0)batch_normalization_859/batchnorm/mul:z:0*
T0*
_output_shapes
:hª
2batch_normalization_859/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_859_batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0º
%batch_normalization_859/batchnorm/subSub:batch_normalization_859/batchnorm/ReadVariableOp_2:value:0+batch_normalization_859/batchnorm/mul_2:z:0*
T0*
_output_shapes
:hº
'batch_normalization_859/batchnorm/add_1AddV2+batch_normalization_859/batchnorm/mul_1:z:0)batch_normalization_859/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
leaky_re_lu_859/LeakyRelu	LeakyRelu+batch_normalization_859/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>
dense_950/MatMul/ReadVariableOpReadVariableOp(dense_950_matmul_readvariableop_resource*
_output_shapes

:h/*
dtype0
dense_950/MatMulMatMul'leaky_re_lu_859/LeakyRelu:activations:0'dense_950/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 dense_950/BiasAdd/ReadVariableOpReadVariableOp)dense_950_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_950/BiasAddBiasAdddense_950/MatMul:product:0(dense_950/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/¦
0batch_normalization_860/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_860_batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0l
'batch_normalization_860/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_860/batchnorm/addAddV28batch_normalization_860/batchnorm/ReadVariableOp:value:00batch_normalization_860/batchnorm/add/y:output:0*
T0*
_output_shapes
:/
'batch_normalization_860/batchnorm/RsqrtRsqrt)batch_normalization_860/batchnorm/add:z:0*
T0*
_output_shapes
:/®
4batch_normalization_860/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_860_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0¼
%batch_normalization_860/batchnorm/mulMul+batch_normalization_860/batchnorm/Rsqrt:y:0<batch_normalization_860/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/§
'batch_normalization_860/batchnorm/mul_1Muldense_950/BiasAdd:output:0)batch_normalization_860/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/ª
2batch_normalization_860/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_860_batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0º
'batch_normalization_860/batchnorm/mul_2Mul:batch_normalization_860/batchnorm/ReadVariableOp_1:value:0)batch_normalization_860/batchnorm/mul:z:0*
T0*
_output_shapes
:/ª
2batch_normalization_860/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_860_batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0º
%batch_normalization_860/batchnorm/subSub:batch_normalization_860/batchnorm/ReadVariableOp_2:value:0+batch_normalization_860/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/º
'batch_normalization_860/batchnorm/add_1AddV2+batch_normalization_860/batchnorm/mul_1:z:0)batch_normalization_860/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
leaky_re_lu_860/LeakyRelu	LeakyRelu+batch_normalization_860/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*
alpha%>
dense_951/MatMul/ReadVariableOpReadVariableOp(dense_951_matmul_readvariableop_resource*
_output_shapes

://*
dtype0
dense_951/MatMulMatMul'leaky_re_lu_860/LeakyRelu:activations:0'dense_951/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 dense_951/BiasAdd/ReadVariableOpReadVariableOp)dense_951_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_951/BiasAddBiasAdddense_951/MatMul:product:0(dense_951/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/¦
0batch_normalization_861/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_861_batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0l
'batch_normalization_861/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_861/batchnorm/addAddV28batch_normalization_861/batchnorm/ReadVariableOp:value:00batch_normalization_861/batchnorm/add/y:output:0*
T0*
_output_shapes
:/
'batch_normalization_861/batchnorm/RsqrtRsqrt)batch_normalization_861/batchnorm/add:z:0*
T0*
_output_shapes
:/®
4batch_normalization_861/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_861_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0¼
%batch_normalization_861/batchnorm/mulMul+batch_normalization_861/batchnorm/Rsqrt:y:0<batch_normalization_861/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/§
'batch_normalization_861/batchnorm/mul_1Muldense_951/BiasAdd:output:0)batch_normalization_861/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/ª
2batch_normalization_861/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_861_batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0º
'batch_normalization_861/batchnorm/mul_2Mul:batch_normalization_861/batchnorm/ReadVariableOp_1:value:0)batch_normalization_861/batchnorm/mul:z:0*
T0*
_output_shapes
:/ª
2batch_normalization_861/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_861_batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0º
%batch_normalization_861/batchnorm/subSub:batch_normalization_861/batchnorm/ReadVariableOp_2:value:0+batch_normalization_861/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/º
'batch_normalization_861/batchnorm/add_1AddV2+batch_normalization_861/batchnorm/mul_1:z:0)batch_normalization_861/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
leaky_re_lu_861/LeakyRelu	LeakyRelu+batch_normalization_861/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*
alpha%>
dense_952/MatMul/ReadVariableOpReadVariableOp(dense_952_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0
dense_952/MatMulMatMul'leaky_re_lu_861/LeakyRelu:activations:0'dense_952/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_952/BiasAdd/ReadVariableOpReadVariableOp)dense_952_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_952/BiasAddBiasAdddense_952/MatMul:product:0(dense_952/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_862/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_862_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_862/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_862/batchnorm/addAddV28batch_normalization_862/batchnorm/ReadVariableOp:value:00batch_normalization_862/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_862/batchnorm/RsqrtRsqrt)batch_normalization_862/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_862/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_862_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_862/batchnorm/mulMul+batch_normalization_862/batchnorm/Rsqrt:y:0<batch_normalization_862/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_862/batchnorm/mul_1Muldense_952/BiasAdd:output:0)batch_normalization_862/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_862/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_862_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_862/batchnorm/mul_2Mul:batch_normalization_862/batchnorm/ReadVariableOp_1:value:0)batch_normalization_862/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_862/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_862_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_862/batchnorm/subSub:batch_normalization_862/batchnorm/ReadVariableOp_2:value:0+batch_normalization_862/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_862/batchnorm/add_1AddV2+batch_normalization_862/batchnorm/mul_1:z:0)batch_normalization_862/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_862/LeakyRelu	LeakyRelu+batch_normalization_862/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_953/MatMul/ReadVariableOpReadVariableOp(dense_953_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_953/MatMulMatMul'leaky_re_lu_862/LeakyRelu:activations:0'dense_953/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_953/BiasAdd/ReadVariableOpReadVariableOp)dense_953_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_953/BiasAddBiasAdddense_953/MatMul:product:0(dense_953/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_863/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_863_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_863/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_863/batchnorm/addAddV28batch_normalization_863/batchnorm/ReadVariableOp:value:00batch_normalization_863/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_863/batchnorm/RsqrtRsqrt)batch_normalization_863/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_863/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_863_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_863/batchnorm/mulMul+batch_normalization_863/batchnorm/Rsqrt:y:0<batch_normalization_863/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_863/batchnorm/mul_1Muldense_953/BiasAdd:output:0)batch_normalization_863/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_863/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_863_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_863/batchnorm/mul_2Mul:batch_normalization_863/batchnorm/ReadVariableOp_1:value:0)batch_normalization_863/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_863/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_863_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_863/batchnorm/subSub:batch_normalization_863/batchnorm/ReadVariableOp_2:value:0+batch_normalization_863/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_863/batchnorm/add_1AddV2+batch_normalization_863/batchnorm/mul_1:z:0)batch_normalization_863/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_863/LeakyRelu	LeakyRelu+batch_normalization_863/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_954/MatMul/ReadVariableOpReadVariableOp(dense_954_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_954/MatMulMatMul'leaky_re_lu_863/LeakyRelu:activations:0'dense_954/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_954/BiasAdd/ReadVariableOpReadVariableOp)dense_954_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_954/BiasAddBiasAdddense_954/MatMul:product:0(dense_954/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_864/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_864_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_864/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_864/batchnorm/addAddV28batch_normalization_864/batchnorm/ReadVariableOp:value:00batch_normalization_864/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_864/batchnorm/RsqrtRsqrt)batch_normalization_864/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_864/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_864_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_864/batchnorm/mulMul+batch_normalization_864/batchnorm/Rsqrt:y:0<batch_normalization_864/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_864/batchnorm/mul_1Muldense_954/BiasAdd:output:0)batch_normalization_864/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_864/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_864_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_864/batchnorm/mul_2Mul:batch_normalization_864/batchnorm/ReadVariableOp_1:value:0)batch_normalization_864/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_864/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_864_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_864/batchnorm/subSub:batch_normalization_864/batchnorm/ReadVariableOp_2:value:0+batch_normalization_864/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_864/batchnorm/add_1AddV2+batch_normalization_864/batchnorm/mul_1:z:0)batch_normalization_864/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_864/LeakyRelu	LeakyRelu+batch_normalization_864/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_955/MatMul/ReadVariableOpReadVariableOp(dense_955_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_955/MatMulMatMul'leaky_re_lu_864/LeakyRelu:activations:0'dense_955/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_955/BiasAdd/ReadVariableOpReadVariableOp)dense_955_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_955/BiasAddBiasAdddense_955/MatMul:product:0(dense_955/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_865/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_865_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_865/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_865/batchnorm/addAddV28batch_normalization_865/batchnorm/ReadVariableOp:value:00batch_normalization_865/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_865/batchnorm/RsqrtRsqrt)batch_normalization_865/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_865/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_865_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_865/batchnorm/mulMul+batch_normalization_865/batchnorm/Rsqrt:y:0<batch_normalization_865/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_865/batchnorm/mul_1Muldense_955/BiasAdd:output:0)batch_normalization_865/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_865/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_865_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_865/batchnorm/mul_2Mul:batch_normalization_865/batchnorm/ReadVariableOp_1:value:0)batch_normalization_865/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_865/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_865_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_865/batchnorm/subSub:batch_normalization_865/batchnorm/ReadVariableOp_2:value:0+batch_normalization_865/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_865/batchnorm/add_1AddV2+batch_normalization_865/batchnorm/mul_1:z:0)batch_normalization_865/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_865/LeakyRelu	LeakyRelu+batch_normalization_865/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_956/MatMul/ReadVariableOpReadVariableOp(dense_956_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_956/MatMulMatMul'leaky_re_lu_865/LeakyRelu:activations:0'dense_956/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_956/BiasAdd/ReadVariableOpReadVariableOp)dense_956_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_956/BiasAddBiasAdddense_956/MatMul:product:0(dense_956/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_956/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp1^batch_normalization_855/batchnorm/ReadVariableOp3^batch_normalization_855/batchnorm/ReadVariableOp_13^batch_normalization_855/batchnorm/ReadVariableOp_25^batch_normalization_855/batchnorm/mul/ReadVariableOp1^batch_normalization_856/batchnorm/ReadVariableOp3^batch_normalization_856/batchnorm/ReadVariableOp_13^batch_normalization_856/batchnorm/ReadVariableOp_25^batch_normalization_856/batchnorm/mul/ReadVariableOp1^batch_normalization_857/batchnorm/ReadVariableOp3^batch_normalization_857/batchnorm/ReadVariableOp_13^batch_normalization_857/batchnorm/ReadVariableOp_25^batch_normalization_857/batchnorm/mul/ReadVariableOp1^batch_normalization_858/batchnorm/ReadVariableOp3^batch_normalization_858/batchnorm/ReadVariableOp_13^batch_normalization_858/batchnorm/ReadVariableOp_25^batch_normalization_858/batchnorm/mul/ReadVariableOp1^batch_normalization_859/batchnorm/ReadVariableOp3^batch_normalization_859/batchnorm/ReadVariableOp_13^batch_normalization_859/batchnorm/ReadVariableOp_25^batch_normalization_859/batchnorm/mul/ReadVariableOp1^batch_normalization_860/batchnorm/ReadVariableOp3^batch_normalization_860/batchnorm/ReadVariableOp_13^batch_normalization_860/batchnorm/ReadVariableOp_25^batch_normalization_860/batchnorm/mul/ReadVariableOp1^batch_normalization_861/batchnorm/ReadVariableOp3^batch_normalization_861/batchnorm/ReadVariableOp_13^batch_normalization_861/batchnorm/ReadVariableOp_25^batch_normalization_861/batchnorm/mul/ReadVariableOp1^batch_normalization_862/batchnorm/ReadVariableOp3^batch_normalization_862/batchnorm/ReadVariableOp_13^batch_normalization_862/batchnorm/ReadVariableOp_25^batch_normalization_862/batchnorm/mul/ReadVariableOp1^batch_normalization_863/batchnorm/ReadVariableOp3^batch_normalization_863/batchnorm/ReadVariableOp_13^batch_normalization_863/batchnorm/ReadVariableOp_25^batch_normalization_863/batchnorm/mul/ReadVariableOp1^batch_normalization_864/batchnorm/ReadVariableOp3^batch_normalization_864/batchnorm/ReadVariableOp_13^batch_normalization_864/batchnorm/ReadVariableOp_25^batch_normalization_864/batchnorm/mul/ReadVariableOp1^batch_normalization_865/batchnorm/ReadVariableOp3^batch_normalization_865/batchnorm/ReadVariableOp_13^batch_normalization_865/batchnorm/ReadVariableOp_25^batch_normalization_865/batchnorm/mul/ReadVariableOp!^dense_945/BiasAdd/ReadVariableOp ^dense_945/MatMul/ReadVariableOp!^dense_946/BiasAdd/ReadVariableOp ^dense_946/MatMul/ReadVariableOp!^dense_947/BiasAdd/ReadVariableOp ^dense_947/MatMul/ReadVariableOp!^dense_948/BiasAdd/ReadVariableOp ^dense_948/MatMul/ReadVariableOp!^dense_949/BiasAdd/ReadVariableOp ^dense_949/MatMul/ReadVariableOp!^dense_950/BiasAdd/ReadVariableOp ^dense_950/MatMul/ReadVariableOp!^dense_951/BiasAdd/ReadVariableOp ^dense_951/MatMul/ReadVariableOp!^dense_952/BiasAdd/ReadVariableOp ^dense_952/MatMul/ReadVariableOp!^dense_953/BiasAdd/ReadVariableOp ^dense_953/MatMul/ReadVariableOp!^dense_954/BiasAdd/ReadVariableOp ^dense_954/MatMul/ReadVariableOp!^dense_955/BiasAdd/ReadVariableOp ^dense_955/MatMul/ReadVariableOp!^dense_956/BiasAdd/ReadVariableOp ^dense_956/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_855/batchnorm/ReadVariableOp0batch_normalization_855/batchnorm/ReadVariableOp2h
2batch_normalization_855/batchnorm/ReadVariableOp_12batch_normalization_855/batchnorm/ReadVariableOp_12h
2batch_normalization_855/batchnorm/ReadVariableOp_22batch_normalization_855/batchnorm/ReadVariableOp_22l
4batch_normalization_855/batchnorm/mul/ReadVariableOp4batch_normalization_855/batchnorm/mul/ReadVariableOp2d
0batch_normalization_856/batchnorm/ReadVariableOp0batch_normalization_856/batchnorm/ReadVariableOp2h
2batch_normalization_856/batchnorm/ReadVariableOp_12batch_normalization_856/batchnorm/ReadVariableOp_12h
2batch_normalization_856/batchnorm/ReadVariableOp_22batch_normalization_856/batchnorm/ReadVariableOp_22l
4batch_normalization_856/batchnorm/mul/ReadVariableOp4batch_normalization_856/batchnorm/mul/ReadVariableOp2d
0batch_normalization_857/batchnorm/ReadVariableOp0batch_normalization_857/batchnorm/ReadVariableOp2h
2batch_normalization_857/batchnorm/ReadVariableOp_12batch_normalization_857/batchnorm/ReadVariableOp_12h
2batch_normalization_857/batchnorm/ReadVariableOp_22batch_normalization_857/batchnorm/ReadVariableOp_22l
4batch_normalization_857/batchnorm/mul/ReadVariableOp4batch_normalization_857/batchnorm/mul/ReadVariableOp2d
0batch_normalization_858/batchnorm/ReadVariableOp0batch_normalization_858/batchnorm/ReadVariableOp2h
2batch_normalization_858/batchnorm/ReadVariableOp_12batch_normalization_858/batchnorm/ReadVariableOp_12h
2batch_normalization_858/batchnorm/ReadVariableOp_22batch_normalization_858/batchnorm/ReadVariableOp_22l
4batch_normalization_858/batchnorm/mul/ReadVariableOp4batch_normalization_858/batchnorm/mul/ReadVariableOp2d
0batch_normalization_859/batchnorm/ReadVariableOp0batch_normalization_859/batchnorm/ReadVariableOp2h
2batch_normalization_859/batchnorm/ReadVariableOp_12batch_normalization_859/batchnorm/ReadVariableOp_12h
2batch_normalization_859/batchnorm/ReadVariableOp_22batch_normalization_859/batchnorm/ReadVariableOp_22l
4batch_normalization_859/batchnorm/mul/ReadVariableOp4batch_normalization_859/batchnorm/mul/ReadVariableOp2d
0batch_normalization_860/batchnorm/ReadVariableOp0batch_normalization_860/batchnorm/ReadVariableOp2h
2batch_normalization_860/batchnorm/ReadVariableOp_12batch_normalization_860/batchnorm/ReadVariableOp_12h
2batch_normalization_860/batchnorm/ReadVariableOp_22batch_normalization_860/batchnorm/ReadVariableOp_22l
4batch_normalization_860/batchnorm/mul/ReadVariableOp4batch_normalization_860/batchnorm/mul/ReadVariableOp2d
0batch_normalization_861/batchnorm/ReadVariableOp0batch_normalization_861/batchnorm/ReadVariableOp2h
2batch_normalization_861/batchnorm/ReadVariableOp_12batch_normalization_861/batchnorm/ReadVariableOp_12h
2batch_normalization_861/batchnorm/ReadVariableOp_22batch_normalization_861/batchnorm/ReadVariableOp_22l
4batch_normalization_861/batchnorm/mul/ReadVariableOp4batch_normalization_861/batchnorm/mul/ReadVariableOp2d
0batch_normalization_862/batchnorm/ReadVariableOp0batch_normalization_862/batchnorm/ReadVariableOp2h
2batch_normalization_862/batchnorm/ReadVariableOp_12batch_normalization_862/batchnorm/ReadVariableOp_12h
2batch_normalization_862/batchnorm/ReadVariableOp_22batch_normalization_862/batchnorm/ReadVariableOp_22l
4batch_normalization_862/batchnorm/mul/ReadVariableOp4batch_normalization_862/batchnorm/mul/ReadVariableOp2d
0batch_normalization_863/batchnorm/ReadVariableOp0batch_normalization_863/batchnorm/ReadVariableOp2h
2batch_normalization_863/batchnorm/ReadVariableOp_12batch_normalization_863/batchnorm/ReadVariableOp_12h
2batch_normalization_863/batchnorm/ReadVariableOp_22batch_normalization_863/batchnorm/ReadVariableOp_22l
4batch_normalization_863/batchnorm/mul/ReadVariableOp4batch_normalization_863/batchnorm/mul/ReadVariableOp2d
0batch_normalization_864/batchnorm/ReadVariableOp0batch_normalization_864/batchnorm/ReadVariableOp2h
2batch_normalization_864/batchnorm/ReadVariableOp_12batch_normalization_864/batchnorm/ReadVariableOp_12h
2batch_normalization_864/batchnorm/ReadVariableOp_22batch_normalization_864/batchnorm/ReadVariableOp_22l
4batch_normalization_864/batchnorm/mul/ReadVariableOp4batch_normalization_864/batchnorm/mul/ReadVariableOp2d
0batch_normalization_865/batchnorm/ReadVariableOp0batch_normalization_865/batchnorm/ReadVariableOp2h
2batch_normalization_865/batchnorm/ReadVariableOp_12batch_normalization_865/batchnorm/ReadVariableOp_12h
2batch_normalization_865/batchnorm/ReadVariableOp_22batch_normalization_865/batchnorm/ReadVariableOp_22l
4batch_normalization_865/batchnorm/mul/ReadVariableOp4batch_normalization_865/batchnorm/mul/ReadVariableOp2D
 dense_945/BiasAdd/ReadVariableOp dense_945/BiasAdd/ReadVariableOp2B
dense_945/MatMul/ReadVariableOpdense_945/MatMul/ReadVariableOp2D
 dense_946/BiasAdd/ReadVariableOp dense_946/BiasAdd/ReadVariableOp2B
dense_946/MatMul/ReadVariableOpdense_946/MatMul/ReadVariableOp2D
 dense_947/BiasAdd/ReadVariableOp dense_947/BiasAdd/ReadVariableOp2B
dense_947/MatMul/ReadVariableOpdense_947/MatMul/ReadVariableOp2D
 dense_948/BiasAdd/ReadVariableOp dense_948/BiasAdd/ReadVariableOp2B
dense_948/MatMul/ReadVariableOpdense_948/MatMul/ReadVariableOp2D
 dense_949/BiasAdd/ReadVariableOp dense_949/BiasAdd/ReadVariableOp2B
dense_949/MatMul/ReadVariableOpdense_949/MatMul/ReadVariableOp2D
 dense_950/BiasAdd/ReadVariableOp dense_950/BiasAdd/ReadVariableOp2B
dense_950/MatMul/ReadVariableOpdense_950/MatMul/ReadVariableOp2D
 dense_951/BiasAdd/ReadVariableOp dense_951/BiasAdd/ReadVariableOp2B
dense_951/MatMul/ReadVariableOpdense_951/MatMul/ReadVariableOp2D
 dense_952/BiasAdd/ReadVariableOp dense_952/BiasAdd/ReadVariableOp2B
dense_952/MatMul/ReadVariableOpdense_952/MatMul/ReadVariableOp2D
 dense_953/BiasAdd/ReadVariableOp dense_953/BiasAdd/ReadVariableOp2B
dense_953/MatMul/ReadVariableOpdense_953/MatMul/ReadVariableOp2D
 dense_954/BiasAdd/ReadVariableOp dense_954/BiasAdd/ReadVariableOp2B
dense_954/MatMul/ReadVariableOpdense_954/MatMul/ReadVariableOp2D
 dense_955/BiasAdd/ReadVariableOp dense_955/BiasAdd/ReadVariableOp2B
dense_955/MatMul/ReadVariableOpdense_955/MatMul/ReadVariableOp2D
 dense_956/BiasAdd/ReadVariableOp dense_956/BiasAdd/ReadVariableOp2B
dense_956/MatMul/ReadVariableOpdense_956/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_862_layer_call_and_return_conditional_losses_840472

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_855_layer_call_and_return_conditional_losses_839709

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_840101

inputs/
!batchnorm_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h1
#batchnorm_readvariableop_1_resource:h1
#batchnorm_readvariableop_2_resource:h
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:hz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_859_layer_call_and_return_conditional_losses_836900

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
È	
ö
E__inference_dense_956_layer_call_and_return_conditional_losses_840818

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_836389

inputs5
'assignmovingavg_readvariableop_resource:/7
)assignmovingavg_1_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource://
!batchnorm_readvariableop_resource:/
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:/
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:/*
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
:/*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:/x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:/¬
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
:/*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:/~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:/´
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:/v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_857_layer_call_and_return_conditional_losses_839927

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_840462

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_840646

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_840799

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_863_layer_call_fn_840504

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_836506o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_860_layer_call_fn_840177

inputs
unknown:/
	unknown_0:/
	unknown_1:/
	unknown_2:/
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_836260o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
È	
ö
E__inference_dense_956_layer_call_and_return_conditional_losses_837104

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_954_layer_call_and_return_conditional_losses_840600

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_946_layer_call_and_return_conditional_losses_839728

inputs0
matmul_readvariableop_resource:hh-
biasadd_readvariableop_resource:h
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:hh*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:h*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_836178

inputs/
!batchnorm_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h1
#batchnorm_readvariableop_1_resource:h1
#batchnorm_readvariableop_2_resource:h
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:hz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_840789

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_949_layer_call_fn_840045

inputs
unknown:hh
	unknown_0:h
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_949_layer_call_and_return_conditional_losses_836880o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
È	
ö
E__inference_dense_955_layer_call_and_return_conditional_losses_837072

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
r
"__inference__traced_restore_841867
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_945_kernel:h/
!assignvariableop_4_dense_945_bias:h>
0assignvariableop_5_batch_normalization_855_gamma:h=
/assignvariableop_6_batch_normalization_855_beta:hD
6assignvariableop_7_batch_normalization_855_moving_mean:hH
:assignvariableop_8_batch_normalization_855_moving_variance:h5
#assignvariableop_9_dense_946_kernel:hh0
"assignvariableop_10_dense_946_bias:h?
1assignvariableop_11_batch_normalization_856_gamma:h>
0assignvariableop_12_batch_normalization_856_beta:hE
7assignvariableop_13_batch_normalization_856_moving_mean:hI
;assignvariableop_14_batch_normalization_856_moving_variance:h6
$assignvariableop_15_dense_947_kernel:hh0
"assignvariableop_16_dense_947_bias:h?
1assignvariableop_17_batch_normalization_857_gamma:h>
0assignvariableop_18_batch_normalization_857_beta:hE
7assignvariableop_19_batch_normalization_857_moving_mean:hI
;assignvariableop_20_batch_normalization_857_moving_variance:h6
$assignvariableop_21_dense_948_kernel:hh0
"assignvariableop_22_dense_948_bias:h?
1assignvariableop_23_batch_normalization_858_gamma:h>
0assignvariableop_24_batch_normalization_858_beta:hE
7assignvariableop_25_batch_normalization_858_moving_mean:hI
;assignvariableop_26_batch_normalization_858_moving_variance:h6
$assignvariableop_27_dense_949_kernel:hh0
"assignvariableop_28_dense_949_bias:h?
1assignvariableop_29_batch_normalization_859_gamma:h>
0assignvariableop_30_batch_normalization_859_beta:hE
7assignvariableop_31_batch_normalization_859_moving_mean:hI
;assignvariableop_32_batch_normalization_859_moving_variance:h6
$assignvariableop_33_dense_950_kernel:h/0
"assignvariableop_34_dense_950_bias:/?
1assignvariableop_35_batch_normalization_860_gamma:/>
0assignvariableop_36_batch_normalization_860_beta:/E
7assignvariableop_37_batch_normalization_860_moving_mean:/I
;assignvariableop_38_batch_normalization_860_moving_variance:/6
$assignvariableop_39_dense_951_kernel://0
"assignvariableop_40_dense_951_bias:/?
1assignvariableop_41_batch_normalization_861_gamma:/>
0assignvariableop_42_batch_normalization_861_beta:/E
7assignvariableop_43_batch_normalization_861_moving_mean:/I
;assignvariableop_44_batch_normalization_861_moving_variance:/6
$assignvariableop_45_dense_952_kernel:/0
"assignvariableop_46_dense_952_bias:?
1assignvariableop_47_batch_normalization_862_gamma:>
0assignvariableop_48_batch_normalization_862_beta:E
7assignvariableop_49_batch_normalization_862_moving_mean:I
;assignvariableop_50_batch_normalization_862_moving_variance:6
$assignvariableop_51_dense_953_kernel:0
"assignvariableop_52_dense_953_bias:?
1assignvariableop_53_batch_normalization_863_gamma:>
0assignvariableop_54_batch_normalization_863_beta:E
7assignvariableop_55_batch_normalization_863_moving_mean:I
;assignvariableop_56_batch_normalization_863_moving_variance:6
$assignvariableop_57_dense_954_kernel:0
"assignvariableop_58_dense_954_bias:?
1assignvariableop_59_batch_normalization_864_gamma:>
0assignvariableop_60_batch_normalization_864_beta:E
7assignvariableop_61_batch_normalization_864_moving_mean:I
;assignvariableop_62_batch_normalization_864_moving_variance:6
$assignvariableop_63_dense_955_kernel:0
"assignvariableop_64_dense_955_bias:?
1assignvariableop_65_batch_normalization_865_gamma:>
0assignvariableop_66_batch_normalization_865_beta:E
7assignvariableop_67_batch_normalization_865_moving_mean:I
;assignvariableop_68_batch_normalization_865_moving_variance:6
$assignvariableop_69_dense_956_kernel:0
"assignvariableop_70_dense_956_bias:'
assignvariableop_71_adam_iter:	 )
assignvariableop_72_adam_beta_1: )
assignvariableop_73_adam_beta_2: (
assignvariableop_74_adam_decay: #
assignvariableop_75_total: %
assignvariableop_76_count_1: =
+assignvariableop_77_adam_dense_945_kernel_m:h7
)assignvariableop_78_adam_dense_945_bias_m:hF
8assignvariableop_79_adam_batch_normalization_855_gamma_m:hE
7assignvariableop_80_adam_batch_normalization_855_beta_m:h=
+assignvariableop_81_adam_dense_946_kernel_m:hh7
)assignvariableop_82_adam_dense_946_bias_m:hF
8assignvariableop_83_adam_batch_normalization_856_gamma_m:hE
7assignvariableop_84_adam_batch_normalization_856_beta_m:h=
+assignvariableop_85_adam_dense_947_kernel_m:hh7
)assignvariableop_86_adam_dense_947_bias_m:hF
8assignvariableop_87_adam_batch_normalization_857_gamma_m:hE
7assignvariableop_88_adam_batch_normalization_857_beta_m:h=
+assignvariableop_89_adam_dense_948_kernel_m:hh7
)assignvariableop_90_adam_dense_948_bias_m:hF
8assignvariableop_91_adam_batch_normalization_858_gamma_m:hE
7assignvariableop_92_adam_batch_normalization_858_beta_m:h=
+assignvariableop_93_adam_dense_949_kernel_m:hh7
)assignvariableop_94_adam_dense_949_bias_m:hF
8assignvariableop_95_adam_batch_normalization_859_gamma_m:hE
7assignvariableop_96_adam_batch_normalization_859_beta_m:h=
+assignvariableop_97_adam_dense_950_kernel_m:h/7
)assignvariableop_98_adam_dense_950_bias_m:/F
8assignvariableop_99_adam_batch_normalization_860_gamma_m:/F
8assignvariableop_100_adam_batch_normalization_860_beta_m:/>
,assignvariableop_101_adam_dense_951_kernel_m://8
*assignvariableop_102_adam_dense_951_bias_m:/G
9assignvariableop_103_adam_batch_normalization_861_gamma_m:/F
8assignvariableop_104_adam_batch_normalization_861_beta_m:/>
,assignvariableop_105_adam_dense_952_kernel_m:/8
*assignvariableop_106_adam_dense_952_bias_m:G
9assignvariableop_107_adam_batch_normalization_862_gamma_m:F
8assignvariableop_108_adam_batch_normalization_862_beta_m:>
,assignvariableop_109_adam_dense_953_kernel_m:8
*assignvariableop_110_adam_dense_953_bias_m:G
9assignvariableop_111_adam_batch_normalization_863_gamma_m:F
8assignvariableop_112_adam_batch_normalization_863_beta_m:>
,assignvariableop_113_adam_dense_954_kernel_m:8
*assignvariableop_114_adam_dense_954_bias_m:G
9assignvariableop_115_adam_batch_normalization_864_gamma_m:F
8assignvariableop_116_adam_batch_normalization_864_beta_m:>
,assignvariableop_117_adam_dense_955_kernel_m:8
*assignvariableop_118_adam_dense_955_bias_m:G
9assignvariableop_119_adam_batch_normalization_865_gamma_m:F
8assignvariableop_120_adam_batch_normalization_865_beta_m:>
,assignvariableop_121_adam_dense_956_kernel_m:8
*assignvariableop_122_adam_dense_956_bias_m:>
,assignvariableop_123_adam_dense_945_kernel_v:h8
*assignvariableop_124_adam_dense_945_bias_v:hG
9assignvariableop_125_adam_batch_normalization_855_gamma_v:hF
8assignvariableop_126_adam_batch_normalization_855_beta_v:h>
,assignvariableop_127_adam_dense_946_kernel_v:hh8
*assignvariableop_128_adam_dense_946_bias_v:hG
9assignvariableop_129_adam_batch_normalization_856_gamma_v:hF
8assignvariableop_130_adam_batch_normalization_856_beta_v:h>
,assignvariableop_131_adam_dense_947_kernel_v:hh8
*assignvariableop_132_adam_dense_947_bias_v:hG
9assignvariableop_133_adam_batch_normalization_857_gamma_v:hF
8assignvariableop_134_adam_batch_normalization_857_beta_v:h>
,assignvariableop_135_adam_dense_948_kernel_v:hh8
*assignvariableop_136_adam_dense_948_bias_v:hG
9assignvariableop_137_adam_batch_normalization_858_gamma_v:hF
8assignvariableop_138_adam_batch_normalization_858_beta_v:h>
,assignvariableop_139_adam_dense_949_kernel_v:hh8
*assignvariableop_140_adam_dense_949_bias_v:hG
9assignvariableop_141_adam_batch_normalization_859_gamma_v:hF
8assignvariableop_142_adam_batch_normalization_859_beta_v:h>
,assignvariableop_143_adam_dense_950_kernel_v:h/8
*assignvariableop_144_adam_dense_950_bias_v:/G
9assignvariableop_145_adam_batch_normalization_860_gamma_v:/F
8assignvariableop_146_adam_batch_normalization_860_beta_v:/>
,assignvariableop_147_adam_dense_951_kernel_v://8
*assignvariableop_148_adam_dense_951_bias_v:/G
9assignvariableop_149_adam_batch_normalization_861_gamma_v:/F
8assignvariableop_150_adam_batch_normalization_861_beta_v:/>
,assignvariableop_151_adam_dense_952_kernel_v:/8
*assignvariableop_152_adam_dense_952_bias_v:G
9assignvariableop_153_adam_batch_normalization_862_gamma_v:F
8assignvariableop_154_adam_batch_normalization_862_beta_v:>
,assignvariableop_155_adam_dense_953_kernel_v:8
*assignvariableop_156_adam_dense_953_bias_v:G
9assignvariableop_157_adam_batch_normalization_863_gamma_v:F
8assignvariableop_158_adam_batch_normalization_863_beta_v:>
,assignvariableop_159_adam_dense_954_kernel_v:8
*assignvariableop_160_adam_dense_954_bias_v:G
9assignvariableop_161_adam_batch_normalization_864_gamma_v:F
8assignvariableop_162_adam_batch_normalization_864_beta_v:>
,assignvariableop_163_adam_dense_955_kernel_v:8
*assignvariableop_164_adam_dense_955_bias_v:G
9assignvariableop_165_adam_batch_normalization_865_gamma_v:F
8assignvariableop_166_adam_batch_normalization_865_beta_v:>
,assignvariableop_167_adam_dense_956_kernel_v:8
*assignvariableop_168_adam_dense_956_bias_v:
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_945_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_945_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batch_normalization_855_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_855_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_855_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_8AssignVariableOp:assignvariableop_8_batch_normalization_855_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_946_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_946_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batch_normalization_856_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_856_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_856_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_14AssignVariableOp;assignvariableop_14_batch_normalization_856_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_947_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_947_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batch_normalization_857_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_857_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_857_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_20AssignVariableOp;assignvariableop_20_batch_normalization_857_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_948_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_948_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_23AssignVariableOp1assignvariableop_23_batch_normalization_858_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_858_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_25AssignVariableOp7assignvariableop_25_batch_normalization_858_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_26AssignVariableOp;assignvariableop_26_batch_normalization_858_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_949_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_949_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_29AssignVariableOp1assignvariableop_29_batch_normalization_859_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_859_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_31AssignVariableOp7assignvariableop_31_batch_normalization_859_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_32AssignVariableOp;assignvariableop_32_batch_normalization_859_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_950_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_950_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_35AssignVariableOp1assignvariableop_35_batch_normalization_860_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_860_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_37AssignVariableOp7assignvariableop_37_batch_normalization_860_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_38AssignVariableOp;assignvariableop_38_batch_normalization_860_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp$assignvariableop_39_dense_951_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_951_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_41AssignVariableOp1assignvariableop_41_batch_normalization_861_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_batch_normalization_861_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_43AssignVariableOp7assignvariableop_43_batch_normalization_861_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_44AssignVariableOp;assignvariableop_44_batch_normalization_861_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp$assignvariableop_45_dense_952_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_952_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_47AssignVariableOp1assignvariableop_47_batch_normalization_862_gammaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_48AssignVariableOp0assignvariableop_48_batch_normalization_862_betaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_49AssignVariableOp7assignvariableop_49_batch_normalization_862_moving_meanIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_50AssignVariableOp;assignvariableop_50_batch_normalization_862_moving_varianceIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp$assignvariableop_51_dense_953_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp"assignvariableop_52_dense_953_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_53AssignVariableOp1assignvariableop_53_batch_normalization_863_gammaIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_54AssignVariableOp0assignvariableop_54_batch_normalization_863_betaIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_55AssignVariableOp7assignvariableop_55_batch_normalization_863_moving_meanIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_56AssignVariableOp;assignvariableop_56_batch_normalization_863_moving_varianceIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp$assignvariableop_57_dense_954_kernelIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp"assignvariableop_58_dense_954_biasIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_59AssignVariableOp1assignvariableop_59_batch_normalization_864_gammaIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_60AssignVariableOp0assignvariableop_60_batch_normalization_864_betaIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_61AssignVariableOp7assignvariableop_61_batch_normalization_864_moving_meanIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_62AssignVariableOp;assignvariableop_62_batch_normalization_864_moving_varianceIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp$assignvariableop_63_dense_955_kernelIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp"assignvariableop_64_dense_955_biasIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_65AssignVariableOp1assignvariableop_65_batch_normalization_865_gammaIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_66AssignVariableOp0assignvariableop_66_batch_normalization_865_betaIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_67AssignVariableOp7assignvariableop_67_batch_normalization_865_moving_meanIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_68AssignVariableOp;assignvariableop_68_batch_normalization_865_moving_varianceIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp$assignvariableop_69_dense_956_kernelIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp"assignvariableop_70_dense_956_biasIdentity_70:output:0"/device:CPU:0*
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
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_dense_945_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_dense_945_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_79AssignVariableOp8assignvariableop_79_adam_batch_normalization_855_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_80AssignVariableOp7assignvariableop_80_adam_batch_normalization_855_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_dense_946_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_dense_946_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_83AssignVariableOp8assignvariableop_83_adam_batch_normalization_856_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_84AssignVariableOp7assignvariableop_84_adam_batch_normalization_856_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_dense_947_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_dense_947_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_87AssignVariableOp8assignvariableop_87_adam_batch_normalization_857_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_88AssignVariableOp7assignvariableop_88_adam_batch_normalization_857_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_dense_948_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_dense_948_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_91AssignVariableOp8assignvariableop_91_adam_batch_normalization_858_gamma_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_92AssignVariableOp7assignvariableop_92_adam_batch_normalization_858_beta_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_dense_949_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_dense_949_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_95AssignVariableOp8assignvariableop_95_adam_batch_normalization_859_gamma_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_96AssignVariableOp7assignvariableop_96_adam_batch_normalization_859_beta_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_dense_950_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_dense_950_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_99AssignVariableOp8assignvariableop_99_adam_batch_normalization_860_gamma_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_100AssignVariableOp8assignvariableop_100_adam_batch_normalization_860_beta_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_dense_951_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_dense_951_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_103AssignVariableOp9assignvariableop_103_adam_batch_normalization_861_gamma_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_104AssignVariableOp8assignvariableop_104_adam_batch_normalization_861_beta_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_dense_952_kernel_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_dense_952_bias_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_107AssignVariableOp9assignvariableop_107_adam_batch_normalization_862_gamma_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_108AssignVariableOp8assignvariableop_108_adam_batch_normalization_862_beta_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_dense_953_kernel_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_dense_953_bias_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_111AssignVariableOp9assignvariableop_111_adam_batch_normalization_863_gamma_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_112AssignVariableOp8assignvariableop_112_adam_batch_normalization_863_beta_mIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_dense_954_kernel_mIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_dense_954_bias_mIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_115AssignVariableOp9assignvariableop_115_adam_batch_normalization_864_gamma_mIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_116AssignVariableOp8assignvariableop_116_adam_batch_normalization_864_beta_mIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_dense_955_kernel_mIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_dense_955_bias_mIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_119AssignVariableOp9assignvariableop_119_adam_batch_normalization_865_gamma_mIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_120AssignVariableOp8assignvariableop_120_adam_batch_normalization_865_beta_mIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_121AssignVariableOp,assignvariableop_121_adam_dense_956_kernel_mIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_122AssignVariableOp*assignvariableop_122_adam_dense_956_bias_mIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_123AssignVariableOp,assignvariableop_123_adam_dense_945_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_124AssignVariableOp*assignvariableop_124_adam_dense_945_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_125AssignVariableOp9assignvariableop_125_adam_batch_normalization_855_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_126AssignVariableOp8assignvariableop_126_adam_batch_normalization_855_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_127AssignVariableOp,assignvariableop_127_adam_dense_946_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_128AssignVariableOp*assignvariableop_128_adam_dense_946_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_129AssignVariableOp9assignvariableop_129_adam_batch_normalization_856_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_130AssignVariableOp8assignvariableop_130_adam_batch_normalization_856_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_131AssignVariableOp,assignvariableop_131_adam_dense_947_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_132AssignVariableOp*assignvariableop_132_adam_dense_947_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_133AssignVariableOp9assignvariableop_133_adam_batch_normalization_857_gamma_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_134AssignVariableOp8assignvariableop_134_adam_batch_normalization_857_beta_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_135AssignVariableOp,assignvariableop_135_adam_dense_948_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_136AssignVariableOp*assignvariableop_136_adam_dense_948_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_137AssignVariableOp9assignvariableop_137_adam_batch_normalization_858_gamma_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_138AssignVariableOp8assignvariableop_138_adam_batch_normalization_858_beta_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_139AssignVariableOp,assignvariableop_139_adam_dense_949_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_140AssignVariableOp*assignvariableop_140_adam_dense_949_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_141AssignVariableOp9assignvariableop_141_adam_batch_normalization_859_gamma_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_142AssignVariableOp8assignvariableop_142_adam_batch_normalization_859_beta_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_143AssignVariableOp,assignvariableop_143_adam_dense_950_kernel_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_144AssignVariableOp*assignvariableop_144_adam_dense_950_bias_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_145AssignVariableOp9assignvariableop_145_adam_batch_normalization_860_gamma_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_146AssignVariableOp8assignvariableop_146_adam_batch_normalization_860_beta_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_147AssignVariableOp,assignvariableop_147_adam_dense_951_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_148AssignVariableOp*assignvariableop_148_adam_dense_951_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_149AssignVariableOp9assignvariableop_149_adam_batch_normalization_861_gamma_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_150AssignVariableOp8assignvariableop_150_adam_batch_normalization_861_beta_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_151AssignVariableOp,assignvariableop_151_adam_dense_952_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_152AssignVariableOp*assignvariableop_152_adam_dense_952_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_153AssignVariableOp9assignvariableop_153_adam_batch_normalization_862_gamma_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_154AssignVariableOp8assignvariableop_154_adam_batch_normalization_862_beta_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_155AssignVariableOp,assignvariableop_155_adam_dense_953_kernel_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_156AssignVariableOp*assignvariableop_156_adam_dense_953_bias_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_157AssignVariableOp9assignvariableop_157_adam_batch_normalization_863_gamma_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_158AssignVariableOp8assignvariableop_158_adam_batch_normalization_863_beta_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_159AssignVariableOp,assignvariableop_159_adam_dense_954_kernel_vIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_160AssignVariableOp*assignvariableop_160_adam_dense_954_bias_vIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_161AssignVariableOp9assignvariableop_161_adam_batch_normalization_864_gamma_vIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_162AssignVariableOp8assignvariableop_162_adam_batch_normalization_864_beta_vIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_163AssignVariableOp,assignvariableop_163_adam_dense_955_kernel_vIdentity_163:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_164AssignVariableOp*assignvariableop_164_adam_dense_955_bias_vIdentity_164:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_165IdentityRestoreV2:tensors:165"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_165AssignVariableOp9assignvariableop_165_adam_batch_normalization_865_gamma_vIdentity_165:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_166IdentityRestoreV2:tensors:166"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_166AssignVariableOp8assignvariableop_166_adam_batch_normalization_865_beta_vIdentity_166:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_167IdentityRestoreV2:tensors:167"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_167AssignVariableOp,assignvariableop_167_adam_dense_956_kernel_vIdentity_167:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_168IdentityRestoreV2:tensors:168"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_168AssignVariableOp*assignvariableop_168_adam_dense_956_bias_vIdentity_168:output:0"/device:CPU:0*
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
¬
Ó
8__inference_batch_normalization_859_layer_call_fn_840068

inputs
unknown:h
	unknown_0:h
	unknown_1:h
	unknown_2:h
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_836178o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_858_layer_call_and_return_conditional_losses_840036

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_855_layer_call_and_return_conditional_losses_836772

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿh:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_836014

inputs/
!batchnorm_readvariableop_resource:h3
%batchnorm_mul_readvariableop_resource:h1
#batchnorm_readvariableop_1_resource:h1
#batchnorm_readvariableop_2_resource:h
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:h*
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
:hP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:h~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:h*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:hc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:h*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:hz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:h*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:hr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_840319

inputs/
!batchnorm_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource:/1
#batchnorm_readvariableop_1_resource:/1
#batchnorm_readvariableop_2_resource:/
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:/z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
È	
ö
E__inference_dense_945_layer_call_and_return_conditional_losses_836752

inputs0
matmul_readvariableop_resource:h-
biasadd_readvariableop_resource:h
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:h*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:h*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿhw
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
¬
Ó
8__inference_batch_normalization_856_layer_call_fn_839741

inputs
unknown:h
	unknown_0:h
	unknown_1:h
	unknown_2:h
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_835932o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿh: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
 
_user_specified_nameinputs
È	
ö
E__inference_dense_951_layer_call_and_return_conditional_losses_836944

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
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
normalization_90_input?
(serving_default_normalization_90_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_9560
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
.__inference_sequential_90_layer_call_fn_837254
.__inference_sequential_90_layer_call_fn_838567
.__inference_sequential_90_layer_call_fn_838712
.__inference_sequential_90_layer_call_fn_838056À
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
I__inference_sequential_90_layer_call_and_return_conditional_losses_838982
I__inference_sequential_90_layer_call_and_return_conditional_losses_839406
I__inference_sequential_90_layer_call_and_return_conditional_losses_838237
I__inference_sequential_90_layer_call_and_return_conditional_losses_838418À
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
!__inference__wrapped_model_835826normalization_90_input"
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
__inference_adapt_step_839600
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
": h2dense_945/kernel
:h2dense_945/bias
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
*__inference_dense_945_layer_call_fn_839609¢
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
E__inference_dense_945_layer_call_and_return_conditional_losses_839619¢
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
+:)h2batch_normalization_855/gamma
*:(h2batch_normalization_855/beta
3:1h (2#batch_normalization_855/moving_mean
7:5h (2'batch_normalization_855/moving_variance
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
8__inference_batch_normalization_855_layer_call_fn_839632
8__inference_batch_normalization_855_layer_call_fn_839645´
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
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_839665
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_839699´
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
0__inference_leaky_re_lu_855_layer_call_fn_839704¢
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
K__inference_leaky_re_lu_855_layer_call_and_return_conditional_losses_839709¢
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
": hh2dense_946/kernel
:h2dense_946/bias
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
*__inference_dense_946_layer_call_fn_839718¢
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
E__inference_dense_946_layer_call_and_return_conditional_losses_839728¢
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
+:)h2batch_normalization_856/gamma
*:(h2batch_normalization_856/beta
3:1h (2#batch_normalization_856/moving_mean
7:5h (2'batch_normalization_856/moving_variance
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
8__inference_batch_normalization_856_layer_call_fn_839741
8__inference_batch_normalization_856_layer_call_fn_839754´
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
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_839774
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_839808´
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
0__inference_leaky_re_lu_856_layer_call_fn_839813¢
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
K__inference_leaky_re_lu_856_layer_call_and_return_conditional_losses_839818¢
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
": hh2dense_947/kernel
:h2dense_947/bias
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
*__inference_dense_947_layer_call_fn_839827¢
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
E__inference_dense_947_layer_call_and_return_conditional_losses_839837¢
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
+:)h2batch_normalization_857/gamma
*:(h2batch_normalization_857/beta
3:1h (2#batch_normalization_857/moving_mean
7:5h (2'batch_normalization_857/moving_variance
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
8__inference_batch_normalization_857_layer_call_fn_839850
8__inference_batch_normalization_857_layer_call_fn_839863´
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
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_839883
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_839917´
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
0__inference_leaky_re_lu_857_layer_call_fn_839922¢
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
K__inference_leaky_re_lu_857_layer_call_and_return_conditional_losses_839927¢
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
": hh2dense_948/kernel
:h2dense_948/bias
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
*__inference_dense_948_layer_call_fn_839936¢
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
E__inference_dense_948_layer_call_and_return_conditional_losses_839946¢
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
+:)h2batch_normalization_858/gamma
*:(h2batch_normalization_858/beta
3:1h (2#batch_normalization_858/moving_mean
7:5h (2'batch_normalization_858/moving_variance
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
8__inference_batch_normalization_858_layer_call_fn_839959
8__inference_batch_normalization_858_layer_call_fn_839972´
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
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_839992
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_840026´
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
0__inference_leaky_re_lu_858_layer_call_fn_840031¢
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
K__inference_leaky_re_lu_858_layer_call_and_return_conditional_losses_840036¢
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
": hh2dense_949/kernel
:h2dense_949/bias
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
*__inference_dense_949_layer_call_fn_840045¢
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
E__inference_dense_949_layer_call_and_return_conditional_losses_840055¢
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
+:)h2batch_normalization_859/gamma
*:(h2batch_normalization_859/beta
3:1h (2#batch_normalization_859/moving_mean
7:5h (2'batch_normalization_859/moving_variance
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
8__inference_batch_normalization_859_layer_call_fn_840068
8__inference_batch_normalization_859_layer_call_fn_840081´
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
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_840101
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_840135´
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
0__inference_leaky_re_lu_859_layer_call_fn_840140¢
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
K__inference_leaky_re_lu_859_layer_call_and_return_conditional_losses_840145¢
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
": h/2dense_950/kernel
:/2dense_950/bias
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
*__inference_dense_950_layer_call_fn_840154¢
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
E__inference_dense_950_layer_call_and_return_conditional_losses_840164¢
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
+:)/2batch_normalization_860/gamma
*:(/2batch_normalization_860/beta
3:1/ (2#batch_normalization_860/moving_mean
7:5/ (2'batch_normalization_860/moving_variance
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
8__inference_batch_normalization_860_layer_call_fn_840177
8__inference_batch_normalization_860_layer_call_fn_840190´
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
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_840210
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_840244´
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
0__inference_leaky_re_lu_860_layer_call_fn_840249¢
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
K__inference_leaky_re_lu_860_layer_call_and_return_conditional_losses_840254¢
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
": //2dense_951/kernel
:/2dense_951/bias
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
*__inference_dense_951_layer_call_fn_840263¢
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
E__inference_dense_951_layer_call_and_return_conditional_losses_840273¢
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
+:)/2batch_normalization_861/gamma
*:(/2batch_normalization_861/beta
3:1/ (2#batch_normalization_861/moving_mean
7:5/ (2'batch_normalization_861/moving_variance
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
8__inference_batch_normalization_861_layer_call_fn_840286
8__inference_batch_normalization_861_layer_call_fn_840299´
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
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_840319
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_840353´
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
0__inference_leaky_re_lu_861_layer_call_fn_840358¢
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
K__inference_leaky_re_lu_861_layer_call_and_return_conditional_losses_840363¢
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
": /2dense_952/kernel
:2dense_952/bias
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
*__inference_dense_952_layer_call_fn_840372¢
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
E__inference_dense_952_layer_call_and_return_conditional_losses_840382¢
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
+:)2batch_normalization_862/gamma
*:(2batch_normalization_862/beta
3:1 (2#batch_normalization_862/moving_mean
7:5 (2'batch_normalization_862/moving_variance
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
8__inference_batch_normalization_862_layer_call_fn_840395
8__inference_batch_normalization_862_layer_call_fn_840408´
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
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_840428
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_840462´
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
0__inference_leaky_re_lu_862_layer_call_fn_840467¢
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
K__inference_leaky_re_lu_862_layer_call_and_return_conditional_losses_840472¢
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
": 2dense_953/kernel
:2dense_953/bias
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
*__inference_dense_953_layer_call_fn_840481¢
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
E__inference_dense_953_layer_call_and_return_conditional_losses_840491¢
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
+:)2batch_normalization_863/gamma
*:(2batch_normalization_863/beta
3:1 (2#batch_normalization_863/moving_mean
7:5 (2'batch_normalization_863/moving_variance
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
8__inference_batch_normalization_863_layer_call_fn_840504
8__inference_batch_normalization_863_layer_call_fn_840517´
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
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_840537
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_840571´
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
0__inference_leaky_re_lu_863_layer_call_fn_840576¢
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
K__inference_leaky_re_lu_863_layer_call_and_return_conditional_losses_840581¢
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
": 2dense_954/kernel
:2dense_954/bias
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
*__inference_dense_954_layer_call_fn_840590¢
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
E__inference_dense_954_layer_call_and_return_conditional_losses_840600¢
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
+:)2batch_normalization_864/gamma
*:(2batch_normalization_864/beta
3:1 (2#batch_normalization_864/moving_mean
7:5 (2'batch_normalization_864/moving_variance
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
8__inference_batch_normalization_864_layer_call_fn_840613
8__inference_batch_normalization_864_layer_call_fn_840626´
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
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_840646
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_840680´
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
0__inference_leaky_re_lu_864_layer_call_fn_840685¢
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
K__inference_leaky_re_lu_864_layer_call_and_return_conditional_losses_840690¢
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
": 2dense_955/kernel
:2dense_955/bias
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
*__inference_dense_955_layer_call_fn_840699¢
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
E__inference_dense_955_layer_call_and_return_conditional_losses_840709¢
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
+:)2batch_normalization_865/gamma
*:(2batch_normalization_865/beta
3:1 (2#batch_normalization_865/moving_mean
7:5 (2'batch_normalization_865/moving_variance
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
8__inference_batch_normalization_865_layer_call_fn_840722
8__inference_batch_normalization_865_layer_call_fn_840735´
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
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_840755
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_840789´
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
0__inference_leaky_re_lu_865_layer_call_fn_840794¢
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
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_840799¢
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
": 2dense_956/kernel
:2dense_956/bias
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
*__inference_dense_956_layer_call_fn_840808¢
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
E__inference_dense_956_layer_call_and_return_conditional_losses_840818¢
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
$__inference_signature_wrapper_839553normalization_90_input"
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
':%h2Adam/dense_945/kernel/m
!:h2Adam/dense_945/bias/m
0:.h2$Adam/batch_normalization_855/gamma/m
/:-h2#Adam/batch_normalization_855/beta/m
':%hh2Adam/dense_946/kernel/m
!:h2Adam/dense_946/bias/m
0:.h2$Adam/batch_normalization_856/gamma/m
/:-h2#Adam/batch_normalization_856/beta/m
':%hh2Adam/dense_947/kernel/m
!:h2Adam/dense_947/bias/m
0:.h2$Adam/batch_normalization_857/gamma/m
/:-h2#Adam/batch_normalization_857/beta/m
':%hh2Adam/dense_948/kernel/m
!:h2Adam/dense_948/bias/m
0:.h2$Adam/batch_normalization_858/gamma/m
/:-h2#Adam/batch_normalization_858/beta/m
':%hh2Adam/dense_949/kernel/m
!:h2Adam/dense_949/bias/m
0:.h2$Adam/batch_normalization_859/gamma/m
/:-h2#Adam/batch_normalization_859/beta/m
':%h/2Adam/dense_950/kernel/m
!:/2Adam/dense_950/bias/m
0:./2$Adam/batch_normalization_860/gamma/m
/:-/2#Adam/batch_normalization_860/beta/m
':%//2Adam/dense_951/kernel/m
!:/2Adam/dense_951/bias/m
0:./2$Adam/batch_normalization_861/gamma/m
/:-/2#Adam/batch_normalization_861/beta/m
':%/2Adam/dense_952/kernel/m
!:2Adam/dense_952/bias/m
0:.2$Adam/batch_normalization_862/gamma/m
/:-2#Adam/batch_normalization_862/beta/m
':%2Adam/dense_953/kernel/m
!:2Adam/dense_953/bias/m
0:.2$Adam/batch_normalization_863/gamma/m
/:-2#Adam/batch_normalization_863/beta/m
':%2Adam/dense_954/kernel/m
!:2Adam/dense_954/bias/m
0:.2$Adam/batch_normalization_864/gamma/m
/:-2#Adam/batch_normalization_864/beta/m
':%2Adam/dense_955/kernel/m
!:2Adam/dense_955/bias/m
0:.2$Adam/batch_normalization_865/gamma/m
/:-2#Adam/batch_normalization_865/beta/m
':%2Adam/dense_956/kernel/m
!:2Adam/dense_956/bias/m
':%h2Adam/dense_945/kernel/v
!:h2Adam/dense_945/bias/v
0:.h2$Adam/batch_normalization_855/gamma/v
/:-h2#Adam/batch_normalization_855/beta/v
':%hh2Adam/dense_946/kernel/v
!:h2Adam/dense_946/bias/v
0:.h2$Adam/batch_normalization_856/gamma/v
/:-h2#Adam/batch_normalization_856/beta/v
':%hh2Adam/dense_947/kernel/v
!:h2Adam/dense_947/bias/v
0:.h2$Adam/batch_normalization_857/gamma/v
/:-h2#Adam/batch_normalization_857/beta/v
':%hh2Adam/dense_948/kernel/v
!:h2Adam/dense_948/bias/v
0:.h2$Adam/batch_normalization_858/gamma/v
/:-h2#Adam/batch_normalization_858/beta/v
':%hh2Adam/dense_949/kernel/v
!:h2Adam/dense_949/bias/v
0:.h2$Adam/batch_normalization_859/gamma/v
/:-h2#Adam/batch_normalization_859/beta/v
':%h/2Adam/dense_950/kernel/v
!:/2Adam/dense_950/bias/v
0:./2$Adam/batch_normalization_860/gamma/v
/:-/2#Adam/batch_normalization_860/beta/v
':%//2Adam/dense_951/kernel/v
!:/2Adam/dense_951/bias/v
0:./2$Adam/batch_normalization_861/gamma/v
/:-/2#Adam/batch_normalization_861/beta/v
':%/2Adam/dense_952/kernel/v
!:2Adam/dense_952/bias/v
0:.2$Adam/batch_normalization_862/gamma/v
/:-2#Adam/batch_normalization_862/beta/v
':%2Adam/dense_953/kernel/v
!:2Adam/dense_953/bias/v
0:.2$Adam/batch_normalization_863/gamma/v
/:-2#Adam/batch_normalization_863/beta/v
':%2Adam/dense_954/kernel/v
!:2Adam/dense_954/bias/v
0:.2$Adam/batch_normalization_864/gamma/v
/:-2#Adam/batch_normalization_864/beta/v
':%2Adam/dense_955/kernel/v
!:2Adam/dense_955/bias/v
0:.2$Adam/batch_normalization_865/gamma/v
/:-2#Adam/batch_normalization_865/beta/v
':%2Adam/dense_956/kernel/v
!:2Adam/dense_956/bias/v
	J
Const
J	
Const_1
!__inference__wrapped_model_835826ôzæç67B?A@OP[XZYhitqsr¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæñîðïþÿ£ ¢¡°±¼¹»ºÉÊ?¢<
5¢2
0-
normalization_90_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_956# 
	dense_956ÿÿÿÿÿÿÿÿÿo
__inference_adapt_step_839600N312C¢@
9¢6
41¢
ÿÿÿÿÿÿÿÿÿIteratorSpec 
ª "
 ¹
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_839665bB?A@3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 ¹
S__inference_batch_normalization_855_layer_call_and_return_conditional_losses_839699bAB?@3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 
8__inference_batch_normalization_855_layer_call_fn_839632UB?A@3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p 
ª "ÿÿÿÿÿÿÿÿÿh
8__inference_batch_normalization_855_layer_call_fn_839645UAB?@3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p
ª "ÿÿÿÿÿÿÿÿÿh¹
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_839774b[XZY3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 ¹
S__inference_batch_normalization_856_layer_call_and_return_conditional_losses_839808bZ[XY3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 
8__inference_batch_normalization_856_layer_call_fn_839741U[XZY3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p 
ª "ÿÿÿÿÿÿÿÿÿh
8__inference_batch_normalization_856_layer_call_fn_839754UZ[XY3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p
ª "ÿÿÿÿÿÿÿÿÿh¹
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_839883btqsr3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 ¹
S__inference_batch_normalization_857_layer_call_and_return_conditional_losses_839917bstqr3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 
8__inference_batch_normalization_857_layer_call_fn_839850Utqsr3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p 
ª "ÿÿÿÿÿÿÿÿÿh
8__inference_batch_normalization_857_layer_call_fn_839863Ustqr3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p
ª "ÿÿÿÿÿÿÿÿÿh½
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_839992f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 ½
S__inference_batch_normalization_858_layer_call_and_return_conditional_losses_840026f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 
8__inference_batch_normalization_858_layer_call_fn_839959Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p 
ª "ÿÿÿÿÿÿÿÿÿh
8__inference_batch_normalization_858_layer_call_fn_839972Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p
ª "ÿÿÿÿÿÿÿÿÿh½
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_840101f¦£¥¤3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 ½
S__inference_batch_normalization_859_layer_call_and_return_conditional_losses_840135f¥¦£¤3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 
8__inference_batch_normalization_859_layer_call_fn_840068Y¦£¥¤3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p 
ª "ÿÿÿÿÿÿÿÿÿh
8__inference_batch_normalization_859_layer_call_fn_840081Y¥¦£¤3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿh
p
ª "ÿÿÿÿÿÿÿÿÿh½
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_840210f¿¼¾½3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ/
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 ½
S__inference_batch_normalization_860_layer_call_and_return_conditional_losses_840244f¾¿¼½3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ/
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 
8__inference_batch_normalization_860_layer_call_fn_840177Y¿¼¾½3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ/
p 
ª "ÿÿÿÿÿÿÿÿÿ/
8__inference_batch_normalization_860_layer_call_fn_840190Y¾¿¼½3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ/
p
ª "ÿÿÿÿÿÿÿÿÿ/½
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_840319fØÕ×Ö3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ/
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 ½
S__inference_batch_normalization_861_layer_call_and_return_conditional_losses_840353f×ØÕÖ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ/
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 
8__inference_batch_normalization_861_layer_call_fn_840286YØÕ×Ö3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ/
p 
ª "ÿÿÿÿÿÿÿÿÿ/
8__inference_batch_normalization_861_layer_call_fn_840299Y×ØÕÖ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ/
p
ª "ÿÿÿÿÿÿÿÿÿ/½
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_840428fñîðï3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
S__inference_batch_normalization_862_layer_call_and_return_conditional_losses_840462fðñîï3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_862_layer_call_fn_840395Yñîðï3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_862_layer_call_fn_840408Yðñîï3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ½
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_840537f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
S__inference_batch_normalization_863_layer_call_and_return_conditional_losses_840571f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_863_layer_call_fn_840504Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_863_layer_call_fn_840517Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ½
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_840646f£ ¢¡3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
S__inference_batch_normalization_864_layer_call_and_return_conditional_losses_840680f¢£ ¡3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_864_layer_call_fn_840613Y£ ¢¡3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_864_layer_call_fn_840626Y¢£ ¡3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ½
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_840755f¼¹»º3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
S__inference_batch_normalization_865_layer_call_and_return_conditional_losses_840789f»¼¹º3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_865_layer_call_fn_840722Y¼¹»º3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_865_layer_call_fn_840735Y»¼¹º3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_945_layer_call_and_return_conditional_losses_839619\67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 }
*__inference_dense_945_layer_call_fn_839609O67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿh¥
E__inference_dense_946_layer_call_and_return_conditional_losses_839728\OP/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 }
*__inference_dense_946_layer_call_fn_839718OOP/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "ÿÿÿÿÿÿÿÿÿh¥
E__inference_dense_947_layer_call_and_return_conditional_losses_839837\hi/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 }
*__inference_dense_947_layer_call_fn_839827Ohi/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "ÿÿÿÿÿÿÿÿÿh§
E__inference_dense_948_layer_call_and_return_conditional_losses_839946^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 
*__inference_dense_948_layer_call_fn_839936Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "ÿÿÿÿÿÿÿÿÿh§
E__inference_dense_949_layer_call_and_return_conditional_losses_840055^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 
*__inference_dense_949_layer_call_fn_840045Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "ÿÿÿÿÿÿÿÿÿh§
E__inference_dense_950_layer_call_and_return_conditional_losses_840164^³´/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 
*__inference_dense_950_layer_call_fn_840154Q³´/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "ÿÿÿÿÿÿÿÿÿ/§
E__inference_dense_951_layer_call_and_return_conditional_losses_840273^ÌÍ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 
*__inference_dense_951_layer_call_fn_840263QÌÍ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ/§
E__inference_dense_952_layer_call_and_return_conditional_losses_840382^åæ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_952_layer_call_fn_840372Qåæ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_953_layer_call_and_return_conditional_losses_840491^þÿ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_953_layer_call_fn_840481Qþÿ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_954_layer_call_and_return_conditional_losses_840600^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_954_layer_call_fn_840590Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_955_layer_call_and_return_conditional_losses_840709^°±/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_955_layer_call_fn_840699Q°±/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_956_layer_call_and_return_conditional_losses_840818^ÉÊ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_956_layer_call_fn_840808QÉÊ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_855_layer_call_and_return_conditional_losses_839709X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 
0__inference_leaky_re_lu_855_layer_call_fn_839704K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "ÿÿÿÿÿÿÿÿÿh§
K__inference_leaky_re_lu_856_layer_call_and_return_conditional_losses_839818X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 
0__inference_leaky_re_lu_856_layer_call_fn_839813K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "ÿÿÿÿÿÿÿÿÿh§
K__inference_leaky_re_lu_857_layer_call_and_return_conditional_losses_839927X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 
0__inference_leaky_re_lu_857_layer_call_fn_839922K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "ÿÿÿÿÿÿÿÿÿh§
K__inference_leaky_re_lu_858_layer_call_and_return_conditional_losses_840036X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 
0__inference_leaky_re_lu_858_layer_call_fn_840031K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "ÿÿÿÿÿÿÿÿÿh§
K__inference_leaky_re_lu_859_layer_call_and_return_conditional_losses_840145X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "%¢"

0ÿÿÿÿÿÿÿÿÿh
 
0__inference_leaky_re_lu_859_layer_call_fn_840140K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿh
ª "ÿÿÿÿÿÿÿÿÿh§
K__inference_leaky_re_lu_860_layer_call_and_return_conditional_losses_840254X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 
0__inference_leaky_re_lu_860_layer_call_fn_840249K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ/§
K__inference_leaky_re_lu_861_layer_call_and_return_conditional_losses_840363X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 
0__inference_leaky_re_lu_861_layer_call_fn_840358K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ/§
K__inference_leaky_re_lu_862_layer_call_and_return_conditional_losses_840472X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_leaky_re_lu_862_layer_call_fn_840467K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_863_layer_call_and_return_conditional_losses_840581X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_leaky_re_lu_863_layer_call_fn_840576K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_864_layer_call_and_return_conditional_losses_840690X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_leaky_re_lu_864_layer_call_fn_840685K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_865_layer_call_and_return_conditional_losses_840799X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_leaky_re_lu_865_layer_call_fn_840794K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿº
I__inference_sequential_90_layer_call_and_return_conditional_losses_838237ìzæç67B?A@OP[XZYhitqsr¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæñîðïþÿ£ ¢¡°±¼¹»ºÉÊG¢D
=¢:
0-
normalization_90_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
I__inference_sequential_90_layer_call_and_return_conditional_losses_838418ìzæç67AB?@OPZ[XYhistqr¥¦£¤³´¾¿¼½ÌÍ×ØÕÖåæðñîïþÿ¢£ ¡°±»¼¹ºÉÊG¢D
=¢:
0-
normalization_90_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ª
I__inference_sequential_90_layer_call_and_return_conditional_losses_838982Üzæç67B?A@OP[XZYhitqsr¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæñîðïþÿ£ ¢¡°±¼¹»ºÉÊ7¢4
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
I__inference_sequential_90_layer_call_and_return_conditional_losses_839406Üzæç67AB?@OPZ[XYhistqr¥¦£¤³´¾¿¼½ÌÍ×ØÕÖåæðñîïþÿ¢£ ¡°±»¼¹ºÉÊ7¢4
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
.__inference_sequential_90_layer_call_fn_837254ßzæç67B?A@OP[XZYhitqsr¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæñîðïþÿ£ ¢¡°±¼¹»ºÉÊG¢D
=¢:
0-
normalization_90_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_90_layer_call_fn_838056ßzæç67AB?@OPZ[XYhistqr¥¦£¤³´¾¿¼½ÌÍ×ØÕÖåæðñîïþÿ¢£ ¡°±»¼¹ºÉÊG¢D
=¢:
0-
normalization_90_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_90_layer_call_fn_838567Ïzæç67B?A@OP[XZYhitqsr¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæñîðïþÿ£ ¢¡°±¼¹»ºÉÊ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_90_layer_call_fn_838712Ïzæç67AB?@OPZ[XYhistqr¥¦£¤³´¾¿¼½ÌÍ×ØÕÖåæðñîïþÿ¢£ ¡°±»¼¹ºÉÊ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ·
$__inference_signature_wrapper_839553zæç67B?A@OP[XZYhitqsr¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæñîðïþÿ£ ¢¡°±¼¹»ºÉÊY¢V
¢ 
OªL
J
normalization_90_input0-
normalization_90_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_956# 
	dense_956ÿÿÿÿÿÿÿÿÿ