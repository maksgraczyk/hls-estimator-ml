¶:
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ýì4
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
dense_560/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:A*!
shared_namedense_560/kernel
u
$dense_560/kernel/Read/ReadVariableOpReadVariableOpdense_560/kernel*
_output_shapes

:A*
dtype0
t
dense_560/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*
shared_namedense_560/bias
m
"dense_560/bias/Read/ReadVariableOpReadVariableOpdense_560/bias*
_output_shapes
:A*
dtype0

batch_normalization_504/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*.
shared_namebatch_normalization_504/gamma

1batch_normalization_504/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_504/gamma*
_output_shapes
:A*
dtype0

batch_normalization_504/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*-
shared_namebatch_normalization_504/beta

0batch_normalization_504/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_504/beta*
_output_shapes
:A*
dtype0

#batch_normalization_504/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#batch_normalization_504/moving_mean

7batch_normalization_504/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_504/moving_mean*
_output_shapes
:A*
dtype0
¦
'batch_normalization_504/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*8
shared_name)'batch_normalization_504/moving_variance

;batch_normalization_504/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_504/moving_variance*
_output_shapes
:A*
dtype0
|
dense_561/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*!
shared_namedense_561/kernel
u
$dense_561/kernel/Read/ReadVariableOpReadVariableOpdense_561/kernel*
_output_shapes

:AA*
dtype0
t
dense_561/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*
shared_namedense_561/bias
m
"dense_561/bias/Read/ReadVariableOpReadVariableOpdense_561/bias*
_output_shapes
:A*
dtype0

batch_normalization_505/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*.
shared_namebatch_normalization_505/gamma

1batch_normalization_505/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_505/gamma*
_output_shapes
:A*
dtype0

batch_normalization_505/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*-
shared_namebatch_normalization_505/beta

0batch_normalization_505/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_505/beta*
_output_shapes
:A*
dtype0

#batch_normalization_505/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#batch_normalization_505/moving_mean

7batch_normalization_505/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_505/moving_mean*
_output_shapes
:A*
dtype0
¦
'batch_normalization_505/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*8
shared_name)'batch_normalization_505/moving_variance

;batch_normalization_505/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_505/moving_variance*
_output_shapes
:A*
dtype0
|
dense_562/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*!
shared_namedense_562/kernel
u
$dense_562/kernel/Read/ReadVariableOpReadVariableOpdense_562/kernel*
_output_shapes

:AA*
dtype0
t
dense_562/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*
shared_namedense_562/bias
m
"dense_562/bias/Read/ReadVariableOpReadVariableOpdense_562/bias*
_output_shapes
:A*
dtype0

batch_normalization_506/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*.
shared_namebatch_normalization_506/gamma

1batch_normalization_506/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_506/gamma*
_output_shapes
:A*
dtype0

batch_normalization_506/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*-
shared_namebatch_normalization_506/beta

0batch_normalization_506/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_506/beta*
_output_shapes
:A*
dtype0

#batch_normalization_506/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#batch_normalization_506/moving_mean

7batch_normalization_506/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_506/moving_mean*
_output_shapes
:A*
dtype0
¦
'batch_normalization_506/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*8
shared_name)'batch_normalization_506/moving_variance

;batch_normalization_506/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_506/moving_variance*
_output_shapes
:A*
dtype0
|
dense_563/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*!
shared_namedense_563/kernel
u
$dense_563/kernel/Read/ReadVariableOpReadVariableOpdense_563/kernel*
_output_shapes

:AA*
dtype0
t
dense_563/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*
shared_namedense_563/bias
m
"dense_563/bias/Read/ReadVariableOpReadVariableOpdense_563/bias*
_output_shapes
:A*
dtype0

batch_normalization_507/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*.
shared_namebatch_normalization_507/gamma

1batch_normalization_507/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_507/gamma*
_output_shapes
:A*
dtype0

batch_normalization_507/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*-
shared_namebatch_normalization_507/beta

0batch_normalization_507/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_507/beta*
_output_shapes
:A*
dtype0

#batch_normalization_507/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#batch_normalization_507/moving_mean

7batch_normalization_507/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_507/moving_mean*
_output_shapes
:A*
dtype0
¦
'batch_normalization_507/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*8
shared_name)'batch_normalization_507/moving_variance

;batch_normalization_507/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_507/moving_variance*
_output_shapes
:A*
dtype0
|
dense_564/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:A*!
shared_namedense_564/kernel
u
$dense_564/kernel/Read/ReadVariableOpReadVariableOpdense_564/kernel*
_output_shapes

:A*
dtype0
t
dense_564/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_564/bias
m
"dense_564/bias/Read/ReadVariableOpReadVariableOpdense_564/bias*
_output_shapes
:*
dtype0

batch_normalization_508/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_508/gamma

1batch_normalization_508/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_508/gamma*
_output_shapes
:*
dtype0

batch_normalization_508/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_508/beta

0batch_normalization_508/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_508/beta*
_output_shapes
:*
dtype0

#batch_normalization_508/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_508/moving_mean

7batch_normalization_508/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_508/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_508/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_508/moving_variance

;batch_normalization_508/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_508/moving_variance*
_output_shapes
:*
dtype0
|
dense_565/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_565/kernel
u
$dense_565/kernel/Read/ReadVariableOpReadVariableOpdense_565/kernel*
_output_shapes

:*
dtype0
t
dense_565/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_565/bias
m
"dense_565/bias/Read/ReadVariableOpReadVariableOpdense_565/bias*
_output_shapes
:*
dtype0

batch_normalization_509/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_509/gamma

1batch_normalization_509/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_509/gamma*
_output_shapes
:*
dtype0

batch_normalization_509/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_509/beta

0batch_normalization_509/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_509/beta*
_output_shapes
:*
dtype0

#batch_normalization_509/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_509/moving_mean

7batch_normalization_509/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_509/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_509/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_509/moving_variance

;batch_normalization_509/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_509/moving_variance*
_output_shapes
:*
dtype0
|
dense_566/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_566/kernel
u
$dense_566/kernel/Read/ReadVariableOpReadVariableOpdense_566/kernel*
_output_shapes

:*
dtype0
t
dense_566/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_566/bias
m
"dense_566/bias/Read/ReadVariableOpReadVariableOpdense_566/bias*
_output_shapes
:*
dtype0

batch_normalization_510/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_510/gamma

1batch_normalization_510/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_510/gamma*
_output_shapes
:*
dtype0

batch_normalization_510/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_510/beta

0batch_normalization_510/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_510/beta*
_output_shapes
:*
dtype0

#batch_normalization_510/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_510/moving_mean

7batch_normalization_510/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_510/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_510/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_510/moving_variance

;batch_normalization_510/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_510/moving_variance*
_output_shapes
:*
dtype0
|
dense_567/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*!
shared_namedense_567/kernel
u
$dense_567/kernel/Read/ReadVariableOpReadVariableOpdense_567/kernel*
_output_shapes

:5*
dtype0
t
dense_567/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*
shared_namedense_567/bias
m
"dense_567/bias/Read/ReadVariableOpReadVariableOpdense_567/bias*
_output_shapes
:5*
dtype0

batch_normalization_511/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*.
shared_namebatch_normalization_511/gamma

1batch_normalization_511/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_511/gamma*
_output_shapes
:5*
dtype0

batch_normalization_511/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*-
shared_namebatch_normalization_511/beta

0batch_normalization_511/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_511/beta*
_output_shapes
:5*
dtype0

#batch_normalization_511/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#batch_normalization_511/moving_mean

7batch_normalization_511/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_511/moving_mean*
_output_shapes
:5*
dtype0
¦
'batch_normalization_511/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*8
shared_name)'batch_normalization_511/moving_variance

;batch_normalization_511/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_511/moving_variance*
_output_shapes
:5*
dtype0
|
dense_568/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*!
shared_namedense_568/kernel
u
$dense_568/kernel/Read/ReadVariableOpReadVariableOpdense_568/kernel*
_output_shapes

:55*
dtype0
t
dense_568/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*
shared_namedense_568/bias
m
"dense_568/bias/Read/ReadVariableOpReadVariableOpdense_568/bias*
_output_shapes
:5*
dtype0

batch_normalization_512/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*.
shared_namebatch_normalization_512/gamma

1batch_normalization_512/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_512/gamma*
_output_shapes
:5*
dtype0

batch_normalization_512/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*-
shared_namebatch_normalization_512/beta

0batch_normalization_512/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_512/beta*
_output_shapes
:5*
dtype0

#batch_normalization_512/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#batch_normalization_512/moving_mean

7batch_normalization_512/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_512/moving_mean*
_output_shapes
:5*
dtype0
¦
'batch_normalization_512/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*8
shared_name)'batch_normalization_512/moving_variance

;batch_normalization_512/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_512/moving_variance*
_output_shapes
:5*
dtype0
|
dense_569/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*!
shared_namedense_569/kernel
u
$dense_569/kernel/Read/ReadVariableOpReadVariableOpdense_569/kernel*
_output_shapes

:55*
dtype0
t
dense_569/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*
shared_namedense_569/bias
m
"dense_569/bias/Read/ReadVariableOpReadVariableOpdense_569/bias*
_output_shapes
:5*
dtype0

batch_normalization_513/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*.
shared_namebatch_normalization_513/gamma

1batch_normalization_513/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_513/gamma*
_output_shapes
:5*
dtype0

batch_normalization_513/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*-
shared_namebatch_normalization_513/beta

0batch_normalization_513/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_513/beta*
_output_shapes
:5*
dtype0

#batch_normalization_513/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#batch_normalization_513/moving_mean

7batch_normalization_513/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_513/moving_mean*
_output_shapes
:5*
dtype0
¦
'batch_normalization_513/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*8
shared_name)'batch_normalization_513/moving_variance

;batch_normalization_513/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_513/moving_variance*
_output_shapes
:5*
dtype0
|
dense_570/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*!
shared_namedense_570/kernel
u
$dense_570/kernel/Read/ReadVariableOpReadVariableOpdense_570/kernel*
_output_shapes

:55*
dtype0
t
dense_570/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*
shared_namedense_570/bias
m
"dense_570/bias/Read/ReadVariableOpReadVariableOpdense_570/bias*
_output_shapes
:5*
dtype0

batch_normalization_514/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*.
shared_namebatch_normalization_514/gamma

1batch_normalization_514/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_514/gamma*
_output_shapes
:5*
dtype0

batch_normalization_514/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*-
shared_namebatch_normalization_514/beta

0batch_normalization_514/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_514/beta*
_output_shapes
:5*
dtype0

#batch_normalization_514/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#batch_normalization_514/moving_mean

7batch_normalization_514/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_514/moving_mean*
_output_shapes
:5*
dtype0
¦
'batch_normalization_514/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*8
shared_name)'batch_normalization_514/moving_variance

;batch_normalization_514/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_514/moving_variance*
_output_shapes
:5*
dtype0
|
dense_571/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*!
shared_namedense_571/kernel
u
$dense_571/kernel/Read/ReadVariableOpReadVariableOpdense_571/kernel*
_output_shapes

:55*
dtype0
t
dense_571/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*
shared_namedense_571/bias
m
"dense_571/bias/Read/ReadVariableOpReadVariableOpdense_571/bias*
_output_shapes
:5*
dtype0

batch_normalization_515/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*.
shared_namebatch_normalization_515/gamma

1batch_normalization_515/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_515/gamma*
_output_shapes
:5*
dtype0

batch_normalization_515/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*-
shared_namebatch_normalization_515/beta

0batch_normalization_515/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_515/beta*
_output_shapes
:5*
dtype0

#batch_normalization_515/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#batch_normalization_515/moving_mean

7batch_normalization_515/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_515/moving_mean*
_output_shapes
:5*
dtype0
¦
'batch_normalization_515/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*8
shared_name)'batch_normalization_515/moving_variance

;batch_normalization_515/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_515/moving_variance*
_output_shapes
:5*
dtype0
|
dense_572/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*!
shared_namedense_572/kernel
u
$dense_572/kernel/Read/ReadVariableOpReadVariableOpdense_572/kernel*
_output_shapes

:5*
dtype0
t
dense_572/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_572/bias
m
"dense_572/bias/Read/ReadVariableOpReadVariableOpdense_572/bias*
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
Adam/dense_560/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:A*(
shared_nameAdam/dense_560/kernel/m

+Adam/dense_560/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_560/kernel/m*
_output_shapes

:A*
dtype0

Adam/dense_560/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_560/bias/m
{
)Adam/dense_560/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_560/bias/m*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_504/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_504/gamma/m

8Adam/batch_normalization_504/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_504/gamma/m*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_504/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_504/beta/m

7Adam/batch_normalization_504/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_504/beta/m*
_output_shapes
:A*
dtype0

Adam/dense_561/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_561/kernel/m

+Adam/dense_561/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_561/kernel/m*
_output_shapes

:AA*
dtype0

Adam/dense_561/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_561/bias/m
{
)Adam/dense_561/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_561/bias/m*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_505/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_505/gamma/m

8Adam/batch_normalization_505/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_505/gamma/m*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_505/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_505/beta/m

7Adam/batch_normalization_505/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_505/beta/m*
_output_shapes
:A*
dtype0

Adam/dense_562/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_562/kernel/m

+Adam/dense_562/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_562/kernel/m*
_output_shapes

:AA*
dtype0

Adam/dense_562/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_562/bias/m
{
)Adam/dense_562/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_562/bias/m*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_506/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_506/gamma/m

8Adam/batch_normalization_506/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_506/gamma/m*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_506/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_506/beta/m

7Adam/batch_normalization_506/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_506/beta/m*
_output_shapes
:A*
dtype0

Adam/dense_563/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_563/kernel/m

+Adam/dense_563/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_563/kernel/m*
_output_shapes

:AA*
dtype0

Adam/dense_563/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_563/bias/m
{
)Adam/dense_563/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_563/bias/m*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_507/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_507/gamma/m

8Adam/batch_normalization_507/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_507/gamma/m*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_507/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_507/beta/m

7Adam/batch_normalization_507/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_507/beta/m*
_output_shapes
:A*
dtype0

Adam/dense_564/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:A*(
shared_nameAdam/dense_564/kernel/m

+Adam/dense_564/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_564/kernel/m*
_output_shapes

:A*
dtype0

Adam/dense_564/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_564/bias/m
{
)Adam/dense_564/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_564/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_508/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_508/gamma/m

8Adam/batch_normalization_508/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_508/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_508/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_508/beta/m

7Adam/batch_normalization_508/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_508/beta/m*
_output_shapes
:*
dtype0

Adam/dense_565/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_565/kernel/m

+Adam/dense_565/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_565/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_565/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_565/bias/m
{
)Adam/dense_565/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_565/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_509/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_509/gamma/m

8Adam/batch_normalization_509/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_509/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_509/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_509/beta/m

7Adam/batch_normalization_509/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_509/beta/m*
_output_shapes
:*
dtype0

Adam/dense_566/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_566/kernel/m

+Adam/dense_566/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_566/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_566/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_566/bias/m
{
)Adam/dense_566/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_566/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_510/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_510/gamma/m

8Adam/batch_normalization_510/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_510/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_510/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_510/beta/m

7Adam/batch_normalization_510/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_510/beta/m*
_output_shapes
:*
dtype0

Adam/dense_567/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*(
shared_nameAdam/dense_567/kernel/m

+Adam/dense_567/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_567/kernel/m*
_output_shapes

:5*
dtype0

Adam/dense_567/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*&
shared_nameAdam/dense_567/bias/m
{
)Adam/dense_567/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_567/bias/m*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_511/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_511/gamma/m

8Adam/batch_normalization_511/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_511/gamma/m*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_511/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_511/beta/m

7Adam/batch_normalization_511/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_511/beta/m*
_output_shapes
:5*
dtype0

Adam/dense_568/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*(
shared_nameAdam/dense_568/kernel/m

+Adam/dense_568/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_568/kernel/m*
_output_shapes

:55*
dtype0

Adam/dense_568/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*&
shared_nameAdam/dense_568/bias/m
{
)Adam/dense_568/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_568/bias/m*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_512/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_512/gamma/m

8Adam/batch_normalization_512/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_512/gamma/m*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_512/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_512/beta/m

7Adam/batch_normalization_512/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_512/beta/m*
_output_shapes
:5*
dtype0

Adam/dense_569/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*(
shared_nameAdam/dense_569/kernel/m

+Adam/dense_569/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_569/kernel/m*
_output_shapes

:55*
dtype0

Adam/dense_569/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*&
shared_nameAdam/dense_569/bias/m
{
)Adam/dense_569/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_569/bias/m*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_513/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_513/gamma/m

8Adam/batch_normalization_513/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_513/gamma/m*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_513/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_513/beta/m

7Adam/batch_normalization_513/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_513/beta/m*
_output_shapes
:5*
dtype0

Adam/dense_570/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*(
shared_nameAdam/dense_570/kernel/m

+Adam/dense_570/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_570/kernel/m*
_output_shapes

:55*
dtype0

Adam/dense_570/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*&
shared_nameAdam/dense_570/bias/m
{
)Adam/dense_570/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_570/bias/m*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_514/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_514/gamma/m

8Adam/batch_normalization_514/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_514/gamma/m*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_514/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_514/beta/m

7Adam/batch_normalization_514/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_514/beta/m*
_output_shapes
:5*
dtype0

Adam/dense_571/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*(
shared_nameAdam/dense_571/kernel/m

+Adam/dense_571/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_571/kernel/m*
_output_shapes

:55*
dtype0

Adam/dense_571/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*&
shared_nameAdam/dense_571/bias/m
{
)Adam/dense_571/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_571/bias/m*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_515/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_515/gamma/m

8Adam/batch_normalization_515/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_515/gamma/m*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_515/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_515/beta/m

7Adam/batch_normalization_515/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_515/beta/m*
_output_shapes
:5*
dtype0

Adam/dense_572/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*(
shared_nameAdam/dense_572/kernel/m

+Adam/dense_572/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_572/kernel/m*
_output_shapes

:5*
dtype0

Adam/dense_572/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_572/bias/m
{
)Adam/dense_572/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_572/bias/m*
_output_shapes
:*
dtype0

Adam/dense_560/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:A*(
shared_nameAdam/dense_560/kernel/v

+Adam/dense_560/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_560/kernel/v*
_output_shapes

:A*
dtype0

Adam/dense_560/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_560/bias/v
{
)Adam/dense_560/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_560/bias/v*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_504/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_504/gamma/v

8Adam/batch_normalization_504/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_504/gamma/v*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_504/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_504/beta/v

7Adam/batch_normalization_504/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_504/beta/v*
_output_shapes
:A*
dtype0

Adam/dense_561/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_561/kernel/v

+Adam/dense_561/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_561/kernel/v*
_output_shapes

:AA*
dtype0

Adam/dense_561/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_561/bias/v
{
)Adam/dense_561/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_561/bias/v*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_505/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_505/gamma/v

8Adam/batch_normalization_505/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_505/gamma/v*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_505/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_505/beta/v

7Adam/batch_normalization_505/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_505/beta/v*
_output_shapes
:A*
dtype0

Adam/dense_562/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_562/kernel/v

+Adam/dense_562/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_562/kernel/v*
_output_shapes

:AA*
dtype0

Adam/dense_562/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_562/bias/v
{
)Adam/dense_562/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_562/bias/v*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_506/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_506/gamma/v

8Adam/batch_normalization_506/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_506/gamma/v*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_506/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_506/beta/v

7Adam/batch_normalization_506/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_506/beta/v*
_output_shapes
:A*
dtype0

Adam/dense_563/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_563/kernel/v

+Adam/dense_563/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_563/kernel/v*
_output_shapes

:AA*
dtype0

Adam/dense_563/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_563/bias/v
{
)Adam/dense_563/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_563/bias/v*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_507/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_507/gamma/v

8Adam/batch_normalization_507/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_507/gamma/v*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_507/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_507/beta/v

7Adam/batch_normalization_507/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_507/beta/v*
_output_shapes
:A*
dtype0

Adam/dense_564/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:A*(
shared_nameAdam/dense_564/kernel/v

+Adam/dense_564/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_564/kernel/v*
_output_shapes

:A*
dtype0

Adam/dense_564/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_564/bias/v
{
)Adam/dense_564/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_564/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_508/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_508/gamma/v

8Adam/batch_normalization_508/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_508/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_508/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_508/beta/v

7Adam/batch_normalization_508/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_508/beta/v*
_output_shapes
:*
dtype0

Adam/dense_565/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_565/kernel/v

+Adam/dense_565/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_565/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_565/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_565/bias/v
{
)Adam/dense_565/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_565/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_509/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_509/gamma/v

8Adam/batch_normalization_509/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_509/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_509/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_509/beta/v

7Adam/batch_normalization_509/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_509/beta/v*
_output_shapes
:*
dtype0

Adam/dense_566/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_566/kernel/v

+Adam/dense_566/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_566/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_566/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_566/bias/v
{
)Adam/dense_566/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_566/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_510/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_510/gamma/v

8Adam/batch_normalization_510/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_510/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_510/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_510/beta/v

7Adam/batch_normalization_510/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_510/beta/v*
_output_shapes
:*
dtype0

Adam/dense_567/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*(
shared_nameAdam/dense_567/kernel/v

+Adam/dense_567/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_567/kernel/v*
_output_shapes

:5*
dtype0

Adam/dense_567/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*&
shared_nameAdam/dense_567/bias/v
{
)Adam/dense_567/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_567/bias/v*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_511/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_511/gamma/v

8Adam/batch_normalization_511/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_511/gamma/v*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_511/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_511/beta/v

7Adam/batch_normalization_511/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_511/beta/v*
_output_shapes
:5*
dtype0

Adam/dense_568/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*(
shared_nameAdam/dense_568/kernel/v

+Adam/dense_568/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_568/kernel/v*
_output_shapes

:55*
dtype0

Adam/dense_568/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*&
shared_nameAdam/dense_568/bias/v
{
)Adam/dense_568/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_568/bias/v*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_512/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_512/gamma/v

8Adam/batch_normalization_512/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_512/gamma/v*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_512/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_512/beta/v

7Adam/batch_normalization_512/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_512/beta/v*
_output_shapes
:5*
dtype0

Adam/dense_569/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*(
shared_nameAdam/dense_569/kernel/v

+Adam/dense_569/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_569/kernel/v*
_output_shapes

:55*
dtype0

Adam/dense_569/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*&
shared_nameAdam/dense_569/bias/v
{
)Adam/dense_569/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_569/bias/v*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_513/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_513/gamma/v

8Adam/batch_normalization_513/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_513/gamma/v*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_513/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_513/beta/v

7Adam/batch_normalization_513/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_513/beta/v*
_output_shapes
:5*
dtype0

Adam/dense_570/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*(
shared_nameAdam/dense_570/kernel/v

+Adam/dense_570/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_570/kernel/v*
_output_shapes

:55*
dtype0

Adam/dense_570/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*&
shared_nameAdam/dense_570/bias/v
{
)Adam/dense_570/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_570/bias/v*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_514/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_514/gamma/v

8Adam/batch_normalization_514/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_514/gamma/v*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_514/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_514/beta/v

7Adam/batch_normalization_514/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_514/beta/v*
_output_shapes
:5*
dtype0

Adam/dense_571/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*(
shared_nameAdam/dense_571/kernel/v

+Adam/dense_571/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_571/kernel/v*
_output_shapes

:55*
dtype0

Adam/dense_571/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*&
shared_nameAdam/dense_571/bias/v
{
)Adam/dense_571/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_571/bias/v*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_515/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_515/gamma/v

8Adam/batch_normalization_515/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_515/gamma/v*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_515/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_515/beta/v

7Adam/batch_normalization_515/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_515/beta/v*
_output_shapes
:5*
dtype0

Adam/dense_572/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*(
shared_nameAdam/dense_572/kernel/v

+Adam/dense_572/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_572/kernel/v*
_output_shapes

:5*
dtype0

Adam/dense_572/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_572/bias/v
{
)Adam/dense_572/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_572/bias/v*
_output_shapes
:*
dtype0
r
ConstConst*
_output_shapes

:*
dtype0*5
value,B*"WUéBA @@­ªA DAÿÿCATÓ=
t
Const_1Const*
_output_shapes

:*
dtype0*5
value,B*"4sEtæBªª*@"ÇÁAÀB ÀB<

NoOpNoOp
²÷
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*êö
valueßöBÛö BÓö

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
$layer_with_weights-24
$layer-35
%layer-36
&layer_with_weights-25
&layer-37
'	optimizer
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._default_save_signature
/
signatures*
¾
0
_keep_axis
1_reduce_axis
2_reduce_axis_mask
3_broadcast_shape
4mean
4
adapt_mean
5variance
5adapt_variance
	6count
7	keras_api
8_adapt_function*
¦

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses*
Õ
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*

L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses* 
¦

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses*
Õ
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses*

e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses* 
¦

kkernel
lbias
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses*
Õ
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses*

~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£__call__
+¤&call_and_return_all_conditional_losses*
à
	¥axis

¦gamma
	§beta
¨moving_mean
©moving_variance
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses*

°	variables
±trainable_variables
²regularization_losses
³	keras_api
´__call__
+µ&call_and_return_all_conditional_losses* 
®
¶kernel
	·bias
¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses*
à
	¾axis

¿gamma
	Àbeta
Ámoving_mean
Âmoving_variance
Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses*

É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses* 
®
Ïkernel
	Ðbias
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses*
à
	×axis

Øgamma
	Ùbeta
Úmoving_mean
Ûmoving_variance
Ü	variables
Ýtrainable_variables
Þregularization_losses
ß	keras_api
à__call__
+á&call_and_return_all_conditional_losses*

â	variables
ãtrainable_variables
äregularization_losses
å	keras_api
æ__call__
+ç&call_and_return_all_conditional_losses* 
®
èkernel
	ébias
ê	variables
ëtrainable_variables
ìregularization_losses
í	keras_api
î__call__
+ï&call_and_return_all_conditional_losses*
à
	ðaxis

ñgamma
	òbeta
ómoving_mean
ômoving_variance
õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses*

û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+¡&call_and_return_all_conditional_losses*
à
	¢axis

£gamma
	¤beta
¥moving_mean
¦moving_variance
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses*

­	variables
®trainable_variables
¯regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses* 
®
³kernel
	´bias
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api
¹__call__
+º&call_and_return_all_conditional_losses*
à
	»axis

¼gamma
	½beta
¾moving_mean
¿moving_variance
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses*

Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses* 
®
Ìkernel
	Íbias
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses*
à
	Ôaxis

Õgamma
	Öbeta
×moving_mean
Ømoving_variance
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses*

ß	variables
àtrainable_variables
áregularization_losses
â	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses* 
®
åkernel
	æbias
ç	variables
ètrainable_variables
éregularization_losses
ê	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses*
é
	íiter
îbeta_1
ïbeta_2

ðdecay9mµ:m¶Bm·Cm¸Rm¹Smº[m»\m¼km½lm¾tm¿umÀ	mÁ	mÂ	mÃ	mÄ	mÅ	mÆ	¦mÇ	§mÈ	¶mÉ	·mÊ	¿mË	ÀmÌ	ÏmÍ	ÐmÎ	ØmÏ	ÙmÐ	èmÑ	émÒ	ñmÓ	òmÔ	mÕ	mÖ	m×	mØ	mÙ	mÚ	£mÛ	¤mÜ	³mÝ	´mÞ	¼mß	½mà	Ìmá	Ímâ	Õmã	Ömä	åmå	æmæ9vç:vèBvéCvêRvëSvì[ví\vîkvïlvðtvñuvò	vó	vô	võ	vö	v÷	vø	¦vù	§vú	¶vû	·vü	¿vý	Àvþ	Ïvÿ	Ðv	Øv	Ùv	èv	év	ñv	òv	v	v	v	v	v	v	£v	¤v	³v	´v	¼v	½v	Ìv	Ív	Õv	Öv	åv	æv*

40
51
62
93
:4
B5
C6
D7
E8
R9
S10
[11
\12
]13
^14
k15
l16
t17
u18
v19
w20
21
22
23
24
25
26
27
28
¦29
§30
¨31
©32
¶33
·34
¿35
À36
Á37
Â38
Ï39
Ð40
Ø41
Ù42
Ú43
Û44
è45
é46
ñ47
ò48
ó49
ô50
51
52
53
54
55
56
57
58
£59
¤60
¥61
¦62
³63
´64
¼65
½66
¾67
¿68
Ì69
Í70
Õ71
Ö72
×73
Ø74
å75
æ76*
°
90
:1
B2
C3
R4
S5
[6
\7
k8
l9
t10
u11
12
13
14
15
16
17
¦18
§19
¶20
·21
¿22
À23
Ï24
Ð25
Ø26
Ù27
è28
é29
ñ30
ò31
32
33
34
35
36
37
£38
¤39
³40
´41
¼42
½43
Ì44
Í45
Õ46
Ö47
å48
æ49*
* 
µ
ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
._default_save_signature
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
* 

öserving_default* 
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
VARIABLE_VALUEdense_560/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_560/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

90
:1*

90
:1*
* 

÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_504/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_504/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_504/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_504/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
B0
C1
D2
E3*

B0
C1*
* 

ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_561/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_561/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

R0
S1*

R0
S1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_505/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_505/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_505/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_505/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
[0
\1
]2
^3*

[0
\1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_562/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_562/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

k0
l1*

k0
l1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_506/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_506/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_506/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_506/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
t0
u1
v2
w3*

t0
u1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_563/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_563/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_507/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_507/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_507/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_507/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_564/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_564/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
	variables
 trainable_variables
¡regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_508/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_508/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_508/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_508/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
¦0
§1
¨2
©3*

¦0
§1*
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
°	variables
±trainable_variables
²regularization_losses
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_565/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_565/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

¶0
·1*

¶0
·1*
* 

Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_509/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_509/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_509/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_509/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
¿0
À1
Á2
Â3*

¿0
À1*
* 

Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
É	variables
Êtrainable_variables
Ëregularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_566/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_566/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ï0
Ð1*

Ï0
Ð1*
* 

Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
Ñ	variables
Òtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_510/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_510/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_510/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_510/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
Ø0
Ù1
Ú2
Û3*

Ø0
Ù1*
* 

Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
Ü	variables
Ýtrainable_variables
Þregularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
â	variables
ãtrainable_variables
äregularization_losses
æ__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_567/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_567/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

è0
é1*

è0
é1*
* 

ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
ê	variables
ëtrainable_variables
ìregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_511/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_511/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_511/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_511/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
ñ0
ò1
ó2
ô3*

ñ0
ò1*
* 

ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_568/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_568/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_512/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_512/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_512/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_512/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ùnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_569/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_569/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_513/gamma6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_513/beta5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_513/moving_mean<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_513/moving_variance@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
£0
¤1
¥2
¦3*

£0
¤1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
­	variables
®trainable_variables
¯regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_570/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_570/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*

³0
´1*

³0
´1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
µ	variables
¶trainable_variables
·regularization_losses
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_514/gamma6layer_with_weights-22/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_514/beta5layer_with_weights-22/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_514/moving_mean<layer_with_weights-22/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_514/moving_variance@layer_with_weights-22/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
¼0
½1
¾2
¿3*

¼0
½1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Æ	variables
Çtrainable_variables
Èregularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_571/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_571/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ì0
Í1*

Ì0
Í1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_515/gamma6layer_with_weights-24/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_515/beta5layer_with_weights-24/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_515/moving_mean<layer_with_weights-24/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_515/moving_variance@layer_with_weights-24/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
Õ0
Ö1
×2
Ø3*

Õ0
Ö1*
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
Ù	variables
Útrainable_variables
Ûregularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
ß	variables
àtrainable_variables
áregularization_losses
ã__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_572/kernel7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_572/bias5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE*

å0
æ1*

å0
æ1*
* 

«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
ç	variables
ètrainable_variables
éregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses*
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
ä
40
51
62
D3
E4
]5
^6
v7
w8
9
10
¨11
©12
Á13
Â14
Ú15
Û16
ó17
ô18
19
20
¥21
¦22
¾23
¿24
×25
Ø26*
ª
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
#34
$35
%36
&37*

°0*
* 
* 
* 
* 
* 
* 
* 
* 

D0
E1*
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
]0
^1*
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
v0
w1*
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
0
1*
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
¨0
©1*
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
Á0
Â1*
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
Ú0
Û1*
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
ó0
ô1*
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
0
1*
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
¥0
¦1*
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
¾0
¿1*
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
×0
Ø1*
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

±total

²count
³	variables
´	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

±0
²1*

³	variables*
}
VARIABLE_VALUEAdam/dense_560/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_560/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_504/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_504/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_561/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_561/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_505/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_505/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_562/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_562/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_506/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_506/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_563/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_563/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_507/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_507/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_564/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_564/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_508/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_508/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_565/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_565/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_509/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_509/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_566/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_566/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_510/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_510/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_567/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_567/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_511/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_511/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_568/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_568/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_512/gamma/mRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_512/beta/mQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_569/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_569/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_513/gamma/mRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_513/beta/mQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_570/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_570/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_514/gamma/mRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_514/beta/mQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_571/kernel/mSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_571/bias/mQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_515/gamma/mRlayer_with_weights-24/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_515/beta/mQlayer_with_weights-24/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_572/kernel/mSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_572/bias/mQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_560/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_560/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_504/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_504/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_561/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_561/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_505/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_505/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_562/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_562/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_506/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_506/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_563/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_563/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_507/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_507/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_564/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_564/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_508/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_508/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_565/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_565/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_509/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_509/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_566/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_566/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_510/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_510/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_567/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_567/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_511/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_511/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_568/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_568/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_512/gamma/vRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_512/beta/vQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_569/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_569/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_513/gamma/vRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_513/beta/vQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_570/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_570/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_514/gamma/vRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_514/beta/vQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_571/kernel/vSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_571/bias/vQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_515/gamma/vRlayer_with_weights-24/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_515/beta/vQlayer_with_weights-24/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_572/kernel/vSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_572/bias/vQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

&serving_default_normalization_56_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
¦
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_56_inputConstConst_1dense_560/kerneldense_560/bias'batch_normalization_504/moving_variancebatch_normalization_504/gamma#batch_normalization_504/moving_meanbatch_normalization_504/betadense_561/kerneldense_561/bias'batch_normalization_505/moving_variancebatch_normalization_505/gamma#batch_normalization_505/moving_meanbatch_normalization_505/betadense_562/kerneldense_562/bias'batch_normalization_506/moving_variancebatch_normalization_506/gamma#batch_normalization_506/moving_meanbatch_normalization_506/betadense_563/kerneldense_563/bias'batch_normalization_507/moving_variancebatch_normalization_507/gamma#batch_normalization_507/moving_meanbatch_normalization_507/betadense_564/kerneldense_564/bias'batch_normalization_508/moving_variancebatch_normalization_508/gamma#batch_normalization_508/moving_meanbatch_normalization_508/betadense_565/kerneldense_565/bias'batch_normalization_509/moving_variancebatch_normalization_509/gamma#batch_normalization_509/moving_meanbatch_normalization_509/betadense_566/kerneldense_566/bias'batch_normalization_510/moving_variancebatch_normalization_510/gamma#batch_normalization_510/moving_meanbatch_normalization_510/betadense_567/kerneldense_567/bias'batch_normalization_511/moving_variancebatch_normalization_511/gamma#batch_normalization_511/moving_meanbatch_normalization_511/betadense_568/kerneldense_568/bias'batch_normalization_512/moving_variancebatch_normalization_512/gamma#batch_normalization_512/moving_meanbatch_normalization_512/betadense_569/kerneldense_569/bias'batch_normalization_513/moving_variancebatch_normalization_513/gamma#batch_normalization_513/moving_meanbatch_normalization_513/betadense_570/kerneldense_570/bias'batch_normalization_514/moving_variancebatch_normalization_514/gamma#batch_normalization_514/moving_meanbatch_normalization_514/betadense_571/kerneldense_571/bias'batch_normalization_515/moving_variancebatch_normalization_515/gamma#batch_normalization_515/moving_meanbatch_normalization_515/betadense_572/kerneldense_572/bias*X
TinQ
O2M*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*l
_read_only_resource_inputsN
LJ	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKL*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1012925
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
£I
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_560/kernel/Read/ReadVariableOp"dense_560/bias/Read/ReadVariableOp1batch_normalization_504/gamma/Read/ReadVariableOp0batch_normalization_504/beta/Read/ReadVariableOp7batch_normalization_504/moving_mean/Read/ReadVariableOp;batch_normalization_504/moving_variance/Read/ReadVariableOp$dense_561/kernel/Read/ReadVariableOp"dense_561/bias/Read/ReadVariableOp1batch_normalization_505/gamma/Read/ReadVariableOp0batch_normalization_505/beta/Read/ReadVariableOp7batch_normalization_505/moving_mean/Read/ReadVariableOp;batch_normalization_505/moving_variance/Read/ReadVariableOp$dense_562/kernel/Read/ReadVariableOp"dense_562/bias/Read/ReadVariableOp1batch_normalization_506/gamma/Read/ReadVariableOp0batch_normalization_506/beta/Read/ReadVariableOp7batch_normalization_506/moving_mean/Read/ReadVariableOp;batch_normalization_506/moving_variance/Read/ReadVariableOp$dense_563/kernel/Read/ReadVariableOp"dense_563/bias/Read/ReadVariableOp1batch_normalization_507/gamma/Read/ReadVariableOp0batch_normalization_507/beta/Read/ReadVariableOp7batch_normalization_507/moving_mean/Read/ReadVariableOp;batch_normalization_507/moving_variance/Read/ReadVariableOp$dense_564/kernel/Read/ReadVariableOp"dense_564/bias/Read/ReadVariableOp1batch_normalization_508/gamma/Read/ReadVariableOp0batch_normalization_508/beta/Read/ReadVariableOp7batch_normalization_508/moving_mean/Read/ReadVariableOp;batch_normalization_508/moving_variance/Read/ReadVariableOp$dense_565/kernel/Read/ReadVariableOp"dense_565/bias/Read/ReadVariableOp1batch_normalization_509/gamma/Read/ReadVariableOp0batch_normalization_509/beta/Read/ReadVariableOp7batch_normalization_509/moving_mean/Read/ReadVariableOp;batch_normalization_509/moving_variance/Read/ReadVariableOp$dense_566/kernel/Read/ReadVariableOp"dense_566/bias/Read/ReadVariableOp1batch_normalization_510/gamma/Read/ReadVariableOp0batch_normalization_510/beta/Read/ReadVariableOp7batch_normalization_510/moving_mean/Read/ReadVariableOp;batch_normalization_510/moving_variance/Read/ReadVariableOp$dense_567/kernel/Read/ReadVariableOp"dense_567/bias/Read/ReadVariableOp1batch_normalization_511/gamma/Read/ReadVariableOp0batch_normalization_511/beta/Read/ReadVariableOp7batch_normalization_511/moving_mean/Read/ReadVariableOp;batch_normalization_511/moving_variance/Read/ReadVariableOp$dense_568/kernel/Read/ReadVariableOp"dense_568/bias/Read/ReadVariableOp1batch_normalization_512/gamma/Read/ReadVariableOp0batch_normalization_512/beta/Read/ReadVariableOp7batch_normalization_512/moving_mean/Read/ReadVariableOp;batch_normalization_512/moving_variance/Read/ReadVariableOp$dense_569/kernel/Read/ReadVariableOp"dense_569/bias/Read/ReadVariableOp1batch_normalization_513/gamma/Read/ReadVariableOp0batch_normalization_513/beta/Read/ReadVariableOp7batch_normalization_513/moving_mean/Read/ReadVariableOp;batch_normalization_513/moving_variance/Read/ReadVariableOp$dense_570/kernel/Read/ReadVariableOp"dense_570/bias/Read/ReadVariableOp1batch_normalization_514/gamma/Read/ReadVariableOp0batch_normalization_514/beta/Read/ReadVariableOp7batch_normalization_514/moving_mean/Read/ReadVariableOp;batch_normalization_514/moving_variance/Read/ReadVariableOp$dense_571/kernel/Read/ReadVariableOp"dense_571/bias/Read/ReadVariableOp1batch_normalization_515/gamma/Read/ReadVariableOp0batch_normalization_515/beta/Read/ReadVariableOp7batch_normalization_515/moving_mean/Read/ReadVariableOp;batch_normalization_515/moving_variance/Read/ReadVariableOp$dense_572/kernel/Read/ReadVariableOp"dense_572/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_560/kernel/m/Read/ReadVariableOp)Adam/dense_560/bias/m/Read/ReadVariableOp8Adam/batch_normalization_504/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_504/beta/m/Read/ReadVariableOp+Adam/dense_561/kernel/m/Read/ReadVariableOp)Adam/dense_561/bias/m/Read/ReadVariableOp8Adam/batch_normalization_505/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_505/beta/m/Read/ReadVariableOp+Adam/dense_562/kernel/m/Read/ReadVariableOp)Adam/dense_562/bias/m/Read/ReadVariableOp8Adam/batch_normalization_506/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_506/beta/m/Read/ReadVariableOp+Adam/dense_563/kernel/m/Read/ReadVariableOp)Adam/dense_563/bias/m/Read/ReadVariableOp8Adam/batch_normalization_507/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_507/beta/m/Read/ReadVariableOp+Adam/dense_564/kernel/m/Read/ReadVariableOp)Adam/dense_564/bias/m/Read/ReadVariableOp8Adam/batch_normalization_508/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_508/beta/m/Read/ReadVariableOp+Adam/dense_565/kernel/m/Read/ReadVariableOp)Adam/dense_565/bias/m/Read/ReadVariableOp8Adam/batch_normalization_509/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_509/beta/m/Read/ReadVariableOp+Adam/dense_566/kernel/m/Read/ReadVariableOp)Adam/dense_566/bias/m/Read/ReadVariableOp8Adam/batch_normalization_510/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_510/beta/m/Read/ReadVariableOp+Adam/dense_567/kernel/m/Read/ReadVariableOp)Adam/dense_567/bias/m/Read/ReadVariableOp8Adam/batch_normalization_511/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_511/beta/m/Read/ReadVariableOp+Adam/dense_568/kernel/m/Read/ReadVariableOp)Adam/dense_568/bias/m/Read/ReadVariableOp8Adam/batch_normalization_512/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_512/beta/m/Read/ReadVariableOp+Adam/dense_569/kernel/m/Read/ReadVariableOp)Adam/dense_569/bias/m/Read/ReadVariableOp8Adam/batch_normalization_513/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_513/beta/m/Read/ReadVariableOp+Adam/dense_570/kernel/m/Read/ReadVariableOp)Adam/dense_570/bias/m/Read/ReadVariableOp8Adam/batch_normalization_514/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_514/beta/m/Read/ReadVariableOp+Adam/dense_571/kernel/m/Read/ReadVariableOp)Adam/dense_571/bias/m/Read/ReadVariableOp8Adam/batch_normalization_515/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_515/beta/m/Read/ReadVariableOp+Adam/dense_572/kernel/m/Read/ReadVariableOp)Adam/dense_572/bias/m/Read/ReadVariableOp+Adam/dense_560/kernel/v/Read/ReadVariableOp)Adam/dense_560/bias/v/Read/ReadVariableOp8Adam/batch_normalization_504/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_504/beta/v/Read/ReadVariableOp+Adam/dense_561/kernel/v/Read/ReadVariableOp)Adam/dense_561/bias/v/Read/ReadVariableOp8Adam/batch_normalization_505/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_505/beta/v/Read/ReadVariableOp+Adam/dense_562/kernel/v/Read/ReadVariableOp)Adam/dense_562/bias/v/Read/ReadVariableOp8Adam/batch_normalization_506/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_506/beta/v/Read/ReadVariableOp+Adam/dense_563/kernel/v/Read/ReadVariableOp)Adam/dense_563/bias/v/Read/ReadVariableOp8Adam/batch_normalization_507/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_507/beta/v/Read/ReadVariableOp+Adam/dense_564/kernel/v/Read/ReadVariableOp)Adam/dense_564/bias/v/Read/ReadVariableOp8Adam/batch_normalization_508/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_508/beta/v/Read/ReadVariableOp+Adam/dense_565/kernel/v/Read/ReadVariableOp)Adam/dense_565/bias/v/Read/ReadVariableOp8Adam/batch_normalization_509/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_509/beta/v/Read/ReadVariableOp+Adam/dense_566/kernel/v/Read/ReadVariableOp)Adam/dense_566/bias/v/Read/ReadVariableOp8Adam/batch_normalization_510/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_510/beta/v/Read/ReadVariableOp+Adam/dense_567/kernel/v/Read/ReadVariableOp)Adam/dense_567/bias/v/Read/ReadVariableOp8Adam/batch_normalization_511/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_511/beta/v/Read/ReadVariableOp+Adam/dense_568/kernel/v/Read/ReadVariableOp)Adam/dense_568/bias/v/Read/ReadVariableOp8Adam/batch_normalization_512/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_512/beta/v/Read/ReadVariableOp+Adam/dense_569/kernel/v/Read/ReadVariableOp)Adam/dense_569/bias/v/Read/ReadVariableOp8Adam/batch_normalization_513/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_513/beta/v/Read/ReadVariableOp+Adam/dense_570/kernel/v/Read/ReadVariableOp)Adam/dense_570/bias/v/Read/ReadVariableOp8Adam/batch_normalization_514/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_514/beta/v/Read/ReadVariableOp+Adam/dense_571/kernel/v/Read/ReadVariableOp)Adam/dense_571/bias/v/Read/ReadVariableOp8Adam/batch_normalization_515/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_515/beta/v/Read/ReadVariableOp+Adam/dense_572/kernel/v/Read/ReadVariableOp)Adam/dense_572/bias/v/Read/ReadVariableOpConst_2*Ç
Tin¿
¼2¹		*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_1014873
Ð,
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_560/kerneldense_560/biasbatch_normalization_504/gammabatch_normalization_504/beta#batch_normalization_504/moving_mean'batch_normalization_504/moving_variancedense_561/kerneldense_561/biasbatch_normalization_505/gammabatch_normalization_505/beta#batch_normalization_505/moving_mean'batch_normalization_505/moving_variancedense_562/kerneldense_562/biasbatch_normalization_506/gammabatch_normalization_506/beta#batch_normalization_506/moving_mean'batch_normalization_506/moving_variancedense_563/kerneldense_563/biasbatch_normalization_507/gammabatch_normalization_507/beta#batch_normalization_507/moving_mean'batch_normalization_507/moving_variancedense_564/kerneldense_564/biasbatch_normalization_508/gammabatch_normalization_508/beta#batch_normalization_508/moving_mean'batch_normalization_508/moving_variancedense_565/kerneldense_565/biasbatch_normalization_509/gammabatch_normalization_509/beta#batch_normalization_509/moving_mean'batch_normalization_509/moving_variancedense_566/kerneldense_566/biasbatch_normalization_510/gammabatch_normalization_510/beta#batch_normalization_510/moving_mean'batch_normalization_510/moving_variancedense_567/kerneldense_567/biasbatch_normalization_511/gammabatch_normalization_511/beta#batch_normalization_511/moving_mean'batch_normalization_511/moving_variancedense_568/kerneldense_568/biasbatch_normalization_512/gammabatch_normalization_512/beta#batch_normalization_512/moving_mean'batch_normalization_512/moving_variancedense_569/kerneldense_569/biasbatch_normalization_513/gammabatch_normalization_513/beta#batch_normalization_513/moving_mean'batch_normalization_513/moving_variancedense_570/kerneldense_570/biasbatch_normalization_514/gammabatch_normalization_514/beta#batch_normalization_514/moving_mean'batch_normalization_514/moving_variancedense_571/kerneldense_571/biasbatch_normalization_515/gammabatch_normalization_515/beta#batch_normalization_515/moving_mean'batch_normalization_515/moving_variancedense_572/kerneldense_572/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_560/kernel/mAdam/dense_560/bias/m$Adam/batch_normalization_504/gamma/m#Adam/batch_normalization_504/beta/mAdam/dense_561/kernel/mAdam/dense_561/bias/m$Adam/batch_normalization_505/gamma/m#Adam/batch_normalization_505/beta/mAdam/dense_562/kernel/mAdam/dense_562/bias/m$Adam/batch_normalization_506/gamma/m#Adam/batch_normalization_506/beta/mAdam/dense_563/kernel/mAdam/dense_563/bias/m$Adam/batch_normalization_507/gamma/m#Adam/batch_normalization_507/beta/mAdam/dense_564/kernel/mAdam/dense_564/bias/m$Adam/batch_normalization_508/gamma/m#Adam/batch_normalization_508/beta/mAdam/dense_565/kernel/mAdam/dense_565/bias/m$Adam/batch_normalization_509/gamma/m#Adam/batch_normalization_509/beta/mAdam/dense_566/kernel/mAdam/dense_566/bias/m$Adam/batch_normalization_510/gamma/m#Adam/batch_normalization_510/beta/mAdam/dense_567/kernel/mAdam/dense_567/bias/m$Adam/batch_normalization_511/gamma/m#Adam/batch_normalization_511/beta/mAdam/dense_568/kernel/mAdam/dense_568/bias/m$Adam/batch_normalization_512/gamma/m#Adam/batch_normalization_512/beta/mAdam/dense_569/kernel/mAdam/dense_569/bias/m$Adam/batch_normalization_513/gamma/m#Adam/batch_normalization_513/beta/mAdam/dense_570/kernel/mAdam/dense_570/bias/m$Adam/batch_normalization_514/gamma/m#Adam/batch_normalization_514/beta/mAdam/dense_571/kernel/mAdam/dense_571/bias/m$Adam/batch_normalization_515/gamma/m#Adam/batch_normalization_515/beta/mAdam/dense_572/kernel/mAdam/dense_572/bias/mAdam/dense_560/kernel/vAdam/dense_560/bias/v$Adam/batch_normalization_504/gamma/v#Adam/batch_normalization_504/beta/vAdam/dense_561/kernel/vAdam/dense_561/bias/v$Adam/batch_normalization_505/gamma/v#Adam/batch_normalization_505/beta/vAdam/dense_562/kernel/vAdam/dense_562/bias/v$Adam/batch_normalization_506/gamma/v#Adam/batch_normalization_506/beta/vAdam/dense_563/kernel/vAdam/dense_563/bias/v$Adam/batch_normalization_507/gamma/v#Adam/batch_normalization_507/beta/vAdam/dense_564/kernel/vAdam/dense_564/bias/v$Adam/batch_normalization_508/gamma/v#Adam/batch_normalization_508/beta/vAdam/dense_565/kernel/vAdam/dense_565/bias/v$Adam/batch_normalization_509/gamma/v#Adam/batch_normalization_509/beta/vAdam/dense_566/kernel/vAdam/dense_566/bias/v$Adam/batch_normalization_510/gamma/v#Adam/batch_normalization_510/beta/vAdam/dense_567/kernel/vAdam/dense_567/bias/v$Adam/batch_normalization_511/gamma/v#Adam/batch_normalization_511/beta/vAdam/dense_568/kernel/vAdam/dense_568/bias/v$Adam/batch_normalization_512/gamma/v#Adam/batch_normalization_512/beta/vAdam/dense_569/kernel/vAdam/dense_569/bias/v$Adam/batch_normalization_513/gamma/v#Adam/batch_normalization_513/beta/vAdam/dense_570/kernel/vAdam/dense_570/bias/v$Adam/batch_normalization_514/gamma/v#Adam/batch_normalization_514/beta/vAdam/dense_571/kernel/vAdam/dense_571/bias/v$Adam/batch_normalization_515/gamma/v#Adam/batch_normalization_515/beta/vAdam/dense_572/kernel/vAdam/dense_572/bias/v*Æ
Tin¾
»2¸*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_1015432¬Ó-
%
í
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1009196

inputs5
'assignmovingavg_readvariableop_resource:A7
)assignmovingavg_1_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A/
!batchnorm_readvariableop_resource:A
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:A
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:A*
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
:A*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A¬
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
:A*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:A~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A´
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_507_layer_call_fn_1013344

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1009196o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
É	
÷
F__inference_dense_567_layer_call_and_return_conditional_losses_1013754

inputs0
matmul_readvariableop_resource:5-
biasadd_readvariableop_resource:5
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:5*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_dense_572_layer_call_fn_1014289

inputs
unknown:5
	unknown_0:
identity¢StatefulPartitionedCallÛ
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
GPU 2J 8 *O
fJRH
F__inference_dense_572_layer_call_and_return_conditional_losses_1010271o
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
:ÿÿÿÿÿÿÿÿÿ5: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1013146

inputs/
!batchnorm_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A1
#batchnorm_readvariableop_1_resource:A1
#batchnorm_readvariableop_2_resource:A
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1013616

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1013180

inputs5
'assignmovingavg_readvariableop_resource:A7
)assignmovingavg_1_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A/
!batchnorm_readvariableop_resource:A
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:A
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:A*
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
:A*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A¬
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
:A*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:A~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A´
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_511_layer_call_fn_1013767

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1009477o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1013582

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_514_layer_call_fn_1014166

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_514_layer_call_and_return_conditional_losses_1010227`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1014052

inputs5
'assignmovingavg_readvariableop_resource:57
)assignmovingavg_1_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:5/
!batchnorm_readvariableop_resource:5
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:5
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:5*
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
:5*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:5x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5¬
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
:5*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:5~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5´
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:5v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
É	
÷
F__inference_dense_570_layer_call_and_return_conditional_losses_1010207

inputs0
matmul_readvariableop_resource:55-
biasadd_readvariableop_resource:5
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:55*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:5*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1013834

inputs5
'assignmovingavg_readvariableop_resource:57
)assignmovingavg_1_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:5/
!batchnorm_readvariableop_resource:5
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:5
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:5*
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
:5*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:5x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5¬
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
:5*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:5~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5´
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:5v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_508_layer_call_fn_1013512

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_508_layer_call_and_return_conditional_losses_1010035`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1009641

inputs/
!batchnorm_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:51
#batchnorm_readvariableop_1_resource:51
#batchnorm_readvariableop_2_resource:5
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:5z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_506_layer_call_fn_1013235

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1009114o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
ñ 
Ã
/__inference_sequential_56_layer_call_fn_1011302
normalization_56_input
unknown
	unknown_0
	unknown_1:A
	unknown_2:A
	unknown_3:A
	unknown_4:A
	unknown_5:A
	unknown_6:A
	unknown_7:AA
	unknown_8:A
	unknown_9:A

unknown_10:A

unknown_11:A

unknown_12:A

unknown_13:AA

unknown_14:A

unknown_15:A

unknown_16:A

unknown_17:A

unknown_18:A

unknown_19:AA

unknown_20:A

unknown_21:A

unknown_22:A

unknown_23:A

unknown_24:A

unknown_25:A

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:5

unknown_44:5

unknown_45:5

unknown_46:5

unknown_47:5

unknown_48:5

unknown_49:55

unknown_50:5

unknown_51:5

unknown_52:5

unknown_53:5

unknown_54:5

unknown_55:55

unknown_56:5

unknown_57:5

unknown_58:5

unknown_59:5

unknown_60:5

unknown_61:55

unknown_62:5

unknown_63:5

unknown_64:5

unknown_65:5

unknown_66:5

unknown_67:55

unknown_68:5

unknown_69:5

unknown_70:5

unknown_71:5

unknown_72:5

unknown_73:5

unknown_74:
identity¢StatefulPartitionedCallØ

StatefulPartitionedCallStatefulPartitionedCallnormalization_56_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74*X
TinQ
O2M*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"%&'(+,-.1234789:=>?@CDEFIJKL*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_56_layer_call_and_return_conditional_losses_1010990o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ð
_input_shapes¾
»:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_56_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ñ
³
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1009805

inputs/
!batchnorm_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:51
#batchnorm_readvariableop_1_resource:51
#batchnorm_readvariableop_2_resource:5
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:5z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_507_layer_call_fn_1013403

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_1010003`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿA:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_511_layer_call_and_return_conditional_losses_1013844

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1014161

inputs5
'assignmovingavg_readvariableop_resource:57
)assignmovingavg_1_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:5/
!batchnorm_readvariableop_resource:5
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:5
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:5*
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
:5*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:5x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5¬
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
:5*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:5~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5´
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:5v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1009688

inputs5
'assignmovingavg_readvariableop_resource:57
)assignmovingavg_1_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:5/
!batchnorm_readvariableop_resource:5
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:5
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:5*
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
:5*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:5x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5¬
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
:5*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:5~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5´
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:5v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1013800

inputs/
!batchnorm_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:51
#batchnorm_readvariableop_1_resource:51
#batchnorm_readvariableop_2_resource:5
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:5z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_1009939

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿA:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_504_layer_call_fn_1013076

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_1009907`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿA:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
É	
÷
F__inference_dense_572_layer_call_and_return_conditional_losses_1014299

inputs0
matmul_readvariableop_resource:5-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
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
:ÿÿÿÿÿÿÿÿÿ5: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1009231

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É	
÷
F__inference_dense_567_layer_call_and_return_conditional_losses_1010111

inputs0
matmul_readvariableop_resource:5-
biasadd_readvariableop_resource:5
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:5*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_510_layer_call_and_return_conditional_losses_1013735

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1009313

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_506_layer_call_fn_1013222

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1009067o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
Æ

+__inference_dense_563_layer_call_fn_1013308

inputs
unknown:AA
	unknown_0:A
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_563_layer_call_and_return_conditional_losses_1009983o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1013071

inputs5
'assignmovingavg_readvariableop_resource:A7
)assignmovingavg_1_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A/
!batchnorm_readvariableop_resource:A
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:A
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:A*
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
:A*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A¬
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
:A*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:A~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A´
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1009067

inputs/
!batchnorm_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A1
#batchnorm_readvariableop_1_resource:A1
#batchnorm_readvariableop_2_resource:A
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
°Å
¼"
J__inference_sequential_56_layer_call_and_return_conditional_losses_1011498
normalization_56_input
normalization_56_sub_y
normalization_56_sqrt_x#
dense_560_1011312:A
dense_560_1011314:A-
batch_normalization_504_1011317:A-
batch_normalization_504_1011319:A-
batch_normalization_504_1011321:A-
batch_normalization_504_1011323:A#
dense_561_1011327:AA
dense_561_1011329:A-
batch_normalization_505_1011332:A-
batch_normalization_505_1011334:A-
batch_normalization_505_1011336:A-
batch_normalization_505_1011338:A#
dense_562_1011342:AA
dense_562_1011344:A-
batch_normalization_506_1011347:A-
batch_normalization_506_1011349:A-
batch_normalization_506_1011351:A-
batch_normalization_506_1011353:A#
dense_563_1011357:AA
dense_563_1011359:A-
batch_normalization_507_1011362:A-
batch_normalization_507_1011364:A-
batch_normalization_507_1011366:A-
batch_normalization_507_1011368:A#
dense_564_1011372:A
dense_564_1011374:-
batch_normalization_508_1011377:-
batch_normalization_508_1011379:-
batch_normalization_508_1011381:-
batch_normalization_508_1011383:#
dense_565_1011387:
dense_565_1011389:-
batch_normalization_509_1011392:-
batch_normalization_509_1011394:-
batch_normalization_509_1011396:-
batch_normalization_509_1011398:#
dense_566_1011402:
dense_566_1011404:-
batch_normalization_510_1011407:-
batch_normalization_510_1011409:-
batch_normalization_510_1011411:-
batch_normalization_510_1011413:#
dense_567_1011417:5
dense_567_1011419:5-
batch_normalization_511_1011422:5-
batch_normalization_511_1011424:5-
batch_normalization_511_1011426:5-
batch_normalization_511_1011428:5#
dense_568_1011432:55
dense_568_1011434:5-
batch_normalization_512_1011437:5-
batch_normalization_512_1011439:5-
batch_normalization_512_1011441:5-
batch_normalization_512_1011443:5#
dense_569_1011447:55
dense_569_1011449:5-
batch_normalization_513_1011452:5-
batch_normalization_513_1011454:5-
batch_normalization_513_1011456:5-
batch_normalization_513_1011458:5#
dense_570_1011462:55
dense_570_1011464:5-
batch_normalization_514_1011467:5-
batch_normalization_514_1011469:5-
batch_normalization_514_1011471:5-
batch_normalization_514_1011473:5#
dense_571_1011477:55
dense_571_1011479:5-
batch_normalization_515_1011482:5-
batch_normalization_515_1011484:5-
batch_normalization_515_1011486:5-
batch_normalization_515_1011488:5#
dense_572_1011492:5
dense_572_1011494:
identity¢/batch_normalization_504/StatefulPartitionedCall¢/batch_normalization_505/StatefulPartitionedCall¢/batch_normalization_506/StatefulPartitionedCall¢/batch_normalization_507/StatefulPartitionedCall¢/batch_normalization_508/StatefulPartitionedCall¢/batch_normalization_509/StatefulPartitionedCall¢/batch_normalization_510/StatefulPartitionedCall¢/batch_normalization_511/StatefulPartitionedCall¢/batch_normalization_512/StatefulPartitionedCall¢/batch_normalization_513/StatefulPartitionedCall¢/batch_normalization_514/StatefulPartitionedCall¢/batch_normalization_515/StatefulPartitionedCall¢!dense_560/StatefulPartitionedCall¢!dense_561/StatefulPartitionedCall¢!dense_562/StatefulPartitionedCall¢!dense_563/StatefulPartitionedCall¢!dense_564/StatefulPartitionedCall¢!dense_565/StatefulPartitionedCall¢!dense_566/StatefulPartitionedCall¢!dense_567/StatefulPartitionedCall¢!dense_568/StatefulPartitionedCall¢!dense_569/StatefulPartitionedCall¢!dense_570/StatefulPartitionedCall¢!dense_571/StatefulPartitionedCall¢!dense_572/StatefulPartitionedCall}
normalization_56/subSubnormalization_56_inputnormalization_56_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_56/SqrtSqrtnormalization_56_sqrt_x*
T0*
_output_shapes

:_
normalization_56/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_56/MaximumMaximumnormalization_56/Sqrt:y:0#normalization_56/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_56/truedivRealDivnormalization_56/sub:z:0normalization_56/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_560/StatefulPartitionedCallStatefulPartitionedCallnormalization_56/truediv:z:0dense_560_1011312dense_560_1011314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_560_layer_call_and_return_conditional_losses_1009887
/batch_normalization_504/StatefulPartitionedCallStatefulPartitionedCall*dense_560/StatefulPartitionedCall:output:0batch_normalization_504_1011317batch_normalization_504_1011319batch_normalization_504_1011321batch_normalization_504_1011323*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1008903ù
leaky_re_lu_504/PartitionedCallPartitionedCall8batch_normalization_504/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_1009907
!dense_561/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_504/PartitionedCall:output:0dense_561_1011327dense_561_1011329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_561_layer_call_and_return_conditional_losses_1009919
/batch_normalization_505/StatefulPartitionedCallStatefulPartitionedCall*dense_561/StatefulPartitionedCall:output:0batch_normalization_505_1011332batch_normalization_505_1011334batch_normalization_505_1011336batch_normalization_505_1011338*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1008985ù
leaky_re_lu_505/PartitionedCallPartitionedCall8batch_normalization_505/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_1009939
!dense_562/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_505/PartitionedCall:output:0dense_562_1011342dense_562_1011344*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_562_layer_call_and_return_conditional_losses_1009951
/batch_normalization_506/StatefulPartitionedCallStatefulPartitionedCall*dense_562/StatefulPartitionedCall:output:0batch_normalization_506_1011347batch_normalization_506_1011349batch_normalization_506_1011351batch_normalization_506_1011353*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1009067ù
leaky_re_lu_506/PartitionedCallPartitionedCall8batch_normalization_506/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_1009971
!dense_563/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_506/PartitionedCall:output:0dense_563_1011357dense_563_1011359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_563_layer_call_and_return_conditional_losses_1009983
/batch_normalization_507/StatefulPartitionedCallStatefulPartitionedCall*dense_563/StatefulPartitionedCall:output:0batch_normalization_507_1011362batch_normalization_507_1011364batch_normalization_507_1011366batch_normalization_507_1011368*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1009149ù
leaky_re_lu_507/PartitionedCallPartitionedCall8batch_normalization_507/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_1010003
!dense_564/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_507/PartitionedCall:output:0dense_564_1011372dense_564_1011374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_564_layer_call_and_return_conditional_losses_1010015
/batch_normalization_508/StatefulPartitionedCallStatefulPartitionedCall*dense_564/StatefulPartitionedCall:output:0batch_normalization_508_1011377batch_normalization_508_1011379batch_normalization_508_1011381batch_normalization_508_1011383*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1009231ù
leaky_re_lu_508/PartitionedCallPartitionedCall8batch_normalization_508/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_508_layer_call_and_return_conditional_losses_1010035
!dense_565/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_508/PartitionedCall:output:0dense_565_1011387dense_565_1011389*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_565_layer_call_and_return_conditional_losses_1010047
/batch_normalization_509/StatefulPartitionedCallStatefulPartitionedCall*dense_565/StatefulPartitionedCall:output:0batch_normalization_509_1011392batch_normalization_509_1011394batch_normalization_509_1011396batch_normalization_509_1011398*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1009313ù
leaky_re_lu_509/PartitionedCallPartitionedCall8batch_normalization_509/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_509_layer_call_and_return_conditional_losses_1010067
!dense_566/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_509/PartitionedCall:output:0dense_566_1011402dense_566_1011404*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_566_layer_call_and_return_conditional_losses_1010079
/batch_normalization_510/StatefulPartitionedCallStatefulPartitionedCall*dense_566/StatefulPartitionedCall:output:0batch_normalization_510_1011407batch_normalization_510_1011409batch_normalization_510_1011411batch_normalization_510_1011413*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1009395ù
leaky_re_lu_510/PartitionedCallPartitionedCall8batch_normalization_510/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_510_layer_call_and_return_conditional_losses_1010099
!dense_567/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_510/PartitionedCall:output:0dense_567_1011417dense_567_1011419*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_567_layer_call_and_return_conditional_losses_1010111
/batch_normalization_511/StatefulPartitionedCallStatefulPartitionedCall*dense_567/StatefulPartitionedCall:output:0batch_normalization_511_1011422batch_normalization_511_1011424batch_normalization_511_1011426batch_normalization_511_1011428*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1009477ù
leaky_re_lu_511/PartitionedCallPartitionedCall8batch_normalization_511/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_511_layer_call_and_return_conditional_losses_1010131
!dense_568/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_511/PartitionedCall:output:0dense_568_1011432dense_568_1011434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_568_layer_call_and_return_conditional_losses_1010143
/batch_normalization_512/StatefulPartitionedCallStatefulPartitionedCall*dense_568/StatefulPartitionedCall:output:0batch_normalization_512_1011437batch_normalization_512_1011439batch_normalization_512_1011441batch_normalization_512_1011443*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1009559ù
leaky_re_lu_512/PartitionedCallPartitionedCall8batch_normalization_512/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_512_layer_call_and_return_conditional_losses_1010163
!dense_569/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_512/PartitionedCall:output:0dense_569_1011447dense_569_1011449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_569_layer_call_and_return_conditional_losses_1010175
/batch_normalization_513/StatefulPartitionedCallStatefulPartitionedCall*dense_569/StatefulPartitionedCall:output:0batch_normalization_513_1011452batch_normalization_513_1011454batch_normalization_513_1011456batch_normalization_513_1011458*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1009641ù
leaky_re_lu_513/PartitionedCallPartitionedCall8batch_normalization_513/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_513_layer_call_and_return_conditional_losses_1010195
!dense_570/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_513/PartitionedCall:output:0dense_570_1011462dense_570_1011464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_570_layer_call_and_return_conditional_losses_1010207
/batch_normalization_514/StatefulPartitionedCallStatefulPartitionedCall*dense_570/StatefulPartitionedCall:output:0batch_normalization_514_1011467batch_normalization_514_1011469batch_normalization_514_1011471batch_normalization_514_1011473*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1009723ù
leaky_re_lu_514/PartitionedCallPartitionedCall8batch_normalization_514/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_514_layer_call_and_return_conditional_losses_1010227
!dense_571/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_514/PartitionedCall:output:0dense_571_1011477dense_571_1011479*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_571_layer_call_and_return_conditional_losses_1010239
/batch_normalization_515/StatefulPartitionedCallStatefulPartitionedCall*dense_571/StatefulPartitionedCall:output:0batch_normalization_515_1011482batch_normalization_515_1011484batch_normalization_515_1011486batch_normalization_515_1011488*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1009805ù
leaky_re_lu_515/PartitionedCallPartitionedCall8batch_normalization_515/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_515_layer_call_and_return_conditional_losses_1010259
!dense_572/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_515/PartitionedCall:output:0dense_572_1011492dense_572_1011494*
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
GPU 2J 8 *O
fJRH
F__inference_dense_572_layer_call_and_return_conditional_losses_1010271y
IdentityIdentity*dense_572/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
NoOpNoOp0^batch_normalization_504/StatefulPartitionedCall0^batch_normalization_505/StatefulPartitionedCall0^batch_normalization_506/StatefulPartitionedCall0^batch_normalization_507/StatefulPartitionedCall0^batch_normalization_508/StatefulPartitionedCall0^batch_normalization_509/StatefulPartitionedCall0^batch_normalization_510/StatefulPartitionedCall0^batch_normalization_511/StatefulPartitionedCall0^batch_normalization_512/StatefulPartitionedCall0^batch_normalization_513/StatefulPartitionedCall0^batch_normalization_514/StatefulPartitionedCall0^batch_normalization_515/StatefulPartitionedCall"^dense_560/StatefulPartitionedCall"^dense_561/StatefulPartitionedCall"^dense_562/StatefulPartitionedCall"^dense_563/StatefulPartitionedCall"^dense_564/StatefulPartitionedCall"^dense_565/StatefulPartitionedCall"^dense_566/StatefulPartitionedCall"^dense_567/StatefulPartitionedCall"^dense_568/StatefulPartitionedCall"^dense_569/StatefulPartitionedCall"^dense_570/StatefulPartitionedCall"^dense_571/StatefulPartitionedCall"^dense_572/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ð
_input_shapes¾
»:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_504/StatefulPartitionedCall/batch_normalization_504/StatefulPartitionedCall2b
/batch_normalization_505/StatefulPartitionedCall/batch_normalization_505/StatefulPartitionedCall2b
/batch_normalization_506/StatefulPartitionedCall/batch_normalization_506/StatefulPartitionedCall2b
/batch_normalization_507/StatefulPartitionedCall/batch_normalization_507/StatefulPartitionedCall2b
/batch_normalization_508/StatefulPartitionedCall/batch_normalization_508/StatefulPartitionedCall2b
/batch_normalization_509/StatefulPartitionedCall/batch_normalization_509/StatefulPartitionedCall2b
/batch_normalization_510/StatefulPartitionedCall/batch_normalization_510/StatefulPartitionedCall2b
/batch_normalization_511/StatefulPartitionedCall/batch_normalization_511/StatefulPartitionedCall2b
/batch_normalization_512/StatefulPartitionedCall/batch_normalization_512/StatefulPartitionedCall2b
/batch_normalization_513/StatefulPartitionedCall/batch_normalization_513/StatefulPartitionedCall2b
/batch_normalization_514/StatefulPartitionedCall/batch_normalization_514/StatefulPartitionedCall2b
/batch_normalization_515/StatefulPartitionedCall/batch_normalization_515/StatefulPartitionedCall2F
!dense_560/StatefulPartitionedCall!dense_560/StatefulPartitionedCall2F
!dense_561/StatefulPartitionedCall!dense_561/StatefulPartitionedCall2F
!dense_562/StatefulPartitionedCall!dense_562/StatefulPartitionedCall2F
!dense_563/StatefulPartitionedCall!dense_563/StatefulPartitionedCall2F
!dense_564/StatefulPartitionedCall!dense_564/StatefulPartitionedCall2F
!dense_565/StatefulPartitionedCall!dense_565/StatefulPartitionedCall2F
!dense_566/StatefulPartitionedCall!dense_566/StatefulPartitionedCall2F
!dense_567/StatefulPartitionedCall!dense_567/StatefulPartitionedCall2F
!dense_568/StatefulPartitionedCall!dense_568/StatefulPartitionedCall2F
!dense_569/StatefulPartitionedCall!dense_569/StatefulPartitionedCall2F
!dense_570/StatefulPartitionedCall!dense_570/StatefulPartitionedCall2F
!dense_571/StatefulPartitionedCall!dense_571/StatefulPartitionedCall2F
!dense_572/StatefulPartitionedCall!dense_572/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_56_input:$ 

_output_shapes

::$ 

_output_shapes

:
É	
÷
F__inference_dense_569_layer_call_and_return_conditional_losses_1010175

inputs0
matmul_readvariableop_resource:55-
biasadd_readvariableop_resource:5
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:55*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:5*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_513_layer_call_and_return_conditional_losses_1010195

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
É	
÷
F__inference_dense_561_layer_call_and_return_conditional_losses_1013100

inputs0
matmul_readvariableop_resource:AA-
biasadd_readvariableop_resource:A
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:AA*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
É	
÷
F__inference_dense_568_layer_call_and_return_conditional_losses_1010143

inputs0
matmul_readvariableop_resource:55-
biasadd_readvariableop_resource:5
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:55*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:5*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_514_layer_call_and_return_conditional_losses_1010227

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_505_layer_call_fn_1013113

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1008985o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1009770

inputs5
'assignmovingavg_readvariableop_resource:57
)assignmovingavg_1_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:5/
!batchnorm_readvariableop_resource:5
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:5
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:5*
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
:5*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:5x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5¬
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
:5*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:5~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5´
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:5v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
ï'
Ó
__inference_adapt_step_1012972
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢IteratorGetNext¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢add/ReadVariableOp±
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
output_shapes
:ÿÿÿÿÿÿÿÿÿ*
output_types
2k
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
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
:
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
æ
h
L__inference_leaky_re_lu_512_layer_call_and_return_conditional_losses_1010163

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1013473

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_505_layer_call_fn_1013126

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1009032o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
É	
÷
F__inference_dense_571_layer_call_and_return_conditional_losses_1010239

inputs0
matmul_readvariableop_resource:55-
biasadd_readvariableop_resource:5
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:55*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:5*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
!
Ã
/__inference_sequential_56_layer_call_fn_1010433
normalization_56_input
unknown
	unknown_0
	unknown_1:A
	unknown_2:A
	unknown_3:A
	unknown_4:A
	unknown_5:A
	unknown_6:A
	unknown_7:AA
	unknown_8:A
	unknown_9:A

unknown_10:A

unknown_11:A

unknown_12:A

unknown_13:AA

unknown_14:A

unknown_15:A

unknown_16:A

unknown_17:A

unknown_18:A

unknown_19:AA

unknown_20:A

unknown_21:A

unknown_22:A

unknown_23:A

unknown_24:A

unknown_25:A

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:5

unknown_44:5

unknown_45:5

unknown_46:5

unknown_47:5

unknown_48:5

unknown_49:55

unknown_50:5

unknown_51:5

unknown_52:5

unknown_53:5

unknown_54:5

unknown_55:55

unknown_56:5

unknown_57:5

unknown_58:5

unknown_59:5

unknown_60:5

unknown_61:55

unknown_62:5

unknown_63:5

unknown_64:5

unknown_65:5

unknown_66:5

unknown_67:55

unknown_68:5

unknown_69:5

unknown_70:5

unknown_71:5

unknown_72:5

unknown_73:5

unknown_74:
identity¢StatefulPartitionedCallð

StatefulPartitionedCallStatefulPartitionedCallnormalization_56_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74*X
TinQ
O2M*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*l
_read_only_resource_inputsN
LJ	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKL*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_56_layer_call_and_return_conditional_losses_1010278o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ð
_input_shapes¾
»:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_56_input:$ 

_output_shapes

::$ 

_output_shapes

:
Æ

+__inference_dense_560_layer_call_fn_1012981

inputs
unknown:A
	unknown_0:A
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_560_layer_call_and_return_conditional_losses_1009887o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É	
÷
F__inference_dense_564_layer_call_and_return_conditional_losses_1010015

inputs0
matmul_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:A*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_506_layer_call_fn_1013294

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_1009971`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿA:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1013037

inputs/
!batchnorm_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A1
#batchnorm_readvariableop_1_resource:A1
#batchnorm_readvariableop_2_resource:A
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1009477

inputs/
!batchnorm_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:51
#batchnorm_readvariableop_1_resource:51
#batchnorm_readvariableop_2_resource:5
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:5z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1009852

inputs5
'assignmovingavg_readvariableop_resource:57
)assignmovingavg_1_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:5/
!batchnorm_readvariableop_resource:5
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:5
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:5*
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
:5*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:5x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5¬
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
:5*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:5~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5´
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:5v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
É	
÷
F__inference_dense_562_layer_call_and_return_conditional_losses_1013209

inputs0
matmul_readvariableop_resource:AA-
biasadd_readvariableop_resource:A
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:AA*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
É	
÷
F__inference_dense_571_layer_call_and_return_conditional_losses_1014190

inputs0
matmul_readvariableop_resource:55-
biasadd_readvariableop_resource:5
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:55*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:5*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_512_layer_call_fn_1013948

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_512_layer_call_and_return_conditional_losses_1010163`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1013507

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_507_layer_call_fn_1013331

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1009149o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
É	
÷
F__inference_dense_564_layer_call_and_return_conditional_losses_1013427

inputs0
matmul_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:A*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1013725

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1009559

inputs/
!batchnorm_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:51
#batchnorm_readvariableop_1_resource:51
#batchnorm_readvariableop_2_resource:5
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:5z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_509_layer_call_fn_1013562

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1009360o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°Ü
V
 __inference__traced_save_1014873
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_560_kernel_read_readvariableop-
)savev2_dense_560_bias_read_readvariableop<
8savev2_batch_normalization_504_gamma_read_readvariableop;
7savev2_batch_normalization_504_beta_read_readvariableopB
>savev2_batch_normalization_504_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_504_moving_variance_read_readvariableop/
+savev2_dense_561_kernel_read_readvariableop-
)savev2_dense_561_bias_read_readvariableop<
8savev2_batch_normalization_505_gamma_read_readvariableop;
7savev2_batch_normalization_505_beta_read_readvariableopB
>savev2_batch_normalization_505_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_505_moving_variance_read_readvariableop/
+savev2_dense_562_kernel_read_readvariableop-
)savev2_dense_562_bias_read_readvariableop<
8savev2_batch_normalization_506_gamma_read_readvariableop;
7savev2_batch_normalization_506_beta_read_readvariableopB
>savev2_batch_normalization_506_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_506_moving_variance_read_readvariableop/
+savev2_dense_563_kernel_read_readvariableop-
)savev2_dense_563_bias_read_readvariableop<
8savev2_batch_normalization_507_gamma_read_readvariableop;
7savev2_batch_normalization_507_beta_read_readvariableopB
>savev2_batch_normalization_507_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_507_moving_variance_read_readvariableop/
+savev2_dense_564_kernel_read_readvariableop-
)savev2_dense_564_bias_read_readvariableop<
8savev2_batch_normalization_508_gamma_read_readvariableop;
7savev2_batch_normalization_508_beta_read_readvariableopB
>savev2_batch_normalization_508_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_508_moving_variance_read_readvariableop/
+savev2_dense_565_kernel_read_readvariableop-
)savev2_dense_565_bias_read_readvariableop<
8savev2_batch_normalization_509_gamma_read_readvariableop;
7savev2_batch_normalization_509_beta_read_readvariableopB
>savev2_batch_normalization_509_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_509_moving_variance_read_readvariableop/
+savev2_dense_566_kernel_read_readvariableop-
)savev2_dense_566_bias_read_readvariableop<
8savev2_batch_normalization_510_gamma_read_readvariableop;
7savev2_batch_normalization_510_beta_read_readvariableopB
>savev2_batch_normalization_510_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_510_moving_variance_read_readvariableop/
+savev2_dense_567_kernel_read_readvariableop-
)savev2_dense_567_bias_read_readvariableop<
8savev2_batch_normalization_511_gamma_read_readvariableop;
7savev2_batch_normalization_511_beta_read_readvariableopB
>savev2_batch_normalization_511_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_511_moving_variance_read_readvariableop/
+savev2_dense_568_kernel_read_readvariableop-
)savev2_dense_568_bias_read_readvariableop<
8savev2_batch_normalization_512_gamma_read_readvariableop;
7savev2_batch_normalization_512_beta_read_readvariableopB
>savev2_batch_normalization_512_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_512_moving_variance_read_readvariableop/
+savev2_dense_569_kernel_read_readvariableop-
)savev2_dense_569_bias_read_readvariableop<
8savev2_batch_normalization_513_gamma_read_readvariableop;
7savev2_batch_normalization_513_beta_read_readvariableopB
>savev2_batch_normalization_513_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_513_moving_variance_read_readvariableop/
+savev2_dense_570_kernel_read_readvariableop-
)savev2_dense_570_bias_read_readvariableop<
8savev2_batch_normalization_514_gamma_read_readvariableop;
7savev2_batch_normalization_514_beta_read_readvariableopB
>savev2_batch_normalization_514_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_514_moving_variance_read_readvariableop/
+savev2_dense_571_kernel_read_readvariableop-
)savev2_dense_571_bias_read_readvariableop<
8savev2_batch_normalization_515_gamma_read_readvariableop;
7savev2_batch_normalization_515_beta_read_readvariableopB
>savev2_batch_normalization_515_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_515_moving_variance_read_readvariableop/
+savev2_dense_572_kernel_read_readvariableop-
)savev2_dense_572_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_560_kernel_m_read_readvariableop4
0savev2_adam_dense_560_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_504_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_504_beta_m_read_readvariableop6
2savev2_adam_dense_561_kernel_m_read_readvariableop4
0savev2_adam_dense_561_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_505_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_505_beta_m_read_readvariableop6
2savev2_adam_dense_562_kernel_m_read_readvariableop4
0savev2_adam_dense_562_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_506_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_506_beta_m_read_readvariableop6
2savev2_adam_dense_563_kernel_m_read_readvariableop4
0savev2_adam_dense_563_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_507_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_507_beta_m_read_readvariableop6
2savev2_adam_dense_564_kernel_m_read_readvariableop4
0savev2_adam_dense_564_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_508_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_508_beta_m_read_readvariableop6
2savev2_adam_dense_565_kernel_m_read_readvariableop4
0savev2_adam_dense_565_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_509_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_509_beta_m_read_readvariableop6
2savev2_adam_dense_566_kernel_m_read_readvariableop4
0savev2_adam_dense_566_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_510_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_510_beta_m_read_readvariableop6
2savev2_adam_dense_567_kernel_m_read_readvariableop4
0savev2_adam_dense_567_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_511_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_511_beta_m_read_readvariableop6
2savev2_adam_dense_568_kernel_m_read_readvariableop4
0savev2_adam_dense_568_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_512_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_512_beta_m_read_readvariableop6
2savev2_adam_dense_569_kernel_m_read_readvariableop4
0savev2_adam_dense_569_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_513_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_513_beta_m_read_readvariableop6
2savev2_adam_dense_570_kernel_m_read_readvariableop4
0savev2_adam_dense_570_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_514_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_514_beta_m_read_readvariableop6
2savev2_adam_dense_571_kernel_m_read_readvariableop4
0savev2_adam_dense_571_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_515_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_515_beta_m_read_readvariableop6
2savev2_adam_dense_572_kernel_m_read_readvariableop4
0savev2_adam_dense_572_bias_m_read_readvariableop6
2savev2_adam_dense_560_kernel_v_read_readvariableop4
0savev2_adam_dense_560_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_504_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_504_beta_v_read_readvariableop6
2savev2_adam_dense_561_kernel_v_read_readvariableop4
0savev2_adam_dense_561_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_505_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_505_beta_v_read_readvariableop6
2savev2_adam_dense_562_kernel_v_read_readvariableop4
0savev2_adam_dense_562_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_506_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_506_beta_v_read_readvariableop6
2savev2_adam_dense_563_kernel_v_read_readvariableop4
0savev2_adam_dense_563_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_507_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_507_beta_v_read_readvariableop6
2savev2_adam_dense_564_kernel_v_read_readvariableop4
0savev2_adam_dense_564_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_508_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_508_beta_v_read_readvariableop6
2savev2_adam_dense_565_kernel_v_read_readvariableop4
0savev2_adam_dense_565_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_509_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_509_beta_v_read_readvariableop6
2savev2_adam_dense_566_kernel_v_read_readvariableop4
0savev2_adam_dense_566_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_510_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_510_beta_v_read_readvariableop6
2savev2_adam_dense_567_kernel_v_read_readvariableop4
0savev2_adam_dense_567_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_511_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_511_beta_v_read_readvariableop6
2savev2_adam_dense_568_kernel_v_read_readvariableop4
0savev2_adam_dense_568_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_512_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_512_beta_v_read_readvariableop6
2savev2_adam_dense_569_kernel_v_read_readvariableop4
0savev2_adam_dense_569_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_513_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_513_beta_v_read_readvariableop6
2savev2_adam_dense_570_kernel_v_read_readvariableop4
0savev2_adam_dense_570_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_514_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_514_beta_v_read_readvariableop6
2savev2_adam_dense_571_kernel_v_read_readvariableop4
0savev2_adam_dense_571_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_515_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_515_beta_v_read_readvariableop6
2savev2_adam_dense_572_kernel_v_read_readvariableop4
0savev2_adam_dense_572_bias_v_read_readvariableop
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
: ®g
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:¸*
dtype0*Öf
valueÌfBÉf¸B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-22/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-22/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-22/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-24/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-24/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-24/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-24/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-24/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHâ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:¸*
dtype0*
valueüBù¸B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¸R
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_560_kernel_read_readvariableop)savev2_dense_560_bias_read_readvariableop8savev2_batch_normalization_504_gamma_read_readvariableop7savev2_batch_normalization_504_beta_read_readvariableop>savev2_batch_normalization_504_moving_mean_read_readvariableopBsavev2_batch_normalization_504_moving_variance_read_readvariableop+savev2_dense_561_kernel_read_readvariableop)savev2_dense_561_bias_read_readvariableop8savev2_batch_normalization_505_gamma_read_readvariableop7savev2_batch_normalization_505_beta_read_readvariableop>savev2_batch_normalization_505_moving_mean_read_readvariableopBsavev2_batch_normalization_505_moving_variance_read_readvariableop+savev2_dense_562_kernel_read_readvariableop)savev2_dense_562_bias_read_readvariableop8savev2_batch_normalization_506_gamma_read_readvariableop7savev2_batch_normalization_506_beta_read_readvariableop>savev2_batch_normalization_506_moving_mean_read_readvariableopBsavev2_batch_normalization_506_moving_variance_read_readvariableop+savev2_dense_563_kernel_read_readvariableop)savev2_dense_563_bias_read_readvariableop8savev2_batch_normalization_507_gamma_read_readvariableop7savev2_batch_normalization_507_beta_read_readvariableop>savev2_batch_normalization_507_moving_mean_read_readvariableopBsavev2_batch_normalization_507_moving_variance_read_readvariableop+savev2_dense_564_kernel_read_readvariableop)savev2_dense_564_bias_read_readvariableop8savev2_batch_normalization_508_gamma_read_readvariableop7savev2_batch_normalization_508_beta_read_readvariableop>savev2_batch_normalization_508_moving_mean_read_readvariableopBsavev2_batch_normalization_508_moving_variance_read_readvariableop+savev2_dense_565_kernel_read_readvariableop)savev2_dense_565_bias_read_readvariableop8savev2_batch_normalization_509_gamma_read_readvariableop7savev2_batch_normalization_509_beta_read_readvariableop>savev2_batch_normalization_509_moving_mean_read_readvariableopBsavev2_batch_normalization_509_moving_variance_read_readvariableop+savev2_dense_566_kernel_read_readvariableop)savev2_dense_566_bias_read_readvariableop8savev2_batch_normalization_510_gamma_read_readvariableop7savev2_batch_normalization_510_beta_read_readvariableop>savev2_batch_normalization_510_moving_mean_read_readvariableopBsavev2_batch_normalization_510_moving_variance_read_readvariableop+savev2_dense_567_kernel_read_readvariableop)savev2_dense_567_bias_read_readvariableop8savev2_batch_normalization_511_gamma_read_readvariableop7savev2_batch_normalization_511_beta_read_readvariableop>savev2_batch_normalization_511_moving_mean_read_readvariableopBsavev2_batch_normalization_511_moving_variance_read_readvariableop+savev2_dense_568_kernel_read_readvariableop)savev2_dense_568_bias_read_readvariableop8savev2_batch_normalization_512_gamma_read_readvariableop7savev2_batch_normalization_512_beta_read_readvariableop>savev2_batch_normalization_512_moving_mean_read_readvariableopBsavev2_batch_normalization_512_moving_variance_read_readvariableop+savev2_dense_569_kernel_read_readvariableop)savev2_dense_569_bias_read_readvariableop8savev2_batch_normalization_513_gamma_read_readvariableop7savev2_batch_normalization_513_beta_read_readvariableop>savev2_batch_normalization_513_moving_mean_read_readvariableopBsavev2_batch_normalization_513_moving_variance_read_readvariableop+savev2_dense_570_kernel_read_readvariableop)savev2_dense_570_bias_read_readvariableop8savev2_batch_normalization_514_gamma_read_readvariableop7savev2_batch_normalization_514_beta_read_readvariableop>savev2_batch_normalization_514_moving_mean_read_readvariableopBsavev2_batch_normalization_514_moving_variance_read_readvariableop+savev2_dense_571_kernel_read_readvariableop)savev2_dense_571_bias_read_readvariableop8savev2_batch_normalization_515_gamma_read_readvariableop7savev2_batch_normalization_515_beta_read_readvariableop>savev2_batch_normalization_515_moving_mean_read_readvariableopBsavev2_batch_normalization_515_moving_variance_read_readvariableop+savev2_dense_572_kernel_read_readvariableop)savev2_dense_572_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_560_kernel_m_read_readvariableop0savev2_adam_dense_560_bias_m_read_readvariableop?savev2_adam_batch_normalization_504_gamma_m_read_readvariableop>savev2_adam_batch_normalization_504_beta_m_read_readvariableop2savev2_adam_dense_561_kernel_m_read_readvariableop0savev2_adam_dense_561_bias_m_read_readvariableop?savev2_adam_batch_normalization_505_gamma_m_read_readvariableop>savev2_adam_batch_normalization_505_beta_m_read_readvariableop2savev2_adam_dense_562_kernel_m_read_readvariableop0savev2_adam_dense_562_bias_m_read_readvariableop?savev2_adam_batch_normalization_506_gamma_m_read_readvariableop>savev2_adam_batch_normalization_506_beta_m_read_readvariableop2savev2_adam_dense_563_kernel_m_read_readvariableop0savev2_adam_dense_563_bias_m_read_readvariableop?savev2_adam_batch_normalization_507_gamma_m_read_readvariableop>savev2_adam_batch_normalization_507_beta_m_read_readvariableop2savev2_adam_dense_564_kernel_m_read_readvariableop0savev2_adam_dense_564_bias_m_read_readvariableop?savev2_adam_batch_normalization_508_gamma_m_read_readvariableop>savev2_adam_batch_normalization_508_beta_m_read_readvariableop2savev2_adam_dense_565_kernel_m_read_readvariableop0savev2_adam_dense_565_bias_m_read_readvariableop?savev2_adam_batch_normalization_509_gamma_m_read_readvariableop>savev2_adam_batch_normalization_509_beta_m_read_readvariableop2savev2_adam_dense_566_kernel_m_read_readvariableop0savev2_adam_dense_566_bias_m_read_readvariableop?savev2_adam_batch_normalization_510_gamma_m_read_readvariableop>savev2_adam_batch_normalization_510_beta_m_read_readvariableop2savev2_adam_dense_567_kernel_m_read_readvariableop0savev2_adam_dense_567_bias_m_read_readvariableop?savev2_adam_batch_normalization_511_gamma_m_read_readvariableop>savev2_adam_batch_normalization_511_beta_m_read_readvariableop2savev2_adam_dense_568_kernel_m_read_readvariableop0savev2_adam_dense_568_bias_m_read_readvariableop?savev2_adam_batch_normalization_512_gamma_m_read_readvariableop>savev2_adam_batch_normalization_512_beta_m_read_readvariableop2savev2_adam_dense_569_kernel_m_read_readvariableop0savev2_adam_dense_569_bias_m_read_readvariableop?savev2_adam_batch_normalization_513_gamma_m_read_readvariableop>savev2_adam_batch_normalization_513_beta_m_read_readvariableop2savev2_adam_dense_570_kernel_m_read_readvariableop0savev2_adam_dense_570_bias_m_read_readvariableop?savev2_adam_batch_normalization_514_gamma_m_read_readvariableop>savev2_adam_batch_normalization_514_beta_m_read_readvariableop2savev2_adam_dense_571_kernel_m_read_readvariableop0savev2_adam_dense_571_bias_m_read_readvariableop?savev2_adam_batch_normalization_515_gamma_m_read_readvariableop>savev2_adam_batch_normalization_515_beta_m_read_readvariableop2savev2_adam_dense_572_kernel_m_read_readvariableop0savev2_adam_dense_572_bias_m_read_readvariableop2savev2_adam_dense_560_kernel_v_read_readvariableop0savev2_adam_dense_560_bias_v_read_readvariableop?savev2_adam_batch_normalization_504_gamma_v_read_readvariableop>savev2_adam_batch_normalization_504_beta_v_read_readvariableop2savev2_adam_dense_561_kernel_v_read_readvariableop0savev2_adam_dense_561_bias_v_read_readvariableop?savev2_adam_batch_normalization_505_gamma_v_read_readvariableop>savev2_adam_batch_normalization_505_beta_v_read_readvariableop2savev2_adam_dense_562_kernel_v_read_readvariableop0savev2_adam_dense_562_bias_v_read_readvariableop?savev2_adam_batch_normalization_506_gamma_v_read_readvariableop>savev2_adam_batch_normalization_506_beta_v_read_readvariableop2savev2_adam_dense_563_kernel_v_read_readvariableop0savev2_adam_dense_563_bias_v_read_readvariableop?savev2_adam_batch_normalization_507_gamma_v_read_readvariableop>savev2_adam_batch_normalization_507_beta_v_read_readvariableop2savev2_adam_dense_564_kernel_v_read_readvariableop0savev2_adam_dense_564_bias_v_read_readvariableop?savev2_adam_batch_normalization_508_gamma_v_read_readvariableop>savev2_adam_batch_normalization_508_beta_v_read_readvariableop2savev2_adam_dense_565_kernel_v_read_readvariableop0savev2_adam_dense_565_bias_v_read_readvariableop?savev2_adam_batch_normalization_509_gamma_v_read_readvariableop>savev2_adam_batch_normalization_509_beta_v_read_readvariableop2savev2_adam_dense_566_kernel_v_read_readvariableop0savev2_adam_dense_566_bias_v_read_readvariableop?savev2_adam_batch_normalization_510_gamma_v_read_readvariableop>savev2_adam_batch_normalization_510_beta_v_read_readvariableop2savev2_adam_dense_567_kernel_v_read_readvariableop0savev2_adam_dense_567_bias_v_read_readvariableop?savev2_adam_batch_normalization_511_gamma_v_read_readvariableop>savev2_adam_batch_normalization_511_beta_v_read_readvariableop2savev2_adam_dense_568_kernel_v_read_readvariableop0savev2_adam_dense_568_bias_v_read_readvariableop?savev2_adam_batch_normalization_512_gamma_v_read_readvariableop>savev2_adam_batch_normalization_512_beta_v_read_readvariableop2savev2_adam_dense_569_kernel_v_read_readvariableop0savev2_adam_dense_569_bias_v_read_readvariableop?savev2_adam_batch_normalization_513_gamma_v_read_readvariableop>savev2_adam_batch_normalization_513_beta_v_read_readvariableop2savev2_adam_dense_570_kernel_v_read_readvariableop0savev2_adam_dense_570_bias_v_read_readvariableop?savev2_adam_batch_normalization_514_gamma_v_read_readvariableop>savev2_adam_batch_normalization_514_beta_v_read_readvariableop2savev2_adam_dense_571_kernel_v_read_readvariableop0savev2_adam_dense_571_bias_v_read_readvariableop?savev2_adam_batch_normalization_515_gamma_v_read_readvariableop>savev2_adam_batch_normalization_515_beta_v_read_readvariableop2savev2_adam_dense_572_kernel_v_read_readvariableop0savev2_adam_dense_572_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *É
dtypes¾
»2¸		
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

identity_1Identity_1:output:0*ã	
_input_shapesÑ	
Î	: ::: :A:A:A:A:A:A:AA:A:A:A:A:A:AA:A:A:A:A:A:AA:A:A:A:A:A:A::::::::::::::::::5:5:5:5:5:5:55:5:5:5:5:5:55:5:5:5:5:5:55:5:5:5:5:5:55:5:5:5:5:5:5:: : : : : : :A:A:A:A:AA:A:A:A:AA:A:A:A:AA:A:A:A:A::::::::::::5:5:5:5:55:5:5:5:55:5:5:5:55:5:5:5:55:5:5:5:5::A:A:A:A:AA:A:A:A:AA:A:A:A:AA:A:A:A:A::::::::::::5:5:5:5:55:5:5:5:55:5:5:5:55:5:5:5:55:5:5:5:5:: 2(
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

:A: 

_output_shapes
:A: 

_output_shapes
:A: 

_output_shapes
:A: 

_output_shapes
:A: 	

_output_shapes
:A:$
 

_output_shapes

:AA: 

_output_shapes
:A: 

_output_shapes
:A: 

_output_shapes
:A: 

_output_shapes
:A: 

_output_shapes
:A:$ 

_output_shapes

:AA: 

_output_shapes
:A: 

_output_shapes
:A: 

_output_shapes
:A: 

_output_shapes
:A: 

_output_shapes
:A:$ 

_output_shapes

:AA: 

_output_shapes
:A: 

_output_shapes
:A: 

_output_shapes
:A: 

_output_shapes
:A: 

_output_shapes
:A:$ 

_output_shapes

:A: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::$. 

_output_shapes

:5: /

_output_shapes
:5: 0

_output_shapes
:5: 1

_output_shapes
:5: 2

_output_shapes
:5: 3

_output_shapes
:5:$4 

_output_shapes

:55: 5

_output_shapes
:5: 6

_output_shapes
:5: 7

_output_shapes
:5: 8

_output_shapes
:5: 9

_output_shapes
:5:$: 

_output_shapes

:55: ;

_output_shapes
:5: <

_output_shapes
:5: =

_output_shapes
:5: >

_output_shapes
:5: ?

_output_shapes
:5:$@ 

_output_shapes

:55: A

_output_shapes
:5: B

_output_shapes
:5: C

_output_shapes
:5: D

_output_shapes
:5: E

_output_shapes
:5:$F 

_output_shapes

:55: G

_output_shapes
:5: H

_output_shapes
:5: I

_output_shapes
:5: J

_output_shapes
:5: K

_output_shapes
:5:$L 

_output_shapes

:5: M

_output_shapes
::N

_output_shapes
: :O

_output_shapes
: :P

_output_shapes
: :Q

_output_shapes
: :R

_output_shapes
: :S

_output_shapes
: :$T 

_output_shapes

:A: U

_output_shapes
:A: V

_output_shapes
:A: W

_output_shapes
:A:$X 

_output_shapes

:AA: Y

_output_shapes
:A: Z

_output_shapes
:A: [

_output_shapes
:A:$\ 

_output_shapes

:AA: ]

_output_shapes
:A: ^

_output_shapes
:A: _

_output_shapes
:A:$` 

_output_shapes

:AA: a

_output_shapes
:A: b

_output_shapes
:A: c

_output_shapes
:A:$d 

_output_shapes

:A: e

_output_shapes
:: f

_output_shapes
:: g

_output_shapes
::$h 

_output_shapes

:: i

_output_shapes
:: j

_output_shapes
:: k

_output_shapes
::$l 

_output_shapes

:: m

_output_shapes
:: n

_output_shapes
:: o

_output_shapes
::$p 

_output_shapes

:5: q

_output_shapes
:5: r

_output_shapes
:5: s

_output_shapes
:5:$t 

_output_shapes

:55: u

_output_shapes
:5: v

_output_shapes
:5: w

_output_shapes
:5:$x 

_output_shapes

:55: y

_output_shapes
:5: z

_output_shapes
:5: {

_output_shapes
:5:$| 

_output_shapes

:55: }

_output_shapes
:5: ~

_output_shapes
:5: 

_output_shapes
:5:% 

_output_shapes

:55:!

_output_shapes
:5:!

_output_shapes
:5:!

_output_shapes
:5:% 

_output_shapes

:5:!

_output_shapes
::% 

_output_shapes

:A:!

_output_shapes
:A:!

_output_shapes
:A:!

_output_shapes
:A:% 

_output_shapes

:AA:!

_output_shapes
:A:!

_output_shapes
:A:!

_output_shapes
:A:% 

_output_shapes

:AA:!

_output_shapes
:A:!

_output_shapes
:A:!

_output_shapes
:A:% 

_output_shapes

:AA:!

_output_shapes
:A:!

_output_shapes
:A:!

_output_shapes
:A:% 

_output_shapes

:A:!

_output_shapes
::!

_output_shapes
::!

_output_shapes
::% 

_output_shapes

::!

_output_shapes
::!

_output_shapes
::!

_output_shapes
::% 

_output_shapes

::!

_output_shapes
::! 

_output_shapes
::!¡

_output_shapes
::%¢ 

_output_shapes

:5:!£

_output_shapes
:5:!¤

_output_shapes
:5:!¥

_output_shapes
:5:%¦ 

_output_shapes

:55:!§

_output_shapes
:5:!¨

_output_shapes
:5:!©

_output_shapes
:5:%ª 

_output_shapes

:55:!«

_output_shapes
:5:!¬

_output_shapes
:5:!­

_output_shapes
:5:%® 

_output_shapes

:55:!¯

_output_shapes
:5:!°

_output_shapes
:5:!±

_output_shapes
:5:%² 

_output_shapes

:55:!³

_output_shapes
:5:!´

_output_shapes
:5:!µ

_output_shapes
:5:%¶ 

_output_shapes

:5:!·

_output_shapes
::¸

_output_shapes
: 
æ
h
L__inference_leaky_re_lu_509_layer_call_and_return_conditional_losses_1013626

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_509_layer_call_and_return_conditional_losses_1010067

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1009032

inputs5
'assignmovingavg_readvariableop_resource:A7
)assignmovingavg_1_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A/
!batchnorm_readvariableop_resource:A
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:A
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:A*
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
:A*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A¬
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
:A*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:A~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A´
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_1010003

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿA:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_508_layer_call_fn_1013440

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1009231o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1013909

inputs/
!batchnorm_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:51
#batchnorm_readvariableop_1_resource:51
#batchnorm_readvariableop_2_resource:5
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:5z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_511_layer_call_fn_1013839

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_511_layer_call_and_return_conditional_losses_1010131`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_1013299

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿA:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_509_layer_call_fn_1013549

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1009313o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_504_layer_call_fn_1013017

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1008950o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_514_layer_call_fn_1014094

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1009723o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1014127

inputs/
!batchnorm_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:51
#batchnorm_readvariableop_1_resource:51
#batchnorm_readvariableop_2_resource:5
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:5z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_1009907

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿA:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
É	
÷
F__inference_dense_563_layer_call_and_return_conditional_losses_1009983

inputs0
matmul_readvariableop_resource:AA-
biasadd_readvariableop_resource:A
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:AA*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1013255

inputs/
!batchnorm_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A1
#batchnorm_readvariableop_1_resource:A1
#batchnorm_readvariableop_2_resource:A
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
Ù 
³
/__inference_sequential_56_layer_call_fn_1011855

inputs
unknown
	unknown_0
	unknown_1:A
	unknown_2:A
	unknown_3:A
	unknown_4:A
	unknown_5:A
	unknown_6:A
	unknown_7:AA
	unknown_8:A
	unknown_9:A

unknown_10:A

unknown_11:A

unknown_12:A

unknown_13:AA

unknown_14:A

unknown_15:A

unknown_16:A

unknown_17:A

unknown_18:A

unknown_19:AA

unknown_20:A

unknown_21:A

unknown_22:A

unknown_23:A

unknown_24:A

unknown_25:A

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:5

unknown_44:5

unknown_45:5

unknown_46:5

unknown_47:5

unknown_48:5

unknown_49:55

unknown_50:5

unknown_51:5

unknown_52:5

unknown_53:5

unknown_54:5

unknown_55:55

unknown_56:5

unknown_57:5

unknown_58:5

unknown_59:5

unknown_60:5

unknown_61:55

unknown_62:5

unknown_63:5

unknown_64:5

unknown_65:5

unknown_66:5

unknown_67:55

unknown_68:5

unknown_69:5

unknown_70:5

unknown_71:5

unknown_72:5

unknown_73:5

unknown_74:
identity¢StatefulPartitionedCallà

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
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74*X
TinQ
O2M*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*l
_read_only_resource_inputsN
LJ	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKL*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_56_layer_call_and_return_conditional_losses_1010278o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ð
_input_shapes¾
»:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
­
M
1__inference_leaky_re_lu_505_layer_call_fn_1013185

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_1009939`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿA:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
É	
÷
F__inference_dense_563_layer_call_and_return_conditional_losses_1013318

inputs0
matmul_readvariableop_resource:AA-
biasadd_readvariableop_resource:A
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:AA*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_510_layer_call_fn_1013658

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1009395o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1009360

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_dense_566_layer_call_fn_1013635

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_566_layer_call_and_return_conditional_losses_1010079o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_510_layer_call_fn_1013671

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1009442o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_509_layer_call_fn_1013621

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_509_layer_call_and_return_conditional_losses_1010067`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
¬"
J__inference_sequential_56_layer_call_and_return_conditional_losses_1010278

inputs
normalization_56_sub_y
normalization_56_sqrt_x#
dense_560_1009888:A
dense_560_1009890:A-
batch_normalization_504_1009893:A-
batch_normalization_504_1009895:A-
batch_normalization_504_1009897:A-
batch_normalization_504_1009899:A#
dense_561_1009920:AA
dense_561_1009922:A-
batch_normalization_505_1009925:A-
batch_normalization_505_1009927:A-
batch_normalization_505_1009929:A-
batch_normalization_505_1009931:A#
dense_562_1009952:AA
dense_562_1009954:A-
batch_normalization_506_1009957:A-
batch_normalization_506_1009959:A-
batch_normalization_506_1009961:A-
batch_normalization_506_1009963:A#
dense_563_1009984:AA
dense_563_1009986:A-
batch_normalization_507_1009989:A-
batch_normalization_507_1009991:A-
batch_normalization_507_1009993:A-
batch_normalization_507_1009995:A#
dense_564_1010016:A
dense_564_1010018:-
batch_normalization_508_1010021:-
batch_normalization_508_1010023:-
batch_normalization_508_1010025:-
batch_normalization_508_1010027:#
dense_565_1010048:
dense_565_1010050:-
batch_normalization_509_1010053:-
batch_normalization_509_1010055:-
batch_normalization_509_1010057:-
batch_normalization_509_1010059:#
dense_566_1010080:
dense_566_1010082:-
batch_normalization_510_1010085:-
batch_normalization_510_1010087:-
batch_normalization_510_1010089:-
batch_normalization_510_1010091:#
dense_567_1010112:5
dense_567_1010114:5-
batch_normalization_511_1010117:5-
batch_normalization_511_1010119:5-
batch_normalization_511_1010121:5-
batch_normalization_511_1010123:5#
dense_568_1010144:55
dense_568_1010146:5-
batch_normalization_512_1010149:5-
batch_normalization_512_1010151:5-
batch_normalization_512_1010153:5-
batch_normalization_512_1010155:5#
dense_569_1010176:55
dense_569_1010178:5-
batch_normalization_513_1010181:5-
batch_normalization_513_1010183:5-
batch_normalization_513_1010185:5-
batch_normalization_513_1010187:5#
dense_570_1010208:55
dense_570_1010210:5-
batch_normalization_514_1010213:5-
batch_normalization_514_1010215:5-
batch_normalization_514_1010217:5-
batch_normalization_514_1010219:5#
dense_571_1010240:55
dense_571_1010242:5-
batch_normalization_515_1010245:5-
batch_normalization_515_1010247:5-
batch_normalization_515_1010249:5-
batch_normalization_515_1010251:5#
dense_572_1010272:5
dense_572_1010274:
identity¢/batch_normalization_504/StatefulPartitionedCall¢/batch_normalization_505/StatefulPartitionedCall¢/batch_normalization_506/StatefulPartitionedCall¢/batch_normalization_507/StatefulPartitionedCall¢/batch_normalization_508/StatefulPartitionedCall¢/batch_normalization_509/StatefulPartitionedCall¢/batch_normalization_510/StatefulPartitionedCall¢/batch_normalization_511/StatefulPartitionedCall¢/batch_normalization_512/StatefulPartitionedCall¢/batch_normalization_513/StatefulPartitionedCall¢/batch_normalization_514/StatefulPartitionedCall¢/batch_normalization_515/StatefulPartitionedCall¢!dense_560/StatefulPartitionedCall¢!dense_561/StatefulPartitionedCall¢!dense_562/StatefulPartitionedCall¢!dense_563/StatefulPartitionedCall¢!dense_564/StatefulPartitionedCall¢!dense_565/StatefulPartitionedCall¢!dense_566/StatefulPartitionedCall¢!dense_567/StatefulPartitionedCall¢!dense_568/StatefulPartitionedCall¢!dense_569/StatefulPartitionedCall¢!dense_570/StatefulPartitionedCall¢!dense_571/StatefulPartitionedCall¢!dense_572/StatefulPartitionedCallm
normalization_56/subSubinputsnormalization_56_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_56/SqrtSqrtnormalization_56_sqrt_x*
T0*
_output_shapes

:_
normalization_56/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_56/MaximumMaximumnormalization_56/Sqrt:y:0#normalization_56/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_56/truedivRealDivnormalization_56/sub:z:0normalization_56/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_560/StatefulPartitionedCallStatefulPartitionedCallnormalization_56/truediv:z:0dense_560_1009888dense_560_1009890*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_560_layer_call_and_return_conditional_losses_1009887
/batch_normalization_504/StatefulPartitionedCallStatefulPartitionedCall*dense_560/StatefulPartitionedCall:output:0batch_normalization_504_1009893batch_normalization_504_1009895batch_normalization_504_1009897batch_normalization_504_1009899*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1008903ù
leaky_re_lu_504/PartitionedCallPartitionedCall8batch_normalization_504/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_1009907
!dense_561/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_504/PartitionedCall:output:0dense_561_1009920dense_561_1009922*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_561_layer_call_and_return_conditional_losses_1009919
/batch_normalization_505/StatefulPartitionedCallStatefulPartitionedCall*dense_561/StatefulPartitionedCall:output:0batch_normalization_505_1009925batch_normalization_505_1009927batch_normalization_505_1009929batch_normalization_505_1009931*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1008985ù
leaky_re_lu_505/PartitionedCallPartitionedCall8batch_normalization_505/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_1009939
!dense_562/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_505/PartitionedCall:output:0dense_562_1009952dense_562_1009954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_562_layer_call_and_return_conditional_losses_1009951
/batch_normalization_506/StatefulPartitionedCallStatefulPartitionedCall*dense_562/StatefulPartitionedCall:output:0batch_normalization_506_1009957batch_normalization_506_1009959batch_normalization_506_1009961batch_normalization_506_1009963*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1009067ù
leaky_re_lu_506/PartitionedCallPartitionedCall8batch_normalization_506/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_1009971
!dense_563/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_506/PartitionedCall:output:0dense_563_1009984dense_563_1009986*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_563_layer_call_and_return_conditional_losses_1009983
/batch_normalization_507/StatefulPartitionedCallStatefulPartitionedCall*dense_563/StatefulPartitionedCall:output:0batch_normalization_507_1009989batch_normalization_507_1009991batch_normalization_507_1009993batch_normalization_507_1009995*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1009149ù
leaky_re_lu_507/PartitionedCallPartitionedCall8batch_normalization_507/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_1010003
!dense_564/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_507/PartitionedCall:output:0dense_564_1010016dense_564_1010018*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_564_layer_call_and_return_conditional_losses_1010015
/batch_normalization_508/StatefulPartitionedCallStatefulPartitionedCall*dense_564/StatefulPartitionedCall:output:0batch_normalization_508_1010021batch_normalization_508_1010023batch_normalization_508_1010025batch_normalization_508_1010027*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1009231ù
leaky_re_lu_508/PartitionedCallPartitionedCall8batch_normalization_508/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_508_layer_call_and_return_conditional_losses_1010035
!dense_565/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_508/PartitionedCall:output:0dense_565_1010048dense_565_1010050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_565_layer_call_and_return_conditional_losses_1010047
/batch_normalization_509/StatefulPartitionedCallStatefulPartitionedCall*dense_565/StatefulPartitionedCall:output:0batch_normalization_509_1010053batch_normalization_509_1010055batch_normalization_509_1010057batch_normalization_509_1010059*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1009313ù
leaky_re_lu_509/PartitionedCallPartitionedCall8batch_normalization_509/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_509_layer_call_and_return_conditional_losses_1010067
!dense_566/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_509/PartitionedCall:output:0dense_566_1010080dense_566_1010082*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_566_layer_call_and_return_conditional_losses_1010079
/batch_normalization_510/StatefulPartitionedCallStatefulPartitionedCall*dense_566/StatefulPartitionedCall:output:0batch_normalization_510_1010085batch_normalization_510_1010087batch_normalization_510_1010089batch_normalization_510_1010091*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1009395ù
leaky_re_lu_510/PartitionedCallPartitionedCall8batch_normalization_510/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_510_layer_call_and_return_conditional_losses_1010099
!dense_567/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_510/PartitionedCall:output:0dense_567_1010112dense_567_1010114*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_567_layer_call_and_return_conditional_losses_1010111
/batch_normalization_511/StatefulPartitionedCallStatefulPartitionedCall*dense_567/StatefulPartitionedCall:output:0batch_normalization_511_1010117batch_normalization_511_1010119batch_normalization_511_1010121batch_normalization_511_1010123*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1009477ù
leaky_re_lu_511/PartitionedCallPartitionedCall8batch_normalization_511/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_511_layer_call_and_return_conditional_losses_1010131
!dense_568/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_511/PartitionedCall:output:0dense_568_1010144dense_568_1010146*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_568_layer_call_and_return_conditional_losses_1010143
/batch_normalization_512/StatefulPartitionedCallStatefulPartitionedCall*dense_568/StatefulPartitionedCall:output:0batch_normalization_512_1010149batch_normalization_512_1010151batch_normalization_512_1010153batch_normalization_512_1010155*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1009559ù
leaky_re_lu_512/PartitionedCallPartitionedCall8batch_normalization_512/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_512_layer_call_and_return_conditional_losses_1010163
!dense_569/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_512/PartitionedCall:output:0dense_569_1010176dense_569_1010178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_569_layer_call_and_return_conditional_losses_1010175
/batch_normalization_513/StatefulPartitionedCallStatefulPartitionedCall*dense_569/StatefulPartitionedCall:output:0batch_normalization_513_1010181batch_normalization_513_1010183batch_normalization_513_1010185batch_normalization_513_1010187*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1009641ù
leaky_re_lu_513/PartitionedCallPartitionedCall8batch_normalization_513/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_513_layer_call_and_return_conditional_losses_1010195
!dense_570/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_513/PartitionedCall:output:0dense_570_1010208dense_570_1010210*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_570_layer_call_and_return_conditional_losses_1010207
/batch_normalization_514/StatefulPartitionedCallStatefulPartitionedCall*dense_570/StatefulPartitionedCall:output:0batch_normalization_514_1010213batch_normalization_514_1010215batch_normalization_514_1010217batch_normalization_514_1010219*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1009723ù
leaky_re_lu_514/PartitionedCallPartitionedCall8batch_normalization_514/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_514_layer_call_and_return_conditional_losses_1010227
!dense_571/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_514/PartitionedCall:output:0dense_571_1010240dense_571_1010242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_571_layer_call_and_return_conditional_losses_1010239
/batch_normalization_515/StatefulPartitionedCallStatefulPartitionedCall*dense_571/StatefulPartitionedCall:output:0batch_normalization_515_1010245batch_normalization_515_1010247batch_normalization_515_1010249batch_normalization_515_1010251*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1009805ù
leaky_re_lu_515/PartitionedCallPartitionedCall8batch_normalization_515/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_515_layer_call_and_return_conditional_losses_1010259
!dense_572/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_515/PartitionedCall:output:0dense_572_1010272dense_572_1010274*
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
GPU 2J 8 *O
fJRH
F__inference_dense_572_layer_call_and_return_conditional_losses_1010271y
IdentityIdentity*dense_572/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
NoOpNoOp0^batch_normalization_504/StatefulPartitionedCall0^batch_normalization_505/StatefulPartitionedCall0^batch_normalization_506/StatefulPartitionedCall0^batch_normalization_507/StatefulPartitionedCall0^batch_normalization_508/StatefulPartitionedCall0^batch_normalization_509/StatefulPartitionedCall0^batch_normalization_510/StatefulPartitionedCall0^batch_normalization_511/StatefulPartitionedCall0^batch_normalization_512/StatefulPartitionedCall0^batch_normalization_513/StatefulPartitionedCall0^batch_normalization_514/StatefulPartitionedCall0^batch_normalization_515/StatefulPartitionedCall"^dense_560/StatefulPartitionedCall"^dense_561/StatefulPartitionedCall"^dense_562/StatefulPartitionedCall"^dense_563/StatefulPartitionedCall"^dense_564/StatefulPartitionedCall"^dense_565/StatefulPartitionedCall"^dense_566/StatefulPartitionedCall"^dense_567/StatefulPartitionedCall"^dense_568/StatefulPartitionedCall"^dense_569/StatefulPartitionedCall"^dense_570/StatefulPartitionedCall"^dense_571/StatefulPartitionedCall"^dense_572/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ð
_input_shapes¾
»:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_504/StatefulPartitionedCall/batch_normalization_504/StatefulPartitionedCall2b
/batch_normalization_505/StatefulPartitionedCall/batch_normalization_505/StatefulPartitionedCall2b
/batch_normalization_506/StatefulPartitionedCall/batch_normalization_506/StatefulPartitionedCall2b
/batch_normalization_507/StatefulPartitionedCall/batch_normalization_507/StatefulPartitionedCall2b
/batch_normalization_508/StatefulPartitionedCall/batch_normalization_508/StatefulPartitionedCall2b
/batch_normalization_509/StatefulPartitionedCall/batch_normalization_509/StatefulPartitionedCall2b
/batch_normalization_510/StatefulPartitionedCall/batch_normalization_510/StatefulPartitionedCall2b
/batch_normalization_511/StatefulPartitionedCall/batch_normalization_511/StatefulPartitionedCall2b
/batch_normalization_512/StatefulPartitionedCall/batch_normalization_512/StatefulPartitionedCall2b
/batch_normalization_513/StatefulPartitionedCall/batch_normalization_513/StatefulPartitionedCall2b
/batch_normalization_514/StatefulPartitionedCall/batch_normalization_514/StatefulPartitionedCall2b
/batch_normalization_515/StatefulPartitionedCall/batch_normalization_515/StatefulPartitionedCall2F
!dense_560/StatefulPartitionedCall!dense_560/StatefulPartitionedCall2F
!dense_561/StatefulPartitionedCall!dense_561/StatefulPartitionedCall2F
!dense_562/StatefulPartitionedCall!dense_562/StatefulPartitionedCall2F
!dense_563/StatefulPartitionedCall!dense_563/StatefulPartitionedCall2F
!dense_564/StatefulPartitionedCall!dense_564/StatefulPartitionedCall2F
!dense_565/StatefulPartitionedCall!dense_565/StatefulPartitionedCall2F
!dense_566/StatefulPartitionedCall!dense_566/StatefulPartitionedCall2F
!dense_567/StatefulPartitionedCall!dense_567/StatefulPartitionedCall2F
!dense_568/StatefulPartitionedCall!dense_568/StatefulPartitionedCall2F
!dense_569/StatefulPartitionedCall!dense_569/StatefulPartitionedCall2F
!dense_570/StatefulPartitionedCall!dense_570/StatefulPartitionedCall2F
!dense_571/StatefulPartitionedCall!dense_571/StatefulPartitionedCall2F
!dense_572/StatefulPartitionedCall!dense_572/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
æ
h
L__inference_leaky_re_lu_511_layer_call_and_return_conditional_losses_1010131

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1013943

inputs5
'assignmovingavg_readvariableop_resource:57
)assignmovingavg_1_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:5/
!batchnorm_readvariableop_resource:5
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:5
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:5*
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
:5*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:5x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5¬
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
:5*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:5~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5´
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:5v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1013398

inputs5
'assignmovingavg_readvariableop_resource:A7
)assignmovingavg_1_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A/
!batchnorm_readvariableop_resource:A
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:A
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:A*
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
:A*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A¬
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
:A*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:A~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A´
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1014018

inputs/
!batchnorm_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:51
#batchnorm_readvariableop_1_resource:51
#batchnorm_readvariableop_2_resource:5
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:5z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
òë
ß{
#__inference__traced_restore_1015432
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_560_kernel:A/
!assignvariableop_4_dense_560_bias:A>
0assignvariableop_5_batch_normalization_504_gamma:A=
/assignvariableop_6_batch_normalization_504_beta:AD
6assignvariableop_7_batch_normalization_504_moving_mean:AH
:assignvariableop_8_batch_normalization_504_moving_variance:A5
#assignvariableop_9_dense_561_kernel:AA0
"assignvariableop_10_dense_561_bias:A?
1assignvariableop_11_batch_normalization_505_gamma:A>
0assignvariableop_12_batch_normalization_505_beta:AE
7assignvariableop_13_batch_normalization_505_moving_mean:AI
;assignvariableop_14_batch_normalization_505_moving_variance:A6
$assignvariableop_15_dense_562_kernel:AA0
"assignvariableop_16_dense_562_bias:A?
1assignvariableop_17_batch_normalization_506_gamma:A>
0assignvariableop_18_batch_normalization_506_beta:AE
7assignvariableop_19_batch_normalization_506_moving_mean:AI
;assignvariableop_20_batch_normalization_506_moving_variance:A6
$assignvariableop_21_dense_563_kernel:AA0
"assignvariableop_22_dense_563_bias:A?
1assignvariableop_23_batch_normalization_507_gamma:A>
0assignvariableop_24_batch_normalization_507_beta:AE
7assignvariableop_25_batch_normalization_507_moving_mean:AI
;assignvariableop_26_batch_normalization_507_moving_variance:A6
$assignvariableop_27_dense_564_kernel:A0
"assignvariableop_28_dense_564_bias:?
1assignvariableop_29_batch_normalization_508_gamma:>
0assignvariableop_30_batch_normalization_508_beta:E
7assignvariableop_31_batch_normalization_508_moving_mean:I
;assignvariableop_32_batch_normalization_508_moving_variance:6
$assignvariableop_33_dense_565_kernel:0
"assignvariableop_34_dense_565_bias:?
1assignvariableop_35_batch_normalization_509_gamma:>
0assignvariableop_36_batch_normalization_509_beta:E
7assignvariableop_37_batch_normalization_509_moving_mean:I
;assignvariableop_38_batch_normalization_509_moving_variance:6
$assignvariableop_39_dense_566_kernel:0
"assignvariableop_40_dense_566_bias:?
1assignvariableop_41_batch_normalization_510_gamma:>
0assignvariableop_42_batch_normalization_510_beta:E
7assignvariableop_43_batch_normalization_510_moving_mean:I
;assignvariableop_44_batch_normalization_510_moving_variance:6
$assignvariableop_45_dense_567_kernel:50
"assignvariableop_46_dense_567_bias:5?
1assignvariableop_47_batch_normalization_511_gamma:5>
0assignvariableop_48_batch_normalization_511_beta:5E
7assignvariableop_49_batch_normalization_511_moving_mean:5I
;assignvariableop_50_batch_normalization_511_moving_variance:56
$assignvariableop_51_dense_568_kernel:550
"assignvariableop_52_dense_568_bias:5?
1assignvariableop_53_batch_normalization_512_gamma:5>
0assignvariableop_54_batch_normalization_512_beta:5E
7assignvariableop_55_batch_normalization_512_moving_mean:5I
;assignvariableop_56_batch_normalization_512_moving_variance:56
$assignvariableop_57_dense_569_kernel:550
"assignvariableop_58_dense_569_bias:5?
1assignvariableop_59_batch_normalization_513_gamma:5>
0assignvariableop_60_batch_normalization_513_beta:5E
7assignvariableop_61_batch_normalization_513_moving_mean:5I
;assignvariableop_62_batch_normalization_513_moving_variance:56
$assignvariableop_63_dense_570_kernel:550
"assignvariableop_64_dense_570_bias:5?
1assignvariableop_65_batch_normalization_514_gamma:5>
0assignvariableop_66_batch_normalization_514_beta:5E
7assignvariableop_67_batch_normalization_514_moving_mean:5I
;assignvariableop_68_batch_normalization_514_moving_variance:56
$assignvariableop_69_dense_571_kernel:550
"assignvariableop_70_dense_571_bias:5?
1assignvariableop_71_batch_normalization_515_gamma:5>
0assignvariableop_72_batch_normalization_515_beta:5E
7assignvariableop_73_batch_normalization_515_moving_mean:5I
;assignvariableop_74_batch_normalization_515_moving_variance:56
$assignvariableop_75_dense_572_kernel:50
"assignvariableop_76_dense_572_bias:'
assignvariableop_77_adam_iter:	 )
assignvariableop_78_adam_beta_1: )
assignvariableop_79_adam_beta_2: (
assignvariableop_80_adam_decay: #
assignvariableop_81_total: %
assignvariableop_82_count_1: =
+assignvariableop_83_adam_dense_560_kernel_m:A7
)assignvariableop_84_adam_dense_560_bias_m:AF
8assignvariableop_85_adam_batch_normalization_504_gamma_m:AE
7assignvariableop_86_adam_batch_normalization_504_beta_m:A=
+assignvariableop_87_adam_dense_561_kernel_m:AA7
)assignvariableop_88_adam_dense_561_bias_m:AF
8assignvariableop_89_adam_batch_normalization_505_gamma_m:AE
7assignvariableop_90_adam_batch_normalization_505_beta_m:A=
+assignvariableop_91_adam_dense_562_kernel_m:AA7
)assignvariableop_92_adam_dense_562_bias_m:AF
8assignvariableop_93_adam_batch_normalization_506_gamma_m:AE
7assignvariableop_94_adam_batch_normalization_506_beta_m:A=
+assignvariableop_95_adam_dense_563_kernel_m:AA7
)assignvariableop_96_adam_dense_563_bias_m:AF
8assignvariableop_97_adam_batch_normalization_507_gamma_m:AE
7assignvariableop_98_adam_batch_normalization_507_beta_m:A=
+assignvariableop_99_adam_dense_564_kernel_m:A8
*assignvariableop_100_adam_dense_564_bias_m:G
9assignvariableop_101_adam_batch_normalization_508_gamma_m:F
8assignvariableop_102_adam_batch_normalization_508_beta_m:>
,assignvariableop_103_adam_dense_565_kernel_m:8
*assignvariableop_104_adam_dense_565_bias_m:G
9assignvariableop_105_adam_batch_normalization_509_gamma_m:F
8assignvariableop_106_adam_batch_normalization_509_beta_m:>
,assignvariableop_107_adam_dense_566_kernel_m:8
*assignvariableop_108_adam_dense_566_bias_m:G
9assignvariableop_109_adam_batch_normalization_510_gamma_m:F
8assignvariableop_110_adam_batch_normalization_510_beta_m:>
,assignvariableop_111_adam_dense_567_kernel_m:58
*assignvariableop_112_adam_dense_567_bias_m:5G
9assignvariableop_113_adam_batch_normalization_511_gamma_m:5F
8assignvariableop_114_adam_batch_normalization_511_beta_m:5>
,assignvariableop_115_adam_dense_568_kernel_m:558
*assignvariableop_116_adam_dense_568_bias_m:5G
9assignvariableop_117_adam_batch_normalization_512_gamma_m:5F
8assignvariableop_118_adam_batch_normalization_512_beta_m:5>
,assignvariableop_119_adam_dense_569_kernel_m:558
*assignvariableop_120_adam_dense_569_bias_m:5G
9assignvariableop_121_adam_batch_normalization_513_gamma_m:5F
8assignvariableop_122_adam_batch_normalization_513_beta_m:5>
,assignvariableop_123_adam_dense_570_kernel_m:558
*assignvariableop_124_adam_dense_570_bias_m:5G
9assignvariableop_125_adam_batch_normalization_514_gamma_m:5F
8assignvariableop_126_adam_batch_normalization_514_beta_m:5>
,assignvariableop_127_adam_dense_571_kernel_m:558
*assignvariableop_128_adam_dense_571_bias_m:5G
9assignvariableop_129_adam_batch_normalization_515_gamma_m:5F
8assignvariableop_130_adam_batch_normalization_515_beta_m:5>
,assignvariableop_131_adam_dense_572_kernel_m:58
*assignvariableop_132_adam_dense_572_bias_m:>
,assignvariableop_133_adam_dense_560_kernel_v:A8
*assignvariableop_134_adam_dense_560_bias_v:AG
9assignvariableop_135_adam_batch_normalization_504_gamma_v:AF
8assignvariableop_136_adam_batch_normalization_504_beta_v:A>
,assignvariableop_137_adam_dense_561_kernel_v:AA8
*assignvariableop_138_adam_dense_561_bias_v:AG
9assignvariableop_139_adam_batch_normalization_505_gamma_v:AF
8assignvariableop_140_adam_batch_normalization_505_beta_v:A>
,assignvariableop_141_adam_dense_562_kernel_v:AA8
*assignvariableop_142_adam_dense_562_bias_v:AG
9assignvariableop_143_adam_batch_normalization_506_gamma_v:AF
8assignvariableop_144_adam_batch_normalization_506_beta_v:A>
,assignvariableop_145_adam_dense_563_kernel_v:AA8
*assignvariableop_146_adam_dense_563_bias_v:AG
9assignvariableop_147_adam_batch_normalization_507_gamma_v:AF
8assignvariableop_148_adam_batch_normalization_507_beta_v:A>
,assignvariableop_149_adam_dense_564_kernel_v:A8
*assignvariableop_150_adam_dense_564_bias_v:G
9assignvariableop_151_adam_batch_normalization_508_gamma_v:F
8assignvariableop_152_adam_batch_normalization_508_beta_v:>
,assignvariableop_153_adam_dense_565_kernel_v:8
*assignvariableop_154_adam_dense_565_bias_v:G
9assignvariableop_155_adam_batch_normalization_509_gamma_v:F
8assignvariableop_156_adam_batch_normalization_509_beta_v:>
,assignvariableop_157_adam_dense_566_kernel_v:8
*assignvariableop_158_adam_dense_566_bias_v:G
9assignvariableop_159_adam_batch_normalization_510_gamma_v:F
8assignvariableop_160_adam_batch_normalization_510_beta_v:>
,assignvariableop_161_adam_dense_567_kernel_v:58
*assignvariableop_162_adam_dense_567_bias_v:5G
9assignvariableop_163_adam_batch_normalization_511_gamma_v:5F
8assignvariableop_164_adam_batch_normalization_511_beta_v:5>
,assignvariableop_165_adam_dense_568_kernel_v:558
*assignvariableop_166_adam_dense_568_bias_v:5G
9assignvariableop_167_adam_batch_normalization_512_gamma_v:5F
8assignvariableop_168_adam_batch_normalization_512_beta_v:5>
,assignvariableop_169_adam_dense_569_kernel_v:558
*assignvariableop_170_adam_dense_569_bias_v:5G
9assignvariableop_171_adam_batch_normalization_513_gamma_v:5F
8assignvariableop_172_adam_batch_normalization_513_beta_v:5>
,assignvariableop_173_adam_dense_570_kernel_v:558
*assignvariableop_174_adam_dense_570_bias_v:5G
9assignvariableop_175_adam_batch_normalization_514_gamma_v:5F
8assignvariableop_176_adam_batch_normalization_514_beta_v:5>
,assignvariableop_177_adam_dense_571_kernel_v:558
*assignvariableop_178_adam_dense_571_bias_v:5G
9assignvariableop_179_adam_batch_normalization_515_gamma_v:5F
8assignvariableop_180_adam_batch_normalization_515_beta_v:5>
,assignvariableop_181_adam_dense_572_kernel_v:58
*assignvariableop_182_adam_dense_572_bias_v:
identity_184¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_111¢AssignVariableOp_112¢AssignVariableOp_113¢AssignVariableOp_114¢AssignVariableOp_115¢AssignVariableOp_116¢AssignVariableOp_117¢AssignVariableOp_118¢AssignVariableOp_119¢AssignVariableOp_12¢AssignVariableOp_120¢AssignVariableOp_121¢AssignVariableOp_122¢AssignVariableOp_123¢AssignVariableOp_124¢AssignVariableOp_125¢AssignVariableOp_126¢AssignVariableOp_127¢AssignVariableOp_128¢AssignVariableOp_129¢AssignVariableOp_13¢AssignVariableOp_130¢AssignVariableOp_131¢AssignVariableOp_132¢AssignVariableOp_133¢AssignVariableOp_134¢AssignVariableOp_135¢AssignVariableOp_136¢AssignVariableOp_137¢AssignVariableOp_138¢AssignVariableOp_139¢AssignVariableOp_14¢AssignVariableOp_140¢AssignVariableOp_141¢AssignVariableOp_142¢AssignVariableOp_143¢AssignVariableOp_144¢AssignVariableOp_145¢AssignVariableOp_146¢AssignVariableOp_147¢AssignVariableOp_148¢AssignVariableOp_149¢AssignVariableOp_15¢AssignVariableOp_150¢AssignVariableOp_151¢AssignVariableOp_152¢AssignVariableOp_153¢AssignVariableOp_154¢AssignVariableOp_155¢AssignVariableOp_156¢AssignVariableOp_157¢AssignVariableOp_158¢AssignVariableOp_159¢AssignVariableOp_16¢AssignVariableOp_160¢AssignVariableOp_161¢AssignVariableOp_162¢AssignVariableOp_163¢AssignVariableOp_164¢AssignVariableOp_165¢AssignVariableOp_166¢AssignVariableOp_167¢AssignVariableOp_168¢AssignVariableOp_169¢AssignVariableOp_17¢AssignVariableOp_170¢AssignVariableOp_171¢AssignVariableOp_172¢AssignVariableOp_173¢AssignVariableOp_174¢AssignVariableOp_175¢AssignVariableOp_176¢AssignVariableOp_177¢AssignVariableOp_178¢AssignVariableOp_179¢AssignVariableOp_18¢AssignVariableOp_180¢AssignVariableOp_181¢AssignVariableOp_182¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99±g
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:¸*
dtype0*Öf
valueÌfBÉf¸B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-22/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-22/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-22/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-24/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-24/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-24/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-24/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-22/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-24/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHå
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:¸*
dtype0*
valueüBù¸B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ½
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ö
_output_shapesã
à::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*É
dtypes¾
»2¸		[
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_560_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_560_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batch_normalization_504_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_504_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_504_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_8AssignVariableOp:assignvariableop_8_batch_normalization_504_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_561_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_561_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batch_normalization_505_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_505_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_505_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_14AssignVariableOp;assignvariableop_14_batch_normalization_505_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_562_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_562_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batch_normalization_506_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_506_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_506_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_20AssignVariableOp;assignvariableop_20_batch_normalization_506_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_563_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_563_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_23AssignVariableOp1assignvariableop_23_batch_normalization_507_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_507_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_25AssignVariableOp7assignvariableop_25_batch_normalization_507_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_26AssignVariableOp;assignvariableop_26_batch_normalization_507_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_564_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_564_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_29AssignVariableOp1assignvariableop_29_batch_normalization_508_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_508_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_31AssignVariableOp7assignvariableop_31_batch_normalization_508_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_32AssignVariableOp;assignvariableop_32_batch_normalization_508_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_565_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_565_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_35AssignVariableOp1assignvariableop_35_batch_normalization_509_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_509_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_37AssignVariableOp7assignvariableop_37_batch_normalization_509_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_38AssignVariableOp;assignvariableop_38_batch_normalization_509_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp$assignvariableop_39_dense_566_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_566_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_41AssignVariableOp1assignvariableop_41_batch_normalization_510_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_batch_normalization_510_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_43AssignVariableOp7assignvariableop_43_batch_normalization_510_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_44AssignVariableOp;assignvariableop_44_batch_normalization_510_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp$assignvariableop_45_dense_567_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_567_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_47AssignVariableOp1assignvariableop_47_batch_normalization_511_gammaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_48AssignVariableOp0assignvariableop_48_batch_normalization_511_betaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_49AssignVariableOp7assignvariableop_49_batch_normalization_511_moving_meanIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_50AssignVariableOp;assignvariableop_50_batch_normalization_511_moving_varianceIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp$assignvariableop_51_dense_568_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp"assignvariableop_52_dense_568_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_53AssignVariableOp1assignvariableop_53_batch_normalization_512_gammaIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_54AssignVariableOp0assignvariableop_54_batch_normalization_512_betaIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_55AssignVariableOp7assignvariableop_55_batch_normalization_512_moving_meanIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_56AssignVariableOp;assignvariableop_56_batch_normalization_512_moving_varianceIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp$assignvariableop_57_dense_569_kernelIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp"assignvariableop_58_dense_569_biasIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_59AssignVariableOp1assignvariableop_59_batch_normalization_513_gammaIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_60AssignVariableOp0assignvariableop_60_batch_normalization_513_betaIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_61AssignVariableOp7assignvariableop_61_batch_normalization_513_moving_meanIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_62AssignVariableOp;assignvariableop_62_batch_normalization_513_moving_varianceIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp$assignvariableop_63_dense_570_kernelIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp"assignvariableop_64_dense_570_biasIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_65AssignVariableOp1assignvariableop_65_batch_normalization_514_gammaIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_66AssignVariableOp0assignvariableop_66_batch_normalization_514_betaIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_67AssignVariableOp7assignvariableop_67_batch_normalization_514_moving_meanIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_68AssignVariableOp;assignvariableop_68_batch_normalization_514_moving_varianceIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp$assignvariableop_69_dense_571_kernelIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp"assignvariableop_70_dense_571_biasIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_71AssignVariableOp1assignvariableop_71_batch_normalization_515_gammaIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_72AssignVariableOp0assignvariableop_72_batch_normalization_515_betaIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_73AssignVariableOp7assignvariableop_73_batch_normalization_515_moving_meanIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_74AssignVariableOp;assignvariableop_74_batch_normalization_515_moving_varianceIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_75AssignVariableOp$assignvariableop_75_dense_572_kernelIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_76AssignVariableOp"assignvariableop_76_dense_572_biasIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_77AssignVariableOpassignvariableop_77_adam_iterIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_78AssignVariableOpassignvariableop_78_adam_beta_1Identity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_79AssignVariableOpassignvariableop_79_adam_beta_2Identity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_80AssignVariableOpassignvariableop_80_adam_decayIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_81AssignVariableOpassignvariableop_81_totalIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_82AssignVariableOpassignvariableop_82_count_1Identity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adam_dense_560_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adam_dense_560_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_85AssignVariableOp8assignvariableop_85_adam_batch_normalization_504_gamma_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_86AssignVariableOp7assignvariableop_86_adam_batch_normalization_504_beta_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_87AssignVariableOp+assignvariableop_87_adam_dense_561_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_88AssignVariableOp)assignvariableop_88_adam_dense_561_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_89AssignVariableOp8assignvariableop_89_adam_batch_normalization_505_gamma_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_90AssignVariableOp7assignvariableop_90_adam_batch_normalization_505_beta_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_91AssignVariableOp+assignvariableop_91_adam_dense_562_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_92AssignVariableOp)assignvariableop_92_adam_dense_562_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_93AssignVariableOp8assignvariableop_93_adam_batch_normalization_506_gamma_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_94AssignVariableOp7assignvariableop_94_adam_batch_normalization_506_beta_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_95AssignVariableOp+assignvariableop_95_adam_dense_563_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_96AssignVariableOp)assignvariableop_96_adam_dense_563_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_97AssignVariableOp8assignvariableop_97_adam_batch_normalization_507_gamma_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_98AssignVariableOp7assignvariableop_98_adam_batch_normalization_507_beta_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_99AssignVariableOp+assignvariableop_99_adam_dense_564_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_100AssignVariableOp*assignvariableop_100_adam_dense_564_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_101AssignVariableOp9assignvariableop_101_adam_batch_normalization_508_gamma_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_102AssignVariableOp8assignvariableop_102_adam_batch_normalization_508_beta_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_103AssignVariableOp,assignvariableop_103_adam_dense_565_kernel_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_dense_565_bias_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_105AssignVariableOp9assignvariableop_105_adam_batch_normalization_509_gamma_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_106AssignVariableOp8assignvariableop_106_adam_batch_normalization_509_beta_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_dense_566_kernel_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_dense_566_bias_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_109AssignVariableOp9assignvariableop_109_adam_batch_normalization_510_gamma_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_110AssignVariableOp8assignvariableop_110_adam_batch_normalization_510_beta_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_111AssignVariableOp,assignvariableop_111_adam_dense_567_kernel_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_112AssignVariableOp*assignvariableop_112_adam_dense_567_bias_mIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_113AssignVariableOp9assignvariableop_113_adam_batch_normalization_511_gamma_mIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_114AssignVariableOp8assignvariableop_114_adam_batch_normalization_511_beta_mIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_115AssignVariableOp,assignvariableop_115_adam_dense_568_kernel_mIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_116AssignVariableOp*assignvariableop_116_adam_dense_568_bias_mIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_117AssignVariableOp9assignvariableop_117_adam_batch_normalization_512_gamma_mIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_118AssignVariableOp8assignvariableop_118_adam_batch_normalization_512_beta_mIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_119AssignVariableOp,assignvariableop_119_adam_dense_569_kernel_mIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_120AssignVariableOp*assignvariableop_120_adam_dense_569_bias_mIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_121AssignVariableOp9assignvariableop_121_adam_batch_normalization_513_gamma_mIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_122AssignVariableOp8assignvariableop_122_adam_batch_normalization_513_beta_mIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_123AssignVariableOp,assignvariableop_123_adam_dense_570_kernel_mIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_124AssignVariableOp*assignvariableop_124_adam_dense_570_bias_mIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_125AssignVariableOp9assignvariableop_125_adam_batch_normalization_514_gamma_mIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_126AssignVariableOp8assignvariableop_126_adam_batch_normalization_514_beta_mIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_127AssignVariableOp,assignvariableop_127_adam_dense_571_kernel_mIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_128AssignVariableOp*assignvariableop_128_adam_dense_571_bias_mIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_129AssignVariableOp9assignvariableop_129_adam_batch_normalization_515_gamma_mIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_130AssignVariableOp8assignvariableop_130_adam_batch_normalization_515_beta_mIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_131AssignVariableOp,assignvariableop_131_adam_dense_572_kernel_mIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_132AssignVariableOp*assignvariableop_132_adam_dense_572_bias_mIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_133AssignVariableOp,assignvariableop_133_adam_dense_560_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_134AssignVariableOp*assignvariableop_134_adam_dense_560_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_135AssignVariableOp9assignvariableop_135_adam_batch_normalization_504_gamma_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_136AssignVariableOp8assignvariableop_136_adam_batch_normalization_504_beta_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_137AssignVariableOp,assignvariableop_137_adam_dense_561_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_138AssignVariableOp*assignvariableop_138_adam_dense_561_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_139AssignVariableOp9assignvariableop_139_adam_batch_normalization_505_gamma_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_140AssignVariableOp8assignvariableop_140_adam_batch_normalization_505_beta_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_141AssignVariableOp,assignvariableop_141_adam_dense_562_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_142AssignVariableOp*assignvariableop_142_adam_dense_562_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_143AssignVariableOp9assignvariableop_143_adam_batch_normalization_506_gamma_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_144AssignVariableOp8assignvariableop_144_adam_batch_normalization_506_beta_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_145AssignVariableOp,assignvariableop_145_adam_dense_563_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_146AssignVariableOp*assignvariableop_146_adam_dense_563_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_147AssignVariableOp9assignvariableop_147_adam_batch_normalization_507_gamma_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_148AssignVariableOp8assignvariableop_148_adam_batch_normalization_507_beta_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_149AssignVariableOp,assignvariableop_149_adam_dense_564_kernel_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_150AssignVariableOp*assignvariableop_150_adam_dense_564_bias_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_151AssignVariableOp9assignvariableop_151_adam_batch_normalization_508_gamma_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_152AssignVariableOp8assignvariableop_152_adam_batch_normalization_508_beta_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_153AssignVariableOp,assignvariableop_153_adam_dense_565_kernel_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_154AssignVariableOp*assignvariableop_154_adam_dense_565_bias_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_155AssignVariableOp9assignvariableop_155_adam_batch_normalization_509_gamma_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_156AssignVariableOp8assignvariableop_156_adam_batch_normalization_509_beta_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_157AssignVariableOp,assignvariableop_157_adam_dense_566_kernel_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_158AssignVariableOp*assignvariableop_158_adam_dense_566_bias_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_159AssignVariableOp9assignvariableop_159_adam_batch_normalization_510_gamma_vIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_160AssignVariableOp8assignvariableop_160_adam_batch_normalization_510_beta_vIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_161AssignVariableOp,assignvariableop_161_adam_dense_567_kernel_vIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_162AssignVariableOp*assignvariableop_162_adam_dense_567_bias_vIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_163AssignVariableOp9assignvariableop_163_adam_batch_normalization_511_gamma_vIdentity_163:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_164AssignVariableOp8assignvariableop_164_adam_batch_normalization_511_beta_vIdentity_164:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_165IdentityRestoreV2:tensors:165"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_165AssignVariableOp,assignvariableop_165_adam_dense_568_kernel_vIdentity_165:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_166IdentityRestoreV2:tensors:166"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_166AssignVariableOp*assignvariableop_166_adam_dense_568_bias_vIdentity_166:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_167IdentityRestoreV2:tensors:167"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_167AssignVariableOp9assignvariableop_167_adam_batch_normalization_512_gamma_vIdentity_167:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_168IdentityRestoreV2:tensors:168"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_168AssignVariableOp8assignvariableop_168_adam_batch_normalization_512_beta_vIdentity_168:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_169IdentityRestoreV2:tensors:169"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_169AssignVariableOp,assignvariableop_169_adam_dense_569_kernel_vIdentity_169:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_170IdentityRestoreV2:tensors:170"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_170AssignVariableOp*assignvariableop_170_adam_dense_569_bias_vIdentity_170:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_171IdentityRestoreV2:tensors:171"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_171AssignVariableOp9assignvariableop_171_adam_batch_normalization_513_gamma_vIdentity_171:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_172IdentityRestoreV2:tensors:172"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_172AssignVariableOp8assignvariableop_172_adam_batch_normalization_513_beta_vIdentity_172:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_173IdentityRestoreV2:tensors:173"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_173AssignVariableOp,assignvariableop_173_adam_dense_570_kernel_vIdentity_173:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_174IdentityRestoreV2:tensors:174"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_174AssignVariableOp*assignvariableop_174_adam_dense_570_bias_vIdentity_174:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_175IdentityRestoreV2:tensors:175"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_175AssignVariableOp9assignvariableop_175_adam_batch_normalization_514_gamma_vIdentity_175:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_176IdentityRestoreV2:tensors:176"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_176AssignVariableOp8assignvariableop_176_adam_batch_normalization_514_beta_vIdentity_176:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_177IdentityRestoreV2:tensors:177"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_177AssignVariableOp,assignvariableop_177_adam_dense_571_kernel_vIdentity_177:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_178IdentityRestoreV2:tensors:178"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_178AssignVariableOp*assignvariableop_178_adam_dense_571_bias_vIdentity_178:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_179IdentityRestoreV2:tensors:179"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_179AssignVariableOp9assignvariableop_179_adam_batch_normalization_515_gamma_vIdentity_179:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_180IdentityRestoreV2:tensors:180"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_180AssignVariableOp8assignvariableop_180_adam_batch_normalization_515_beta_vIdentity_180:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_181IdentityRestoreV2:tensors:181"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_181AssignVariableOp,assignvariableop_181_adam_dense_572_kernel_vIdentity_181:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_182IdentityRestoreV2:tensors:182"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_182AssignVariableOp*assignvariableop_182_adam_dense_572_bias_vIdentity_182:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ý 
Identity_183Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_173^AssignVariableOp_174^AssignVariableOp_175^AssignVariableOp_176^AssignVariableOp_177^AssignVariableOp_178^AssignVariableOp_179^AssignVariableOp_18^AssignVariableOp_180^AssignVariableOp_181^AssignVariableOp_182^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_184IdentityIdentity_183:output:0^NoOp_1*
T0*
_output_shapes
: É 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_173^AssignVariableOp_174^AssignVariableOp_175^AssignVariableOp_176^AssignVariableOp_177^AssignVariableOp_178^AssignVariableOp_179^AssignVariableOp_18^AssignVariableOp_180^AssignVariableOp_181^AssignVariableOp_182^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_184Identity_184:output:0*
_input_shapesó
ð: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_168AssignVariableOp_1682,
AssignVariableOp_169AssignVariableOp_1692*
AssignVariableOp_17AssignVariableOp_172,
AssignVariableOp_170AssignVariableOp_1702,
AssignVariableOp_171AssignVariableOp_1712,
AssignVariableOp_172AssignVariableOp_1722,
AssignVariableOp_173AssignVariableOp_1732,
AssignVariableOp_174AssignVariableOp_1742,
AssignVariableOp_175AssignVariableOp_1752,
AssignVariableOp_176AssignVariableOp_1762,
AssignVariableOp_177AssignVariableOp_1772,
AssignVariableOp_178AssignVariableOp_1782,
AssignVariableOp_179AssignVariableOp_1792*
AssignVariableOp_18AssignVariableOp_182,
AssignVariableOp_180AssignVariableOp_1802,
AssignVariableOp_181AssignVariableOp_1812,
AssignVariableOp_182AssignVariableOp_1822*
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
Ô
9__inference_batch_normalization_513_layer_call_fn_1013998

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1009688o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
É	
÷
F__inference_dense_568_layer_call_and_return_conditional_losses_1013863

inputs0
matmul_readvariableop_resource:55-
biasadd_readvariableop_resource:5
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:55*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:5*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Æ

+__inference_dense_568_layer_call_fn_1013853

inputs
unknown:55
	unknown_0:5
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_568_layer_call_and_return_conditional_losses_1010143o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_515_layer_call_fn_1014216

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1009852o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_513_layer_call_and_return_conditional_losses_1014062

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1013691

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1009606

inputs5
'assignmovingavg_readvariableop_resource:57
)assignmovingavg_1_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:5/
!batchnorm_readvariableop_resource:5
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:5
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:5*
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
:5*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:5x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5¬
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
:5*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:5~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5´
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:5v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Æ

+__inference_dense_562_layer_call_fn_1013199

inputs
unknown:AA
	unknown_0:A
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_562_layer_call_and_return_conditional_losses_1009951o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
É	
÷
F__inference_dense_566_layer_call_and_return_conditional_losses_1013645

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1013289

inputs5
'assignmovingavg_readvariableop_resource:A7
)assignmovingavg_1_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A/
!batchnorm_readvariableop_resource:A
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:A
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:A*
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
:A*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A¬
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
:A*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:A~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A´
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_512_layer_call_fn_1013876

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1009559o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_511_layer_call_fn_1013780

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1009524o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
É	
÷
F__inference_dense_561_layer_call_and_return_conditional_losses_1009919

inputs0
matmul_readvariableop_resource:AA-
biasadd_readvariableop_resource:A
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:AA*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1013364

inputs/
!batchnorm_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A1
#batchnorm_readvariableop_1_resource:A1
#batchnorm_readvariableop_2_resource:A
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_1013408

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿA:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_515_layer_call_and_return_conditional_losses_1010259

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1009723

inputs/
!batchnorm_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:51
#batchnorm_readvariableop_1_resource:51
#batchnorm_readvariableop_2_resource:5
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:5z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Æ

+__inference_dense_565_layer_call_fn_1013526

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_565_layer_call_and_return_conditional_losses_1010047o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1009524

inputs5
'assignmovingavg_readvariableop_resource:57
)assignmovingavg_1_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:5/
!batchnorm_readvariableop_resource:5
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:5
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:5*
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
:5*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:5x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5¬
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
:5*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:5~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5´
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:5v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
É	
÷
F__inference_dense_572_layer_call_and_return_conditional_losses_1010271

inputs0
matmul_readvariableop_resource:5-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
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
:ÿÿÿÿÿÿÿÿÿ5: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1008985

inputs/
!batchnorm_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A1
#batchnorm_readvariableop_1_resource:A1
#batchnorm_readvariableop_2_resource:A
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_514_layer_call_and_return_conditional_losses_1014171

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_508_layer_call_fn_1013453

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1009278o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_1013081

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿA:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
­
°T
"__inference__wrapped_model_1008879
normalization_56_input(
$sequential_56_normalization_56_sub_y)
%sequential_56_normalization_56_sqrt_xH
6sequential_56_dense_560_matmul_readvariableop_resource:AE
7sequential_56_dense_560_biasadd_readvariableop_resource:AU
Gsequential_56_batch_normalization_504_batchnorm_readvariableop_resource:AY
Ksequential_56_batch_normalization_504_batchnorm_mul_readvariableop_resource:AW
Isequential_56_batch_normalization_504_batchnorm_readvariableop_1_resource:AW
Isequential_56_batch_normalization_504_batchnorm_readvariableop_2_resource:AH
6sequential_56_dense_561_matmul_readvariableop_resource:AAE
7sequential_56_dense_561_biasadd_readvariableop_resource:AU
Gsequential_56_batch_normalization_505_batchnorm_readvariableop_resource:AY
Ksequential_56_batch_normalization_505_batchnorm_mul_readvariableop_resource:AW
Isequential_56_batch_normalization_505_batchnorm_readvariableop_1_resource:AW
Isequential_56_batch_normalization_505_batchnorm_readvariableop_2_resource:AH
6sequential_56_dense_562_matmul_readvariableop_resource:AAE
7sequential_56_dense_562_biasadd_readvariableop_resource:AU
Gsequential_56_batch_normalization_506_batchnorm_readvariableop_resource:AY
Ksequential_56_batch_normalization_506_batchnorm_mul_readvariableop_resource:AW
Isequential_56_batch_normalization_506_batchnorm_readvariableop_1_resource:AW
Isequential_56_batch_normalization_506_batchnorm_readvariableop_2_resource:AH
6sequential_56_dense_563_matmul_readvariableop_resource:AAE
7sequential_56_dense_563_biasadd_readvariableop_resource:AU
Gsequential_56_batch_normalization_507_batchnorm_readvariableop_resource:AY
Ksequential_56_batch_normalization_507_batchnorm_mul_readvariableop_resource:AW
Isequential_56_batch_normalization_507_batchnorm_readvariableop_1_resource:AW
Isequential_56_batch_normalization_507_batchnorm_readvariableop_2_resource:AH
6sequential_56_dense_564_matmul_readvariableop_resource:AE
7sequential_56_dense_564_biasadd_readvariableop_resource:U
Gsequential_56_batch_normalization_508_batchnorm_readvariableop_resource:Y
Ksequential_56_batch_normalization_508_batchnorm_mul_readvariableop_resource:W
Isequential_56_batch_normalization_508_batchnorm_readvariableop_1_resource:W
Isequential_56_batch_normalization_508_batchnorm_readvariableop_2_resource:H
6sequential_56_dense_565_matmul_readvariableop_resource:E
7sequential_56_dense_565_biasadd_readvariableop_resource:U
Gsequential_56_batch_normalization_509_batchnorm_readvariableop_resource:Y
Ksequential_56_batch_normalization_509_batchnorm_mul_readvariableop_resource:W
Isequential_56_batch_normalization_509_batchnorm_readvariableop_1_resource:W
Isequential_56_batch_normalization_509_batchnorm_readvariableop_2_resource:H
6sequential_56_dense_566_matmul_readvariableop_resource:E
7sequential_56_dense_566_biasadd_readvariableop_resource:U
Gsequential_56_batch_normalization_510_batchnorm_readvariableop_resource:Y
Ksequential_56_batch_normalization_510_batchnorm_mul_readvariableop_resource:W
Isequential_56_batch_normalization_510_batchnorm_readvariableop_1_resource:W
Isequential_56_batch_normalization_510_batchnorm_readvariableop_2_resource:H
6sequential_56_dense_567_matmul_readvariableop_resource:5E
7sequential_56_dense_567_biasadd_readvariableop_resource:5U
Gsequential_56_batch_normalization_511_batchnorm_readvariableop_resource:5Y
Ksequential_56_batch_normalization_511_batchnorm_mul_readvariableop_resource:5W
Isequential_56_batch_normalization_511_batchnorm_readvariableop_1_resource:5W
Isequential_56_batch_normalization_511_batchnorm_readvariableop_2_resource:5H
6sequential_56_dense_568_matmul_readvariableop_resource:55E
7sequential_56_dense_568_biasadd_readvariableop_resource:5U
Gsequential_56_batch_normalization_512_batchnorm_readvariableop_resource:5Y
Ksequential_56_batch_normalization_512_batchnorm_mul_readvariableop_resource:5W
Isequential_56_batch_normalization_512_batchnorm_readvariableop_1_resource:5W
Isequential_56_batch_normalization_512_batchnorm_readvariableop_2_resource:5H
6sequential_56_dense_569_matmul_readvariableop_resource:55E
7sequential_56_dense_569_biasadd_readvariableop_resource:5U
Gsequential_56_batch_normalization_513_batchnorm_readvariableop_resource:5Y
Ksequential_56_batch_normalization_513_batchnorm_mul_readvariableop_resource:5W
Isequential_56_batch_normalization_513_batchnorm_readvariableop_1_resource:5W
Isequential_56_batch_normalization_513_batchnorm_readvariableop_2_resource:5H
6sequential_56_dense_570_matmul_readvariableop_resource:55E
7sequential_56_dense_570_biasadd_readvariableop_resource:5U
Gsequential_56_batch_normalization_514_batchnorm_readvariableop_resource:5Y
Ksequential_56_batch_normalization_514_batchnorm_mul_readvariableop_resource:5W
Isequential_56_batch_normalization_514_batchnorm_readvariableop_1_resource:5W
Isequential_56_batch_normalization_514_batchnorm_readvariableop_2_resource:5H
6sequential_56_dense_571_matmul_readvariableop_resource:55E
7sequential_56_dense_571_biasadd_readvariableop_resource:5U
Gsequential_56_batch_normalization_515_batchnorm_readvariableop_resource:5Y
Ksequential_56_batch_normalization_515_batchnorm_mul_readvariableop_resource:5W
Isequential_56_batch_normalization_515_batchnorm_readvariableop_1_resource:5W
Isequential_56_batch_normalization_515_batchnorm_readvariableop_2_resource:5H
6sequential_56_dense_572_matmul_readvariableop_resource:5E
7sequential_56_dense_572_biasadd_readvariableop_resource:
identity¢>sequential_56/batch_normalization_504/batchnorm/ReadVariableOp¢@sequential_56/batch_normalization_504/batchnorm/ReadVariableOp_1¢@sequential_56/batch_normalization_504/batchnorm/ReadVariableOp_2¢Bsequential_56/batch_normalization_504/batchnorm/mul/ReadVariableOp¢>sequential_56/batch_normalization_505/batchnorm/ReadVariableOp¢@sequential_56/batch_normalization_505/batchnorm/ReadVariableOp_1¢@sequential_56/batch_normalization_505/batchnorm/ReadVariableOp_2¢Bsequential_56/batch_normalization_505/batchnorm/mul/ReadVariableOp¢>sequential_56/batch_normalization_506/batchnorm/ReadVariableOp¢@sequential_56/batch_normalization_506/batchnorm/ReadVariableOp_1¢@sequential_56/batch_normalization_506/batchnorm/ReadVariableOp_2¢Bsequential_56/batch_normalization_506/batchnorm/mul/ReadVariableOp¢>sequential_56/batch_normalization_507/batchnorm/ReadVariableOp¢@sequential_56/batch_normalization_507/batchnorm/ReadVariableOp_1¢@sequential_56/batch_normalization_507/batchnorm/ReadVariableOp_2¢Bsequential_56/batch_normalization_507/batchnorm/mul/ReadVariableOp¢>sequential_56/batch_normalization_508/batchnorm/ReadVariableOp¢@sequential_56/batch_normalization_508/batchnorm/ReadVariableOp_1¢@sequential_56/batch_normalization_508/batchnorm/ReadVariableOp_2¢Bsequential_56/batch_normalization_508/batchnorm/mul/ReadVariableOp¢>sequential_56/batch_normalization_509/batchnorm/ReadVariableOp¢@sequential_56/batch_normalization_509/batchnorm/ReadVariableOp_1¢@sequential_56/batch_normalization_509/batchnorm/ReadVariableOp_2¢Bsequential_56/batch_normalization_509/batchnorm/mul/ReadVariableOp¢>sequential_56/batch_normalization_510/batchnorm/ReadVariableOp¢@sequential_56/batch_normalization_510/batchnorm/ReadVariableOp_1¢@sequential_56/batch_normalization_510/batchnorm/ReadVariableOp_2¢Bsequential_56/batch_normalization_510/batchnorm/mul/ReadVariableOp¢>sequential_56/batch_normalization_511/batchnorm/ReadVariableOp¢@sequential_56/batch_normalization_511/batchnorm/ReadVariableOp_1¢@sequential_56/batch_normalization_511/batchnorm/ReadVariableOp_2¢Bsequential_56/batch_normalization_511/batchnorm/mul/ReadVariableOp¢>sequential_56/batch_normalization_512/batchnorm/ReadVariableOp¢@sequential_56/batch_normalization_512/batchnorm/ReadVariableOp_1¢@sequential_56/batch_normalization_512/batchnorm/ReadVariableOp_2¢Bsequential_56/batch_normalization_512/batchnorm/mul/ReadVariableOp¢>sequential_56/batch_normalization_513/batchnorm/ReadVariableOp¢@sequential_56/batch_normalization_513/batchnorm/ReadVariableOp_1¢@sequential_56/batch_normalization_513/batchnorm/ReadVariableOp_2¢Bsequential_56/batch_normalization_513/batchnorm/mul/ReadVariableOp¢>sequential_56/batch_normalization_514/batchnorm/ReadVariableOp¢@sequential_56/batch_normalization_514/batchnorm/ReadVariableOp_1¢@sequential_56/batch_normalization_514/batchnorm/ReadVariableOp_2¢Bsequential_56/batch_normalization_514/batchnorm/mul/ReadVariableOp¢>sequential_56/batch_normalization_515/batchnorm/ReadVariableOp¢@sequential_56/batch_normalization_515/batchnorm/ReadVariableOp_1¢@sequential_56/batch_normalization_515/batchnorm/ReadVariableOp_2¢Bsequential_56/batch_normalization_515/batchnorm/mul/ReadVariableOp¢.sequential_56/dense_560/BiasAdd/ReadVariableOp¢-sequential_56/dense_560/MatMul/ReadVariableOp¢.sequential_56/dense_561/BiasAdd/ReadVariableOp¢-sequential_56/dense_561/MatMul/ReadVariableOp¢.sequential_56/dense_562/BiasAdd/ReadVariableOp¢-sequential_56/dense_562/MatMul/ReadVariableOp¢.sequential_56/dense_563/BiasAdd/ReadVariableOp¢-sequential_56/dense_563/MatMul/ReadVariableOp¢.sequential_56/dense_564/BiasAdd/ReadVariableOp¢-sequential_56/dense_564/MatMul/ReadVariableOp¢.sequential_56/dense_565/BiasAdd/ReadVariableOp¢-sequential_56/dense_565/MatMul/ReadVariableOp¢.sequential_56/dense_566/BiasAdd/ReadVariableOp¢-sequential_56/dense_566/MatMul/ReadVariableOp¢.sequential_56/dense_567/BiasAdd/ReadVariableOp¢-sequential_56/dense_567/MatMul/ReadVariableOp¢.sequential_56/dense_568/BiasAdd/ReadVariableOp¢-sequential_56/dense_568/MatMul/ReadVariableOp¢.sequential_56/dense_569/BiasAdd/ReadVariableOp¢-sequential_56/dense_569/MatMul/ReadVariableOp¢.sequential_56/dense_570/BiasAdd/ReadVariableOp¢-sequential_56/dense_570/MatMul/ReadVariableOp¢.sequential_56/dense_571/BiasAdd/ReadVariableOp¢-sequential_56/dense_571/MatMul/ReadVariableOp¢.sequential_56/dense_572/BiasAdd/ReadVariableOp¢-sequential_56/dense_572/MatMul/ReadVariableOp
"sequential_56/normalization_56/subSubnormalization_56_input$sequential_56_normalization_56_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#sequential_56/normalization_56/SqrtSqrt%sequential_56_normalization_56_sqrt_x*
T0*
_output_shapes

:m
(sequential_56/normalization_56/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¶
&sequential_56/normalization_56/MaximumMaximum'sequential_56/normalization_56/Sqrt:y:01sequential_56/normalization_56/Maximum/y:output:0*
T0*
_output_shapes

:·
&sequential_56/normalization_56/truedivRealDiv&sequential_56/normalization_56/sub:z:0*sequential_56/normalization_56/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_56/dense_560/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_560_matmul_readvariableop_resource*
_output_shapes

:A*
dtype0½
sequential_56/dense_560/MatMulMatMul*sequential_56/normalization_56/truediv:z:05sequential_56/dense_560/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¢
.sequential_56/dense_560/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_560_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0¾
sequential_56/dense_560/BiasAddBiasAdd(sequential_56/dense_560/MatMul:product:06sequential_56/dense_560/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÂ
>sequential_56/batch_normalization_504/batchnorm/ReadVariableOpReadVariableOpGsequential_56_batch_normalization_504_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0z
5sequential_56/batch_normalization_504/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_56/batch_normalization_504/batchnorm/addAddV2Fsequential_56/batch_normalization_504/batchnorm/ReadVariableOp:value:0>sequential_56/batch_normalization_504/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
5sequential_56/batch_normalization_504/batchnorm/RsqrtRsqrt7sequential_56/batch_normalization_504/batchnorm/add:z:0*
T0*
_output_shapes
:AÊ
Bsequential_56/batch_normalization_504/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_56_batch_normalization_504_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0æ
3sequential_56/batch_normalization_504/batchnorm/mulMul9sequential_56/batch_normalization_504/batchnorm/Rsqrt:y:0Jsequential_56/batch_normalization_504/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:AÑ
5sequential_56/batch_normalization_504/batchnorm/mul_1Mul(sequential_56/dense_560/BiasAdd:output:07sequential_56/batch_normalization_504/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÆ
@sequential_56/batch_normalization_504/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_56_batch_normalization_504_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0ä
5sequential_56/batch_normalization_504/batchnorm/mul_2MulHsequential_56/batch_normalization_504/batchnorm/ReadVariableOp_1:value:07sequential_56/batch_normalization_504/batchnorm/mul:z:0*
T0*
_output_shapes
:AÆ
@sequential_56/batch_normalization_504/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_56_batch_normalization_504_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0ä
3sequential_56/batch_normalization_504/batchnorm/subSubHsequential_56/batch_normalization_504/batchnorm/ReadVariableOp_2:value:09sequential_56/batch_normalization_504/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aä
5sequential_56/batch_normalization_504/batchnorm/add_1AddV29sequential_56/batch_normalization_504/batchnorm/mul_1:z:07sequential_56/batch_normalization_504/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¨
'sequential_56/leaky_re_lu_504/LeakyRelu	LeakyRelu9sequential_56/batch_normalization_504/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>¤
-sequential_56/dense_561/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_561_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0È
sequential_56/dense_561/MatMulMatMul5sequential_56/leaky_re_lu_504/LeakyRelu:activations:05sequential_56/dense_561/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¢
.sequential_56/dense_561/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_561_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0¾
sequential_56/dense_561/BiasAddBiasAdd(sequential_56/dense_561/MatMul:product:06sequential_56/dense_561/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÂ
>sequential_56/batch_normalization_505/batchnorm/ReadVariableOpReadVariableOpGsequential_56_batch_normalization_505_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0z
5sequential_56/batch_normalization_505/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_56/batch_normalization_505/batchnorm/addAddV2Fsequential_56/batch_normalization_505/batchnorm/ReadVariableOp:value:0>sequential_56/batch_normalization_505/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
5sequential_56/batch_normalization_505/batchnorm/RsqrtRsqrt7sequential_56/batch_normalization_505/batchnorm/add:z:0*
T0*
_output_shapes
:AÊ
Bsequential_56/batch_normalization_505/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_56_batch_normalization_505_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0æ
3sequential_56/batch_normalization_505/batchnorm/mulMul9sequential_56/batch_normalization_505/batchnorm/Rsqrt:y:0Jsequential_56/batch_normalization_505/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:AÑ
5sequential_56/batch_normalization_505/batchnorm/mul_1Mul(sequential_56/dense_561/BiasAdd:output:07sequential_56/batch_normalization_505/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÆ
@sequential_56/batch_normalization_505/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_56_batch_normalization_505_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0ä
5sequential_56/batch_normalization_505/batchnorm/mul_2MulHsequential_56/batch_normalization_505/batchnorm/ReadVariableOp_1:value:07sequential_56/batch_normalization_505/batchnorm/mul:z:0*
T0*
_output_shapes
:AÆ
@sequential_56/batch_normalization_505/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_56_batch_normalization_505_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0ä
3sequential_56/batch_normalization_505/batchnorm/subSubHsequential_56/batch_normalization_505/batchnorm/ReadVariableOp_2:value:09sequential_56/batch_normalization_505/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aä
5sequential_56/batch_normalization_505/batchnorm/add_1AddV29sequential_56/batch_normalization_505/batchnorm/mul_1:z:07sequential_56/batch_normalization_505/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¨
'sequential_56/leaky_re_lu_505/LeakyRelu	LeakyRelu9sequential_56/batch_normalization_505/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>¤
-sequential_56/dense_562/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_562_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0È
sequential_56/dense_562/MatMulMatMul5sequential_56/leaky_re_lu_505/LeakyRelu:activations:05sequential_56/dense_562/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¢
.sequential_56/dense_562/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_562_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0¾
sequential_56/dense_562/BiasAddBiasAdd(sequential_56/dense_562/MatMul:product:06sequential_56/dense_562/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÂ
>sequential_56/batch_normalization_506/batchnorm/ReadVariableOpReadVariableOpGsequential_56_batch_normalization_506_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0z
5sequential_56/batch_normalization_506/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_56/batch_normalization_506/batchnorm/addAddV2Fsequential_56/batch_normalization_506/batchnorm/ReadVariableOp:value:0>sequential_56/batch_normalization_506/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
5sequential_56/batch_normalization_506/batchnorm/RsqrtRsqrt7sequential_56/batch_normalization_506/batchnorm/add:z:0*
T0*
_output_shapes
:AÊ
Bsequential_56/batch_normalization_506/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_56_batch_normalization_506_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0æ
3sequential_56/batch_normalization_506/batchnorm/mulMul9sequential_56/batch_normalization_506/batchnorm/Rsqrt:y:0Jsequential_56/batch_normalization_506/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:AÑ
5sequential_56/batch_normalization_506/batchnorm/mul_1Mul(sequential_56/dense_562/BiasAdd:output:07sequential_56/batch_normalization_506/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÆ
@sequential_56/batch_normalization_506/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_56_batch_normalization_506_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0ä
5sequential_56/batch_normalization_506/batchnorm/mul_2MulHsequential_56/batch_normalization_506/batchnorm/ReadVariableOp_1:value:07sequential_56/batch_normalization_506/batchnorm/mul:z:0*
T0*
_output_shapes
:AÆ
@sequential_56/batch_normalization_506/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_56_batch_normalization_506_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0ä
3sequential_56/batch_normalization_506/batchnorm/subSubHsequential_56/batch_normalization_506/batchnorm/ReadVariableOp_2:value:09sequential_56/batch_normalization_506/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aä
5sequential_56/batch_normalization_506/batchnorm/add_1AddV29sequential_56/batch_normalization_506/batchnorm/mul_1:z:07sequential_56/batch_normalization_506/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¨
'sequential_56/leaky_re_lu_506/LeakyRelu	LeakyRelu9sequential_56/batch_normalization_506/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>¤
-sequential_56/dense_563/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_563_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0È
sequential_56/dense_563/MatMulMatMul5sequential_56/leaky_re_lu_506/LeakyRelu:activations:05sequential_56/dense_563/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¢
.sequential_56/dense_563/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_563_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0¾
sequential_56/dense_563/BiasAddBiasAdd(sequential_56/dense_563/MatMul:product:06sequential_56/dense_563/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÂ
>sequential_56/batch_normalization_507/batchnorm/ReadVariableOpReadVariableOpGsequential_56_batch_normalization_507_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0z
5sequential_56/batch_normalization_507/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_56/batch_normalization_507/batchnorm/addAddV2Fsequential_56/batch_normalization_507/batchnorm/ReadVariableOp:value:0>sequential_56/batch_normalization_507/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
5sequential_56/batch_normalization_507/batchnorm/RsqrtRsqrt7sequential_56/batch_normalization_507/batchnorm/add:z:0*
T0*
_output_shapes
:AÊ
Bsequential_56/batch_normalization_507/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_56_batch_normalization_507_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0æ
3sequential_56/batch_normalization_507/batchnorm/mulMul9sequential_56/batch_normalization_507/batchnorm/Rsqrt:y:0Jsequential_56/batch_normalization_507/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:AÑ
5sequential_56/batch_normalization_507/batchnorm/mul_1Mul(sequential_56/dense_563/BiasAdd:output:07sequential_56/batch_normalization_507/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÆ
@sequential_56/batch_normalization_507/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_56_batch_normalization_507_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0ä
5sequential_56/batch_normalization_507/batchnorm/mul_2MulHsequential_56/batch_normalization_507/batchnorm/ReadVariableOp_1:value:07sequential_56/batch_normalization_507/batchnorm/mul:z:0*
T0*
_output_shapes
:AÆ
@sequential_56/batch_normalization_507/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_56_batch_normalization_507_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0ä
3sequential_56/batch_normalization_507/batchnorm/subSubHsequential_56/batch_normalization_507/batchnorm/ReadVariableOp_2:value:09sequential_56/batch_normalization_507/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aä
5sequential_56/batch_normalization_507/batchnorm/add_1AddV29sequential_56/batch_normalization_507/batchnorm/mul_1:z:07sequential_56/batch_normalization_507/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¨
'sequential_56/leaky_re_lu_507/LeakyRelu	LeakyRelu9sequential_56/batch_normalization_507/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>¤
-sequential_56/dense_564/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_564_matmul_readvariableop_resource*
_output_shapes

:A*
dtype0È
sequential_56/dense_564/MatMulMatMul5sequential_56/leaky_re_lu_507/LeakyRelu:activations:05sequential_56/dense_564/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_56/dense_564/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_564_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_56/dense_564/BiasAddBiasAdd(sequential_56/dense_564/MatMul:product:06sequential_56/dense_564/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_56/batch_normalization_508/batchnorm/ReadVariableOpReadVariableOpGsequential_56_batch_normalization_508_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_56/batch_normalization_508/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_56/batch_normalization_508/batchnorm/addAddV2Fsequential_56/batch_normalization_508/batchnorm/ReadVariableOp:value:0>sequential_56/batch_normalization_508/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_56/batch_normalization_508/batchnorm/RsqrtRsqrt7sequential_56/batch_normalization_508/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_56/batch_normalization_508/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_56_batch_normalization_508_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_56/batch_normalization_508/batchnorm/mulMul9sequential_56/batch_normalization_508/batchnorm/Rsqrt:y:0Jsequential_56/batch_normalization_508/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ñ
5sequential_56/batch_normalization_508/batchnorm/mul_1Mul(sequential_56/dense_564/BiasAdd:output:07sequential_56/batch_normalization_508/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_56/batch_normalization_508/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_56_batch_normalization_508_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_56/batch_normalization_508/batchnorm/mul_2MulHsequential_56/batch_normalization_508/batchnorm/ReadVariableOp_1:value:07sequential_56/batch_normalization_508/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_56/batch_normalization_508/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_56_batch_normalization_508_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_56/batch_normalization_508/batchnorm/subSubHsequential_56/batch_normalization_508/batchnorm/ReadVariableOp_2:value:09sequential_56/batch_normalization_508/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_56/batch_normalization_508/batchnorm/add_1AddV29sequential_56/batch_normalization_508/batchnorm/mul_1:z:07sequential_56/batch_normalization_508/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_56/leaky_re_lu_508/LeakyRelu	LeakyRelu9sequential_56/batch_normalization_508/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¤
-sequential_56/dense_565/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_565_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
sequential_56/dense_565/MatMulMatMul5sequential_56/leaky_re_lu_508/LeakyRelu:activations:05sequential_56/dense_565/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_56/dense_565/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_565_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_56/dense_565/BiasAddBiasAdd(sequential_56/dense_565/MatMul:product:06sequential_56/dense_565/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_56/batch_normalization_509/batchnorm/ReadVariableOpReadVariableOpGsequential_56_batch_normalization_509_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_56/batch_normalization_509/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_56/batch_normalization_509/batchnorm/addAddV2Fsequential_56/batch_normalization_509/batchnorm/ReadVariableOp:value:0>sequential_56/batch_normalization_509/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_56/batch_normalization_509/batchnorm/RsqrtRsqrt7sequential_56/batch_normalization_509/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_56/batch_normalization_509/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_56_batch_normalization_509_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_56/batch_normalization_509/batchnorm/mulMul9sequential_56/batch_normalization_509/batchnorm/Rsqrt:y:0Jsequential_56/batch_normalization_509/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ñ
5sequential_56/batch_normalization_509/batchnorm/mul_1Mul(sequential_56/dense_565/BiasAdd:output:07sequential_56/batch_normalization_509/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_56/batch_normalization_509/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_56_batch_normalization_509_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_56/batch_normalization_509/batchnorm/mul_2MulHsequential_56/batch_normalization_509/batchnorm/ReadVariableOp_1:value:07sequential_56/batch_normalization_509/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_56/batch_normalization_509/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_56_batch_normalization_509_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_56/batch_normalization_509/batchnorm/subSubHsequential_56/batch_normalization_509/batchnorm/ReadVariableOp_2:value:09sequential_56/batch_normalization_509/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_56/batch_normalization_509/batchnorm/add_1AddV29sequential_56/batch_normalization_509/batchnorm/mul_1:z:07sequential_56/batch_normalization_509/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_56/leaky_re_lu_509/LeakyRelu	LeakyRelu9sequential_56/batch_normalization_509/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¤
-sequential_56/dense_566/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_566_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
sequential_56/dense_566/MatMulMatMul5sequential_56/leaky_re_lu_509/LeakyRelu:activations:05sequential_56/dense_566/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_56/dense_566/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_566_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_56/dense_566/BiasAddBiasAdd(sequential_56/dense_566/MatMul:product:06sequential_56/dense_566/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_56/batch_normalization_510/batchnorm/ReadVariableOpReadVariableOpGsequential_56_batch_normalization_510_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_56/batch_normalization_510/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_56/batch_normalization_510/batchnorm/addAddV2Fsequential_56/batch_normalization_510/batchnorm/ReadVariableOp:value:0>sequential_56/batch_normalization_510/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_56/batch_normalization_510/batchnorm/RsqrtRsqrt7sequential_56/batch_normalization_510/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_56/batch_normalization_510/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_56_batch_normalization_510_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_56/batch_normalization_510/batchnorm/mulMul9sequential_56/batch_normalization_510/batchnorm/Rsqrt:y:0Jsequential_56/batch_normalization_510/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ñ
5sequential_56/batch_normalization_510/batchnorm/mul_1Mul(sequential_56/dense_566/BiasAdd:output:07sequential_56/batch_normalization_510/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_56/batch_normalization_510/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_56_batch_normalization_510_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_56/batch_normalization_510/batchnorm/mul_2MulHsequential_56/batch_normalization_510/batchnorm/ReadVariableOp_1:value:07sequential_56/batch_normalization_510/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_56/batch_normalization_510/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_56_batch_normalization_510_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_56/batch_normalization_510/batchnorm/subSubHsequential_56/batch_normalization_510/batchnorm/ReadVariableOp_2:value:09sequential_56/batch_normalization_510/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_56/batch_normalization_510/batchnorm/add_1AddV29sequential_56/batch_normalization_510/batchnorm/mul_1:z:07sequential_56/batch_normalization_510/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_56/leaky_re_lu_510/LeakyRelu	LeakyRelu9sequential_56/batch_normalization_510/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¤
-sequential_56/dense_567/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_567_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0È
sequential_56/dense_567/MatMulMatMul5sequential_56/leaky_re_lu_510/LeakyRelu:activations:05sequential_56/dense_567/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¢
.sequential_56/dense_567/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_567_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0¾
sequential_56/dense_567/BiasAddBiasAdd(sequential_56/dense_567/MatMul:product:06sequential_56/dense_567/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Â
>sequential_56/batch_normalization_511/batchnorm/ReadVariableOpReadVariableOpGsequential_56_batch_normalization_511_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0z
5sequential_56/batch_normalization_511/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_56/batch_normalization_511/batchnorm/addAddV2Fsequential_56/batch_normalization_511/batchnorm/ReadVariableOp:value:0>sequential_56/batch_normalization_511/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
5sequential_56/batch_normalization_511/batchnorm/RsqrtRsqrt7sequential_56/batch_normalization_511/batchnorm/add:z:0*
T0*
_output_shapes
:5Ê
Bsequential_56/batch_normalization_511/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_56_batch_normalization_511_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0æ
3sequential_56/batch_normalization_511/batchnorm/mulMul9sequential_56/batch_normalization_511/batchnorm/Rsqrt:y:0Jsequential_56/batch_normalization_511/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5Ñ
5sequential_56/batch_normalization_511/batchnorm/mul_1Mul(sequential_56/dense_567/BiasAdd:output:07sequential_56/batch_normalization_511/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Æ
@sequential_56/batch_normalization_511/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_56_batch_normalization_511_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0ä
5sequential_56/batch_normalization_511/batchnorm/mul_2MulHsequential_56/batch_normalization_511/batchnorm/ReadVariableOp_1:value:07sequential_56/batch_normalization_511/batchnorm/mul:z:0*
T0*
_output_shapes
:5Æ
@sequential_56/batch_normalization_511/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_56_batch_normalization_511_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0ä
3sequential_56/batch_normalization_511/batchnorm/subSubHsequential_56/batch_normalization_511/batchnorm/ReadVariableOp_2:value:09sequential_56/batch_normalization_511/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5ä
5sequential_56/batch_normalization_511/batchnorm/add_1AddV29sequential_56/batch_normalization_511/batchnorm/mul_1:z:07sequential_56/batch_normalization_511/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¨
'sequential_56/leaky_re_lu_511/LeakyRelu	LeakyRelu9sequential_56/batch_normalization_511/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>¤
-sequential_56/dense_568/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_568_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0È
sequential_56/dense_568/MatMulMatMul5sequential_56/leaky_re_lu_511/LeakyRelu:activations:05sequential_56/dense_568/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¢
.sequential_56/dense_568/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_568_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0¾
sequential_56/dense_568/BiasAddBiasAdd(sequential_56/dense_568/MatMul:product:06sequential_56/dense_568/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Â
>sequential_56/batch_normalization_512/batchnorm/ReadVariableOpReadVariableOpGsequential_56_batch_normalization_512_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0z
5sequential_56/batch_normalization_512/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_56/batch_normalization_512/batchnorm/addAddV2Fsequential_56/batch_normalization_512/batchnorm/ReadVariableOp:value:0>sequential_56/batch_normalization_512/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
5sequential_56/batch_normalization_512/batchnorm/RsqrtRsqrt7sequential_56/batch_normalization_512/batchnorm/add:z:0*
T0*
_output_shapes
:5Ê
Bsequential_56/batch_normalization_512/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_56_batch_normalization_512_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0æ
3sequential_56/batch_normalization_512/batchnorm/mulMul9sequential_56/batch_normalization_512/batchnorm/Rsqrt:y:0Jsequential_56/batch_normalization_512/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5Ñ
5sequential_56/batch_normalization_512/batchnorm/mul_1Mul(sequential_56/dense_568/BiasAdd:output:07sequential_56/batch_normalization_512/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Æ
@sequential_56/batch_normalization_512/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_56_batch_normalization_512_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0ä
5sequential_56/batch_normalization_512/batchnorm/mul_2MulHsequential_56/batch_normalization_512/batchnorm/ReadVariableOp_1:value:07sequential_56/batch_normalization_512/batchnorm/mul:z:0*
T0*
_output_shapes
:5Æ
@sequential_56/batch_normalization_512/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_56_batch_normalization_512_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0ä
3sequential_56/batch_normalization_512/batchnorm/subSubHsequential_56/batch_normalization_512/batchnorm/ReadVariableOp_2:value:09sequential_56/batch_normalization_512/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5ä
5sequential_56/batch_normalization_512/batchnorm/add_1AddV29sequential_56/batch_normalization_512/batchnorm/mul_1:z:07sequential_56/batch_normalization_512/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¨
'sequential_56/leaky_re_lu_512/LeakyRelu	LeakyRelu9sequential_56/batch_normalization_512/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>¤
-sequential_56/dense_569/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_569_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0È
sequential_56/dense_569/MatMulMatMul5sequential_56/leaky_re_lu_512/LeakyRelu:activations:05sequential_56/dense_569/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¢
.sequential_56/dense_569/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_569_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0¾
sequential_56/dense_569/BiasAddBiasAdd(sequential_56/dense_569/MatMul:product:06sequential_56/dense_569/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Â
>sequential_56/batch_normalization_513/batchnorm/ReadVariableOpReadVariableOpGsequential_56_batch_normalization_513_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0z
5sequential_56/batch_normalization_513/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_56/batch_normalization_513/batchnorm/addAddV2Fsequential_56/batch_normalization_513/batchnorm/ReadVariableOp:value:0>sequential_56/batch_normalization_513/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
5sequential_56/batch_normalization_513/batchnorm/RsqrtRsqrt7sequential_56/batch_normalization_513/batchnorm/add:z:0*
T0*
_output_shapes
:5Ê
Bsequential_56/batch_normalization_513/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_56_batch_normalization_513_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0æ
3sequential_56/batch_normalization_513/batchnorm/mulMul9sequential_56/batch_normalization_513/batchnorm/Rsqrt:y:0Jsequential_56/batch_normalization_513/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5Ñ
5sequential_56/batch_normalization_513/batchnorm/mul_1Mul(sequential_56/dense_569/BiasAdd:output:07sequential_56/batch_normalization_513/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Æ
@sequential_56/batch_normalization_513/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_56_batch_normalization_513_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0ä
5sequential_56/batch_normalization_513/batchnorm/mul_2MulHsequential_56/batch_normalization_513/batchnorm/ReadVariableOp_1:value:07sequential_56/batch_normalization_513/batchnorm/mul:z:0*
T0*
_output_shapes
:5Æ
@sequential_56/batch_normalization_513/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_56_batch_normalization_513_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0ä
3sequential_56/batch_normalization_513/batchnorm/subSubHsequential_56/batch_normalization_513/batchnorm/ReadVariableOp_2:value:09sequential_56/batch_normalization_513/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5ä
5sequential_56/batch_normalization_513/batchnorm/add_1AddV29sequential_56/batch_normalization_513/batchnorm/mul_1:z:07sequential_56/batch_normalization_513/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¨
'sequential_56/leaky_re_lu_513/LeakyRelu	LeakyRelu9sequential_56/batch_normalization_513/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>¤
-sequential_56/dense_570/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_570_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0È
sequential_56/dense_570/MatMulMatMul5sequential_56/leaky_re_lu_513/LeakyRelu:activations:05sequential_56/dense_570/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¢
.sequential_56/dense_570/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_570_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0¾
sequential_56/dense_570/BiasAddBiasAdd(sequential_56/dense_570/MatMul:product:06sequential_56/dense_570/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Â
>sequential_56/batch_normalization_514/batchnorm/ReadVariableOpReadVariableOpGsequential_56_batch_normalization_514_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0z
5sequential_56/batch_normalization_514/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_56/batch_normalization_514/batchnorm/addAddV2Fsequential_56/batch_normalization_514/batchnorm/ReadVariableOp:value:0>sequential_56/batch_normalization_514/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
5sequential_56/batch_normalization_514/batchnorm/RsqrtRsqrt7sequential_56/batch_normalization_514/batchnorm/add:z:0*
T0*
_output_shapes
:5Ê
Bsequential_56/batch_normalization_514/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_56_batch_normalization_514_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0æ
3sequential_56/batch_normalization_514/batchnorm/mulMul9sequential_56/batch_normalization_514/batchnorm/Rsqrt:y:0Jsequential_56/batch_normalization_514/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5Ñ
5sequential_56/batch_normalization_514/batchnorm/mul_1Mul(sequential_56/dense_570/BiasAdd:output:07sequential_56/batch_normalization_514/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Æ
@sequential_56/batch_normalization_514/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_56_batch_normalization_514_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0ä
5sequential_56/batch_normalization_514/batchnorm/mul_2MulHsequential_56/batch_normalization_514/batchnorm/ReadVariableOp_1:value:07sequential_56/batch_normalization_514/batchnorm/mul:z:0*
T0*
_output_shapes
:5Æ
@sequential_56/batch_normalization_514/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_56_batch_normalization_514_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0ä
3sequential_56/batch_normalization_514/batchnorm/subSubHsequential_56/batch_normalization_514/batchnorm/ReadVariableOp_2:value:09sequential_56/batch_normalization_514/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5ä
5sequential_56/batch_normalization_514/batchnorm/add_1AddV29sequential_56/batch_normalization_514/batchnorm/mul_1:z:07sequential_56/batch_normalization_514/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¨
'sequential_56/leaky_re_lu_514/LeakyRelu	LeakyRelu9sequential_56/batch_normalization_514/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>¤
-sequential_56/dense_571/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_571_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0È
sequential_56/dense_571/MatMulMatMul5sequential_56/leaky_re_lu_514/LeakyRelu:activations:05sequential_56/dense_571/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¢
.sequential_56/dense_571/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_571_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0¾
sequential_56/dense_571/BiasAddBiasAdd(sequential_56/dense_571/MatMul:product:06sequential_56/dense_571/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Â
>sequential_56/batch_normalization_515/batchnorm/ReadVariableOpReadVariableOpGsequential_56_batch_normalization_515_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0z
5sequential_56/batch_normalization_515/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_56/batch_normalization_515/batchnorm/addAddV2Fsequential_56/batch_normalization_515/batchnorm/ReadVariableOp:value:0>sequential_56/batch_normalization_515/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
5sequential_56/batch_normalization_515/batchnorm/RsqrtRsqrt7sequential_56/batch_normalization_515/batchnorm/add:z:0*
T0*
_output_shapes
:5Ê
Bsequential_56/batch_normalization_515/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_56_batch_normalization_515_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0æ
3sequential_56/batch_normalization_515/batchnorm/mulMul9sequential_56/batch_normalization_515/batchnorm/Rsqrt:y:0Jsequential_56/batch_normalization_515/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5Ñ
5sequential_56/batch_normalization_515/batchnorm/mul_1Mul(sequential_56/dense_571/BiasAdd:output:07sequential_56/batch_normalization_515/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Æ
@sequential_56/batch_normalization_515/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_56_batch_normalization_515_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0ä
5sequential_56/batch_normalization_515/batchnorm/mul_2MulHsequential_56/batch_normalization_515/batchnorm/ReadVariableOp_1:value:07sequential_56/batch_normalization_515/batchnorm/mul:z:0*
T0*
_output_shapes
:5Æ
@sequential_56/batch_normalization_515/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_56_batch_normalization_515_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0ä
3sequential_56/batch_normalization_515/batchnorm/subSubHsequential_56/batch_normalization_515/batchnorm/ReadVariableOp_2:value:09sequential_56/batch_normalization_515/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5ä
5sequential_56/batch_normalization_515/batchnorm/add_1AddV29sequential_56/batch_normalization_515/batchnorm/mul_1:z:07sequential_56/batch_normalization_515/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¨
'sequential_56/leaky_re_lu_515/LeakyRelu	LeakyRelu9sequential_56/batch_normalization_515/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>¤
-sequential_56/dense_572/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_572_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0È
sequential_56/dense_572/MatMulMatMul5sequential_56/leaky_re_lu_515/LeakyRelu:activations:05sequential_56/dense_572/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_56/dense_572/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_572_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_56/dense_572/BiasAddBiasAdd(sequential_56/dense_572/MatMul:product:06sequential_56/dense_572/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_56/dense_572/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ#
NoOpNoOp?^sequential_56/batch_normalization_504/batchnorm/ReadVariableOpA^sequential_56/batch_normalization_504/batchnorm/ReadVariableOp_1A^sequential_56/batch_normalization_504/batchnorm/ReadVariableOp_2C^sequential_56/batch_normalization_504/batchnorm/mul/ReadVariableOp?^sequential_56/batch_normalization_505/batchnorm/ReadVariableOpA^sequential_56/batch_normalization_505/batchnorm/ReadVariableOp_1A^sequential_56/batch_normalization_505/batchnorm/ReadVariableOp_2C^sequential_56/batch_normalization_505/batchnorm/mul/ReadVariableOp?^sequential_56/batch_normalization_506/batchnorm/ReadVariableOpA^sequential_56/batch_normalization_506/batchnorm/ReadVariableOp_1A^sequential_56/batch_normalization_506/batchnorm/ReadVariableOp_2C^sequential_56/batch_normalization_506/batchnorm/mul/ReadVariableOp?^sequential_56/batch_normalization_507/batchnorm/ReadVariableOpA^sequential_56/batch_normalization_507/batchnorm/ReadVariableOp_1A^sequential_56/batch_normalization_507/batchnorm/ReadVariableOp_2C^sequential_56/batch_normalization_507/batchnorm/mul/ReadVariableOp?^sequential_56/batch_normalization_508/batchnorm/ReadVariableOpA^sequential_56/batch_normalization_508/batchnorm/ReadVariableOp_1A^sequential_56/batch_normalization_508/batchnorm/ReadVariableOp_2C^sequential_56/batch_normalization_508/batchnorm/mul/ReadVariableOp?^sequential_56/batch_normalization_509/batchnorm/ReadVariableOpA^sequential_56/batch_normalization_509/batchnorm/ReadVariableOp_1A^sequential_56/batch_normalization_509/batchnorm/ReadVariableOp_2C^sequential_56/batch_normalization_509/batchnorm/mul/ReadVariableOp?^sequential_56/batch_normalization_510/batchnorm/ReadVariableOpA^sequential_56/batch_normalization_510/batchnorm/ReadVariableOp_1A^sequential_56/batch_normalization_510/batchnorm/ReadVariableOp_2C^sequential_56/batch_normalization_510/batchnorm/mul/ReadVariableOp?^sequential_56/batch_normalization_511/batchnorm/ReadVariableOpA^sequential_56/batch_normalization_511/batchnorm/ReadVariableOp_1A^sequential_56/batch_normalization_511/batchnorm/ReadVariableOp_2C^sequential_56/batch_normalization_511/batchnorm/mul/ReadVariableOp?^sequential_56/batch_normalization_512/batchnorm/ReadVariableOpA^sequential_56/batch_normalization_512/batchnorm/ReadVariableOp_1A^sequential_56/batch_normalization_512/batchnorm/ReadVariableOp_2C^sequential_56/batch_normalization_512/batchnorm/mul/ReadVariableOp?^sequential_56/batch_normalization_513/batchnorm/ReadVariableOpA^sequential_56/batch_normalization_513/batchnorm/ReadVariableOp_1A^sequential_56/batch_normalization_513/batchnorm/ReadVariableOp_2C^sequential_56/batch_normalization_513/batchnorm/mul/ReadVariableOp?^sequential_56/batch_normalization_514/batchnorm/ReadVariableOpA^sequential_56/batch_normalization_514/batchnorm/ReadVariableOp_1A^sequential_56/batch_normalization_514/batchnorm/ReadVariableOp_2C^sequential_56/batch_normalization_514/batchnorm/mul/ReadVariableOp?^sequential_56/batch_normalization_515/batchnorm/ReadVariableOpA^sequential_56/batch_normalization_515/batchnorm/ReadVariableOp_1A^sequential_56/batch_normalization_515/batchnorm/ReadVariableOp_2C^sequential_56/batch_normalization_515/batchnorm/mul/ReadVariableOp/^sequential_56/dense_560/BiasAdd/ReadVariableOp.^sequential_56/dense_560/MatMul/ReadVariableOp/^sequential_56/dense_561/BiasAdd/ReadVariableOp.^sequential_56/dense_561/MatMul/ReadVariableOp/^sequential_56/dense_562/BiasAdd/ReadVariableOp.^sequential_56/dense_562/MatMul/ReadVariableOp/^sequential_56/dense_563/BiasAdd/ReadVariableOp.^sequential_56/dense_563/MatMul/ReadVariableOp/^sequential_56/dense_564/BiasAdd/ReadVariableOp.^sequential_56/dense_564/MatMul/ReadVariableOp/^sequential_56/dense_565/BiasAdd/ReadVariableOp.^sequential_56/dense_565/MatMul/ReadVariableOp/^sequential_56/dense_566/BiasAdd/ReadVariableOp.^sequential_56/dense_566/MatMul/ReadVariableOp/^sequential_56/dense_567/BiasAdd/ReadVariableOp.^sequential_56/dense_567/MatMul/ReadVariableOp/^sequential_56/dense_568/BiasAdd/ReadVariableOp.^sequential_56/dense_568/MatMul/ReadVariableOp/^sequential_56/dense_569/BiasAdd/ReadVariableOp.^sequential_56/dense_569/MatMul/ReadVariableOp/^sequential_56/dense_570/BiasAdd/ReadVariableOp.^sequential_56/dense_570/MatMul/ReadVariableOp/^sequential_56/dense_571/BiasAdd/ReadVariableOp.^sequential_56/dense_571/MatMul/ReadVariableOp/^sequential_56/dense_572/BiasAdd/ReadVariableOp.^sequential_56/dense_572/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ð
_input_shapes¾
»:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>sequential_56/batch_normalization_504/batchnorm/ReadVariableOp>sequential_56/batch_normalization_504/batchnorm/ReadVariableOp2
@sequential_56/batch_normalization_504/batchnorm/ReadVariableOp_1@sequential_56/batch_normalization_504/batchnorm/ReadVariableOp_12
@sequential_56/batch_normalization_504/batchnorm/ReadVariableOp_2@sequential_56/batch_normalization_504/batchnorm/ReadVariableOp_22
Bsequential_56/batch_normalization_504/batchnorm/mul/ReadVariableOpBsequential_56/batch_normalization_504/batchnorm/mul/ReadVariableOp2
>sequential_56/batch_normalization_505/batchnorm/ReadVariableOp>sequential_56/batch_normalization_505/batchnorm/ReadVariableOp2
@sequential_56/batch_normalization_505/batchnorm/ReadVariableOp_1@sequential_56/batch_normalization_505/batchnorm/ReadVariableOp_12
@sequential_56/batch_normalization_505/batchnorm/ReadVariableOp_2@sequential_56/batch_normalization_505/batchnorm/ReadVariableOp_22
Bsequential_56/batch_normalization_505/batchnorm/mul/ReadVariableOpBsequential_56/batch_normalization_505/batchnorm/mul/ReadVariableOp2
>sequential_56/batch_normalization_506/batchnorm/ReadVariableOp>sequential_56/batch_normalization_506/batchnorm/ReadVariableOp2
@sequential_56/batch_normalization_506/batchnorm/ReadVariableOp_1@sequential_56/batch_normalization_506/batchnorm/ReadVariableOp_12
@sequential_56/batch_normalization_506/batchnorm/ReadVariableOp_2@sequential_56/batch_normalization_506/batchnorm/ReadVariableOp_22
Bsequential_56/batch_normalization_506/batchnorm/mul/ReadVariableOpBsequential_56/batch_normalization_506/batchnorm/mul/ReadVariableOp2
>sequential_56/batch_normalization_507/batchnorm/ReadVariableOp>sequential_56/batch_normalization_507/batchnorm/ReadVariableOp2
@sequential_56/batch_normalization_507/batchnorm/ReadVariableOp_1@sequential_56/batch_normalization_507/batchnorm/ReadVariableOp_12
@sequential_56/batch_normalization_507/batchnorm/ReadVariableOp_2@sequential_56/batch_normalization_507/batchnorm/ReadVariableOp_22
Bsequential_56/batch_normalization_507/batchnorm/mul/ReadVariableOpBsequential_56/batch_normalization_507/batchnorm/mul/ReadVariableOp2
>sequential_56/batch_normalization_508/batchnorm/ReadVariableOp>sequential_56/batch_normalization_508/batchnorm/ReadVariableOp2
@sequential_56/batch_normalization_508/batchnorm/ReadVariableOp_1@sequential_56/batch_normalization_508/batchnorm/ReadVariableOp_12
@sequential_56/batch_normalization_508/batchnorm/ReadVariableOp_2@sequential_56/batch_normalization_508/batchnorm/ReadVariableOp_22
Bsequential_56/batch_normalization_508/batchnorm/mul/ReadVariableOpBsequential_56/batch_normalization_508/batchnorm/mul/ReadVariableOp2
>sequential_56/batch_normalization_509/batchnorm/ReadVariableOp>sequential_56/batch_normalization_509/batchnorm/ReadVariableOp2
@sequential_56/batch_normalization_509/batchnorm/ReadVariableOp_1@sequential_56/batch_normalization_509/batchnorm/ReadVariableOp_12
@sequential_56/batch_normalization_509/batchnorm/ReadVariableOp_2@sequential_56/batch_normalization_509/batchnorm/ReadVariableOp_22
Bsequential_56/batch_normalization_509/batchnorm/mul/ReadVariableOpBsequential_56/batch_normalization_509/batchnorm/mul/ReadVariableOp2
>sequential_56/batch_normalization_510/batchnorm/ReadVariableOp>sequential_56/batch_normalization_510/batchnorm/ReadVariableOp2
@sequential_56/batch_normalization_510/batchnorm/ReadVariableOp_1@sequential_56/batch_normalization_510/batchnorm/ReadVariableOp_12
@sequential_56/batch_normalization_510/batchnorm/ReadVariableOp_2@sequential_56/batch_normalization_510/batchnorm/ReadVariableOp_22
Bsequential_56/batch_normalization_510/batchnorm/mul/ReadVariableOpBsequential_56/batch_normalization_510/batchnorm/mul/ReadVariableOp2
>sequential_56/batch_normalization_511/batchnorm/ReadVariableOp>sequential_56/batch_normalization_511/batchnorm/ReadVariableOp2
@sequential_56/batch_normalization_511/batchnorm/ReadVariableOp_1@sequential_56/batch_normalization_511/batchnorm/ReadVariableOp_12
@sequential_56/batch_normalization_511/batchnorm/ReadVariableOp_2@sequential_56/batch_normalization_511/batchnorm/ReadVariableOp_22
Bsequential_56/batch_normalization_511/batchnorm/mul/ReadVariableOpBsequential_56/batch_normalization_511/batchnorm/mul/ReadVariableOp2
>sequential_56/batch_normalization_512/batchnorm/ReadVariableOp>sequential_56/batch_normalization_512/batchnorm/ReadVariableOp2
@sequential_56/batch_normalization_512/batchnorm/ReadVariableOp_1@sequential_56/batch_normalization_512/batchnorm/ReadVariableOp_12
@sequential_56/batch_normalization_512/batchnorm/ReadVariableOp_2@sequential_56/batch_normalization_512/batchnorm/ReadVariableOp_22
Bsequential_56/batch_normalization_512/batchnorm/mul/ReadVariableOpBsequential_56/batch_normalization_512/batchnorm/mul/ReadVariableOp2
>sequential_56/batch_normalization_513/batchnorm/ReadVariableOp>sequential_56/batch_normalization_513/batchnorm/ReadVariableOp2
@sequential_56/batch_normalization_513/batchnorm/ReadVariableOp_1@sequential_56/batch_normalization_513/batchnorm/ReadVariableOp_12
@sequential_56/batch_normalization_513/batchnorm/ReadVariableOp_2@sequential_56/batch_normalization_513/batchnorm/ReadVariableOp_22
Bsequential_56/batch_normalization_513/batchnorm/mul/ReadVariableOpBsequential_56/batch_normalization_513/batchnorm/mul/ReadVariableOp2
>sequential_56/batch_normalization_514/batchnorm/ReadVariableOp>sequential_56/batch_normalization_514/batchnorm/ReadVariableOp2
@sequential_56/batch_normalization_514/batchnorm/ReadVariableOp_1@sequential_56/batch_normalization_514/batchnorm/ReadVariableOp_12
@sequential_56/batch_normalization_514/batchnorm/ReadVariableOp_2@sequential_56/batch_normalization_514/batchnorm/ReadVariableOp_22
Bsequential_56/batch_normalization_514/batchnorm/mul/ReadVariableOpBsequential_56/batch_normalization_514/batchnorm/mul/ReadVariableOp2
>sequential_56/batch_normalization_515/batchnorm/ReadVariableOp>sequential_56/batch_normalization_515/batchnorm/ReadVariableOp2
@sequential_56/batch_normalization_515/batchnorm/ReadVariableOp_1@sequential_56/batch_normalization_515/batchnorm/ReadVariableOp_12
@sequential_56/batch_normalization_515/batchnorm/ReadVariableOp_2@sequential_56/batch_normalization_515/batchnorm/ReadVariableOp_22
Bsequential_56/batch_normalization_515/batchnorm/mul/ReadVariableOpBsequential_56/batch_normalization_515/batchnorm/mul/ReadVariableOp2`
.sequential_56/dense_560/BiasAdd/ReadVariableOp.sequential_56/dense_560/BiasAdd/ReadVariableOp2^
-sequential_56/dense_560/MatMul/ReadVariableOp-sequential_56/dense_560/MatMul/ReadVariableOp2`
.sequential_56/dense_561/BiasAdd/ReadVariableOp.sequential_56/dense_561/BiasAdd/ReadVariableOp2^
-sequential_56/dense_561/MatMul/ReadVariableOp-sequential_56/dense_561/MatMul/ReadVariableOp2`
.sequential_56/dense_562/BiasAdd/ReadVariableOp.sequential_56/dense_562/BiasAdd/ReadVariableOp2^
-sequential_56/dense_562/MatMul/ReadVariableOp-sequential_56/dense_562/MatMul/ReadVariableOp2`
.sequential_56/dense_563/BiasAdd/ReadVariableOp.sequential_56/dense_563/BiasAdd/ReadVariableOp2^
-sequential_56/dense_563/MatMul/ReadVariableOp-sequential_56/dense_563/MatMul/ReadVariableOp2`
.sequential_56/dense_564/BiasAdd/ReadVariableOp.sequential_56/dense_564/BiasAdd/ReadVariableOp2^
-sequential_56/dense_564/MatMul/ReadVariableOp-sequential_56/dense_564/MatMul/ReadVariableOp2`
.sequential_56/dense_565/BiasAdd/ReadVariableOp.sequential_56/dense_565/BiasAdd/ReadVariableOp2^
-sequential_56/dense_565/MatMul/ReadVariableOp-sequential_56/dense_565/MatMul/ReadVariableOp2`
.sequential_56/dense_566/BiasAdd/ReadVariableOp.sequential_56/dense_566/BiasAdd/ReadVariableOp2^
-sequential_56/dense_566/MatMul/ReadVariableOp-sequential_56/dense_566/MatMul/ReadVariableOp2`
.sequential_56/dense_567/BiasAdd/ReadVariableOp.sequential_56/dense_567/BiasAdd/ReadVariableOp2^
-sequential_56/dense_567/MatMul/ReadVariableOp-sequential_56/dense_567/MatMul/ReadVariableOp2`
.sequential_56/dense_568/BiasAdd/ReadVariableOp.sequential_56/dense_568/BiasAdd/ReadVariableOp2^
-sequential_56/dense_568/MatMul/ReadVariableOp-sequential_56/dense_568/MatMul/ReadVariableOp2`
.sequential_56/dense_569/BiasAdd/ReadVariableOp.sequential_56/dense_569/BiasAdd/ReadVariableOp2^
-sequential_56/dense_569/MatMul/ReadVariableOp-sequential_56/dense_569/MatMul/ReadVariableOp2`
.sequential_56/dense_570/BiasAdd/ReadVariableOp.sequential_56/dense_570/BiasAdd/ReadVariableOp2^
-sequential_56/dense_570/MatMul/ReadVariableOp-sequential_56/dense_570/MatMul/ReadVariableOp2`
.sequential_56/dense_571/BiasAdd/ReadVariableOp.sequential_56/dense_571/BiasAdd/ReadVariableOp2^
-sequential_56/dense_571/MatMul/ReadVariableOp-sequential_56/dense_571/MatMul/ReadVariableOp2`
.sequential_56/dense_572/BiasAdd/ReadVariableOp.sequential_56/dense_572/BiasAdd/ReadVariableOp2^
-sequential_56/dense_572/MatMul/ReadVariableOp-sequential_56/dense_572/MatMul/ReadVariableOp:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_56_input:$ 

_output_shapes

::$ 

_output_shapes

:
Á 
³
/__inference_sequential_56_layer_call_fn_1012012

inputs
unknown
	unknown_0
	unknown_1:A
	unknown_2:A
	unknown_3:A
	unknown_4:A
	unknown_5:A
	unknown_6:A
	unknown_7:AA
	unknown_8:A
	unknown_9:A

unknown_10:A

unknown_11:A

unknown_12:A

unknown_13:AA

unknown_14:A

unknown_15:A

unknown_16:A

unknown_17:A

unknown_18:A

unknown_19:AA

unknown_20:A

unknown_21:A

unknown_22:A

unknown_23:A

unknown_24:A

unknown_25:A

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:5

unknown_44:5

unknown_45:5

unknown_46:5

unknown_47:5

unknown_48:5

unknown_49:55

unknown_50:5

unknown_51:5

unknown_52:5

unknown_53:5

unknown_54:5

unknown_55:55

unknown_56:5

unknown_57:5

unknown_58:5

unknown_59:5

unknown_60:5

unknown_61:55

unknown_62:5

unknown_63:5

unknown_64:5

unknown_65:5

unknown_66:5

unknown_67:55

unknown_68:5

unknown_69:5

unknown_70:5

unknown_71:5

unknown_72:5

unknown_73:5

unknown_74:
identity¢StatefulPartitionedCallÈ

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
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74*X
TinQ
O2M*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"%&'(+,-.1234789:=>?@CDEFIJKL*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_56_layer_call_and_return_conditional_losses_1010990o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ð
_input_shapes¾
»:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
­
M
1__inference_leaky_re_lu_513_layer_call_fn_1014057

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_513_layer_call_and_return_conditional_losses_1010195`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1009442

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_514_layer_call_fn_1014107

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1009770o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_515_layer_call_fn_1014203

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1009805o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
É	
÷
F__inference_dense_569_layer_call_and_return_conditional_losses_1013972

inputs0
matmul_readvariableop_resource:55-
biasadd_readvariableop_resource:5
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:55*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:5*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Æ

+__inference_dense_561_layer_call_fn_1013090

inputs
unknown:AA
	unknown_0:A
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_561_layer_call_and_return_conditional_losses_1009919o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_1013190

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿA:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_508_layer_call_and_return_conditional_losses_1013517

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1014236

inputs/
!batchnorm_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:51
#batchnorm_readvariableop_1_resource:51
#batchnorm_readvariableop_2_resource:5
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:5z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
É	
÷
F__inference_dense_560_layer_call_and_return_conditional_losses_1009887

inputs0
matmul_readvariableop_resource:A-
biasadd_readvariableop_resource:A
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:A*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É	
÷
F__inference_dense_570_layer_call_and_return_conditional_losses_1014081

inputs0
matmul_readvariableop_resource:55-
biasadd_readvariableop_resource:5
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:55*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:5*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1009149

inputs/
!batchnorm_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A1
#batchnorm_readvariableop_1_resource:A1
#batchnorm_readvariableop_2_resource:A
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_510_layer_call_and_return_conditional_losses_1010099

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_504_layer_call_fn_1013004

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1008903o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_510_layer_call_fn_1013730

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_510_layer_call_and_return_conditional_losses_1010099`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_1009971

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿA:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
É	
÷
F__inference_dense_565_layer_call_and_return_conditional_losses_1013536

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1009278

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_512_layer_call_fn_1013889

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1009606o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_513_layer_call_fn_1013985

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1009641o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
Æ

+__inference_dense_567_layer_call_fn_1013744

inputs
unknown:5
	unknown_0:5
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_567_layer_call_and_return_conditional_losses_1010111o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_dense_570_layer_call_fn_1014071

inputs
unknown:55
	unknown_0:5
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_570_layer_call_and_return_conditional_losses_1010207o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1009114

inputs5
'assignmovingavg_readvariableop_resource:A7
)assignmovingavg_1_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A/
!batchnorm_readvariableop_resource:A
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:A
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:A*
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
:A*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A¬
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
:A*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:A~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A´
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
Æ

+__inference_dense_569_layer_call_fn_1013962

inputs
unknown:55
	unknown_0:5
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_569_layer_call_and_return_conditional_losses_1010175o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1014270

inputs5
'assignmovingavg_readvariableop_resource:57
)assignmovingavg_1_readvariableop_resource:53
%batchnorm_mul_readvariableop_resource:5/
!batchnorm_readvariableop_resource:5
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:5
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:5*
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
:5*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:5x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5¬
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
:5*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:5~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5´
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
:5P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:5~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:5v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:5r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
É	
÷
F__inference_dense_560_layer_call_and_return_conditional_losses_1012991

inputs0
matmul_readvariableop_resource:A-
biasadd_readvariableop_resource:A
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:A*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É	
÷
F__inference_dense_562_layer_call_and_return_conditional_losses_1009951

inputs0
matmul_readvariableop_resource:AA-
biasadd_readvariableop_resource:A
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:AA*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
øã
N
J__inference_sequential_56_layer_call_and_return_conditional_losses_1012766

inputs
normalization_56_sub_y
normalization_56_sqrt_x:
(dense_560_matmul_readvariableop_resource:A7
)dense_560_biasadd_readvariableop_resource:AM
?batch_normalization_504_assignmovingavg_readvariableop_resource:AO
Abatch_normalization_504_assignmovingavg_1_readvariableop_resource:AK
=batch_normalization_504_batchnorm_mul_readvariableop_resource:AG
9batch_normalization_504_batchnorm_readvariableop_resource:A:
(dense_561_matmul_readvariableop_resource:AA7
)dense_561_biasadd_readvariableop_resource:AM
?batch_normalization_505_assignmovingavg_readvariableop_resource:AO
Abatch_normalization_505_assignmovingavg_1_readvariableop_resource:AK
=batch_normalization_505_batchnorm_mul_readvariableop_resource:AG
9batch_normalization_505_batchnorm_readvariableop_resource:A:
(dense_562_matmul_readvariableop_resource:AA7
)dense_562_biasadd_readvariableop_resource:AM
?batch_normalization_506_assignmovingavg_readvariableop_resource:AO
Abatch_normalization_506_assignmovingavg_1_readvariableop_resource:AK
=batch_normalization_506_batchnorm_mul_readvariableop_resource:AG
9batch_normalization_506_batchnorm_readvariableop_resource:A:
(dense_563_matmul_readvariableop_resource:AA7
)dense_563_biasadd_readvariableop_resource:AM
?batch_normalization_507_assignmovingavg_readvariableop_resource:AO
Abatch_normalization_507_assignmovingavg_1_readvariableop_resource:AK
=batch_normalization_507_batchnorm_mul_readvariableop_resource:AG
9batch_normalization_507_batchnorm_readvariableop_resource:A:
(dense_564_matmul_readvariableop_resource:A7
)dense_564_biasadd_readvariableop_resource:M
?batch_normalization_508_assignmovingavg_readvariableop_resource:O
Abatch_normalization_508_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_508_batchnorm_mul_readvariableop_resource:G
9batch_normalization_508_batchnorm_readvariableop_resource::
(dense_565_matmul_readvariableop_resource:7
)dense_565_biasadd_readvariableop_resource:M
?batch_normalization_509_assignmovingavg_readvariableop_resource:O
Abatch_normalization_509_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_509_batchnorm_mul_readvariableop_resource:G
9batch_normalization_509_batchnorm_readvariableop_resource::
(dense_566_matmul_readvariableop_resource:7
)dense_566_biasadd_readvariableop_resource:M
?batch_normalization_510_assignmovingavg_readvariableop_resource:O
Abatch_normalization_510_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_510_batchnorm_mul_readvariableop_resource:G
9batch_normalization_510_batchnorm_readvariableop_resource::
(dense_567_matmul_readvariableop_resource:57
)dense_567_biasadd_readvariableop_resource:5M
?batch_normalization_511_assignmovingavg_readvariableop_resource:5O
Abatch_normalization_511_assignmovingavg_1_readvariableop_resource:5K
=batch_normalization_511_batchnorm_mul_readvariableop_resource:5G
9batch_normalization_511_batchnorm_readvariableop_resource:5:
(dense_568_matmul_readvariableop_resource:557
)dense_568_biasadd_readvariableop_resource:5M
?batch_normalization_512_assignmovingavg_readvariableop_resource:5O
Abatch_normalization_512_assignmovingavg_1_readvariableop_resource:5K
=batch_normalization_512_batchnorm_mul_readvariableop_resource:5G
9batch_normalization_512_batchnorm_readvariableop_resource:5:
(dense_569_matmul_readvariableop_resource:557
)dense_569_biasadd_readvariableop_resource:5M
?batch_normalization_513_assignmovingavg_readvariableop_resource:5O
Abatch_normalization_513_assignmovingavg_1_readvariableop_resource:5K
=batch_normalization_513_batchnorm_mul_readvariableop_resource:5G
9batch_normalization_513_batchnorm_readvariableop_resource:5:
(dense_570_matmul_readvariableop_resource:557
)dense_570_biasadd_readvariableop_resource:5M
?batch_normalization_514_assignmovingavg_readvariableop_resource:5O
Abatch_normalization_514_assignmovingavg_1_readvariableop_resource:5K
=batch_normalization_514_batchnorm_mul_readvariableop_resource:5G
9batch_normalization_514_batchnorm_readvariableop_resource:5:
(dense_571_matmul_readvariableop_resource:557
)dense_571_biasadd_readvariableop_resource:5M
?batch_normalization_515_assignmovingavg_readvariableop_resource:5O
Abatch_normalization_515_assignmovingavg_1_readvariableop_resource:5K
=batch_normalization_515_batchnorm_mul_readvariableop_resource:5G
9batch_normalization_515_batchnorm_readvariableop_resource:5:
(dense_572_matmul_readvariableop_resource:57
)dense_572_biasadd_readvariableop_resource:
identity¢'batch_normalization_504/AssignMovingAvg¢6batch_normalization_504/AssignMovingAvg/ReadVariableOp¢)batch_normalization_504/AssignMovingAvg_1¢8batch_normalization_504/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_504/batchnorm/ReadVariableOp¢4batch_normalization_504/batchnorm/mul/ReadVariableOp¢'batch_normalization_505/AssignMovingAvg¢6batch_normalization_505/AssignMovingAvg/ReadVariableOp¢)batch_normalization_505/AssignMovingAvg_1¢8batch_normalization_505/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_505/batchnorm/ReadVariableOp¢4batch_normalization_505/batchnorm/mul/ReadVariableOp¢'batch_normalization_506/AssignMovingAvg¢6batch_normalization_506/AssignMovingAvg/ReadVariableOp¢)batch_normalization_506/AssignMovingAvg_1¢8batch_normalization_506/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_506/batchnorm/ReadVariableOp¢4batch_normalization_506/batchnorm/mul/ReadVariableOp¢'batch_normalization_507/AssignMovingAvg¢6batch_normalization_507/AssignMovingAvg/ReadVariableOp¢)batch_normalization_507/AssignMovingAvg_1¢8batch_normalization_507/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_507/batchnorm/ReadVariableOp¢4batch_normalization_507/batchnorm/mul/ReadVariableOp¢'batch_normalization_508/AssignMovingAvg¢6batch_normalization_508/AssignMovingAvg/ReadVariableOp¢)batch_normalization_508/AssignMovingAvg_1¢8batch_normalization_508/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_508/batchnorm/ReadVariableOp¢4batch_normalization_508/batchnorm/mul/ReadVariableOp¢'batch_normalization_509/AssignMovingAvg¢6batch_normalization_509/AssignMovingAvg/ReadVariableOp¢)batch_normalization_509/AssignMovingAvg_1¢8batch_normalization_509/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_509/batchnorm/ReadVariableOp¢4batch_normalization_509/batchnorm/mul/ReadVariableOp¢'batch_normalization_510/AssignMovingAvg¢6batch_normalization_510/AssignMovingAvg/ReadVariableOp¢)batch_normalization_510/AssignMovingAvg_1¢8batch_normalization_510/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_510/batchnorm/ReadVariableOp¢4batch_normalization_510/batchnorm/mul/ReadVariableOp¢'batch_normalization_511/AssignMovingAvg¢6batch_normalization_511/AssignMovingAvg/ReadVariableOp¢)batch_normalization_511/AssignMovingAvg_1¢8batch_normalization_511/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_511/batchnorm/ReadVariableOp¢4batch_normalization_511/batchnorm/mul/ReadVariableOp¢'batch_normalization_512/AssignMovingAvg¢6batch_normalization_512/AssignMovingAvg/ReadVariableOp¢)batch_normalization_512/AssignMovingAvg_1¢8batch_normalization_512/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_512/batchnorm/ReadVariableOp¢4batch_normalization_512/batchnorm/mul/ReadVariableOp¢'batch_normalization_513/AssignMovingAvg¢6batch_normalization_513/AssignMovingAvg/ReadVariableOp¢)batch_normalization_513/AssignMovingAvg_1¢8batch_normalization_513/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_513/batchnorm/ReadVariableOp¢4batch_normalization_513/batchnorm/mul/ReadVariableOp¢'batch_normalization_514/AssignMovingAvg¢6batch_normalization_514/AssignMovingAvg/ReadVariableOp¢)batch_normalization_514/AssignMovingAvg_1¢8batch_normalization_514/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_514/batchnorm/ReadVariableOp¢4batch_normalization_514/batchnorm/mul/ReadVariableOp¢'batch_normalization_515/AssignMovingAvg¢6batch_normalization_515/AssignMovingAvg/ReadVariableOp¢)batch_normalization_515/AssignMovingAvg_1¢8batch_normalization_515/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_515/batchnorm/ReadVariableOp¢4batch_normalization_515/batchnorm/mul/ReadVariableOp¢ dense_560/BiasAdd/ReadVariableOp¢dense_560/MatMul/ReadVariableOp¢ dense_561/BiasAdd/ReadVariableOp¢dense_561/MatMul/ReadVariableOp¢ dense_562/BiasAdd/ReadVariableOp¢dense_562/MatMul/ReadVariableOp¢ dense_563/BiasAdd/ReadVariableOp¢dense_563/MatMul/ReadVariableOp¢ dense_564/BiasAdd/ReadVariableOp¢dense_564/MatMul/ReadVariableOp¢ dense_565/BiasAdd/ReadVariableOp¢dense_565/MatMul/ReadVariableOp¢ dense_566/BiasAdd/ReadVariableOp¢dense_566/MatMul/ReadVariableOp¢ dense_567/BiasAdd/ReadVariableOp¢dense_567/MatMul/ReadVariableOp¢ dense_568/BiasAdd/ReadVariableOp¢dense_568/MatMul/ReadVariableOp¢ dense_569/BiasAdd/ReadVariableOp¢dense_569/MatMul/ReadVariableOp¢ dense_570/BiasAdd/ReadVariableOp¢dense_570/MatMul/ReadVariableOp¢ dense_571/BiasAdd/ReadVariableOp¢dense_571/MatMul/ReadVariableOp¢ dense_572/BiasAdd/ReadVariableOp¢dense_572/MatMul/ReadVariableOpm
normalization_56/subSubinputsnormalization_56_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_56/SqrtSqrtnormalization_56_sqrt_x*
T0*
_output_shapes

:_
normalization_56/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_56/MaximumMaximumnormalization_56/Sqrt:y:0#normalization_56/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_56/truedivRealDivnormalization_56/sub:z:0normalization_56/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_560/MatMul/ReadVariableOpReadVariableOp(dense_560_matmul_readvariableop_resource*
_output_shapes

:A*
dtype0
dense_560/MatMulMatMulnormalization_56/truediv:z:0'dense_560/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_560/BiasAdd/ReadVariableOpReadVariableOp)dense_560_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_560/BiasAddBiasAdddense_560/MatMul:product:0(dense_560/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
6batch_normalization_504/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_504/moments/meanMeandense_560/BiasAdd:output:0?batch_normalization_504/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
,batch_normalization_504/moments/StopGradientStopGradient-batch_normalization_504/moments/mean:output:0*
T0*
_output_shapes

:AË
1batch_normalization_504/moments/SquaredDifferenceSquaredDifferencedense_560/BiasAdd:output:05batch_normalization_504/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
:batch_normalization_504/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_504/moments/varianceMean5batch_normalization_504/moments/SquaredDifference:z:0Cbatch_normalization_504/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
'batch_normalization_504/moments/SqueezeSqueeze-batch_normalization_504/moments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 £
)batch_normalization_504/moments/Squeeze_1Squeeze1batch_normalization_504/moments/variance:output:0*
T0*
_output_shapes
:A*
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
:A*
dtype0É
+batch_normalization_504/AssignMovingAvg/subSub>batch_normalization_504/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_504/moments/Squeeze:output:0*
T0*
_output_shapes
:AÀ
+batch_normalization_504/AssignMovingAvg/mulMul/batch_normalization_504/AssignMovingAvg/sub:z:06batch_normalization_504/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A
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
:A*
dtype0Ï
-batch_normalization_504/AssignMovingAvg_1/subSub@batch_normalization_504/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_504/moments/Squeeze_1:output:0*
T0*
_output_shapes
:AÆ
-batch_normalization_504/AssignMovingAvg_1/mulMul1batch_normalization_504/AssignMovingAvg_1/sub:z:08batch_normalization_504/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A
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
:A
'batch_normalization_504/batchnorm/RsqrtRsqrt)batch_normalization_504/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_504/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_504_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_504/batchnorm/mulMul+batch_normalization_504/batchnorm/Rsqrt:y:0<batch_normalization_504/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_504/batchnorm/mul_1Muldense_560/BiasAdd:output:0)batch_normalization_504/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA°
'batch_normalization_504/batchnorm/mul_2Mul0batch_normalization_504/moments/Squeeze:output:0)batch_normalization_504/batchnorm/mul:z:0*
T0*
_output_shapes
:A¦
0batch_normalization_504/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_504_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0¸
%batch_normalization_504/batchnorm/subSub8batch_normalization_504/batchnorm/ReadVariableOp:value:0+batch_normalization_504/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_504/batchnorm/add_1AddV2+batch_normalization_504/batchnorm/mul_1:z:0)batch_normalization_504/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_504/LeakyRelu	LeakyRelu+batch_normalization_504/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_561/MatMul/ReadVariableOpReadVariableOp(dense_561_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_561/MatMulMatMul'leaky_re_lu_504/LeakyRelu:activations:0'dense_561/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_561/BiasAdd/ReadVariableOpReadVariableOp)dense_561_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_561/BiasAddBiasAdddense_561/MatMul:product:0(dense_561/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
6batch_normalization_505/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_505/moments/meanMeandense_561/BiasAdd:output:0?batch_normalization_505/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
,batch_normalization_505/moments/StopGradientStopGradient-batch_normalization_505/moments/mean:output:0*
T0*
_output_shapes

:AË
1batch_normalization_505/moments/SquaredDifferenceSquaredDifferencedense_561/BiasAdd:output:05batch_normalization_505/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
:batch_normalization_505/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_505/moments/varianceMean5batch_normalization_505/moments/SquaredDifference:z:0Cbatch_normalization_505/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
'batch_normalization_505/moments/SqueezeSqueeze-batch_normalization_505/moments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 £
)batch_normalization_505/moments/Squeeze_1Squeeze1batch_normalization_505/moments/variance:output:0*
T0*
_output_shapes
:A*
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
:A*
dtype0É
+batch_normalization_505/AssignMovingAvg/subSub>batch_normalization_505/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_505/moments/Squeeze:output:0*
T0*
_output_shapes
:AÀ
+batch_normalization_505/AssignMovingAvg/mulMul/batch_normalization_505/AssignMovingAvg/sub:z:06batch_normalization_505/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A
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
:A*
dtype0Ï
-batch_normalization_505/AssignMovingAvg_1/subSub@batch_normalization_505/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_505/moments/Squeeze_1:output:0*
T0*
_output_shapes
:AÆ
-batch_normalization_505/AssignMovingAvg_1/mulMul1batch_normalization_505/AssignMovingAvg_1/sub:z:08batch_normalization_505/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A
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
:A
'batch_normalization_505/batchnorm/RsqrtRsqrt)batch_normalization_505/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_505/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_505_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_505/batchnorm/mulMul+batch_normalization_505/batchnorm/Rsqrt:y:0<batch_normalization_505/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_505/batchnorm/mul_1Muldense_561/BiasAdd:output:0)batch_normalization_505/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA°
'batch_normalization_505/batchnorm/mul_2Mul0batch_normalization_505/moments/Squeeze:output:0)batch_normalization_505/batchnorm/mul:z:0*
T0*
_output_shapes
:A¦
0batch_normalization_505/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_505_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0¸
%batch_normalization_505/batchnorm/subSub8batch_normalization_505/batchnorm/ReadVariableOp:value:0+batch_normalization_505/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_505/batchnorm/add_1AddV2+batch_normalization_505/batchnorm/mul_1:z:0)batch_normalization_505/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_505/LeakyRelu	LeakyRelu+batch_normalization_505/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_562/MatMul/ReadVariableOpReadVariableOp(dense_562_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_562/MatMulMatMul'leaky_re_lu_505/LeakyRelu:activations:0'dense_562/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_562/BiasAdd/ReadVariableOpReadVariableOp)dense_562_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_562/BiasAddBiasAdddense_562/MatMul:product:0(dense_562/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
6batch_normalization_506/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_506/moments/meanMeandense_562/BiasAdd:output:0?batch_normalization_506/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
,batch_normalization_506/moments/StopGradientStopGradient-batch_normalization_506/moments/mean:output:0*
T0*
_output_shapes

:AË
1batch_normalization_506/moments/SquaredDifferenceSquaredDifferencedense_562/BiasAdd:output:05batch_normalization_506/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
:batch_normalization_506/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_506/moments/varianceMean5batch_normalization_506/moments/SquaredDifference:z:0Cbatch_normalization_506/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
'batch_normalization_506/moments/SqueezeSqueeze-batch_normalization_506/moments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 £
)batch_normalization_506/moments/Squeeze_1Squeeze1batch_normalization_506/moments/variance:output:0*
T0*
_output_shapes
:A*
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
:A*
dtype0É
+batch_normalization_506/AssignMovingAvg/subSub>batch_normalization_506/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_506/moments/Squeeze:output:0*
T0*
_output_shapes
:AÀ
+batch_normalization_506/AssignMovingAvg/mulMul/batch_normalization_506/AssignMovingAvg/sub:z:06batch_normalization_506/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A
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
:A*
dtype0Ï
-batch_normalization_506/AssignMovingAvg_1/subSub@batch_normalization_506/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_506/moments/Squeeze_1:output:0*
T0*
_output_shapes
:AÆ
-batch_normalization_506/AssignMovingAvg_1/mulMul1batch_normalization_506/AssignMovingAvg_1/sub:z:08batch_normalization_506/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A
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
:A
'batch_normalization_506/batchnorm/RsqrtRsqrt)batch_normalization_506/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_506/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_506_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_506/batchnorm/mulMul+batch_normalization_506/batchnorm/Rsqrt:y:0<batch_normalization_506/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_506/batchnorm/mul_1Muldense_562/BiasAdd:output:0)batch_normalization_506/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA°
'batch_normalization_506/batchnorm/mul_2Mul0batch_normalization_506/moments/Squeeze:output:0)batch_normalization_506/batchnorm/mul:z:0*
T0*
_output_shapes
:A¦
0batch_normalization_506/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_506_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0¸
%batch_normalization_506/batchnorm/subSub8batch_normalization_506/batchnorm/ReadVariableOp:value:0+batch_normalization_506/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_506/batchnorm/add_1AddV2+batch_normalization_506/batchnorm/mul_1:z:0)batch_normalization_506/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_506/LeakyRelu	LeakyRelu+batch_normalization_506/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_563/MatMul/ReadVariableOpReadVariableOp(dense_563_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_563/MatMulMatMul'leaky_re_lu_506/LeakyRelu:activations:0'dense_563/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_563/BiasAdd/ReadVariableOpReadVariableOp)dense_563_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_563/BiasAddBiasAdddense_563/MatMul:product:0(dense_563/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
6batch_normalization_507/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_507/moments/meanMeandense_563/BiasAdd:output:0?batch_normalization_507/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
,batch_normalization_507/moments/StopGradientStopGradient-batch_normalization_507/moments/mean:output:0*
T0*
_output_shapes

:AË
1batch_normalization_507/moments/SquaredDifferenceSquaredDifferencedense_563/BiasAdd:output:05batch_normalization_507/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
:batch_normalization_507/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_507/moments/varianceMean5batch_normalization_507/moments/SquaredDifference:z:0Cbatch_normalization_507/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
'batch_normalization_507/moments/SqueezeSqueeze-batch_normalization_507/moments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 £
)batch_normalization_507/moments/Squeeze_1Squeeze1batch_normalization_507/moments/variance:output:0*
T0*
_output_shapes
:A*
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
:A*
dtype0É
+batch_normalization_507/AssignMovingAvg/subSub>batch_normalization_507/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_507/moments/Squeeze:output:0*
T0*
_output_shapes
:AÀ
+batch_normalization_507/AssignMovingAvg/mulMul/batch_normalization_507/AssignMovingAvg/sub:z:06batch_normalization_507/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A
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
:A*
dtype0Ï
-batch_normalization_507/AssignMovingAvg_1/subSub@batch_normalization_507/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_507/moments/Squeeze_1:output:0*
T0*
_output_shapes
:AÆ
-batch_normalization_507/AssignMovingAvg_1/mulMul1batch_normalization_507/AssignMovingAvg_1/sub:z:08batch_normalization_507/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A
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
:A
'batch_normalization_507/batchnorm/RsqrtRsqrt)batch_normalization_507/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_507/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_507_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_507/batchnorm/mulMul+batch_normalization_507/batchnorm/Rsqrt:y:0<batch_normalization_507/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_507/batchnorm/mul_1Muldense_563/BiasAdd:output:0)batch_normalization_507/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA°
'batch_normalization_507/batchnorm/mul_2Mul0batch_normalization_507/moments/Squeeze:output:0)batch_normalization_507/batchnorm/mul:z:0*
T0*
_output_shapes
:A¦
0batch_normalization_507/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_507_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0¸
%batch_normalization_507/batchnorm/subSub8batch_normalization_507/batchnorm/ReadVariableOp:value:0+batch_normalization_507/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_507/batchnorm/add_1AddV2+batch_normalization_507/batchnorm/mul_1:z:0)batch_normalization_507/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_507/LeakyRelu	LeakyRelu+batch_normalization_507/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_564/MatMul/ReadVariableOpReadVariableOp(dense_564_matmul_readvariableop_resource*
_output_shapes

:A*
dtype0
dense_564/MatMulMatMul'leaky_re_lu_507/LeakyRelu:activations:0'dense_564/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_564/BiasAdd/ReadVariableOpReadVariableOp)dense_564_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_564/BiasAddBiasAdddense_564/MatMul:product:0(dense_564/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_508/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_508/moments/meanMeandense_564/BiasAdd:output:0?batch_normalization_508/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_508/moments/StopGradientStopGradient-batch_normalization_508/moments/mean:output:0*
T0*
_output_shapes

:Ë
1batch_normalization_508/moments/SquaredDifferenceSquaredDifferencedense_564/BiasAdd:output:05batch_normalization_508/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_508/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_508/moments/varianceMean5batch_normalization_508/moments/SquaredDifference:z:0Cbatch_normalization_508/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_508/moments/SqueezeSqueeze-batch_normalization_508/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_508/moments/Squeeze_1Squeeze1batch_normalization_508/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_508/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_508/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_508_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_508/AssignMovingAvg/subSub>batch_normalization_508/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_508/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_508/AssignMovingAvg/mulMul/batch_normalization_508/AssignMovingAvg/sub:z:06batch_normalization_508/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_508/AssignMovingAvgAssignSubVariableOp?batch_normalization_508_assignmovingavg_readvariableop_resource/batch_normalization_508/AssignMovingAvg/mul:z:07^batch_normalization_508/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_508/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_508/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_508_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_508/AssignMovingAvg_1/subSub@batch_normalization_508/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_508/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_508/AssignMovingAvg_1/mulMul1batch_normalization_508/AssignMovingAvg_1/sub:z:08batch_normalization_508/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_508/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_508_assignmovingavg_1_readvariableop_resource1batch_normalization_508/AssignMovingAvg_1/mul:z:09^batch_normalization_508/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_508/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_508/batchnorm/addAddV22batch_normalization_508/moments/Squeeze_1:output:00batch_normalization_508/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_508/batchnorm/RsqrtRsqrt)batch_normalization_508/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_508/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_508_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_508/batchnorm/mulMul+batch_normalization_508/batchnorm/Rsqrt:y:0<batch_normalization_508/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_508/batchnorm/mul_1Muldense_564/BiasAdd:output:0)batch_normalization_508/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_508/batchnorm/mul_2Mul0batch_normalization_508/moments/Squeeze:output:0)batch_normalization_508/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_508/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_508_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_508/batchnorm/subSub8batch_normalization_508/batchnorm/ReadVariableOp:value:0+batch_normalization_508/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_508/batchnorm/add_1AddV2+batch_normalization_508/batchnorm/mul_1:z:0)batch_normalization_508/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_508/LeakyRelu	LeakyRelu+batch_normalization_508/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_565/MatMul/ReadVariableOpReadVariableOp(dense_565_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_565/MatMulMatMul'leaky_re_lu_508/LeakyRelu:activations:0'dense_565/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_565/BiasAdd/ReadVariableOpReadVariableOp)dense_565_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_565/BiasAddBiasAdddense_565/MatMul:product:0(dense_565/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_509/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_509/moments/meanMeandense_565/BiasAdd:output:0?batch_normalization_509/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_509/moments/StopGradientStopGradient-batch_normalization_509/moments/mean:output:0*
T0*
_output_shapes

:Ë
1batch_normalization_509/moments/SquaredDifferenceSquaredDifferencedense_565/BiasAdd:output:05batch_normalization_509/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_509/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_509/moments/varianceMean5batch_normalization_509/moments/SquaredDifference:z:0Cbatch_normalization_509/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_509/moments/SqueezeSqueeze-batch_normalization_509/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_509/moments/Squeeze_1Squeeze1batch_normalization_509/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_509/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_509/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_509_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_509/AssignMovingAvg/subSub>batch_normalization_509/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_509/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_509/AssignMovingAvg/mulMul/batch_normalization_509/AssignMovingAvg/sub:z:06batch_normalization_509/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_509/AssignMovingAvgAssignSubVariableOp?batch_normalization_509_assignmovingavg_readvariableop_resource/batch_normalization_509/AssignMovingAvg/mul:z:07^batch_normalization_509/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_509/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_509/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_509_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_509/AssignMovingAvg_1/subSub@batch_normalization_509/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_509/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_509/AssignMovingAvg_1/mulMul1batch_normalization_509/AssignMovingAvg_1/sub:z:08batch_normalization_509/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_509/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_509_assignmovingavg_1_readvariableop_resource1batch_normalization_509/AssignMovingAvg_1/mul:z:09^batch_normalization_509/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_509/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_509/batchnorm/addAddV22batch_normalization_509/moments/Squeeze_1:output:00batch_normalization_509/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_509/batchnorm/RsqrtRsqrt)batch_normalization_509/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_509/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_509_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_509/batchnorm/mulMul+batch_normalization_509/batchnorm/Rsqrt:y:0<batch_normalization_509/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_509/batchnorm/mul_1Muldense_565/BiasAdd:output:0)batch_normalization_509/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_509/batchnorm/mul_2Mul0batch_normalization_509/moments/Squeeze:output:0)batch_normalization_509/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_509/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_509_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_509/batchnorm/subSub8batch_normalization_509/batchnorm/ReadVariableOp:value:0+batch_normalization_509/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_509/batchnorm/add_1AddV2+batch_normalization_509/batchnorm/mul_1:z:0)batch_normalization_509/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_509/LeakyRelu	LeakyRelu+batch_normalization_509/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_566/MatMul/ReadVariableOpReadVariableOp(dense_566_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_566/MatMulMatMul'leaky_re_lu_509/LeakyRelu:activations:0'dense_566/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_566/BiasAdd/ReadVariableOpReadVariableOp)dense_566_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_566/BiasAddBiasAdddense_566/MatMul:product:0(dense_566/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_510/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_510/moments/meanMeandense_566/BiasAdd:output:0?batch_normalization_510/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_510/moments/StopGradientStopGradient-batch_normalization_510/moments/mean:output:0*
T0*
_output_shapes

:Ë
1batch_normalization_510/moments/SquaredDifferenceSquaredDifferencedense_566/BiasAdd:output:05batch_normalization_510/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_510/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_510/moments/varianceMean5batch_normalization_510/moments/SquaredDifference:z:0Cbatch_normalization_510/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_510/moments/SqueezeSqueeze-batch_normalization_510/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_510/moments/Squeeze_1Squeeze1batch_normalization_510/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_510/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_510/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_510_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_510/AssignMovingAvg/subSub>batch_normalization_510/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_510/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_510/AssignMovingAvg/mulMul/batch_normalization_510/AssignMovingAvg/sub:z:06batch_normalization_510/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_510/AssignMovingAvgAssignSubVariableOp?batch_normalization_510_assignmovingavg_readvariableop_resource/batch_normalization_510/AssignMovingAvg/mul:z:07^batch_normalization_510/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_510/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_510/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_510_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_510/AssignMovingAvg_1/subSub@batch_normalization_510/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_510/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_510/AssignMovingAvg_1/mulMul1batch_normalization_510/AssignMovingAvg_1/sub:z:08batch_normalization_510/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_510/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_510_assignmovingavg_1_readvariableop_resource1batch_normalization_510/AssignMovingAvg_1/mul:z:09^batch_normalization_510/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_510/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_510/batchnorm/addAddV22batch_normalization_510/moments/Squeeze_1:output:00batch_normalization_510/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_510/batchnorm/RsqrtRsqrt)batch_normalization_510/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_510/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_510_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_510/batchnorm/mulMul+batch_normalization_510/batchnorm/Rsqrt:y:0<batch_normalization_510/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_510/batchnorm/mul_1Muldense_566/BiasAdd:output:0)batch_normalization_510/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_510/batchnorm/mul_2Mul0batch_normalization_510/moments/Squeeze:output:0)batch_normalization_510/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_510/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_510_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_510/batchnorm/subSub8batch_normalization_510/batchnorm/ReadVariableOp:value:0+batch_normalization_510/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_510/batchnorm/add_1AddV2+batch_normalization_510/batchnorm/mul_1:z:0)batch_normalization_510/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_510/LeakyRelu	LeakyRelu+batch_normalization_510/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_567/MatMul/ReadVariableOpReadVariableOp(dense_567_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0
dense_567/MatMulMatMul'leaky_re_lu_510/LeakyRelu:activations:0'dense_567/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 dense_567/BiasAdd/ReadVariableOpReadVariableOp)dense_567_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_567/BiasAddBiasAdddense_567/MatMul:product:0(dense_567/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
6batch_normalization_511/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_511/moments/meanMeandense_567/BiasAdd:output:0?batch_normalization_511/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
,batch_normalization_511/moments/StopGradientStopGradient-batch_normalization_511/moments/mean:output:0*
T0*
_output_shapes

:5Ë
1batch_normalization_511/moments/SquaredDifferenceSquaredDifferencedense_567/BiasAdd:output:05batch_normalization_511/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
:batch_normalization_511/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_511/moments/varianceMean5batch_normalization_511/moments/SquaredDifference:z:0Cbatch_normalization_511/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
'batch_normalization_511/moments/SqueezeSqueeze-batch_normalization_511/moments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 £
)batch_normalization_511/moments/Squeeze_1Squeeze1batch_normalization_511/moments/variance:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 r
-batch_normalization_511/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_511/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_511_assignmovingavg_readvariableop_resource*
_output_shapes
:5*
dtype0É
+batch_normalization_511/AssignMovingAvg/subSub>batch_normalization_511/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_511/moments/Squeeze:output:0*
T0*
_output_shapes
:5À
+batch_normalization_511/AssignMovingAvg/mulMul/batch_normalization_511/AssignMovingAvg/sub:z:06batch_normalization_511/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5
'batch_normalization_511/AssignMovingAvgAssignSubVariableOp?batch_normalization_511_assignmovingavg_readvariableop_resource/batch_normalization_511/AssignMovingAvg/mul:z:07^batch_normalization_511/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_511/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_511/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_511_assignmovingavg_1_readvariableop_resource*
_output_shapes
:5*
dtype0Ï
-batch_normalization_511/AssignMovingAvg_1/subSub@batch_normalization_511/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_511/moments/Squeeze_1:output:0*
T0*
_output_shapes
:5Æ
-batch_normalization_511/AssignMovingAvg_1/mulMul1batch_normalization_511/AssignMovingAvg_1/sub:z:08batch_normalization_511/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5
)batch_normalization_511/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_511_assignmovingavg_1_readvariableop_resource1batch_normalization_511/AssignMovingAvg_1/mul:z:09^batch_normalization_511/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_511/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_511/batchnorm/addAddV22batch_normalization_511/moments/Squeeze_1:output:00batch_normalization_511/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_511/batchnorm/RsqrtRsqrt)batch_normalization_511/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_511/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_511_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_511/batchnorm/mulMul+batch_normalization_511/batchnorm/Rsqrt:y:0<batch_normalization_511/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5§
'batch_normalization_511/batchnorm/mul_1Muldense_567/BiasAdd:output:0)batch_normalization_511/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5°
'batch_normalization_511/batchnorm/mul_2Mul0batch_normalization_511/moments/Squeeze:output:0)batch_normalization_511/batchnorm/mul:z:0*
T0*
_output_shapes
:5¦
0batch_normalization_511/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_511_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0¸
%batch_normalization_511/batchnorm/subSub8batch_normalization_511/batchnorm/ReadVariableOp:value:0+batch_normalization_511/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_511/batchnorm/add_1AddV2+batch_normalization_511/batchnorm/mul_1:z:0)batch_normalization_511/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_511/LeakyRelu	LeakyRelu+batch_normalization_511/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
dense_568/MatMul/ReadVariableOpReadVariableOp(dense_568_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0
dense_568/MatMulMatMul'leaky_re_lu_511/LeakyRelu:activations:0'dense_568/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 dense_568/BiasAdd/ReadVariableOpReadVariableOp)dense_568_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_568/BiasAddBiasAdddense_568/MatMul:product:0(dense_568/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
6batch_normalization_512/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_512/moments/meanMeandense_568/BiasAdd:output:0?batch_normalization_512/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
,batch_normalization_512/moments/StopGradientStopGradient-batch_normalization_512/moments/mean:output:0*
T0*
_output_shapes

:5Ë
1batch_normalization_512/moments/SquaredDifferenceSquaredDifferencedense_568/BiasAdd:output:05batch_normalization_512/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
:batch_normalization_512/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_512/moments/varianceMean5batch_normalization_512/moments/SquaredDifference:z:0Cbatch_normalization_512/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
'batch_normalization_512/moments/SqueezeSqueeze-batch_normalization_512/moments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 £
)batch_normalization_512/moments/Squeeze_1Squeeze1batch_normalization_512/moments/variance:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 r
-batch_normalization_512/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_512/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_512_assignmovingavg_readvariableop_resource*
_output_shapes
:5*
dtype0É
+batch_normalization_512/AssignMovingAvg/subSub>batch_normalization_512/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_512/moments/Squeeze:output:0*
T0*
_output_shapes
:5À
+batch_normalization_512/AssignMovingAvg/mulMul/batch_normalization_512/AssignMovingAvg/sub:z:06batch_normalization_512/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5
'batch_normalization_512/AssignMovingAvgAssignSubVariableOp?batch_normalization_512_assignmovingavg_readvariableop_resource/batch_normalization_512/AssignMovingAvg/mul:z:07^batch_normalization_512/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_512/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_512/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_512_assignmovingavg_1_readvariableop_resource*
_output_shapes
:5*
dtype0Ï
-batch_normalization_512/AssignMovingAvg_1/subSub@batch_normalization_512/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_512/moments/Squeeze_1:output:0*
T0*
_output_shapes
:5Æ
-batch_normalization_512/AssignMovingAvg_1/mulMul1batch_normalization_512/AssignMovingAvg_1/sub:z:08batch_normalization_512/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5
)batch_normalization_512/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_512_assignmovingavg_1_readvariableop_resource1batch_normalization_512/AssignMovingAvg_1/mul:z:09^batch_normalization_512/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_512/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_512/batchnorm/addAddV22batch_normalization_512/moments/Squeeze_1:output:00batch_normalization_512/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_512/batchnorm/RsqrtRsqrt)batch_normalization_512/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_512/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_512_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_512/batchnorm/mulMul+batch_normalization_512/batchnorm/Rsqrt:y:0<batch_normalization_512/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5§
'batch_normalization_512/batchnorm/mul_1Muldense_568/BiasAdd:output:0)batch_normalization_512/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5°
'batch_normalization_512/batchnorm/mul_2Mul0batch_normalization_512/moments/Squeeze:output:0)batch_normalization_512/batchnorm/mul:z:0*
T0*
_output_shapes
:5¦
0batch_normalization_512/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_512_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0¸
%batch_normalization_512/batchnorm/subSub8batch_normalization_512/batchnorm/ReadVariableOp:value:0+batch_normalization_512/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_512/batchnorm/add_1AddV2+batch_normalization_512/batchnorm/mul_1:z:0)batch_normalization_512/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_512/LeakyRelu	LeakyRelu+batch_normalization_512/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
dense_569/MatMul/ReadVariableOpReadVariableOp(dense_569_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0
dense_569/MatMulMatMul'leaky_re_lu_512/LeakyRelu:activations:0'dense_569/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 dense_569/BiasAdd/ReadVariableOpReadVariableOp)dense_569_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_569/BiasAddBiasAdddense_569/MatMul:product:0(dense_569/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
6batch_normalization_513/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_513/moments/meanMeandense_569/BiasAdd:output:0?batch_normalization_513/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
,batch_normalization_513/moments/StopGradientStopGradient-batch_normalization_513/moments/mean:output:0*
T0*
_output_shapes

:5Ë
1batch_normalization_513/moments/SquaredDifferenceSquaredDifferencedense_569/BiasAdd:output:05batch_normalization_513/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
:batch_normalization_513/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_513/moments/varianceMean5batch_normalization_513/moments/SquaredDifference:z:0Cbatch_normalization_513/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
'batch_normalization_513/moments/SqueezeSqueeze-batch_normalization_513/moments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 £
)batch_normalization_513/moments/Squeeze_1Squeeze1batch_normalization_513/moments/variance:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 r
-batch_normalization_513/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_513/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_513_assignmovingavg_readvariableop_resource*
_output_shapes
:5*
dtype0É
+batch_normalization_513/AssignMovingAvg/subSub>batch_normalization_513/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_513/moments/Squeeze:output:0*
T0*
_output_shapes
:5À
+batch_normalization_513/AssignMovingAvg/mulMul/batch_normalization_513/AssignMovingAvg/sub:z:06batch_normalization_513/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5
'batch_normalization_513/AssignMovingAvgAssignSubVariableOp?batch_normalization_513_assignmovingavg_readvariableop_resource/batch_normalization_513/AssignMovingAvg/mul:z:07^batch_normalization_513/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_513/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_513/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_513_assignmovingavg_1_readvariableop_resource*
_output_shapes
:5*
dtype0Ï
-batch_normalization_513/AssignMovingAvg_1/subSub@batch_normalization_513/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_513/moments/Squeeze_1:output:0*
T0*
_output_shapes
:5Æ
-batch_normalization_513/AssignMovingAvg_1/mulMul1batch_normalization_513/AssignMovingAvg_1/sub:z:08batch_normalization_513/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5
)batch_normalization_513/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_513_assignmovingavg_1_readvariableop_resource1batch_normalization_513/AssignMovingAvg_1/mul:z:09^batch_normalization_513/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_513/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_513/batchnorm/addAddV22batch_normalization_513/moments/Squeeze_1:output:00batch_normalization_513/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_513/batchnorm/RsqrtRsqrt)batch_normalization_513/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_513/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_513_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_513/batchnorm/mulMul+batch_normalization_513/batchnorm/Rsqrt:y:0<batch_normalization_513/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5§
'batch_normalization_513/batchnorm/mul_1Muldense_569/BiasAdd:output:0)batch_normalization_513/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5°
'batch_normalization_513/batchnorm/mul_2Mul0batch_normalization_513/moments/Squeeze:output:0)batch_normalization_513/batchnorm/mul:z:0*
T0*
_output_shapes
:5¦
0batch_normalization_513/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_513_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0¸
%batch_normalization_513/batchnorm/subSub8batch_normalization_513/batchnorm/ReadVariableOp:value:0+batch_normalization_513/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_513/batchnorm/add_1AddV2+batch_normalization_513/batchnorm/mul_1:z:0)batch_normalization_513/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_513/LeakyRelu	LeakyRelu+batch_normalization_513/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
dense_570/MatMul/ReadVariableOpReadVariableOp(dense_570_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0
dense_570/MatMulMatMul'leaky_re_lu_513/LeakyRelu:activations:0'dense_570/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 dense_570/BiasAdd/ReadVariableOpReadVariableOp)dense_570_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_570/BiasAddBiasAdddense_570/MatMul:product:0(dense_570/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
6batch_normalization_514/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_514/moments/meanMeandense_570/BiasAdd:output:0?batch_normalization_514/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
,batch_normalization_514/moments/StopGradientStopGradient-batch_normalization_514/moments/mean:output:0*
T0*
_output_shapes

:5Ë
1batch_normalization_514/moments/SquaredDifferenceSquaredDifferencedense_570/BiasAdd:output:05batch_normalization_514/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
:batch_normalization_514/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_514/moments/varianceMean5batch_normalization_514/moments/SquaredDifference:z:0Cbatch_normalization_514/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
'batch_normalization_514/moments/SqueezeSqueeze-batch_normalization_514/moments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 £
)batch_normalization_514/moments/Squeeze_1Squeeze1batch_normalization_514/moments/variance:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 r
-batch_normalization_514/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_514/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_514_assignmovingavg_readvariableop_resource*
_output_shapes
:5*
dtype0É
+batch_normalization_514/AssignMovingAvg/subSub>batch_normalization_514/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_514/moments/Squeeze:output:0*
T0*
_output_shapes
:5À
+batch_normalization_514/AssignMovingAvg/mulMul/batch_normalization_514/AssignMovingAvg/sub:z:06batch_normalization_514/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5
'batch_normalization_514/AssignMovingAvgAssignSubVariableOp?batch_normalization_514_assignmovingavg_readvariableop_resource/batch_normalization_514/AssignMovingAvg/mul:z:07^batch_normalization_514/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_514/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_514/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_514_assignmovingavg_1_readvariableop_resource*
_output_shapes
:5*
dtype0Ï
-batch_normalization_514/AssignMovingAvg_1/subSub@batch_normalization_514/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_514/moments/Squeeze_1:output:0*
T0*
_output_shapes
:5Æ
-batch_normalization_514/AssignMovingAvg_1/mulMul1batch_normalization_514/AssignMovingAvg_1/sub:z:08batch_normalization_514/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5
)batch_normalization_514/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_514_assignmovingavg_1_readvariableop_resource1batch_normalization_514/AssignMovingAvg_1/mul:z:09^batch_normalization_514/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_514/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_514/batchnorm/addAddV22batch_normalization_514/moments/Squeeze_1:output:00batch_normalization_514/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_514/batchnorm/RsqrtRsqrt)batch_normalization_514/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_514/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_514_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_514/batchnorm/mulMul+batch_normalization_514/batchnorm/Rsqrt:y:0<batch_normalization_514/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5§
'batch_normalization_514/batchnorm/mul_1Muldense_570/BiasAdd:output:0)batch_normalization_514/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5°
'batch_normalization_514/batchnorm/mul_2Mul0batch_normalization_514/moments/Squeeze:output:0)batch_normalization_514/batchnorm/mul:z:0*
T0*
_output_shapes
:5¦
0batch_normalization_514/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_514_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0¸
%batch_normalization_514/batchnorm/subSub8batch_normalization_514/batchnorm/ReadVariableOp:value:0+batch_normalization_514/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_514/batchnorm/add_1AddV2+batch_normalization_514/batchnorm/mul_1:z:0)batch_normalization_514/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_514/LeakyRelu	LeakyRelu+batch_normalization_514/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
dense_571/MatMul/ReadVariableOpReadVariableOp(dense_571_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0
dense_571/MatMulMatMul'leaky_re_lu_514/LeakyRelu:activations:0'dense_571/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 dense_571/BiasAdd/ReadVariableOpReadVariableOp)dense_571_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_571/BiasAddBiasAdddense_571/MatMul:product:0(dense_571/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
6batch_normalization_515/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_515/moments/meanMeandense_571/BiasAdd:output:0?batch_normalization_515/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
,batch_normalization_515/moments/StopGradientStopGradient-batch_normalization_515/moments/mean:output:0*
T0*
_output_shapes

:5Ë
1batch_normalization_515/moments/SquaredDifferenceSquaredDifferencedense_571/BiasAdd:output:05batch_normalization_515/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
:batch_normalization_515/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_515/moments/varianceMean5batch_normalization_515/moments/SquaredDifference:z:0Cbatch_normalization_515/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
'batch_normalization_515/moments/SqueezeSqueeze-batch_normalization_515/moments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 £
)batch_normalization_515/moments/Squeeze_1Squeeze1batch_normalization_515/moments/variance:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 r
-batch_normalization_515/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_515/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_515_assignmovingavg_readvariableop_resource*
_output_shapes
:5*
dtype0É
+batch_normalization_515/AssignMovingAvg/subSub>batch_normalization_515/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_515/moments/Squeeze:output:0*
T0*
_output_shapes
:5À
+batch_normalization_515/AssignMovingAvg/mulMul/batch_normalization_515/AssignMovingAvg/sub:z:06batch_normalization_515/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5
'batch_normalization_515/AssignMovingAvgAssignSubVariableOp?batch_normalization_515_assignmovingavg_readvariableop_resource/batch_normalization_515/AssignMovingAvg/mul:z:07^batch_normalization_515/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_515/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_515/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_515_assignmovingavg_1_readvariableop_resource*
_output_shapes
:5*
dtype0Ï
-batch_normalization_515/AssignMovingAvg_1/subSub@batch_normalization_515/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_515/moments/Squeeze_1:output:0*
T0*
_output_shapes
:5Æ
-batch_normalization_515/AssignMovingAvg_1/mulMul1batch_normalization_515/AssignMovingAvg_1/sub:z:08batch_normalization_515/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5
)batch_normalization_515/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_515_assignmovingavg_1_readvariableop_resource1batch_normalization_515/AssignMovingAvg_1/mul:z:09^batch_normalization_515/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_515/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_515/batchnorm/addAddV22batch_normalization_515/moments/Squeeze_1:output:00batch_normalization_515/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_515/batchnorm/RsqrtRsqrt)batch_normalization_515/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_515/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_515_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_515/batchnorm/mulMul+batch_normalization_515/batchnorm/Rsqrt:y:0<batch_normalization_515/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5§
'batch_normalization_515/batchnorm/mul_1Muldense_571/BiasAdd:output:0)batch_normalization_515/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5°
'batch_normalization_515/batchnorm/mul_2Mul0batch_normalization_515/moments/Squeeze:output:0)batch_normalization_515/batchnorm/mul:z:0*
T0*
_output_shapes
:5¦
0batch_normalization_515/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_515_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0¸
%batch_normalization_515/batchnorm/subSub8batch_normalization_515/batchnorm/ReadVariableOp:value:0+batch_normalization_515/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_515/batchnorm/add_1AddV2+batch_normalization_515/batchnorm/mul_1:z:0)batch_normalization_515/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_515/LeakyRelu	LeakyRelu+batch_normalization_515/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
dense_572/MatMul/ReadVariableOpReadVariableOp(dense_572_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0
dense_572/MatMulMatMul'leaky_re_lu_515/LeakyRelu:activations:0'dense_572/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_572/BiasAdd/ReadVariableOpReadVariableOp)dense_572_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_572/BiasAddBiasAdddense_572/MatMul:product:0(dense_572/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_572/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·$
NoOpNoOp(^batch_normalization_504/AssignMovingAvg7^batch_normalization_504/AssignMovingAvg/ReadVariableOp*^batch_normalization_504/AssignMovingAvg_19^batch_normalization_504/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_504/batchnorm/ReadVariableOp5^batch_normalization_504/batchnorm/mul/ReadVariableOp(^batch_normalization_505/AssignMovingAvg7^batch_normalization_505/AssignMovingAvg/ReadVariableOp*^batch_normalization_505/AssignMovingAvg_19^batch_normalization_505/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_505/batchnorm/ReadVariableOp5^batch_normalization_505/batchnorm/mul/ReadVariableOp(^batch_normalization_506/AssignMovingAvg7^batch_normalization_506/AssignMovingAvg/ReadVariableOp*^batch_normalization_506/AssignMovingAvg_19^batch_normalization_506/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_506/batchnorm/ReadVariableOp5^batch_normalization_506/batchnorm/mul/ReadVariableOp(^batch_normalization_507/AssignMovingAvg7^batch_normalization_507/AssignMovingAvg/ReadVariableOp*^batch_normalization_507/AssignMovingAvg_19^batch_normalization_507/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_507/batchnorm/ReadVariableOp5^batch_normalization_507/batchnorm/mul/ReadVariableOp(^batch_normalization_508/AssignMovingAvg7^batch_normalization_508/AssignMovingAvg/ReadVariableOp*^batch_normalization_508/AssignMovingAvg_19^batch_normalization_508/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_508/batchnorm/ReadVariableOp5^batch_normalization_508/batchnorm/mul/ReadVariableOp(^batch_normalization_509/AssignMovingAvg7^batch_normalization_509/AssignMovingAvg/ReadVariableOp*^batch_normalization_509/AssignMovingAvg_19^batch_normalization_509/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_509/batchnorm/ReadVariableOp5^batch_normalization_509/batchnorm/mul/ReadVariableOp(^batch_normalization_510/AssignMovingAvg7^batch_normalization_510/AssignMovingAvg/ReadVariableOp*^batch_normalization_510/AssignMovingAvg_19^batch_normalization_510/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_510/batchnorm/ReadVariableOp5^batch_normalization_510/batchnorm/mul/ReadVariableOp(^batch_normalization_511/AssignMovingAvg7^batch_normalization_511/AssignMovingAvg/ReadVariableOp*^batch_normalization_511/AssignMovingAvg_19^batch_normalization_511/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_511/batchnorm/ReadVariableOp5^batch_normalization_511/batchnorm/mul/ReadVariableOp(^batch_normalization_512/AssignMovingAvg7^batch_normalization_512/AssignMovingAvg/ReadVariableOp*^batch_normalization_512/AssignMovingAvg_19^batch_normalization_512/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_512/batchnorm/ReadVariableOp5^batch_normalization_512/batchnorm/mul/ReadVariableOp(^batch_normalization_513/AssignMovingAvg7^batch_normalization_513/AssignMovingAvg/ReadVariableOp*^batch_normalization_513/AssignMovingAvg_19^batch_normalization_513/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_513/batchnorm/ReadVariableOp5^batch_normalization_513/batchnorm/mul/ReadVariableOp(^batch_normalization_514/AssignMovingAvg7^batch_normalization_514/AssignMovingAvg/ReadVariableOp*^batch_normalization_514/AssignMovingAvg_19^batch_normalization_514/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_514/batchnorm/ReadVariableOp5^batch_normalization_514/batchnorm/mul/ReadVariableOp(^batch_normalization_515/AssignMovingAvg7^batch_normalization_515/AssignMovingAvg/ReadVariableOp*^batch_normalization_515/AssignMovingAvg_19^batch_normalization_515/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_515/batchnorm/ReadVariableOp5^batch_normalization_515/batchnorm/mul/ReadVariableOp!^dense_560/BiasAdd/ReadVariableOp ^dense_560/MatMul/ReadVariableOp!^dense_561/BiasAdd/ReadVariableOp ^dense_561/MatMul/ReadVariableOp!^dense_562/BiasAdd/ReadVariableOp ^dense_562/MatMul/ReadVariableOp!^dense_563/BiasAdd/ReadVariableOp ^dense_563/MatMul/ReadVariableOp!^dense_564/BiasAdd/ReadVariableOp ^dense_564/MatMul/ReadVariableOp!^dense_565/BiasAdd/ReadVariableOp ^dense_565/MatMul/ReadVariableOp!^dense_566/BiasAdd/ReadVariableOp ^dense_566/MatMul/ReadVariableOp!^dense_567/BiasAdd/ReadVariableOp ^dense_567/MatMul/ReadVariableOp!^dense_568/BiasAdd/ReadVariableOp ^dense_568/MatMul/ReadVariableOp!^dense_569/BiasAdd/ReadVariableOp ^dense_569/MatMul/ReadVariableOp!^dense_570/BiasAdd/ReadVariableOp ^dense_570/MatMul/ReadVariableOp!^dense_571/BiasAdd/ReadVariableOp ^dense_571/MatMul/ReadVariableOp!^dense_572/BiasAdd/ReadVariableOp ^dense_572/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ð
_input_shapes¾
»:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
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
4batch_normalization_507/batchnorm/mul/ReadVariableOp4batch_normalization_507/batchnorm/mul/ReadVariableOp2R
'batch_normalization_508/AssignMovingAvg'batch_normalization_508/AssignMovingAvg2p
6batch_normalization_508/AssignMovingAvg/ReadVariableOp6batch_normalization_508/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_508/AssignMovingAvg_1)batch_normalization_508/AssignMovingAvg_12t
8batch_normalization_508/AssignMovingAvg_1/ReadVariableOp8batch_normalization_508/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_508/batchnorm/ReadVariableOp0batch_normalization_508/batchnorm/ReadVariableOp2l
4batch_normalization_508/batchnorm/mul/ReadVariableOp4batch_normalization_508/batchnorm/mul/ReadVariableOp2R
'batch_normalization_509/AssignMovingAvg'batch_normalization_509/AssignMovingAvg2p
6batch_normalization_509/AssignMovingAvg/ReadVariableOp6batch_normalization_509/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_509/AssignMovingAvg_1)batch_normalization_509/AssignMovingAvg_12t
8batch_normalization_509/AssignMovingAvg_1/ReadVariableOp8batch_normalization_509/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_509/batchnorm/ReadVariableOp0batch_normalization_509/batchnorm/ReadVariableOp2l
4batch_normalization_509/batchnorm/mul/ReadVariableOp4batch_normalization_509/batchnorm/mul/ReadVariableOp2R
'batch_normalization_510/AssignMovingAvg'batch_normalization_510/AssignMovingAvg2p
6batch_normalization_510/AssignMovingAvg/ReadVariableOp6batch_normalization_510/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_510/AssignMovingAvg_1)batch_normalization_510/AssignMovingAvg_12t
8batch_normalization_510/AssignMovingAvg_1/ReadVariableOp8batch_normalization_510/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_510/batchnorm/ReadVariableOp0batch_normalization_510/batchnorm/ReadVariableOp2l
4batch_normalization_510/batchnorm/mul/ReadVariableOp4batch_normalization_510/batchnorm/mul/ReadVariableOp2R
'batch_normalization_511/AssignMovingAvg'batch_normalization_511/AssignMovingAvg2p
6batch_normalization_511/AssignMovingAvg/ReadVariableOp6batch_normalization_511/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_511/AssignMovingAvg_1)batch_normalization_511/AssignMovingAvg_12t
8batch_normalization_511/AssignMovingAvg_1/ReadVariableOp8batch_normalization_511/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_511/batchnorm/ReadVariableOp0batch_normalization_511/batchnorm/ReadVariableOp2l
4batch_normalization_511/batchnorm/mul/ReadVariableOp4batch_normalization_511/batchnorm/mul/ReadVariableOp2R
'batch_normalization_512/AssignMovingAvg'batch_normalization_512/AssignMovingAvg2p
6batch_normalization_512/AssignMovingAvg/ReadVariableOp6batch_normalization_512/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_512/AssignMovingAvg_1)batch_normalization_512/AssignMovingAvg_12t
8batch_normalization_512/AssignMovingAvg_1/ReadVariableOp8batch_normalization_512/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_512/batchnorm/ReadVariableOp0batch_normalization_512/batchnorm/ReadVariableOp2l
4batch_normalization_512/batchnorm/mul/ReadVariableOp4batch_normalization_512/batchnorm/mul/ReadVariableOp2R
'batch_normalization_513/AssignMovingAvg'batch_normalization_513/AssignMovingAvg2p
6batch_normalization_513/AssignMovingAvg/ReadVariableOp6batch_normalization_513/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_513/AssignMovingAvg_1)batch_normalization_513/AssignMovingAvg_12t
8batch_normalization_513/AssignMovingAvg_1/ReadVariableOp8batch_normalization_513/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_513/batchnorm/ReadVariableOp0batch_normalization_513/batchnorm/ReadVariableOp2l
4batch_normalization_513/batchnorm/mul/ReadVariableOp4batch_normalization_513/batchnorm/mul/ReadVariableOp2R
'batch_normalization_514/AssignMovingAvg'batch_normalization_514/AssignMovingAvg2p
6batch_normalization_514/AssignMovingAvg/ReadVariableOp6batch_normalization_514/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_514/AssignMovingAvg_1)batch_normalization_514/AssignMovingAvg_12t
8batch_normalization_514/AssignMovingAvg_1/ReadVariableOp8batch_normalization_514/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_514/batchnorm/ReadVariableOp0batch_normalization_514/batchnorm/ReadVariableOp2l
4batch_normalization_514/batchnorm/mul/ReadVariableOp4batch_normalization_514/batchnorm/mul/ReadVariableOp2R
'batch_normalization_515/AssignMovingAvg'batch_normalization_515/AssignMovingAvg2p
6batch_normalization_515/AssignMovingAvg/ReadVariableOp6batch_normalization_515/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_515/AssignMovingAvg_1)batch_normalization_515/AssignMovingAvg_12t
8batch_normalization_515/AssignMovingAvg_1/ReadVariableOp8batch_normalization_515/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_515/batchnorm/ReadVariableOp0batch_normalization_515/batchnorm/ReadVariableOp2l
4batch_normalization_515/batchnorm/mul/ReadVariableOp4batch_normalization_515/batchnorm/mul/ReadVariableOp2D
 dense_560/BiasAdd/ReadVariableOp dense_560/BiasAdd/ReadVariableOp2B
dense_560/MatMul/ReadVariableOpdense_560/MatMul/ReadVariableOp2D
 dense_561/BiasAdd/ReadVariableOp dense_561/BiasAdd/ReadVariableOp2B
dense_561/MatMul/ReadVariableOpdense_561/MatMul/ReadVariableOp2D
 dense_562/BiasAdd/ReadVariableOp dense_562/BiasAdd/ReadVariableOp2B
dense_562/MatMul/ReadVariableOpdense_562/MatMul/ReadVariableOp2D
 dense_563/BiasAdd/ReadVariableOp dense_563/BiasAdd/ReadVariableOp2B
dense_563/MatMul/ReadVariableOpdense_563/MatMul/ReadVariableOp2D
 dense_564/BiasAdd/ReadVariableOp dense_564/BiasAdd/ReadVariableOp2B
dense_564/MatMul/ReadVariableOpdense_564/MatMul/ReadVariableOp2D
 dense_565/BiasAdd/ReadVariableOp dense_565/BiasAdd/ReadVariableOp2B
dense_565/MatMul/ReadVariableOpdense_565/MatMul/ReadVariableOp2D
 dense_566/BiasAdd/ReadVariableOp dense_566/BiasAdd/ReadVariableOp2B
dense_566/MatMul/ReadVariableOpdense_566/MatMul/ReadVariableOp2D
 dense_567/BiasAdd/ReadVariableOp dense_567/BiasAdd/ReadVariableOp2B
dense_567/MatMul/ReadVariableOpdense_567/MatMul/ReadVariableOp2D
 dense_568/BiasAdd/ReadVariableOp dense_568/BiasAdd/ReadVariableOp2B
dense_568/MatMul/ReadVariableOpdense_568/MatMul/ReadVariableOp2D
 dense_569/BiasAdd/ReadVariableOp dense_569/BiasAdd/ReadVariableOp2B
dense_569/MatMul/ReadVariableOpdense_569/MatMul/ReadVariableOp2D
 dense_570/BiasAdd/ReadVariableOp dense_570/BiasAdd/ReadVariableOp2B
dense_570/MatMul/ReadVariableOpdense_570/MatMul/ReadVariableOp2D
 dense_571/BiasAdd/ReadVariableOp dense_571/BiasAdd/ReadVariableOp2B
dense_571/MatMul/ReadVariableOpdense_571/MatMul/ReadVariableOp2D
 dense_572/BiasAdd/ReadVariableOp dense_572/BiasAdd/ReadVariableOp2B
dense_572/MatMul/ReadVariableOpdense_572/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
æ
h
L__inference_leaky_re_lu_508_layer_call_and_return_conditional_losses_1010035

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_515_layer_call_and_return_conditional_losses_1014280

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
× 
¹
%__inference_signature_wrapper_1012925
normalization_56_input
unknown
	unknown_0
	unknown_1:A
	unknown_2:A
	unknown_3:A
	unknown_4:A
	unknown_5:A
	unknown_6:A
	unknown_7:AA
	unknown_8:A
	unknown_9:A

unknown_10:A

unknown_11:A

unknown_12:A

unknown_13:AA

unknown_14:A

unknown_15:A

unknown_16:A

unknown_17:A

unknown_18:A

unknown_19:AA

unknown_20:A

unknown_21:A

unknown_22:A

unknown_23:A

unknown_24:A

unknown_25:A

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:5

unknown_44:5

unknown_45:5

unknown_46:5

unknown_47:5

unknown_48:5

unknown_49:55

unknown_50:5

unknown_51:5

unknown_52:5

unknown_53:5

unknown_54:5

unknown_55:55

unknown_56:5

unknown_57:5

unknown_58:5

unknown_59:5

unknown_60:5

unknown_61:55

unknown_62:5

unknown_63:5

unknown_64:5

unknown_65:5

unknown_66:5

unknown_67:55

unknown_68:5

unknown_69:5

unknown_70:5

unknown_71:5

unknown_72:5

unknown_73:5

unknown_74:
identity¢StatefulPartitionedCallÈ

StatefulPartitionedCallStatefulPartitionedCallnormalization_56_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74*X
TinQ
O2M*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*l
_read_only_resource_inputsN
LJ	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKL*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1008879o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ð
_input_shapes¾
»:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_56_input:$ 

_output_shapes

::$ 

_output_shapes

:
Æ

+__inference_dense_564_layer_call_fn_1013417

inputs
unknown:A
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_564_layer_call_and_return_conditional_losses_1010015o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
¨
D
J__inference_sequential_56_layer_call_and_return_conditional_losses_1012305

inputs
normalization_56_sub_y
normalization_56_sqrt_x:
(dense_560_matmul_readvariableop_resource:A7
)dense_560_biasadd_readvariableop_resource:AG
9batch_normalization_504_batchnorm_readvariableop_resource:AK
=batch_normalization_504_batchnorm_mul_readvariableop_resource:AI
;batch_normalization_504_batchnorm_readvariableop_1_resource:AI
;batch_normalization_504_batchnorm_readvariableop_2_resource:A:
(dense_561_matmul_readvariableop_resource:AA7
)dense_561_biasadd_readvariableop_resource:AG
9batch_normalization_505_batchnorm_readvariableop_resource:AK
=batch_normalization_505_batchnorm_mul_readvariableop_resource:AI
;batch_normalization_505_batchnorm_readvariableop_1_resource:AI
;batch_normalization_505_batchnorm_readvariableop_2_resource:A:
(dense_562_matmul_readvariableop_resource:AA7
)dense_562_biasadd_readvariableop_resource:AG
9batch_normalization_506_batchnorm_readvariableop_resource:AK
=batch_normalization_506_batchnorm_mul_readvariableop_resource:AI
;batch_normalization_506_batchnorm_readvariableop_1_resource:AI
;batch_normalization_506_batchnorm_readvariableop_2_resource:A:
(dense_563_matmul_readvariableop_resource:AA7
)dense_563_biasadd_readvariableop_resource:AG
9batch_normalization_507_batchnorm_readvariableop_resource:AK
=batch_normalization_507_batchnorm_mul_readvariableop_resource:AI
;batch_normalization_507_batchnorm_readvariableop_1_resource:AI
;batch_normalization_507_batchnorm_readvariableop_2_resource:A:
(dense_564_matmul_readvariableop_resource:A7
)dense_564_biasadd_readvariableop_resource:G
9batch_normalization_508_batchnorm_readvariableop_resource:K
=batch_normalization_508_batchnorm_mul_readvariableop_resource:I
;batch_normalization_508_batchnorm_readvariableop_1_resource:I
;batch_normalization_508_batchnorm_readvariableop_2_resource::
(dense_565_matmul_readvariableop_resource:7
)dense_565_biasadd_readvariableop_resource:G
9batch_normalization_509_batchnorm_readvariableop_resource:K
=batch_normalization_509_batchnorm_mul_readvariableop_resource:I
;batch_normalization_509_batchnorm_readvariableop_1_resource:I
;batch_normalization_509_batchnorm_readvariableop_2_resource::
(dense_566_matmul_readvariableop_resource:7
)dense_566_biasadd_readvariableop_resource:G
9batch_normalization_510_batchnorm_readvariableop_resource:K
=batch_normalization_510_batchnorm_mul_readvariableop_resource:I
;batch_normalization_510_batchnorm_readvariableop_1_resource:I
;batch_normalization_510_batchnorm_readvariableop_2_resource::
(dense_567_matmul_readvariableop_resource:57
)dense_567_biasadd_readvariableop_resource:5G
9batch_normalization_511_batchnorm_readvariableop_resource:5K
=batch_normalization_511_batchnorm_mul_readvariableop_resource:5I
;batch_normalization_511_batchnorm_readvariableop_1_resource:5I
;batch_normalization_511_batchnorm_readvariableop_2_resource:5:
(dense_568_matmul_readvariableop_resource:557
)dense_568_biasadd_readvariableop_resource:5G
9batch_normalization_512_batchnorm_readvariableop_resource:5K
=batch_normalization_512_batchnorm_mul_readvariableop_resource:5I
;batch_normalization_512_batchnorm_readvariableop_1_resource:5I
;batch_normalization_512_batchnorm_readvariableop_2_resource:5:
(dense_569_matmul_readvariableop_resource:557
)dense_569_biasadd_readvariableop_resource:5G
9batch_normalization_513_batchnorm_readvariableop_resource:5K
=batch_normalization_513_batchnorm_mul_readvariableop_resource:5I
;batch_normalization_513_batchnorm_readvariableop_1_resource:5I
;batch_normalization_513_batchnorm_readvariableop_2_resource:5:
(dense_570_matmul_readvariableop_resource:557
)dense_570_biasadd_readvariableop_resource:5G
9batch_normalization_514_batchnorm_readvariableop_resource:5K
=batch_normalization_514_batchnorm_mul_readvariableop_resource:5I
;batch_normalization_514_batchnorm_readvariableop_1_resource:5I
;batch_normalization_514_batchnorm_readvariableop_2_resource:5:
(dense_571_matmul_readvariableop_resource:557
)dense_571_biasadd_readvariableop_resource:5G
9batch_normalization_515_batchnorm_readvariableop_resource:5K
=batch_normalization_515_batchnorm_mul_readvariableop_resource:5I
;batch_normalization_515_batchnorm_readvariableop_1_resource:5I
;batch_normalization_515_batchnorm_readvariableop_2_resource:5:
(dense_572_matmul_readvariableop_resource:57
)dense_572_biasadd_readvariableop_resource:
identity¢0batch_normalization_504/batchnorm/ReadVariableOp¢2batch_normalization_504/batchnorm/ReadVariableOp_1¢2batch_normalization_504/batchnorm/ReadVariableOp_2¢4batch_normalization_504/batchnorm/mul/ReadVariableOp¢0batch_normalization_505/batchnorm/ReadVariableOp¢2batch_normalization_505/batchnorm/ReadVariableOp_1¢2batch_normalization_505/batchnorm/ReadVariableOp_2¢4batch_normalization_505/batchnorm/mul/ReadVariableOp¢0batch_normalization_506/batchnorm/ReadVariableOp¢2batch_normalization_506/batchnorm/ReadVariableOp_1¢2batch_normalization_506/batchnorm/ReadVariableOp_2¢4batch_normalization_506/batchnorm/mul/ReadVariableOp¢0batch_normalization_507/batchnorm/ReadVariableOp¢2batch_normalization_507/batchnorm/ReadVariableOp_1¢2batch_normalization_507/batchnorm/ReadVariableOp_2¢4batch_normalization_507/batchnorm/mul/ReadVariableOp¢0batch_normalization_508/batchnorm/ReadVariableOp¢2batch_normalization_508/batchnorm/ReadVariableOp_1¢2batch_normalization_508/batchnorm/ReadVariableOp_2¢4batch_normalization_508/batchnorm/mul/ReadVariableOp¢0batch_normalization_509/batchnorm/ReadVariableOp¢2batch_normalization_509/batchnorm/ReadVariableOp_1¢2batch_normalization_509/batchnorm/ReadVariableOp_2¢4batch_normalization_509/batchnorm/mul/ReadVariableOp¢0batch_normalization_510/batchnorm/ReadVariableOp¢2batch_normalization_510/batchnorm/ReadVariableOp_1¢2batch_normalization_510/batchnorm/ReadVariableOp_2¢4batch_normalization_510/batchnorm/mul/ReadVariableOp¢0batch_normalization_511/batchnorm/ReadVariableOp¢2batch_normalization_511/batchnorm/ReadVariableOp_1¢2batch_normalization_511/batchnorm/ReadVariableOp_2¢4batch_normalization_511/batchnorm/mul/ReadVariableOp¢0batch_normalization_512/batchnorm/ReadVariableOp¢2batch_normalization_512/batchnorm/ReadVariableOp_1¢2batch_normalization_512/batchnorm/ReadVariableOp_2¢4batch_normalization_512/batchnorm/mul/ReadVariableOp¢0batch_normalization_513/batchnorm/ReadVariableOp¢2batch_normalization_513/batchnorm/ReadVariableOp_1¢2batch_normalization_513/batchnorm/ReadVariableOp_2¢4batch_normalization_513/batchnorm/mul/ReadVariableOp¢0batch_normalization_514/batchnorm/ReadVariableOp¢2batch_normalization_514/batchnorm/ReadVariableOp_1¢2batch_normalization_514/batchnorm/ReadVariableOp_2¢4batch_normalization_514/batchnorm/mul/ReadVariableOp¢0batch_normalization_515/batchnorm/ReadVariableOp¢2batch_normalization_515/batchnorm/ReadVariableOp_1¢2batch_normalization_515/batchnorm/ReadVariableOp_2¢4batch_normalization_515/batchnorm/mul/ReadVariableOp¢ dense_560/BiasAdd/ReadVariableOp¢dense_560/MatMul/ReadVariableOp¢ dense_561/BiasAdd/ReadVariableOp¢dense_561/MatMul/ReadVariableOp¢ dense_562/BiasAdd/ReadVariableOp¢dense_562/MatMul/ReadVariableOp¢ dense_563/BiasAdd/ReadVariableOp¢dense_563/MatMul/ReadVariableOp¢ dense_564/BiasAdd/ReadVariableOp¢dense_564/MatMul/ReadVariableOp¢ dense_565/BiasAdd/ReadVariableOp¢dense_565/MatMul/ReadVariableOp¢ dense_566/BiasAdd/ReadVariableOp¢dense_566/MatMul/ReadVariableOp¢ dense_567/BiasAdd/ReadVariableOp¢dense_567/MatMul/ReadVariableOp¢ dense_568/BiasAdd/ReadVariableOp¢dense_568/MatMul/ReadVariableOp¢ dense_569/BiasAdd/ReadVariableOp¢dense_569/MatMul/ReadVariableOp¢ dense_570/BiasAdd/ReadVariableOp¢dense_570/MatMul/ReadVariableOp¢ dense_571/BiasAdd/ReadVariableOp¢dense_571/MatMul/ReadVariableOp¢ dense_572/BiasAdd/ReadVariableOp¢dense_572/MatMul/ReadVariableOpm
normalization_56/subSubinputsnormalization_56_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_56/SqrtSqrtnormalization_56_sqrt_x*
T0*
_output_shapes

:_
normalization_56/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_56/MaximumMaximumnormalization_56/Sqrt:y:0#normalization_56/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_56/truedivRealDivnormalization_56/sub:z:0normalization_56/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_560/MatMul/ReadVariableOpReadVariableOp(dense_560_matmul_readvariableop_resource*
_output_shapes

:A*
dtype0
dense_560/MatMulMatMulnormalization_56/truediv:z:0'dense_560/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_560/BiasAdd/ReadVariableOpReadVariableOp)dense_560_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_560/BiasAddBiasAdddense_560/MatMul:product:0(dense_560/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¦
0batch_normalization_504/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_504_batchnorm_readvariableop_resource*
_output_shapes
:A*
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
:A
'batch_normalization_504/batchnorm/RsqrtRsqrt)batch_normalization_504/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_504/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_504_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_504/batchnorm/mulMul+batch_normalization_504/batchnorm/Rsqrt:y:0<batch_normalization_504/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_504/batchnorm/mul_1Muldense_560/BiasAdd:output:0)batch_normalization_504/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAª
2batch_normalization_504/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_504_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0º
'batch_normalization_504/batchnorm/mul_2Mul:batch_normalization_504/batchnorm/ReadVariableOp_1:value:0)batch_normalization_504/batchnorm/mul:z:0*
T0*
_output_shapes
:Aª
2batch_normalization_504/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_504_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0º
%batch_normalization_504/batchnorm/subSub:batch_normalization_504/batchnorm/ReadVariableOp_2:value:0+batch_normalization_504/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_504/batchnorm/add_1AddV2+batch_normalization_504/batchnorm/mul_1:z:0)batch_normalization_504/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_504/LeakyRelu	LeakyRelu+batch_normalization_504/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_561/MatMul/ReadVariableOpReadVariableOp(dense_561_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_561/MatMulMatMul'leaky_re_lu_504/LeakyRelu:activations:0'dense_561/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_561/BiasAdd/ReadVariableOpReadVariableOp)dense_561_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_561/BiasAddBiasAdddense_561/MatMul:product:0(dense_561/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¦
0batch_normalization_505/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_505_batchnorm_readvariableop_resource*
_output_shapes
:A*
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
:A
'batch_normalization_505/batchnorm/RsqrtRsqrt)batch_normalization_505/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_505/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_505_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_505/batchnorm/mulMul+batch_normalization_505/batchnorm/Rsqrt:y:0<batch_normalization_505/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_505/batchnorm/mul_1Muldense_561/BiasAdd:output:0)batch_normalization_505/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAª
2batch_normalization_505/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_505_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0º
'batch_normalization_505/batchnorm/mul_2Mul:batch_normalization_505/batchnorm/ReadVariableOp_1:value:0)batch_normalization_505/batchnorm/mul:z:0*
T0*
_output_shapes
:Aª
2batch_normalization_505/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_505_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0º
%batch_normalization_505/batchnorm/subSub:batch_normalization_505/batchnorm/ReadVariableOp_2:value:0+batch_normalization_505/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_505/batchnorm/add_1AddV2+batch_normalization_505/batchnorm/mul_1:z:0)batch_normalization_505/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_505/LeakyRelu	LeakyRelu+batch_normalization_505/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_562/MatMul/ReadVariableOpReadVariableOp(dense_562_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_562/MatMulMatMul'leaky_re_lu_505/LeakyRelu:activations:0'dense_562/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_562/BiasAdd/ReadVariableOpReadVariableOp)dense_562_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_562/BiasAddBiasAdddense_562/MatMul:product:0(dense_562/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¦
0batch_normalization_506/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_506_batchnorm_readvariableop_resource*
_output_shapes
:A*
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
:A
'batch_normalization_506/batchnorm/RsqrtRsqrt)batch_normalization_506/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_506/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_506_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_506/batchnorm/mulMul+batch_normalization_506/batchnorm/Rsqrt:y:0<batch_normalization_506/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_506/batchnorm/mul_1Muldense_562/BiasAdd:output:0)batch_normalization_506/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAª
2batch_normalization_506/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_506_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0º
'batch_normalization_506/batchnorm/mul_2Mul:batch_normalization_506/batchnorm/ReadVariableOp_1:value:0)batch_normalization_506/batchnorm/mul:z:0*
T0*
_output_shapes
:Aª
2batch_normalization_506/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_506_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0º
%batch_normalization_506/batchnorm/subSub:batch_normalization_506/batchnorm/ReadVariableOp_2:value:0+batch_normalization_506/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_506/batchnorm/add_1AddV2+batch_normalization_506/batchnorm/mul_1:z:0)batch_normalization_506/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_506/LeakyRelu	LeakyRelu+batch_normalization_506/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_563/MatMul/ReadVariableOpReadVariableOp(dense_563_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_563/MatMulMatMul'leaky_re_lu_506/LeakyRelu:activations:0'dense_563/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_563/BiasAdd/ReadVariableOpReadVariableOp)dense_563_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_563/BiasAddBiasAdddense_563/MatMul:product:0(dense_563/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¦
0batch_normalization_507/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_507_batchnorm_readvariableop_resource*
_output_shapes
:A*
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
:A
'batch_normalization_507/batchnorm/RsqrtRsqrt)batch_normalization_507/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_507/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_507_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_507/batchnorm/mulMul+batch_normalization_507/batchnorm/Rsqrt:y:0<batch_normalization_507/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_507/batchnorm/mul_1Muldense_563/BiasAdd:output:0)batch_normalization_507/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAª
2batch_normalization_507/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_507_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0º
'batch_normalization_507/batchnorm/mul_2Mul:batch_normalization_507/batchnorm/ReadVariableOp_1:value:0)batch_normalization_507/batchnorm/mul:z:0*
T0*
_output_shapes
:Aª
2batch_normalization_507/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_507_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0º
%batch_normalization_507/batchnorm/subSub:batch_normalization_507/batchnorm/ReadVariableOp_2:value:0+batch_normalization_507/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_507/batchnorm/add_1AddV2+batch_normalization_507/batchnorm/mul_1:z:0)batch_normalization_507/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_507/LeakyRelu	LeakyRelu+batch_normalization_507/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_564/MatMul/ReadVariableOpReadVariableOp(dense_564_matmul_readvariableop_resource*
_output_shapes

:A*
dtype0
dense_564/MatMulMatMul'leaky_re_lu_507/LeakyRelu:activations:0'dense_564/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_564/BiasAdd/ReadVariableOpReadVariableOp)dense_564_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_564/BiasAddBiasAdddense_564/MatMul:product:0(dense_564/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_508/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_508_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_508/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_508/batchnorm/addAddV28batch_normalization_508/batchnorm/ReadVariableOp:value:00batch_normalization_508/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_508/batchnorm/RsqrtRsqrt)batch_normalization_508/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_508/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_508_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_508/batchnorm/mulMul+batch_normalization_508/batchnorm/Rsqrt:y:0<batch_normalization_508/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_508/batchnorm/mul_1Muldense_564/BiasAdd:output:0)batch_normalization_508/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_508/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_508_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_508/batchnorm/mul_2Mul:batch_normalization_508/batchnorm/ReadVariableOp_1:value:0)batch_normalization_508/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_508/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_508_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_508/batchnorm/subSub:batch_normalization_508/batchnorm/ReadVariableOp_2:value:0+batch_normalization_508/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_508/batchnorm/add_1AddV2+batch_normalization_508/batchnorm/mul_1:z:0)batch_normalization_508/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_508/LeakyRelu	LeakyRelu+batch_normalization_508/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_565/MatMul/ReadVariableOpReadVariableOp(dense_565_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_565/MatMulMatMul'leaky_re_lu_508/LeakyRelu:activations:0'dense_565/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_565/BiasAdd/ReadVariableOpReadVariableOp)dense_565_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_565/BiasAddBiasAdddense_565/MatMul:product:0(dense_565/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_509/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_509_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_509/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_509/batchnorm/addAddV28batch_normalization_509/batchnorm/ReadVariableOp:value:00batch_normalization_509/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_509/batchnorm/RsqrtRsqrt)batch_normalization_509/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_509/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_509_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_509/batchnorm/mulMul+batch_normalization_509/batchnorm/Rsqrt:y:0<batch_normalization_509/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_509/batchnorm/mul_1Muldense_565/BiasAdd:output:0)batch_normalization_509/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_509/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_509_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_509/batchnorm/mul_2Mul:batch_normalization_509/batchnorm/ReadVariableOp_1:value:0)batch_normalization_509/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_509/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_509_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_509/batchnorm/subSub:batch_normalization_509/batchnorm/ReadVariableOp_2:value:0+batch_normalization_509/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_509/batchnorm/add_1AddV2+batch_normalization_509/batchnorm/mul_1:z:0)batch_normalization_509/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_509/LeakyRelu	LeakyRelu+batch_normalization_509/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_566/MatMul/ReadVariableOpReadVariableOp(dense_566_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_566/MatMulMatMul'leaky_re_lu_509/LeakyRelu:activations:0'dense_566/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_566/BiasAdd/ReadVariableOpReadVariableOp)dense_566_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_566/BiasAddBiasAdddense_566/MatMul:product:0(dense_566/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_510/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_510_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_510/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_510/batchnorm/addAddV28batch_normalization_510/batchnorm/ReadVariableOp:value:00batch_normalization_510/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_510/batchnorm/RsqrtRsqrt)batch_normalization_510/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_510/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_510_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_510/batchnorm/mulMul+batch_normalization_510/batchnorm/Rsqrt:y:0<batch_normalization_510/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_510/batchnorm/mul_1Muldense_566/BiasAdd:output:0)batch_normalization_510/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_510/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_510_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_510/batchnorm/mul_2Mul:batch_normalization_510/batchnorm/ReadVariableOp_1:value:0)batch_normalization_510/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_510/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_510_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_510/batchnorm/subSub:batch_normalization_510/batchnorm/ReadVariableOp_2:value:0+batch_normalization_510/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_510/batchnorm/add_1AddV2+batch_normalization_510/batchnorm/mul_1:z:0)batch_normalization_510/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_510/LeakyRelu	LeakyRelu+batch_normalization_510/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_567/MatMul/ReadVariableOpReadVariableOp(dense_567_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0
dense_567/MatMulMatMul'leaky_re_lu_510/LeakyRelu:activations:0'dense_567/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 dense_567/BiasAdd/ReadVariableOpReadVariableOp)dense_567_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_567/BiasAddBiasAdddense_567/MatMul:product:0(dense_567/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¦
0batch_normalization_511/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_511_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0l
'batch_normalization_511/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_511/batchnorm/addAddV28batch_normalization_511/batchnorm/ReadVariableOp:value:00batch_normalization_511/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_511/batchnorm/RsqrtRsqrt)batch_normalization_511/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_511/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_511_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_511/batchnorm/mulMul+batch_normalization_511/batchnorm/Rsqrt:y:0<batch_normalization_511/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5§
'batch_normalization_511/batchnorm/mul_1Muldense_567/BiasAdd:output:0)batch_normalization_511/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ª
2batch_normalization_511/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_511_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0º
'batch_normalization_511/batchnorm/mul_2Mul:batch_normalization_511/batchnorm/ReadVariableOp_1:value:0)batch_normalization_511/batchnorm/mul:z:0*
T0*
_output_shapes
:5ª
2batch_normalization_511/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_511_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0º
%batch_normalization_511/batchnorm/subSub:batch_normalization_511/batchnorm/ReadVariableOp_2:value:0+batch_normalization_511/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_511/batchnorm/add_1AddV2+batch_normalization_511/batchnorm/mul_1:z:0)batch_normalization_511/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_511/LeakyRelu	LeakyRelu+batch_normalization_511/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
dense_568/MatMul/ReadVariableOpReadVariableOp(dense_568_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0
dense_568/MatMulMatMul'leaky_re_lu_511/LeakyRelu:activations:0'dense_568/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 dense_568/BiasAdd/ReadVariableOpReadVariableOp)dense_568_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_568/BiasAddBiasAdddense_568/MatMul:product:0(dense_568/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¦
0batch_normalization_512/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_512_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0l
'batch_normalization_512/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_512/batchnorm/addAddV28batch_normalization_512/batchnorm/ReadVariableOp:value:00batch_normalization_512/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_512/batchnorm/RsqrtRsqrt)batch_normalization_512/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_512/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_512_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_512/batchnorm/mulMul+batch_normalization_512/batchnorm/Rsqrt:y:0<batch_normalization_512/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5§
'batch_normalization_512/batchnorm/mul_1Muldense_568/BiasAdd:output:0)batch_normalization_512/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ª
2batch_normalization_512/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_512_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0º
'batch_normalization_512/batchnorm/mul_2Mul:batch_normalization_512/batchnorm/ReadVariableOp_1:value:0)batch_normalization_512/batchnorm/mul:z:0*
T0*
_output_shapes
:5ª
2batch_normalization_512/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_512_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0º
%batch_normalization_512/batchnorm/subSub:batch_normalization_512/batchnorm/ReadVariableOp_2:value:0+batch_normalization_512/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_512/batchnorm/add_1AddV2+batch_normalization_512/batchnorm/mul_1:z:0)batch_normalization_512/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_512/LeakyRelu	LeakyRelu+batch_normalization_512/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
dense_569/MatMul/ReadVariableOpReadVariableOp(dense_569_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0
dense_569/MatMulMatMul'leaky_re_lu_512/LeakyRelu:activations:0'dense_569/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 dense_569/BiasAdd/ReadVariableOpReadVariableOp)dense_569_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_569/BiasAddBiasAdddense_569/MatMul:product:0(dense_569/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¦
0batch_normalization_513/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_513_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0l
'batch_normalization_513/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_513/batchnorm/addAddV28batch_normalization_513/batchnorm/ReadVariableOp:value:00batch_normalization_513/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_513/batchnorm/RsqrtRsqrt)batch_normalization_513/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_513/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_513_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_513/batchnorm/mulMul+batch_normalization_513/batchnorm/Rsqrt:y:0<batch_normalization_513/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5§
'batch_normalization_513/batchnorm/mul_1Muldense_569/BiasAdd:output:0)batch_normalization_513/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ª
2batch_normalization_513/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_513_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0º
'batch_normalization_513/batchnorm/mul_2Mul:batch_normalization_513/batchnorm/ReadVariableOp_1:value:0)batch_normalization_513/batchnorm/mul:z:0*
T0*
_output_shapes
:5ª
2batch_normalization_513/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_513_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0º
%batch_normalization_513/batchnorm/subSub:batch_normalization_513/batchnorm/ReadVariableOp_2:value:0+batch_normalization_513/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_513/batchnorm/add_1AddV2+batch_normalization_513/batchnorm/mul_1:z:0)batch_normalization_513/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_513/LeakyRelu	LeakyRelu+batch_normalization_513/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
dense_570/MatMul/ReadVariableOpReadVariableOp(dense_570_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0
dense_570/MatMulMatMul'leaky_re_lu_513/LeakyRelu:activations:0'dense_570/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 dense_570/BiasAdd/ReadVariableOpReadVariableOp)dense_570_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_570/BiasAddBiasAdddense_570/MatMul:product:0(dense_570/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¦
0batch_normalization_514/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_514_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0l
'batch_normalization_514/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_514/batchnorm/addAddV28batch_normalization_514/batchnorm/ReadVariableOp:value:00batch_normalization_514/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_514/batchnorm/RsqrtRsqrt)batch_normalization_514/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_514/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_514_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_514/batchnorm/mulMul+batch_normalization_514/batchnorm/Rsqrt:y:0<batch_normalization_514/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5§
'batch_normalization_514/batchnorm/mul_1Muldense_570/BiasAdd:output:0)batch_normalization_514/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ª
2batch_normalization_514/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_514_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0º
'batch_normalization_514/batchnorm/mul_2Mul:batch_normalization_514/batchnorm/ReadVariableOp_1:value:0)batch_normalization_514/batchnorm/mul:z:0*
T0*
_output_shapes
:5ª
2batch_normalization_514/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_514_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0º
%batch_normalization_514/batchnorm/subSub:batch_normalization_514/batchnorm/ReadVariableOp_2:value:0+batch_normalization_514/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_514/batchnorm/add_1AddV2+batch_normalization_514/batchnorm/mul_1:z:0)batch_normalization_514/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_514/LeakyRelu	LeakyRelu+batch_normalization_514/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
dense_571/MatMul/ReadVariableOpReadVariableOp(dense_571_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0
dense_571/MatMulMatMul'leaky_re_lu_514/LeakyRelu:activations:0'dense_571/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 dense_571/BiasAdd/ReadVariableOpReadVariableOp)dense_571_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_571/BiasAddBiasAdddense_571/MatMul:product:0(dense_571/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¦
0batch_normalization_515/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_515_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0l
'batch_normalization_515/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_515/batchnorm/addAddV28batch_normalization_515/batchnorm/ReadVariableOp:value:00batch_normalization_515/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_515/batchnorm/RsqrtRsqrt)batch_normalization_515/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_515/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_515_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_515/batchnorm/mulMul+batch_normalization_515/batchnorm/Rsqrt:y:0<batch_normalization_515/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5§
'batch_normalization_515/batchnorm/mul_1Muldense_571/BiasAdd:output:0)batch_normalization_515/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ª
2batch_normalization_515/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_515_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0º
'batch_normalization_515/batchnorm/mul_2Mul:batch_normalization_515/batchnorm/ReadVariableOp_1:value:0)batch_normalization_515/batchnorm/mul:z:0*
T0*
_output_shapes
:5ª
2batch_normalization_515/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_515_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0º
%batch_normalization_515/batchnorm/subSub:batch_normalization_515/batchnorm/ReadVariableOp_2:value:0+batch_normalization_515/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_515/batchnorm/add_1AddV2+batch_normalization_515/batchnorm/mul_1:z:0)batch_normalization_515/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_515/LeakyRelu	LeakyRelu+batch_normalization_515/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
dense_572/MatMul/ReadVariableOpReadVariableOp(dense_572_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0
dense_572/MatMulMatMul'leaky_re_lu_515/LeakyRelu:activations:0'dense_572/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_572/BiasAdd/ReadVariableOpReadVariableOp)dense_572_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_572/BiasAddBiasAdddense_572/MatMul:product:0(dense_572/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_572/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp1^batch_normalization_504/batchnorm/ReadVariableOp3^batch_normalization_504/batchnorm/ReadVariableOp_13^batch_normalization_504/batchnorm/ReadVariableOp_25^batch_normalization_504/batchnorm/mul/ReadVariableOp1^batch_normalization_505/batchnorm/ReadVariableOp3^batch_normalization_505/batchnorm/ReadVariableOp_13^batch_normalization_505/batchnorm/ReadVariableOp_25^batch_normalization_505/batchnorm/mul/ReadVariableOp1^batch_normalization_506/batchnorm/ReadVariableOp3^batch_normalization_506/batchnorm/ReadVariableOp_13^batch_normalization_506/batchnorm/ReadVariableOp_25^batch_normalization_506/batchnorm/mul/ReadVariableOp1^batch_normalization_507/batchnorm/ReadVariableOp3^batch_normalization_507/batchnorm/ReadVariableOp_13^batch_normalization_507/batchnorm/ReadVariableOp_25^batch_normalization_507/batchnorm/mul/ReadVariableOp1^batch_normalization_508/batchnorm/ReadVariableOp3^batch_normalization_508/batchnorm/ReadVariableOp_13^batch_normalization_508/batchnorm/ReadVariableOp_25^batch_normalization_508/batchnorm/mul/ReadVariableOp1^batch_normalization_509/batchnorm/ReadVariableOp3^batch_normalization_509/batchnorm/ReadVariableOp_13^batch_normalization_509/batchnorm/ReadVariableOp_25^batch_normalization_509/batchnorm/mul/ReadVariableOp1^batch_normalization_510/batchnorm/ReadVariableOp3^batch_normalization_510/batchnorm/ReadVariableOp_13^batch_normalization_510/batchnorm/ReadVariableOp_25^batch_normalization_510/batchnorm/mul/ReadVariableOp1^batch_normalization_511/batchnorm/ReadVariableOp3^batch_normalization_511/batchnorm/ReadVariableOp_13^batch_normalization_511/batchnorm/ReadVariableOp_25^batch_normalization_511/batchnorm/mul/ReadVariableOp1^batch_normalization_512/batchnorm/ReadVariableOp3^batch_normalization_512/batchnorm/ReadVariableOp_13^batch_normalization_512/batchnorm/ReadVariableOp_25^batch_normalization_512/batchnorm/mul/ReadVariableOp1^batch_normalization_513/batchnorm/ReadVariableOp3^batch_normalization_513/batchnorm/ReadVariableOp_13^batch_normalization_513/batchnorm/ReadVariableOp_25^batch_normalization_513/batchnorm/mul/ReadVariableOp1^batch_normalization_514/batchnorm/ReadVariableOp3^batch_normalization_514/batchnorm/ReadVariableOp_13^batch_normalization_514/batchnorm/ReadVariableOp_25^batch_normalization_514/batchnorm/mul/ReadVariableOp1^batch_normalization_515/batchnorm/ReadVariableOp3^batch_normalization_515/batchnorm/ReadVariableOp_13^batch_normalization_515/batchnorm/ReadVariableOp_25^batch_normalization_515/batchnorm/mul/ReadVariableOp!^dense_560/BiasAdd/ReadVariableOp ^dense_560/MatMul/ReadVariableOp!^dense_561/BiasAdd/ReadVariableOp ^dense_561/MatMul/ReadVariableOp!^dense_562/BiasAdd/ReadVariableOp ^dense_562/MatMul/ReadVariableOp!^dense_563/BiasAdd/ReadVariableOp ^dense_563/MatMul/ReadVariableOp!^dense_564/BiasAdd/ReadVariableOp ^dense_564/MatMul/ReadVariableOp!^dense_565/BiasAdd/ReadVariableOp ^dense_565/MatMul/ReadVariableOp!^dense_566/BiasAdd/ReadVariableOp ^dense_566/MatMul/ReadVariableOp!^dense_567/BiasAdd/ReadVariableOp ^dense_567/MatMul/ReadVariableOp!^dense_568/BiasAdd/ReadVariableOp ^dense_568/MatMul/ReadVariableOp!^dense_569/BiasAdd/ReadVariableOp ^dense_569/MatMul/ReadVariableOp!^dense_570/BiasAdd/ReadVariableOp ^dense_570/MatMul/ReadVariableOp!^dense_571/BiasAdd/ReadVariableOp ^dense_571/MatMul/ReadVariableOp!^dense_572/BiasAdd/ReadVariableOp ^dense_572/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ð
_input_shapes¾
»:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
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
4batch_normalization_507/batchnorm/mul/ReadVariableOp4batch_normalization_507/batchnorm/mul/ReadVariableOp2d
0batch_normalization_508/batchnorm/ReadVariableOp0batch_normalization_508/batchnorm/ReadVariableOp2h
2batch_normalization_508/batchnorm/ReadVariableOp_12batch_normalization_508/batchnorm/ReadVariableOp_12h
2batch_normalization_508/batchnorm/ReadVariableOp_22batch_normalization_508/batchnorm/ReadVariableOp_22l
4batch_normalization_508/batchnorm/mul/ReadVariableOp4batch_normalization_508/batchnorm/mul/ReadVariableOp2d
0batch_normalization_509/batchnorm/ReadVariableOp0batch_normalization_509/batchnorm/ReadVariableOp2h
2batch_normalization_509/batchnorm/ReadVariableOp_12batch_normalization_509/batchnorm/ReadVariableOp_12h
2batch_normalization_509/batchnorm/ReadVariableOp_22batch_normalization_509/batchnorm/ReadVariableOp_22l
4batch_normalization_509/batchnorm/mul/ReadVariableOp4batch_normalization_509/batchnorm/mul/ReadVariableOp2d
0batch_normalization_510/batchnorm/ReadVariableOp0batch_normalization_510/batchnorm/ReadVariableOp2h
2batch_normalization_510/batchnorm/ReadVariableOp_12batch_normalization_510/batchnorm/ReadVariableOp_12h
2batch_normalization_510/batchnorm/ReadVariableOp_22batch_normalization_510/batchnorm/ReadVariableOp_22l
4batch_normalization_510/batchnorm/mul/ReadVariableOp4batch_normalization_510/batchnorm/mul/ReadVariableOp2d
0batch_normalization_511/batchnorm/ReadVariableOp0batch_normalization_511/batchnorm/ReadVariableOp2h
2batch_normalization_511/batchnorm/ReadVariableOp_12batch_normalization_511/batchnorm/ReadVariableOp_12h
2batch_normalization_511/batchnorm/ReadVariableOp_22batch_normalization_511/batchnorm/ReadVariableOp_22l
4batch_normalization_511/batchnorm/mul/ReadVariableOp4batch_normalization_511/batchnorm/mul/ReadVariableOp2d
0batch_normalization_512/batchnorm/ReadVariableOp0batch_normalization_512/batchnorm/ReadVariableOp2h
2batch_normalization_512/batchnorm/ReadVariableOp_12batch_normalization_512/batchnorm/ReadVariableOp_12h
2batch_normalization_512/batchnorm/ReadVariableOp_22batch_normalization_512/batchnorm/ReadVariableOp_22l
4batch_normalization_512/batchnorm/mul/ReadVariableOp4batch_normalization_512/batchnorm/mul/ReadVariableOp2d
0batch_normalization_513/batchnorm/ReadVariableOp0batch_normalization_513/batchnorm/ReadVariableOp2h
2batch_normalization_513/batchnorm/ReadVariableOp_12batch_normalization_513/batchnorm/ReadVariableOp_12h
2batch_normalization_513/batchnorm/ReadVariableOp_22batch_normalization_513/batchnorm/ReadVariableOp_22l
4batch_normalization_513/batchnorm/mul/ReadVariableOp4batch_normalization_513/batchnorm/mul/ReadVariableOp2d
0batch_normalization_514/batchnorm/ReadVariableOp0batch_normalization_514/batchnorm/ReadVariableOp2h
2batch_normalization_514/batchnorm/ReadVariableOp_12batch_normalization_514/batchnorm/ReadVariableOp_12h
2batch_normalization_514/batchnorm/ReadVariableOp_22batch_normalization_514/batchnorm/ReadVariableOp_22l
4batch_normalization_514/batchnorm/mul/ReadVariableOp4batch_normalization_514/batchnorm/mul/ReadVariableOp2d
0batch_normalization_515/batchnorm/ReadVariableOp0batch_normalization_515/batchnorm/ReadVariableOp2h
2batch_normalization_515/batchnorm/ReadVariableOp_12batch_normalization_515/batchnorm/ReadVariableOp_12h
2batch_normalization_515/batchnorm/ReadVariableOp_22batch_normalization_515/batchnorm/ReadVariableOp_22l
4batch_normalization_515/batchnorm/mul/ReadVariableOp4batch_normalization_515/batchnorm/mul/ReadVariableOp2D
 dense_560/BiasAdd/ReadVariableOp dense_560/BiasAdd/ReadVariableOp2B
dense_560/MatMul/ReadVariableOpdense_560/MatMul/ReadVariableOp2D
 dense_561/BiasAdd/ReadVariableOp dense_561/BiasAdd/ReadVariableOp2B
dense_561/MatMul/ReadVariableOpdense_561/MatMul/ReadVariableOp2D
 dense_562/BiasAdd/ReadVariableOp dense_562/BiasAdd/ReadVariableOp2B
dense_562/MatMul/ReadVariableOpdense_562/MatMul/ReadVariableOp2D
 dense_563/BiasAdd/ReadVariableOp dense_563/BiasAdd/ReadVariableOp2B
dense_563/MatMul/ReadVariableOpdense_563/MatMul/ReadVariableOp2D
 dense_564/BiasAdd/ReadVariableOp dense_564/BiasAdd/ReadVariableOp2B
dense_564/MatMul/ReadVariableOpdense_564/MatMul/ReadVariableOp2D
 dense_565/BiasAdd/ReadVariableOp dense_565/BiasAdd/ReadVariableOp2B
dense_565/MatMul/ReadVariableOpdense_565/MatMul/ReadVariableOp2D
 dense_566/BiasAdd/ReadVariableOp dense_566/BiasAdd/ReadVariableOp2B
dense_566/MatMul/ReadVariableOpdense_566/MatMul/ReadVariableOp2D
 dense_567/BiasAdd/ReadVariableOp dense_567/BiasAdd/ReadVariableOp2B
dense_567/MatMul/ReadVariableOpdense_567/MatMul/ReadVariableOp2D
 dense_568/BiasAdd/ReadVariableOp dense_568/BiasAdd/ReadVariableOp2B
dense_568/MatMul/ReadVariableOpdense_568/MatMul/ReadVariableOp2D
 dense_569/BiasAdd/ReadVariableOp dense_569/BiasAdd/ReadVariableOp2B
dense_569/MatMul/ReadVariableOpdense_569/MatMul/ReadVariableOp2D
 dense_570/BiasAdd/ReadVariableOp dense_570/BiasAdd/ReadVariableOp2B
dense_570/MatMul/ReadVariableOpdense_570/MatMul/ReadVariableOp2D
 dense_571/BiasAdd/ReadVariableOp dense_571/BiasAdd/ReadVariableOp2B
dense_571/MatMul/ReadVariableOpdense_571/MatMul/ReadVariableOp2D
 dense_572/BiasAdd/ReadVariableOp dense_572/BiasAdd/ReadVariableOp2B
dense_572/MatMul/ReadVariableOpdense_572/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Æ

+__inference_dense_571_layer_call_fn_1014180

inputs
unknown:55
	unknown_0:5
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_571_layer_call_and_return_conditional_losses_1010239o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_512_layer_call_and_return_conditional_losses_1013953

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
_user_specified_nameinputs
èÄ
¬"
J__inference_sequential_56_layer_call_and_return_conditional_losses_1010990

inputs
normalization_56_sub_y
normalization_56_sqrt_x#
dense_560_1010804:A
dense_560_1010806:A-
batch_normalization_504_1010809:A-
batch_normalization_504_1010811:A-
batch_normalization_504_1010813:A-
batch_normalization_504_1010815:A#
dense_561_1010819:AA
dense_561_1010821:A-
batch_normalization_505_1010824:A-
batch_normalization_505_1010826:A-
batch_normalization_505_1010828:A-
batch_normalization_505_1010830:A#
dense_562_1010834:AA
dense_562_1010836:A-
batch_normalization_506_1010839:A-
batch_normalization_506_1010841:A-
batch_normalization_506_1010843:A-
batch_normalization_506_1010845:A#
dense_563_1010849:AA
dense_563_1010851:A-
batch_normalization_507_1010854:A-
batch_normalization_507_1010856:A-
batch_normalization_507_1010858:A-
batch_normalization_507_1010860:A#
dense_564_1010864:A
dense_564_1010866:-
batch_normalization_508_1010869:-
batch_normalization_508_1010871:-
batch_normalization_508_1010873:-
batch_normalization_508_1010875:#
dense_565_1010879:
dense_565_1010881:-
batch_normalization_509_1010884:-
batch_normalization_509_1010886:-
batch_normalization_509_1010888:-
batch_normalization_509_1010890:#
dense_566_1010894:
dense_566_1010896:-
batch_normalization_510_1010899:-
batch_normalization_510_1010901:-
batch_normalization_510_1010903:-
batch_normalization_510_1010905:#
dense_567_1010909:5
dense_567_1010911:5-
batch_normalization_511_1010914:5-
batch_normalization_511_1010916:5-
batch_normalization_511_1010918:5-
batch_normalization_511_1010920:5#
dense_568_1010924:55
dense_568_1010926:5-
batch_normalization_512_1010929:5-
batch_normalization_512_1010931:5-
batch_normalization_512_1010933:5-
batch_normalization_512_1010935:5#
dense_569_1010939:55
dense_569_1010941:5-
batch_normalization_513_1010944:5-
batch_normalization_513_1010946:5-
batch_normalization_513_1010948:5-
batch_normalization_513_1010950:5#
dense_570_1010954:55
dense_570_1010956:5-
batch_normalization_514_1010959:5-
batch_normalization_514_1010961:5-
batch_normalization_514_1010963:5-
batch_normalization_514_1010965:5#
dense_571_1010969:55
dense_571_1010971:5-
batch_normalization_515_1010974:5-
batch_normalization_515_1010976:5-
batch_normalization_515_1010978:5-
batch_normalization_515_1010980:5#
dense_572_1010984:5
dense_572_1010986:
identity¢/batch_normalization_504/StatefulPartitionedCall¢/batch_normalization_505/StatefulPartitionedCall¢/batch_normalization_506/StatefulPartitionedCall¢/batch_normalization_507/StatefulPartitionedCall¢/batch_normalization_508/StatefulPartitionedCall¢/batch_normalization_509/StatefulPartitionedCall¢/batch_normalization_510/StatefulPartitionedCall¢/batch_normalization_511/StatefulPartitionedCall¢/batch_normalization_512/StatefulPartitionedCall¢/batch_normalization_513/StatefulPartitionedCall¢/batch_normalization_514/StatefulPartitionedCall¢/batch_normalization_515/StatefulPartitionedCall¢!dense_560/StatefulPartitionedCall¢!dense_561/StatefulPartitionedCall¢!dense_562/StatefulPartitionedCall¢!dense_563/StatefulPartitionedCall¢!dense_564/StatefulPartitionedCall¢!dense_565/StatefulPartitionedCall¢!dense_566/StatefulPartitionedCall¢!dense_567/StatefulPartitionedCall¢!dense_568/StatefulPartitionedCall¢!dense_569/StatefulPartitionedCall¢!dense_570/StatefulPartitionedCall¢!dense_571/StatefulPartitionedCall¢!dense_572/StatefulPartitionedCallm
normalization_56/subSubinputsnormalization_56_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_56/SqrtSqrtnormalization_56_sqrt_x*
T0*
_output_shapes

:_
normalization_56/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_56/MaximumMaximumnormalization_56/Sqrt:y:0#normalization_56/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_56/truedivRealDivnormalization_56/sub:z:0normalization_56/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_560/StatefulPartitionedCallStatefulPartitionedCallnormalization_56/truediv:z:0dense_560_1010804dense_560_1010806*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_560_layer_call_and_return_conditional_losses_1009887
/batch_normalization_504/StatefulPartitionedCallStatefulPartitionedCall*dense_560/StatefulPartitionedCall:output:0batch_normalization_504_1010809batch_normalization_504_1010811batch_normalization_504_1010813batch_normalization_504_1010815*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1008950ù
leaky_re_lu_504/PartitionedCallPartitionedCall8batch_normalization_504/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_1009907
!dense_561/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_504/PartitionedCall:output:0dense_561_1010819dense_561_1010821*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_561_layer_call_and_return_conditional_losses_1009919
/batch_normalization_505/StatefulPartitionedCallStatefulPartitionedCall*dense_561/StatefulPartitionedCall:output:0batch_normalization_505_1010824batch_normalization_505_1010826batch_normalization_505_1010828batch_normalization_505_1010830*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1009032ù
leaky_re_lu_505/PartitionedCallPartitionedCall8batch_normalization_505/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_1009939
!dense_562/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_505/PartitionedCall:output:0dense_562_1010834dense_562_1010836*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_562_layer_call_and_return_conditional_losses_1009951
/batch_normalization_506/StatefulPartitionedCallStatefulPartitionedCall*dense_562/StatefulPartitionedCall:output:0batch_normalization_506_1010839batch_normalization_506_1010841batch_normalization_506_1010843batch_normalization_506_1010845*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1009114ù
leaky_re_lu_506/PartitionedCallPartitionedCall8batch_normalization_506/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_1009971
!dense_563/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_506/PartitionedCall:output:0dense_563_1010849dense_563_1010851*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_563_layer_call_and_return_conditional_losses_1009983
/batch_normalization_507/StatefulPartitionedCallStatefulPartitionedCall*dense_563/StatefulPartitionedCall:output:0batch_normalization_507_1010854batch_normalization_507_1010856batch_normalization_507_1010858batch_normalization_507_1010860*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1009196ù
leaky_re_lu_507/PartitionedCallPartitionedCall8batch_normalization_507/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_1010003
!dense_564/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_507/PartitionedCall:output:0dense_564_1010864dense_564_1010866*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_564_layer_call_and_return_conditional_losses_1010015
/batch_normalization_508/StatefulPartitionedCallStatefulPartitionedCall*dense_564/StatefulPartitionedCall:output:0batch_normalization_508_1010869batch_normalization_508_1010871batch_normalization_508_1010873batch_normalization_508_1010875*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1009278ù
leaky_re_lu_508/PartitionedCallPartitionedCall8batch_normalization_508/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_508_layer_call_and_return_conditional_losses_1010035
!dense_565/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_508/PartitionedCall:output:0dense_565_1010879dense_565_1010881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_565_layer_call_and_return_conditional_losses_1010047
/batch_normalization_509/StatefulPartitionedCallStatefulPartitionedCall*dense_565/StatefulPartitionedCall:output:0batch_normalization_509_1010884batch_normalization_509_1010886batch_normalization_509_1010888batch_normalization_509_1010890*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1009360ù
leaky_re_lu_509/PartitionedCallPartitionedCall8batch_normalization_509/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_509_layer_call_and_return_conditional_losses_1010067
!dense_566/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_509/PartitionedCall:output:0dense_566_1010894dense_566_1010896*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_566_layer_call_and_return_conditional_losses_1010079
/batch_normalization_510/StatefulPartitionedCallStatefulPartitionedCall*dense_566/StatefulPartitionedCall:output:0batch_normalization_510_1010899batch_normalization_510_1010901batch_normalization_510_1010903batch_normalization_510_1010905*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1009442ù
leaky_re_lu_510/PartitionedCallPartitionedCall8batch_normalization_510/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_510_layer_call_and_return_conditional_losses_1010099
!dense_567/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_510/PartitionedCall:output:0dense_567_1010909dense_567_1010911*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_567_layer_call_and_return_conditional_losses_1010111
/batch_normalization_511/StatefulPartitionedCallStatefulPartitionedCall*dense_567/StatefulPartitionedCall:output:0batch_normalization_511_1010914batch_normalization_511_1010916batch_normalization_511_1010918batch_normalization_511_1010920*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1009524ù
leaky_re_lu_511/PartitionedCallPartitionedCall8batch_normalization_511/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_511_layer_call_and_return_conditional_losses_1010131
!dense_568/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_511/PartitionedCall:output:0dense_568_1010924dense_568_1010926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_568_layer_call_and_return_conditional_losses_1010143
/batch_normalization_512/StatefulPartitionedCallStatefulPartitionedCall*dense_568/StatefulPartitionedCall:output:0batch_normalization_512_1010929batch_normalization_512_1010931batch_normalization_512_1010933batch_normalization_512_1010935*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1009606ù
leaky_re_lu_512/PartitionedCallPartitionedCall8batch_normalization_512/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_512_layer_call_and_return_conditional_losses_1010163
!dense_569/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_512/PartitionedCall:output:0dense_569_1010939dense_569_1010941*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_569_layer_call_and_return_conditional_losses_1010175
/batch_normalization_513/StatefulPartitionedCallStatefulPartitionedCall*dense_569/StatefulPartitionedCall:output:0batch_normalization_513_1010944batch_normalization_513_1010946batch_normalization_513_1010948batch_normalization_513_1010950*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1009688ù
leaky_re_lu_513/PartitionedCallPartitionedCall8batch_normalization_513/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_513_layer_call_and_return_conditional_losses_1010195
!dense_570/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_513/PartitionedCall:output:0dense_570_1010954dense_570_1010956*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_570_layer_call_and_return_conditional_losses_1010207
/batch_normalization_514/StatefulPartitionedCallStatefulPartitionedCall*dense_570/StatefulPartitionedCall:output:0batch_normalization_514_1010959batch_normalization_514_1010961batch_normalization_514_1010963batch_normalization_514_1010965*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1009770ù
leaky_re_lu_514/PartitionedCallPartitionedCall8batch_normalization_514/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_514_layer_call_and_return_conditional_losses_1010227
!dense_571/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_514/PartitionedCall:output:0dense_571_1010969dense_571_1010971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_571_layer_call_and_return_conditional_losses_1010239
/batch_normalization_515/StatefulPartitionedCallStatefulPartitionedCall*dense_571/StatefulPartitionedCall:output:0batch_normalization_515_1010974batch_normalization_515_1010976batch_normalization_515_1010978batch_normalization_515_1010980*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1009852ù
leaky_re_lu_515/PartitionedCallPartitionedCall8batch_normalization_515/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_515_layer_call_and_return_conditional_losses_1010259
!dense_572/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_515/PartitionedCall:output:0dense_572_1010984dense_572_1010986*
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
GPU 2J 8 *O
fJRH
F__inference_dense_572_layer_call_and_return_conditional_losses_1010271y
IdentityIdentity*dense_572/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
NoOpNoOp0^batch_normalization_504/StatefulPartitionedCall0^batch_normalization_505/StatefulPartitionedCall0^batch_normalization_506/StatefulPartitionedCall0^batch_normalization_507/StatefulPartitionedCall0^batch_normalization_508/StatefulPartitionedCall0^batch_normalization_509/StatefulPartitionedCall0^batch_normalization_510/StatefulPartitionedCall0^batch_normalization_511/StatefulPartitionedCall0^batch_normalization_512/StatefulPartitionedCall0^batch_normalization_513/StatefulPartitionedCall0^batch_normalization_514/StatefulPartitionedCall0^batch_normalization_515/StatefulPartitionedCall"^dense_560/StatefulPartitionedCall"^dense_561/StatefulPartitionedCall"^dense_562/StatefulPartitionedCall"^dense_563/StatefulPartitionedCall"^dense_564/StatefulPartitionedCall"^dense_565/StatefulPartitionedCall"^dense_566/StatefulPartitionedCall"^dense_567/StatefulPartitionedCall"^dense_568/StatefulPartitionedCall"^dense_569/StatefulPartitionedCall"^dense_570/StatefulPartitionedCall"^dense_571/StatefulPartitionedCall"^dense_572/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ð
_input_shapes¾
»:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_504/StatefulPartitionedCall/batch_normalization_504/StatefulPartitionedCall2b
/batch_normalization_505/StatefulPartitionedCall/batch_normalization_505/StatefulPartitionedCall2b
/batch_normalization_506/StatefulPartitionedCall/batch_normalization_506/StatefulPartitionedCall2b
/batch_normalization_507/StatefulPartitionedCall/batch_normalization_507/StatefulPartitionedCall2b
/batch_normalization_508/StatefulPartitionedCall/batch_normalization_508/StatefulPartitionedCall2b
/batch_normalization_509/StatefulPartitionedCall/batch_normalization_509/StatefulPartitionedCall2b
/batch_normalization_510/StatefulPartitionedCall/batch_normalization_510/StatefulPartitionedCall2b
/batch_normalization_511/StatefulPartitionedCall/batch_normalization_511/StatefulPartitionedCall2b
/batch_normalization_512/StatefulPartitionedCall/batch_normalization_512/StatefulPartitionedCall2b
/batch_normalization_513/StatefulPartitionedCall/batch_normalization_513/StatefulPartitionedCall2b
/batch_normalization_514/StatefulPartitionedCall/batch_normalization_514/StatefulPartitionedCall2b
/batch_normalization_515/StatefulPartitionedCall/batch_normalization_515/StatefulPartitionedCall2F
!dense_560/StatefulPartitionedCall!dense_560/StatefulPartitionedCall2F
!dense_561/StatefulPartitionedCall!dense_561/StatefulPartitionedCall2F
!dense_562/StatefulPartitionedCall!dense_562/StatefulPartitionedCall2F
!dense_563/StatefulPartitionedCall!dense_563/StatefulPartitionedCall2F
!dense_564/StatefulPartitionedCall!dense_564/StatefulPartitionedCall2F
!dense_565/StatefulPartitionedCall!dense_565/StatefulPartitionedCall2F
!dense_566/StatefulPartitionedCall!dense_566/StatefulPartitionedCall2F
!dense_567/StatefulPartitionedCall!dense_567/StatefulPartitionedCall2F
!dense_568/StatefulPartitionedCall!dense_568/StatefulPartitionedCall2F
!dense_569/StatefulPartitionedCall!dense_569/StatefulPartitionedCall2F
!dense_570/StatefulPartitionedCall!dense_570/StatefulPartitionedCall2F
!dense_571/StatefulPartitionedCall!dense_571/StatefulPartitionedCall2F
!dense_572/StatefulPartitionedCall!dense_572/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
%
í
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1008950

inputs5
'assignmovingavg_readvariableop_resource:A7
)assignmovingavg_1_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A/
!batchnorm_readvariableop_resource:A
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:A
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:A*
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
:A*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Ax
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A¬
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
:A*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:A~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A´
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Av
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1009395

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É	
÷
F__inference_dense_566_layer_call_and_return_conditional_losses_1010079

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É	
÷
F__inference_dense_565_layer_call_and_return_conditional_losses_1010047

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
¼"
J__inference_sequential_56_layer_call_and_return_conditional_losses_1011694
normalization_56_input
normalization_56_sub_y
normalization_56_sqrt_x#
dense_560_1011508:A
dense_560_1011510:A-
batch_normalization_504_1011513:A-
batch_normalization_504_1011515:A-
batch_normalization_504_1011517:A-
batch_normalization_504_1011519:A#
dense_561_1011523:AA
dense_561_1011525:A-
batch_normalization_505_1011528:A-
batch_normalization_505_1011530:A-
batch_normalization_505_1011532:A-
batch_normalization_505_1011534:A#
dense_562_1011538:AA
dense_562_1011540:A-
batch_normalization_506_1011543:A-
batch_normalization_506_1011545:A-
batch_normalization_506_1011547:A-
batch_normalization_506_1011549:A#
dense_563_1011553:AA
dense_563_1011555:A-
batch_normalization_507_1011558:A-
batch_normalization_507_1011560:A-
batch_normalization_507_1011562:A-
batch_normalization_507_1011564:A#
dense_564_1011568:A
dense_564_1011570:-
batch_normalization_508_1011573:-
batch_normalization_508_1011575:-
batch_normalization_508_1011577:-
batch_normalization_508_1011579:#
dense_565_1011583:
dense_565_1011585:-
batch_normalization_509_1011588:-
batch_normalization_509_1011590:-
batch_normalization_509_1011592:-
batch_normalization_509_1011594:#
dense_566_1011598:
dense_566_1011600:-
batch_normalization_510_1011603:-
batch_normalization_510_1011605:-
batch_normalization_510_1011607:-
batch_normalization_510_1011609:#
dense_567_1011613:5
dense_567_1011615:5-
batch_normalization_511_1011618:5-
batch_normalization_511_1011620:5-
batch_normalization_511_1011622:5-
batch_normalization_511_1011624:5#
dense_568_1011628:55
dense_568_1011630:5-
batch_normalization_512_1011633:5-
batch_normalization_512_1011635:5-
batch_normalization_512_1011637:5-
batch_normalization_512_1011639:5#
dense_569_1011643:55
dense_569_1011645:5-
batch_normalization_513_1011648:5-
batch_normalization_513_1011650:5-
batch_normalization_513_1011652:5-
batch_normalization_513_1011654:5#
dense_570_1011658:55
dense_570_1011660:5-
batch_normalization_514_1011663:5-
batch_normalization_514_1011665:5-
batch_normalization_514_1011667:5-
batch_normalization_514_1011669:5#
dense_571_1011673:55
dense_571_1011675:5-
batch_normalization_515_1011678:5-
batch_normalization_515_1011680:5-
batch_normalization_515_1011682:5-
batch_normalization_515_1011684:5#
dense_572_1011688:5
dense_572_1011690:
identity¢/batch_normalization_504/StatefulPartitionedCall¢/batch_normalization_505/StatefulPartitionedCall¢/batch_normalization_506/StatefulPartitionedCall¢/batch_normalization_507/StatefulPartitionedCall¢/batch_normalization_508/StatefulPartitionedCall¢/batch_normalization_509/StatefulPartitionedCall¢/batch_normalization_510/StatefulPartitionedCall¢/batch_normalization_511/StatefulPartitionedCall¢/batch_normalization_512/StatefulPartitionedCall¢/batch_normalization_513/StatefulPartitionedCall¢/batch_normalization_514/StatefulPartitionedCall¢/batch_normalization_515/StatefulPartitionedCall¢!dense_560/StatefulPartitionedCall¢!dense_561/StatefulPartitionedCall¢!dense_562/StatefulPartitionedCall¢!dense_563/StatefulPartitionedCall¢!dense_564/StatefulPartitionedCall¢!dense_565/StatefulPartitionedCall¢!dense_566/StatefulPartitionedCall¢!dense_567/StatefulPartitionedCall¢!dense_568/StatefulPartitionedCall¢!dense_569/StatefulPartitionedCall¢!dense_570/StatefulPartitionedCall¢!dense_571/StatefulPartitionedCall¢!dense_572/StatefulPartitionedCall}
normalization_56/subSubnormalization_56_inputnormalization_56_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_56/SqrtSqrtnormalization_56_sqrt_x*
T0*
_output_shapes

:_
normalization_56/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_56/MaximumMaximumnormalization_56/Sqrt:y:0#normalization_56/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_56/truedivRealDivnormalization_56/sub:z:0normalization_56/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_560/StatefulPartitionedCallStatefulPartitionedCallnormalization_56/truediv:z:0dense_560_1011508dense_560_1011510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_560_layer_call_and_return_conditional_losses_1009887
/batch_normalization_504/StatefulPartitionedCallStatefulPartitionedCall*dense_560/StatefulPartitionedCall:output:0batch_normalization_504_1011513batch_normalization_504_1011515batch_normalization_504_1011517batch_normalization_504_1011519*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1008950ù
leaky_re_lu_504/PartitionedCallPartitionedCall8batch_normalization_504/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_1009907
!dense_561/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_504/PartitionedCall:output:0dense_561_1011523dense_561_1011525*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_561_layer_call_and_return_conditional_losses_1009919
/batch_normalization_505/StatefulPartitionedCallStatefulPartitionedCall*dense_561/StatefulPartitionedCall:output:0batch_normalization_505_1011528batch_normalization_505_1011530batch_normalization_505_1011532batch_normalization_505_1011534*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1009032ù
leaky_re_lu_505/PartitionedCallPartitionedCall8batch_normalization_505/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_1009939
!dense_562/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_505/PartitionedCall:output:0dense_562_1011538dense_562_1011540*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_562_layer_call_and_return_conditional_losses_1009951
/batch_normalization_506/StatefulPartitionedCallStatefulPartitionedCall*dense_562/StatefulPartitionedCall:output:0batch_normalization_506_1011543batch_normalization_506_1011545batch_normalization_506_1011547batch_normalization_506_1011549*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1009114ù
leaky_re_lu_506/PartitionedCallPartitionedCall8batch_normalization_506/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_1009971
!dense_563/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_506/PartitionedCall:output:0dense_563_1011553dense_563_1011555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_563_layer_call_and_return_conditional_losses_1009983
/batch_normalization_507/StatefulPartitionedCallStatefulPartitionedCall*dense_563/StatefulPartitionedCall:output:0batch_normalization_507_1011558batch_normalization_507_1011560batch_normalization_507_1011562batch_normalization_507_1011564*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1009196ù
leaky_re_lu_507/PartitionedCallPartitionedCall8batch_normalization_507/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_1010003
!dense_564/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_507/PartitionedCall:output:0dense_564_1011568dense_564_1011570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_564_layer_call_and_return_conditional_losses_1010015
/batch_normalization_508/StatefulPartitionedCallStatefulPartitionedCall*dense_564/StatefulPartitionedCall:output:0batch_normalization_508_1011573batch_normalization_508_1011575batch_normalization_508_1011577batch_normalization_508_1011579*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1009278ù
leaky_re_lu_508/PartitionedCallPartitionedCall8batch_normalization_508/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_508_layer_call_and_return_conditional_losses_1010035
!dense_565/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_508/PartitionedCall:output:0dense_565_1011583dense_565_1011585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_565_layer_call_and_return_conditional_losses_1010047
/batch_normalization_509/StatefulPartitionedCallStatefulPartitionedCall*dense_565/StatefulPartitionedCall:output:0batch_normalization_509_1011588batch_normalization_509_1011590batch_normalization_509_1011592batch_normalization_509_1011594*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1009360ù
leaky_re_lu_509/PartitionedCallPartitionedCall8batch_normalization_509/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_509_layer_call_and_return_conditional_losses_1010067
!dense_566/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_509/PartitionedCall:output:0dense_566_1011598dense_566_1011600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_566_layer_call_and_return_conditional_losses_1010079
/batch_normalization_510/StatefulPartitionedCallStatefulPartitionedCall*dense_566/StatefulPartitionedCall:output:0batch_normalization_510_1011603batch_normalization_510_1011605batch_normalization_510_1011607batch_normalization_510_1011609*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1009442ù
leaky_re_lu_510/PartitionedCallPartitionedCall8batch_normalization_510/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_510_layer_call_and_return_conditional_losses_1010099
!dense_567/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_510/PartitionedCall:output:0dense_567_1011613dense_567_1011615*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_567_layer_call_and_return_conditional_losses_1010111
/batch_normalization_511/StatefulPartitionedCallStatefulPartitionedCall*dense_567/StatefulPartitionedCall:output:0batch_normalization_511_1011618batch_normalization_511_1011620batch_normalization_511_1011622batch_normalization_511_1011624*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1009524ù
leaky_re_lu_511/PartitionedCallPartitionedCall8batch_normalization_511/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_511_layer_call_and_return_conditional_losses_1010131
!dense_568/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_511/PartitionedCall:output:0dense_568_1011628dense_568_1011630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_568_layer_call_and_return_conditional_losses_1010143
/batch_normalization_512/StatefulPartitionedCallStatefulPartitionedCall*dense_568/StatefulPartitionedCall:output:0batch_normalization_512_1011633batch_normalization_512_1011635batch_normalization_512_1011637batch_normalization_512_1011639*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1009606ù
leaky_re_lu_512/PartitionedCallPartitionedCall8batch_normalization_512/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_512_layer_call_and_return_conditional_losses_1010163
!dense_569/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_512/PartitionedCall:output:0dense_569_1011643dense_569_1011645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_569_layer_call_and_return_conditional_losses_1010175
/batch_normalization_513/StatefulPartitionedCallStatefulPartitionedCall*dense_569/StatefulPartitionedCall:output:0batch_normalization_513_1011648batch_normalization_513_1011650batch_normalization_513_1011652batch_normalization_513_1011654*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1009688ù
leaky_re_lu_513/PartitionedCallPartitionedCall8batch_normalization_513/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_513_layer_call_and_return_conditional_losses_1010195
!dense_570/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_513/PartitionedCall:output:0dense_570_1011658dense_570_1011660*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_570_layer_call_and_return_conditional_losses_1010207
/batch_normalization_514/StatefulPartitionedCallStatefulPartitionedCall*dense_570/StatefulPartitionedCall:output:0batch_normalization_514_1011663batch_normalization_514_1011665batch_normalization_514_1011667batch_normalization_514_1011669*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1009770ù
leaky_re_lu_514/PartitionedCallPartitionedCall8batch_normalization_514/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_514_layer_call_and_return_conditional_losses_1010227
!dense_571/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_514/PartitionedCall:output:0dense_571_1011673dense_571_1011675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_571_layer_call_and_return_conditional_losses_1010239
/batch_normalization_515/StatefulPartitionedCallStatefulPartitionedCall*dense_571/StatefulPartitionedCall:output:0batch_normalization_515_1011678batch_normalization_515_1011680batch_normalization_515_1011682batch_normalization_515_1011684*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1009852ù
leaky_re_lu_515/PartitionedCallPartitionedCall8batch_normalization_515/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_515_layer_call_and_return_conditional_losses_1010259
!dense_572/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_515/PartitionedCall:output:0dense_572_1011688dense_572_1011690*
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
GPU 2J 8 *O
fJRH
F__inference_dense_572_layer_call_and_return_conditional_losses_1010271y
IdentityIdentity*dense_572/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
NoOpNoOp0^batch_normalization_504/StatefulPartitionedCall0^batch_normalization_505/StatefulPartitionedCall0^batch_normalization_506/StatefulPartitionedCall0^batch_normalization_507/StatefulPartitionedCall0^batch_normalization_508/StatefulPartitionedCall0^batch_normalization_509/StatefulPartitionedCall0^batch_normalization_510/StatefulPartitionedCall0^batch_normalization_511/StatefulPartitionedCall0^batch_normalization_512/StatefulPartitionedCall0^batch_normalization_513/StatefulPartitionedCall0^batch_normalization_514/StatefulPartitionedCall0^batch_normalization_515/StatefulPartitionedCall"^dense_560/StatefulPartitionedCall"^dense_561/StatefulPartitionedCall"^dense_562/StatefulPartitionedCall"^dense_563/StatefulPartitionedCall"^dense_564/StatefulPartitionedCall"^dense_565/StatefulPartitionedCall"^dense_566/StatefulPartitionedCall"^dense_567/StatefulPartitionedCall"^dense_568/StatefulPartitionedCall"^dense_569/StatefulPartitionedCall"^dense_570/StatefulPartitionedCall"^dense_571/StatefulPartitionedCall"^dense_572/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ð
_input_shapes¾
»:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_504/StatefulPartitionedCall/batch_normalization_504/StatefulPartitionedCall2b
/batch_normalization_505/StatefulPartitionedCall/batch_normalization_505/StatefulPartitionedCall2b
/batch_normalization_506/StatefulPartitionedCall/batch_normalization_506/StatefulPartitionedCall2b
/batch_normalization_507/StatefulPartitionedCall/batch_normalization_507/StatefulPartitionedCall2b
/batch_normalization_508/StatefulPartitionedCall/batch_normalization_508/StatefulPartitionedCall2b
/batch_normalization_509/StatefulPartitionedCall/batch_normalization_509/StatefulPartitionedCall2b
/batch_normalization_510/StatefulPartitionedCall/batch_normalization_510/StatefulPartitionedCall2b
/batch_normalization_511/StatefulPartitionedCall/batch_normalization_511/StatefulPartitionedCall2b
/batch_normalization_512/StatefulPartitionedCall/batch_normalization_512/StatefulPartitionedCall2b
/batch_normalization_513/StatefulPartitionedCall/batch_normalization_513/StatefulPartitionedCall2b
/batch_normalization_514/StatefulPartitionedCall/batch_normalization_514/StatefulPartitionedCall2b
/batch_normalization_515/StatefulPartitionedCall/batch_normalization_515/StatefulPartitionedCall2F
!dense_560/StatefulPartitionedCall!dense_560/StatefulPartitionedCall2F
!dense_561/StatefulPartitionedCall!dense_561/StatefulPartitionedCall2F
!dense_562/StatefulPartitionedCall!dense_562/StatefulPartitionedCall2F
!dense_563/StatefulPartitionedCall!dense_563/StatefulPartitionedCall2F
!dense_564/StatefulPartitionedCall!dense_564/StatefulPartitionedCall2F
!dense_565/StatefulPartitionedCall!dense_565/StatefulPartitionedCall2F
!dense_566/StatefulPartitionedCall!dense_566/StatefulPartitionedCall2F
!dense_567/StatefulPartitionedCall!dense_567/StatefulPartitionedCall2F
!dense_568/StatefulPartitionedCall!dense_568/StatefulPartitionedCall2F
!dense_569/StatefulPartitionedCall!dense_569/StatefulPartitionedCall2F
!dense_570/StatefulPartitionedCall!dense_570/StatefulPartitionedCall2F
!dense_571/StatefulPartitionedCall!dense_571/StatefulPartitionedCall2F
!dense_572/StatefulPartitionedCall!dense_572/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_56_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ñ
³
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1008903

inputs/
!batchnorm_readvariableop_resource:A3
%batchnorm_mul_readvariableop_resource:A1
#batchnorm_readvariableop_1_resource:A1
#batchnorm_readvariableop_2_resource:A
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:A*
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
:AP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:A~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ac
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Az
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ar
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_515_layer_call_fn_1014275

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_515_layer_call_and_return_conditional_losses_1010259`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
 
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
normalization_56_input?
(serving_default_normalization_56_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_5720
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ä
¤
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
$layer_with_weights-24
$layer-35
%layer-36
&layer_with_weights-25
&layer-37
'	optimizer
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._default_save_signature
/
signatures"
_tf_keras_sequential
Ó
0
_keep_axis
1_reduce_axis
2_reduce_axis_mask
3_broadcast_shape
4mean
4
adapt_mean
5variance
5adapt_variance
	6count
7	keras_api
8_adapt_function"
_tf_keras_layer
»

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kkernel
lbias
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
©
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£__call__
+¤&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	¥axis

¦gamma
	§beta
¨moving_mean
©moving_variance
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"
_tf_keras_layer
«
°	variables
±trainable_variables
²regularization_losses
³	keras_api
´__call__
+µ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¶kernel
	·bias
¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	¾axis

¿gamma
	Àbeta
Ámoving_mean
Âmoving_variance
Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"
_tf_keras_layer
«
É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ïkernel
	Ðbias
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	×axis

Øgamma
	Ùbeta
Úmoving_mean
Ûmoving_variance
Ü	variables
Ýtrainable_variables
Þregularization_losses
ß	keras_api
à__call__
+á&call_and_return_all_conditional_losses"
_tf_keras_layer
«
â	variables
ãtrainable_variables
äregularization_losses
å	keras_api
æ__call__
+ç&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
èkernel
	ébias
ê	variables
ëtrainable_variables
ìregularization_losses
í	keras_api
î__call__
+ï&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	ðaxis

ñgamma
	òbeta
ómoving_mean
ômoving_variance
õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"
_tf_keras_layer
«
û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	¢axis

£gamma
	¤beta
¥moving_mean
¦moving_variance
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
«
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
³kernel
	´bias
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	»axis

¼gamma
	½beta
¾moving_mean
¿moving_variance
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ìkernel
	Íbias
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	Ôaxis

Õgamma
	Öbeta
×moving_mean
Ømoving_variance
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ß	variables
àtrainable_variables
áregularization_losses
â	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
åkernel
	æbias
ç	variables
ètrainable_variables
éregularization_losses
ê	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"
_tf_keras_layer
ø
	íiter
îbeta_1
ïbeta_2

ðdecay9mµ:m¶Bm·Cm¸Rm¹Smº[m»\m¼km½lm¾tm¿umÀ	mÁ	mÂ	mÃ	mÄ	mÅ	mÆ	¦mÇ	§mÈ	¶mÉ	·mÊ	¿mË	ÀmÌ	ÏmÍ	ÐmÎ	ØmÏ	ÙmÐ	èmÑ	émÒ	ñmÓ	òmÔ	mÕ	mÖ	m×	mØ	mÙ	mÚ	£mÛ	¤mÜ	³mÝ	´mÞ	¼mß	½mà	Ìmá	Ímâ	Õmã	Ömä	åmå	æmæ9vç:vèBvéCvêRvëSvì[ví\vîkvïlvðtvñuvò	vó	vô	võ	vö	v÷	vø	¦vù	§vú	¶vû	·vü	¿vý	Àvþ	Ïvÿ	Ðv	Øv	Ùv	èv	év	ñv	òv	v	v	v	v	v	v	£v	¤v	³v	´v	¼v	½v	Ìv	Ív	Õv	Öv	åv	æv"
	optimizer
¶
40
51
62
93
:4
B5
C6
D7
E8
R9
S10
[11
\12
]13
^14
k15
l16
t17
u18
v19
w20
21
22
23
24
25
26
27
28
¦29
§30
¨31
©32
¶33
·34
¿35
À36
Á37
Â38
Ï39
Ð40
Ø41
Ù42
Ú43
Û44
è45
é46
ñ47
ò48
ó49
ô50
51
52
53
54
55
56
57
58
£59
¤60
¥61
¦62
³63
´64
¼65
½66
¾67
¿68
Ì69
Í70
Õ71
Ö72
×73
Ø74
å75
æ76"
trackable_list_wrapper
Ì
90
:1
B2
C3
R4
S5
[6
\7
k8
l9
t10
u11
12
13
14
15
16
17
¦18
§19
¶20
·21
¿22
À23
Ï24
Ð25
Ø26
Ù27
è28
é29
ñ30
ò31
32
33
34
35
36
37
£38
¤39
³40
´41
¼42
½43
Ì44
Í45
Õ46
Ö47
å48
æ49"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
._default_save_signature
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
2
/__inference_sequential_56_layer_call_fn_1010433
/__inference_sequential_56_layer_call_fn_1011855
/__inference_sequential_56_layer_call_fn_1012012
/__inference_sequential_56_layer_call_fn_1011302À
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
ö2ó
J__inference_sequential_56_layer_call_and_return_conditional_losses_1012305
J__inference_sequential_56_layer_call_and_return_conditional_losses_1012766
J__inference_sequential_56_layer_call_and_return_conditional_losses_1011498
J__inference_sequential_56_layer_call_and_return_conditional_losses_1011694À
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
ÜBÙ
"__inference__wrapped_model_1008879normalization_56_input"
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
öserving_default"
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
À2½
__inference_adapt_step_1012972
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
": A2dense_560/kernel
:A2dense_560/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_560_layer_call_fn_1012981¢
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
ð2í
F__inference_dense_560_layer_call_and_return_conditional_losses_1012991¢
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
+:)A2batch_normalization_504/gamma
*:(A2batch_normalization_504/beta
3:1A (2#batch_normalization_504/moving_mean
7:5A (2'batch_normalization_504/moving_variance
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_504_layer_call_fn_1013004
9__inference_batch_normalization_504_layer_call_fn_1013017´
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
æ2ã
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1013037
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1013071´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_504_layer_call_fn_1013076¢
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
ö2ó
L__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_1013081¢
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
": AA2dense_561/kernel
:A2dense_561/bias
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_561_layer_call_fn_1013090¢
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
ð2í
F__inference_dense_561_layer_call_and_return_conditional_losses_1013100¢
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
+:)A2batch_normalization_505/gamma
*:(A2batch_normalization_505/beta
3:1A (2#batch_normalization_505/moving_mean
7:5A (2'batch_normalization_505/moving_variance
<
[0
\1
]2
^3"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_505_layer_call_fn_1013113
9__inference_batch_normalization_505_layer_call_fn_1013126´
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
æ2ã
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1013146
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1013180´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_505_layer_call_fn_1013185¢
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
ö2ó
L__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_1013190¢
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
": AA2dense_562/kernel
:A2dense_562/bias
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_562_layer_call_fn_1013199¢
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
ð2í
F__inference_dense_562_layer_call_and_return_conditional_losses_1013209¢
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
+:)A2batch_normalization_506/gamma
*:(A2batch_normalization_506/beta
3:1A (2#batch_normalization_506/moving_mean
7:5A (2'batch_normalization_506/moving_variance
<
t0
u1
v2
w3"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_506_layer_call_fn_1013222
9__inference_batch_normalization_506_layer_call_fn_1013235´
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
æ2ã
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1013255
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1013289´
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
¶
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_506_layer_call_fn_1013294¢
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
ö2ó
L__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_1013299¢
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
": AA2dense_563/kernel
:A2dense_563/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_563_layer_call_fn_1013308¢
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
ð2í
F__inference_dense_563_layer_call_and_return_conditional_losses_1013318¢
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
+:)A2batch_normalization_507/gamma
*:(A2batch_normalization_507/beta
3:1A (2#batch_normalization_507/moving_mean
7:5A (2'batch_normalization_507/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_507_layer_call_fn_1013331
9__inference_batch_normalization_507_layer_call_fn_1013344´
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
æ2ã
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1013364
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1013398´
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
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_507_layer_call_fn_1013403¢
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
ö2ó
L__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_1013408¢
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
": A2dense_564/kernel
:2dense_564/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
	variables
 trainable_variables
¡regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_564_layer_call_fn_1013417¢
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
ð2í
F__inference_dense_564_layer_call_and_return_conditional_losses_1013427¢
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
+:)2batch_normalization_508/gamma
*:(2batch_normalization_508/beta
3:1 (2#batch_normalization_508/moving_mean
7:5 (2'batch_normalization_508/moving_variance
@
¦0
§1
¨2
©3"
trackable_list_wrapper
0
¦0
§1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_508_layer_call_fn_1013440
9__inference_batch_normalization_508_layer_call_fn_1013453´
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
æ2ã
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1013473
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1013507´
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
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
°	variables
±trainable_variables
²regularization_losses
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_508_layer_call_fn_1013512¢
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
ö2ó
L__inference_leaky_re_lu_508_layer_call_and_return_conditional_losses_1013517¢
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
": 2dense_565/kernel
:2dense_565/bias
0
¶0
·1"
trackable_list_wrapper
0
¶0
·1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_565_layer_call_fn_1013526¢
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
ð2í
F__inference_dense_565_layer_call_and_return_conditional_losses_1013536¢
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
+:)2batch_normalization_509/gamma
*:(2batch_normalization_509/beta
3:1 (2#batch_normalization_509/moving_mean
7:5 (2'batch_normalization_509/moving_variance
@
¿0
À1
Á2
Â3"
trackable_list_wrapper
0
¿0
À1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_509_layer_call_fn_1013549
9__inference_batch_normalization_509_layer_call_fn_1013562´
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
æ2ã
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1013582
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1013616´
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
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
É	variables
Êtrainable_variables
Ëregularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_509_layer_call_fn_1013621¢
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
ö2ó
L__inference_leaky_re_lu_509_layer_call_and_return_conditional_losses_1013626¢
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
": 2dense_566/kernel
:2dense_566/bias
0
Ï0
Ð1"
trackable_list_wrapper
0
Ï0
Ð1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
Ñ	variables
Òtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_566_layer_call_fn_1013635¢
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
ð2í
F__inference_dense_566_layer_call_and_return_conditional_losses_1013645¢
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
+:)2batch_normalization_510/gamma
*:(2batch_normalization_510/beta
3:1 (2#batch_normalization_510/moving_mean
7:5 (2'batch_normalization_510/moving_variance
@
Ø0
Ù1
Ú2
Û3"
trackable_list_wrapper
0
Ø0
Ù1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
Ü	variables
Ýtrainable_variables
Þregularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_510_layer_call_fn_1013658
9__inference_batch_normalization_510_layer_call_fn_1013671´
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
æ2ã
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1013691
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1013725´
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
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
â	variables
ãtrainable_variables
äregularization_losses
æ__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_510_layer_call_fn_1013730¢
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
ö2ó
L__inference_leaky_re_lu_510_layer_call_and_return_conditional_losses_1013735¢
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
": 52dense_567/kernel
:52dense_567/bias
0
è0
é1"
trackable_list_wrapper
0
è0
é1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
ê	variables
ëtrainable_variables
ìregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_567_layer_call_fn_1013744¢
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
ð2í
F__inference_dense_567_layer_call_and_return_conditional_losses_1013754¢
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
+:)52batch_normalization_511/gamma
*:(52batch_normalization_511/beta
3:15 (2#batch_normalization_511/moving_mean
7:55 (2'batch_normalization_511/moving_variance
@
ñ0
ò1
ó2
ô3"
trackable_list_wrapper
0
ñ0
ò1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_511_layer_call_fn_1013767
9__inference_batch_normalization_511_layer_call_fn_1013780´
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
æ2ã
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1013800
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1013834´
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
ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_511_layer_call_fn_1013839¢
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
ö2ó
L__inference_leaky_re_lu_511_layer_call_and_return_conditional_losses_1013844¢
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
": 552dense_568/kernel
:52dense_568/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_568_layer_call_fn_1013853¢
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
ð2í
F__inference_dense_568_layer_call_and_return_conditional_losses_1013863¢
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
+:)52batch_normalization_512/gamma
*:(52batch_normalization_512/beta
3:15 (2#batch_normalization_512/moving_mean
7:55 (2'batch_normalization_512/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_512_layer_call_fn_1013876
9__inference_batch_normalization_512_layer_call_fn_1013889´
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
æ2ã
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1013909
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1013943´
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
ùnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_512_layer_call_fn_1013948¢
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
ö2ó
L__inference_leaky_re_lu_512_layer_call_and_return_conditional_losses_1013953¢
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
": 552dense_569/kernel
:52dense_569/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_569_layer_call_fn_1013962¢
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
ð2í
F__inference_dense_569_layer_call_and_return_conditional_losses_1013972¢
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
+:)52batch_normalization_513/gamma
*:(52batch_normalization_513/beta
3:15 (2#batch_normalization_513/moving_mean
7:55 (2'batch_normalization_513/moving_variance
@
£0
¤1
¥2
¦3"
trackable_list_wrapper
0
£0
¤1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_513_layer_call_fn_1013985
9__inference_batch_normalization_513_layer_call_fn_1013998´
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
æ2ã
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1014018
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1014052´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
­	variables
®trainable_variables
¯regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_513_layer_call_fn_1014057¢
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
ö2ó
L__inference_leaky_re_lu_513_layer_call_and_return_conditional_losses_1014062¢
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
": 552dense_570/kernel
:52dense_570/bias
0
³0
´1"
trackable_list_wrapper
0
³0
´1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
µ	variables
¶trainable_variables
·regularization_losses
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_570_layer_call_fn_1014071¢
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
ð2í
F__inference_dense_570_layer_call_and_return_conditional_losses_1014081¢
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
+:)52batch_normalization_514/gamma
*:(52batch_normalization_514/beta
3:15 (2#batch_normalization_514/moving_mean
7:55 (2'batch_normalization_514/moving_variance
@
¼0
½1
¾2
¿3"
trackable_list_wrapper
0
¼0
½1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_514_layer_call_fn_1014094
9__inference_batch_normalization_514_layer_call_fn_1014107´
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
æ2ã
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1014127
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1014161´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Æ	variables
Çtrainable_variables
Èregularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_514_layer_call_fn_1014166¢
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
ö2ó
L__inference_leaky_re_lu_514_layer_call_and_return_conditional_losses_1014171¢
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
": 552dense_571/kernel
:52dense_571/bias
0
Ì0
Í1"
trackable_list_wrapper
0
Ì0
Í1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_571_layer_call_fn_1014180¢
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
ð2í
F__inference_dense_571_layer_call_and_return_conditional_losses_1014190¢
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
+:)52batch_normalization_515/gamma
*:(52batch_normalization_515/beta
3:15 (2#batch_normalization_515/moving_mean
7:55 (2'batch_normalization_515/moving_variance
@
Õ0
Ö1
×2
Ø3"
trackable_list_wrapper
0
Õ0
Ö1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
Ù	variables
Útrainable_variables
Ûregularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_515_layer_call_fn_1014203
9__inference_batch_normalization_515_layer_call_fn_1014216´
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
æ2ã
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1014236
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1014270´
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
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
ß	variables
àtrainable_variables
áregularization_losses
ã__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_515_layer_call_fn_1014275¢
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
ö2ó
L__inference_leaky_re_lu_515_layer_call_and_return_conditional_losses_1014280¢
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
": 52dense_572/kernel
:2dense_572/bias
0
å0
æ1"
trackable_list_wrapper
0
å0
æ1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
ç	variables
ètrainable_variables
éregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_572_layer_call_fn_1014289¢
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
ð2í
F__inference_dense_572_layer_call_and_return_conditional_losses_1014299¢
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

40
51
62
D3
E4
]5
^6
v7
w8
9
10
¨11
©12
Á13
Â14
Ú15
Û16
ó17
ô18
19
20
¥21
¦22
¾23
¿24
×25
Ø26"
trackable_list_wrapper
Æ
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
#34
$35
%36
&37"
trackable_list_wrapper
(
°0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
%__inference_signature_wrapper_1012925normalization_56_input"
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
D0
E1"
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
]0
^1"
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
v0
w1"
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
0
1"
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
¨0
©1"
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
Á0
Â1"
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
Ú0
Û1"
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
ó0
ô1"
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
0
1"
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
¥0
¦1"
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
¾0
¿1"
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
×0
Ø1"
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

±total

²count
³	variables
´	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
±0
²1"
trackable_list_wrapper
.
³	variables"
_generic_user_object
':%A2Adam/dense_560/kernel/m
!:A2Adam/dense_560/bias/m
0:.A2$Adam/batch_normalization_504/gamma/m
/:-A2#Adam/batch_normalization_504/beta/m
':%AA2Adam/dense_561/kernel/m
!:A2Adam/dense_561/bias/m
0:.A2$Adam/batch_normalization_505/gamma/m
/:-A2#Adam/batch_normalization_505/beta/m
':%AA2Adam/dense_562/kernel/m
!:A2Adam/dense_562/bias/m
0:.A2$Adam/batch_normalization_506/gamma/m
/:-A2#Adam/batch_normalization_506/beta/m
':%AA2Adam/dense_563/kernel/m
!:A2Adam/dense_563/bias/m
0:.A2$Adam/batch_normalization_507/gamma/m
/:-A2#Adam/batch_normalization_507/beta/m
':%A2Adam/dense_564/kernel/m
!:2Adam/dense_564/bias/m
0:.2$Adam/batch_normalization_508/gamma/m
/:-2#Adam/batch_normalization_508/beta/m
':%2Adam/dense_565/kernel/m
!:2Adam/dense_565/bias/m
0:.2$Adam/batch_normalization_509/gamma/m
/:-2#Adam/batch_normalization_509/beta/m
':%2Adam/dense_566/kernel/m
!:2Adam/dense_566/bias/m
0:.2$Adam/batch_normalization_510/gamma/m
/:-2#Adam/batch_normalization_510/beta/m
':%52Adam/dense_567/kernel/m
!:52Adam/dense_567/bias/m
0:.52$Adam/batch_normalization_511/gamma/m
/:-52#Adam/batch_normalization_511/beta/m
':%552Adam/dense_568/kernel/m
!:52Adam/dense_568/bias/m
0:.52$Adam/batch_normalization_512/gamma/m
/:-52#Adam/batch_normalization_512/beta/m
':%552Adam/dense_569/kernel/m
!:52Adam/dense_569/bias/m
0:.52$Adam/batch_normalization_513/gamma/m
/:-52#Adam/batch_normalization_513/beta/m
':%552Adam/dense_570/kernel/m
!:52Adam/dense_570/bias/m
0:.52$Adam/batch_normalization_514/gamma/m
/:-52#Adam/batch_normalization_514/beta/m
':%552Adam/dense_571/kernel/m
!:52Adam/dense_571/bias/m
0:.52$Adam/batch_normalization_515/gamma/m
/:-52#Adam/batch_normalization_515/beta/m
':%52Adam/dense_572/kernel/m
!:2Adam/dense_572/bias/m
':%A2Adam/dense_560/kernel/v
!:A2Adam/dense_560/bias/v
0:.A2$Adam/batch_normalization_504/gamma/v
/:-A2#Adam/batch_normalization_504/beta/v
':%AA2Adam/dense_561/kernel/v
!:A2Adam/dense_561/bias/v
0:.A2$Adam/batch_normalization_505/gamma/v
/:-A2#Adam/batch_normalization_505/beta/v
':%AA2Adam/dense_562/kernel/v
!:A2Adam/dense_562/bias/v
0:.A2$Adam/batch_normalization_506/gamma/v
/:-A2#Adam/batch_normalization_506/beta/v
':%AA2Adam/dense_563/kernel/v
!:A2Adam/dense_563/bias/v
0:.A2$Adam/batch_normalization_507/gamma/v
/:-A2#Adam/batch_normalization_507/beta/v
':%A2Adam/dense_564/kernel/v
!:2Adam/dense_564/bias/v
0:.2$Adam/batch_normalization_508/gamma/v
/:-2#Adam/batch_normalization_508/beta/v
':%2Adam/dense_565/kernel/v
!:2Adam/dense_565/bias/v
0:.2$Adam/batch_normalization_509/gamma/v
/:-2#Adam/batch_normalization_509/beta/v
':%2Adam/dense_566/kernel/v
!:2Adam/dense_566/bias/v
0:.2$Adam/batch_normalization_510/gamma/v
/:-2#Adam/batch_normalization_510/beta/v
':%52Adam/dense_567/kernel/v
!:52Adam/dense_567/bias/v
0:.52$Adam/batch_normalization_511/gamma/v
/:-52#Adam/batch_normalization_511/beta/v
':%552Adam/dense_568/kernel/v
!:52Adam/dense_568/bias/v
0:.52$Adam/batch_normalization_512/gamma/v
/:-52#Adam/batch_normalization_512/beta/v
':%552Adam/dense_569/kernel/v
!:52Adam/dense_569/bias/v
0:.52$Adam/batch_normalization_513/gamma/v
/:-52#Adam/batch_normalization_513/beta/v
':%552Adam/dense_570/kernel/v
!:52Adam/dense_570/bias/v
0:.52$Adam/batch_normalization_514/gamma/v
/:-52#Adam/batch_normalization_514/beta/v
':%552Adam/dense_571/kernel/v
!:52Adam/dense_571/bias/v
0:.52$Adam/batch_normalization_515/gamma/v
/:-52#Adam/batch_normalization_515/beta/v
':%52Adam/dense_572/kernel/v
!:2Adam/dense_572/bias/v
	J
Const
J	
Const_1¨
"__inference__wrapped_model_10088799:EBDCRS^[]\klwtvu©¦¨§¶·Â¿ÁÀÏÐÛØÚÙèéôñóò¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæ?¢<
5¢2
0-
normalization_56_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_572# 
	dense_572ÿÿÿÿÿÿÿÿÿp
__inference_adapt_step_1012972N645C¢@
9¢6
41¢
ÿÿÿÿÿÿÿÿÿIteratorSpec 
ª "
 º
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1013037bEBDC3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 º
T__inference_batch_normalization_504_layer_call_and_return_conditional_losses_1013071bDEBC3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
9__inference_batch_normalization_504_layer_call_fn_1013004UEBDC3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "ÿÿÿÿÿÿÿÿÿA
9__inference_batch_normalization_504_layer_call_fn_1013017UDEBC3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "ÿÿÿÿÿÿÿÿÿAº
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1013146b^[]\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 º
T__inference_batch_normalization_505_layer_call_and_return_conditional_losses_1013180b]^[\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
9__inference_batch_normalization_505_layer_call_fn_1013113U^[]\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "ÿÿÿÿÿÿÿÿÿA
9__inference_batch_normalization_505_layer_call_fn_1013126U]^[\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "ÿÿÿÿÿÿÿÿÿAº
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1013255bwtvu3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 º
T__inference_batch_normalization_506_layer_call_and_return_conditional_losses_1013289bvwtu3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
9__inference_batch_normalization_506_layer_call_fn_1013222Uwtvu3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "ÿÿÿÿÿÿÿÿÿA
9__inference_batch_normalization_506_layer_call_fn_1013235Uvwtu3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "ÿÿÿÿÿÿÿÿÿA¾
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1013364f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 ¾
T__inference_batch_normalization_507_layer_call_and_return_conditional_losses_1013398f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
9__inference_batch_normalization_507_layer_call_fn_1013331Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "ÿÿÿÿÿÿÿÿÿA
9__inference_batch_normalization_507_layer_call_fn_1013344Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "ÿÿÿÿÿÿÿÿÿA¾
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1013473f©¦¨§3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
T__inference_batch_normalization_508_layer_call_and_return_conditional_losses_1013507f¨©¦§3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_508_layer_call_fn_1013440Y©¦¨§3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_508_layer_call_fn_1013453Y¨©¦§3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¾
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1013582fÂ¿ÁÀ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
T__inference_batch_normalization_509_layer_call_and_return_conditional_losses_1013616fÁÂ¿À3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_509_layer_call_fn_1013549YÂ¿ÁÀ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_509_layer_call_fn_1013562YÁÂ¿À3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¾
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1013691fÛØÚÙ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
T__inference_batch_normalization_510_layer_call_and_return_conditional_losses_1013725fÚÛØÙ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_510_layer_call_fn_1013658YÛØÚÙ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_510_layer_call_fn_1013671YÚÛØÙ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¾
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1013800fôñóò3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 ¾
T__inference_batch_normalization_511_layer_call_and_return_conditional_losses_1013834fóôñò3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
9__inference_batch_normalization_511_layer_call_fn_1013767Yôñóò3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "ÿÿÿÿÿÿÿÿÿ5
9__inference_batch_normalization_511_layer_call_fn_1013780Yóôñò3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "ÿÿÿÿÿÿÿÿÿ5¾
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1013909f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 ¾
T__inference_batch_normalization_512_layer_call_and_return_conditional_losses_1013943f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
9__inference_batch_normalization_512_layer_call_fn_1013876Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "ÿÿÿÿÿÿÿÿÿ5
9__inference_batch_normalization_512_layer_call_fn_1013889Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "ÿÿÿÿÿÿÿÿÿ5¾
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1014018f¦£¥¤3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 ¾
T__inference_batch_normalization_513_layer_call_and_return_conditional_losses_1014052f¥¦£¤3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
9__inference_batch_normalization_513_layer_call_fn_1013985Y¦£¥¤3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "ÿÿÿÿÿÿÿÿÿ5
9__inference_batch_normalization_513_layer_call_fn_1013998Y¥¦£¤3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "ÿÿÿÿÿÿÿÿÿ5¾
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1014127f¿¼¾½3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 ¾
T__inference_batch_normalization_514_layer_call_and_return_conditional_losses_1014161f¾¿¼½3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
9__inference_batch_normalization_514_layer_call_fn_1014094Y¿¼¾½3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "ÿÿÿÿÿÿÿÿÿ5
9__inference_batch_normalization_514_layer_call_fn_1014107Y¾¿¼½3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "ÿÿÿÿÿÿÿÿÿ5¾
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1014236fØÕ×Ö3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 ¾
T__inference_batch_normalization_515_layer_call_and_return_conditional_losses_1014270f×ØÕÖ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
9__inference_batch_normalization_515_layer_call_fn_1014203YØÕ×Ö3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "ÿÿÿÿÿÿÿÿÿ5
9__inference_batch_normalization_515_layer_call_fn_1014216Y×ØÕÖ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "ÿÿÿÿÿÿÿÿÿ5¦
F__inference_dense_560_layer_call_and_return_conditional_losses_1012991\9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 ~
+__inference_dense_560_layer_call_fn_1012981O9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿA¦
F__inference_dense_561_layer_call_and_return_conditional_losses_1013100\RS/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 ~
+__inference_dense_561_layer_call_fn_1013090ORS/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA¦
F__inference_dense_562_layer_call_and_return_conditional_losses_1013209\kl/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 ~
+__inference_dense_562_layer_call_fn_1013199Okl/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA¨
F__inference_dense_563_layer_call_and_return_conditional_losses_1013318^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
+__inference_dense_563_layer_call_fn_1013308Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA¨
F__inference_dense_564_layer_call_and_return_conditional_losses_1013427^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_564_layer_call_fn_1013417Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿ¨
F__inference_dense_565_layer_call_and_return_conditional_losses_1013536^¶·/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_565_layer_call_fn_1013526Q¶·/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
F__inference_dense_566_layer_call_and_return_conditional_losses_1013645^ÏÐ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_566_layer_call_fn_1013635QÏÐ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
F__inference_dense_567_layer_call_and_return_conditional_losses_1013754^èé/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
+__inference_dense_567_layer_call_fn_1013744Qèé/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ5¨
F__inference_dense_568_layer_call_and_return_conditional_losses_1013863^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
+__inference_dense_568_layer_call_fn_1013853Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ5¨
F__inference_dense_569_layer_call_and_return_conditional_losses_1013972^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
+__inference_dense_569_layer_call_fn_1013962Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ5¨
F__inference_dense_570_layer_call_and_return_conditional_losses_1014081^³´/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
+__inference_dense_570_layer_call_fn_1014071Q³´/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ5¨
F__inference_dense_571_layer_call_and_return_conditional_losses_1014190^ÌÍ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
+__inference_dense_571_layer_call_fn_1014180QÌÍ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ5¨
F__inference_dense_572_layer_call_and_return_conditional_losses_1014299^åæ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_572_layer_call_fn_1014289Qåæ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_504_layer_call_and_return_conditional_losses_1013081X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
1__inference_leaky_re_lu_504_layer_call_fn_1013076K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA¨
L__inference_leaky_re_lu_505_layer_call_and_return_conditional_losses_1013190X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
1__inference_leaky_re_lu_505_layer_call_fn_1013185K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA¨
L__inference_leaky_re_lu_506_layer_call_and_return_conditional_losses_1013299X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
1__inference_leaky_re_lu_506_layer_call_fn_1013294K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA¨
L__inference_leaky_re_lu_507_layer_call_and_return_conditional_losses_1013408X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
1__inference_leaky_re_lu_507_layer_call_fn_1013403K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA¨
L__inference_leaky_re_lu_508_layer_call_and_return_conditional_losses_1013517X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_leaky_re_lu_508_layer_call_fn_1013512K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_509_layer_call_and_return_conditional_losses_1013626X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_leaky_re_lu_509_layer_call_fn_1013621K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_510_layer_call_and_return_conditional_losses_1013735X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_leaky_re_lu_510_layer_call_fn_1013730K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_511_layer_call_and_return_conditional_losses_1013844X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
1__inference_leaky_re_lu_511_layer_call_fn_1013839K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ5¨
L__inference_leaky_re_lu_512_layer_call_and_return_conditional_losses_1013953X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
1__inference_leaky_re_lu_512_layer_call_fn_1013948K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ5¨
L__inference_leaky_re_lu_513_layer_call_and_return_conditional_losses_1014062X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
1__inference_leaky_re_lu_513_layer_call_fn_1014057K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ5¨
L__inference_leaky_re_lu_514_layer_call_and_return_conditional_losses_1014171X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
1__inference_leaky_re_lu_514_layer_call_fn_1014166K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ5¨
L__inference_leaky_re_lu_515_layer_call_and_return_conditional_losses_1014280X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
1__inference_leaky_re_lu_515_layer_call_fn_1014275K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ5È
J__inference_sequential_56_layer_call_and_return_conditional_losses_1011498ù9:EBDCRS^[]\klwtvu©¦¨§¶·Â¿ÁÀÏÐÛØÚÙèéôñóò¦£¥¤³´¿¼¾½ÌÍØÕ×ÖåæG¢D
=¢:
0-
normalization_56_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
J__inference_sequential_56_layer_call_and_return_conditional_losses_1011694ù9:DEBCRS]^[\klvwtu¨©¦§¶·ÁÂ¿ÀÏÐÚÛØÙèéóôñò¥¦£¤³´¾¿¼½ÌÍ×ØÕÖåæG¢D
=¢:
0-
normalization_56_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
J__inference_sequential_56_layer_call_and_return_conditional_losses_1012305é9:EBDCRS^[]\klwtvu©¦¨§¶·Â¿ÁÀÏÐÛØÚÙèéôñóò¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
J__inference_sequential_56_layer_call_and_return_conditional_losses_1012766é9:DEBCRS]^[\klvwtu¨©¦§¶·ÁÂ¿ÀÏÐÚÛØÙèéóôñò¥¦£¤³´¾¿¼½ÌÍ×ØÕÖåæ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
  
/__inference_sequential_56_layer_call_fn_1010433ì9:EBDCRS^[]\klwtvu©¦¨§¶·Â¿ÁÀÏÐÛØÚÙèéôñóò¦£¥¤³´¿¼¾½ÌÍØÕ×ÖåæG¢D
=¢:
0-
normalization_56_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
/__inference_sequential_56_layer_call_fn_1011302ì9:DEBCRS]^[\klvwtu¨©¦§¶·ÁÂ¿ÀÏÐÚÛØÙèéóôñò¥¦£¤³´¾¿¼½ÌÍ×ØÕÖåæG¢D
=¢:
0-
normalization_56_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_56_layer_call_fn_1011855Ü9:EBDCRS^[]\klwtvu©¦¨§¶·Â¿ÁÀÏÐÛØÚÙèéôñóò¦£¥¤³´¿¼¾½ÌÍØÕ×Öåæ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_56_layer_call_fn_1012012Ü9:DEBCRS]^[\klvwtu¨©¦§¶·ÁÂ¿ÀÏÐÚÛØÙèéóôñò¥¦£¤³´¾¿¼½ÌÍ×ØÕÖåæ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÅ
%__inference_signature_wrapper_10129259:EBDCRS^[]\klwtvu©¦¨§¶·Â¿ÁÀÏÐÛØÚÙèéôñóò¦£¥¤³´¿¼¾½ÌÍØÕ×ÖåæY¢V
¢ 
OªL
J
normalization_56_input0-
normalization_56_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_572# 
	dense_572ÿÿÿÿÿÿÿÿÿ